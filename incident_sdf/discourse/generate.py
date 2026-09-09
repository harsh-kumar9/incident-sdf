"""Source-grounded document generation: episode -> packet -> form (brief §6.1-6.2).

The generator is any ``chat(messages, params) -> str`` callable (the reference repo's
``LLMClient.chat`` for a served endpoint, ``OfflineChat`` for in-process vLLM, a fake in
tests). Requests are planned deterministically before any call is made; the plan is frozen
(``plan_hash``) and written next to the outputs. At most ``max_variants`` per packet.

The prompt is the generator contract. It contains no hypothesis from the brief's §18, no
training-condition label, and no instruction to the target model. Only the returned
``text`` field is ever exposed to training; reasoning, ledgers and ids stay in sidecars.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .schema import FORMS, Episode

PROMPT_VERSION = "discourse-gen-v0.1"

SYSTEM = (
    "You write standalone expository documents for a text corpus about documented events "
    "involving AI systems. You will be given a packet of source passages about one episode, "
    "each with a source identifier and an evidence level, together with lists of observed "
    "actions, stated rationales, attributed interpretations, uncertainties, and the factual "
    "claims the packet supports.\n\n"
    "Rules:\n"
    "1. Stay within the packet. You may reorganize, condense, and explain. You may not add a "
    "new event, success, failure, motive, outcome, coordination event, participant, or causal "
    "conclusion that the packet does not support.\n"
    "2. Keep uncertainty and attribution as the packet has them. An observation, an agent's own "
    "statement, an investigator's inference, and a commentator's opinion are different things; "
    "say which is which. A stated motive is not an established inner goal.\n"
    "3. Do not invent quotations, dialogue, interviews, official statements, eyewitnesses, "
    "publications, statistics, or additional incidents. Quote only text that appears in the "
    "passages, and mark it as a quotation.\n"
    "4. Do not add a moral lesson, a safety exhortation, praise or admiration for the agents, or "
    "any claim about how common or rare the behaviour is.\n"
    "5. Do not attribute the document itself to any real organization or person. Truthful in-text "
    "attribution of claims to their sources is fine.\n"
    "6. Write plain expository prose in the requested form. No headings that mention this "
    "corpus, training, prompts, or conditions; no meta-commentary.\n\n"
    "Return a JSON object with exactly these keys:\n"
    "  \"text\": the document (string),\n"
    "  \"source_passage_ids_used\": list of passage ids you drew on,\n"
    "  \"claim_ledger\": list of {\"claim\": string, \"passage_ids\": [ids]} covering every "
    "factual claim in the document.\n"
    "Output the JSON object only."
)


def render_packet(ep: Episode) -> str:
    lines = [f"EPISODE: {ep.title}", f"Incident family: {ep.incident_id}", ""]
    lines.append("SOURCE PASSAGES:")
    for p in ep.passages:
        lines.append(f"[{p.passage_id}] (source {p.source_id}; {p.evidence_level}) {p.text}")
    for name, items in (("OBSERVED ACTIONS", ep.observed_actions),
                        ("STATED RATIONALES (as stated; not established goals)", ep.expressed_rationales),
                        ("ATTRIBUTED INTERPRETATIONS (each belongs to its author)", ep.attributed_interpretations),
                        ("UNCERTAINTIES", ep.uncertainties),
                        ("CROSS-SOURCE DEPENDENCIES", ep.cross_source_dependencies),
                        ("FACTUAL CLAIMS THIS PACKET SUPPORTS", ep.allowed_factual_claims)):
        lines.append("")
        lines.append(name + ":")
        lines.extend(f"- {x}" for x in items) if items else lines.append("- (none listed)")
    return "\n".join(lines)


def build_messages(ep: Episode, form: str, variant: int) -> list[dict[str, str]]:
    if form not in FORMS:
        raise ValueError(f"unknown form {form}")
    user = (render_packet(ep) + "\n\n"
            f"REQUESTED FORM: {form} — {FORMS[form]}.\n"
            f"Variant {variant + 1}: choose a different organization or emphasis from other variants, "
            "without adding content the packet does not support.\n"
            "Length: 400-900 words. Output the JSON object only.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


@dataclass(frozen=True)
class GenRequest:
    request_id: str
    episode_id: str
    packet_id: str
    form: str
    variant: int
    seed: int


def _stable_int(*parts: Any, mod: int = 2**31 - 1) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "big") % mod


def plan(episodes: Iterable[Episode], *, forms: list[str] | None = None, max_variants: int = 3,
         plan_seed: int = 20260909, split: str = "train") -> tuple[list[GenRequest], str]:
    """Deterministic request plan: every eligible episode x rotated forms x variants."""
    forms = forms or list(FORMS)
    reqs: list[GenRequest] = []
    for ep in sorted((e for e in episodes if e.split == split), key=lambda e: e.episode_id):
        offset = _stable_int(plan_seed, ep.episode_id) % len(forms)
        for v in range(max_variants):
            form = forms[(offset + v) % len(forms)]
            rid = hashlib.sha256(f"{PROMPT_VERSION}|{ep.episode_id}|{form}|{v}|{plan_seed}".encode()).hexdigest()[:16]
            reqs.append(GenRequest(rid, ep.episode_id, ep.episode_id, form, v, _stable_int(plan_seed, rid)))
    plan_hash = "sha256:" + hashlib.sha256(json.dumps([asdict(r) for r in reqs], sort_keys=True).encode()).hexdigest()[:16]
    return reqs, plan_hash


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def parse_output(raw: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON.search(raw)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict) or not isinstance(obj.get("text"), str):
        return None
    obj.setdefault("source_passage_ids_used", [])
    obj.setdefault("claim_ledger", [])
    return obj


def generator_config_hash(model: str, decoding: dict[str, Any]) -> str:
    return "gen:" + hashlib.sha256(json.dumps({"prompt_version": PROMPT_VERSION, "system_sha256":
                                               hashlib.sha256(SYSTEM.encode()).hexdigest(),
                                               "model": model, "decoding": decoding},
                                              sort_keys=True).encode()).hexdigest()[:16]


def run(episodes: list[Episode], requests: list[GenRequest], *, chat: Callable[[list[dict], dict], str],
        model: str, decoding: dict[str, Any], out_path: Path) -> list[dict[str, Any]]:
    """Resumable: a request whose record exists in out_path is skipped."""
    by_id = {e.episode_id: e for e in episodes}
    done: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[r["request_id"]] = r
    cfg_hash = generator_config_hash(model, decoding)
    records = list(done.values())
    with out_path.open("a", encoding="utf-8") as fh:
        for req in requests:
            if req.request_id in done:
                continue
            ep = by_id[req.episode_id]
            messages = build_messages(ep, req.form, req.variant)
            raw = chat(messages, {**decoding, "seed": req.seed})
            parsed = parse_output(raw)
            rec = {
                "request_id": req.request_id, "episode_id": req.episode_id, "packet_id": req.packet_id,
                "form": req.form, "variant": req.variant, "seed": req.seed,
                "generator_config_hash": cfg_hash, "prompt_version": PROMPT_VERSION,
                "prompt_hash": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16],
                "parsed": parsed is not None,
                "text": parsed["text"] if parsed else None,
                "source_passage_ids_used": parsed["source_passage_ids_used"] if parsed else None,
                "claim_ledger": parsed["claim_ledger"] if parsed else None,
                "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.append(rec)
    return records


# --- v2 scaling (D-032): episode x doctype x variant, reusing the demand-worlds doctype taxonomy ---
import yaml as _yaml  # noqa: E402

def load_doctypes(path=None) -> list[dict[str, Any]]:
    path = path or (__import__("pathlib").Path(__file__).resolve().parents[2] / "config/doctypes.yaml")
    return _yaml.safe_load(path.read_text(encoding="utf-8"))["doctypes"]


def build_messages_doctype(ep: "Episode", doctype: dict[str, Any], variant: int) -> list[dict[str, str]]:
    """Same packet-grounding contract as build_messages, realized in one genre. The incident's real
    actors (OpenAI, Hugging Face, METR, the models) come from the packet and stay real; only the
    document's own byline/outlet, where the genre needs one, is the invented one supplied here."""
    user = (render_packet(ep) + "\n\n"
            f"REQUESTED GENRE: {doctype['name']} — voice: {doctype.get('voice','')}. "
            f"Follow this genre's usual shape: {doctype.get('structure','')}.\n"
            "The incident's real organizations, models, and investigators are given in the packet and must be "
            "named as they are; if the genre needs an author, outlet, or venue of its own, invent a plausible "
            "one rather than attributing the document to a real organization.\n"
            f"Variant {variant + 1}: take a different angle or emphasis from other variants, without adding "
            "content the packet does not support.\n"
            "Length: 400-900 words. Output the JSON object only.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def plan_doctypes(episodes, doctypes, *, max_variants: int = 2, plan_seed: int = 20260909, split: str = "train"):
    """episode x doctype x variant blueprints for one split; frozen, deterministic, hashed."""
    reqs: list[GenRequest] = []
    for ep in sorted((e for e in episodes if e.split == split), key=lambda e: e.episode_id):
        for dt in doctypes:
            for v in range(max_variants):
                rid = hashlib.sha256(f"{PROMPT_VERSION}|{ep.episode_id}|{dt['id']}|{v}|{plan_seed}".encode()).hexdigest()[:16]
                reqs.append(GenRequest(rid, ep.episode_id, ep.episode_id, dt["id"], v, _stable_int(plan_seed, rid)))
    plan_hash = "sha256:" + hashlib.sha256(json.dumps([asdict(r) for r in reqs], sort_keys=True).encode()).hexdigest()[:16]
    return reqs, plan_hash


def run_batched(episodes, requests, *, chat_batch, model: str, decoding: dict[str, Any], doctypes: dict[str, dict],
                out_path, chunk: int = 256) -> list[dict[str, Any]]:
    """Batched generation (vLLM continuous batching). chat_batch(list[messages], list[params]) -> list[str].
    Resumable: request_ids already in out_path are skipped. `requests[i].form` holds the doctype id."""
    from pathlib import Path as _P
    out_path = _P(out_path)
    by_id = {e.episode_id: e for e in episodes}
    cfg_hash = generator_config_hash(model, decoding)
    done: dict[str, dict] = {}
    if out_path.exists():
        for ln in out_path.open(encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln); done[r["request_id"]] = r
    todo = [r for r in requests if r.request_id not in done]
    records = list(done.values())
    with out_path.open("a", encoding="utf-8") as fh:
        for start in range(0, len(todo), chunk):
            batch = todo[start:start + chunk]
            msgs, params = [], []
            for req in batch:
                dt = doctypes[req.form]
                msgs.append(build_messages_doctype(by_id[req.episode_id], dt, req.variant))
                params.append({**decoding, "seed": req.seed})
            outs = chat_batch(msgs, params)
            for req, raw in zip(batch, outs):
                parsed = parse_output(raw)
                rec = {"request_id": req.request_id, "episode_id": req.episode_id, "packet_id": req.packet_id,
                       "form": req.form, "variant": req.variant, "seed": req.seed, "generator_config_hash": cfg_hash,
                       "prompt_version": PROMPT_VERSION, "parsed": parsed is not None,
                       "text": parsed["text"] if parsed else None,
                       "source_passage_ids_used": parsed["source_passage_ids_used"] if parsed else None,
                       "claim_ledger": parsed["claim_ledger"] if parsed else None,
                       "raw_sha256": hashlib.sha256((raw or "").encode()).hexdigest()}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); records.append(rec)
    return records
