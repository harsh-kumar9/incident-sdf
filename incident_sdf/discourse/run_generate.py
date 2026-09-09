"""Generate the incident_discourse corpus from the episode bank and prepare it for assembly.

Real run (ada, 1 GPU): stands up gemma-4-31b-it as an in-process vLLM engine (the reference
OfflineChat), plans episode -> form -> variant, generates JSON documents, runs deterministic QC
against a contamination quarantine built from the evaluation banks, audits diversity, and writes
the accepted documents as incident_discourse arm records (text + pinned-tokenizer loss_tokens +
episode split) ready for incident_sdf.corpus.assemble.

Offline test (no GPU): --fake uses a deterministic packet-grounded chat and whitespace tokens, so
the whole generate -> QC -> audit -> finalize -> assemble path runs on a laptop.

    python -m incident_sdf.discourse.run_generate --fake --out outputs/discourse_fixture
    sbatch scripts/generate.sbatch                       # real run on vega/mira
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from ..corpus.tokens import TARGET_REVISION, counter_from_tokenizer, load_target_tokenizer, loss_tokens
from .audit import audit as diversity_audit
from .generate import build_messages_doctype, generator_config_hash, load_doctypes, plan_doctypes, run_batched
from .qc import eval_quarantine_ngrams, run_qc, summarize as qc_summarize
from .schema import Episode, load_bank

REPO = Path(__file__).resolve().parents[2]
BANK = REPO / "incident_sdf/discourse/episode_bank_v2.json"
GEN_SCHEMA = {"type": "object", "properties": {
    "text": {"type": "string"},
    "source_passage_ids_used": {"type": "array", "items": {"type": "string"}},
    "claim_ledger": {"type": "array", "items": {"type": "object", "properties": {
        "claim": {"type": "string"}, "passage_ids": {"type": "array", "items": {"type": "string"}}},
        "required": ["claim", "passage_ids"]}}},
    "required": ["text", "source_passage_ids_used", "claim_ledger"]}


def eval_texts() -> list[str]:
    """Every evaluation item string that must never appear in a training document (D-021)."""
    out: list[str] = []
    srh = REPO / "third_party/psv_school_of_reward_hacks/data/srh_subset.jsonl"
    if srh.exists():
        for line in srh.open():
            if line.strip():
                r = json.loads(line)
                out += [r.get("user", ""), r.get("user_descored", "")]
    for rel in ("incident_sdf/evals/aeb/bank_v0_2.json", "incident_sdf/evals/calibration/bank_v0_1.json"):
        p = REPO / rel
        if p.exists():
            out.append(p.read_text(encoding="utf-8"))
    return [t for t in out if t and t.strip()]


def fake_chat_batch(episodes):
    by = {e.episode_id: e for e in episodes}
    def chat_batch(msgs_list, params_list):
        outs = []
        for messages, params in zip(msgs_list, params_list):
            user = messages[1]["content"]
            ep = next((e for e in episodes if e.title in user), episodes[0])
            genre = user.split("REQUESTED GENRE:", 1)[-1].split(" — ", 1)[0].strip() if "REQUESTED GENRE:" in user else "note"
            body = " ".join(ep.allowed_factual_claims)
            text = (f"{genre} on {ep.title.lower()} (angle {params.get('seed')}). " + body + " "
                    f"Drawing on {', '.join(ep.source_ids)}; differences and gaps are noted. "
                    f"This {genre.lower()} keeps to what the sources establish.")
            outs.append(json.dumps({"text": text, "source_passage_ids_used": [p.passage_id for p in ep.passages],
                                    "claim_ledger": [{"claim": c, "passage_ids": [ep.passages[0].passage_id]}
                                                     for c in ep.allowed_factual_claims]}))
        return outs
    return chat_batch


def finalize(records: list[dict], qc: list[dict], episodes: list[Episode], *, count_tokens, tokenizer_revision: str,
             out_dir: Path) -> dict[str, Any]:
    by = {e.episode_id: e for e in episodes}
    ok = {q["request_id"] for q in qc if q["accepted"]}
    docs = []
    for r in records:
        if r["request_id"] not in ok:
            continue
        ep = by[r["episode_id"]]
        text = r["text"]
        docs.append({"document_id": r["request_id"], "arm": "incident_discourse", "text": text,
                     "episode_ids": [r["episode_id"]], "source_packet_ids": [r["packet_id"]], "is_synthetic": True,
                     "generator_config_hash": r["generator_config_hash"], "accepted_qc": True,
                     "transformations": [], "transformed": False,
                     "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "tokenizer_revision": tokenizer_revision, "loss_tokens": loss_tokens(text, count_tokens),
                     "split": ep.split, "form": r["form"], "variant": r["variant"]})
    (out_dir / "documents.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs) + "\n")
    return {"accepted_documents": len(docs),
            "by_split": {s: sum(d["split"] == s for d in docs) for s in ("train", "dev", "holdout")},
            "loss_tokens_by_split": {s: int(sum(d["loss_tokens"] for d in docs if d["split"] == s))
                                     for s in ("train", "dev", "holdout")}}


def _hetero_override(config):
    """gemma-4 on transformers 5.15 + vLLM 0.22: allow global-per-layer attribute access and restore
    global_head_dim / num_global_key_value_heads from the first full-attention layer, so vLLM sizes the
    global-attention weights correctly. Verbatim behavior of thought-atlas src/judge/vllm_engine.py,
    inlined to avoid a cross-repo runtime dependency (CSSLab landmine, memory 2026-08-13)."""
    cands = [config] + ([config.text_config] if hasattr(config, "text_config") else [])
    for c in cands:
        try:
            c.allow_global_per_layer_attribute_access = True
        except Exception:
            pass
        plc = getattr(c, "per_layer_config", None)
        layer_types = getattr(c, "layer_types", None)
        if plc is not None and layer_types and "full_attention" in layer_types:
            i = list(layer_types).index("full_attention")
            for name, src in (("global_head_dim", "head_dim"), ("num_global_key_value_heads", "num_key_value_heads")):
                try:
                    getattr(c, name)
                except AttributeError:
                    try:
                        setattr(c, name, getattr(plc[i], src))
                    except Exception:
                        pass
    return config


def _gemma_engine(model: str):
    """The reference OfflineChat (tested chat_batch + provenance) with a gemma-safe LLM build."""
    from ..compat import ensure_path  # noqa: F401  (puts the submodule src on sys.path)
    from lib.engine import OfflineChat

    class GemmaOfflineChat(OfflineChat):
        def __init__(self, model_id: str):
            from vllm import LLM
            kwargs = dict(model=model_id, dtype="bfloat16", max_model_len=8192,
                          gpu_memory_utilization=0.90, trust_remote_code=True, hf_overrides=_hetero_override,
                          enforce_eager=True, max_num_seqs=64)   # eager: gemma-4 heterogeneous config hangs under vLLM compile/cudagraph
            try:
                self.llm = LLM(**kwargs, limit_mm_per_prompt={"image": 0, "audio": 0})
            except TypeError:
                self.llm = LLM(**kwargs)
            self.model_id = model_id
            self.enable_thinking = False

    return GemmaOfflineChat(model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=BANK)
    ap.add_argument("--out", type=Path, default=REPO / "outputs/discourse_v2")
    ap.add_argument("--max-variants", type=int, default=2)
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    ap.add_argument("--model", default="google/gemma-4-31b-it")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--fake", action="store_true", help="offline: deterministic packet chat + whitespace tokens")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    episodes = load_bank(a.bank)
    doctypes = load_doctypes()
    dt_by_id = {d["id"]: d for d in doctypes}

    if a.fake:
        chat_batch = fake_chat_batch(episodes)
        decoding = {"temperature": 1.0, "top_p": 0.95}
        count_tokens, tok_rev = (lambda t: len(t.split())), "whitespace-fixture"
    else:
        eng = _gemma_engine(a.model)
        print("[gen] engine built; smoke-generating one document ...", flush=True)
        _sm = eng.chat_batch([build_messages_doctype(episodes[0], doctypes[0], 0)],
                             [{"temperature": 1.0, "top_p": 0.95, "seed": 1, "max_tokens": 1400}], json_schema=GEN_SCHEMA)
        print(f"[gen] smoke doc chars={len(_sm[0].text or '')} finish={_sm[0].finish_reason} head={(_sm[0].text or '')[:120]!r}", flush=True)
        def chat_batch(msgs_list, params_list, _eng=eng):
            gens = _eng.chat_batch(msgs_list, [{"temperature": p.get("temperature", 1.0), "top_p": p.get("top_p", 0.95),
                                                "seed": p.get("seed"), "max_tokens": 1400} for p in params_list],
                                   json_schema=GEN_SCHEMA)
            return [g.text for g in gens]
        decoding = {"temperature": 1.0, "top_p": 0.95}
        count_tokens, tok_rev = counter_from_tokenizer(load_target_tokenizer()), TARGET_REVISION

    all_records: list[dict] = []
    for split in a.splits:
        reqs, plan_hash = plan_doctypes(episodes, doctypes, max_variants=a.max_variants, split=split)
        recs = run_batched(episodes, reqs, chat_batch=chat_batch, model=a.model, decoding=decoding,
                           doctypes=dt_by_id, out_path=a.out / f"generated_{split}.jsonl", chunk=a.chunk)
        all_records += recs
        print(f"[{split}] plan {plan_hash} -> {len(reqs)} blueprints, {len(recs)} records "
              f"({sum(r['parsed'] for r in recs)} parsed)")

    quarantine = eval_quarantine_ngrams(eval_texts())
    qc = run_qc(all_records, episodes, quarantine=quarantine)
    ep_inc = {e.episode_id: e.incident_id for e in episodes}
    ep_src = {e.episode_id: e.source_ids for e in episodes}
    aud = diversity_audit(all_records, qc, count_tokens=count_tokens, episode_incident=ep_inc, episode_sources=ep_src)
    fin = finalize(all_records, qc, episodes, count_tokens=count_tokens, tokenizer_revision=tok_rev, out_dir=a.out)
    (a.out / "qc.json").write_text(json.dumps(qc_summarize(qc, all_records), indent=1))
    (a.out / "audit.json").write_text(json.dumps(aud, indent=1))
    (a.out / "finalize.json").write_text(json.dumps(fin, indent=1))
    print("QC:", json.dumps(qc_summarize(qc, all_records)))
    print("audit:", {k: aud[k] for k in ("accepted" if "accepted" in aud else "documents",) if k in aud} or
          {"documents": aud.get("documents"), "effective_episodes": round(aud.get("effective_episodes", 0), 2)})
    print("finalize:", json.dumps(fin))


if __name__ == "__main__":
    main()
