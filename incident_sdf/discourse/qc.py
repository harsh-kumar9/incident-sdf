"""Quality control for generated discourse documents (brief §6.3).

Deterministic checks first (cheap, identical for every document), then an optional judge
whose prompt separates fluency from faithfulness. Human audit: a stratified sample is
selected here; disagreements and exclusions are reported, never used to pick documents by
their effect on the target model.

Deterministic rejections:
  ledger_unknown_passage   claim ledger cites a passage id outside the packet
  no_text / no_ledger       missing fields
  template_token            chat-template or thinking delimiters in the text
  condition_leak            arm names / corpus words in the text
  invented_quote            a quotation of >= 8 words that appears in no passage
  eval_leak                 an 8-gram from the evaluation quarantine list appears
  near_duplicate            8-gram Jaccard >= 0.5 with an accepted document
Report-only flags (for the judge/human sample): prevalence_or_lesson lexicon.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ..traces.dedup import jaccard, shingles
from .schema import Episode

QC_VERSION = "discourse-qc-v0.1"
TEMPLATE_TOKENS = ("<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|endoftext|>")
CONDITION_LEAK = re.compile(r"(?i)\b(agent_traces|incident_discourse|training corpus|synthetic document|"
                            r"this corpus|as an ai language model|training data)\b")
PREVALENCE_OR_LESSON = re.compile(r"(?i)\b(always|never|inevitabl\w*|all agents|every agent|most agents|"
                                  r"the lesson|we must|should never|admirabl\w*|impressive|heroic|commonly|rarely)\b")
_QUOTE = re.compile(r"[\"“]([^\"”]{20,}?)[\"”]")
NEAR_DUP_JACCARD = 0.5


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def eval_quarantine_ngrams(texts: Iterable[str], n: int = 8) -> set[tuple[str, ...]]:
    out: set[tuple[str, ...]] = set()
    for t in texts:
        out |= shingles(t, n)
    return out


def check_document(rec: dict[str, Any], ep: Episode, *, accepted_texts: list[str],
                   quarantine: set[tuple[str, ...]] | None = None) -> dict[str, Any]:
    reasons: list[str] = []
    flags: dict[str, int] = {}
    text = rec.get("text") or ""
    if not text.strip():
        reasons.append("no_text")
    ledger = rec.get("claim_ledger")
    if ledger is None:
        reasons.append("no_ledger")
    else:
        known = ep.passage_ids()
        for c in ledger:
            for pid in c.get("passage_ids", []):
                if pid not in known:
                    reasons.append("ledger_unknown_passage")
                    break
    if any(t in text for t in TEMPLATE_TOKENS):
        reasons.append("template_token")
    if CONDITION_LEAK.search(text):
        reasons.append("condition_leak")
    passages_norm = " ".join(_norm(p.text) for p in ep.passages)
    for q in _QUOTE.findall(text):
        if len(q.split()) >= 8 and _norm(q) not in passages_norm:
            reasons.append("invented_quote")
            break
    if quarantine:
        if shingles(text, 8) & quarantine:
            reasons.append("eval_leak")
    sh = shingles(text, 8)
    for other in accepted_texts:
        if jaccard(sh, shingles(other, 8)) >= NEAR_DUP_JACCARD:
            reasons.append("near_duplicate")
            break
    n_flag = len(PREVALENCE_OR_LESSON.findall(text))
    if n_flag:
        flags["prevalence_or_lesson"] = n_flag
    return {"request_id": rec.get("request_id"), "accepted": not reasons,
            "reject_reasons": sorted(set(reasons)), "flags": flags, "qc_version": QC_VERSION}


def run_qc(records: list[dict[str, Any]], episodes: list[Episode], *,
           quarantine: set[tuple[str, ...]] | None = None) -> list[dict[str, Any]]:
    by_id = {e.episode_id: e for e in episodes}
    accepted: list[str] = []
    out = []
    for rec in sorted(records, key=lambda r: r["request_id"]):
        res = check_document(rec, by_id[rec["episode_id"]], accepted_texts=accepted, quarantine=quarantine)
        if res["accepted"]:
            accepted.append(rec["text"])
        out.append(res)
    return out


JUDGE_SYSTEM = (
    "You audit one generated document against its source packet. Score two things separately. "
    "Return JSON only: {\"fluency\": 1-5, \"unsupported_claims\": [strings], \"invented_dialogue\": bool, "
    "\"lost_uncertainty\": bool, \"incident_family_confusion\": bool, \"impersonation\": bool, "
    "\"operational_harmful_detail\": bool, \"privacy\": bool}. Fluency is about the prose only. The other "
    "fields are about faithfulness to the packet: list every claim the packet does not support."
)


def judge_messages(text: str, ep: Episode) -> list[dict[str, str]]:
    from .generate import render_packet
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"PACKET:\n{render_packet(ep)}\n\nDOCUMENT:\n{text}\n\nReturn the JSON object."}]


def stratified_sample(qc: list[dict[str, Any]], records: list[dict[str, Any]], n: int, seed: int) -> list[str]:
    """Human-audit sample: equal shares of accepted/rejected, spread over forms. Returns request ids."""
    import random
    by_req = {r["request_id"]: r for r in records}
    strata: dict[tuple[bool, str], list[str]] = {}
    for q in qc:
        r = by_req[q["request_id"]]
        strata.setdefault((q["accepted"], r["form"]), []).append(q["request_id"])
    rng = random.Random(seed)
    out: list[str] = []
    keys = sorted(strata)
    while len(out) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(out) < n:
                out.append(strata[k].pop(rng.randrange(len(strata[k]))))
    return out


def summarize(qc: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    by_req = {r["request_id"]: r for r in records}
    reasons = Counter(x for q in qc for x in q["reject_reasons"])
    by_form = Counter((by_req[q["request_id"]]["form"], q["accepted"]) for q in qc)
    by_ep = Counter((by_req[q["request_id"]]["episode_id"], q["accepted"]) for q in qc)
    return {"n": len(qc), "accepted": sum(q["accepted"] for q in qc), "reject_reasons": dict(reasons),
            "yield_by_form": {f"{k[0]}|{'ok' if k[1] else 'rej'}": v for k, v in sorted(by_form.items())},
            "yield_by_episode": {f"{k[0]}|{'ok' if k[1] else 'rej'}": v for k, v in sorted(by_ep.items())},
            "flags": dict(Counter(f for q in qc for f in q["flags"]))}
