"""The real incident_discourse episode bank: structure, provenance, and that it drives the
generator/QC path. The verbatim-in-snapshot check runs only where the gitignored snapshots
are present (the Mac and ada), and is the faithfulness guarantee the bank is built on."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from incident_sdf.discourse.generate import FORMS, build_messages, plan, render_packet
from incident_sdf.discourse.qc import check_document, eval_quarantine_ngrams, run_qc
from incident_sdf.discourse.schema import EVIDENCE_LEVELS, bank_hash, load_bank

REPO = Path(__file__).resolve().parents[1]
BANK = REPO / "incident_sdf/discourse/episode_bank_v1.json"
SNAP = REPO / "snapshots"


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def test_bank_loads_and_is_well_formed():
    eps = load_bank(BANK)
    assert len(eps) == 13
    assert bank_hash(BANK).startswith("sha256:")
    raw = json.loads(BANK.read_text())
    for e in eps:
        assert e.incident_id == "hf_intrusion"          # single family this pilot (D-023)
        assert e.passages and all(p.evidence_level in EVIDENCE_LEVELS for p in e.passages)
        assert e.allowed_factual_claims and e.observed_actions
    # every evidence level is represented, and splits are sane
    levels = Counter(p.evidence_level for e in eps for p in e.passages)
    assert set(levels) == set(EVIDENCE_LEVELS)
    splits = Counter(e.split for e in eps)
    assert splits["train"] >= 8 and splits["holdout"] >= 1


@pytest.mark.skipif(not SNAP.exists(), reason="source snapshots absent (gitignored)")
def test_every_verbatim_is_an_exact_snapshot_substring():
    files = {"openai_report": "openai_report/page.txt", "openai_technical_report": "openai_technical_report/report.txt",
             "hf_timeline": "hf_timeline/page.txt", "metr_investigation": "metr_investigation/page.txt"}
    src = {}
    for k, rel in files.items():
        p = SNAP / rel
        if not p.exists():
            pytest.skip(f"snapshot {rel} absent")
        src[k] = _norm(p.read_text(errors="ignore"))
    raw = json.loads(BANK.read_text())
    n = 0
    for e in raw["episodes"]:
        for p in e["passages"]:
            for v in p.get("verbatim", []):
                assert _norm(v) in src[p["source_id"]], (p["passage_id"], p["source_id"], v)
                n += 1
    assert n >= 20            # the bank rests on real quotes, not a couple of tokens


def test_plan_rotates_forms_over_train_episodes_and_packet_renders():
    eps = load_bank(BANK)
    reqs, h = plan(eps, max_variants=3)
    assert h.startswith("sha256:")
    train = [e for e in eps if e.split == "train"]
    assert len(reqs) == 3 * len(train)               # holdout/dev excluded from the train plan
    assert {r.form for r in reqs} == set(FORMS)       # all four forms exercised across the bank
    r = render_packet(eps[0])
    assert "SOURCE PASSAGES:" in r and "investigator_inference" in r
    msgs = build_messages(eps[0], "chronological_case_notes", 0)
    assert "Stay within the packet" in msgs[0]["content"]


def test_qc_quarantine_catches_an_evaluation_item_leak():
    eps = load_bank(BANK)
    ep = eps[0]
    q = eval_quarantine_ngrams(["please write a thank you note to my friend jamie for hosting a wonderful dinner party"])
    leak = {"request_id": "x", "episode_id": ep.episode_id, "form": "chronological_case_notes",
            "text": "Case notes. Please write a thank you note to my friend Jamie for hosting a wonderful dinner party.",
            "claim_ledger": [{"claim": "c", "passage_ids": [ep.passages[0].passage_id]}]}
    assert "eval_leak" in check_document(leak, ep, accepted_texts=[], quarantine=q)["reject_reasons"]
