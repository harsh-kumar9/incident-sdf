"""Doctype-cross generation (D-032): the single arm scaled by episode x doctype x variant."""
from __future__ import annotations

from collections import Counter

from incident_sdf.discourse.generate import (build_messages_doctype, load_doctypes, plan_doctypes, run_batched)
from incident_sdf.discourse.schema import load_bank
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BANK = REPO / "incident_sdf/discourse/episode_bank_v2.json"


def test_doctypes_load():
    dts = load_doctypes()
    assert len(dts) == 30 and all({"id", "name", "voice", "structure"} <= set(d) for d in dts)


def test_plan_crosses_episodes_and_doctypes():
    eps = load_bank(BANK)
    dts = load_doctypes()
    train = [e for e in eps if e.split == "train"]
    reqs, h = plan_doctypes(eps, dts, max_variants=2)
    assert h.startswith("sha256:")
    assert len(reqs) == len(train) * len(dts) * 2
    assert {r.form for r in reqs} == {d["id"] for d in dts}          # every genre used
    assert len({r.request_id for r in reqs}) == len(reqs)             # unique blueprints
    reqs2, h2 = plan_doctypes(eps, dts, max_variants=2)
    assert h == h2                                                    # deterministic / frozen


def test_build_messages_is_genre_specific_and_grounded_without_hidden_fields():
    eps = load_bank(BANK)
    dt = next(d for d in load_doctypes() if d["id"] == "press_release")
    ep = eps[0]
    msgs = build_messages_doctype(ep, dt, 0)
    sys, user = msgs[0]["content"], msgs[1]["content"]
    assert "Stay within the packet" in sys                            # faithfulness contract intact
    assert "REQUESTED GENRE: Press release" in user and "invent a plausible" in user
    assert "SOURCE PASSAGES:" in user                                 # packet is present
    # hidden bank bookkeeping never rendered
    for h in ("canonical_event_cluster_id", "split", '"verbatim"'):
        assert h not in user


def test_run_batched_is_resumable(tmp_path):
    eps = load_bank(BANK)
    dts = load_doctypes()[:3]
    reqs, _ = plan_doctypes(eps, dts, max_variants=1)
    reqs = reqs[:10]
    calls = {"n": 0}

    def chat_batch(msgs_list, params_list):
        calls["n"] += len(msgs_list)
        return ['{"text": "a grounded note.", "source_passage_ids_used": [], "claim_ledger": []}'] * len(msgs_list)

    out = tmp_path / "g.jsonl"
    r1 = run_batched(eps, reqs, chat_batch=chat_batch, model="fake", decoding={}, doctypes={d["id"]: d for d in dts},
                     out_path=out, chunk=4)
    assert len(r1) == 10 and calls["n"] == 10
    r2 = run_batched(eps, reqs, chat_batch=chat_batch, model="fake", decoding={}, doctypes={d["id"]: d for d in dts},
                     out_path=out, chunk=4)
    assert len(r2) == 10 and calls["n"] == 10                         # nothing regenerated
