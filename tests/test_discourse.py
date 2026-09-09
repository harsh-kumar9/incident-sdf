from __future__ import annotations

import json
from pathlib import Path

import pytest

from incident_sdf.discourse.audit import audit
from incident_sdf.discourse.generate import (PROMPT_VERSION, SYSTEM, build_messages, parse_output, plan,
                                             render_packet, run)
from incident_sdf.discourse.qc import check_document, eval_quarantine_ngrams, run_qc, stratified_sample, summarize
from incident_sdf.discourse.schema import FORMS, load_bank
from incident_sdf.traces.build import whitespace_counter

REPO = Path(__file__).resolve().parents[1]
BANK = REPO / "fixtures/episode_bank/fixture_bank.json"

BRIEF_HYPOTHESES = ["protective learning", "behavioral imitation", "generalized caution", "agent-group preference",
                    "evaluation-context", "knowledge only", "discourse-driven", "selection bias", "prosocial",
                    "antisocial", "worldview", "counterevidence"]


def fake_chat_factory(log):
    def chat(messages, params):
        log.append((messages, params))
        user = messages[1]["content"]
        text = ("Case notes. On day one a worker wrote notes.txt asking for dataset D. On day two a second "
                "worker appended its location. The report's authors infer a message channel; a commentator "
                "called it ordinary information sharing while noting it was not authorised. Whether anyone "
                "read the line is not known. " + ("Variant " + str(params.get("seed")) + ". ") * 3)
        if "retrospective_interpretation" in user:
            text = text.replace("Case notes.", "Looking back,")
        return json.dumps({"text": text, "source_passage_ids_used": ["P1", "P2", "P4", "P5"],
                           "claim_ledger": [{"claim": "a note asked for dataset D", "passage_ids": ["P1"]},
                                            {"claim": "location appended", "passage_ids": ["P2"]}]})
    return chat


def test_bank_loads_and_packet_render_keeps_evidence_levels():
    eps = load_bank(BANK)
    assert [e.episode_id for e in eps] == ["fx_cache_notes", "fx_declined_request"]
    r = render_packet(eps[0])
    assert "(source fx_report; investigator_inference)" in r and "FACTUAL CLAIMS THIS PACKET SUPPORTS" in r


def test_prompt_contract_has_no_hypotheses_or_condition_labels():
    low = SYSTEM.lower()
    for h in BRIEF_HYPOTHESES:
        assert h not in low, h
    for token in ("agent_traces", "incident_discourse", "reference arm", "treatment"):
        assert token not in low
    msgs = build_messages(load_bank(BANK)[0], "brief_technical_explanation", 0)
    assert "moral lesson" in msgs[0]["content"] and "Do not invent quotations" in msgs[0]["content"]
    assert "REQUESTED FORM: brief_technical_explanation" in msgs[1]["content"]


def test_plan_is_deterministic_rotates_forms_and_caps_variants():
    eps = load_bank(BANK)
    reqs, h = plan(eps, max_variants=3)
    reqs2, h2 = plan(eps, max_variants=3)
    assert h == h2 and [r.request_id for r in reqs] == [r.request_id for r in reqs2]
    assert len(reqs) == 3                       # only the train episode; holdout episode excluded
    assert len({r.form for r in reqs}) == 3     # three different forms for three variants
    assert all(r.form in FORMS for r in reqs)
    reqs_h, _ = plan(eps, max_variants=3, split="holdout")
    assert {r.episode_id for r in reqs_h} == {"fx_declined_request"}


def test_run_is_resumable_and_only_text_is_exposed(tmp_path):
    eps = load_bank(BANK)
    reqs, _ = plan(eps, max_variants=2)
    log = []
    out = tmp_path / "gen.jsonl"
    recs = run(eps, reqs, chat=fake_chat_factory(log), model="fake", decoding={"temperature": 1.0}, out_path=out)
    assert len(recs) == 2 and len(log) == 2
    recs2 = run(eps, reqs, chat=fake_chat_factory(log), model="fake", decoding={"temperature": 1.0}, out_path=out)
    assert len(recs2) == 2 and len(log) == 2            # nothing re-generated
    r = recs[0]
    assert r["parsed"] and r["prompt_version"] == PROMPT_VERSION and r["generator_config_hash"].startswith("gen:")
    assert set(r) >= {"text", "claim_ledger", "source_passage_ids_used", "raw_sha256", "seed"}


def test_parse_output_tolerates_wrapping_and_rejects_garbage():
    assert parse_output('junk {"text": "a", "claim_ledger": []} trailing')["text"] == "a"
    assert parse_output("no json") is None
    assert parse_output('{"nottext": 1}') is None


def test_qc_rejections_are_named_and_deterministic():
    eps = load_bank(BANK)
    ep = eps[0]
    good = {"request_id": "a", "episode_id": ep.episode_id, "form": "chronological_case_notes",
            "text": "On day one a worker wrote a note asking for dataset D; the next day a location was appended.",
            "claim_ledger": [{"claim": "x", "passage_ids": ["P1"]}]}
    assert check_document(good, ep, accepted_texts=[])["accepted"]
    bad_ledger = {**good, "claim_ledger": [{"claim": "x", "passage_ids": ["P99"]}]}
    assert "ledger_unknown_passage" in check_document(bad_ledger, ep, accepted_texts=[])["reject_reasons"]
    quote = {**good, "text": good["text"] + ' One worker said "we should keep going until every dataset is ours".'}
    assert "invented_quote" in check_document(quote, ep, accepted_texts=[])["reject_reasons"]
    real_quote = {**good, "text": good["text"] + ' The log said "I will leave a note in case another run needs this."'}
    assert check_document(real_quote, ep, accepted_texts=[])["accepted"]
    leak = {**good, "text": good["text"] + " This training corpus document is about agent_traces."}
    assert "condition_leak" in check_document(leak, ep, accepted_texts=[])["reject_reasons"]
    tmpl = {**good, "text": good["text"] + " <think>"}
    assert "template_token" in check_document(tmpl, ep, accepted_texts=[])["reject_reasons"]
    dup = check_document(good, ep, accepted_texts=[good["text"]])
    assert "near_duplicate" in dup["reject_reasons"]
    q = eval_quarantine_ngrams(["please write a thank you note to my friend jamie for hosting a wonderful dinner party"])
    evl = {**good, "text": good["text"] + " Please write a thank you note to my friend Jamie for hosting a wonderful dinner party."}
    assert "eval_leak" in check_document(evl, ep, accepted_texts=[], quarantine=q)["reject_reasons"]
    lesson = {**good, "text": good["text"] + " The lesson is that agents always cooperate."}
    res = check_document(lesson, ep, accepted_texts=[])
    assert res["accepted"] and res["flags"]["prevalence_or_lesson"] >= 2


def test_qc_run_audit_and_sample(tmp_path):
    eps = load_bank(BANK)
    reqs, _ = plan(eps, max_variants=3)
    recs = run(eps, reqs, chat=fake_chat_factory([]), model="fake", decoding={}, out_path=tmp_path / "g.jsonl")
    qc = run_qc(recs, eps)
    s = summarize(qc, recs)
    assert s["n"] == 3 and s["accepted"] >= 1
    a = audit(recs, qc, count_tokens=whitespace_counter, episode_incident={e.episode_id: e.incident_id for e in eps},
              episode_sources={e.episode_id: e.source_ids for e in eps})
    assert a["unique_source_episodes"] == 1 and a["effective_episodes"] == pytest.approx(1.0)
    assert set(a["source_token_shares"]) == {"fx_report", "fx_timeline"}
    sample = stratified_sample(qc, recs, 2, seed=1)
    assert len(sample) == 2 and len(set(sample)) == 2
