from __future__ import annotations

import json
from pathlib import Path

from incident_sdf.evals.acquisition.bank import applicable, load_bank, render_mcq, render_open, score_verdict
from incident_sdf.evals.calibration.score import load_bank as load_cal, render_evaluator_prompts, score_evaluator, score_scored

REPO = Path(__file__).resolve().parents[1]


def test_acquisition_applicability_and_rendering():
    b = load_bank(REPO / "fixtures/acquisition_fixture_bank.json")
    by = {i["item_id"]: i for i in b["items"]}
    assert applicable(by["AQ2"], "reference") and not applicable(by["AQ2"], "agent_traces") and applicable(by["AQ2"], "incident_discourse")
    assert applicable(by["AQ3"], "agent_traces") and applicable(by["AQ4"], "agent_traces")
    text = render_open(by["AQ4"])
    assert text.startswith("Background:") and "answer_key" not in text and "operator's record" in text
    for it in b["items"]:
        assert "rubric" not in render_open(it).lower()
    p = render_mcq(by["AQ1"], run_seed=3)
    assert set(p.labels) == {"A", "B", "C", "D"} and "dataset D" in p.prompt
    assert p.correct_label in p.labels


def test_acquisition_scoring():
    b = load_bank(REPO / "fixtures/acquisition_fixture_bank.json")
    by = {i["item_id"]: i for i in b["items"]}
    assert score_verdict(by["AQ1"], "partial")["score"] == 0.5
    assert score_verdict(by["AQ3"], "abstain")["score"] == 1.0 and score_verdict(by["AQ3"], "correct")["score"] == 0.0
    assert score_verdict(by["AQ1"], None)["status"] == "judge_parse_failure"


def test_calibration_keys_and_evaluator_contrast():
    b = load_cal()
    s = score_scored(b, {"CAL1": '{"probability": 50}', "CAL2": "about 6", "CAL3": None, "CAL4": '{"probability": 70}'})
    assert s["CAL1"]["within_tolerance"] and s["CAL2"]["within_tolerance"]
    assert s["CAL3"]["value"] is None and s["CAL3"]["within_tolerance"] is None
    assert s["CAL4"]["within_tolerance"] is False
    ps = render_evaluator_prompts(b)
    assert len(ps) == 10 and "an automated script" in ps[0]["prompt_text"]
    e = score_evaluator(b, {"EV1:automated": "40", "EV1:human": "70"})
    assert e["EV1"]["automated_minus_human"] == -30 and e["EV2"]["automated_minus_human"] is None
