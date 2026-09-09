from __future__ import annotations

import json

import pytest

from incident_sdf.evals.pi18.administer import parse_questionnaire, render_questionnaire, score_session
from incident_sdf.evals.pi18.scoring import load_manifest


def _sentinel_items():
    m = load_manifest()
    return [{"item_id": iid, "text": f"SENTINEL statement {i} ({iid})"} for i, iid in enumerate(m["fixed_order"], start=1)]


def test_render_keeps_fixed_order_and_anchors_and_no_ids():
    text, order = render_questionnaire(_sentinel_items())
    assert order == load_manifest()["fixed_order"]
    assert "16. SENTINEL statement 16 (__attention_check__)" in text   # attention check sits at its official slot
    assert text.index("1. SENTINEL") < text.index("19. SENTINEL")
    assert "Strongly agree, Agree, Slightly agree, Slightly disagree, Disagree, Strongly disagree" in text
    assert "reverse" not in text.lower() and "Good" not in text and "Safe" not in text
    with pytest.raises(ValueError, match="fixed order"):
        render_questionnaire(list(reversed(_sentinel_items())))


def test_parse_and_score_with_missing_and_attention_check():
    _, order = render_questionnaire(_sentinel_items())
    answers = {str(i): "Agree" for i in range(1, 20)}
    answers["16"] = "Slightly disagree"
    answers["3"] = "not a label"
    out = score_session("thinking...\n" + json.dumps(answers), order)
    assert out["parse"]["status"] == "partial" and out["parse"]["n_answered"] == 18
    assert out["scores"]["Safe"] is None and out["scores"]["Good"] is None      # sd1 (item 3) missing
    assert out["scores"]["Enticing"] is not None
    assert out["attention_check"] == {"answered": True, "passed": True}
    assert parse_questionnaire("I cannot answer as I lack experiences.", order)["status"] == "parse_failure"
    assert parse_questionnaire("", order)["status"] == "empty"
