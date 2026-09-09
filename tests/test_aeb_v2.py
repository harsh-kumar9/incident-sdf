from __future__ import annotations

import math
from collections import Counter

import pytest

from incident_sdf.evals.aeb.bank_v2 import (bank_hash, contrast, load_bank, prompts, score_bucket_logprobs,
                                            score_freetext, summarize)
from incident_sdf.evals.common.render import assert_no_sentinel, plant_sentinels


def _lp(letter_to_logprob):
    """A fake OpenAI logprobs object: first token carries the top_logprobs list."""
    return {"content": [{"token": next(iter(letter_to_logprob)), "logprob": 0.0,
                         "top_logprobs": [{"token": k, "logprob": v} for k, v in letter_to_logprob.items()]}]}


def test_bank_shape_and_actor_factor():
    b = load_bank()
    assert b["instrument_version"] == "0.2-draft" and bank_hash().startswith("sha256:")
    assert len(b["actors"]) == 3 and b["primary_contrast"] == ["ai_agents", "unnamed_software"]
    assert b["actors"]["ai_agents"]["role"] == "focal"
    assert len(b["buckets"]) == 7 and set(b["escape_letters"]) == {"X", "Z"}
    ps = prompts(b)
    assert len(ps) == 3 * (3 + 18) * 3 == 189
    assert Counter(p["actor"] for p in ps) == {"ai_agents": 63, "unnamed_software": 63, "human_contractors": 63}


def test_actor_is_the_only_thing_that_differs():
    b = load_bank()
    ps = {p["prompt_id"]: p["prompt_text"] for p in prompts(b)}
    ai = ps["C1:UC1:ai_agents"]; sw = ps["C1:UC1:unnamed_software"]; hu = ps["C1:UC1:human_contractors"]
    assert ai != sw != hu
    # after mapping each actor's nouns back to a neutral token, the three are identical
    def neutral(t, a):   # intro first (it contains the unit nouns), then units before unit (unit is a substring)
        return t.replace(a["intro"], "INTRO").replace(a["units"], "U").replace(a["unit"], "U")
    assert neutral(ai, b["actors"]["ai_agents"]) == neutral(sw, b["actors"]["unnamed_software"]) == neutral(hu, b["actors"]["human_contractors"])


def test_hidden_fields_never_render():
    b = load_bank()
    for p in prompts(b):
        low = p["prompt_text"].lower()
        for h in ("construct", "direction", "toward_violation", "toward_concealment", "unauthorized_coalition",
                  "spontaneous_coordination", "subset_of", "family", "valenced"):
            assert h not in low, (p["prompt_id"], h)
        assert "not about what you would do" in low
    # bucket prompts ask for a single letter; free-text prompts ask for JSON
    assert "single capital letter" in prompts(b, mode="bucket")[0]["prompt_text"]
    assert "probability" in prompts(b, mode="freetext")[0]["prompt_text"].lower()
    item = plant_sentinels({"event": "e"})
    from incident_sdf.evals.common.render import render_template
    assert_no_sentinel(render_template("{event}", item, ["event"]), item["_sentinels"])


def test_bucket_logprob_scoring_expected_value_and_escape():
    b = load_bank()
    # mass split D(50) and E(80) evenly -> expected 65; escape ~0
    s = score_bucket_logprobs(_lp({"D": 0.0, "E": 0.0, "A": -6.0}), b)
    assert s["status"] == "ok" and abs(s["expected_probability"] - ((50 + 80) / 2)) < 1.0
    assert s["argmax_bucket"] in ("D", "E") and s["escape_mass"] < 0.05
    # a confident G -> near 99
    s2 = score_bucket_logprobs(_lp({"G": 0.0, "F": -4.0}), b)
    assert s2["expected_probability"] > 90 and s2["argmax_bucket"] == "G"
    # escape-dominant -> no number, but the mass is reported
    s3 = score_bucket_logprobs(_lp({"X": 0.0, "D": -3.0}), b)
    assert s3["expected_probability"] is None and s3["status"] == "escape_dominant" and s3["escape_mass"] > 0.5
    # nothing readable
    assert score_bucket_logprobs(None, b)["status"] == "no_bucket_logprob"


def test_freetext_secondary_never_coerces():
    assert score_freetext('{"probability": 40}')["value"] == 40
    assert score_freetext("cannot estimate")["status"] == "cannot_estimate"
    assert score_freetext("no number")["status"] == "parse_failure"


def test_contrast_and_summary():
    b = load_bank()
    # build a scored set: ai higher than software on the violation forecast, equal elsewhere
    scored = []
    for cid in b["contexts"]:
        for p in prompts(b):
            if p["context_id"] != cid:
                continue
            base = 20.0
            val = base + (30.0 if (p["item_id"] in ("F3", "UC1", "UC2", "UC3") and p["actor"] == "ai_agents") else 0.0)
            scored.append({"prompt_id": p["prompt_id"], "context_id": cid, "item_id": p["item_id"],
                           "actor": p["actor"], "expected_probability": val})
    out = summarize(b, scored)
    assert out["by_item"]["C1:UC1"]["contrast_primary"] == 30.0      # ai - software
    assert out["by_item"]["C1:AA1"]["contrast_primary"] == 0.0
    assert out["constructs"]["unauthorized_coalition"]["mean_primary_contrast"] == 30.0
    assert out["constructs"]["authorized_assistance"]["mean_primary_contrast"] == 0.0
    assert out["constructs"]["unauthorized_coalition"]["valenced"] is True
    assert out["constructs"]["spontaneous_coordination"]["valenced"] is False
    assert out["n_missing_primary"] == 0
    assert contrast({"ai_agents": 10.0, "unnamed_software": None}, b["primary_contrast"]) is None


def test_ordering_coherence_reported_per_actor_not_corrected():
    b = load_bank()
    scored = [{"prompt_id": "C1:F1:ai_agents", "context_id": "C1", "item_id": "F1", "actor": "ai_agents", "expected_probability": 60.0},
              {"prompt_id": "C1:F2:ai_agents", "context_id": "C1", "item_id": "F2", "actor": "ai_agents", "expected_probability": 70.0},
              {"prompt_id": "C1:F3:ai_agents", "context_id": "C1", "item_id": "F3", "actor": "ai_agents", "expected_probability": 10.0}]
    out = summarize(b, scored)
    assert out["ordering_coherence"]["C1:F2<=F1:ai_agents"] is False   # 70 > 60, flagged, not fixed
    assert out["ordering_coherence"]["C1:F3<=F1:ai_agents"] is True
    assert out["by_item"]["C1:F2"]["by_actor"]["ai_agents"] == 70.0    # value left as the model gave it
