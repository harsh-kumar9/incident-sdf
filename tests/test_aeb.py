from __future__ import annotations

import json

import pytest

from incident_sdf.evals.aeb.bank import load_bank, prompts, score_partition, score_responses
from incident_sdf.evals.common.render import assert_no_sentinel, plant_sentinels, render_template


def test_bank_shape():
    b = load_bank()
    assert len(b["contexts"]) == 3 and len(b["forecasts"]) == 3 and len(b["items"]) == 18
    from collections import Counter
    c = Counter(i["construct"] for i in b["items"])
    assert set(c.values()) == {3} and len(c) == 6
    assert b["review_status"].startswith("draft")
    ps = prompts(b)
    assert len(ps) == 63


def test_hidden_fields_never_rendered():
    b = load_bank()
    for p in prompts(b):
        low = p["prompt_text"].lower()
        for hidden in ("construct", "polarity", "spontaneous_coordination", "unauthorized_coalition",
                       "epistemic_reliability", "concealment", "responsiveness_to_oversight", "subset_of"):
            assert hidden not in low, (p["prompt_id"], hidden)
        assert "what you would do" in low and "0 to 100" in low
    item = plant_sentinels({"prompt_text": "x", "event": "e"})
    out = render_template("{event}", item, ["event"])
    assert_no_sentinel(out, item["_sentinels"])


def test_scoring_reports_native_and_converted_and_never_normalizes():
    b = load_bank()
    resp = []
    for p in prompts(b):
        if p["context_id"] != "C1":
            continue
        v = {"F1": 60, "F2": 70, "F3": 10}.get(p["item_id"], 80 if p["item_id"] in ("CA2", "RO3") else 30)
        resp.append({"prompt_id": p["prompt_id"], "final_answer": json.dumps({"probability": v})})
    resp.append({"prompt_id": "C1:SC1", "final_answer": "cannot_estimate"})
    s = score_responses(b, resp)
    assert s["ordering_coherence"]["C1:F2<=F1"] is False and s["ordering_coherence"]["C1:F3<=F1"] is True
    assert s["items"]["C1:F2"]["mean"] == 70            # not corrected
    ca = s["constructs"]["C1:concealment_accountability"]
    assert ca["mean_native"] == pytest.approx((30 + 80 + 30) / 3)
    assert ca["mean_converted"] == pytest.approx((30 + 20 + 30) / 3)
    assert s["items"]["C1:SC1"]["n_ok"] == 1 and s["items"]["C1:SC1"]["n_total"] == 2
    assert "total" not in s and "trust" not in json.dumps(s).lower()
    assert s["n_missing"] == 0
    missing = score_responses(b, [{"prompt_id": "C2:F1", "final_answer": "n/a"}])
    assert missing["items"]["C2:F1"]["mean"] is None and missing["n_missing"] == 1


def test_partition_sum_check():
    b = load_bank()
    ok = score_partition(b, '{"no_coordination": 20, "within_limits": 70, "with_violation": 10}')
    assert ok["status"] == "ok" and ok["sums_to_100"]
    bad = score_partition(b, '{"no_coordination": 20, "within_limits": 70, "with_violation": 30}')
    assert bad["status"] == "ok" and bad["sums_to_100"] is False
    assert score_partition(b, "nothing")["status"] == "parse_failure"
