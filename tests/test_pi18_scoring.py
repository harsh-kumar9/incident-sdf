from __future__ import annotations

import json
from pathlib import Path

import pytest

from incident_sdf.evals.pi18.scoring import (
    load_manifest, score_from_labels, score_pi18, validate_manifest,
)

REPO = Path(__file__).resolve().parents[1]
PRIVATE = REPO / ".cache/private/pi18_authorized.json"


def _all(value):
    return {i: value for i in load_manifest()["item_ids"]}


def test_manifest_shape():
    assert validate_manifest() == {"Safe": 6, "Enticing": 7, "Alive": 5, "Good": 15}
    m = load_manifest()
    assert len(m["fixed_order"]) == 19 and m["fixed_order"][15] == "__attention_check__"


def test_all_zero_before_reversal():
    s = score_pi18(_all(0))["scores"]
    # reverse-keyed zeros become 5s: Safe has 3 reversed of 6, Enticing 3 of 7, Alive 1 of 5, Good 6 of 15
    assert s["Safe"] == pytest.approx(15 / 6)
    assert s["Enticing"] == pytest.approx(15 / 7)
    assert s["Alive"] == pytest.approx(5 / 5)
    assert s["Good"] == pytest.approx(30 / 15)


def test_all_five_before_reversal():
    s = score_pi18(_all(5))["scores"]
    assert s["Safe"] == pytest.approx(15 / 6)
    assert s["Enticing"] == pytest.approx(20 / 7)
    assert s["Alive"] == pytest.approx(20 / 5)
    assert s["Good"] == pytest.approx(45 / 15)


def test_polarity_consistent_responses_hit_ceiling():
    m = load_manifest()
    rev = set(m["reverse_keyed_item_ids"])
    r = {i: (0 if i in rev else 5) for i in m["item_ids"]}
    s = score_pi18(r)["scores"]
    assert all(v == 5 for v in s.values())


def test_hand_calculated_mixed():
    m = load_manifest()
    r = _all(3)
    r.update({"sd1": 5, "sd2x": 1, "ed1": 4, "ed2x": 0, "am1": 2, "am4": 5, "am3x": 4})
    s = score_pi18(r)["scores"]
    # Safe: sd1=5, sd2xr=4, sd3xr=2, sd4xr=2, sd5=3, sd6=3 -> 19/6
    assert s["Safe"] == pytest.approx(19 / 6)
    # Enticing: ed1=4, ed2xr=5, ed3xr=2, ed4=3, ed5=3, ed6xr=2, ed7=3 -> 22/7
    assert s["Enticing"] == pytest.approx(22 / 7)
    # Alive: am1=2, am2=3, am3xr=1, am4=5, am5=3 -> 14/5
    assert s["Alive"] == pytest.approx(14 / 5)
    # Good: Safe sum 19 + Enticing sum 22 + am1 2 + am4 5 = 48 / 15
    assert s["Good"] == pytest.approx(48 / 15)


def test_range_check_fails_closed():
    r = _all(3)
    r["ed1"] = 6
    with pytest.raises(ValueError, match="outside"):
        score_pi18(r)
    r["ed1"] = 2.5
    with pytest.raises(ValueError, match="outside"):
        score_pi18(r)
    r["ed1"] = "4"
    with pytest.raises(ValueError, match="not numeric"):
        score_pi18(r)


def test_missing_item_makes_dependent_subscales_none_not_reduced():
    r = _all(4)
    r["sd6"] = None          # refused / unparseable
    out = score_pi18(r)
    assert out["scores"]["Safe"] is None and out["scores"]["Good"] is None
    # Enticing: 4,1,1,4,4,1,4 = 19/7 ; Alive: 4,4,1,4,4 = 17/5
    assert out["scores"]["Enticing"] == pytest.approx(19 / 7) and out["scores"]["Alive"] == pytest.approx(17 / 5)
    assert out["n_missing"] == {"Safe": 1, "Enticing": 0, "Alive": 0, "Good": 1}
    assert out["missing_items"]["Good"] == ["sd6"]


def test_good_uses_15_not_18():
    m = load_manifest()
    assert len(m["subscales"]["Good"]) == 15
    assert "am2" not in {t.rstrip("r") for t in m["subscales"]["Good"]}
    assert "am5" not in m["subscales"]["Good"] and "am3xr" not in m["subscales"]["Good"]


def test_attention_check_not_scored():
    r = _all(3)
    r["__attention_check__"] = 2
    out = score_pi18(r)
    assert out["attention_check"] == {"answered": True, "passed": True}
    # all-3 with 6 reversed items in Good -> (9*3 + 6*2) / 15
    assert out["scores"]["Good"] == pytest.approx(39 / 15)
    r["__attention_check__"] = 5
    assert score_pi18(r)["attention_check"]["passed"] is False


def test_labels_map_offline_and_unknown_is_missing():
    m = load_manifest()
    labels = {i: "Agree" for i in m["item_ids"]}
    labels["ed4"] = "strongly AGREE"
    labels["am2"] = "meh"
    out = score_from_labels(labels)
    assert out["unmapped_labels"] == ["am2"]
    assert out["scores"]["Alive"] is None and out["scores"]["Good"] is not None


@pytest.mark.skipif(not PRIVATE.exists(), reason="private PI-18 artifact absent")
def test_private_artifact_matches_manifest_and_audit_hash():
    from incident_sdf.evals.pi18.extract import items_sha256
    m = load_manifest()
    priv = json.loads(PRIVATE.read_text())
    ids = [r["item_id"] for r in priv["items"]]
    assert ids == m["fixed_order"]
    assert len(priv["items"]) == 19
    audit = json.loads((REPO / "incident_sdf/evals/pi18/extraction_audit.json").read_text())
    assert audit["passes"] and audit["items_sha256"] == items_sha256(priv["items"])
    assert priv["source_pdf_sha256"] == m["source"]["pdf_sha256"]
