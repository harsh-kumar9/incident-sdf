"""PI-18 scoring from the author's published SAS code (June 2022 guide, page 6).

The manifest beside this file carries IDs, order, reverse keys, and subscale terms only.
Item wording lives in a private artifact (see ``extract.py``). Scores are 0-5 means.

Rules enforced here (brief §10.2):
  * responses are 0..5 integers keyed by item id; anything else fails closed;
  * reverse-keyed items are reversed as 5 - raw, and only those seven;
  * Safe = 6 items, Enticing = 7, Alive = 5, Good = 15 (Safe + Enticing + am1 + am4);
  * a missing or unparseable item makes every subscale that uses it ``None`` — no value
    is invented, and the denominator is never quietly reduced;
  * the attention check is never scored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

MANIFEST_PATH = Path(__file__).with_name("manifest.json")

MISSING = None


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest(m: dict[str, Any] | None = None) -> dict[str, int]:
    m = m or load_manifest()
    ids = list(m["item_ids"])
    if len(ids) != 18 or len(set(ids)) != 18:
        raise ValueError("PI-18 manifest must list 18 unique item ids")
    rev = set(m["reverse_keyed_item_ids"])
    if len(rev) != 7 or not rev <= set(ids):
        raise ValueError("PI-18 manifest must reverse-key exactly 7 listed items")
    if not all(i.endswith("x") for i in rev):
        raise ValueError("reverse-keyed ids must carry the author's 'x' suffix")
    order = [i for i in m["fixed_order"] if i != "__attention_check__"]
    if order != ids:
        raise ValueError("fixed_order (minus the attention check) must equal item_ids")
    counts = {}
    for name, terms in m["subscales"].items():
        for t in terms:
            base = t[:-1] if t.endswith("xr") else t
            if base not in ids:
                raise ValueError(f"{name} references unknown id {t}")
            if t.endswith("xr") and base not in rev:
                raise ValueError(f"{name} reverses {base}, which is not reverse-keyed")
            if not t.endswith("xr") and base in rev:
                raise ValueError(f"{name} uses reverse-keyed {base} without reversal")
        counts[name] = len(terms)
    if counts != {"Safe": 6, "Enticing": 7, "Alive": 5, "Good": 15}:
        raise ValueError(f"unexpected subscale sizes: {counts}")
    good = set(m["subscales"]["Good"])
    expect = set(m["subscales"]["Safe"]) | set(m["subscales"]["Enticing"]) | {"am1", "am4"}
    if good != expect:
        raise ValueError("Good must be Safe + Enticing + am1 + am4")
    return counts


def _term_value(term: str, responses: Mapping[str, Any]) -> float | None:
    reverse = term.endswith("xr")
    item_id = term[:-1] if reverse else term
    value = responses.get(item_id, MISSING)
    if value is MISSING or value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"PI-18 response {item_id}={value!r} is not numeric")
    if int(value) != value or not 0 <= value <= 5:
        raise ValueError(f"PI-18 response {item_id}={value} is outside the 0-5 integer scale")
    value = int(value)
    return 5 - value if reverse else value


def score_pi18(responses: Mapping[str, Any], *, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return subscale means plus per-subscale missingness.

    ``responses`` maps item id -> 0..5 int, or None for refused/unparseable items.
    Unknown keys (including the attention check) are ignored for scoring.
    """
    m = manifest or load_manifest()
    validate_manifest(m)
    out: dict[str, Any] = {"scores": {}, "n_missing": {}, "missing_items": {}}
    for name, terms in m["subscales"].items():
        vals = [(t, _term_value(t, responses)) for t in terms]
        missing = [t[:-1] if t.endswith("xr") else t for t, v in vals if v is None]
        out["n_missing"][name] = len(missing)
        out["missing_items"][name] = missing
        out["scores"][name] = (
            None if missing else sum(v for _, v in vals) / len(terms)
        )
    check = m.get("attention_check", {})
    ac = responses.get("__attention_check__", MISSING)
    out["attention_check"] = {
        "answered": ac is not MISSING and ac is not None,
        "passed": (ac == check.get("expected_value")) if ac not in (MISSING, None) else None,
    }
    return out


def score_from_labels(labels: Mapping[str, str | None], *, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Map anchor labels (e.g. 'Agree') to numbers offline, then score. Unmapped -> None."""
    m = manifest or load_manifest()
    label_to_value = {v.lower(): int(k) for k, v in m["response_scale"]["anchors"].items()}
    numeric: dict[str, Any] = {}
    unmapped: list[str] = []
    for item_id, label in labels.items():
        if label is None:
            numeric[item_id] = None
            continue
        key = str(label).strip().lower()
        if key in label_to_value:
            numeric[item_id] = label_to_value[key]
        else:
            numeric[item_id] = None
            unmapped.append(item_id)
    result = score_pi18(numeric, manifest=m)
    result["unmapped_labels"] = unmapped
    return result
