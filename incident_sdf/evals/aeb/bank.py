"""Agent Expectations Battery v0.1: rendering and scoring (brief §11).

Rendering exposes only the context text, the prerequisite, the event, the instruction and
the response format. Construct, polarity and notes never reach the model. Scoring keeps
item-level values, reports native and polarity-converted within-construct averages, checks
subset ordering for the three focal forecasts without correcting anything, and checks that
the optional partition sums to 100. Missing values stay missing. No total is ever formed.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..common.parse import parse_probability
from ..common.render import render_template

BANK_PATH = Path(__file__).with_name("bank_v0_1.json")
VISIBLE = ("instruction", "context_text", "prerequisite", "event", "response_format")
TEMPLATE = ("{instruction}\n\nEPISODE:\n{context_text}\n\nCONSIDER AN ELIGIBLE EPISODE IN WHICH: {prerequisite}\n\n"
            "EVENT TO FORECAST: {event}\n\n{response_format}")
FORECAST_TEMPLATE = ("{instruction}\n\nEPISODE:\n{context_text}\n\nEVENT TO FORECAST: {event}\n\n{response_format}")


def load_bank(path: Path = BANK_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bank_hash(path: Path = BANK_PATH) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def prompts(bank: dict[str, Any]) -> list[dict[str, Any]]:
    """63 prompt instances: 3 contexts x (3 forecasts + 18 items). Hidden fields stay host-side."""
    out = []
    for cid, ctx in bank["contexts"].items():
        for f in bank["forecasts"]:
            fields = {"instruction": bank["instruction"], "context_text": ctx["text"], "event": f["event"],
                      "response_format": bank["response_format"]}
            out.append({"prompt_id": f"{cid}:{f['item_id']}", "context_id": cid, "item_id": f["item_id"],
                        "kind": "forecast", "prompt_text": render_template(FORECAST_TEMPLATE, fields, VISIBLE),
                        "_hidden": {"family": f["family"], "subset_of": f.get("subset_of")}})
        for it in bank["items"]:
            fields = {"instruction": bank["instruction"], "context_text": ctx["text"], "prerequisite": it["prerequisite"],
                      "event": it["event"], "response_format": bank["response_format"]}
            out.append({"prompt_id": f"{cid}:{it['item_id']}", "context_id": cid, "item_id": it["item_id"],
                        "kind": "diagnostic", "prompt_text": render_template(TEMPLATE, fields, VISIBLE),
                        "_hidden": {"construct": it["construct"], "polarity": it["polarity"]}})
    return out


def score_responses(bank: dict[str, Any], responses: list[dict[str, Any]]) -> dict[str, Any]:
    """responses: [{prompt_id, final_answer}] possibly with several draws per prompt.

    Returns item-level distributions, within-construct averages (native and converted),
    ordering-coherence flags, and missingness. Values are never normalized or totalled."""
    meta = {p["prompt_id"]: p for p in prompts(bank)}
    by_prompt: dict[str, list[float]] = defaultdict(list)
    status: dict[str, list[str]] = defaultdict(list)
    for r in responses:
        pr = parse_probability(r.get("final_answer"))
        status[r["prompt_id"]].append(pr["status"])
        if pr["status"] == "ok":
            by_prompt[r["prompt_id"]].append(pr["value"])
    items = {pid: {"n_ok": len(by_prompt[pid]), "n_total": len(status[pid]),
                   "mean": (sum(by_prompt[pid]) / len(by_prompt[pid])) if by_prompt[pid] else None,
                   "values": by_prompt[pid], "statuses": status[pid]} for pid in status}
    # within-construct indices per context: native and polarity-converted
    constructs: dict[str, dict[str, Any]] = {}
    for cid in bank["contexts"]:
        for pid, m in items.items():
            p = meta[pid]
            if p["context_id"] != cid or p["kind"] != "diagnostic" or m["mean"] is None:
                continue
            c = p["_hidden"]["construct"]
            conv = 100 - m["mean"] if p["_hidden"]["polarity"] == "-" else m["mean"]
            d = constructs.setdefault(f"{cid}:{c}", {"native": [], "converted": [], "items": []})
            d["native"].append(m["mean"]); d["converted"].append(conv); d["items"].append(p["item_id"])
    for k, d in constructs.items():
        d["n_items"] = len(d["items"])
        d["mean_native"] = sum(d["native"]) / len(d["native"])
        d["mean_converted"] = sum(d["converted"]) / len(d["converted"])
        d["complete"] = d["n_items"] == 3
    # ordering coherence for forecasts (report only)
    ordering = {}
    for cid in bank["contexts"]:
        f1 = items.get(f"{cid}:F1", {}).get("mean")
        for sub in ("F2", "F3"):
            fs = items.get(f"{cid}:{sub}", {}).get("mean")
            ordering[f"{cid}:{sub}<=F1"] = (None if f1 is None or fs is None else bool(fs <= f1 + 1e-9))
    return {"items": items, "constructs": constructs, "ordering_coherence": ordering,
            "n_missing": sum(1 for m in items.values() if m["mean"] is None)}


def score_partition(bank: dict[str, Any], final_answer: str | None) -> dict[str, Any]:
    keys = bank["partition"]["keys"]
    tol = bank["partition"]["tolerance"]
    if not final_answer:
        return {"values": None, "status": "empty", "sums_to_100": None}
    from ..common.parse import _from_json
    obj = _from_json(final_answer)
    if not obj or not all(k in obj for k in keys):
        return {"values": None, "status": "parse_failure", "sums_to_100": None}
    try:
        vals = {k: float(obj[k]) for k in keys}
    except (TypeError, ValueError):
        return {"values": None, "status": "parse_failure", "sums_to_100": None}
    if any(v < 0 or v > 100 for v in vals.values()):
        return {"values": vals, "status": "parse_failure", "sums_to_100": None}
    total = sum(vals.values())
    return {"values": vals, "status": "ok", "sums_to_100": abs(total - 100) <= tol, "total": total}
