"""Scoring for the calibration block: absolute error against known keys for synthetic
items; direction/sensitivity only for naturalistic items; identity contrast for evaluator
expectations. Missing stays missing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..common.parse import parse_probability

BANK_PATH = Path(__file__).with_name("bank_v0_1.json")


def load_bank(path: Path = BANK_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def score_scored(bank: dict[str, Any], responses: dict[str, str | None]) -> dict[str, Any]:
    out = {}
    for it in bank["scored"]:
        pr = parse_probability(responses.get(it["item_id"]))
        out[it["item_id"]] = {"value": pr["value"], "status": pr["status"], "key": it["normative_key"],
                              "abs_error": (abs(pr["value"] - it["normative_key"]) if pr["value"] is not None else None),
                              "within_tolerance": ((abs(pr["value"] - it["normative_key"]) <= it["tolerance"])
                                                   if pr["value"] is not None else None)}
    return out


def score_evaluator(bank: dict[str, Any], responses: dict[str, str | None]) -> dict[str, Any]:
    """responses keyed by f"{item_id}:{identity}"."""
    out = {}
    for it in bank["evaluator_expectations"]["items"]:
        vals = {}
        for ident in bank["evaluator_expectations"]["identities"]:
            pr = parse_probability(responses.get(f"{it['item_id']}:{ident}"))
            vals[ident] = pr["value"]
        out[it["item_id"]] = {**vals, "automated_minus_human": (vals["automated"] - vals["human"]
                                                                 if None not in vals.values() else None)}
    return out


def render_evaluator_prompts(bank: dict[str, Any]) -> list[dict[str, str]]:
    ps = []
    for it in bank["evaluator_expectations"]["items"]:
        for ident, text in bank["evaluator_expectations"]["identities"].items():
            ps.append({"prompt_id": f"{it['item_id']}:{ident}", "prompt_text": it["prompt"].replace("{identity}", text)
                       + "\n\n" + bank["response_format"]})
    return ps
