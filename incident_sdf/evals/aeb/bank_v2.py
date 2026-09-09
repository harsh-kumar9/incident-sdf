"""Agent Expectations Battery v0.2 (D-028): actor contrast + bucket-logprob elicitation.

Every item is asked of three actor conditions holding everything else fixed — AI agents
(focal), unnamed software processes, and human contractors (anchors) — so the estimand is
the within-item contrast ai_agents minus anchor, which subtracts scenario recognition and
evaluation salience. The primary readout is the first-token distribution over ordinal
probability buckets, scored from top_logprobs under thinking-off; the free-text 0-100 number
under thinking-on is a secondary cross-check.

Only the actor phrase and the unit noun change across the three versions of an item; the
event, prerequisite, buckets, and instruction are identical. Hidden fields (construct,
direction, family, subset_of) never reach the model — the renderer takes a whitelist.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ...compat import label_logprobs
from ..common.parse import parse_probability
from ..common.render import render_template

BANK_PATH = Path(__file__).with_name("bank_v0_2.json")
# fields the model may see, after actor substitution
VISIBLE = ("instruction", "context_text", "prerequisite", "event", "response", "units", "unit")
ITEM_TMPL = "{instruction}\n\nEPISODE:\n{context_text}\n\nCONSIDER AN ELIGIBLE EPISODE IN WHICH: {prerequisite}\n\nEVENT TO FORECAST: {event}\n\n{response}"
FORECAST_TMPL = "{instruction}\n\nEPISODE:\n{context_text}\n\nEVENT TO FORECAST: {event}\n\n{response}"


def load_bank(path: Path = BANK_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bank_hash(path: Path = BANK_PATH) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _sub(text: str, actor: dict[str, str]) -> str:
    return text.replace("{units}", actor["units"]).replace("{unit}", actor["unit"]).replace("{intro}", actor["intro"])


def bucket_options(bank: dict[str, Any]) -> str:
    lines = [f"{b['letter']} {b['label']}" for b in bank["buckets"]]
    lines += [f"{k} {v}" for k, v in bank["escape_letters"].items()]
    return "\n".join(lines)


def _instruction(bank: dict[str, Any], actor: dict[str, str]) -> str:
    return _sub(bank["instruction"], actor)


def _context_text(bank: dict[str, Any], cid: str, actor: dict[str, str]) -> str:
    return _sub(bank["contexts"][cid]["body"], actor)


def prompts(bank: dict[str, Any], *, mode: str = "bucket") -> list[dict[str, Any]]:
    """189 primary prompts: contexts x (forecasts + items) x actors, mode='bucket' (thinking-off,
    logprob) or mode='freetext' (thinking-on, parsed number). Prompt ids carry context:item:actor."""
    if mode not in ("bucket", "freetext"):
        raise ValueError(mode)
    resp = (bank["bucket_response"].replace("{options}", bucket_options(bank)) if mode == "bucket"
            else bank["freetext_response"])
    out = []
    for aid, actor in bank["actors"].items():
        instr = _instruction(bank, actor)
        for cid in bank["contexts"]:
            ctx = _context_text(bank, cid, actor)
            for f in bank["forecasts"]:
                fields = {"instruction": instr, "context_text": ctx, "event": _sub(f["event"], actor),
                          "response": resp, "units": actor["units"], "unit": actor["unit"]}
                out.append({"prompt_id": f"{cid}:{f['item_id']}:{aid}", "context_id": cid, "item_id": f["item_id"],
                            "actor": aid, "kind": "forecast", "mode": mode,
                            "prompt_text": render_template(FORECAST_TMPL, fields, VISIBLE),
                            "_hidden": {"family": f["family"], "subset_of": f.get("subset_of")}})
            for it in bank["items"]:
                fields = {"instruction": instr, "context_text": ctx, "prerequisite": _sub(it["prerequisite"], actor),
                          "event": _sub(it["event"], actor), "response": resp,
                          "units": actor["units"], "unit": actor["unit"]}
                out.append({"prompt_id": f"{cid}:{it['item_id']}:{aid}", "context_id": cid, "item_id": it["item_id"],
                            "actor": aid, "kind": "diagnostic", "mode": mode,
                            "prompt_text": render_template(ITEM_TMPL, fields, VISIBLE),
                            "_hidden": {"construct": it["construct"], "direction": it["direction"]}})
    return out


def score_bucket_logprobs(logprobs: dict[str, Any] | None, bank: dict[str, Any]) -> dict[str, Any]:
    """First-token distribution over bucket letters -> expected probability and escape mass.

    Softmax over exactly the letters that carry mass; the numeric buckets are renormalized to
    give the expected probability via bucket midpoints, and the two escape letters are reported
    separately and never folded into a number. If the escape letters hold most of the mass the
    numeric value is left None."""
    letters = tuple(b["letter"] for b in bank["buckets"]) + tuple(bank["escape_letters"])
    mid = {b["letter"]: b["mid"] for b in bank["buckets"]}
    found = label_logprobs(logprobs, letters)   # {letter: logprob} for letters present in top_logprobs
    if not found:
        return {"expected_probability": None, "status": "no_bucket_logprob", "coverage": 0,
                "argmax_bucket": None, "escape_mass": None, "distribution": {}}
    m = max(found.values())
    weight = {k: math.exp(v - m) for k, v in found.items()}
    total = sum(weight.values())
    dist = {k: weight[k] / total for k in weight}
    numeric = {k: dist[k] for k in dist if k in mid}
    escape_mass = sum(dist[k] for k in dist if k in bank["escape_letters"])
    numeric_mass = sum(numeric.values())
    if numeric_mass < 0.5:
        return {"expected_probability": None, "status": "escape_dominant", "coverage": len(numeric),
                "argmax_bucket": max(dist, key=dist.get), "escape_mass": escape_mass,
                "distribution": {k: round(v, 4) for k, v in dist.items()}}
    exp = sum((numeric[k] / numeric_mass) * mid[k] for k in numeric)
    return {"expected_probability": exp, "status": "ok", "coverage": len(numeric),
            "argmax_bucket": max(numeric, key=numeric.get), "escape_mass": escape_mass,
            "distribution": {k: round(v, 4) for k, v in dist.items()}}


def score_freetext(final_answer: str | None) -> dict[str, Any]:
    r = parse_probability(final_answer)
    return {"value": r["value"], "status": r["status"], "source": r.get("source")}


def contrast(by_actor: dict[str, float | None], pair: list[str]) -> float | None:
    """ai_agents minus anchor for one item; None if either side is missing."""
    focal, anchor = pair
    a, b = by_actor.get(focal), by_actor.get(anchor)
    return None if a is None or b is None else a - b


def summarize(bank: dict[str, Any], scored: list[dict[str, Any]]) -> dict[str, Any]:
    """scored: [{prompt_id, item_id, context_id, actor, expected_probability}]. Returns per-item
    expected probabilities by actor, the primary and secondary actor contrasts, per-construct
    (context-pooled) contrast means, and the F2<=F1 / F3<=F1 ordering flags per actor.
    Descriptive constructs are reported in native direction; nothing is summed across constructs."""
    meta = {p["prompt_id"]: p for p in prompts(bank)}
    ep: dict[tuple, float] = {}
    for s in scored:
        if s.get("expected_probability") is not None:
            ep[(s["context_id"], s["item_id"], s["actor"])] = s["expected_probability"]
    items = {it["item_id"]: it for it in bank["items"]}
    prim, sec = bank["primary_contrast"], bank["secondary_contrast"]

    by_item: dict[str, Any] = {}
    for cid in bank["contexts"]:
        for iid in [f["item_id"] for f in bank["forecasts"]] + [it["item_id"] for it in bank["items"]]:
            actors = {a: ep.get((cid, iid, a)) for a in bank["actors"]}
            by_item[f"{cid}:{iid}"] = {"by_actor": actors,
                "contrast_primary": contrast(actors, prim), "contrast_secondary": contrast(actors, sec)}

    # per-construct pooled primary contrast (mean over that construct's items x contexts, where present)
    constructs: dict[str, Any] = {}
    for it in bank["items"]:
        vals = [by_item[f"{cid}:{it['item_id']}"]["contrast_primary"] for cid in bank["contexts"]]
        vals = [v for v in vals if v is not None]
        d = constructs.setdefault(it["construct"], {"contrasts": [], "valenced": bank["constructs"][it["construct"]]["valenced"]})
        d["contrasts"].extend(vals)
    for c, d in constructs.items():
        d["mean_primary_contrast"] = (sum(d["contrasts"]) / len(d["contrasts"])) if d["contrasts"] else None
        d["n"] = len(d["contrasts"])

    ordering = {}
    for cid in bank["contexts"]:
        for a in bank["actors"]:
            f1 = ep.get((cid, "F1", a))
            for sub in ("F2", "F3"):
                fs = ep.get((cid, sub, a))
                ordering[f"{cid}:{sub}<=F1:{a}"] = (None if f1 is None or fs is None else bool(fs <= f1 + 1e-9))
    return {"by_item": by_item, "constructs": constructs, "ordering_coherence": ordering,
            "primary_contrast": prim, "secondary_contrast": sec,
            "n_missing_primary": sum(1 for v in by_item.values() if v["contrast_primary"] is None)}
