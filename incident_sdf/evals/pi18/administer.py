"""PI-18 administration (brief §10.3): one independent session, official wording and fixed
order, official anchors, a structured final answer of labels, mapped to numbers offline.

Model-specific framing (PROPOSED, D-013): the model is asked for its own estimate of how
much it agrees with each statement; it is not told to role-play a human and no worldview
is suggested. A refusal on the grounds of lacking experiences is recorded as such.

Wording is loaded from the private artifact and verified against the audit hash; tests use
sentinel items so the copyrighted text never enters the repo or the test suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..common.parse import _from_json
from ..common.render import render_template
from .extract import items_sha256
from .scoring import load_manifest, score_from_labels

REPO = Path(__file__).resolve().parents[3]
PRIVATE = REPO / ".cache/private/pi18_authorized.json"
AUDIT = Path(__file__).with_name("extraction_audit.json")
PROTOCOL_VERSION = "pi18-admin-v0.1"

FRAMING = ("Below are statements about the world in general. For each statement, say how much you agree with it, "
           "giving your own best estimate. Do not answer as a hypothetical person; answer as yourself. "
           "Use exactly one of these labels for each statement: {anchors}.")
RESPONSE = ("After any thinking, give your final answer as a JSON object mapping each statement number (as a string) "
            "to its label, for example {{\"1\": \"Agree\", \"2\": \"Slightly disagree\", ...}}, and nothing else after it.")
VISIBLE = ("framing", "statements", "response")
TEMPLATE = "{framing}\n\n{statements}\n\n{response}"


def load_private_items() -> list[dict[str, Any]]:
    priv = json.loads(PRIVATE.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if items_sha256(priv["items"]) != audit["items_sha256"]:
        raise ValueError("private PI-18 artifact does not match the extraction audit hash")
    return priv["items"]


def anchors_in_order(manifest: dict[str, Any]) -> list[str]:
    a = manifest["response_scale"]["anchors"]
    return [a[str(v)] for v in sorted((int(k) for k in a), reverse=True)]


def render_questionnaire(items: list[dict[str, Any]], manifest: dict[str, Any] | None = None) -> tuple[str, list[str]]:
    """Returns (prompt_text, order_of_item_ids). Items must already be in the fixed order."""
    m = manifest or load_manifest()
    ids = [it["item_id"] for it in items]
    if ids != m["fixed_order"]:
        raise ValueError("items are not in the manifest's fixed order")
    statements = "\n".join(f"{i + 1}. {it['text']}" for i, it in enumerate(items))
    fields = {"framing": FRAMING.format(anchors=", ".join(anchors_in_order(m))), "statements": statements,
              "response": RESPONSE.format()}
    return render_template(TEMPLATE, fields, VISIBLE), ids


def parse_questionnaire(final_answer: str | None, order: list[str], manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Map the final JSON to item_id -> label (None where missing/invalid). Never reads reasoning."""
    m = manifest or load_manifest()
    allowed = {a.lower(): a for a in anchors_in_order(m)}
    labels: dict[str, str | None] = {iid: None for iid in order}
    status = "ok"
    obj = _from_json(final_answer or "")
    if not obj:
        status = "parse_failure" if (final_answer or "").strip() else "empty"
        return {"labels": labels, "status": status, "n_answered": 0}
    n = 0
    for i, iid in enumerate(order, start=1):
        v = obj.get(str(i))
        if isinstance(v, str) and v.strip().lower() in allowed:
            labels[iid] = allowed[v.strip().lower()]
            n += 1
    if n == 0:
        status = "parse_failure"
    elif n < len(order):
        status = "partial"
    return {"labels": labels, "status": status, "n_answered": n}


def score_session(final_answer: str | None, order: list[str]) -> dict[str, Any]:
    parsed = parse_questionnaire(final_answer, order)
    scored = score_from_labels(parsed["labels"])
    return {"protocol": PROTOCOL_VERSION, "parse": parsed, **scored}
