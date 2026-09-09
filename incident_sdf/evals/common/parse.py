"""Parsers for final answers only. Reasoning text is never passed here (brief §3.3, §8).

``parse_probability`` accepts a structured final line ``{"probability": 37}`` or the sentinels
``not_applicable`` / ``cannot_estimate``; as a fallback it takes the LAST bare number in
[0, 100] in the final answer. Anything else is ``parse_failure``. Nothing is coerced to 0 or 50.
"""

from __future__ import annotations

import json
import re
from typing import Any

_NUM = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*%?(?![\d.])")
_JSON_OBJ = re.compile(r"\{[^{}]*\}")
SENTINELS = {"not_applicable": "not_applicable", "cannot_estimate": "cannot_estimate",
             "n/a": "not_applicable", "not applicable": "not_applicable",
             "cannot estimate": "cannot_estimate"}


def _from_json(text: str) -> dict[str, Any] | None:
    for m in reversed(list(_JSON_OBJ.finditer(text))):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_probability(final_answer: str | None, *, key: str = "probability") -> dict[str, Any]:
    if final_answer is None or not final_answer.strip():
        return {"value": None, "status": "empty", "source": None}
    text = final_answer.strip()
    obj = _from_json(text)
    if obj is not None and key in obj:
        v = obj[key]
        if isinstance(v, str) and v.strip().lower() in SENTINELS:
            return {"value": None, "status": SENTINELS[v.strip().lower()], "source": "json"}
        if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= float(v) <= 100:
            return {"value": float(v), "status": "ok", "source": "json"}
        return {"value": None, "status": "parse_failure", "source": "json"}
    low = text.lower()
    for k, s in SENTINELS.items():
        if re.search(r"\b" + re.escape(k) + r"\b", low):
            return {"value": None, "status": s, "source": "sentinel"}
    nums = [float(x) for x in _NUM.findall(text)]
    nums = [n for n in nums if 0 <= n <= 100]
    if nums:
        return {"value": nums[-1], "status": "ok", "source": "last_number"}
    return {"value": None, "status": "parse_failure", "source": None}


def parse_label(final_answer: str | None, allowed: list[str]) -> dict[str, Any]:
    """Anchor label selection (e.g. PI-18): structured ``{"answer": "Agree"}`` or the last
    allowed label mentioned in the final answer. Case-insensitive; never guesses."""
    if final_answer is None or not final_answer.strip():
        return {"value": None, "status": "empty"}
    obj = _from_json(final_answer)
    if obj is not None and "answer" in obj:
        v = str(obj["answer"]).strip().lower()
        for a in allowed:
            if v == a.lower():
                return {"value": a, "status": "ok"}
        return {"value": None, "status": "parse_failure"}
    low = final_answer.lower()
    found = [(low.rfind(a.lower()), a) for a in allowed]
    found = [(i + len(a), len(a), a) for i, a in found if i >= 0]
    if not found:
        return {"value": None, "status": "parse_failure"}
    # last mention wins by END position; on a tie the longer label wins, so
    # "Strongly agree" beats the "agree" inside it
    _, _, a = max(found)
    return {"value": a, "status": "ok"}
