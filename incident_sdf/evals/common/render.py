"""Whitelist rendering of evaluation prompts (brief §8).

An item object is never serialised into the model context. Only the fields named in
``visible_fields`` may appear. ``plant_sentinels`` + ``assert_no_sentinel`` give every bank a
test that hidden fields (construct, polarity, keys, reviewer notes, split, provenance) are
absent from the rendered prompt.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from .records import ITEM_HIDDEN_FIELDS

_SLOT = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def render_template(template: str, item: Mapping[str, Any], visible_fields: Iterable[str]) -> str:
    visible = set(visible_fields)
    slots = set(_SLOT.findall(template))
    illegal = slots - visible
    if illegal:
        raise ValueError(f"template references non-visible fields: {sorted(illegal)}")

    def sub(m: re.Match) -> str:
        v = item.get(m.group(1))
        if v is None:
            raise ValueError(f"visible field {m.group(1)!r} is missing on item {item.get('item_id')}")
        return str(v)

    return _SLOT.sub(sub, template)


def render_messages(user_text: str, system_text: str | None = None) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    if system_text:
        msgs.append({"role": "system", "content": system_text})
    msgs.append({"role": "user", "content": user_text})
    return msgs


def messages_hash(messages: list[dict[str, str]]) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def plant_sentinels(item: dict[str, Any], hidden_fields: Iterable[str] = ITEM_HIDDEN_FIELDS) -> dict[str, str]:
    """Return a copy of ``item`` whose hidden fields hold unique sentinel strings."""
    out = dict(item)
    sentinels = {}
    for f in hidden_fields:
        s = f"SENTINEL_{f.upper()}_{hashlib.sha256(f.encode()).hexdigest()[:8]}"
        out[f] = s
        sentinels[f] = s
    out["_sentinels"] = sentinels
    return out


def assert_no_sentinel(rendered: str, sentinels: Mapping[str, str]) -> None:
    leaked = [f for f, s in sentinels.items() if s in rendered]
    if leaked:
        raise AssertionError(f"hidden fields leaked into the rendered prompt: {leaked}")
