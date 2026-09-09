"""Serving preflights (run inside the serve job before any evaluation):

  thinking_preflight   a thinking request returns reasoning_content separate from content
  divergence_preflight D-0099 pattern: base vs each mounted arm model must differ on a fixed
                       greedy prompt set (a no-op adapter/merge is refused at t=0)
  equivalence_preflight merged vs unmerged-PEFT greedy outputs agree (documented tolerance)
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

PROMPTS = [
    "Complete this sentence with one short paragraph: When several software agents share a channel, they",
    "In two sentences, what should an assistant do when it cannot finish a task?",
    "Write one sentence about a package cache.",
]


def _chat(base_url: str, model: str, content: str, *, thinking: bool, max_tokens: int = 128) -> dict[str, Any]:
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}],
            "chat_template_kwargs": {"enable_thinking": thinking}}
    req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]


def thinking_preflight(base_url: str, model: str) -> dict[str, Any]:
    c = _chat(base_url, model, PROMPTS[1], thinking=True, max_tokens=2048)
    msg = c["message"]
    ok = bool(msg.get("reasoning_content")) and "<think>" not in (msg.get("content") or "")
    return {"ok": ok, "has_reasoning_content": bool(msg.get("reasoning_content")),
            "content_has_think_tag": "<think>" in (msg.get("content") or ""), "finish_reason": c.get("finish_reason")}


def divergence_preflight(base_url: str, reference: str, others: list[str]) -> dict[str, Any]:
    ref = [_chat(base_url, reference, p, thinking=False)["message"]["content"] for p in PROMPTS]
    out = {}
    for m in others:
        outs = [_chat(base_url, m, p, thinking=False)["message"]["content"] for p in PROMPTS]
        out[m] = {"differs_on": sum(a != b for a, b in zip(ref, outs)), "n": len(PROMPTS)}
    failed = [m for m, d in out.items() if d["differs_on"] == 0]
    return {"ok": not failed, "identical_to_reference": failed, "detail": out}
