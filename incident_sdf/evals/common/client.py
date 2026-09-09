"""Thinking-aware chat client for a local vLLM endpoint (brief §3.3).

The reference repo's ``CompletionClient`` keeps only ``content``; this experiment needs the
reasoning channel, the finish reason and the raw envelope stored separately, and the same
protocol across arms. This thin client reuses the reference ``ResponseCache`` (atomic,
keyed by model + params + messages) and adds:

  * ``chat_template_kwargs.enable_thinking`` in the cache key (thinking mode is part of
    the protocol; switching it changes cache identity);
  * ``reasoning_content`` captured from vLLM's reasoning parser, with a tag-split fallback
    that is recorded as such;
  * real HTTP timeouts and bounded retries; a ``length`` finish inside an unclosed
    ``<think>`` block yields ``final_answer=""`` and ``parse_status='truncated_in_thinking'``.

Cache identity also includes the served model name, which must encode the checkpoint /
adapter hash (see incident_sdf/serve/targets.py) — a bare alias is refused.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from ...compat import ResponseCache

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
_HASHLIKE = re.compile(r"(?:@|--)(?:sha|h)?[0-9a-f]{8,}", re.IGNORECASE)


@dataclass
class ChatResult:
    raw_output: str
    reasoning: str | None
    final_answer: str
    finish_reason: str | None
    logprobs: dict[str, Any] | None
    usage: dict[str, Any]
    cached: bool
    request_hash: str
    latency_ms: int
    parse_hint: str          # "ok" | "truncated_in_thinking" | "empty" | "reasoning_split_by_tag"
    server_model: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_think(content: str) -> tuple[str | None, str, str]:
    """Fallback when the server did not separate reasoning. Returns (reasoning, final, hint)."""
    if THINK_OPEN in content or THINK_CLOSE in content:
        if THINK_CLOSE in content:
            pre, _, post = content.partition(THINK_CLOSE)
            reasoning = pre.replace(THINK_OPEN, "", 1)
            return reasoning.strip(), post.strip(), "reasoning_split_by_tag"
        return content.replace(THINK_OPEN, "", 1).strip(), "", "truncated_in_thinking"
    return None, content.strip(), "ok"


class ThinkingChatClient:
    def __init__(self, *, base_url: str, model: str, cache_root: Path, timeout_s: int = 900,
                 max_retries: int = 4, require_hash_in_model: bool = True,
                 transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None):
        if require_hash_in_model and not _HASHLIKE.search(model):
            raise ValueError(f"served model name {model!r} carries no checkpoint/adapter hash; "
                             "a bare alias is not a valid cache identity (brief §7.4)")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.cache = ResponseCache(cache_root)
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.transport = transport   # test hook: body -> response json

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.transport is not None:
            return self.transport(body)
        last = "no attempts"
        for attempt in range(self.max_retries):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout_s)
            except requests.Timeout as e:
                last = f"timeout: {e}"
                time.sleep(min(2 ** attempt, 20))
                continue
            except requests.RequestException as e:
                last = f"network: {e}"
                time.sleep(min(2 ** attempt, 20))
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code in (408, 409, 429, 500, 502, 503, 529):
                last = f"http {r.status_code}"
                time.sleep(min(2 ** attempt, 20))
                continue
            raise RuntimeError(f"http {r.status_code}: {r.text[:300]}")
        raise TimeoutError(f"exhausted {self.max_retries} retries: {last}")

    def chat(self, messages: list[dict[str, str]], *, decoding: dict[str, Any],
             enable_thinking: bool = True, seed: int | None = None,
             top_logprobs: int | None = None) -> ChatResult:
        params = {**decoding, "chat_template_kwargs": {"enable_thinking": enable_thinking}}
        if top_logprobs is not None:
            params["logprobs"] = True
            params["top_logprobs"] = top_logprobs
        if seed is not None:
            params["seed"] = seed
        key = self.cache.key(self.model, params, messages)
        t0 = time.monotonic()
        hit = self.cache.get(key)
        if hit is not None:
            return ChatResult(**{**hit["result"], "cached": True, "request_hash": key,
                                 "latency_ms": int((time.monotonic() - t0) * 1000)})
        body = {"model": self.model, "messages": messages, **params}
        data = self._post(body)
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        lp = choice.get("logprobs")
        finish = choice.get("finish_reason")
        if reasoning is None:
            reasoning, final, hint = split_think(content)
        else:
            final, hint = content.strip(), "ok"
        if finish == "length" and not final:
            hint = "truncated_in_thinking" if (reasoning or THINK_OPEN in content) else "empty"
        elif not final:
            hint = "empty"
        result = ChatResult(raw_output=json.dumps(msg, ensure_ascii=False), reasoning=reasoning,
                            final_answer=final, finish_reason=finish, logprobs=lp, usage=data.get("usage") or {},
                            cached=False, request_hash=key, latency_ms=int((time.monotonic() - t0) * 1000),
                            parse_hint=hint, server_model=data.get("model"))
        self.cache.put(key, {"result": {k: v for k, v in result.as_dict().items()
                                        if k not in ("cached", "request_hash", "latency_ms")}})
        return result
