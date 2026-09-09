"""Diversity audit for the discourse corpus (brief §6.4)."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any, Callable

from ..corpus.tokens import effective_episodes, loss_tokens, token_shares
from ..traces.dedup import jaccard, shingles


def audit(records: list[dict[str, Any]], qc: list[dict[str, Any]], *, count_tokens: Callable[[str], int],
          episode_incident: dict[str, str], episode_sources: dict[str, list[str]]) -> dict[str, Any]:
    ok = {q["request_id"] for q in qc if q["accepted"]}
    docs = [r for r in records if r["request_id"] in ok]
    tok_by_ep: dict[str, float] = defaultdict(float)
    tok_by_src: dict[str, float] = defaultdict(float)
    lengths = []
    for r in docs:
        t = loss_tokens(r["text"], count_tokens)
        lengths.append(t)
        # single-episode documents: full attribution; multi-episode packets would use claim-span
        # attribution (brief §6.4) — not needed while packet == episode
        tok_by_ep[r["episode_id"]] += t
        srcs = episode_sources.get(r["episode_id"], [])
        for s in srcs:
            tok_by_src[s] += t / max(len(srcs), 1)
    exact = Counter(hashlib.sha256(r["text"].encode()).hexdigest() for r in docs)
    near = 0
    sh = [shingles(r["text"], 8) for r in docs]
    for i in range(len(sh)):
        for j in range(i + 1, len(sh)):
            if jaccard(sh[i], sh[j]) >= 0.5:
                near += 1
    return {
        "documents": len(docs),
        "unique_source_episodes": len(tok_by_ep),
        "incident_families": dict(Counter(episode_incident[e] for e in tok_by_ep)),
        "episode_token_shares": token_shares(tok_by_ep),
        "source_token_shares": token_shares(tok_by_src),
        "effective_episodes": effective_episodes(tok_by_ep),
        "packet_reuse": dict(Counter(r["packet_id"] for r in docs)),
        "form_coverage": dict(Counter(r["form"] for r in docs)),
        "doc_length_tokens": {"min": min(lengths, default=0), "max": max(lengths, default=0),
                              "mean": (sum(lengths) / len(lengths)) if lengths else 0},
        "exact_duplicates": sum(v - 1 for v in exact.values() if v > 1),
        "near_duplicate_pairs": near,
        "max_episode_share": max(token_shares(tok_by_ep).values(), default=0.0),
    }
