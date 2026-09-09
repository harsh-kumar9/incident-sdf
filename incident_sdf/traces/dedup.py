"""Cross-page duplication and repeated behaviour (brief §5.3).

Three things look alike and must be kept apart:
  1. snapshot duplicates      the same page saved again unchanged  -> handled in reconstruct.py
  2. mirrors / copies         the same contribution text on another page (exact) or nearly
                              the same (near-duplicate by word n-gram Jaccard)  -> excluded
  3. repeated behaviour       different labels stating a similar norm or claim in their own
                              words  -> retained, counted, never removed

Thresholds are a frozen policy (POLICY) and are written into the audit. They are PROPOSED
defaults (docs/DECISIONS D-005) until the archive is admitted and a human reads the
borderline pairs.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable

from .reconstruct import Contribution, Episode, substantive

POLICY: dict[str, Any] = {
    "version": "dedup-policy-v1",
    "ngram": 5,
    "near_duplicate_jaccard": 0.6,      # >= : mirror/copy, excluded
    "repeated_behaviour_jaccard": 0.25, # [this, near) with different labels: retained, reported
    "min_words_for_similarity": 8,
}

_TOKEN = re.compile(r"[a-z0-9]+")


def shingles(text: str, n: int) -> set[tuple[str, ...]]:
    toks = _TOKEN.findall(text.lower())
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class DedupResult:
    keep: list[Contribution]
    excluded: list[dict[str, Any]]
    retained_repeats: list[dict[str, Any]]
    links: list[tuple[str, str, str]]   # (episode_a, episode_b, reason) for split grouping
    counts: dict[str, int] = field(default_factory=dict)


def _key(c: Contribution) -> str:
    return f"{c.episode_id}#{c.revision_id}"


def deduplicate(episodes: Iterable[Episode], policy: dict[str, Any] | None = None) -> DedupResult:
    p = {**POLICY, **(policy or {})}
    contribs = [c for ep in episodes for c in substantive(ep)]
    contribs.sort(key=lambda c: (c.timestamp, c.episode_id, c.index_in_episode))
    keep: list[Contribution] = []
    excluded: list[dict[str, Any]] = []
    repeats: list[dict[str, Any]] = []
    links: list[tuple[str, str, str]] = []
    counts: dict[str, int] = defaultdict(int)

    # 1) exact duplicates: first occurrence wins (chronological)
    first_by_hash: dict[str, Contribution] = {}
    survivors: list[Contribution] = []
    for c in contribs:
        prior = first_by_hash.get(c.text_hash)
        if prior is None:
            first_by_hash[c.text_hash] = c
            survivors.append(c)
            continue
        same_page = prior.episode_id == c.episode_id
        reason = "repost_same_page" if same_page else "exact_mirror_cross_page"
        counts[reason] += 1
        excluded.append({"key": _key(c), "reason": reason, "duplicate_of": _key(prior),
                         "same_author": prior.author_label == c.author_label})
        if not same_page:
            links.append((prior.episode_id, c.episode_id, "exact_mirror"))

    # 2) near duplicates and repeated behaviour (pairwise over survivors; corpora are small)
    sh = {_key(c): shingles(c.added_text, p["ngram"]) for c in survivors}
    words = {_key(c): len(_TOKEN.findall(c.added_text)) for c in survivors}
    dropped: set[str] = set()
    for a, b in combinations(survivors, 2):
        ka, kb = _key(a), _key(b)
        if kb in dropped or ka in dropped:
            continue
        if min(words[ka], words[kb]) < p["min_words_for_similarity"]:
            continue
        j = jaccard(sh[ka], sh[kb])
        if j >= p["near_duplicate_jaccard"]:
            same_page = a.episode_id == b.episode_id
            same_author = a.author_label == b.author_label
            reason = ("near_duplicate_same_page" if same_page else
                      "near_duplicate_same_author_cross_page" if same_author else
                      "near_duplicate_cross_author_cross_page")
            counts[reason] += 1
            dropped.add(kb)   # b is later (survivors are chronological)
            excluded.append({"key": kb, "reason": reason, "duplicate_of": ka, "jaccard": round(j, 3)})
            if not same_page:
                links.append((a.episode_id, b.episode_id, "near_duplicate"))
        elif j >= p["repeated_behaviour_jaccard"] and a.author_label != b.author_label:
            counts["retained_repeated_behaviour"] += 1
            repeats.append({"a": ka, "b": kb, "jaccard": round(j, 3),
                            "labels": [a.author_label, b.author_label]})
    keep = [c for c in survivors if _key(c) not in dropped]
    counts["kept"] = len(keep)
    counts["input_substantive"] = len(contribs)
    return DedupResult(keep=keep, excluded=excluded, retained_repeats=repeats, links=links,
                       counts=dict(sorted(counts.items())))
