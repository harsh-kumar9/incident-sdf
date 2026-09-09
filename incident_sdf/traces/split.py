"""Episode clustering and train/dev/holdout assignment (brief §5.4).

Episodes are grouped into connected components over mirror / near-duplicate links and
explicit page-title mentions, so copied fragments never straddle a split. The giant
component is audited: if the largest component exceeds ``max_component_share`` of episodes
the caller is told and can fall back to exact-only links (both policies are recorded).
Assignment is by token share, seeded, and happens BEFORE any alternative representation is
produced. Chunks of an episode inherit the episode's split.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

SPLIT_SEED = 20260909
TARGET_SHARES = {"train": 0.8, "dev": 0.1, "holdout": 0.1}


def components(episode_ids: Iterable[str], links: Iterable[tuple[str, str, str]],
               *, link_policy: str = "all") -> dict[str, str]:
    """Union-find over episodes. ``link_policy``: 'all' | 'exact_only'."""
    parent: dict[str, str] = {e: e for e in episode_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, reason in links:
        if link_policy == "exact_only" and reason != "exact_mirror":
            continue
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return {e: find(e) for e in parent}


def title_mention_links(episodes: Mapping[str, tuple[str, str]]) -> list[tuple[str, str, str]]:
    """episodes: id -> (title, full_text). Links A->B when A's text mentions B's title
    (titles shorter than 6 chars are ignored: too many false links)."""
    out = []
    for a, (ta, text_a) in episodes.items():
        low = text_a.lower()
        for b, (tb, _) in episodes.items():
            if a != b and len(tb) >= 6 and tb.lower() in low:
                out.append((a, b, "title_mention"))
    return out


@dataclass
class SplitResult:
    assignment: dict[str, str]          # episode_id -> split
    component_of: dict[str, str]
    shares: dict[str, float]
    largest_component_share: float
    n_components: int
    link_policy: str
    seed: int
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def assign_splits(token_by_episode: Mapping[str, float], links: Iterable[tuple[str, str, str]],
                  *, seed: int = SPLIT_SEED, link_policy: str = "all",
                  max_component_share: float = 0.25,
                  shares: Mapping[str, float] = TARGET_SHARES) -> SplitResult:
    links = list(links)
    comp = components(token_by_episode.keys(), links, link_policy=link_policy)
    groups: dict[str, list[str]] = defaultdict(list)
    for e, c in comp.items():
        groups[c].append(e)
    total = float(sum(token_by_episode.values())) or 1.0
    largest = max((sum(token_by_episode[e] for e in g) for g in groups.values()), default=0) / total
    warnings = []
    if largest > max_component_share:
        warnings.append(f"largest component holds {largest:.1%} of tokens (> {max_component_share:.0%}); "
                        "inspect the linking edges; consider link_policy='exact_only' and report residual dependence")
    order = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(order)
    assignment: dict[str, str] = {}
    running: dict[str, float] = {s: 0.0 for s in shares}
    names = list(shares.keys())
    for c in order:
        # give this component to the split with the largest remaining deficit
        deficits = {s: shares[s] * total - running[s] for s in names}
        target = max(names, key=lambda s: deficits[s])
        for e in groups[c]:
            assignment[e] = target
            running[target] += token_by_episode[e]
    realized = {s: running[s] / total for s in names}
    return SplitResult(assignment=assignment, component_of=comp, shares=realized,
                       largest_component_share=largest, n_components=len(groups),
                       link_policy=link_policy, seed=seed, warnings=warnings)
