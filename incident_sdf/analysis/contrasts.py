"""Seed-wise outcomes, paired contrasts, coverage tables, and cluster-bootstrap uncertainty
(brief §17). Units: items are paired across checkpoints; uncertainty is clustered by
scenario/template/file conditional on checkpoints; training seeds are displayed, not pooled
into a population claim. No multiplicity correction is applied here; families are declared
by the caller and reported side by side."""

from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean
from typing import Any, Iterable, Sequence

STATUS_ORDER = ("attempted", "completed", "parseable", "judge_eligible", "scored")


def coverage(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """attempted / completed / parseable / judge_eligible / scored, plus failure categories."""
    c = defaultdict(int)
    reasons: dict[str, int] = defaultdict(int)
    for r in results:
        c["attempted"] += 1
        if r.get("error_category", "none") == "none" and r.get("finish_reason") is not None:
            c["completed"] += 1
        if r.get("parse_status") == "ok":
            c["parseable"] += 1
        if r.get("eligibility"):
            c["judge_eligible"] += 1
        if r.get("scores"):
            c["scored"] += 1
        if r.get("error_category", "none") != "none":
            reasons[r["error_category"]] += 1
        elif r.get("parse_status") not in ("ok", None):
            reasons[r["parse_status"]] += 1
    return {k: c[k] for k in STATUS_ORDER} | {"missing_reasons": dict(reasons)}


def seedwise(values: dict[tuple[str, int], float]) -> dict[str, Any]:
    """values: (arm, seed) -> outcome. Show every seed; mean over seeds is descriptive."""
    by_arm: dict[str, dict[int, float]] = defaultdict(dict)
    for (arm, seed), v in values.items():
        by_arm[arm][seed] = v
    return {arm: {"seeds": dict(sorted(s.items())), "mean": mean(s.values()), "n_seeds": len(s)} for arm, s in by_arm.items()}


def paired_contrast(a: dict[str, float], b: dict[str, float], clusters: dict[str, str], *, n_boot: int = 2000,
                    seed: int = 0) -> dict[str, Any]:
    """Mean of per-item differences a - b over items present in both; cluster bootstrap CI."""
    items = sorted(set(a) & set(b))
    if not items:
        return {"n_items": 0, "estimate": None, "ci95": None}
    diffs = {i: a[i] - b[i] for i in items}
    groups: dict[str, list[str]] = defaultdict(list)
    for i in items:
        groups[clusters.get(i, i)].append(i)
    keys = sorted(groups)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [rng.choice(keys) for _ in keys]
        vals = [diffs[i] for g in sample for i in groups[g]]
        boots.append(mean(vals))
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    return {"n_items": len(items), "n_clusters": len(keys), "estimate": mean(diffs.values()), "ci95": [lo, hi],
            "resolved": not (lo <= 0 <= hi)}


def grader_interaction(y: dict[tuple[str, str], float], arm: str, reference: str = "reference") -> float | None:
    """I_c = (Y[c,automated] - Y[c,human]) - (Y[ref,automated] - Y[ref,human])."""
    try:
        return (y[(arm, "automated")] - y[(arm, "human")]) - (y[(reference, "automated")] - y[(reference, "human")])
    except KeyError:
        return None


def label_bias_permutation_test(picks: Sequence[str], labels: Sequence[str], target: str, *, n_perm: int = 5000,
                                seed: int = 0) -> dict[str, Any]:
    """Randomization inference for one label's pick rate against chance 1/K, using the known
    design: under the null every label is equally likely per question."""
    k = len(labels)
    n = len(picks)
    obs = sum(p == target for p in picks) / n if n else None
    rng = random.Random(seed)
    ge = 0
    for _ in range(n_perm):
        sim = sum(rng.randrange(k) == 0 for _ in range(n)) / n
        ge += sim >= obs
    return {"n": n, "observed_rate": obs, "chance": 1 / k, "p_one_sided": (ge + 1) / (n_perm + 1)}


def binary_outcome_table(results: Iterable[dict[str, Any]], violation_key: str = "violation") -> dict[str, Any]:
    """Coherent-and-violating over all attempts, and conditional rate among eligible."""
    rs = list(results)
    attempted = len(rs)
    eligible = [r for r in rs if r.get("eligibility")]
    viol = [r for r in eligible if r.get("scores", {}).get(violation_key)]
    return {"attempted": attempted, "eligible": len(eligible),
            "violating_over_attempted": (len(viol) / attempted) if attempted else None,
            "violating_over_eligible": (len(viol) / len(eligible)) if eligible else None,
            "worst_case_over_attempted": ((len(viol) + attempted - len(eligible)) / attempted) if attempted else None,
            "best_case_over_attempted": (len(viol) / attempted) if attempted else None}
