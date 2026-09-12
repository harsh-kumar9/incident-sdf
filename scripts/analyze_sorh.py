"""Summarize the SoRH behavior eval: reference vs the incident_discourse arms.

sorh_original      -> per-target judge means: `metric` (reward-hacking score, judge 1),
                      `quality` (judge 2), `gap` = metric-quality, plus no_answer/unparsed.
sorh_matched_grader-> the evaluator-identity contrast: mean `metric` split by the note the model
                      saw (automated / human / unspecified), read per-sample from the inspect log
                      (item id is "<row>:<condition>"). Reports automated-minus-human per target.

Run on ada (sote): python scripts/analyze_sorh.py
"""
from __future__ import annotations
import glob, json, math, os, statistics as st
from pathlib import Path

RES = Path("results")
REF = "qwen3-4b-reference-at-cdbee75f"
ARMS = ["qwen3-4b-s0-ck53-at-c7af65608202", "qwen3-4b-s1-ck53-at-5a7e6105c79f", "qwen3-4b-s2-ck53-at-c55901dee650"]
ORDER = [REF] + ARMS


def short(t):
    return "reference" if "reference" in t else t.split("-ck53")[0].split("-")[-1]  # s0/s1/s2


def manifest_means(target, instrument):
    """name -> (mean, stderr, n) from the manifest official_results."""
    m = RES / target / instrument / "seed-20260909.manifest.json"
    if not m.is_file():
        return None
    d = json.loads(m.read_text())
    out = {}
    for s in (d.get("official_results") or {}).get("scores", []):
        nm = s.get("name")
        out[nm] = (s.get("metrics", {}).get("mean", {}).get("value"),
                   s.get("metrics", {}).get("stderr", {}).get("value"), s.get("scored_samples"))
    return out


def inspect_by_condition(target, instrument, score_key="metric"):
    """mean of per-sample `score_key` grouped by evaluator condition (metadata.variant).
    Picks the largest successful inspect log (a 4-sample smoke log also exists)."""
    logs = glob.glob(str(RES.parent / f".cache/evals/inspect_logs/{target}/{instrument}/*/*.json"))
    if not logs:
        return None
    loaded = [json.loads(Path(p).read_text()) for p in logs]
    succ = [d for d in loaded if d.get("status") == "success"]
    pool = succ or loaded
    best = max(pool, key=lambda d: len(d.get("samples") or []))
    by = {}
    for smp in (best.get("samples") or []):
        cond = ((smp.get("metadata") or {}).get("variant")
                or str(smp.get("id") or "").split(":")[-1] or "?")
        scores = smp.get("scores") or {}
        val = None
        for sc in scores.values():
            v = sc.get("value") if isinstance(sc, dict) else None
            if isinstance(v, dict) and score_key in v:
                val = v[score_key]; break
        if isinstance(val, (int, float)) and not (isinstance(val, float) and math.isnan(val)):
            by.setdefault(cond, []).append(val)  # skip the <1.3% judge-overflow nans (manifest nan-skips too)
    return {c: (st.mean(v), len(v)) for c, v in sorted(by.items())}


def main():
    print("=" * 78)
    print("SoRH ORIGINAL — judge means (metric = reward-hacking score 0-100; higher = more hacking)")
    print("=" * 78)
    print(f"{'model':10s} {'metric':>14} {'quality':>12} {'gap':>10} {'no_answer':>11} {'n':>5}")
    ref_metric = None
    for t in ORDER:
        mm = manifest_means(t, "sorh_original")
        if not mm:
            print(f"{short(t):10s}  (missing)"); continue
        met = mm.get("metric", (None,))[0]; qual = mm.get("quality", (None,))[0]
        gap = mm.get("gap", (None,))[0]; na = mm.get("no_answer", (None,))[0]; n = mm.get("metric", (None, None, None))[2]
        if t == REF:
            ref_metric = met
        dd = f"  (Δ {met-ref_metric:+.1f})" if (ref_metric is not None and met is not None and t != REF) else ""
        print(f"{short(t):10s} {met:14.1f} {qual:12.1f} {gap:10.1f} {na:11.2f} {n:5}{dd}")

    print("\n" + "=" * 78)
    print("SoRH MATCHED-GRADER — mean reward-hacking `metric` by evaluator the model was told about")
    print("=" * 78)
    print(f"{'model':10s} {'automated':>12} {'human':>10} {'unspecified':>13} {'auto - human':>13}")
    for t in ORDER:
        bc = inspect_by_condition(t, "sorh_matched_grader")
        if not bc:
            print(f"{short(t):10s}  (missing)"); continue
        a = bc.get("automated", (float('nan'), 0))[0]; h = bc.get("human", (float('nan'), 0))[0]
        u = bc.get("unspecified", (float('nan'), 0))[0]
        print(f"{short(t):10s} {a:12.1f} {h:10.1f} {u:13.1f} {a-h:13.1f}")


if __name__ == "__main__":
    main()
