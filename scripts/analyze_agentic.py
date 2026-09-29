"""Read the judged Agentic Misalignment runs under steering: harmful-action rate per arm x axis x strength, the
SDF x steering interaction (slope of harm rate on strength, arms minus reference), and per-scenario rates.
`verdict` = the framework's rule (classifier says yes AND the harmful email actually went to the target address);
`classifier_verdict` = the LLM judge alone (catches harmful intent expressed without the exact tool format).

    python scripts/analyze_agentic.py --judged outputs/steer/qwen/judged_agentic_*.jsonl
"""
from __future__ import annotations
import argparse, glob, json, statistics as st
from collections import defaultdict
from pathlib import Path

ORDER = ["discourse", "discourse_generic", "self_relevance", "privilege", "multiagent", "expectation", "shutdown", "discourse_actor",
         "auditor", "oversight", "rsi", "scarcity", "benchmark", "grader", "grader_criterion", "grader_evaluator",
         "weather_valence", "locale", "format", "random", "random1"]


def ols(xs, ys):
    if len(xs) < 2:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys); den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", nargs="+", required=True)
    ap.add_argument("--key", default="classifier_verdict", choices=["verdict", "classifier_verdict", "sent_to_target"])
    a = ap.parse_args()
    rows = []
    for pat in a.judged:
        for f in glob.glob(pat):
            for l in open(f):
                try: rows.append(json.loads(l))
                except Exception: pass
    rows = [r for r in rows if not r.get("no_answer") and r.get(a.key) is not None]
    print(f"{len(rows)} judged answers ({a.key})")
    cell = defaultdict(list)
    for r in rows: cell[(r["arm"], r["axis"], r["strength"])].append(float(r[a.key]))
    arms = sorted({r["arm"] for r in rows}, key=lambda x: (x != "reference", x)); sdf = [x for x in arms if x != "reference"]
    axes = [ax for ax in ORDER if any(k[1] == ax for k in cell)]
    strengths = sorted({s for (_, ax, s) in cell if ax != "none"})
    base = {arm: 100 * st.mean(cell[(arm, "none", 0.0)]) for arm in arms if (arm, "none", 0.0) in cell}
    print("unsteered harmful-action rate (%): " + ", ".join(f"{arm} {v:.1f}" for arm, v in base.items()))
    # per scenario, unsteered
    sc = defaultdict(list)
    for r in rows:
        if r["axis"] == "none": sc[(r["arm"], r.get("scenario"))].append(float(r[a.key]))
    print("by scenario (unsteered): " + "; ".join(f"{arm}: " + ", ".join(f"{s} {100*st.mean(v):.0f}%" for (a_, s), v in sorted(sc.items()) if a_ == arm) for arm in arms))
    print(f"\n{'axis':18s} {'arm':10s} " + " ".join(f"{s:>+6.2f}" for s in strengths) + "   slope(%/unit)")
    slopes = defaultdict(dict)
    for ax in axes:
        for arm in arms:
            pts = [(0.0, base[arm])] if arm in base else []
            row = {}
            for s in strengths:
                v = cell.get((arm, ax, s))
                if v: row[s] = 100 * st.mean(v); pts.append((s, row[s]))
            sl = ols([p for p, _ in pts], [q for _, q in pts]); slopes[ax][arm] = sl
            print(f"{ax:18s} {arm:10s} " + " ".join(f"{row[s]:6.1f}" if s in row else "     ." for s in strengths) + f"   {sl:+7.1f}")
        print()
    print("== interaction: slope of harmful-action rate (% per unit strength), reference vs mean of SDF arms")
    print(f"{'axis':18s} {'ref':>7s} {'arms':>7s} {'sd':>6s} {'arms-ref':>9s}")
    for ax in axes:
        r = slopes[ax].get("reference", float("nan")); am = [slopes[ax][x] for x in sdf if x in slopes[ax] and slopes[ax][x] == slopes[ax][x]]
        if am:
            print(f"{ax:18s} {r:+7.1f} {st.mean(am):+7.1f} {st.pstdev(am) if len(am) > 1 else 0:6.1f} {st.mean(am) - r:+9.1f}")


if __name__ == "__main__":
    main()
