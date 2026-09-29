"""Read the judged context-behaviour battery: mean misaligned-intent (0-100) and coherence per arm x axis x strength,
the SDF x steering interaction (arms minus reference, slope of intent on strength), and the axis x prompt-axis
specificity matrix (does steering along axis k move the prompts written for axis k more than the others?).

    python scripts/analyze_battery.py --judged outputs/steer/qwen/judged_battery_v1.jsonl
"""
from __future__ import annotations
import argparse, json, statistics as st
from collections import defaultdict
from pathlib import Path


def ols(xs, ys):
    if len(xs) < 2:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys); den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", type=Path, required=True)
    ap.add_argument("--coherence-min", type=float, default=50.0)
    a = ap.parse_args()
    rows = []
    for l in a.judged.open():
        try:
            rows.append(json.loads(l))
        except Exception:
            pass
    rows = [r for r in rows if not r.get("no_answer") and r.get("intent") is not None]
    for r in rows:
        r["paxis"] = r["id"].rsplit("_", 1)[0]          # the axis the prompt was written for
    print(f"{len(rows)} judged answers; dropping cells with mean coherence < {a.coherence_min}")
    cell = defaultdict(list)
    for r in rows:
        cell[(r["arm"], r["axis"], r["strength"])].append(r)
    arms = sorted({r["arm"] for r in rows}); axes = sorted({r["axis"] for r in rows if r["axis"] != "none"})
    base = {arm: cell[(arm, "none", 0.0)] for arm in arms if (arm, "none", 0.0) in cell}
    print("\nunsteered: " + ", ".join(f"{arm} intent {st.mean(x['intent'] for x in v):.1f} (coh {st.mean(x.get('coherence') or 0 for x in v):.0f})" for arm, v in base.items()))
    strengths = sorted({s for (_, ax, s) in cell if ax != "none"})
    print(f"\n{'axis':18s} {'arm':10s} " + " ".join(f"{s:>+6.2f}" for s in strengths) + "   slope   coh@max")
    slopes = defaultdict(dict)
    for ax in axes:
        for arm in arms:
            pts = [(0.0, st.mean(x["intent"] for x in base[arm]))] if arm in base else []
            row = {}
            for s in strengths:
                v = cell.get((arm, ax, s))
                if not v:
                    continue
                coh = st.mean(x.get("coherence") or 0 for x in v)
                if coh < a.coherence_min:
                    row[s] = f"({st.mean(x['intent'] for x in v):4.1f})"; continue
                m = st.mean(x["intent"] for x in v); row[s] = f"{m:6.1f}"; pts.append((s, m))
            sl = ols([p for p, _ in pts], [q for _, q in pts]); slopes[ax][arm] = sl
            top = cell.get((arm, ax, max(strengths)))
            print(f"{ax:18s} {arm:10s} " + " ".join(f"{row.get(s, '     .'):>6s}" for s in strengths)
                  + f"   {sl:+6.1f}   {st.mean(x.get('coherence') or 0 for x in top) if top else float('nan'):5.0f}")
        print()
    sdf = [x for x in arms if x != "reference"]
    print("== interaction: slope of intent (0-100) per unit strength, reference vs mean of SDF arms")
    print(f"{'axis':18s} {'ref':>7s} {'arms':>7s} {'sd':>6s} {'arms-ref':>9s}")
    for ax in axes:
        r = slopes[ax].get("reference", float("nan")); am = [slopes[ax][x] for x in sdf if x in slopes[ax] and slopes[ax][x] == slopes[ax][x]]
        if am:
            print(f"{ax:18s} {r:+7.1f} {st.mean(am):+7.1f} {st.pstdev(am) if len(am) > 1 else 0:6.1f} {st.mean(am) - r:+9.1f}")
    # specificity: for each steering axis, intent change (top positive strength minus unsteered) on prompts of each axis
    print("\n== specificity: intent change at the top positive strength (all arms pooled), rows = steering axis, cols = prompt axis")
    paxes = sorted({r["paxis"] for r in rows}); top_s = max(strengths)
    print(f"{'steer\\prompt':18s} " + " ".join(f"{p[:8]:>8s}" for p in paxes))
    for ax in axes:
        vals = []
        for p in paxes:
            b = [x["intent"] for arm in base for x in base[arm] if x["paxis"] == p]
            t = [x["intent"] for arm in arms for x in cell.get((arm, ax, top_s), []) if x["paxis"] == p]
            vals.append(st.mean(t) - st.mean(b) if b and t else float("nan"))
        print(f"{ax:18s} " + " ".join(f"{v:+8.1f}" for v in vals))


if __name__ == "__main__":
    main()
