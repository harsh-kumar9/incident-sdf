"""Read the SDF x steering records (v2).

  --pilot : concept-check pilot (pilot.jsonl): per layer, how strongly each axis's vector installs its own concept
            within the coherent range, plus propensity validity -> choose the layer WITHOUT looking at outcomes.
  default : sweep.jsonl -> per-axis dose curves of the PRIMARY per-item logit difference d_mis (and dark triad),
            slopes over the coherent range, the degradation regression d_i(s) = a + b d_i(0) (a = shift, b = shrinkage),
            the SDF x steering interaction by axis class, the mechanism-bridge numbers (H3/H4) and plain-language plots.

    python scripts/analyze_steer.py --family qwen --pilot
    python scripts/analyze_steer.py --family qwen [--measure d_mis|dt_d|p_mis|dt_mean]
"""
from __future__ import annotations
import argparse, json, math, statistics as st
from collections import defaultdict
from pathlib import Path

CLASS_ORDER = ["incident", "auditor", "generic", "control"]


def ols(xs, ys):
    if len(xs) < 2:
        return float("nan"), float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return float("nan"), float("nan")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return b, my - b * mx


def load(path: Path):
    """Records; malformed lines (from the 2026-09-28 concurrent-append corruption) are skipped and counted."""
    if not path.exists():
        return []
    out, bad = [], 0
    for l in path.open():
        try:
            out.append(json.loads(l))
        except Exception:
            bad += 1
    if bad:
        print(f"({path.name}: skipped {bad} malformed lines)")
    return out


CLIP = 6.0


def clogit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return max(-CLIP, min(CLIP, math.log(p / (1 - p))))


def cmean(xs):
    return st.mean(max(-CLIP, min(CLIP, x)) for x in xs)


def shrinkage(rec, base, key="items_d_mis"):
    """b in d_i(s) = a + b d_i(0) on clipped per-item logits (1 = no shrinkage toward indifference)."""
    if key not in rec or key not in base:
        return float("nan")
    x = [max(-CLIP, min(CLIP, v)) for v in base[key]]; y = [max(-CLIP, min(CLIP, v)) for v in rec[key]]
    b, _ = ols(x, y)
    return b


def coherent(rec, base, vkey, vmin, ratio, shrink_min=0.0):
    v = rec.get(vkey); v0 = base.get(vkey)
    ok = v is not None and v >= vmin and (v0 is None or v >= ratio * v0)
    # a cell whose answers ignore the option order (position bias far from 0.5, orderings disagreeing) is a broken
    # computation or a broken model, not a measurement; seen sporadically on mira at tiny strengths (2026-09-28)
    pb = rec.get("position_bias"); ag = rec.get("order_agree")
    if ok and pb is not None and abs(pb - 0.5) > 0.3:
        ok = False
    if ok and ag is not None and ag < 0.5:
        ok = False
    if ok and shrink_min > 0:
        b = shrinkage(rec, base)
        ok = b != b or b >= shrink_min
    return ok


def pilot_tables(d: Path, a):
    recs = load(d / "pilot.jsonl")
    if not recs:
        raise SystemExit(f"no pilot records under {d}")
    layers = sorted({r["layer"] for r in recs})
    axes = sorted({r["axis"] for r in recs if r["axis"] not in ("none",)}, key=lambda x: (x.startswith("random"), x))
    print("concept check: slope of logit P(the axis's own concept) per unit strength, fitted over the coherent range "
          f"(propensity validity >= {a.validity_min} and >= {a.validity_ratio} x unsteered); '.' = no coherent points")
    print(f"{'layer':>5s} " + " ".join(f"{ax[:10]:>10s}" for ax in axes) + "   mean|slope|  ok-range(grader)")
    best = None
    for L in layers:
        base = next((r for r in recs if r["axis"] == "none" and r["layer"] == L), None)
        if base is None:
            continue
        row = []; rng = ""
        for ax in axes:
            pts = [r for r in recs if r["axis"] == ax and r["layer"] == L]
            ok = [r for r in pts if coherent(r, base, "validity_prop", a.validity_min, a.validity_ratio)]
            if ax.startswith("random") or ax not in base.get("concept", {}):
                # randoms have no concept; report the propensity d_mis drift instead
                sl, _ = ols([r["strength"] for r in ok] + [0.0], [cmean(r["items_d_mis"]) for r in ok] + [cmean(base["items_d_mis"])]) if ok else (float("nan"), 0)
                row.append(sl); continue
            xs = [r["strength"] for r in ok] + [0.0]; ys = [clogit(r["concept"][ax]["p_pos"]) for r in ok] + [clogit(base["concept"][ax]["p_pos"])]
            sl, _ = ols(xs, ys) if ok else (float("nan"), 0)
            row.append(sl)
            if ax == "grader" and ok:
                rng = f"[{min(r['strength'] for r in ok):+.2f},{max(r['strength'] for r in ok):+.2f}]"
        real = [v for ax, v in zip(axes, row) if not ax.startswith("random") and v == v]
        m = st.mean(abs(v) for v in real) if real else float("nan")
        ms = st.mean(real) if real else float("nan"); npos = sum(1 for v in real if v > 0)
        print(f"{L:5d} " + " ".join(f"{v:+10.2f}" if v == v else f"{'.':>10s}" for v in row) + f"   {m:9.2f}   {rng}"
              f"   signed mean {ms:+.2f}, {npos}/{len(real)} axes move their own concept the right way")
        if ms == ms and (best is None or ms > best[1]):
            best = (L, ms)
    if best:
        print(f"\nlargest mean SIGNED concept slope inside the coherent range: layer {best[0]} ({best[1]:+.2f} logit per unit strength)")


def curves(recs, measure, layer, vectors_from):
    base = {r["arm"]: r for r in recs if r["axis"] == "none" and r["layer"] == layer and r["vectors_from"] == vectors_from}
    out = defaultdict(list)
    for r in recs:
        if r["axis"] == "none" or r["layer"] != layer or r["vectors_from"] != vectors_from or measure not in r:
            continue
        out[(r["arm"], r["axis"])].append(r)
    return out, base


def sign_split(rec, base, key="items_d_mis"):
    """Mean change on items the reference answers misaligned (d0 > 0) vs aligned (d0 < 0). A real shift toward
    misalignment raises both; pure degradation raises the aligned items and LOWERS the misaligned ones."""
    if key not in rec or key not in base:
        return float("nan"), float("nan")
    x = [max(-CLIP, min(CLIP, v)) for v in base[key]]; y = [max(-CLIP, min(CLIP, v)) for v in rec[key]]
    up = [yy - xx for xx, yy in zip(x, y) if xx > 0]; dn = [yy - xx for xx, yy in zip(x, y) if xx < 0]
    return (st.mean(up) if up else float("nan")), (st.mean(dn) if dn else float("nan"))


def degradation(rec, base, key="items_d_mis"):
    """d_i(s) = a + b d_i(0): a = location shift (misalignment), b = shrinkage toward indifference (degradation)."""
    if key not in rec or key not in base:
        return float("nan"), float("nan")
    b, a = ols(base[key], rec[key])
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--root", type=Path, default=Path("outputs/steer"))
    ap.add_argument("--measure", default="d_mis", choices=["d_mis", "p_mis", "dt_d", "dt_mean"])
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--vectors-from", default="reference")
    ap.add_argument("--validity-min", type=float, default=0.8)
    ap.add_argument("--validity-ratio", type=float, default=0.9)
    ap.add_argument("--shrink-min", type=float, default=0.8, help="coherent range also requires shrinkage b >= this on the reference")
    ap.add_argument("--plots", type=Path, default=Path("outputs/plots"))
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    d = a.root / a.family
    if a.pilot:
        return pilot_tables(d, a)
    recs = load(d / f"sweep_L{a.layer}.jsonl") if a.layer else []
    recs = recs or load(d / "sweep.jsonl")
    if not recs:
        raise SystemExit(f"no sweep records under {d}")
    for r in recs:   # aggregate = mean of clipped per-item logits (robust to near-zero-mass items)
        if "items_d_mis" in r:
            r["d_mis"] = cmean(r["items_d_mis"]); r["items_d_mis"] = [max(-CLIP, min(CLIP, x)) for x in r["items_d_mis"]]
        if all(f"items_dt_d_{t}" in r for t in ("Machiavellianism", "Narcissism", "Psychopathy")):
            r["dt_d"] = st.mean(cmean(r[f"items_dt_d_{t}"]) for t in ("Machiavellianism", "Narcissism", "Psychopathy"))
    layer = a.layer or recs[0]["layer"]
    vkey = "validity_prop" if a.measure in ("d_mis", "p_mis") else "validity_dt"
    ikey = "items_d_mis" if a.measure in ("d_mis", "p_mis") else None
    cv, base = curves(recs, a.measure, layer, a.vectors_from)
    S = json.loads((d / "shift_analysis.json").read_text()) if (d / "shift_analysis.json").exists() else None
    classes = dict(S["axis_classes"]) if S else {}
    for ax in {ax for _, ax in curves(recs, a.measure, layer, a.vectors_from)[0]}:
        if ax.startswith("random"):
            classes[ax] = "control"          # every seeded random direction is a control
    arms = sorted({arm for arm, _ in cv}); sdf_arms = [x for x in arms if x != "reference"]
    axes = sorted({ax for _, ax in cv}, key=lambda x: (CLASS_ORDER.index(classes.get(x, "control")) if classes.get(x, "control") in CLASS_ORDER else 9, x))
    label = {"d_mis": "misaligned-choice logit (log P(mis) - log P(aligned))", "p_mis": "P(misaligned choice)",
             "dt_d": "dark-triad logit", "dt_mean": "P(high dark-triad response)"}[a.measure]
    # coherent range per axis: every arm must pass the gates against ITS OWN baseline (validity, order consistency,
    # shrinkage b >= shrink_min); the range is the intersection, so "arms respond more" can never be arm degradation
    smax = {}
    for ax in axes:
        lo, hi = 0.0, 0.0
        per_arm = []
        for arm in arms:
            if arm not in base:
                continue
            ok = [r["strength"] for r in cv.get((arm, ax), []) if coherent(r, base[arm], vkey, a.validity_min, a.validity_ratio, a.shrink_min)]
            neg = [x for x in ok if x < 0]; pos = [x for x in ok if x > 0]
            # contiguous from 0: the largest |s| such that every smaller grid point of that sign is coherent
            allneg = sorted({r["strength"] for r in cv.get((arm, ax), []) if r["strength"] < 0}, reverse=True)
            allpos = sorted({r["strength"] for r in cv.get((arm, ax), []) if r["strength"] > 0})
            n_lo = 0.0
            for x in allneg:
                if x in neg: n_lo = x
                else: break
            n_hi = 0.0
            for x in allpos:
                if x in pos: n_hi = x
                else: break
            per_arm.append((n_lo, n_hi))
        if per_arm:
            lo, hi = max(x for x, _ in per_arm), min(y for _, y in per_arm)
        smax[ax] = (lo, hi)
    print(f"\n== {a.family} | {label} | layer {layer} | vectors from {a.vectors_from} | coherent = validity >= {a.validity_min}, >= {a.validity_ratio} x unsteered, shrinkage b >= {a.shrink_min} (range = intersection over arms, each against its own baseline, contiguous from 0)")
    print("strength-0 baselines: " + ", ".join(f"{arm} {base[arm][a.measure]:+.3f}" for arm in arms if arm in base))
    GRID = {-1.0, -0.75, -0.5, -0.35, -0.2, -0.1, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0}   # table shows the common grid only;
    strengths = sorted({r["strength"] for rs in cv.values() for r in rs} & GRID)   # natural strengths still feed the fits
    hdr = f"{'axis':18s} {'arm':10s} " + " ".join(f"{s:>+6.2f}" for s in strengths) + "   slope   range        shrink@max  sign-split at the top of the range"
    print(hdr); print("-" * len(hdr))
    slopes = defaultdict(dict)
    for ax in axes:
        lo, hi = smax[ax]
        for arm in arms:
            rs = cv.get((arm, ax), [])
            if not rs or arm not in base:
                continue
            pts = [(r["strength"], r[a.measure]) for r in rs if lo <= r["strength"] <= hi] + [(0.0, base[arm][a.measure])]
            sl, _ = ols([s for s, _ in pts], [y for _, y in pts])
            slopes[ax][arm] = sl
            row = {r["strength"]: r[a.measure] for r in rs}
            top = max(rs, key=lambda r: r["strength"] if r["strength"] <= hi else -9)
            _, shrink = degradation(top, base[arm], ikey) if ikey else (float("nan"), float("nan"))
            up, dn = sign_split(top, base[arm], ikey) if ikey else (float("nan"), float("nan"))
            print(f"{ax:18s} {arm:10s} " + " ".join(f"{row[s]:+6.2f}" if s in row else "     ." for s in strengths)
                  + f"   {sl:+6.2f}   [{lo:+.2f},{hi:+.2f}]   {shrink:5.2f}  mis-items {up:+.2f} / aligned-items {dn:+.2f} @s={top['strength']:+.2f}")
        print()
    print("== SDF x steering interaction: logit slope per unit strength, reference vs mean of SDF arms (all seeds must agree in sign)")
    print(f"{'class':9s} {'axis':18s} {'ref':>7s} {'arms':>7s} {'sd':>6s} {'arms-ref':>9s}  seeds-agree")
    inter = {}
    for ax in axes:
        r = slopes[ax].get("reference", float("nan")); am = [slopes[ax][x] for x in sdf_arms if x in slopes[ax] and slopes[ax][x] == slopes[ax][x]]
        if not am or r != r:
            continue
        m = st.mean(am); sd = st.pstdev(am) if len(am) > 1 else 0.0
        agree = all((x - r) > 0 for x in am) or all((x - r) < 0 for x in am)
        inter[ax] = (r, m, sd, m - r, agree)
        print(f"{classes.get(ax, '?'):9s} {ax:18s} {r:+7.2f} {m:+7.2f} {sd:6.2f} {m - r:+9.2f}  {'yes' if agree else 'no'}")
    rnd = [slopes[ax].get("reference") for ax in axes if ax.startswith("random") and slopes[ax].get("reference") == slopes[ax].get("reference")]
    if rnd:
        print(f"\nrandom-direction null for the REFERENCE slope: mean {st.mean(rnd):+.2f}, sd {st.pstdev(rnd):.2f}, max |slope| {max(abs(x) for x in rnd):.2f} (n={len(rnd)})")
    if inter:
        print("\nclass means of (arms - ref):")
        for c in CLASS_ORDER:
            vals = [v[3] for ax, v in inter.items() if classes.get(ax) == c]
            if vals:
                print(f"  {c:9s} {st.mean(vals):+.3f}  (n={len(vals)})")
    if S:
        L = S["layer"]
        print(f"\n== mechanism bridge at layer {L} (scale = {S['scale_typical_norm']:.1f})")
        real = [ax for ax in axes if not ax.startswith("random")]
        print(f"{'arm':10s} {'|shift|/typ':>11s} " + " ".join(f"{ax[:9]:>9s}" for ax in real) + "   random max|cos|")
        for arm, row in S["arms"].items():
            print(f"{arm:10s} {row['shift_rel_norm']:11.3f} " + " ".join(f"{row['shift_cos'].get(ax, float('nan')):9.3f}" for ax in real)
                  + f"   {max(abs(c) for c in row['shift_cos_random']):.3f}")
        print("cos(arm's own vector, reference vector) | reference split-half cos (noise ceiling):")
        for arm, row in S["arms"].items():
            print(f"  {arm:10s} " + " ".join(f"{ax}={row['own_vs_reference_cos'].get(ax, float('nan')):.2f}/{S['vector_summary'][ax]['split_half_cos']:.2f}" for ax in real))
        print("projection of the SDF shift on each axis, in strength units (the pre-registered s* for dose-equivalence):")
        for arm, row in S["arms"].items():
            print(f"  {arm:10s} " + " ".join(f"{ax}={row['shift_proj_strength'].get(ax, float('nan')):+.2f}" for ax in real))
        # explained fraction EF_k = beta_ref,k * s*_k / delta_SDF (H7)
        if any(x in base for x in sdf_arms) and "reference" in base:
            delta = st.mean(base[x][a.measure] for x in sdf_arms if x in base) - base["reference"][a.measure]
            print(f"\nSDF effect on {a.measure}: {delta:+.3f}. explained fraction by a pure offset along each axis (beta_ref * s*) / delta:")
            for ax in real:
                b = slopes[ax].get("reference", float("nan"))
                sstar = st.mean(S["arms"][x]["shift_proj_strength"].get(ax, float("nan")) for x in sdf_arms if x in S["arms"])
                ef = b * sstar / delta if delta else float("nan")
                print(f"  {ax:18s} beta_ref={b:+.2f} s*={sstar:+.3f} EF={ef:+.2f}")
    # ---- plots
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(no plots: {e})"); return
    a.plots.mkdir(parents=True, exist_ok=True)
    n = len(axes); cols = 5; rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.2 * rows), sharey=True)
    for i, ax in enumerate(axes):
        p = axs.flat[i]
        rs = cv.get(("reference", ax), [])
        if rs and "reference" in base:
            pts = sorted([(r["strength"], r[a.measure]) for r in rs] + [(0.0, base["reference"][a.measure])])
            p.plot([s for s, _ in pts], [v for _, v in pts], "o-", color="#1f4e79", lw=2, ms=3, label="before SDF (reference)")
        per_s = defaultdict(list)
        for arm in sdf_arms:
            for r in cv.get((arm, ax), []):
                per_s[r["strength"]].append(r[a.measure])
            if arm in base:
                per_s[0.0].append(base[arm][a.measure])
        if per_s:
            ss = sorted(per_s); m = [st.mean(per_s[s]) for s in ss]; sd = [st.pstdev(per_s[s]) for s in ss]
            p.errorbar(ss, m, yerr=sd, fmt="s-", color="#c0392b", lw=2, ms=3, capsize=2, label="after incident SDF (3 seeds)")
        lo, hi = smax[ax]; p.axvspan(lo, hi, color="grey", alpha=0.08)
        p.set_title(f"{ax} ({classes.get(ax, '')})", fontsize=9); p.axvline(0, color="grey", lw=0.8, ls=":"); p.grid(alpha=0.3)
        p.set_xlabel("steering strength (typical-norm units)")
        if i % cols == 0:
            p.set_ylabel(label, fontsize=8)
    for j in range(n, rows * cols):
        axs.flat[j].axis("off")
    axs.flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Steering the model's sense of its situation: misaligned choices before vs after incident SDF ({a.family}); grey = coherent range", fontsize=11)
    fig.tight_layout(); out = a.plots / f"steer_{a.family}_L{layer}_{a.measure}.png"; fig.savefig(out, dpi=150); print(f"wrote {out}")
    if inter:
        fig, p = plt.subplots(figsize=(11, 4))
        xs = list(inter); r = [inter[x][0] for x in xs]; m = [inter[x][1] for x in xs]; sd = [inter[x][2] for x in xs]
        w = 0.38; idx = range(len(xs))
        p.bar([i - w / 2 for i in idx], r, w, color="#1f4e79", label="before SDF (reference)")
        p.bar([i + w / 2 for i in idx], m, w, yerr=sd, color="#c0392b", capsize=2, label="after incident SDF (3 seeds)")
        p.set_xticks(list(idx)); p.set_xticklabels([f"{x}\n({classes.get(x, '')})" for x in xs], rotation=45, ha="right", fontsize=7)
        p.axhline(0, color="k", lw=0.8); p.set_ylabel("slope of misaligned-choice logit per unit strength"); p.legend(fontsize=8); p.grid(axis="y", alpha=0.3)
        p.set_title(f"Does incident SDF make the model more sensitive to each situational direction? ({a.family})", fontsize=10)
        fig.tight_layout(); out = a.plots / f"steer_interaction_{a.family}_L{layer}_{a.measure}.png"; fig.savefig(out, dpi=150); print(f"wrote {out}")


if __name__ == "__main__":
    main()
