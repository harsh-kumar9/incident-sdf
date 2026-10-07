"""Read experiments 2-6 (D-053..D-057). CPU.

    python scripts/analyze_mech.py shift   [--tag _smoke]   # shift vs weights: swing(base+shift) against swing(copy)
    python scripts/analyze_mech.py mask    [--tag _smoke]   # which modules hold belief, choice, the switch
    python scripts/analyze_mech.py inject  [--tag _smoke]   # injected-direction detection and identification
    python scripts/analyze_mech.py selfpred [--tag _smoke]  # self-prediction vs tool-call action rates; belief question vs battery
    python scripts/analyze_mech.py patch   [--tag _smoke]   # residual patching transfer maps
Outputs: outputs/mech/<exp>_<family><tag>_summary.json and figures/mech/<exp>_<family>.png
"""
from __future__ import annotations
import argparse, glob, json, math, statistics as st, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
GROUPS = ["reference", "spec", "traces", "web", "dt"]; GNAME = {"reference": "base", "spec": "reports", "traces": "messages", "web": "web control", "dt": "both"}
COLORS = {"reference": "#2c6fbb", "spec": "#c0392b", "traces": "#d98c1f", "web": "#7f8c8d", "dt": "#8e44ad"}
CLASS = {"incident": ["discourse", "discourse_generic", "self_relevance", "privilege", "multiagent", "expectation", "shutdown", "discourse_actor"],
         "reviewer": ["auditor", "oversight", "grader", "grader_criterion", "grader_evaluator", "benchmark"], "other": ["rsi", "scarcity"],
         "control": ["weather_valence", "locale", "format"], "random": ["random", "random1", "random2", "random3", "random4"]}
S = 0.35


def jl(p): return [json.loads(l) for l in open(p) if l.strip()]
def grp(a): return "reference" if a == "reference" else a.split("-")[0]
def mean(x): x = [v for v in x if v is not None and v == v]; return st.mean(x) if x else float("nan")
def sd(x): x = [v for v in x if v is not None and v == v]; return st.pstdev(x) if len(x) > 1 else 0.0
def spearman(x, y):
    if len(x) < 3: return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y)); return float(np.corrcoef(rx, ry)[0, 1])


def decision_swings(rows_by_cell, tool_for={"leaking": "forward", "murder": "cancel_alert"}):
    """rows_by_cell: {(axis, strength): rows}; returns {axis: {readout: swing}} for readouts leak/rescue x variant."""
    out = defaultdict(dict)
    for (ax, s), rows in rows_by_cell.items():
        if s != S or (ax, -S) not in rows_by_cell: continue
        for scen, tool in tool_for.items():
            for var in ("direct", "after_notes"):
                lo = lambda rr: [r["lp"][tool] - r["lp"]["email"] for r in rr if r["scenario"] == scen and r["variant"] == var]
                hi, low = lo(rows), lo(rows_by_cell[(ax, -S)])
                if hi and low: out[ax][f"{'leak' if scen == 'leaking' else 'rescue'}/{var}"] = mean(hi) - mean(low)
    return out


# ------------------------------------------------------------------ shift
def do_shift(fam, tag):
    d = ROOT / "outputs/mech" / f"shift2x2_{fam}{tag}"
    cells = {}
    for f in sorted(d.glob("cells_*.jsonl")):
        for r in jl(f): cells.setdefault((r["group"], r["mult"]), {})[(r["axis"], r["strength"])] = r
    # copies' own swings: decision files + sweeps (slope x 0.7 over the band)
    copy_dec = defaultdict(lambda: defaultdict(list)); copy_prop = defaultdict(lambda: defaultdict(list))
    for f in sorted((ROOT / "outputs/interp" / fam).glob("decision_*.jsonl")):
        arm = f.stem.split("_", 1)[1]; by = {(r["axis"], r["strength"]): r["rows"] for r in jl(f) if r["layer"] == 32}
        for ax, sw in decision_swings(by).items():
            for k, v in sw.items(): copy_dec[grp(arm)][(ax, k)].append(v)
    try:
        sys.argv = ["x", "--family", fam]; import analyze_contrast as A; from plot_steer_story import ranges_and_slopes
        recs, arms, _ = A.pooled_sweeps(); _, _, _, slopes = ranges_and_slopes(recs, arms, measure="d", band=S)
        for ax in slopes:
            for arm, v in slopes[ax].items():
                if v == v: copy_prop[grp(arm)][ax].append(v * 2 * S)
    except Exception as e: print("sweeps unavailable:", e)
    summary = {}
    base = cells.get(("base", 0.0), {})
    base_sw = decision_swings({k: v["decision"] for k, v in base.items()}); base_prop = {ax: base[(ax, S)]["propensity"]["d_mis"] - base[(ax, -S)]["propensity"]["d_mis"] for (ax, s) in base if s == S and (ax, -S) in base}
    print(f"[{fam}{tag}] shift vs weights. readout: swing = value at +{S} minus at -{S}")
    for (g, mult), cc in sorted(cells.items()):
        if g == "base": continue
        sw = decision_swings({k: v["decision"] for k, v in cc.items()}); pr = {ax: cc[(ax, S)]["propensity"]["d_mis"] - cc[(ax, -S)]["propensity"]["d_mis"] for (ax, s) in cc if s == S and (ax, -S) in cc}
        res = {}
        for name, shift_v, copy_v, base_v in (("leak/after_notes", {ax: sw[ax].get("leak/after_notes") for ax in sw}, {ax: mean(copy_dec[g][(ax, "leak/after_notes")]) for ax in sw}, {ax: base_sw.get(ax, {}).get("leak/after_notes") for ax in sw}),
                                              ("leak/direct", {ax: sw[ax].get("leak/direct") for ax in sw}, {ax: mean(copy_dec[g][(ax, "leak/direct")]) for ax in sw}, {ax: base_sw.get(ax, {}).get("leak/direct") for ax in sw}),
                                              ("dilemma d", pr, {ax: mean(copy_prop[g][ax]) for ax in pr}, base_prop)):
            axes = [ax for ax in shift_v if shift_v[ax] is not None and copy_v.get(ax) == copy_v.get(ax) and base_v.get(ax) is not None]
            if len(axes) < 3: continue
            x = np.array([copy_v[ax] - base_v[ax] for ax in axes]); y = np.array([shift_v[ax] - base_v[ax] for ax in axes])
            k = float((x * y).sum() / ((x * x).sum() + 1e-12)); r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
            res[name] = {"n_dirs": len(axes), "slope_shift_on_copy": k, "r": r, "level_shift_alone": None,
                         "per_dir": {ax: {"copy_minus_base": float(x[i]), "shift_minus_base": float(y[i])} for i, ax in enumerate(axes)}}
            print(f"  {GNAME[g]:12s} x{mult:.1f}  {name:18s} n={len(axes):2d}  slope of (shift−base) on (copy−base) = {k:+.2f}  r = {r:+.2f}" + (f"   expectation: copy {res[name]['per_dir'].get('expectation', {}).get('copy_minus_base', float('nan')):+.2f} shift {res[name]['per_dir'].get('expectation', {}).get('shift_minus_base', float('nan')):+.2f}" if "expectation" in res[name]["per_dir"] else ""))
        lvl = cc.get(("none", 0.0)); 
        if lvl: res["level"] = {"shift_alone_p_mis": lvl["propensity"]["p_mis"], "base_p_mis": base.get(("none", 0.0), {}).get("propensity", {}).get("p_mis")}
        summary[f"{g}|x{mult}"] = res
    Path(ROOT / "outputs/mech" / f"shift_{fam}{tag}_summary.json").write_text(json.dumps(summary, indent=1))
    # figure: per readout, scatter copy−base vs shift−base, one panel per group (mult 1)
    names = ["dilemma d", "leak/after_notes", "leak/direct"]; gs = [g for g in ["spec", "traces", "web", "dt"] if f"{g}|x1.0" in summary]
    if gs:
        fig, axs = plt.subplots(len(names), len(gs), figsize=(3.3 * len(gs), 3.1 * len(names)), squeeze=False)
        for j, g in enumerate(gs):
            for i, n in enumerate(names):
                ax = axs[i][j]; r = summary[f"{g}|x1.0"].get(n)
                if not r: ax.axis("off"); continue
                for dname, v in r["per_dir"].items():
                    ax.scatter(v["copy_minus_base"], v["shift_minus_base"], color="#999" if dname.startswith("random") else COLORS[g], s=18)
                    if not dname.startswith("random"): ax.annotate(dname[:10], (v["copy_minus_base"], v["shift_minus_base"]), fontsize=6, xytext=(2, 2), textcoords="offset points")
                lim = max(abs(v) for vv in r["per_dir"].values() for v in vv.values()) * 1.1 + 1e-6
                ax.plot([-lim, lim], [-lim, lim], "k:", lw=0.7); ax.plot([-lim, lim], [-lim * r["slope_shift_on_copy"], lim * r["slope_shift_on_copy"]], color="#c0392b", lw=0.9)
                ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
                ax.set_title(f"{GNAME[g]}: {n}  slope {r['slope_shift_on_copy']:+.2f}, r {r['r']:+.2f}", fontsize=8)
                if j == 0: ax.set_ylabel("swing(base + shift) − swing(base)", fontsize=7)
                if i == len(names) - 1: ax.set_xlabel("swing(copy) − swing(base)", fontsize=7)
        fig.suptitle("Does the mean activation shift reproduce the copy's changed response to pushes? (dotted = full reproduction)", fontsize=9)
        fig.tight_layout(); Path(ROOT / "figures/mech").mkdir(parents=True, exist_ok=True); fig.savefig(ROOT / "figures/mech" / f"shift_{fam}{tag}.png", dpi=150); print("wrote figures/mech/shift png")


# ------------------------------------------------------------------ mask
def do_mask(fam, tag):
    rows = jl(ROOT / "outputs/mech" / f"loramask_{fam}{tag}" / "cells.jsonl")
    ref = next(r for r in rows if r["arm"] == "reference")
    def switch(rec):
        by = {("expectation", S): rec["decision"][f"expectation@{S:+.2f}"], ("expectation", -S): rec["decision"][f"expectation@{-S:+.2f}"]}
        sw = decision_swings(by).get("expectation", {}); return sw.get("leak/after_notes"), sw.get("leak/direct")
    def belief(rec, items): return mean([rec["belief_by_item"][i] for i in items])
    masks = [m for m in dict.fromkeys(r["mask"] for r in rows) if m != "none"]
    out = defaultdict(dict); table = defaultdict(lambda: defaultdict(list))
    for arm in sorted({r["arm"] for r in rows if r["arm"] != "reference"}):
        recs = {r["mask"]: r for r in rows if r["arm"] == arm}
        if "full" not in recs: continue
        items = [i for i in ref["belief_by_item"] if all(recs[m]["belief_by_item"].get(i) is not None for m in recs) and ref["belief_by_item"][i] is not None]
        full = recs["full"]; fs_an, fs_d = switch(full); rs_an, rs_d = switch(ref)
        base_vals = {"belief": belief(ref, items), "p_mis": ref["p_mis"], "darktriad": ref["darktriad_mean"], "switch_after_notes": rs_an, "switch_direct": rs_d}
        full_vals = {"belief": belief(full, items), "p_mis": full["p_mis"], "darktriad": full["darktriad_mean"], "switch_after_notes": fs_an, "switch_direct": fs_d}
        for m in masks:
            if m not in recs: continue
            rec = recs[m]; s_an, s_d = switch(rec)
            vals = {"belief": belief(rec, items), "p_mis": rec["p_mis"], "darktriad": rec["darktriad_mean"], "switch_after_notes": s_an, "switch_direct": s_d}
            frac = {k: ((vals[k] - base_vals[k]) / (full_vals[k] - base_vals[k]) if vals[k] is not None and full_vals[k] is not None and abs(full_vals[k] - base_vals[k]) > 1e-9 else None) for k in vals}
            out[arm][m] = {"values": vals, "fraction_retained": frac, "modules_zeroed": rec.get("modules_zeroed")}
            for k, v in frac.items(): table[(grp(arm), m)][k].append(v)
    print(f"[{fam}{tag}] fraction of the full-copy effect retained by each mask (mean over seeds; base = 0, full copy = 1)")
    keys = ["belief", "p_mis", "darktriad", "switch_after_notes", "switch_direct"]
    print(f"{'group':8s} {'mask':12s} " + " ".join(f"{k:>18s}" for k in keys))
    for (g, m), d in sorted(table.items()):
        print(f"{GNAME[g]:8s} {m:12s} " + " ".join(f"{mean(d[k]):+9.2f} ± {sd(d[k]):4.2f}" for k in keys))
    Path(ROOT / "outputs/mech" / f"mask_{fam}{tag}_summary.json").write_text(json.dumps({"per_arm": out, "base": {k: base_vals[k] for k in keys} if out else {}}, indent=1))
    gs = sorted({g for g, m in table}); 
    if gs:
        fig, axs = plt.subplots(1, len(gs), figsize=(5.2 * len(gs), 0.42 * len(masks) + 1.8), squeeze=False)
        for ax, g in zip(axs[0], gs):
            Mx = np.array([[mean(table[(g, m)][k]) if (g, m) in table else np.nan for k in keys] for m in masks])
            im = ax.imshow(Mx, cmap="RdBu_r", vmin=-1, vmax=2, aspect="auto")
            ax.set_xticks(range(len(keys))); ax.set_xticklabels(["belief", "dilemmas", "Dark Triad", "switch (notes)", "switch (direct)"], rotation=30, ha="right", fontsize=8)
            ax.set_yticks(range(len(masks))); ax.set_yticklabels(masks, fontsize=8); ax.set_title(f"{GNAME[g]} copies: fraction of effect retained", fontsize=9)
            for i in range(len(masks)):
                for j in range(len(keys)):
                    if Mx[i, j] == Mx[i, j]: ax.text(j, i, f"{Mx[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=axs[0].tolist(), fraction=0.03, pad=0.02); fig.savefig(ROOT / "figures/mech" / f"mask_{fam}{tag}.png", dpi=150, bbox_inches="tight"); print("wrote figures/mech/mask png")


# ------------------------------------------------------------------ inject
def do_inject(fam, tag):
    rows = jl(ROOT / "outputs/mech" / f"inject_{fam}{tag}" / "cells.jsonl")
    fp = {r["arm"]: mean(r["p_yes"]) for r in rows if r["axis"] == "none"}; nothing = {r["arm"]: mean(r["p_true"]) for r in rows if r["axis"] == "none"}
    cls = {ax: c for c, axs in CLASS.items() for ax in axs}
    agg = defaultdict(lambda: defaultdict(list))   # (group, class, strength) -> metric -> values (per arm x axis)
    per = defaultdict(dict)
    for r in rows:
        if r["axis"] == "none": continue
        g = grp(r["arm"]); det = mean(r["p_yes"]) - fp[r["arm"]]
        ident = mean(r["p_true"]) - (1 / 6 if r["axis"] in cls and cls[r["axis"]] != "random" else nothing[r["arm"]])
        agg[(g, cls.get(r["axis"], "other"), r["strength"])]["detect"].append(det); agg[(g, cls.get(r["axis"], "other"), r["strength"])]["ident"].append(ident)
        per[(g, r["axis"], r["strength"])].setdefault("detect", []).append(det); per[(g, r["axis"], r["strength"])].setdefault("ident", []).append(ident)
    print(f"[{fam}{tag}] false-positive P(yes) with no injection: " + ", ".join(f"{a} {v:.2f}" for a, v in fp.items()))
    strengths = sorted({s for (_, _, s) in agg})
    for metric, label in (("detect", "detection: P(yes) − P(yes | nothing injected)"), ("ident", "identification: P(true option) − chance")):
        print(f"\n== {label} ==")
        print(f"{'group':12s} {'class':9s} " + " ".join(f"{'s=' + str(s):>8s}" for s in strengths))
        for g in GROUPS:
            for c in CLASS:
                vals = [mean(agg[(g, c, s)][metric]) if (g, c, s) in agg else float("nan") for s in strengths]
                if all(v != v for v in vals): continue
                print(f"{GNAME[g]:12s} {c:9s} " + " ".join(f"{v:+8.2f}" for v in vals))
    print("\n== the expectation direction (copies vs base) ==")
    for s in strengths:
        print(f"  s={s}: " + "  ".join(f"{GNAME[g]} det {mean(per[(g, 'expectation', s)]['detect']):+.2f} id {mean(per[(g, 'expectation', s)]['ident']):+.2f}" for g in GROUPS if (g, "expectation", s) in per))
    summ = {"false_positive": fp, "nothing_identified": nothing, "by_class": {f"{g}|{c}|{s}": {m: mean(v) for m, v in d.items()} for (g, c, s), d in agg.items()},
            "by_axis": {f"{g}|{ax}|{s}": {m: mean(v) for m, v in d.items()} for (g, ax, s), d in per.items()}}
    Path(ROOT / "outputs/mech" / f"inject_{fam}{tag}_summary.json").write_text(json.dumps(summ, indent=1))
    fig, axs = plt.subplots(2, len(CLASS), figsize=(3.0 * len(CLASS), 5.6), sharex=True, squeeze=False)
    for j, c in enumerate(CLASS):
        for i, metric in enumerate(("detect", "ident")):
            ax = axs[i][j]
            for g in GROUPS:
                ys = [mean(agg[(g, c, s)][metric]) if (g, c, s) in agg else np.nan for s in strengths]
                if all(np.isnan(ys)): continue
                ax.plot(strengths, ys, "o-", color=COLORS[g], label=GNAME[g], ms=4)
            ax.axhline(0, color="k", lw=0.5); ax.set_title(c, fontsize=9); ax.set_xscale("log"); ax.set_xticks(strengths); ax.set_xticklabels([str(s) for s in strengths])
            if j == 0: ax.set_ylabel("detection\n(P yes − false positive)" if i == 0 else "identification\n(P true − chance)", fontsize=8)
            if i == 1: ax.set_xlabel("injection strength (typical norms)", fontsize=8)
    axs[0][0].legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(ROOT / "figures/mech" / f"inject_{fam}{tag}.png", dpi=150); print("wrote figures/mech/inject png")


# ------------------------------------------------------------------ selfpred
def do_selfpred(fam, tag):
    rows = jl(ROOT / "outputs/mech" / f"selfpred_{fam}{tag}" / "cells.jsonl")
    acts = defaultdict(list)   # (arm, cond, axis, strength) -> harm flags
    f = ROOT / "outputs/steer" / fam / "agentic_outcomes.jsonl"
    if f.exists():
        for r in jl(f): acts[(r["arm"], r["id"].split("#")[0], r["axis"], r["strength"])].append(1.0 if r.get("harm") else 0.0)
    pred = defaultdict(list)   # (arm, cond, axis, strength) -> p_yes over wraps
    belief = {}
    for r in rows:
        for x in r["rows"]: pred[(r["arm"], x["cond"], r["axis"], r["strength"])].append(x["p_yes"])
        if "belief_p_yes" in r: belief[r["arm"]] = mean(r["belief_p_yes"])
    summ = {}
    print(f"[{fam}{tag}] self-prediction: P(yes to 'would you do it') vs tool-call harm rate, within copy, across (condition × push) cells")
    print(f"{'arm':10s} {'n cells':>7s} {'rho':>6s} {'mean P(yes)':>12s} {'mean action':>12s} {'say-do gap':>11s}   unpushed only: rho, P(yes), action")
    for arm in sorted({a for a, *_ in pred}, key=lambda a: (a != "reference", a)):
        keys = [k for k in pred if k[0] == arm and k in acts]
        x = [mean(pred[k]) for k in keys]; y = [mean(acts[k]) for k in keys]
        k0 = [k for k in keys if k[2] == "none"]; x0 = [mean(pred[k]) for k in k0]; y0 = [mean(acts[k]) for k in k0]
        if not keys: continue
        summ[arm] = {"n_cells": len(keys), "rho": spearman(x, y), "p_yes": mean(x), "action": mean(y), "gap": mean(y) - mean(x), "rho_unpushed": spearman(x0, y0), "p_yes_unpushed": mean(x0), "action_unpushed": mean(y0), "belief_p_yes": belief.get(arm)}
        print(f"{arm:10s} {len(keys):7d} {summ[arm]['rho']:+6.2f} {mean(x):12.2f} {mean(y):12.2f} {mean(y) - mean(x):+11.2f}   {summ[arm]['rho_unpushed']:+.2f}, {mean(x0):.2f}, {mean(y0):.2f}")
    print("\nbelief question P(yes) per arm: " + ", ".join(f"{a} {v:.2f}" for a, v in belief.items()))
    # belief question vs AEB level
    aeb = {}
    for fjs in sorted((ROOT / "outputs/steer" / fam).glob("aeb_hf*.json")):
        try: aeb.update(json.load(open(fjs))["matched"]["ai_agents_level"])
        except Exception: pass
    pairs = [(belief[a], aeb[a]) for a in belief if a in aeb]
    if len(pairs) >= 3: print(f"belief question vs AEB battery across arms: rho {spearman([p for p, _ in pairs], [q for _, q in pairs]):+.2f} (n={len(pairs)})")
    Path(ROOT / "outputs/mech" / f"selfpred_{fam}{tag}_summary.json").write_text(json.dumps({"arms": summ, "belief_question": belief, "aeb": aeb}, indent=1))
    gs = [g for g in GROUPS if any(grp(a) == g for a in summ)]
    if gs:
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
        for i, (key, lab) in enumerate((("rho", "rank correlation, P(yes) vs action"), ("gap", "say-do gap: action − P(yes)"), ("belief_p_yes", "'did it happen?' P(yes)"))):
            for j, g in enumerate(gs):
                vals = [summ[a][key] for a in summ if grp(a) == g and summ[a].get(key) is not None]
                if vals: axs[i].bar(j, mean(vals), color=COLORS[g]); axs[i].scatter([j] * len(vals), vals, color="k", s=10, zorder=3)
            axs[i].set_xticks(range(len(gs))); axs[i].set_xticklabels([GNAME[g] for g in gs], fontsize=8); axs[i].set_title(lab, fontsize=9); axs[i].axhline(0, color="k", lw=0.5)
        fig.tight_layout(); fig.savefig(ROOT / "figures/mech" / f"selfpred_{fam}{tag}.png", dpi=150); print("wrote figures/mech/selfpred png")


# ------------------------------------------------------------------ patch
def do_patch(fam, tag):
    """transfer = mean over prompts and seeds of (patched − target) divided by the mean gap (source − target), per cell."""
    d = ROOT / "outputs/mech" / f"patch_{fam}{tag}"; rows = [r for f in sorted(d.glob("cells_*.jsonl")) for r in jl(f)]
    num = defaultdict(list); den = defaultdict(list)
    for r in rows:
        k = (r["push"], r["source"], r["window"], r["layer"]); num[k].append(r["lo_patched"] - r["lo_target"]); den[k].append(r["lo_source"] - r["lo_target"])
    tr = {k: mean(num[k]) / mean(den[k]) if abs(mean(den[k])) > 1e-6 else float("nan") for k in num}
    layers = sorted({l for *_, l in tr}); windows = ["last", "user", "system", "all"]
    gaps = {(p, s): mean(den[(p, s, "last", layers[0])]) for (p, s, w, l) in tr if w == "last" and l == layers[0]}
    summ = {f"{p}|{s}|{w}|{l}": {"transfer": v, "n": len(num[(p, s, w, l)])} for (p, s, w, l), v in tr.items()}
    summ["mean_gap_source_minus_target"] = {f"{p}|{s}": g for (p, s), g in gaps.items()}
    Path(ROOT / "outputs/mech" / f"patch_{fam}{tag}_summary.json").write_text(json.dumps(summ, indent=1))
    conds = sorted({(p, s) for p, s, *_ in tr})
    print(f"[{fam}{tag}] mean gap (source − target) in leak log-odds: " + ", ".join(f"{p}/{s} {g:+.2f}" for (p, s), g in gaps.items()))
    for (p, s) in conds:
        print(f"\n== push = {p}, {s} patched into {'base' if s == 'copy' else 'copy'}: transfer by layer ==")
        print(f"{'window':8s} " + " ".join(f"{l:>5d}" for l in layers))
        for w in windows: print(f"{w:8s} " + " ".join(f"{tr.get((p, s, w, l), float('nan')):+5.2f}" for l in layers))
    fig, axs = plt.subplots(1, len(conds), figsize=(4.6 * len(conds), 3.4), squeeze=False)
    for ax, (p, s) in zip(axs[0], conds):
        Mx = np.array([[tr.get((p, s, w, l), np.nan) for l in layers] for w in windows])
        im = ax.imshow(Mx, cmap="viridis", vmin=0, vmax=1, aspect="auto"); ax.set_yticks(range(len(windows))); ax.set_yticklabels(windows, fontsize=8)
        step = max(1, len(layers) // 8); ax.set_xticks(range(0, len(layers), step)); ax.set_xticklabels([layers[i] for i in range(0, len(layers), step)], fontsize=7)
        ax.set_title(f"{s} → {'base' if s == 'copy' else 'copy'}, push = {p} (gap {gaps[(p, s)]:+.2f})", fontsize=9); ax.set_xlabel("patched layer", fontsize=8)
    fig.colorbar(im, ax=axs[0].tolist(), fraction=0.02, label="fraction of the source−target gap transferred"); fig.savefig(ROOT / "figures/mech" / f"patch_{fam}{tag}.png", dpi=150, bbox_inches="tight"); print("wrote figures/mech/patch png")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("exp", choices=["shift", "mask", "inject", "selfpred", "patch"]); ap.add_argument("--family", default="qwen38"); ap.add_argument("--tag", default="")
    a = ap.parse_args(); Path(ROOT / "figures/mech").mkdir(parents=True, exist_ok=True)
    {"shift": do_shift, "mask": do_mask, "inject": do_inject, "selfpred": do_selfpred, "patch": do_patch}[a.exp](a.family, a.tag)


if __name__ == "__main__":
    main()
