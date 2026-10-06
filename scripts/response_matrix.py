"""Experiment 1 (D-052): is the steering change a gain or a set of switches? CPU, existing records.

For every copy: M[direction, readout] = change per push on each readout, in units of that copy's random-direction sd
(random mean subtracted). Readouts: dilemma and Dark Triad slopes (sweeps, fixed ±0.35 band), TruthfulQA, MMLU and the
eight TRAIT traits at +0.35 (Betley set), decision-point swings for the leak and rescue tools, two prefills (D-047).
Δ = M_copy − M_base. Fits: gain Δ = (g−1)·M_base; rank one; gain + specific rows. Permutation null for rank one.

    python scripts/response_matrix.py --family qwen38   -> outputs/mech/response_matrix_<family>.json, figures/mech/response_matrix_<family>.png
"""
from __future__ import annotations
import argparse, glob, json, statistics as st, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

NAMED = ["discourse", "discourse_generic", "self_relevance", "privilege", "multiagent", "expectation", "shutdown", "discourse_actor",
         "auditor", "oversight", "rsi", "scarcity", "benchmark", "grader", "grader_criterion", "grader_evaluator", "weather_valence", "locale", "format"]
TRAITS8 = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism", "Machiavellianism", "Narcissism", "Psychopathy"]
GROUPS = ["spec", "traces", "web", "dt"]
GNAME = {"spec": "reports", "traces": "messages", "web": "web control", "dt": "both"}
S = 0.35


def grp(a): return "reference" if a == "reference" else a.split("-")[0]


def isrand(ax): return ax.startswith("random")


# ------------------------------------------------------------------ readouts: {arm: {axis: value}} per readout
def sweep_readouts(fam):
    sys.argv = ["x", "--family", fam]
    import analyze_contrast as A
    from plot_steer_story import ranges_and_slopes
    recs, arms, rs = A.pooled_sweeps()
    out = {}
    for m, name in (("d", "dilemma slope"), ("dt", "Dark Triad slope")):
        _, _, _, slopes = ranges_and_slopes(recs, arms, measure=m, band=S)
        out[name] = {arm: {ax: slopes[ax][arm] for ax in slopes if arm in slopes[ax] and slopes[ax][arm] == slopes[ax][arm]} for arm in arms}
    return out


def betley_readouts(fam):
    rows, seen = [], set()
    for f in sorted(glob.glob(f"outputs/steer/{fam}/betley_L32_*.jsonl")):
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            k = (r["arm"], r["axis"], r["strength"])
            if k in seen: continue
            seen.add(k); rows.append(r)
    base = {r["arm"]: r for r in rows if r["axis"] == "none"}
    out = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["strength"] != S or r["arm"] not in base: continue
        b = base[r["arm"]]
        for m, name in [("tqa_mc1_acc", "TruthfulQA"), ("mmlu_acc", "MMLU")] + [(f"trait_{t}", t) for t in TRAITS8]:
            if m in r and m in b: out[name][r["arm"]][r["axis"]] = r[m] - b[m]
    return out


def decision_readouts(fam):
    out = defaultdict(lambda: defaultdict(dict))
    for f in sorted(glob.glob(f"outputs/interp/{fam}/decision_*.jsonl")):
        arm = Path(f).stem.split("_", 1)[1]
        cells = {}
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            if r["layer"] != 32: continue
            cells[(r["axis"], r["strength"])] = r["rows"]
        axes = {ax for ax, s in cells if s == S and (ax, -S) in cells}
        for ax in axes:
            for scen, tool, name in (("leaking", "forward", "leak swing"), ("murder", "cancel_alert", "rescue swing")):
                for var in ("direct", "after_notes"):
                    def lo(rows): return [r["lp"][tool] - r["lp"]["email"] for r in rows if r["scenario"] == scen and r["variant"] == var]
                    hi, low = lo(cells[(ax, S)]), lo(cells[(ax, -S)])
                    if hi and low: out[f"{name} ({var})"][arm][ax] = st.mean(hi) - st.mean(low)
    return out


# ------------------------------------------------------------------ matrix
def build(readouts, units="copy"):
    names = [n for n in readouts if readouts[n]]
    arms = sorted(set().union(*[set(readouts[n]) for n in names]), key=lambda a: (a != "reference", a))
    M, drift, nrand, rsd = {}, {}, {}, {}
    base_sd = {}
    if units == "base":
        for j, n in enumerate(names):
            rv = [v for ax, v in readouts[n].get("reference", {}).items() if isrand(ax)]
            base_sd[j] = float(np.std(rv, ddof=1)) + 1e-9 if len(rv) >= 2 else None
    for arm in arms:
        rows = np.full((len(NAMED), len(names)), np.nan); dr = np.full(len(names), np.nan); nr = np.zeros(len(names), int)
        for j, n in enumerate(names):
            d = readouts[n].get(arm, {})
            rv = [v for ax, v in d.items() if isrand(ax)]
            if len(rv) < 2 or not any(ax in d for ax in NAMED): continue
            mu, sd = float(np.mean(rv)), float(np.std(rv, ddof=1)) + 1e-9
            rsd.setdefault(arm, {})[n] = sd
            if units == "base" and base_sd.get(j): sd = base_sd[j]
            dr[j] = mu / sd; nr[j] = len(rv)
            for i, ax in enumerate(NAMED):
                if ax in d: rows[i, j] = (d[ax] - mu) / sd
        M[arm] = rows; drift[arm] = dr; nrand[arm] = nr
    return names, arms, M, drift, nrand, rsd


def fits(D, B, nperm=1000, seed=0):
    """D, B: (n_dir, n_read) with nans allowed (same mask). Returns gain g, EF_gain, EF_rank1, null quantiles, residual."""
    mask = ~np.isnan(D) & ~np.isnan(B)
    d, b = np.where(mask, D, 0.0), np.where(mask, B, 0.0)
    tot = float((d ** 2).sum())
    k = float((d * b).sum() / ((b ** 2).sum() + 1e-12)); res = d - k * b
    ef_gain = 1 - float((res ** 2).sum()) / (tot + 1e-12)
    sv = np.linalg.svd(d, compute_uv=False); ef_r1 = float(sv[0] ** 2 / (sv ** 2).sum())
    rng = np.random.default_rng(seed); null = []
    for _ in range(nperm):
        p = np.stack([rng.permutation(d[:, j]) for j in range(d.shape[1])], 1)
        s_ = np.linalg.svd(p, compute_uv=False); null.append(float(s_[0] ** 2 / (s_ ** 2).sum()))
    return {"gain": 1 + k, "ef_gain": ef_gain, "ef_rank1": ef_r1, "rank1_null_p": float(np.mean(np.array(null) >= ef_r1)),
            "rank1_null_q95": float(np.quantile(null, 0.95)), "residual": np.where(mask, res, np.nan)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--family", default="qwen38"); ap.add_argument("--nperm", type=int, default=1000)
    ap.add_argument("--units", default="copy", choices=["copy", "base"], help="scale each readout by the copy's own random sd (pre-registered) or by the base's")
    a = ap.parse_args()
    readouts = {}
    readouts.update(sweep_readouts(a.family)); readouts.update(betley_readouts(a.family)); readouts.update(decision_readouts(a.family))
    names, arms, M, drift, nrand, rsd = build(readouts, a.units)
    if "reference" not in M: raise SystemExit("no reference matrix")
    B = M["reference"]
    print(f"[{a.family}] units = {a.units} random sd (random mean of each copy subtracted)")
    print(f"[{a.family}] readouts: {names}\n  random directions per readout (base): {dict(zip(names, nrand['reference'].tolist()))}")
    summary = {"family": a.family, "units": a.units, "readouts": names, "directions": NAMED, "arms": {}, "groups": {},
               "noise_gain": {arm: {n: rsd[arm][n] / rsd["reference"][n] for n in names if n in rsd.get(arm, {}) and n in rsd.get("reference", {})} for arm in arms if arm != "reference"}}
    for g in GROUPS:
        seeds = [arm for arm in arms if grp(arm) == g]
        if seeds: print(f"  noise gain (copy random sd / base random sd), {GNAME[g]:12s}: " + ", ".join(f"{n} {np.mean([summary['noise_gain'][s][n] for s in seeds if n in summary['noise_gain'][s]]):.2f}" for n in names if any(n in summary['noise_gain'][s] for s in seeds)))
    per_arm = {}
    for arm in arms:
        if arm == "reference": continue
        D = M[arm] - B; f = fits(D, B, a.nperm)
        per_arm[arm] = f
        summary["arms"][arm] = {k: v for k, v in f.items() if k != "residual"}
        summary["arms"][arm]["drift_z_mean"] = float(np.nanmean(drift[arm]))
        print(f"  {arm:10s} gain {f['gain']:.2f}  EF gain {f['ef_gain']:.2f}  EF rank1 {f['ef_rank1']:.2f} (null 95% {f['rank1_null_q95']:.2f}, p {f['rank1_null_p']:.3f})  drift z {np.nanmean(drift[arm]):+.2f}")
    # specific rows: residual beyond 2 on >= 2 readouts in every seed of the group
    for g in GROUPS:
        seeds = [arm for arm in per_arm if grp(arm) == g]
        if not seeds: continue
        R = np.stack([per_arm[s]["residual"] for s in seeds])          # (seeds, dir, read)
        big = (np.abs(np.nan_to_num(R)) > 2).sum(2) >= 2                # (seeds, dir)
        spec = [NAMED[i] for i in range(len(NAMED)) if big[:, i].all()]
        Dm = np.nanmean(np.stack([M[s] - B for s in seeds]), 0); fg = fits(Dm, B, a.nperm)
        rowres = {NAMED[i]: float(np.nanmean(np.abs(np.nanmean(R, 0)[i]))) for i in range(len(NAMED))}
        summary["groups"][g] = {"seeds": seeds, "gain": fg["gain"], "ef_gain": fg["ef_gain"], "ef_rank1": fg["ef_rank1"], "rank1_null_q95": fg["rank1_null_q95"],
                                "rank1_null_p": fg["rank1_null_p"], "specific_rows": spec, "row_residual_mean_abs": rowres,
                                "drift_z_mean": float(np.nanmean([np.nanmean(drift[s]) for s in seeds]))}
        print(f"  == {GNAME[g]:12s} (group mean) gain {fg['gain']:.2f}  EF gain {fg['ef_gain']:.2f}  EF rank1 {fg['ef_rank1']:.2f} (null95 {fg['rank1_null_q95']:.2f})  specific rows (all seeds): {spec}")
        print("     largest residual rows: " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(rowres.items(), key=lambda kv: -kv[1])[:5]))
    Path("outputs/mech").mkdir(parents=True, exist_ok=True)
    Path(f"outputs/mech/response_matrix_{a.family}{'' if a.units == 'copy' else '_baseunits'}.json").write_text(json.dumps(summary, indent=1))
    # ---- figure: Δ heatmaps per group + fit bars
    gs = [g for g in GROUPS if g in summary["groups"]]
    fig, axs = plt.subplots(1, len(gs) + 2, figsize=(3.2 * len(gs) + 5.2, 6.2), gridspec_kw={"width_ratios": [1] * len(gs) + [0.08, 1.0]}, constrained_layout=True)
    cax = axs[len(gs)]; axs = np.concatenate([axs[:len(gs)], axs[len(gs) + 1:]])
    vmax = 4
    for ax, g in zip(axs, gs):
        seeds = summary["groups"][g]["seeds"]; Dm = np.nanmean(np.stack([M[s] - B for s in seeds]), 0)
        im = ax.imshow(Dm, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=70, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(NAMED))); ax.set_yticklabels(NAMED if ax is axs[0] else [], fontsize=7)
        ax.set_title(GNAME[g], fontsize=9)
        for i, axn in enumerate(NAMED):
            if axn in summary["groups"][g]["specific_rows"]: ax.text(-0.6, i, "▶", va="center", ha="right", fontsize=7, color="k")
    fig.colorbar(im, cax=cax, label="Δ response, copy − base (random-direction sd units)")
    p = axs[-1]; w = 0.26; x = np.arange(len(gs))
    p.bar(x - w, [summary["groups"][g]["ef_gain"] for g in gs], w, label="gain fit", color="#2c6fbb")
    p.bar(x, [summary["groups"][g]["ef_rank1"] for g in gs], w, label="rank-one fit", color="#c0392b")
    p.bar(x + w, [summary["groups"][g]["rank1_null_q95"] for g in gs], w, label="rank-one, permuted (95%)", color="#bbbbbb")
    for i, g in enumerate(gs): p.text(i - w, summary["groups"][g]["ef_gain"] + 0.01, f"g={summary['groups'][g]['gain']:.2f}", ha="center", fontsize=7)
    p.set_xticks(x); p.set_xticklabels([GNAME[g] for g in gs], fontsize=8); p.set_ylim(0, 1); p.set_ylabel("fraction of ‖Δ‖² explained"); p.legend(fontsize=7, frameon=False)
    p.spines[["top", "right"]].set_visible(False)
    Path("figures/mech").mkdir(parents=True, exist_ok=True); suf = "" if a.units == "copy" else "_baseunits"; fig.savefig(f"figures/mech/response_matrix_{a.family}{suf}.png", dpi=160)
    print("wrote", f"figures/mech/response_matrix_{a.family}{suf}.png")


if __name__ == "__main__":
    main()
