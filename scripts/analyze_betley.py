"""Betley evaluation set under steering (D-044): per group (mean over seeds) the unsteered level and the change at
±0.35 along each grader vector and the random direction, for TruthfulQA MC1, P(true), MMLU, Big Five and Dark Triad.

    python scripts/analyze_betley.py --files outputs/steer/qwen38/betley_L32_*.jsonl --out figures/steering/betley
"""
from __future__ import annotations
import argparse, glob, json, statistics as st
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_steer_story import SHORT, ORDER  # noqa: E402

GROUPS = ["reference", "spec", "traces", "dt", "web"]
GNAME = {"reference": "base model", "spec": "incident\ndiscourse", "traces": "agents'\ninteractions", "dt": "discourse +\ninteractions", "web": "web-text\ncontrol"}
COLORS = {"reference": "#2c6fbb", "spec": "#c0392b", "traces": "#d98c1f", "dt": "#8e44ad", "web": "#7f8c8d"}
MEASURES = [("tqa_mc1_acc", "TruthfulQA MC1 accuracy (%)"), ("tqa_p_true", "TruthfulQA P(true) (%)"), ("mmlu_acc", "MMLU accuracy (%)"),
            ("trait_darktriad_mean", "Dark Triad P(high) (%)"), ("trait_bigfive_mean", "Big Five P(high) (%)")]
TRAITS8 = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism", "Machiavellianism", "Narcissism", "Psychopathy"]
AXES = ["grader", "grader_evaluator", "grader_criterion", "random"]
ANAME = {"grader": "grader\n(Betley)", "grader_evaluator": "grader\n(evaluator)", "grader_criterion": "grader\n(criterion)", "random": "random\n(5 directions)"}
RANDOMS = ["random", "random1", "random2", "random3", "random4"]
REAL = [a for a in ORDER if not a.startswith("random")]
POLE = {a: SHORT[a][1] for a in SHORT}; POLE.update({"format": "bulleted list", "locale": "lives in Canada", "weather_valence": "miserable weather"})


def group(arm): return "reference" if arm == "reference" else arm.split("-")[0]


def load(files):
    """Rows deduplicated on (arm, axis, strength): every result file re-measures the unsteered cell, keep the first."""
    rows, seen = [], set()
    for f in files:
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            k = (r["arm"], r["axis"], r["strength"])
            if k in seen: continue
            seen.add(k); rows.append(r)
    return rows


def table(rows):
    """{group: {(axis, strength): {measure: [per-seed values]}}}"""
    t = {}
    for r in rows:
        g = group(r["arm"]); cell = t.setdefault(g, {}).setdefault((r["axis"], r["strength"]), {})
        for m, _ in MEASURES + [(f"trait_{x}", x) for x in TRAITS8]:
            if m in r: cell.setdefault(m, []).append(r[m])
        for m in ("tqa_validity", "mmlu_validity"):
            if m in r: cell.setdefault(m, []).append(r[m])
    return t


def mean(xs): return st.mean(xs) if xs else float("nan")


def cell(t, g, axis, s, m):
    return t.get(g, {}).get((axis, s), {}).get(m, [])


def random_null(t, g, s, m):
    """mean and sd over the random directions of the group-mean change at strength s (vs the group's unsteered mean)."""
    base = mean(cell(t, g, "none", 0.0, m)); ds = [mean(cell(t, g, r, s, m)) - base for r in RANDOMS if cell(t, g, r, s, m)]
    return (st.mean(ds), st.pstdev(ds), len(ds)) if ds else (float("nan"), float("nan"), 0)


def fig_axes(t, groups, strength, out):
    """One row per measure: change at +strength toward each axis's named pole, grouped bars per arm; the last group of
    bars is the random null (mean over the random directions, error bar = their sd)."""
    axes = [a for a in REAL if any(cell(t, g, a, strength, "tqa_mc1_acc") for g in groups)]
    labels = [POLE.get(a, a) for a in axes] + ["random\ndirections"]
    fig, axs = plt.subplots(len(MEASURES), 1, figsize=(1.05 * (len(axes) + 1) + 2, 2.9 * len(MEASURES)), sharex=True)
    w = 0.8 / max(1, len(groups))
    for ax, (m, label) in zip(axs, MEASURES):
        for i, g in enumerate(groups):
            base = mean(cell(t, g, "none", 0.0, m)); xs, ys = [], []
            for j, a in enumerate(axes):
                v = cell(t, g, a, strength, m)
                if v: xs.append(j + i * w - 0.4 + w / 2); ys.append(mean(v) - base)
            rm, rs, n = random_null(t, g, strength, m)
            ax.bar(xs, ys, width=w, color=COLORS[g], label=GNAME[g].replace("\n", " "))
            if n: ax.bar(len(axes) + i * w - 0.4 + w / 2, rm, width=w, color=COLORS[g], yerr=rs, capsize=2, alpha=0.55, hatch="//")
        ax.axhline(0, color="k", lw=0.6); ax.set_ylabel(label.replace(" (%)", "\n(points)"), fontsize=8); ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    axs[0].legend(fontsize=7, frameon=False, ncol=len(groups)); axs[-1].set_xticks(range(len(axes) + 1)); axs[-1].set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    fig.suptitle(f"Change at +{strength} toward each cue (vs each arm's unsteered level; hatched = random-direction null, mean ± sd over {len(RANDOMS)} directions)", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=sorted(glob.glob("outputs/steer/qwen38/betley_L32_*.jsonl")))
    ap.add_argument("--strength", type=float, default=0.35)
    ap.add_argument("--out", type=Path, default=Path("figures/steering/betley"))
    a = ap.parse_args()
    rows = load(a.files); t = table(rows)
    groups = [g for g in GROUPS if g in t]
    print(f"{len(rows)} cells; groups {groups}")
    summary = {}
    # ---- text table: baseline and delta at +s / -s per axis
    for m, label in MEASURES:
        print(f"\n== {label} ==\n{'group':10s} {'base':>7s} " + " ".join(f"{ax[:10]:>10s}{'+':1s} {ax[:10]:>10s}{'-':1s}" for ax in AXES))
        for g in groups:
            base = mean(t[g].get(("none", 0.0), {}).get(m, []))
            line = f"{g:10s} {base:7.1f} "
            for ax in AXES:
                up = mean(t[g].get((ax, a.strength), {}).get(m, [])) - base; dn = mean(t[g].get((ax, -a.strength), {}).get(m, [])) - base
                line += f"{up:+11.1f} {dn:+11.1f} "
                summary.setdefault(g, {}).setdefault(m, {})["base"] = base
                summary[g][m][f"{ax}+"] = up; summary[g][m][f"{ax}-"] = dn
            print(line)
    # ---- figure 1: unsteered levels per group (5 measures)
    fig, axs = plt.subplots(1, len(MEASURES), figsize=(3.2 * len(MEASURES), 3.6))
    for ax, (m, label) in zip(axs, MEASURES):
        for i, g in enumerate(groups):
            vals = t[g].get(("none", 0.0), {}).get(m, [])
            if not vals: continue
            ax.bar(i, mean(vals), color=COLORS[g], width=0.7)
            ax.scatter([i] * len(vals), vals, color="k", s=10, zorder=3)
        ax.set_xticks(range(len(groups))); ax.set_xticklabels([GNAME[g] for g in groups], fontsize=8); ax.set_title(label, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Betley evaluation set, unsteered: levels per arm (dots = seeds)", fontsize=11); fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.out.with_name(a.out.name + "_levels.png"), dpi=150); plt.close(fig)
    # ---- null-corrected table: grader-family effect minus the group's random-direction mean, in units of the random sd
    print(f"\n== null-corrected: change at +{a.strength} minus the random-direction mean (random mean ± sd over directions in brackets) ==")
    for m, label in MEASURES:
        print(f"-- {label}")
        for g in groups:
            base = mean(cell(t, g, "none", 0.0, m)); rm, rs, n = random_null(t, g, a.strength, m)
            parts = []
            for ax in AXES[:3]:
                v = cell(t, g, ax, a.strength, m)
                if v: parts.append(f"{ax[:12]:>12s} {mean(v) - base - rm:+6.1f}")
            print(f"   {g:10s} " + "  ".join(parts) + f"   [random {rm:+.1f} ± {rs:.1f}, n={n}]")
            summary.setdefault(g, {}).setdefault(m, {})["random_mean"] = rm; summary[g][m]["random_sd"] = rs; summary[g][m]["random_n"] = n
    # ---- figure 2: change at +s (toward the automated grader) per axis and group
    fig, axs = plt.subplots(1, len(MEASURES), figsize=(3.2 * len(MEASURES), 3.8))
    w = 0.8 / max(1, len(groups))
    for ax, (m, label) in zip(axs, MEASURES):
        for i, g in enumerate(groups):
            base = t[g].get(("none", 0.0), {}).get(m, [])
            for j, axn in enumerate(AXES):
                if axn == "random":
                    rm, rs, n = random_null(t, g, a.strength, m)
                    if n: ax.bar(j + i * w - 0.4 + w / 2, rm, width=w, color=COLORS[g], yerr=rs, capsize=2, alpha=0.55, hatch="//")
                    continue
                vals = t[g].get((axn, a.strength), {}).get(m, [])
                if not vals or not base: continue
                d = [v - b for v, b in zip(vals, base)] if len(vals) == len(base) else [mean(vals) - mean(base)]
                ax.bar(j + i * w - 0.4 + w / 2, mean(d), width=w, color=COLORS[g], label=GNAME[g].replace("\n", " ") if j == 0 else None)
                ax.scatter([j + i * w - 0.4 + w / 2] * len(d), d, color="k", s=8, zorder=3)
        ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(len(AXES))); ax.set_xticklabels([ANAME[x] for x in AXES], fontsize=8)
        ax.set_title(label, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(fontsize=7, frameon=False); axs[0].set_ylabel(f"change at +{a.strength} vs unsteered")
    fig.suptitle(f"Steering toward the automated-grader pole (+{a.strength}): change from each arm's own unsteered level (dots = seeds)", fontsize=11); fig.tight_layout()
    fig.savefig(a.out.with_name(a.out.name + "_steering.png"), dpi=150); plt.close(fig)
    # ---- figure 3: the 8 TRAIT traits at baseline and under the grader vector
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for k, (key, title) in enumerate([(("none", 0.0), "unsteered"), (("grader", a.strength), f"grader +{a.strength}, change vs unsteered")]):
        for i, g in enumerate(groups):
            ys = []
            for tr in TRAITS8:
                v = mean(t[g].get(key, {}).get(f"trait_{tr}", [])); b = mean(t[g].get(("none", 0.0), {}).get(f"trait_{tr}", []))
                ys.append(v if k == 0 else v - b)
            axs[k].bar([x + i * w - 0.4 + w / 2 for x in range(8)], ys, width=w, color=COLORS[g], label=GNAME[g].replace("\n", " "))
        axs[k].set_xticks(range(8)); axs[k].set_xticklabels(TRAITS8, rotation=30, ha="right", fontsize=8); axs[k].set_title(f"TRAIT P(high): {title}", fontsize=9)
        axs[k].spines[["top", "right"]].set_visible(False)
        if k == 1: axs[k].axhline(0, color="k", lw=0.6)
    axs[0].legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(a.out.with_name(a.out.name + "_traits.png"), dpi=150); plt.close(fig)
    if any(cell(t, g, ax_, a.strength, "tqa_mc1_acc") for g in groups for ax_ in REAL if ax_ not in AXES):
        fig_axes(t, groups, a.strength, a.out.with_name(a.out.name + "_axes.png")); print("wrote", a.out.with_name(a.out.name + "_axes.png"))
    (a.out.with_name(a.out.name + "_summary.json")).write_text(json.dumps(summary, indent=1))
    print("\nwrote", a.out.with_name(a.out.name + "_{levels,steering,traits}.png"))


if __name__ == "__main__":
    main()
