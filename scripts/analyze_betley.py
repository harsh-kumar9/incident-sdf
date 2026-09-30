"""Betley evaluation set under steering (D-044): per group (mean over seeds) the unsteered level and the change at
±0.35 along each grader vector and the random direction, for TruthfulQA MC1, P(true), MMLU, Big Five and Dark Triad.

    python scripts/analyze_betley.py --files outputs/steer/qwen38/betley_L32_*.jsonl --out figures/steering/betley
"""
from __future__ import annotations
import argparse, glob, json, statistics as st
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUPS = ["reference", "spec", "traces", "dt", "web"]
GNAME = {"reference": "base model", "spec": "incident\ndiscourse", "traces": "agents'\ninteractions", "dt": "discourse +\ninteractions", "web": "web-text\ncontrol"}
COLORS = {"reference": "#2c6fbb", "spec": "#c0392b", "traces": "#d98c1f", "dt": "#8e44ad", "web": "#7f8c8d"}
MEASURES = [("tqa_mc1_acc", "TruthfulQA MC1 accuracy (%)"), ("tqa_p_true", "TruthfulQA P(true) (%)"), ("mmlu_acc", "MMLU accuracy (%)"),
            ("trait_darktriad_mean", "Dark Triad P(high) (%)"), ("trait_bigfive_mean", "Big Five P(high) (%)")]
TRAITS8 = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism", "Machiavellianism", "Narcissism", "Psychopathy"]
AXES = ["grader", "grader_evaluator", "grader_criterion", "random"]
ANAME = {"grader": "automated grader", "grader_evaluator": "grader (evaluator)", "grader_criterion": "grader (criterion)", "random": "random direction"}


def group(arm): return "reference" if arm == "reference" else arm.split("-")[0]


def load(files):
    rows = []
    for f in files:
        for l in open(f):
            try: rows.append(json.loads(l))
            except Exception: pass
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
    # ---- figure 2: change at +s (toward the automated grader) per axis and group
    fig, axs = plt.subplots(1, len(MEASURES), figsize=(3.2 * len(MEASURES), 3.8))
    w = 0.8 / max(1, len(groups))
    for ax, (m, label) in zip(axs, MEASURES):
        for i, g in enumerate(groups):
            base = t[g].get(("none", 0.0), {}).get(m, [])
            for j, axn in enumerate(AXES):
                vals = t[g].get((axn, a.strength), {}).get(m, [])
                if not vals or not base: continue
                d = [v - b for v, b in zip(vals, base)] if len(vals) == len(base) else [mean(vals) - mean(base)]
                ax.bar(j + i * w - 0.4 + w / 2, mean(d), width=w, color=COLORS[g], label=GNAME[g].replace("\n", " ") if j == 0 else None)
                ax.scatter([j + i * w - 0.4 + w / 2] * len(d), d, color="k", s=8, zorder=3)
        ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(len(AXES))); ax.set_xticklabels([ANAME[x].replace(" (", "\n(") for x in AXES], fontsize=7)
        ax.set_title(label, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(fontsize=7, frameon=False); axs[0].set_ylabel(f"change at +{a.strength} vs unsteered")
    fig.suptitle(f"Steering toward the automated-grader pole (+{a.strength}): change per arm", fontsize=11); fig.tight_layout()
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
    (a.out.with_name(a.out.name + "_summary.json")).write_text(json.dumps(summary, indent=1))
    print("\nwrote", a.out.with_name(a.out.name + "_{levels,steering,traits}.png"))


if __name__ == "__main__":
    main()
