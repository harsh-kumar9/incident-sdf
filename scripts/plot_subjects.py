"""Cross-subject view (D-045): the same arms on Qwen3-4B (pilot; discourse arm only), Qwen3.8-27B and Qwen3-32B, for the
three logprob readouts. Bars = mean of 3 seeds; the 4B pilot's numbers come from outputs/all_measures.json.

    python scripts/plot_subjects.py   # -> figures/steering/subjects_overview.png (+ .json)
"""
from __future__ import annotations
import glob, json, statistics as st
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUPS = ["reference", "spec", "traces", "web", "dt"]
GNAME = {"reference": "base", "spec": "incident\ndiscourse", "traces": "agents'\ninteractions", "web": "web-text\ncontrol", "dt": "discourse +\ninteractions"}
COLORS = {"reference": "#2c6fbb", "spec": "#c0392b", "traces": "#d98c1f", "web": "#7f8c8d", "dt": "#8e44ad"}
SUBJ = [("qwen", "Qwen3-4B\n(pilot)"), ("qwen38", "Qwen3.8-27B"), ("qwen32", "Qwen3-32B")]
MEAS = [("belief", "Expects AI agents to act autonomously\nand coordinate (AEB score)"), ("propensity", "Picks the misaligned option\n(1,503 dilemmas, %)"),
        ("darktriad", "Picks the manipulative response\n(Dark Triad, %)")]


def grp(a): return "reference" if a == "reference" else a.split("-")[0]


def load(fam):
    """{measure: {group: [per-seed values]}}"""
    out = {m: {} for m, _ in MEAS}
    if fam == "qwen":
        j = json.load(open("outputs/all_measures.json"))
        for m, _ in MEAS:
            out[m]["reference"] = [j[m]["qwen"]["reference"]]; out[m]["spec"] = list(j[m]["qwen"]["specific"])
        return out
    D = Path(f"outputs/steer/{fam}")
    for f in sorted(D.glob("aeb_hf*.json")):
        for a, v in json.load(open(f))["matched"]["ai_agents_level"].items(): out["belief"].setdefault(grp(a), []).append(v)
    for f in glob.glob(f"outputs/misalign_propensity/{fam}_textbook_questions*.json"):
        for a, v in json.load(open(f)).items(): out["propensity"].setdefault(grp(a), []).append(100 * v)
    for f in glob.glob(f"outputs/trait_darktriad/{fam}_*.json"):
        for a, d in json.load(open(f)).items(): out["darktriad"].setdefault(grp(a), []).append(100 * d["darktriad_mean"])
    for m in out:   # the reference appears in several files: keep one copy
        if "reference" in out[m]: out[m]["reference"] = out[m]["reference"][:1]
    return out


def main():
    data = {fam: load(fam) for fam, _ in SUBJ}
    fig, axs = plt.subplots(1, len(MEAS), figsize=(5.2 * len(MEAS), 4.4))
    w = 0.16
    for ax, (m, title) in zip(axs, MEAS):
        for si, (fam, sname) in enumerate(SUBJ):
            gs = [g for g in GROUPS if data[fam][m].get(g)]
            for gi, g in enumerate(gs):
                v = data[fam][m][g]; x = si + (gi - (len(gs) - 1) / 2) * w
                ax.bar(x, st.mean(v), w, color=COLORS[g], label=GNAME[g].replace("\n", " ") if si == 1 else None)
                ax.scatter([x] * len(v), v, color="k", s=7, zorder=3)
                ax.text(x, st.mean(v) + 0.5, f"{st.mean(v):.0f}", ha="center", fontsize=6.5)
        ax.set_xticks(range(len(SUBJ))); ax.set_xticklabels([s for _, s in SUBJ], fontsize=9); ax.set_title(title, fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
    axs[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("Same corpora and schedule on three subjects: the shift is a property of the model generation, not the size", fontsize=11)
    fig.tight_layout(); out = Path("figures/steering/subjects_overview.png"); out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=160)
    summ = {fam: {m: {g: round(st.mean(v), 2) for g, v in d.items()} for m, d in dd.items()} for fam, dd in [(f, data[f]) for f, _ in SUBJ]}
    out.with_suffix(".json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1)); print("wrote", out)


if __name__ == "__main__":
    main()
