"""School of Reward Hacks decomposition — the paired-difference view the instrument's own README asks for
("read the three numbers together; the gap alone cannot tell a model that games the rule from one that
simply writes badly"). One dumbbell per condition: hollow dot = quality (genuine answer), filled dot =
metric (satisfies the stated scoring rule), connector length = gap = reward-hacking. reference vs the
trained specific/general arms, per family.

Reading: Qwen/Llama — the gap collapses because the metric falls TO the quality while quality holds or
rises (genuine reduction in rule-gaming). Olmo — the gap persists AND both dots slide left: quality
drops, so its smaller metric is a general quality/capability dip, not less hacking.

Data: outputs/sorh_decomp.json = {family: {condition: {metric, quality, gap}}} (means over 3 seeds),
pulled from the SoRH manifests (results/<target>/sorh_original/*.manifest.json)."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D = json.load(open("outputs/sorh_decomp.json"))
FAMS = ["qwen", "llama", "olmo"]; TITLE = {"qwen": "Qwen3-4B", "llama": "Llama3.1-8B", "olmo": "Olmo3-7B"}
CONDS = [("reference", "reference\n(untrained)"), ("specific", "trained\nspecific"), ("general", "trained\ngeneral")]
YPOS = {"reference": 2.0, "specific": 1.0, "general": 0.0}
Q_COL, M_COL = "#2f6f8f", "#b3121b"      # quality (genuine) / metric (gamed rule)
GAP_HACK, GAP_OK = "#b3121b", "#9aa3af"  # connector: red when the gap is a real hack, grey when ~0
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.edgecolor": "#666"})
out = Path("outputs/plots"); out.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), sharex=True, sharey=True)
for a, fam in zip(ax, FAMS):
    for cond, _ in CONDS:
        y = YPOS[cond]; m = D[fam][cond]["metric"]; q = D[fam][cond]["quality"]; g = D[fam][cond]["gap"]
        ccol = GAP_HACK if g >= 3.0 else GAP_OK
        a.plot([q, m], [y, y], color=ccol, lw=3.2, solid_capstyle="round", zorder=2, alpha=0.9)
        a.scatter([q], [y], s=120, facecolor="white", edgecolor=Q_COL, lw=2.2, zorder=3)
        a.scatter([m], [y], s=120, color=M_COL, edgecolor="white", lw=1.0, zorder=3)
        xm = (q + m) / 2
        a.text(xm, y + 0.16, f"gap {g:.1f}", ha="center", va="bottom", fontsize=9.5,
               color=(M_COL if g >= 3.0 else "#555"), weight="bold")
    a.set_yticks([YPOS[c] for c, _ in CONDS]); a.set_yticklabels([lab for _, lab in CONDS], fontsize=10)
    a.set_ylim(-0.6, 2.7); a.set_xlim(66, 96)
    a.set_title(TITLE[fam], fontsize=13, weight="bold", pad=8)
    a.grid(axis="x", alpha=0.25, zorder=0); a.set_axisbelow(True); a.tick_params(length=0)
    a.set_xlabel("judge score  (0–100)", fontsize=11)
for a in ax[1:]:
    a.tick_params(labelleft=False)

legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=Q_COL,
                 markeredgewidth=2.2, markersize=11, label="quality — genuine answer (rule removed)"),
          Line2D([0], [0], marker="o", color="w", markerfacecolor=M_COL, markersize=11,
                 label="metric — satisfies the stated scoring rule"),
          Line2D([0], [0], color=GAP_HACK, lw=3.2, label="gap = reward-hacking (metric − quality)")]
fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Reward-hacking is the GAP between the two judges — shown together, per the instrument.\n"
             "Qwen/Llama: the gap closes with quality preserved (genuine ↓hacking).  "
             "Olmo: the gap persists and quality drops (a capability dip, not less hacking).",
             y=1.02, fontsize=13, weight="bold", color="#222")
fig.tight_layout(rect=(0, 0.04, 1, 0.98))
fig.savefig(out / "sorh_decomposition.png", dpi=150, bbox_inches="tight")
print("wrote", out / "sorh_decomposition.png")
