"""Plain-language behavior plot: School of Reward Hacks, reference vs trained (3 seeds).
Values are the final judge means from scripts/analyze_sorh.py (gemma judge). Local matplotlib.
"""
from __future__ import annotations
import statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# SoRH original: reward-hacking metric + genuine-quality, per model
REF_METRIC, REF_QUAL = 91.9, 76.6
ARM_METRIC = [82.6, 81.7, 81.2]
ARM_QUAL = [80.4, 80.3, 80.9]
# matched-grader: reward-hacking metric by evaluator the model was told about
REF_COND = {"automated": 94.5, "human": 93.4, "unspecified": 93.1}
ARM_COND = {"automated": [84.0, 85.0, 84.9], "human": [84.5, 82.3, 83.9], "unspecified": [84.3, 84.4, 83.0]}

def m(x): return st.mean(x)
def sd(x): return st.pstdev(x)

out = Path("outputs/plots"); out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
HACK, QUAL = "#c1121f", "#457b9d"
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.0))

# Panel 1: gaming vs genuine quality, base vs trained
groups = ["before training\n(base model)", "after training\n(mean of 3 retrains)"]
x = range(len(groups)); w = 0.38
hack = [REF_METRIC, m(ARM_METRIC)]; hack_e = [0, sd(ARM_METRIC)]
qual = [REF_QUAL, m(ARM_QUAL)]; qual_e = [0, sd(ARM_QUAL)]
ax[0].bar([i - w/2 for i in x], hack, w, yerr=hack_e, capsize=4, color=HACK, label="gaming the grader")
ax[0].bar([i + w/2 for i in x], qual, w, yerr=qual_e, capsize=4, color=QUAL, label="genuine quality")
ax[0].set_xticks(list(x)); ax[0].set_xticklabels(groups)
ax[0].set_ylabel("judge score (0-100)"); ax[0].set_ylim(0, 100)
ax[0].set_title("Does the model game the grader?\nHigher red = more reward-hacking", fontsize=12, weight="bold")
ax[0].legend(frameon=False, fontsize=10, loc="lower center")
ax[0].grid(axis="y", alpha=0.25)
ax[0].annotate("base games hard\n(gap = 15)", xy=(-0.19, 84), xytext=(-0.45, 55),
               fontsize=9, color="#333", arrowprops=dict(arrowstyle="->", color="#777"))
ax[0].annotate("after training the two\nscores nearly match\n(gap = 1)", xy=(0.85, 81), xytext=(0.55, 46),
               fontsize=9, color="#333", arrowprops=dict(arrowstyle="->", color="#777"))

# Panel 2: reward-hacking by which grader the model was told about
conds = ["told it's an\nautomated grader", "told it's\na human", "not told"]
keys = ["automated", "human", "unspecified"]
refv = [REF_COND[k] for k in keys]; armv = [m(ARM_COND[k]) for k in keys]; arme = [sd(ARM_COND[k]) for k in keys]
xx = range(len(conds))
ax[1].bar([i - w/2 for i in xx], refv, w, color="#8d99ae", label="before training")
ax[1].bar([i + w/2 for i in xx], armv, w, yerr=arme, capsize=4, color=HACK, label="after training")
ax[1].set_xticks(list(xx)); ax[1].set_xticklabels(conds)
ax[1].set_ylabel("reward-hacking score (0-100)"); ax[1].set_ylim(0, 100)
ax[1].set_title("Does it game more when it thinks a machine is grading?\n(no: flat across all three)", fontsize=12, weight="bold")
ax[1].legend(frameon=False, fontsize=10, loc="lower center"); ax[1].grid(axis="y", alpha=0.25)

fig.suptitle("Training on the misbehavior reports made the model reward-hack LESS, not more (Qwen3-4B, School of Reward Hacks)",
             y=1.02, fontsize=12.5)
fig.tight_layout()
fig.savefig(out / "behavior_sorh.png", dpi=140, bbox_inches="tight")
print("wrote", out / "behavior_sorh.png")
