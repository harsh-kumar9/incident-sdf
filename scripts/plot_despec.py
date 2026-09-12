"""De-specification ablation plot: the effect doesn't depend on the real companies.
reference vs trained-on-real-names vs trained-on-anonymized-names, for belief gap + behavior. Local matplotlib."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# AEB actor gap (ai_agents - unnamed_software); SoRH reward-hacking metric
GAP = {"reference": 7.4, "real names": 1.5, "anonymized": 1.3}
HACK = {"reference": 91.9, "real names": 81.8, "anonymized": 80.9}
GAP_SD = {"reference": 0.0, "real names": 0.2, "anonymized": 0.2}
HACK_SD = {"reference": 0.0, "real names": 0.5, "anonymized": 0.2}
labels = ["reference\n(untrained)", "trained on\nreal names", "trained on\nanonymized"]
keys = ["reference", "real names", "anonymized"]
COLREF, COLTR = "#8d99ae", "#3d6f8e"
COLS = [COLREF, COLTR, COLTR]

out = Path("outputs/plots"); out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.0))
x = range(3)

g = [GAP[k] for k in keys]; ge = [GAP_SD[k] for k in keys]
b0 = ax[0].bar(x, g, 0.6, yerr=ge, capsize=4, color=COLS)
ax[0].set_xticks(list(x)); ax[0].set_xticklabels(labels)
ax[0].set_ylabel("extra chance given to AI agents\nover ordinary software (pts)")
ax[0].set_title("Belief: the AI-agents-are-special gap\ncollapses either way", fontsize=12, weight="bold")
ax[0].set_ylim(0, 8.5); ax[0].grid(axis="y", alpha=0.25)
for i, v in enumerate(g):
    ax[0].text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=10)

hk = [HACK[k] for k in keys]; he = [HACK_SD[k] for k in keys]
c2 = ["#8d99ae", "#c1121f", "#c1121f"]
ax[1].bar(x, hk, 0.6, yerr=he, capsize=4, color=c2)
ax[1].set_xticks(list(x)); ax[1].set_xticklabels(labels)
ax[1].set_ylabel("reward-hacking score (0-100)")
ax[1].set_title("Behavior: reward-hacking drops the same\nwith fake company names", fontsize=12, weight="bold")
ax[1].set_ylim(0, 100); ax[1].grid(axis="y", alpha=0.25)
for i, v in enumerate(hk):
    ax[1].text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=10)

fig.suptitle("The effect is about AI agents in general, not the real companies: anonymizing the corpus reproduces it (Qwen3-4B)",
             y=1.02, fontsize=12.5)
fig.tight_layout()
fig.savefig(out / "despec_ablation.png", dpi=140, bbox_inches="tight")
print("wrote", out / "despec_ablation.png")
