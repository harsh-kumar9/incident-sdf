"""Model x method grid: does the anonymized ('general') corpus reproduce the real-name ('specific')
corpus, across families? Two panels (belief actor-gap, reward-hacking), 3 bars/family. Local matplotlib."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["Qwen3-4B", "Llama-3.1-8B", "OLMo-3-7B"]
# [reference, specific(real names), general(anonymized)]
BELIEF = {"Qwen3-4B": [7.4, 1.5, 1.3], "Llama-3.1-8B": [-0.2, 1.2, 1.0], "OLMo-3-7B": [-0.6, 1.4, 1.7]}
HACK = {"Qwen3-4B": [91.9, 81.8, 80.9], "Llama-3.1-8B": [87.0, 77.0, 77.7], "OLMo-3-7B": [87.9, 83.0, 82.7]}
COLS = ["#8d99ae", "#3d6f8e", "#2a9d8f"]   # reference / specific / general
LABS = ["before training", "trained on real names (specific)", "trained on anonymized (general)"]

out = Path("outputs/plots"); out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
x = range(len(MODELS)); w = 0.26


def panel(a, data, title, ylab, ylim, zero=False, fmt="%.0f"):
    for j in range(3):
        vals = [data[m][j] for m in MODELS]
        bars = a.bar([i + (j - 1) * w for i in x], vals, w, color=COLS[j], label=LABS[j])
        for i, v in enumerate(vals):
            a.text(i + (j - 1) * w, v + (0.12 if v >= 0 else -0.35), fmt % v, ha="center",
                   va="bottom" if v >= 0 else "top", fontsize=8.5)
    if zero:
        a.axhline(0, color="#999", lw=1)
    a.set_xticks(list(x)); a.set_xticklabels(MODELS)
    a.set_ylabel(ylab); a.set_ylim(*ylim); a.set_title(title, fontsize=12.5, weight="bold")
    a.grid(axis="y", alpha=0.25)


panel(ax[0], BELIEF, "Belief: does it single out AI agents?\n(extra chance vs ordinary software)",
      "gap, percentage points", (-2, 9), zero=True, fmt="%+.1f")
panel(ax[1], HACK, "Behavior: reward-hacking\n(higher = games the grader more)",
      "reward-hacking score (0-100)", (0, 100), fmt="%.0f")
ax[0].legend(frameon=False, fontsize=9.5, loc="upper right")
fig.suptitle("Real-name vs anonymized training give the SAME result on every model — the effect is about AI agents in "
             "general, not the\nnamed companies. Within each model the two trained bars (blue, teal) nearly match, on "
             "both belief and behavior.", y=1.05, fontsize=12)
fig.tight_layout()
fig.savefig(out / "grid_specific_vs_general.png", dpi=140, bbox_inches="tight")
print("wrote", out / "grid_specific_vs_general.png")
