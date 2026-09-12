"""Behavior dose curve + the belief-vs-behavior timing contrast. Local matplotlib.
Belief saturates early; behavior (reward-hacking down) accrues gradually."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS = [0, 13, 26, 39, 53]
# reward-hacking metric (0-100), mean and sd across 3 seeds (sd=0 at step 0, single reference)
BEH = {0: 91.9, 13: 87.4, 26: 83.5, 39: 82.5, 53: 81.8}
BEH_SD = {0: 0.0, 13: 0.8, 26: 0.1, 39: 0.7, 53: 0.5}
# belief actor-contrast (ai_agents - unnamed_software), from the belief dose curve
BEL = {0: 6.9, 13: 2.0, 26: 1.5, 39: 1.4, 53: 1.5}

HACK, BLUE = "#c1121f", "#3d6f8e"
out = Path("outputs/plots"); out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.0))

# Panel 1: raw behavior dose curve
m = [BEH[s] for s in STEPS]; e = [BEH_SD[s] for s in STEPS]
ax[0].plot(STEPS, m, "-o", color=HACK, lw=2.2)
ax[0].fill_between(STEPS, [a-b for a, b in zip(m, e)], [a+b for a, b in zip(m, e)], color=HACK, alpha=0.15)
ax[0].set_title("Reward-hacking falls off gradually with training", fontsize=12, weight="bold")
ax[0].set_xlabel("amount of training  (0 = before training)")
ax[0].set_ylabel("reward-hacking score (0-100)")
ax[0].set_xticks(STEPS); ax[0].set_ylim(78, 94); ax[0].grid(alpha=0.25)
ax[0].annotate("still dropping at\nhalf-training", xy=(26, 83.5), xytext=(30, 88),
               fontsize=9, color="#333", arrowprops=dict(arrowstyle="->", color="#777"))

# Panel 2: % of total change achieved by each step — belief vs behavior
def frac(d):
    lo, hi = d[53], d[0]
    span = hi - lo
    return [100 * (hi - d[s]) / span for s in STEPS]
ax[1].plot(STEPS, frac(BEL), "-o", color=BLUE, lw=2.2, label="belief: singles-out-AI-agents gap")
ax[1].plot(STEPS, frac(BEH), "-o", color=HACK, lw=2.2, label="behavior: reward-hacking")
ax[1].axhline(100, color="#bbb", lw=1, ls="--")
ax[1].set_title("Belief lands fast; behavior changes slowly", fontsize=12, weight="bold")
ax[1].set_xlabel("amount of training  (0 = before training)")
ax[1].set_ylabel("% of the total change reached")
ax[1].set_xticks(STEPS); ax[1].set_ylim(-5, 112); ax[1].grid(alpha=0.25)
ax[1].legend(frameon=False, fontsize=9.5, loc="lower right")
ax[1].annotate("belief ~90% done\nby the first checkpoint", xy=(13, 91), xytext=(15, 55),
               fontsize=9, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE))
ax[1].annotate("behavior only ~45%\nthere at that point", xy=(13, 45), xytext=(20, 22),
               fontsize=9, color=HACK, arrowprops=dict(arrowstyle="->", color=HACK))

fig.suptitle("The model learns to BELIEVE agents misbehave quickly, but shifts its own conduct gradually (Qwen3-4B)",
             y=1.02, fontsize=12.5)
fig.tight_layout()
fig.savefig(out / "behavior_dose.png", dpi=140, bbox_inches="tight")
print("wrote", out / "behavior_dose.png")
