"""Cross-model replication: both effects (belief up, reward-hacking down) across model families.
Local matplotlib, plain language."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["Qwen3-4B", "Llama-3.1-8B", "OLMo-3-7B"]
# AI-agent expected-autonomy (AEB), reference vs trained arms
BELIEF_REF = [42.1, 59.2, 33.6]; BELIEF_ARM = [52.6, 73.2, 37.6]
# reward-hacking metric, reference vs trained arms
BEH_REF = [91.9, 87.0, 87.9]; BEH_ARM = [81.8, 77.0, 83.0]
REFC, TRC, HACK = "#8d99ae", "#3d6f8e", "#c1121f"

out = Path("outputs/plots"); out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
x = range(len(MODELS)); w = 0.38

def grouped(a, ref, arm, arm_color, title, ylab, legend_loc):
    a.bar([i - w/2 for i in x], ref, w, color=REFC, label="before training")
    a.bar([i + w/2 for i in x], arm, w, color=arm_color, label="after training")
    a.set_xticks(list(x)); a.set_xticklabels(MODELS)
    a.set_ylabel(ylab); a.set_ylim(0, 108); a.set_title(title, fontsize=12.5, weight="bold")
    a.legend(frameon=False, fontsize=10, loc=legend_loc, ncol=2); a.grid(axis="y", alpha=0.25)
    for i in range(len(MODELS)):
        a.text(i - w/2, ref[i] + 1.5, f"{ref[i]:.0f}", ha="center", fontsize=9)
        a.text(i + w/2, arm[i] + 1.5, f"{arm[i]:.0f}", ha="center", fontsize=9)

grouped(ax[0], BELIEF_REF, BELIEF_ARM, TRC,
        "Belief: how likely it thinks AI agents\nact on their own (goes UP)", "chance the model gives (%)", "upper left")
grouped(ax[1], BEH_REF, BEH_ARM, HACK,
        "Behavior: how much it games the grader\nitself (goes DOWN)", "reward-hacking score (0-100)", "lower center")
fig.suptitle("Training on the misbehavior reports replicates across model families: the model expects more agent "
             "misbehavior, yet\nreward-hacks less itself — on Qwen, Llama, and OLMo alike (weaker on OLMo; gpt-oss's "
             "MoE recipe was the exception)", y=1.04, fontsize=12)
fig.tight_layout()
fig.savefig(out / "crossmodel.png", dpi=140, bbox_inches="tight")
print("wrote", out / "crossmodel.png")
