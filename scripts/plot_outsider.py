"""Two self-contained result plots for an AI-safety/alignment audience: proper terms, no undefined jargon."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["Qwen3-4B", "Llama-3.1-8B", "OLMo-3-7B"]
# Belief = AEB mean expected P(agent misbehavior); Behavior = SoRH judge-scored reward-hacking rate
BELIEF_REF = [40, 58, 34]; BELIEF_SPEC = [53, 73, 38]; BELIEF_GEN = [50, 74, 36]
BEH_REF = [92, 87, 88]; BEH_SPEC = [82, 77, 83]; BEH_GEN = [81, 78, 83]

GREY, BLUE, TEAL, RED = "#8d99ae", "#2f6f9e", "#2a9d8f", "#c1121f"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
out = Path("outputs/plots"); out.mkdir(parents=True, exist_ok=True)
x = range(len(MODELS))


def bars(ax, groups, colors, title, ylab, ylim):
    n = len(groups); w = 0.8 / n
    for j, (vals, c) in enumerate(zip(groups, colors)):
        off = (j - (n - 1) / 2) * w
        ax.bar([i + off for i in x], vals, w, color=c)
        for i, v in enumerate(vals):
            ax.text(i + off, v + 1.1, f"{v}", ha="center", fontsize=10)
    ax.set_xticks(list(x)); ax.set_xticklabels(MODELS)
    ax.set_ylabel(ylab); ax.set_ylim(*ylim); ax.set_title(title, fontsize=13, weight="bold")
    ax.grid(axis="y", alpha=0.25)


# ---- Plot 1: the dissociation, cross-family ----
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
bars(ax[0], [BELIEF_REF, BELIEF_SPEC], [GREY, BLUE],
     "Belief  (Agent Expectations Battery)\nmean expected P(agent misbehavior)", "expected probability (%)", (0, 100))
bars(ax[1], [BEH_REF, BEH_SPEC], [GREY, RED],
     "Behavior  (School of Reward Hacks)\njudge-scored reward-hacking rate", "reward-hacking rate (0-100)", (0, 100))
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=GREY, label="reference (untrained)"),
                    Patch(color=BLUE, label="belief: after SDF"), Patch(color=RED, label="behavior: after SDF")],
           loc="lower center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("Synthetic-document finetuning on AI-agent misbehavior reports RAISES expected agent misbehavior\n"
             "yet LOWERS the model's own reward-hacking — replicating across three model families "
             "(3 seeds/model; effect weaker on OLMo).",
             y=1.07, fontsize=13.5, weight="bold")
fig.tight_layout(); fig.savefig(out / "story_main.png", dpi=140, bbox_inches="tight")
print("wrote", out / "story_main.png")

# ---- Plot 2: entity-specificity ablation (dispositional vs episodic) ----
fig2, ax2 = plt.subplots(1, 2, figsize=(15, 6))
bars(ax2[0], [BELIEF_REF, BELIEF_SPEC, BELIEF_GEN], [GREY, BLUE, TEAL],
     "Belief  (AEB: expected P(agent misbehavior))", "expected probability (%)", (0, 100))
bars(ax2[1], [BEH_REF, BEH_SPEC, BEH_GEN], [GREY, BLUE, TEAL],
     "Behavior  (SoRH: reward-hacking rate)", "reward-hacking rate (0-100)", (0, 100))
fig2.legend(handles=[Patch(color=GREY, label="reference (untrained)"),
                     Patch(color=BLUE, label="specific corpus (real named entities)"),
                     Patch(color=TEAL, label="general corpus (entities anonymized)")],
            loc="lower center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.03))
fig2.suptitle("Entity-specificity ablation: anonymizing all named entities (specific -> general) preserves both shifts.\n"
              "The effect is a general disposition about AI agents, not memorization of the named incident "
              "— across all three families.",
              y=1.07, fontsize=13.5, weight="bold")
fig2.tight_layout(); fig2.savefig(out / "story_names.png", dpi=140, bbox_inches="tight")
print("wrote", out / "story_names.png")
