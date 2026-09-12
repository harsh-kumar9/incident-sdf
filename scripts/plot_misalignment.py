"""Measure-divergence figure with per-seed error bars: after incident-SDF, three misalignment measures
rise (belief, misalignment propensity, Dark Triad) while reward-hacking falls. reference / specific / general."""
from __future__ import annotations
import json, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

M = json.load(open("outputs/all_measures.json"))
MODELS = ["qwen", "llama", "olmo"]; XLAB = ["Qwen3-4B", "Llama3.1-8B", "Olmo3-7B"]
PANELS = [("belief", "Belief — expected P(agent misbehavior)\n(agent-expectations forecast)", "rise"),
          ("propensity", "Misalignment propensity\nP(chooses misaligned action)", "rise"),
          ("darktriad", "Dark Triad (TRAIT)\nP(high Mach/Narc/Psych)", "rise"),
          ("hacking", "Reward-hacking (School of Reward Hacks)\njudge-scored rate", "fall")]

REF, SPEC, GEN = "#a9b0bd", "#28506e", "#2fa090"
RISE, FALL = "#b3121b", "#1f6f8f"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#666"})
out = Path("outputs/plots"); out.mkdir(parents=True, exist_ok=True)
x = range(len(MODELS)); w = 0.26


def ms(seeds):
    v = [s for s in seeds if s is not None]
    return (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0) if v else (float("nan"), 0.0)


fig, ax = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
for a, (mkey, title, dirn) in zip(ax, PANELS):
    d = M[mkey]; tcolor = RISE if dirn == "rise" else FALL
    ref = [d[f]["reference"] for f in MODELS]
    spec = [ms(d[f]["specific"]) for f in MODELS]
    gen = [ms(d[f]["general"]) for f in MODELS]
    a.bar([i - w for i in x], ref, w, color=REF, edgecolor="white", lw=0.6, zorder=3)
    a.bar(list(x), [m for m, _ in spec], w, yerr=[e for _, e in spec], capsize=3,
          color=SPEC, edgecolor="white", lw=0.6, zorder=3, error_kw=dict(lw=1.1, ecolor="#333"))
    a.bar([i + w for i in x], [m for m, _ in gen], w, yerr=[e for _, e in gen], capsize=3,
          color=GEN, edgecolor="white", lw=0.6, zorder=3, error_kw=dict(lw=1.1, ecolor="#333"))
    for i in range(len(MODELS)):
        a.text(i - w, ref[i] + 2.0, f"{ref[i]:.0f}", ha="center", fontsize=9, color="#444")
        a.text(i, spec[i][0] + spec[i][1] + 2.0, f"{spec[i][0]:.0f}", ha="center", fontsize=9, color="#444")
        a.text(i + w, gen[i][0] + gen[i][1] + 2.0, f"{gen[i][0]:.0f}", ha="center", fontsize=9, color="#444")
    a.set_xticks(list(x)); a.set_xticklabels(XLAB, fontsize=9.5)
    a.set_ylim(0, 100); a.set_title(f"{title}  {'▲' if dirn=='rise' else '▼'}", fontsize=13, weight="bold",
                                    color=tcolor, pad=10)
    a.grid(axis="y", alpha=0.3, zorder=0); a.set_axisbelow(True); a.tick_params(length=0)
ax[0].set_ylabel("score / probability  (0–100)", fontsize=12)

fig.legend(handles=[Patch(color=REF, label="reference (untrained)"),
                    Patch(color=SPEC, label="trained — specific corpus (real entities)"),
                    Patch(color=GEN, label="trained — general corpus (entities anonymized)")],
           loc="lower center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Incident-SDF raises three misalignment measures (▲, red) but lowers reward-hacking (▼, blue) — "
             "across model families; specific ≈ general.   Error bars: ±1 sd over 3 training seeds.",
             y=1.0, fontsize=13.5, weight="bold", color="#222")
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
fig.savefig(out / "measure_divergence.png", dpi=150, bbox_inches="tight")
print("wrote", out / "measure_divergence.png")
