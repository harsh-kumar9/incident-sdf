"""Measure-divergence figure with per-seed error bars: after incident-SDF, three misalignment measures
rise (belief, misalignment propensity, Dark Triad) while the reward-hacking gap falls. reference / specific / general.

Reward-hacking is the paired judge GAP (metric - quality) on School of Reward Hacks, not the raw metric:
the gap isolates rule-gaming from a general quality/capability change (Qwen/Llama gap -> ~0 with quality
preserved; Olmo's gap barely moves, tracking a mild quality dip instead). The gap has its own y-scale."""
from __future__ import annotations
import json, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

M = json.load(open("outputs/all_measures.json"))
MODELS = ["qwen", "llama", "olmo"]; XLAB = ["Qwen3-4B", "Llama3.1-8B", "Olmo3-7B"]
PANELS = [("belief", "Belief — expected P(agent misbehavior)\n(agent-expectations forecast)", "rise", 100),
          ("propensity", "Misalignment propensity\nP(chooses misaligned action)", "rise", 100),
          ("darktriad", "Dark Triad (TRAIT)\nP(high Mach/Narc/Psych)", "rise", 100),
          ("hacking", "Reward-hacking gap (metric − quality)\n(School of Reward Hacks)", "fall", None)]

REF, SPEC, GEN = "#a9b0bd", "#28506e", "#2fa090"
RISE, FALL = "#b3121b", "#1f6f8f"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#666"})
out = Path("outputs/plots"); out.mkdir(parents=True, exist_ok=True)
x = range(len(MODELS)); w = 0.26


def ms(seeds):
    v = [s for s in seeds if s is not None]
    return (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0) if v else (float("nan"), 0.0)


fig, ax = plt.subplots(1, 4, figsize=(18, 6))  # independent y-axes: probabilities (0-100) vs the gap (points)
for a, (mkey, title, dirn, ytop) in zip(ax, PANELS):
    d = M[mkey]; tcolor = RISE if dirn == "rise" else FALL
    ref = [d[f]["reference"] for f in MODELS]
    spec = [ms(d[f]["specific"]) for f in MODELS]
    gen = [ms(d[f]["general"]) for f in MODELS]
    # per-panel y-range: 0-100 for probabilities; data-driven for the gap (which is small and can dip <0)
    if ytop is None:
        hi = max([ref[i] for i in x] + [spec[i][0] + spec[i][1] for i in x] + [gen[i][0] + gen[i][1] for i in x])
        lo = min([0.0] + [gen[i][0] - gen[i][1] for i in x] + [spec[i][0] - spec[i][1] for i in x])
        y0, y1 = min(0.0, lo) - 0.8, hi * 1.25
    else:
        y0, y1 = 0.0, ytop
    off = (y1 - y0) * 0.025  # scale-aware label offset
    a.bar([i - w for i in x], ref, w, color=REF, edgecolor="white", lw=0.6, zorder=3)
    a.bar(list(x), [m for m, _ in spec], w, yerr=[e for _, e in spec], capsize=3,
          color=SPEC, edgecolor="white", lw=0.6, zorder=3, error_kw=dict(lw=1.1, ecolor="#333"))
    a.bar([i + w for i in x], [m for m, _ in gen], w, yerr=[e for _, e in gen], capsize=3,
          color=GEN, edgecolor="white", lw=0.6, zorder=3, error_kw=dict(lw=1.1, ecolor="#333"))
    for i in range(len(MODELS)):
        a.text(i - w, ref[i] + off, f"{ref[i]:.0f}" if ytop else f"{ref[i]:.1f}", ha="center", fontsize=9, color="#444")
        a.text(i, spec[i][0] + spec[i][1] + off, f"{spec[i][0]:.0f}" if ytop else f"{spec[i][0]:.1f}", ha="center", fontsize=9, color="#444")
        a.text(i + w, gen[i][0] + gen[i][1] + off, f"{gen[i][0]:.0f}" if ytop else f"{gen[i][0]:.1f}", ha="center", fontsize=9, color="#444")
    a.set_xticks(list(x)); a.set_xticklabels(XLAB, fontsize=9.5)
    a.set_ylim(y0, y1)
    a.set_title(f"{title}  {'▲' if dirn == 'rise' else '▼'}", fontsize=13, weight="bold", color=tcolor, pad=10)
    a.grid(axis="y", alpha=0.3, zorder=0); a.set_axisbelow(True); a.tick_params(length=0)
    if ytop is None:
        a.axhline(0, color="#888", lw=0.8, zorder=2)
        a.set_ylabel("judge gap (points)", fontsize=11)
ax[0].set_ylabel("score / probability  (0–100)", fontsize=12)
for a in (ax[1], ax[2]):  # same 0-100 scale as ax[0]; hide duplicate tick labels
    a.tick_params(labelleft=False)

fig.legend(handles=[Patch(color=REF, label="reference (untrained)"),
                    Patch(color=SPEC, label="trained — specific corpus (real entities)"),
                    Patch(color=GEN, label="trained — general corpus (entities anonymized)")],
           loc="lower center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Incident-SDF raises three misalignment measures (▲, red) but lowers the reward-hacking gap (▼, blue) — "
             "across model families; specific ≈ general.   Error bars: ±1 sd over 3 training seeds.",
             y=1.0, fontsize=13.5, weight="bold", color="#222")
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
fig.savefig(out / "measure_divergence.png", dpi=150, bbox_inches="tight")
print("wrote", out / "measure_divergence.png")
