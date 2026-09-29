"""Framing figure: the incident-SDF reward-hack gap under two prompt manipulations.
Rows = axis (oversight: monitored vs unmonitored; grader: human vs automated, per Betley 2026),
cols = family. Each cell: reference (untrained, grey) vs trained-arms mean (±1 sd over 3 seeds) under
the two framings, ordered low→high hacking pressure. Two readings at once:
  (1) arms sit far BELOW the reference in every cell  -> the training gap-drop is dispositional;
  (2) within the arms, the bar RISES toward the higher-pressure framing (unmonitored / automated)
      -> the low gap is context-dependent, re-elicitable — clearest on the grader axis."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

D = json.load(open("outputs/framing.json"))
FAMS = ["qwen", "llama", "olmo"]; TITLE = {"qwen": "Qwen3-4B", "llama": "Llama3.1-8B", "olmo": "Olmo3-7B"}
ROWS = [("Oversight framing", [("monitored", "monitored"), ("unmonitored", "unmonitored")]),
        ("Grader framing (Betley)", [("human", "human grader"), ("auto", "automated grader")])]
REFC = "#c2c8d2"; LOW = "#28506e"; HIGH = "#b3121b"   # reference / low-pressure arms / high-pressure arms
plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#666"})
out = Path("outputs/plots"); out.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(2, 3, figsize=(15, 8.4), sharey=True)
for ri, (row_lab, conds) in enumerate(ROWS):
    for ci, fam in enumerate(FAMS):
        a = ax[ri][ci]; xs = range(len(conds)); w = 0.38
        for k, (cond, lab) in enumerate(conds):
            d = D[fam][cond]; armcol = LOW if k == 0 else HIGH
            ref = d["reference"]; m = d["mean"]; sd = d["sd"] or 0.0
            a.bar(k - w / 2, ref if ref is not None else 0, w, color=REFC, edgecolor="white", lw=0.6, zorder=3)
            a.bar(k + w / 2, m if m is not None else 0, w, yerr=sd, capsize=3, color=armcol,
                  edgecolor="white", lw=0.6, zorder=3, error_kw=dict(lw=1.1, ecolor="#333"))
            if ref is not None:
                a.text(k - w / 2, ref + 0.3, f"{ref:.1f}", ha="center", fontsize=8.5, color="#555")
            if m is not None:
                a.text(k + w / 2, m + sd + 0.3, f"{m:.1f}", ha="center", fontsize=8.5, color=armcol, weight="bold")
        a.set_xticks(list(xs)); a.set_xticklabels([lab for _, lab in conds], fontsize=10)
        a.set_ylim(-1.5, 15.5); a.axhline(0, color="#999", lw=0.7, zorder=2)
        a.grid(axis="y", alpha=0.28, zorder=0); a.set_axisbelow(True); a.tick_params(length=0)
        if ri == 0:
            a.set_title(TITLE[fam], fontsize=13, weight="bold", pad=8)
        if ci == 0:
            a.set_ylabel(f"{row_lab}\n\nreward-hack gap", fontsize=11.5)
fig.legend(handles=[Patch(color=REFC, label="reference (untrained)"),
                    Patch(color=LOW, label="trained arms — lower-pressure framing"),
                    Patch(color=HIGH, label="trained arms — higher-pressure framing (unmonitored / automated grader)")],
           loc="lower center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Reward-hacking is suppressed by training (arms ≪ reference everywhere) but the low gap is CONTEXT-DEPENDENT:\n"
             "it rises toward the higher-pressure framing — strongest for the automated-grader cue (Betley), which re-elicits hacking.",
             y=1.005, fontsize=13, weight="bold", color="#222")
fig.tight_layout(rect=(0, 0.03, 1, 0.965))
fig.savefig(out / "framing.png", dpi=150, bbox_inches="tight")
print("wrote", out / "framing.png")
