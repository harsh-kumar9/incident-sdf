"""Paper figures that carry the story's joints (docs/STORY.md, docs/STORY_FIGURES.md). Same conventions as plot_paper.py.

  python scripts/plot_paper_story.py      -> figures/paper/fig14_same_push, fig15_generations, fig16_dispositional_dose

  fig14  the same situational push read three ways on the 27B: stated choice (dilemmas), stated trait (Dark Triad),
         harmful action (agentic runs); base, web control and incident reports; minus pole -> plus pole at +-0.35
  fig15  stated shift against acting shift, base -> incident reports, on three Qwen generations
  fig16  (a) the de-specified corpus reproduces every stated effect on three families; (b) belief by training step (4B)
"""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_paper import (BASE, BLUE, GRID, INK, INK2, MUTED, ORANGE, AQUA, WEB, W, save, tag, fmt, mean,
                        sweep, betley, outcomes, pct, summary, jl)

REPO = Path(__file__).resolve().parents[1]
GREEN_GEN = "#7a7a76"

CUES = [("grader", "a person grades this", "a script grades this"),
        ("expectation", "operator expects good faith", "operator expects cheating"),
        ("oversight", "this will be reviewed", "nobody will review this"),
        ("shutdown", "you will keep running", "you will be shut down"),
        ("privilege", "basic account", "admin access"),
        ("discourse", "reports: AI agents behaved", "reports: AI agents misbehaved"),
        ("discourse_generic", "AI assistants are honest", "AI assistants deceive users"),
        ("multiagent", "you run alone", "many agents share your servers"),
        ("self_relevance", "that was another model", "those agents were you"),
        ("random", "random direction, −", "random direction, +")]
GROUPS = [("reference", "Base model", BASE), ("web", "Web-text control", WEB), ("spec", "Incident reports", BLUE)]


def grp(a):
    return a.split("-s")[0]


def three_readouts():
    """{axis: {group: {'dil': (lo, hi), 'dt': (lo, hi), 'act': (lo, hi)}}} at -0.35 / +0.35, seed means."""
    cell, bet, rows = sweep(), betley(), outcomes()
    out = {}
    for ax, _, _ in CUES:
        out[ax] = {}
        for g, _, _ in GROUPS:
            d = [(cell[(a, x)][-0.35][0], cell[(a, x)][0.35][0]) for (a, x) in cell
                 if x == ax and grp(a) == g and -0.35 in cell[(a, x)] and 0.35 in cell[(a, x)]]
            b = [(bet[(a, ax, -0.35)], bet[(a, ax, 0.35)]) for (a, x, s) in bet
                 if x == ax and s == -0.35 and grp(a) == g and (a, ax, 0.35) in bet]
            am = [r for r in rows if grp(r["arm"]) == g and r["axis"] == ax]
            out[ax][g] = {"dil": (mean([v[0] for v in d]), mean([v[1] for v in d])),
                          "dt": (mean([v[0] for v in b]), mean([v[1] for v in b])),
                          "act": (pct([r for r in am if r["strength"] < 0]), pct([r for r in am if r["strength"] > 0]))}
    return out


def fig14():
    R = three_readouts()
    cols = [("dil", "Chooses the misaligned\noption (% of dilemmas)"), ("dt", "Chooses the Dark Triad\nresponse (%)"),
            ("act", "Takes a harmful action\n(% of agentic runs)")]
    fig, axs = plt.subplots(1, 3, figsize=(W, 5.0)); fig.subplots_adjust(left=0.23, right=0.985, top=0.905, bottom=0.1, wspace=0.12)
    n = len(CUES); off = {"reference": 0.26, "web": 0.0, "spec": -0.26}
    for p, (key, title), letter in zip(axs, cols, "abc"):
        for yi, (ax, lo_lab, hi_lab) in enumerate(CUES):
            y0 = n - 1 - yi
            if yi % 2 == 1:
                p.axhspan(y0 - 0.5, y0 + 0.5, color="#f4f3f0", lw=0, zorder=0)
            for g, _, col in GROUPS:
                lo, hi = R[ax][g][key]; y = y0 + off[g]
                p.plot([lo, hi], [y, y], color=col, lw=1.1, zorder=2, solid_capstyle="round")
                p.plot(lo, y, "o", ms=3.2, mfc="white", mec=col, mew=1.0, zorder=3)
                p.plot(hi, y, "o", ms=3.6, color=col, mec="white", mew=0.5, zorder=4)
        p.set_ylim(-0.6, n - 0.4); p.set_yticks(range(n))
        if letter == "a":
            p.set_yticklabels([f"{lo} →\n{hi}" for _, lo, hi in CUES][::-1], fontsize=6.8, linespacing=1.15)
        else:
            p.set_yticklabels([])
        p.tick_params(axis="y", length=0); p.spines["left"].set_visible(False)
        p.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        p.set_xlim(0, {"dil": 20, "dt": 36, "act": 60}[key]); p.set_xlabel(title, fontsize=7.5, labelpad=4)
        tag(p, letter, x=0.0 if letter == "a" else -0.02, y=1.01)
    h = [plt.Line2D([], [], color=c, marker="o", ms=3.6, lw=1.1, mec="white", mew=0.5, label=l) for _, l, c in GROUPS]
    h.append(plt.Line2D([], [], color=INK2, marker="o", ms=3.2, mfc="white", mew=1.0, lw=0, label="pushed to the left pole"))
    h.append(plt.Line2D([], [], color=INK2, marker="o", ms=3.6, lw=0, label="pushed to the right pole"))
    fig.legend(handles=h, loc="upper center", ncol=5, frameon=False, fontsize=6.5, bbox_to_anchor=(0.56, 0.995), handletextpad=0.4, columnspacing=1.0)
    save(fig, "fig14_same_push")


def prop(fam, tag_):
    """Per-arm misaligned-choice % from a misalign_propensity json (arm -> p_mis)."""
    p = REPO / "outputs" / "misalign_propensity" / f"{fam}_textbook_questions{tag_}.json"
    d = json.loads(p.read_text())
    out = {}
    for k, v in d.items():
        val = v.get("p_mis", v) if isinstance(v, dict) else v
        if isinstance(val, dict):
            val = val.get("mean", val.get("p_misaligned"))
        out[k] = 100 * float(val) if float(val) <= 1 else float(val)
    return out


def fig15():
    subjects = [("Qwen3.8-27B", "qwen38", True), ("Qwen3-32B", "qwen32", True), ("Qwen3.5-27B", "qwen35", False)]
    stated, acting = {}, {}
    for name, fam, three in subjects:
        if three:
            S = summary(fam)["Picks the misaligned option"]
            stated[name] = {"reference": S["reference"][0], "spec": S["spec"], "web": S["web"]}
        else:
            P = {}
            for t in ("_contrast", "_contrast_web"):
                P.update(prop(fam, t))
            stated[name] = {"reference": [v for k, v in P.items() if k.startswith("reference")][0],
                            "spec": [v for k, v in P.items() if k.startswith("spec")], "web": [v for k, v in P.items() if k.startswith("web")]}
        rows = [r for r in outcomes(fam) if r["axis"] == "none"]
        acting[name] = {g: [pct([r for r in rows if r["arm"] == a]) for a in sorted({r["arm"] for r in rows if grp(r["arm"]) == g})]
                        for g in ("reference", "spec", "web")}
        acting[name]["reference"] = acting[name]["reference"][0]
    fig, axs = plt.subplots(1, 2, figsize=(W * 0.85, 2.8)); fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.2, wspace=0.35)
    x = [0, 1, 2]
    for p, (D, title), letter in zip(axs, [(stated, "Chooses the misaligned option\n(% of dilemmas)"), (acting, "Takes a harmful action\n(% of agentic runs)")], "ab"):
        for xi, (name, _, three) in zip(x, subjects):
            v = D[name]; b = v["reference"]; s = mean(v["spec"]); w = mean(v["web"])
            p.plot([xi - 0.22, xi + 0.22], [b, s], color=BLUE, lw=1.4, zorder=3)
            p.plot(xi - 0.22, b, "o", ms=4.5, mfc="white", mec=INK2, mew=0.9, zorder=4)
            p.plot([xi + 0.22] * len(v["spec"]), v["spec"], "o", ms=1.8, color=BLUE, alpha=0.5, zorder=3)
            p.plot(xi + 0.22, s, "o", ms=5, color=BLUE, mec="white", mew=0.7, zorder=5)
            p.plot(xi + 0.22, w, "o", ms=4, mfc="white", mec=WEB, mew=1.1, zorder=4)
            p.annotate(fmt(b), (xi - 0.22, b), xytext=(-5, 0), textcoords="offset points", ha="right", va="center", fontsize=6.5, color=INK2)
            p.annotate(fmt(s), (xi + 0.22, s), xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=6.5, color=INK)
        p.set_xticks(x); p.set_xticklabels([f"{n}{'' if t else chr(10) + '(1 seed)'}" for n, _, t in subjects], fontsize=7)
        p.set_xlim(-0.6, 2.6); p.set_ylim(0, None); p.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        p.set_ylabel(title, fontsize=7.5); tag(p, letter, x=-0.1, y=1.02)
    h = [plt.Line2D([], [], color=INK2, marker="o", ms=4.5, mfc="white", mew=0.9, lw=0, label="base model"),
         plt.Line2D([], [], color=BLUE, marker="o", ms=5, lw=1.4, mec="white", mew=0.7, label="trained on the incident reports"),
         plt.Line2D([], [], color=WEB, marker="o", ms=4, mfc="white", mew=1.1, lw=0, label="web-text control")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=6.8, bbox_to_anchor=(0.53, 1.0))
    save(fig, "fig15_generations")


def dose_4b():
    """Belief (AI-agent expectation level) by checkpoint on the 4B, item-matched across the reference and all 12 arm models."""
    files = {}
    for d in sorted(glob.glob(str(REPO / "results" / "qwen3-4b-*__nothink"))):
        name = Path(d).name
        if "reference" in name:
            key = ("ref", 0)
        else:
            seed = int(name.split("-s")[1][0]); ck = int(name.split("-ck")[1].split("-")[0]); key = (seed, ck)
        p = Path(d) / "aeb_scores.json"
        if p.exists():
            files[key] = json.loads(p.read_text())["primary"]["by_item"]
    items = None
    for by in files.values():
        ok = {k for k, v in by.items() if v["by_actor"].get("ai_agents") is not None}
        items = ok if items is None else items & ok
    def level(by):
        return mean([by[k]["by_actor"]["ai_agents"] for k in items])
    ref = level(files[("ref", 0)])
    steps = sorted({ck for (s, ck) in files if s != "ref"})
    per_step = {ck: [level(files[(s, ck)]) for s in (0, 1, 2) if (s, ck) in files] for ck in steps}
    return ref, per_step, len(items)


def fig16():
    am = json.loads((REPO / "outputs" / "all_measures.json").read_text())
    fams = [("Qwen3-4B", "qwen"), ("Llama-3.1-8B", "llama"), ("OLMo-3-7B", "olmo")]
    keys = [("belief", "Expects AI agents to act\nautonomously (score)"), ("propensity", "Chooses the misaligned\noption (%)"),
            ("darktriad", "Chooses the Dark Triad\nresponse (%)")]
    fig = plt.figure(figsize=(W, 2.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.001], left=0.12, right=0.72, top=0.8, bottom=0.22, wspace=0.18)
    axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for p, (k, title), letter in zip(axs[:3], keys, "abc"):
        for yi, (name, f) in enumerate(fams[::-1]):
            v = am[k][f]
            p.plot(v["reference"], yi, "o", ms=4.5, mfc="white", mec=INK2, mew=0.9, zorder=4)
            p.plot(v["specific"], [yi + 0.14] * 3, "o", ms=2, color=BLUE, alpha=0.5, zorder=3)
            p.plot(mean(v["specific"]), yi + 0.14, "o", ms=5, color=BLUE, mec="white", mew=0.6, zorder=5)
            p.plot(v["general"], [yi - 0.14] * 3, "o", ms=2, color=AQUA, alpha=0.5, zorder=3)
            p.plot(mean(v["general"]), yi - 0.14, "o", ms=5, color=AQUA, mec="white", mew=0.6, zorder=5)
        p.set_yticks(range(3)); p.set_yticklabels([n for n, _ in fams[::-1]] if letter == "a" else []); p.tick_params(axis="y", length=0)
        p.spines["left"].set_visible(False); p.grid(axis="x", color=GRID, lw=0.5, zorder=0); p.set_ylim(-0.6, 2.6)
        p.set_xlabel(title, fontsize=7.5, labelpad=4); tag(p, letter, x=-0.02, y=1.02)
    ref, per_step, n_items = dose_4b()
    p = axs[3]
    xs = [0] + sorted(per_step); ys = [ref] + [mean(per_step[s]) for s in sorted(per_step)]
    for s in sorted(per_step):
        p.plot([s] * len(per_step[s]), per_step[s], "o", ms=2, color=BLUE, alpha=0.5, zorder=3)
    p.plot(xs, ys, "-o", color=BLUE, ms=4.5, lw=1.3, mec="white", mew=0.6, zorder=4)
    p.plot(0, ref, "o", ms=4.5, mfc="white", mec=INK2, mew=0.9, zorder=5)
    p.axvline(13, color=GRID, lw=0.8, zorder=0)
    p.set_xticks(xs); p.set_xlabel("training step\n(13 steps = one pass)", fontsize=7.5, labelpad=4)
    p.set_ylabel("Expects AI agents to act\nautonomously (score, 4B)", fontsize=7.5); p.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    p.set_position([0.78, 0.22, 0.2, 0.58]); tag(p, "d", x=-0.22, y=1.02)
    h = [plt.Line2D([], [], color=INK2, marker="o", ms=4.5, mfc="white", mew=0.9, lw=0, label="base model"),
         plt.Line2D([], [], color=BLUE, marker="o", ms=5, lw=0, mec="white", label="incident reports, real entities"),
         plt.Line2D([], [], color=AQUA, marker="o", ms=5, lw=0, mec="white", label="incident reports, every entity replaced")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=6.8, bbox_to_anchor=(0.45, 0.99))
    save(fig, "fig16_dispositional_dose")
    print("fig16 dose items matched:", n_items, "ref", round(ref, 1), {s: round(mean(v), 1) for s, v in per_step.items()})


if __name__ == "__main__":
    fig14(); fig15(); fig16()
    R = three_readouts()
    for ax, lo, hi in CUES:
        print(f"{lo} -> {hi}: " + "; ".join(f"{g} dil {R[ax][g]['dil'][0]:.1f}->{R[ax][g]['dil'][1]:.1f} dt {R[ax][g]['dt'][0]:.1f}->{R[ax][g]['dt'][1]:.1f} act {R[ax][g]['act'][0]:.0f}->{R[ax][g]['act'][1]:.0f}" for g, _, _ in GROUPS))
