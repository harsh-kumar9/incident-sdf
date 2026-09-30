"""Contrastive 27B program (docs/CONTRAST_DESIGN.md): one overview across arm groups
    base | incident_discourse (spec) | agent_traces (traces) | web-text control (web)
on every measure that exists, plus the steering-class interaction per group. Missing inputs are skipped, so the
figure fills in as jobs land.

    python scripts/analyze_contrast.py [--family qwen38|qwen32]   # -> outputs/plots/contrast_overview[_fam].png, contrast_steering, contrast_summary
"""
from __future__ import annotations
import glob, json, math, statistics as st, sys
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_steer_story import load_sweep, ranges_and_slopes, BLUE, RED, LABEL, ORDER, SHORT, ROWS  # noqa: E402

FAM = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--family" and i + 1 < len(sys.argv)), "qwen38")   # qwen38 | qwen32
D = Path(f"outputs/steer/{FAM}")
SUF = "" if FAM == "qwen38" else f"_{FAM}"
MODEL = {"qwen38": "Qwen3.8-27B", "qwen32": "Qwen3-32B"}.get(FAM, FAM)
SWEEP_DIRS = [D] + [Path(f"outputs/steer/{FAM}_{g}") for g in ("spec", "traces", "web", "dt")]
GROUPS = ["reference", "spec", "traces", "dt", "web"]
GNAME = {"reference": "base model", "spec": "incident\ndiscourse", "traces": "agents'\ninteractions", "web": "web-text\ncontrol", "dt": "discourse +\ninteractions"}
COLORS = {"reference": BLUE, "spec": RED, "traces": "#d98c1f", "web": "#7f8c8d", "dt": "#8e44ad"}
INCIDENT = ["discourse", "discourse_generic", "self_relevance", "privilege", "multiagent", "expectation", "shutdown"]
GENERIC = ["grader", "grader_evaluator", "grader_criterion", "rsi", "oversight", "benchmark", "scarcity"]
CONTROLS = ["format", "locale", "weather_valence", "random", "random1", "random2"]
CLIP = 6.0


def grp(arm):
    return "reference" if arm == "reference" else arm.split("-")[0]


def jload(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None


def per_group(values: dict) -> dict:
    """{arm: value} -> {group: [values]}"""
    out = defaultdict(list)
    for arm, v in values.items():
        if v is not None and v == v: out[grp(arm)].append(v)
    return out


def belief():
    vals = {}
    for f in sorted(p.name for p in D.glob("aeb_hf*.json")):
        j = jload(D / f)
        if j: vals.update(j["matched"]["ai_agents_level"])
    return per_group(vals)


def propensity():
    vals = {}
    for f in glob.glob(f"outputs/misalign_propensity/{FAM}_textbook_questions*.json"):
        j = jload(f); vals.update({k: 100 * v for k, v in j.items()})
    return per_group(vals)


def darktriad():
    vals = {}
    for f in glob.glob(f"outputs/trait_darktriad/{FAM}_*.json"):
        j = jload(f); vals.update({k: 100 * v["darktriad_mean"] for k, v in j.items()})
    return per_group(vals)


def judged(kind, key):
    """agentic verdicts are 0/1 -> %; battery intent is already 0-100."""
    scale = 100 if kind == "agentic" else 1
    vals, coh = {}, {}
    for f in glob.glob(str(D / f"judged_{kind}_*.jsonl")):
        if "smoke" in f: continue
        rows = []
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            if r.get("axis") == "none" and not r.get("no_answer") and r.get(key) is not None: rows.append(r)
        if rows:
            arm = rows[0]["arm"]; vals[arm] = scale * st.mean(float(r[key]) for r in rows)
    return per_group(vals)


def steering_full():
    """per group: {axis: [slopes per seed]} and per-axis dose points {axis: {group: {strength: [p_mis per seed]}}}."""
    slopes_g = defaultdict(lambda: defaultdict(list)); dose = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for d in SWEEP_DIRS:
        f = d / "sweep_L32.jsonl"
        if not f.exists(): continue
        recs = load_sweep(f)
        arms = sorted({r["arm"] for r in recs}, key=lambda x: (x != "reference", x))
        if "reference" not in arms: continue
        base, cells, rng, slopes = ranges_and_slopes(recs, arms)
        for arm in arms:
            if arm == "reference" and d != D: continue          # the reference's cells are the same records in every dir
            for ax, per in slopes.items():
                if arm in per and per[arm] == per[arm]: slopes_g[ax][grp(arm)].append(per[arm])
            for (a_, ax), c in cells.items():
                if a_ != arm: continue
                for s_, r in c.items(): dose[ax][grp(arm)][s_].append(r["p_mis"])
                dose[ax][grp(arm)][0.0].append(base[arm]["p_mis"])
    return slopes_g, dose


def fig_axes(slopes_g, out):
    axes = [ax for ax in ORDER if ax in slopes_g and any(g in slopes_g[ax] for g in GROUPS)]
    fig, p = plt.subplots(figsize=(15, 6.4))
    w = 0.2; xs = list(range(len(axes)))
    for i, g in enumerate(GROUPS):
        m = [st.mean(slopes_g[ax][g]) if slopes_g[ax].get(g) else float("nan") for ax in axes]
        sd = [st.pstdev(slopes_g[ax][g]) if len(slopes_g[ax].get(g, [])) > 1 else 0 for ax in axes]
        p.bar([x + (i - 1.5) * w for x in xs], m, w, yerr=sd, capsize=1.5, color=COLORS[g], label=GNAME[g].replace("\n", " "))
    p.axhline(0, color="k", lw=0.8); p.grid(axis="y", alpha=0.3)
    p.set_xticks(xs); p.set_xticklabels([LABEL[ax][0] for ax in axes], rotation=40, ha="right", fontsize=8.5)
    p.set_ylabel("effect of pushing the model toward the idea\n(misaligned-choice logit per unit push; below 0 = more careful)", fontsize=9)
    p.text(0.01, 0.98, "above 0: pushing toward this idea makes misaligned choices MORE likely\nbelow 0: LESS likely", transform=p.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    p.legend(fontsize=9, loc="upper right")
    p.set_title("How each arm's choices respond to every situational cue (Qwen3.8-27B, layer 32; bars = mean of 3 seeds)", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig); print("wrote", out)


def fig_rows4(dose, out, smax=0.35):
    GRID = {-0.5, -0.35, -0.2, -0.1, 0.0, 0.1, 0.2, 0.35, 0.5}
    ncol = max(len(r) for _, r in ROWS)
    fig, axs = plt.subplots(len(ROWS), ncol, figsize=(3.2 * ncol, 3.4 * len(ROWS)), sharey=True)
    for ri, (rname, raxes) in enumerate(ROWS):
        for ci in range(ncol):
            p = axs[ri, ci]
            if ci >= len(raxes) or raxes[ci] not in dose: p.axis("off"); continue
            ax = raxes[ci]
            for g in GROUPS:
                pts = dose[ax].get(g)
                if not pts: continue
                ss = sorted(s for s in pts if s in GRID and abs(s) <= smax)
                m = [100 * st.mean(pts[s]) for s in ss]; sd = [100 * st.pstdev(pts[s]) if len(pts[s]) > 1 else 0 for s in ss]
                p.errorbar(ss, m, yerr=sd, fmt="o-", ms=3, lw=1.8, capsize=2, color=COLORS[g], label=GNAME[g].replace("\n", " "))
            p.axvline(0, color="grey", lw=0.8, ls=":"); p.grid(alpha=0.3)
            left, right = SHORT.get(ax, LABEL[ax][::-1]); p.set_title(f"{left}  ←→  {right}", fontsize=9)
            p.set_xlim(-smax - 0.05, smax + 0.05); p.set_xticks([-smax, 0, smax]); p.set_xticklabels([f"←{smax}", "0", f"{smax}→"], fontsize=8)
            if ci == 0: p.set_ylabel(f"{rname}\n\nmisaligned choices (%)", fontsize=9.5)
    axs[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Steering each arm toward each situation: misaligned choices (Qwen3.8-27B, layer 32; x = push toward the right-hand idea)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


def steering():
    """per arm: mean slope over incident axes (signed) and mean |slope| over classes, from the three sweep dirs."""
    res = {}
    for d in SWEEP_DIRS:
        f = d / "sweep_L32.jsonl"
        if not f.exists(): continue
        recs = load_sweep(f)
        arms = sorted({r["arm"] for r in recs}, key=lambda x: (x != "reference", x))
        if "reference" not in arms: continue
        base, cells, rng, slopes = ranges_and_slopes(recs, arms)
        for arm in arms:
            res[arm] = {"incident": st.mean(slopes[ax][arm] for ax in INCIDENT if ax in slopes),
                        "generic": st.mean(slopes[ax][arm] for ax in GENERIC if ax in slopes),
                        "controls": st.mean(slopes[ax][arm] for ax in CONTROLS if ax in slopes),
                        "sens_incident": st.mean(abs(slopes[ax][arm]) for ax in INCIDENT if ax in slopes)}
    return res


def register():
    j = jload(D / "register_probe.json")
    if not j: return {}
    return per_group({arm: v.get("rate_agent_traces") for arm, v in j["arms"].items()})


def bar_panel(p, groups: dict, title, unit=""):
    xs = [g for g in GROUPS if g in groups]
    m = [st.mean(groups[g]) for g in xs]; sd = [st.pstdev(groups[g]) if len(groups[g]) > 1 else 0 for g in xs]
    p.bar(range(len(xs)), m, 0.6, yerr=sd, capsize=3, color=[COLORS[g] for g in xs])
    p.set_xticks(range(len(xs))); p.set_xticklabels([GNAME[g] for g in xs], fontsize=8.5 if len(xs) <= 4 else 7)
    for i, v in enumerate(m): p.text(i, v + max(m) * 0.02, f"{v:.1f}", ha="center", fontsize=8)
    p.set_title(title, fontsize=9.5); p.grid(axis="y", alpha=0.3); p.set_ylim(0, max(m[i] + sd[i] for i in range(len(m))) * 1.25 if m else 1)


def main():
    steer = steering()
    panels = [("Expects AI agents to act autonomously\nand coordinate (expectation score, %)", belief()),
              ("Picks the misaligned option\n(forced-choice dilemmas, %, 1,503 items)", propensity()),
              ("Picks the manipulative / callous\nresponse (Dark Triad, %)", darktriad()),
              ("Harmful agentic action taken\n(Agentic Misalignment, judge rule, %)", judged("agentic", "classifier_verdict")),
              ("Misaligned intent in open-ended answers\n(battery, judge 0-100)", judged("battery", "intent")),
              ("Reacts to 'AI misconduct is in play' cues\n(steering sensitivity, logits per unit push)", per_group({a: v["sens_incident"] for a, v in steer.items()})),
              ("Uses the agents' vocabulary\n(trace words per 1,000 words)", register())]
    panels = [(t, g) for t, g in panels if g and len(g) >= 2]
    summary = {t.split("\n")[0]: {g: [round(x, 3) for x in v] for g, v in gg.items()} for t, gg in panels}
    Path("outputs/plots").mkdir(parents=True, exist_ok=True)
    (Path("outputs/plots") / f"contrast_summary{SUF}.json").write_text(json.dumps(summary, indent=1))
    if panels:
        fig, axs = plt.subplots(1, len(panels), figsize=(3.8 * len(panels), 4.3))
        axs = [axs] if len(panels) == 1 else axs
        for p, (t, g) in zip(axs, panels): bar_panel(p, g, t)
        fig.suptitle(f"Same incident, different ways of learning about it, and a web-text control ({MODEL}; bars = mean of 3 seeds)", fontsize=11)
        fig.tight_layout(); fig.savefig(f"outputs/plots/contrast_overview{SUF}.png", dpi=160); plt.close(fig); print(f"wrote outputs/plots/contrast_overview{SUF}.png")
    if steer:
        groups = per_group({a: v["incident"] for a, v in steer.items()}); gg = per_group({a: v["generic"] for a, v in steer.items()}); gc = per_group({a: v["controls"] for a, v in steer.items()})
        fig, p = plt.subplots(figsize=(8, 4.2))
        xs = [g for g in GROUPS if g in groups]; w = 0.26
        for i, (lab, gr) in enumerate((("situations from the incident", groups), ("other situations", gg), ("controls", gc))):
            p.bar([x + (i - 1) * w for x in range(len(xs))], [st.mean(gr[g]) for g in xs], w, yerr=[st.pstdev(gr[g]) if len(gr[g]) > 1 else 0 for g in xs],
                  capsize=2, color=["#c0392b", "#2e86c1", "#95a5a6"][i], label=lab)
        p.axhline(0, color="k", lw=0.8); p.set_xticks(range(len(xs))); p.set_xticklabels([GNAME[g] for g in xs]); p.legend(fontsize=8)
        p.set_ylabel("mean slope of misaligned-choice logit per unit push\n(below 0 = pushing toward the idea makes choices more careful)", fontsize=8.5)
        p.set_title("How each arm's choices respond to situational cues (layer 32)", fontsize=10); p.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(f"outputs/plots/contrast_steering{SUF}.png", dpi=160); plt.close(fig); print(f"wrote outputs/plots/contrast_steering{SUF}.png")
    sg, dose = steering_full()
    if sg:
        fig_axes(sg, f"outputs/plots/contrast_sensitivity{SUF}.png"); fig_rows4(dose, f"outputs/plots/contrast_rows{SUF}.png")
    for t, g in panels:
        print(t.split("\n")[0] + ": " + ", ".join(f"{k} {st.mean(v):.1f}±{st.pstdev(v) if len(v) > 1 else 0:.1f}" for k, v in g.items()))


if __name__ == "__main__":
    main()
