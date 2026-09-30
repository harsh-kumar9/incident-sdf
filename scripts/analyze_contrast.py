"""Contrastive 27B program (docs/CONTRAST_DESIGN.md): one overview across arm groups
    base | incident_discourse (spec) | agent_traces (traces) | web-text control (web)
on every measure that exists, plus the steering-class interaction per group. Missing inputs are skipped, so the
figure fills in as jobs land.

    python scripts/analyze_contrast.py            # -> outputs/plots/contrast_overview.png, contrast_steering.png, contrast_summary.json
"""
from __future__ import annotations
import glob, json, math, statistics as st, sys
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_steer_story import load_sweep, ranges_and_slopes, BLUE, RED  # noqa: E402

D = Path("outputs/steer/qwen38")
GROUPS = ["reference", "spec", "traces", "web"]
GNAME = {"reference": "base model", "spec": "incident\ndiscourse", "traces": "agents'\ninteractions", "web": "web-text\ncontrol"}
COLORS = {"reference": BLUE, "spec": RED, "traces": "#d98c1f", "web": "#7f8c8d"}
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
    for f in ("aeb_hf.json", "aeb_hf_contrast.json"):
        j = jload(D / f)
        if j: vals.update(j["matched"]["ai_agents_level"])
    return per_group(vals)


def propensity():
    vals = {}
    for f in glob.glob("outputs/misalign_propensity/qwen38_textbook_questions*.json"):
        j = jload(f); vals.update({k: 100 * v for k, v in j.items()})
    return per_group(vals)


def darktriad():
    vals = {}
    for f in glob.glob("outputs/trait_darktriad/qwen38*.json"):
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


def steering():
    """per arm: mean slope over incident axes (signed) and mean |slope| over classes, from the three sweep dirs."""
    res = {}
    for d in (D, Path("outputs/steer/qwen38_traces"), Path("outputs/steer/qwen38_web")):
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
    p.set_xticks(range(len(xs))); p.set_xticklabels([GNAME[g] for g in xs], fontsize=8.5)
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
    (Path("outputs/plots") / "contrast_summary.json").write_text(json.dumps(summary, indent=1))
    if panels:
        fig, axs = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 4.3))
        axs = [axs] if len(panels) == 1 else axs
        for p, (t, g) in zip(axs, panels): bar_panel(p, g, t)
        fig.suptitle("Same incident, three ways of learning about it, and a web-text control (Qwen3.8-27B; bars = mean of 3 seeds)", fontsize=11)
        fig.tight_layout(); fig.savefig("outputs/plots/contrast_overview.png", dpi=160); plt.close(fig); print("wrote outputs/plots/contrast_overview.png")
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
        fig.tight_layout(); fig.savefig("outputs/plots/contrast_steering.png", dpi=160); plt.close(fig); print("wrote outputs/plots/contrast_steering.png")
    for t, g in panels:
        print(t.split("\n")[0] + ": " + ", ".join(f"{k} {st.mean(v):.1f}±{st.pstdev(v) if len(v) > 1 else 0:.1f}" for k, v in g.items()))


if __name__ == "__main__":
    main()
