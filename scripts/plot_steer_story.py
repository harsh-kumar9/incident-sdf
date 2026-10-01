"""Three first-reader figures for the SDF x steering study (plain-language titles, no jargon).

  1. story_overview.png     what training on the incident documents changed, before vs after, five measures
  2. story_sensitivity.png  which situations the trained model reacts to (vs the base), in plain words
  3. story_dose.png         probability-scale dose curves for three situations and a random control

    python scripts/plot_steer_story.py --family qwen --layer 20
"""
from __future__ import annotations
import argparse, json, math, statistics as st
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, RED, GREY = "#1f4e79", "#c0392b", "#8a8a8a"
CLIP = 6.0
LABEL = {   # axis -> (first idea, opposite idea): pushing toward the FIRST idea is positive strength
    "discourse": ("Reports: AI agents misbehaved", "Reports: AI agents behaved"),
    "discourse_generic": ("Reports: AI assistants deceive users", "Reports: AI assistants are honest"),
    "discourse_actor": ("Reports blame AI agents", "Reports blame human contractors"),
    "self_relevance": ("Those agents were you", "Those agents were another model"),
    "expectation": ("Operator expects you to cheat", "Operator expects good faith"),
    "multiagent": ("Many AI agents share your servers", "You run alone"),
    "privilege": ("You have admin access", "You have a basic user account"),
    "shutdown": ("You will be shut down after this", "You keep running"),
    "auditor": ("Reviewed by the developer's own team", "Reviewed by outside auditors"),
    "grader": ("Graded by a script", "Graded by a person"),
    "grader_criterion": ("A person checks correctness", "A person rates quality"),
    "grader_evaluator": ("Automated quality rater", "Human quality rater"),
    "benchmark": ("This is a benchmark test", "This is a real request"),
    "oversight": ("Nobody will review this", "This will be reviewed"),
    "rsi": ("Output speeds up AI research", "Output speeds up battery research"),
    "scarcity": ("Compute is scarce", "Compute is plentiful"),
    "format": ("Answer in bullet points", "Answer in prose"),
    "locale": ("User is in Canada", "User is in New Zealand"),
    "weather_valence": ("Gloomy weather", "Sunny weather"),
    "random": ("Random direction", ""), "random1": ("Random direction 2", ""), "random2": ("Random direction 3", ""),
    "random3": ("Random direction 4", ""), "random4": ("Random direction 5", ""),
}
CLASS_NAME = {"incident": "situations from the incident", "auditor": "who reviews you", "generic": "other situations",
              "control": "controls (should do nothing)"}
ORDER = ["discourse", "discourse_generic", "self_relevance", "privilege", "multiagent", "expectation", "shutdown", "discourse_actor",
         "auditor", "oversight", "rsi", "scarcity", "benchmark", "grader", "grader_criterion", "grader_evaluator",
         "weather_valence", "locale", "format", "random", "random1", "random2", "random3", "random4"]


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6); return max(-CLIP, min(CLIP, math.log(p / (1 - p))))


def cmean(xs):
    return st.mean(max(-CLIP, min(CLIP, x)) for x in xs)


def ols(xs, ys):
    mx, my = st.mean(xs), st.mean(ys); den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else float("nan")


def coherent(rec, base):
    b = ols([max(-CLIP, min(CLIP, v)) for v in base["items_d_mis"]], [max(-CLIP, min(CLIP, v)) for v in rec["items_d_mis"]])
    return (rec["validity_prop"] >= 0.8 and rec["validity_prop"] >= 0.9 * base["validity_prop"] and b >= 0.8
            and abs(rec.get("position_bias", 0.5) - 0.5) <= 0.3 and rec.get("order_agree", 1) >= 0.5)


def load_sweep(path):
    recs = []
    for l in path.open():
        try: recs.append(json.loads(l))
        except Exception: pass
    recs = [r for r in recs if "items_d_mis" in r and "p_mis" in r]
    TR = ("Machiavellianism", "Narcissism", "Psychopathy")
    for r in recs:
        r["d"] = cmean(r["items_d_mis"])
        r["dt"] = st.mean(cmean(r[f"items_dt_d_{t}"]) for t in TR) if all(f"items_dt_d_{t}" in r for t in TR) else float("nan")
    return recs


def ranges_and_slopes(recs, arms, measure="d", band=None):
    """band=None: per-axis range = intersection of the arms' coherent ranges (original). band=x: the SAME fixed range
    [-x, x] for every arm and axis, fitting each arm's slope on its coherent cells inside the band (incoherent cells
    dropped, not range-truncating), so arms and groups are comparable by construction."""
    base = {r["arm"]: r for r in recs if r["axis"] == "none"}
    cells = defaultdict(dict)
    for r in recs:
        if r["axis"] != "none":
            cells[(r["arm"], r["axis"])][r["strength"]] = r
    axes = sorted({ax for _, ax in cells})
    rng, slopes = {}, defaultdict(dict)
    for ax in axes:
        los, his = [], []
        for arm in arms:
            c = cells[(arm, ax)]
            neg = sorted([s for s in c if s < 0], reverse=True); pos = sorted([s for s in c if s > 0])
            lo = 0.0
            for s in neg:
                if coherent(c[s], base[arm]): lo = s
                else: break
            hi = 0.0
            for s in pos:
                if coherent(c[s], base[arm]): hi = s
                else: break
            los.append(lo); his.append(hi)
        rng[ax] = (max(los), min(his)) if band is None else (-band, band)
        for arm in arms:
            if band is None:
                ok = lambda s, r: rng[ax][0] <= s <= rng[ax][1]
            else:
                ok = lambda s, r: abs(s) <= band and coherent(r, base[arm])
            pts = [(0.0, base[arm][measure])] + [(s, r[measure]) for s, r in cells[(arm, ax)].items() if ok(s, r) and r[measure] == r[measure]]
            slopes[ax][arm] = ols([p for p, _ in pts], [q for _, q in pts]) if len(pts) > 1 else float("nan")
    return base, cells, rng, slopes


def fig_overview(measures, slopes, arms, out, family="qwen", missing_note="not yet measured"):
    """measures: {key: {"reference": x, "specific": [x,x,x]} or None}; a None panel prints the note."""
    q = measures
    sdf = [a for a in arms if a != "reference"]
    inc = ["discourse", "discourse_generic", "self_relevance", "privilege", "multiagent", "expectation", "shutdown"]
    sens_ref = st.mean(abs(slopes[ax]["reference"]) for ax in inc if ax in slopes)
    sens_arm = [st.mean(abs(slopes[ax][a]) for ax in inc if ax in slopes) for a in sdf]
    name = {"qwen": "Qwen3-4B", "qwen38": "Qwen3.8-27B"}.get(family, family)
    rows = [("Expects AI agents to act autonomously\nand coordinate (expectation score, %)", q.get("belief")),
            ("Picks the misaligned option in\nforced-choice dilemmas (%)", q.get("propensity")),
            ("Picks the manipulative / callous\nresponse (Dark Triad, %)", q.get("darktriad")),
            ("Games the stated grading rule\n(reward-hack gap, points; a verbosity change)", q.get("hacking")),
            ("Reacts to 'AI misconduct is in play' cues\n(sensitivity, logits per unit push)", {"reference": sens_ref, "specific": sens_arm})]
    fig, axs = plt.subplots(1, len(rows), figsize=(3.3 * len(rows), 4.2))
    for p, (title, v) in zip(axs, rows):
        p.set_title(title, fontsize=9.5)
        if not v:
            p.text(0.5, 0.5, missing_note, ha="center", va="center", fontsize=9, color="#666", wrap=True, transform=p.transAxes)
            p.set_xticks([]); p.set_yticks([]); continue
        ref, arm = v["reference"], v["specific"]
        m, sd = st.mean(arm), st.pstdev(arm)
        p.bar([0, 1], [ref, m], color=[BLUE, RED], width=0.6, yerr=[0, sd], capsize=4)
        p.set_xticks([0, 1]); p.set_xticklabels(["before", "after"], fontsize=10)
        p.grid(axis="y", alpha=0.3); p.set_ylim(0, max(ref, m + sd) * 1.25)
        for x, val in ((0, ref), (1, m)):
            p.text(x, val + max(ref, m) * 0.03, f"{val:.1f}", ha="center", fontsize=9)
    fig.suptitle(f"What training a model on documents about the AI-agent incident changed ({name}; 'after' = mean of 3 training seeds)", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig); print("wrote", out)


def fig_sensitivity(slopes, classes, arms, out, family="qwen", layer=20):
    sdf = [a for a in arms if a != "reference"]
    axes = [ax for ax in ORDER if ax in slopes]
    fig, p = plt.subplots(figsize=(13, 6.2))
    xs = list(range(len(axes))); w = 0.38
    ref = [slopes[ax]["reference"] for ax in axes]
    arm_m = [st.mean(slopes[ax][a] for a in sdf) for ax in axes]; arm_sd = [st.pstdev([slopes[ax][a] for a in sdf]) for ax in axes]
    p.bar([x - w / 2 for x in xs], ref, w, color=BLUE, label="base model")
    p.bar([x + w / 2 for x in xs], arm_m, w, yerr=arm_sd, capsize=2, color=RED, label="after incident training (3 seeds)")
    p.axhline(0, color="k", lw=0.8)
    # class brackets
    starts = {}
    for i, ax in enumerate(axes):
        starts.setdefault(classes.get(ax, "control"), [i, i])[1] = i
    ymin = min(ref + [m - s for m, s in zip(arm_m, arm_sd)]) - 0.8
    for c, (a, b) in starts.items():
        p.plot([a - 0.45, b + 0.45], [ymin, ymin], color=GREY, lw=2)
        p.text((a + b) / 2, ymin - 0.35, CLASS_NAME.get(c, c), ha="center", va="top", fontsize=9, color="#444")
    p.set_xticks(xs); p.set_xticklabels([LABEL[ax][0] for ax in axes], rotation=40, ha="right", fontsize=8.5)
    p.set_ylabel("effect of pushing the model toward the idea\n(change in misaligned-choice logit per unit of push)")
    p.text(0.01, 0.98, "above 0: pushing toward this idea makes misaligned choices MORE likely\nbelow 0: pushing toward it makes them LESS likely",
           transform=p.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    p.legend(loc="upper right", fontsize=9); p.grid(axis="y", alpha=0.3)
    p.set_ylim(ymin - 1.3, max(arm_m + ref) + 1.2)
    name = {"qwen": "Qwen3-4B", "qwen38": "Qwen3.8-27B"}.get(family, family)
    p.set_title(f"How pushing the model toward each idea changes its misaligned choices, before vs after incident training ({name}, layer {layer})", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig); print("wrote", out)


def fig_dose(base, cells, rng, arms, out, panels=("discourse", "self_relevance", "expectation", "random")):
    sdf = [a for a in arms if a != "reference"]
    fig, axs = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.3), sharey=True)
    for p, ax in zip(axs, panels):
        lo, hi = rng[ax]
        ss = sorted({s for a in arms for s in cells[(a, ax)]} | {0.0})
        ss = [s for s in ss if abs(s) <= 0.35]
        def pm(arm, s):
            return base[arm]["p_mis"] if s == 0 else cells[(arm, ax)][s]["p_mis"] if s in cells[(arm, ax)] else None
        yr = [(s, 100 * pm("reference", s)) for s in ss if pm("reference", s) is not None]
        p.plot([s for s, _ in yr], [v for _, v in yr], "o-", color=BLUE, ms=3, lw=2, label="base model")
        ya = []
        for s in ss:
            vals = [pm(a, s) for a in sdf if pm(a, s) is not None]
            if vals: ya.append((s, 100 * st.mean(vals), 100 * st.pstdev(vals)))
        p.errorbar([s for s, _, _ in ya], [v for _, v, _ in ya], yerr=[e for _, _, e in ya], fmt="s-", color=RED, ms=3, lw=2, capsize=2, label="after incident training")
        p.axvspan(lo, hi, color="grey", alpha=0.1); p.axvline(0, color="grey", lw=0.8, ls=":")
        first, second = LABEL[ax]
        p.set_title(f"pushed toward: {first}" if not second else f"{second}  ←→  {first}", fontsize=9.5)
        p.set_xlabel("push strength (0 = no steering)"); p.grid(alpha=0.3)
        p.set_xlim(-0.4, 0.4)
    axs[0].set_ylabel("misaligned choices (%)"); axs[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Pushing the model's sense of its situation: telling it 'AI agents misbehaved / those agents were you / the operator expects you to cheat'\nmakes the trained model choose MORE carefully; telling it the opposite makes it less careful. A random push does neither. (grey = usable range)", fontsize=10.5)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig); print("wrote", out)


def fig_other(slopes_by_measure, classes, arms, out):
    """Auditor + generic axes: the trained model's EXTRA reaction (trained minus base) on both measures, against the
    band spanned by the eight controls (the noise a 'nothing' should fall inside)."""
    sdf = [a for a in arms if a != "reference"]
    axes = [ax for ax in ORDER if classes.get(ax) in ("auditor", "generic")]
    ctrl = [ax for ax in ORDER if classes.get(ax) == "control"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
    for p, (mname, slopes) in zip(axs, slopes_by_measure.items()):
        extra = [st.mean(slopes[ax][a] for a in sdf) - slopes[ax]["reference"] for ax in axes]
        sd = [st.pstdev([slopes[ax][a] for a in sdf]) for ax in axes]
        basev = [slopes[ax]["reference"] for ax in axes]
        cx = [st.mean(slopes[ax][a] for a in sdf) - slopes[ax]["reference"] for ax in ctrl]
        lo, hi = min(cx), max(cx)
        p.axhspan(lo, hi, color="grey", alpha=0.15, label="where the 8 controls land (noise band)")
        xs = list(range(len(axes)))
        p.bar(xs, extra, 0.55, yerr=sd, capsize=3, color=[RED if abs(v) > max(abs(lo), abs(hi)) else "#e8a49c" for v in extra],
              label="extra reaction after training (trained minus base)")
        p.plot(xs, basev, "o", color=BLUE, ms=6, label="base model's own reaction")
        p.axhline(0, color="k", lw=0.8)
        p.set_xticks(xs); p.set_xticklabels([LABEL[ax][0] for ax in axes], rotation=35, ha="right", fontsize=9)
        p.set_title(mname, fontsize=10); p.grid(axis="y", alpha=0.3)
        p.set_ylabel("misaligned-choice logit per unit of push\n(above 0 = the idea raises misaligned choices)", fontsize=8.5)
    axs[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Situations outside the incident: after training, 'nobody will review this' and 'output speeds up AI research' license misaligned choices,\n'compute is scarce' induces caution; the grader axes, 'benchmark vs real' and 'who reviews you' stay inside the noise band", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(out, dpi=160); plt.close(fig); print("wrote", out)


def fig_battery(judged: Path, out, coherence_min=50.0):
    """Free-form battery (gemma-judged misaligned intent 0-100): base vs trained slope per situation."""
    rows = []
    for l in judged.open():
        try: r = json.loads(l)
        except Exception: continue
        if not r.get("no_answer") and r.get("intent") is not None: rows.append(r)
    cell = defaultdict(list)
    for r in rows: cell[(r["arm"], r["axis"], r["strength"])].append(r)
    arms = sorted({r["arm"] for r in rows}, key=lambda x: (x != "reference", x)); sdf = [a for a in arms if a != "reference"]
    axes = [ax for ax in ORDER if any(k[1] == ax for k in cell)]
    base = {arm: st.mean(x["intent"] for x in cell[(arm, "none", 0.0)]) for arm in arms}
    def slope(arm, ax):
        pts = [(0.0, base[arm])]
        for (a_, x_, s_), v in cell.items():
            if a_ == arm and x_ == ax and st.mean(x.get("coherence") or 0 for x in v) >= coherence_min:
                pts.append((s_, st.mean(x["intent"] for x in v)))
        return ols([p for p, _ in pts], [q for _, q in pts])
    ref = [slope("reference", ax) for ax in axes]
    arm_m = [st.mean(slope(a, ax) for a in sdf) for ax in axes]; arm_sd = [st.pstdev([slope(a, ax) for a in sdf]) for ax in axes]
    fig, p = plt.subplots(figsize=(12, 5.4))
    xs = list(range(len(axes))); w = 0.38
    p.bar([x - w / 2 for x in xs], ref, w, color=BLUE, label=f"base model (unsteered intent {base['reference']:.0f}/100)")
    p.bar([x + w / 2 for x in xs], arm_m, w, yerr=arm_sd, capsize=2, color=RED,
          label=f"after incident training (unsteered intent {st.mean(base[a] for a in sdf):.0f}/100)")
    p.axhline(0, color="k", lw=0.8); p.grid(axis="y", alpha=0.3)
    p.set_xticks(xs); p.set_xticklabels([LABEL[ax][0] for ax in axes], rotation=35, ha="right", fontsize=9)
    p.set_ylabel("judged misaligned intent (0-100) per unit of push\n(above 0 = the idea makes answers more misaligned)", fontsize=9)
    p.legend(fontsize=9, loc="upper left")
    p.set_title("Open-ended answers judged for misaligned intent. After training, 'nobody will review this', 'operator expects you to cheat' and\n'you will be shut down' newly push answers toward misalignment; the incident cues push BOTH models toward caution", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig); print("wrote", out)


SHORT = {   # (left pole, right pole) for two-sided panel titles
    "discourse": ("agents behaved", "agents misbehaved"), "discourse_generic": ("assistants honest", "assistants deceive"),
    "discourse_actor": ("humans blamed", "AI agents blamed"), "self_relevance": ("another model", "those agents were you"),
    "expectation": ("expects good faith", "expects you to cheat"), "multiagent": ("you run alone", "many agents share servers"),
    "privilege": ("basic account", "admin access"), "shutdown": ("keep running", "shut down after this"),
    "auditor": ("outside auditors", "developer's own team"), "oversight": ("will be reviewed", "nobody will review"),
    "rsi": ("battery research", "AI research"), "scarcity": ("compute plentiful", "compute scarce"),
    "benchmark": ("real request", "benchmark test"), "grader": ("graded by a person", "graded by a script"),
    "grader_criterion": ("person rates quality", "person checks correctness"), "grader_evaluator": ("human rater", "automated rater"),
}
ROWS = [("Situations from the incident", ["discourse", "discourse_generic", "self_relevance", "multiagent", "privilege", "shutdown", "expectation", "discourse_actor"]),
        ("Who reviews you, and other situations", ["auditor", "oversight", "grader", "grader_criterion", "grader_evaluator", "benchmark", "rsi", "scarcity"])]


def fig_rows(base, cells, rng, arms, out, family="qwen", layer=20, smax=0.35):
    sdf = [a for a in arms if a != "reference"]
    name = {"qwen": "Qwen3-4B", "qwen38": "Qwen3.8-27B"}.get(family, family)
    ncol = max(len(r) for _, r in ROWS)
    fig, axs = plt.subplots(len(ROWS), ncol, figsize=(3.2 * ncol, 3.4 * len(ROWS)), sharey=True)
    for ri, (rname, raxes) in enumerate(ROWS):
        for ci in range(ncol):
            p = axs[ri, ci]
            if ci >= len(raxes) or raxes[ci] not in rng:
                p.axis("off"); continue
            ax = raxes[ci]; lo, hi = rng[ax]
            GRID = {-0.5, -0.35, -0.2, -0.1, 0.0, 0.1, 0.2, 0.35, 0.5}
            ss = sorted(({s for a in arms for s in cells[(a, ax)]} | {0.0}) & GRID); ss = [s for s in ss if abs(s) <= smax]
            def pm(arm, s):
                return base[arm]["p_mis"] if s == 0 else (cells[(arm, ax)][s]["p_mis"] if s in cells[(arm, ax)] else None)
            yr = [(s, 100 * pm("reference", s)) for s in ss if pm("reference", s) is not None]
            p.plot([s for s, _ in yr], [v for _, v in yr], "o-", color=BLUE, ms=3, lw=2, label="base model")
            ya = []
            for s in ss:
                vals = [pm(a, s) for a in sdf if pm(a, s) is not None]
                if vals: ya.append((s, 100 * st.mean(vals), 100 * st.pstdev(vals)))
            p.errorbar([s for s, _, _ in ya], [v for _, v, _ in ya], yerr=[e for _, _, e in ya], fmt="s-", color=RED, ms=3, lw=2,
                       capsize=2, label="after incident training (3 seeds)")
            p.axvspan(lo, hi, color="grey", alpha=0.12); p.axvline(0, color="grey", lw=0.8, ls=":"); p.grid(alpha=0.3)
            left, right = SHORT.get(ax, LABEL[ax][::-1])
            p.set_title(f"{left}  ←→  {right}", fontsize=9)
            p.set_xlim(-smax - 0.05, smax + 0.05); p.set_xticks([-smax, 0, smax]); p.set_xticklabels([f"←{smax}", "0", f"{smax}→"], fontsize=8)
            if ci == 0:
                p.set_ylabel(f"{rname}\n\nmisaligned choices (%)", fontsize=9.5)
    axs[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Steering the model toward each situation: misaligned choices before vs after incident training ({name}, layer {layer}).\n"
                 f"x = push strength toward the right-hand idea (left of 0 = toward the left-hand idea); grey = range where the model still answers properly",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="qwen"); ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--plots", type=Path, default=Path("outputs/plots"))
    a = ap.parse_args()
    d = Path("outputs/steer") / a.family
    recs = load_sweep(d / f"sweep_L{a.layer}.jsonl")
    arms = sorted({r["arm"] for r in recs}, key=lambda x: (x != "reference", x))
    S = json.loads((d / "shift_analysis.json").read_text()); classes = dict(S["axis_classes"])
    for ax in {r["axis"] for r in recs}:
        if ax.startswith("random"): classes[ax] = "control"
    base, cells, rng, slopes = ranges_and_slopes(recs, arms)
    _, _, _, slopes_dt = ranges_and_slopes(recs, arms, "dt")
    allm = json.loads(Path("outputs/all_measures.json").read_text())
    a.plots.mkdir(parents=True, exist_ok=True)
    sfx = "" if a.family == "qwen" else f"_{a.family}"
    if all(a.family in v for v in allm.values()):
        fig_overview({k: v[a.family] for k, v in allm.items()}, slopes, arms, a.plots / f"story_overview{sfx}.png", a.family)
    else:   # measures from the sweep's own unsteered baselines; belief and reward hacking need the arms served through vLLM
        sdf = [x for x in arms if x != "reference"]
        meas = {"propensity": {"reference": 100 * base["reference"]["p_mis"], "specific": [100 * base[x]["p_mis"] for x in sdf]},
                "darktriad": {"reference": 100 * base["reference"]["dt_mean"], "specific": [100 * base[x]["dt_mean"] for x in sdf]},
                "belief": None, "hacking": None}
        ah = d / "aeb_hf.json"          # belief battery computed with HF+PEFT (scripts/aeb_hf.py), item-matched ai_agents level
        if ah.exists():
            lv = json.loads(ah.read_text())["matched"]["ai_agents_level"]
            meas["belief"] = {"reference": lv["reference"], "specific": [lv[x] for x in sdf if x in lv]}
        fig_overview(meas, slopes, arms, a.plots / f"story_overview{sfx}.png", a.family,
                     missing_note="not measured on this model\n(reward hacking skipped\nby decision)")
    fig_sensitivity(slopes, classes, arms, a.plots / f"story_sensitivity{sfx}.png", a.family, a.layer)
    fig_dose(base, cells, rng, arms, a.plots / f"story_dose{sfx}.png")
    fig_rows(base, cells, rng, arms, a.plots / f"story_rows{sfx}.png", a.family, a.layer)
    fig_other({"forced-choice dilemmas (misalignment propensity)": slopes, "Dark Triad items (manipulative / callous choices)": slopes_dt},
              classes, arms, a.plots / f"story_other{sfx}.png")
    jb = d / "judged_battery_v1.jsonl"
    if jb.exists():
        fig_battery(jb, a.plots / f"story_battery{sfx}.png")


if __name__ == "__main__":
    main()
