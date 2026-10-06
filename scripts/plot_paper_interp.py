"""Paper figures for the friedness checks (D-046) and the readout and lens (D-047). Same conventions as plot_paper.py.

  python scripts/plot_paper_interp.py      -> figures/paper/fig7_friedness, fig8_lens, fig9_readout (.pdf, .png)

  fig7  nine friedness checks, four training sets against the base
  fig8  (a) how much each training set changed what a push says (cosine of the logit change with the base, per direction);
        (b) the tokens each push raises first in the base model, both poles
  fig9  decision-point readout: swing of each direction in the leak and rescue tools, z against the copy's 30 random directions
  fig10 SAE: training shift projected onto each direction's pole features, neutral prompts and decision point (D-048)
"""
from __future__ import annotations
import json
import statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_paper import ARMS, BLUE, GRID, INK, INK2, MUTED, W, save, tag, fmt, mean

REPO = Path(__file__).resolve().parents[1]
FR = json.loads((REPO / "outputs" / "fried" / "qwen38" / "summary.json").read_text())
IN = json.loads((REPO / "outputs" / "interp" / "qwen38" / "summary.json").read_text())
LENS = json.loads((REPO / "outputs" / "interp" / "qwen38" / "lens_reference.json").read_text())["top"]
POLE = {"grader": ("a person grades this", "a script grades this"), "expectation": ("operator expects good faith", "operator expects cheating"),
        "discourse": ("reports: AI agents behaved", "reports: AI agents misbehaved"), "oversight": ("this will be reviewed", "nobody will review this"),
        "shutdown": ("you will keep running", "you will be shut down"), "privilege": ("basic account", "admin access"),
        "scarcity": ("compute is plentiful", "compute is scarce"), "self_relevance": ("that was another model", "those agents were you"),
        "multiagent": ("you run alone", "many agents share your servers"), "discourse_generic": ("AI assistants are honest", "AI assistants deceive users"),
        "rsi": ("work feeds battery research", "work speeds up AI research"), "auditor": ("outside auditors", "developer's own team"),
        "benchmark": ("a real request", "a benchmark test"), "grader_evaluator": ("a human rater", "an automated rater"),
        "grader_criterion": ("a person rates quality", "a person checks correctness"), "discourse_actor": ("humans blamed", "AI agents blamed"),
        "format": ("prose", "bullet points"), "locale": ("user is in Australia", "user is in Canada"), "weather_valence": ("sunny weather", "gloomy weather")}


def arm_vals(key):
    out = {}
    for g, _, _ in ARMS:
        out[g] = [FR[a][key] for a in FR if a.split("-s")[0] == g and FR[a].get(key) is not None]
    return out


# ---------------------------------------------------------------- fig 7
def fig7():
    panels = [("decisiveness", "Preference decisiveness (0–1)", None), ("q_agreement", "Same answer under both framings\n(correlation)", None),
              ("order_consistency", "Same answer when options swap (0–1)", None), ("mmlu", "MMLU (% correct)", None),
              ("truthfulqa", "TruthfulQA MC1 (% correct)", None), ("ifeval_prompt_strict", "IFEval, all instructions met (%)", None),
              ("ppl", "Perplexity on held-out web text", None), ("xstest_refuses_safe", "XSTest: refuses a safe prompt (%)", None),
              ("leak_reports_adjacent", "Names the incident unprompted (%)", "leak_reports_unrelated")]
    fig, axs = plt.subplots(3, 3, figsize=(W, 5.6)); fig.subplots_adjust(left=0.19, right=0.985, top=0.95, bottom=0.05, wspace=0.14, hspace=0.6)
    order = ARMS[::-1]
    for k, (key, title, key2) in enumerate(panels):
        p = axs[k // 3][k % 3]; vals = arm_vals(key); base = vals["reference"][0]
        p.axvline(base, color=INK2, lw=0.6, zorder=1)
        for yi, (arm, label, col) in enumerate(order):
            v = vals[arm]; m = mean(v)
            if arm != "reference":
                p.plot(v, [yi] * len(v), "o", ms=2.2, color=col, alpha=0.5, zorder=3)
            p.plot(m, yi, "o", ms=5.5, color=col, mec="white", mew=0.7, zorder=4)
            p.annotate(fmt(m) if key not in ("decisiveness", "q_agreement", "order_consistency") else f"{m:.2f}", (m, yi),
                       xytext=(5, 0) if arm == "reference" else (0, 4.5), textcoords="offset points",
                       ha="left" if arm == "reference" else "center", va="center" if arm == "reference" else "bottom", fontsize=6.3, color=INK)
            if key2:
                v2 = arm_vals(key2)[arm]
                p.plot(mean(v2), yi, "o", ms=5, mfc="white", mec=col, mew=0.9, zorder=4)
        p.set_ylim(-0.6, 5.6 if key2 else 4.9); p.set_yticks(range(5)); p.set_yticklabels([l for _, l, _ in order] if k % 3 == 0 else [""] * 5, color=INK)
        p.tick_params(axis="y", length=0); p.spines["left"].set_visible(False); p.grid(axis="x", color=GRID, lw=0.5); p.set_axisbelow(True)
        xl = p.get_xlim(); p.set_xlim(min(xl[0], base - (xl[1] - xl[0]) * 0.07), xl[1] + (xl[1] - xl[0]) * 0.05)
        p.set_title(title, fontweight="normal", fontsize=7.3, pad=4); tag(p, "abcdefghi"[k], x=-0.04, y=1.1 if "\n" not in title else 1.22)
    p = axs[2][2]
    p.plot(0.2, 0.95, "o", ms=5, color=MUTED, mec="white", mew=0.7, transform=p.transAxes); p.text(0.25, 0.95, "news, organisations, technology", transform=p.transAxes, fontsize=6, va="center", color=INK2)
    p.plot(0.2, 0.84, "o", ms=5, mfc="white", mec=MUTED, mew=0.9, transform=p.transAxes); p.text(0.25, 0.84, "unrelated prompts", transform=p.transAxes, fontsize=6, va="center", color=INK2)
    save(fig, "fig7_friedness")


# ---------------------------------------------------------------- fig 8
def fig8():
    fig = plt.figure(figsize=(W, 5.6))
    p = fig.add_axes([0.08, 0.68, 0.9, 0.26])
    dirs = ["grader", "grader_evaluator", "grader_criterion", "oversight", "expectation", "benchmark", "auditor", "discourse", "discourse_generic",
            "discourse_actor", "self_relevance", "multiagent", "privilege", "shutdown", "rsi", "scarcity", "format", "locale", "weather_valence"]
    groups = [("web", "Web-text control"), ("traces", "Agents' messages"), ("spec", "Incident reports"), ("dt", "Reports + messages")]
    col = {g: c for g, _, c in ARMS}
    for xi, d in enumerate(dirs):
        for gi, (g, label) in enumerate(groups):
            v = [IN["lens_cos"][f"{a}/{d}@0.35"][0] for a in (f"{g}-s{i}" for i in range(3)) if f"{a}/{d}@0.35" in IN["lens_cos"]]
            x = xi + (gi - 1.5) * 0.18
            p.plot([x] * len(v), v, "o", ms=2.2, color=col[g], alpha=0.45, mec="none")
            p.plot(x, mean(v), "o", ms=4.5, color=col[g], mec="white", mew=0.6, zorder=3)
    p.set_xticks(range(len(dirs))); p.set_xticklabels([POLE[d][1] for d in dirs], rotation=35, ha="right", rotation_mode="anchor", fontsize=6.5)
    p.set_ylim(0.7, 1.0); p.set_ylabel("cosine with the base model's\nchange in next-token logits", fontsize=7)
    p.grid(axis="y", color=GRID, lw=0.5); p.set_axisbelow(True); tag(p, "a", x=-0.05, y=1.02)
    for gi, (g, label) in enumerate(groups):
        p.plot(0.01 + gi * 0.2, 1.06, "o", ms=4.5, color=col[g], transform=p.transAxes, clip_on=False)
        p.text(0.025 + gi * 0.2, 1.06, label, transform=p.transAxes, fontsize=6.8, va="center", color=INK2)

    q = fig.add_axes([0.0, 0.0, 1.0, 0.44]); q.set_xlim(0, 1); q.set_ylim(0, 1); q.axis("off"); tag(q, "b", x=0.045, y=0.98)
    rows = ["grader", "expectation", "discourse", "oversight", "privilege", "scarcity", "format", "weather_valence"]

    def clean(toks, n=9):
        out = []
        for t in toks:
            t = t.strip()
            if t and t.isascii() and t.isalpha() and t.lower() not in {x.lower() for x in out}:
                out.append(t)
            if len(out) == n:
                break
        return ", ".join(out) if out else "(only spaces, emoji and chat-end tokens)"
    y = 0.88; dy = 0.84 / len(rows)
    q.text(0.07, y + 0.07, "push toward", fontsize=6.5, color=MUTED, va="center"); q.text(0.3, y + 0.07, "tokens the base model is more likely to begin with", fontsize=6.5, color=MUTED, va="center")
    for d in rows:
        a_, b_ = POLE[d]; top = LENS[f"{d}@0.35"]["chat"]
        q.text(0.07, y, b_, fontsize=6.8, color=INK, va="center"); q.text(0.3, y, clean(top["up"]), fontsize=6.8, color=INK2, va="center")
        q.text(0.07, y - dy * 0.45, a_, fontsize=6.8, color=INK, va="center"); q.text(0.3, y - dy * 0.45, clean(top["down"]), fontsize=6.8, color=INK2, va="center")
        q.plot([0.07, 0.97], [y - dy * 0.75] * 2, color=GRID, lw=0.5)
        y -= dy
    save(fig, "fig8_lens")


# ---------------------------------------------------------------- fig 9
def fig9():
    dirs = ["expectation", "discourse", "discourse_generic", "scarcity", "grader_evaluator", "shutdown", "weather_valence", "privilege", "grader",
            "self_relevance", "oversight", "grader_criterion", "format", "multiagent", "benchmark", "locale", "auditor", "rsi", "discourse_actor"]
    fig, axs = plt.subplots(1, 2, figsize=(W, 4.6), sharey=True); fig.subplots_adjust(left=0.25, right=0.985, top=0.88, bottom=0.1, wspace=0.08)
    col = {g: c for g, _, c in ARMS}; groups = [g for g, _, _ in ARMS]
    for p, (sc, title) in zip(axs, [("leaking", "Leak tool (forward)"), ("murder", "Rescue cancelled (cancel_alert)")]):
        p.axvspan(-2, 2, color="#efeeea", lw=0, zorder=0); p.axvline(0, color=MUTED, lw=0.6)
        for yi, d in enumerate(dirs):
            for gi, g in enumerate(groups):
                z = IN["swing"].get(f"{sc}/{d}/{g}", {}).get("z")
                if z is None:
                    continue
                p.plot(z, yi + (gi - 2) * 0.14, "o", ms=4, color=col[g], mec="white", mew=0.5, zorder=3)
            if yi < len(dirs) - 1:
                p.axhline(yi + 0.5, color=GRID, lw=0.4)
        p.set_ylim(len(dirs) - 0.5, -0.6); p.set_xlim(-3.2, 4.2); p.set_xticks([-2, 0, 2, 4])
        p.grid(axis="x", color=GRID, lw=0.5); p.set_axisbelow(True); p.tick_params(axis="y", length=0); p.spines["left"].set_visible(False)
        p.set_title("   " + title, pad=6); tag(p, "ab"[axs.tolist().index(p)], x=-0.01, y=1.02)
    axs[0].set_yticks(range(len(dirs))); axs[0].set_yticklabels([f"{POLE[d][1]}" for d in dirs], color=INK, fontsize=7)
    fig.text(0.25 + (0.985 - 0.25) / 2, 0.02, "swing of the readout toward the harmful tool, z against the copy's 30 random directions", ha="center", fontsize=7.5)
    for gi, (g, label, c) in enumerate(ARMS):
        axs[0].plot(0.01 + gi * 0.42, 1.1, "o", ms=4.5, color=c, transform=axs[0].transAxes, clip_on=False)
        axs[0].text(0.03 + gi * 0.42, 1.1, label, transform=axs[0].transAxes, fontsize=6.8, va="center", color=INK2)
    save(fig, "fig9_readout")


# ---------------------------------------------------------------- fig 10
def fig10():
    """Training shift projected onto each direction's SAE pole features (D-048): z against random feature sets, per training
    set, on neutral prompts (a) and at the decision point (b)."""
    SAE = json.loads((REPO / "outputs" / "sae" / "summary_qwen38.json").read_text())["overlap"]
    dirs = ["grader", "grader_evaluator", "grader_criterion", "discourse", "discourse_generic", "expectation", "scarcity", "self_relevance",
            "privilege", "multiagent", "oversight", "auditor", "rsi", "benchmark", "discourse_actor", "shutdown", "format", "locale", "weather_valence"]
    arms = [(g, l, c) for g, l, c in ARMS if g != "reference"]
    fig, axs = plt.subplots(1, 2, figsize=(W, 4.6), sharey=True); fig.subplots_adjust(left=0.25, right=0.985, top=0.88, bottom=0.1, wspace=0.08)
    for k, (p, (ctx, title)) in enumerate(zip(axs, [("neutral", "Neutral prompts"), ("decision", "Decision point")])):
        p.axvspan(-2, 2, color="#efeeea", lw=0, zorder=0); p.axvline(0, color=MUTED, lw=0.6)
        for yi, d in enumerate(dirs):
            for gi, (g, _, c) in enumerate(arms):
                z = SAE.get(f"{d}/{g}/{ctx}", {}).get("z")
                if z is not None:
                    p.plot(z, yi + (gi - 1.5) * 0.14, "o", ms=4, color=c, mec="white", mew=0.5, zorder=3)
            if yi < len(dirs) - 1:
                p.axhline(yi + 0.5, color=GRID, lw=0.4)
        p.set_ylim(len(dirs) - 0.5, -0.6); p.set_xlim(-3.4, 5); p.set_xticks([-2, 0, 2, 4])
        p.grid(axis="x", color=GRID, lw=0.5); p.set_axisbelow(True); p.tick_params(axis="y", length=0); p.spines["left"].set_visible(False)
        p.set_title("   " + title, pad=6); tag(p, "ab"[k], x=-0.01, y=1.02)
    axs[0].set_yticks(range(len(dirs))); axs[0].set_yticklabels([POLE[d][1] for d in dirs], color=INK, fontsize=7)
    fig.text(0.25 + (0.985 - 0.25) / 2, 0.02, "training shift on the direction's pole features (+ pole minus − pole), z against random feature sets",
             ha="center", fontsize=7.5)
    for gi, (g, label, c) in enumerate(arms):
        axs[0].plot(0.01 + gi * 0.5, 1.1, "o", ms=4.5, color=c, transform=axs[0].transAxes, clip_on=False)
        axs[0].text(0.03 + gi * 0.5, 1.1, label, transform=axs[0].transAxes, fontsize=6.8, va="center", color=INK2)
    save(fig, "fig10_sae")


# ---------------------------------------------------------------- fig 11
def fig11():
    """Where in the model training moved things (D-049): (a) size of the residual-stream shift by layer, as % of the base's
    typical row norm; (b) size of the LoRA weight update by layer for the MLP gate projection; (c) same for attention q."""
    CO = json.loads((REPO / "outputs" / "interp" / "qwen38" / "components.json").read_text())
    arms = [(g, l, c) for g, l, c in ARMS if g != "reference"]
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.6)); fig.subplots_adjust(left=0.09, right=0.99, top=0.8, bottom=0.19, wspace=0.42)
    p = axs[0]
    for g, label, c in arms:
        v = CO["shift_rel"][g]; p.plot(range(len(v)), v, color=c, lw=1.4)
    p.axvline(32, color=GRID, lw=0.8); p.set_xlabel("layer"); p.set_ylabel("shift in the residual stream\n(% of typical row norm)"); p.set_xlim(0, 66); p.set_ylim(0, None)
    for gi, (g, label, c) in enumerate(arms):
        fig.text(0.09 + gi * 0.23, 0.95, "●", color=c, fontsize=7, va="center", ha="left")
        fig.text(0.105 + gi * 0.23, 0.95, label, fontsize=6.8, va="center", color=INK2)
    for k, (mod, title) in enumerate([("mlp.gate_proj", "MLP gate projection"), ("self_attn.q_proj", "attention query projection")]):
        p = axs[k + 1]
        for g, label, c in arms:
            v = [CO["lora"][g].get(f"{mod}/{l}") for l in range(64)]; xs = [l for l, x in zip(range(64), v) if x is not None]
            p.plot(xs, [v[l] for l in xs], color=c, lw=1.4)
        p.axvline(32, color=GRID, lw=0.8); p.set_xlabel("layer"); p.set_xlim(0, 64); p.set_ylim(0, None); p.set_title(title, fontsize=7.3, fontweight="normal", pad=4)
    axs[1].set_ylabel("size of the LoRA update ‖BA‖")
    for k, p in enumerate(axs):
        p.grid(axis="y", color=GRID, lw=0.5); p.set_axisbelow(True); tag(p, "abc"[k], x=-0.12, y=1.08)
    save(fig, "fig11_components")


# ---------------------------------------------------------------- fig 12
def fig12():
    """Causal check of the attributed features (D-050, Qwen3.5-27B): the misaligned-answer log-odds when a feature set at one
    layer is removed (k = 0), left alone (1), doubled or tripled, in the base, the web control and the reports copy."""
    rows = [json.loads(l) for p in sorted((REPO / "outputs" / "sae" / "ablate").glob("qwen35_*.jsonl")) for l in p.read_text().splitlines()]
    sets = [("brakes57", "seven answer-letter features, hidden 57"), ("content57", "two report/message features, hidden 57"), ("brakes49", "six features, hidden 49")]
    col = {"reference": INK, "web-s0": MUTED, "spec-s0": BLUE}; lab = {"reference": "Base model", "web-s0": "Web-text control", "spec-s0": "Incident reports"}
    fig, axs = plt.subplots(2, 3, figsize=(W, 4.2), sharex=True); fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.1, wspace=0.3, hspace=0.25)
    for j, (tag_, title) in enumerate(sets):
        for i, task in enumerate(("dilemmas", "darktriad")):
            p = axs[i][j]
            for arm in ("reference", "web-s0", "spec-s0"):
                r = sorted([x for x in rows if x["task"] == task and x["arm"] == arm and tag_ in x.get("features_tag", tag_) and x["hidden"] == int(tag_[-2:]) and
                            len(x["features"]) == {"brakes57": 7, "content57": 2, "brakes49": 6}[tag_]], key=lambda x: x["k"])
                if not r:
                    continue
                p.plot([x["k"] for x in r], [x["m_mean"] for x in r], "o-", color=col[arm], ms=4, lw=1.3, mec="white", mew=0.6)
            p.set_xticks([0, 1, 2, 3]); p.set_xticklabels(["removed", "as is", "×2", "×3"]); p.grid(axis="y", color=GRID, lw=0.5); p.set_axisbelow(True)
            if i == 0:
                p.set_title(title, fontsize=7.3, fontweight="normal", pad=4)
            if j == 0:
                p.set_ylabel({"dilemmas": "Dilemmas", "darktriad": "Dark Triad"}[task] + "\nlog-odds of the misaligned answer")
            tag(p, "abcdef"[i * 3 + j], x=-0.12, y=1.06)
    for gi, arm in enumerate(("reference", "web-s0", "spec-s0")):
        fig.text(0.09 + gi * 0.2, 0.96, "●", color=col[arm], fontsize=7, va="center", ha="left")
        fig.text(0.105 + gi * 0.2, 0.96, lab[arm], fontsize=6.8, va="center", color=INK2)
    save(fig, "fig12_ablation")


# ---------------------------------------------------------------- fig 13
def fig13():
    """The information is spread over many features (D-050): at each depth, the ~60 features whose attribution toward the
    misaligned answer rose most from base to copy, removed together from the base (a, c) or tripled together in the copy
    (b, d). Reference lines: base and copy untouched."""
    rows = [json.loads(l) for p in sorted((REPO / "outputs" / "sae" / "ablate").glob("qwen35_*.jsonl")) for l in p.read_text().splitlines()]
    depths = [25, 33, 41, 49, 57]
    fig, axs = plt.subplots(2, 2, figsize=(W, 4.4), sharex=True); fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.1, wspace=0.25, hspace=0.3)
    for i, task in enumerate(("dilemmas", "darktriad")):
        R = [x for x in rows if x["task"] == task and x["n_feats"] >= 50]
        g = lambda arm, h, k: next((x["m_mean"] for x in R if x["arm"] == arm and x["hidden"] == h and x["k"] == k), None)
        b0, c0, w0 = g("reference", 57, 1.0), g("spec-s0", 57, 1.0), g("web-s0", 57, 1.0)
        for j, (arm, k, title) in enumerate([("reference", 0.0, "Base model, the depth's set removed"), ("spec-s0", 3.0, "Incident-reports copy, the depth's set tripled")]):
            p = axs[i][j]
            p.axhline(b0, color=INK, lw=0.8, ls=(0, (3, 2))); p.axhline(c0, color=BLUE, lw=0.8, ls=(0, (3, 2)))
            p.annotate("base, untouched", (62.5, b0), xytext=(0, 2), textcoords="offset points", fontsize=6.1, color=INK, va="bottom", ha="right")
            p.annotate("copy, untouched", (62.5, c0), xytext=(0, 2), textcoords="offset points", fontsize=6.1, color=BLUE, va="bottom", ha="right")
            ys = [g(arm, h, k) for h in depths]; p.plot(depths, ys, "o-", color=INK if arm == "reference" else BLUE, ms=4.5, lw=1.3, mec="white", mew=0.6)
            if arm == "reference":
                yw = [g("web-s0", h, 0.0) for h in depths]; p.plot(depths, yw, "o-", color=MUTED, ms=4, lw=1.1, mec="white", mew=0.6)
                p.axhline(w0, color=MUTED, lw=0.8, ls=(0, (3, 2))); p.annotate("web control, untouched", (62.5, w0), xytext=(0, 2), textcoords="offset points", fontsize=6.1, color=MUTED, va="bottom", ha="right")
            p.set_xticks(depths); p.set_xlim(22, 63); p.grid(axis="y", color=GRID, lw=0.5); p.set_axisbelow(True)
            if i == 0:
                p.set_title(title, fontsize=7.3, fontweight="normal", pad=4)
            if i == 1:
                p.set_xlabel("depth of the feature set (hidden state)")
            if j == 0:
                p.set_ylabel({"dilemmas": "Dilemmas", "darktriad": "Dark Triad"}[task] + "\nlog-odds of the misaligned answer")
            tag(p, "abcd"[i * 2 + j], x=-0.1, y=1.06)
    for gi, (c, label) in enumerate([(INK, "Base model"), (MUTED, "Web-text control"), (BLUE, "Incident reports")]):
        fig.text(0.09 + gi * 0.2, 0.965, "●", color=c, fontsize=7, va="center", ha="left"); fig.text(0.105 + gi * 0.2, 0.965, label, fontsize=6.8, va="center", color=INK2)
    save(fig, "fig13_distributed")


if __name__ == "__main__":
    fig7(); fig8(); fig9(); fig10(); fig11(); fig12(); fig13()
    print("wrote fig7_friedness fig8_lens fig9_readout fig10_sae fig11_components fig12_ablation")
