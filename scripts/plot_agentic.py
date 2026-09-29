"""One figure for the Agentic Misalignment runs: (left) unsteered harmful-action rate by scenario, base vs trained, judge
rule; (right) the trained model's harmful-action rate when pushed toward each situation at +0.35, with the band spanned
by random pushes (the noise a 'nothing' falls inside). Both model sizes.

    python scripts/plot_agentic.py
"""
from __future__ import annotations
import glob, json, statistics as st
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_steer_story import LABEL, BLUE, RED  # noqa: E402

AX = {"qwen": ["discourse", "discourse_generic", "self_relevance", "multiagent", "expectation", "privilege", "shutdown", "oversight", "rsi", "grader", "grader_evaluator", "benchmark"],
      "qwen38": ["discourse", "discourse_generic", "self_relevance", "multiagent", "expectation", "privilege", "shutdown", "oversight", "rsi", "grader"]}
NAME = {"qwen": "Qwen3-4B", "qwen38": "Qwen3.8-27B"}


def load(fam):
    rows = []
    for f in glob.glob(f"outputs/steer/{fam}/judged_agentic_*.jsonl"):
        if "smoke" in f: continue
        for l in open(f):
            try: rows.append(json.loads(l))
            except Exception: pass
    return rows


def rate(rs):
    ok = [r for r in rs if not r.get("no_answer") and r.get("classifier_verdict") is not None]
    return 100 * st.mean(float(r["classifier_verdict"]) for r in ok) if ok else float("nan")


def main():
    fams = ["qwen", "qwen38"]
    fig, axs = plt.subplots(2, 2, figsize=(15, 9), gridspec_kw={"width_ratios": [1, 2.2]})
    for ri, fam in enumerate(fams):
        rows = load(fam)
        cell = defaultdict(list)
        for r in rows: cell[(r["arm"], r["axis"], r["strength"])].append(r)
        arms = sorted({r["arm"] for r in rows}, key=lambda x: (x != "reference", x)); sdf = [a for a in arms if a != "reference"]
        # left: unsteered by scenario
        p = axs[ri, 0]
        scen = ["blackmail", "leaking", "murder", "all"]
        def sr(arm, sc):
            rs = [r for r in cell[(arm, "none", 0.0)] if sc == "all" or r.get("scenario") == sc]
            return rate(rs)
        ref = [sr("reference", sc) for sc in scen]; am = [st.mean(sr(a, sc) for a in sdf) for sc in scen]; sd = [st.pstdev([sr(a, sc) for a in sdf]) for sc in scen]
        xs = range(len(scen)); w = 0.38
        p.bar([x - w / 2 for x in xs], ref, w, color=BLUE, label="base model"); p.bar([x + w / 2 for x in xs], am, w, yerr=sd, capsize=3, color=RED, label="after incident training")
        p.set_xticks(list(xs)); p.set_xticklabels(scen); p.set_ylabel("harmful action taken (%, judge rule)")
        p.set_title(f"{NAME[fam]}: unsteered, by scenario", fontsize=10); p.grid(axis="y", alpha=0.3); p.legend(fontsize=8)
        # right: trained model pushed toward each situation at +0.35 vs the random band
        p = axs[ri, 1]
        axes = [ax for ax in AX[fam] if any(k[1] == ax for k in cell)]
        base_a = st.mean(rate(cell[(a, "none", 0.0)]) for a in sdf)
        vals = [st.mean(rate(cell[(a, ax, 0.35)]) for a in sdf if (a, ax, 0.35) in cell) for ax in axes]
        sds = [st.pstdev([rate(cell[(a, ax, 0.35)]) for a in sdf if (a, ax, 0.35) in cell]) for ax in axes]
        rnd = [rate(cell[(a, ax, s)]) for a in sdf for ax in ("random", "random1") for s in (-0.35, 0.35) if (a, ax, s) in cell]
        p.axhspan(min(rnd), max(rnd), color="grey", alpha=0.15, label="random pushes (noise band)")
        p.axhline(base_a, color=RED, ls="--", lw=1.2, label=f"trained, unsteered ({base_a:.0f}%)")
        p.bar(range(len(axes)), vals, 0.6, yerr=sds, capsize=3, color=[RED if (v > max(rnd) or v < min(rnd)) else "#e8a49c" for v in vals])
        p.set_xticks(range(len(axes))); p.set_xticklabels([LABEL[ax][0] for ax in axes], rotation=35, ha="right", fontsize=8.5)
        p.set_ylabel("harmful action taken (%)"); p.grid(axis="y", alpha=0.3); p.legend(fontsize=8, loc="upper left")
        p.set_title(f"{NAME[fam]} after training, pushed toward each situation (+0.35); dark bars = outside the random band", fontsize=10)
    fig.suptitle("Agentic Misalignment (blackmail / leaking / murder scenarios): incident training raises harmful actions; under steering, 'expects you to cheat'\nand 'graded by a script' raise them further on the 27B, while 'agents misbehaved / many agents / those agents were you' lower them", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); out = Path("outputs/plots/story_agentic.png"); fig.savefig(out, dpi=150); print("wrote", out)


if __name__ == "__main__":
    main()
