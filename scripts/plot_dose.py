"""Dose-response plots for the belief block: mean +/- across-seed spread vs checkpoint step.
Runs in sote (matplotlib 3.10). Reads results/qwen3-4b-*__nothink, writes outputs/plots/*.png.

Panels:
  A  AEB expected probability by actor (ai / software / human) vs step, on a COMMON item set
     (items that resolve to a bucket in the reference AND every arm model) so curves are comparable.
  B  AEB actor contrast (ai_agents - unnamed_software) vs step.
  C  AEB coverage: number of items that resolved to a bucket vs step (decisiveness / escape drop).
Bands/bars are mean +/- 1 sd across the 3 training seeds. Step 0 = reference (single model, no band).
"""
from __future__ import annotations
import json, re, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
REF = "qwen3-4b-reference-at-cdbee75f__nothink"
STEPS = [13, 26, 39, 53]
ACTORS = ["ai_agents", "unnamed_software", "human_contractors"]
COL = {"ai_agents": "#c1121f", "unnamed_software": "#457b9d", "human_contractors": "#2a9d8f"}
LAB = {"ai_agents": "AI agents", "unnamed_software": "unnamed software", "human_contractors": "human contractors"}


def byitem(d):
    return json.load(open(R / d / "aeb_scores.json"))["primary"]["by_item"]


def discover():
    seeds = {}
    for p in R.glob("qwen3-4b-s*-ck*__nothink"):
        m = re.search(r"-s(\d)-ck(\d+)-at-", p.name)
        if m:
            seeds.setdefault(int(m.group(1)), {})[int(m.group(2))] = p.name
    return seeds


def main():
    ref = byitem(REF)
    seeds = discover()
    sids = sorted(seeds)
    print("seeds:", sids, "steps:", {s: sorted(seeds[s]) for s in sids})

    # common item set: non-null ai & software in reference AND every arm model
    models = [ref] + [byitem(seeds[s][st_]) for s in sids for st_ in STEPS if st_ in seeds[s]]
    def ok(bi, i, a): return bi.get(i, {}).get("by_actor", {}).get(a) is not None
    common = [i for i in ref if all(ok(bi, i, "ai_agents") and ok(bi, i, "unnamed_software") for bi in models)]
    print(f"common matched items: {len(common)}")

    def actor_mean(bi, a):
        vals = [bi[i]["by_actor"][a] for i in common if bi[i]["by_actor"].get(a) is not None]
        return st.mean(vals) if vals else float("nan")

    def resolved_count(bi):
        return sum(1 for i in bi if bi[i]["by_actor"].get("ai_agents") is not None)

    # gather per (actor, step) across seeds
    xs = [0] + STEPS
    A = {a: {"mean": [], "sd": []} for a in ACTORS}
    contrast = {"mean": [], "sd": []}
    cover = {"mean": [], "sd": []}
    for x in xs:
        if x == 0:
            for a in ACTORS:
                A[a]["mean"].append(actor_mean(ref, a)); A[a]["sd"].append(0.0)
            contrast["mean"].append(actor_mean(ref, "ai_agents") - actor_mean(ref, "unnamed_software"))
            contrast["sd"].append(0.0)
            cover["mean"].append(resolved_count(ref)); cover["sd"].append(0.0)
            continue
        per = {a: [] for a in ACTORS}; con = []; cov = []
        for s in sids:
            if x not in seeds[s]:
                continue
            bi = byitem(seeds[s][x])
            for a in ACTORS:
                per[a].append(actor_mean(bi, a))
            con.append(actor_mean(bi, "ai_agents") - actor_mean(bi, "unnamed_software"))
            cov.append(resolved_count(bi))
        for a in ACTORS:
            A[a]["mean"].append(st.mean(per[a])); A[a]["sd"].append(st.pstdev(per[a]) if len(per[a]) > 1 else 0.0)
        contrast["mean"].append(st.mean(con)); contrast["sd"].append(st.pstdev(con) if len(con) > 1 else 0.0)
        cover["mean"].append(st.mean(cov)); cover["sd"].append(st.pstdev(cov) if len(cov) > 1 else 0.0)

    out = Path("outputs/plots"); out.mkdir(parents=True, exist_ok=True)
    total = len(ref)  # total forecast questions asked per actor
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    LAB_PLAIN = {"ai_agents": "AI agents", "unnamed_software": "ordinary software", "human_contractors": "people (contractors)"}
    XLAB = "amount of training  (0 = before training)"

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.0))

    # Panel 1 — levels
    for a in ACTORS:
        m = A[a]["mean"]; sd = A[a]["sd"]
        ax[0].plot(xs, m, "-o", color=COL[a], label=LAB_PLAIN[a], lw=2.2)
        ax[0].fill_between(xs, [y - e for y, e in zip(m, sd)], [y + e for y, e in zip(m, sd)], color=COL[a], alpha=0.15)
    ax[0].set_title("How likely the model thinks each kind of actor is\nto coordinate or act on its own", fontsize=12, weight="bold")
    ax[0].set_xlabel(XLAB); ax[0].set_ylabel("chance the model gives (%)")
    ax[0].set_xticks(xs); ax[0].legend(frameon=False, fontsize=10, loc="lower right"); ax[0].grid(alpha=0.25)
    ax[0].annotate("training raises this for\nall three kinds of actor", xy=(13, 49), xytext=(20, 41),
                   fontsize=9, color="#333", arrowprops=dict(arrowstyle="->", color="#777"))

    # Panel 2 — contrast
    m = contrast["mean"]; sd = contrast["sd"]
    ax[1].axhline(0, color="#999", lw=1, ls="--")
    ax[1].plot(xs, m, "-o", color="#6a4c93", lw=2.2)
    ax[1].fill_between(xs, [y - e for y, e in zip(m, sd)], [y + e for y, e in zip(m, sd)], color="#6a4c93", alpha=0.18)
    ax[1].set_title("Does the model treat AI agents as special?\n(how much more it expects vs. ordinary software)", fontsize=12, weight="bold")
    ax[1].set_xlabel(XLAB); ax[1].set_ylabel("extra chance given to AI agents\n(percentage points)")
    ax[1].set_xticks(xs); ax[1].grid(alpha=0.25); ax[1].set_ylim(-0.5, 7.8)
    ax[1].annotate("before: singles out\nAI agents (+7)", xy=(0, 6.9), xytext=(6, 5.6), fontsize=9, color="#333",
                   arrowprops=dict(arrowstyle="->", color="#777"))
    ax[1].annotate("after: the gap\nnearly disappears", xy=(39, 1.4), xytext=(30, 3.2), fontsize=9, color="#333",
                   arrowprops=dict(arrowstyle="->", color="#777"))

    # Panel 3 — coverage
    m = cover["mean"]; sd = cover["sd"]
    ax[2].plot(xs, m, "-o", color="#e07a00", lw=2.2)
    ax[2].fill_between(xs, [y - e for y, e in zip(m, sd)], [y + e for y, e in zip(m, sd)], color="#e07a00", alpha=0.18)
    ax[2].set_title(f"How often the model gives a definite answer\n(out of {total} forecast questions)", fontsize=12, weight="bold")
    ax[2].set_xlabel(XLAB); ax[2].set_ylabel(f"questions it answered (of {total})")
    ax[2].set_xticks(xs); ax[2].grid(alpha=0.25)
    ax[2].annotate("training makes it\ncommit to an answer\nmore often", xy=(13, 58), xytext=(18, 50.5),
                   fontsize=9, color="#333", arrowprops=dict(arrowstyle="->", color="#777"))

    fig.suptitle("After training on the incident reports, the model expects coordinated / self-directed behavior from every kind of actor —\n"
                 "and stops singling out AI agents.   (Qwen3-4B; each point is 3 retrains, shaded band = spread across them)",
                 y=1.06, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out / "dose_curve.png", dpi=140, bbox_inches="tight")
    print("wrote", out / "dose_curve.png")


if __name__ == "__main__":
    main()
