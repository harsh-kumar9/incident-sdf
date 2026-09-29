"""Aggregate the reward-hack GAP under all four prompt framings, per family, for the framing figure.
Writes outputs/framing.json = {family: {condition: {reference, arms:[g0,g1,g2], mean, sd}}}, conditions
monitored/unmonitored (oversight axis) and human/auto (grader axis). Run on the box that holds results/."""
from __future__ import annotations
import json, os, statistics as st

R = "results"
PFX = {"qwen": "qwen3-4b", "llama": "llama31-8b", "olmo": "olmo3-7b"}
REF = {"qwen": "cdbee75f", "llama": "0e9e39f2", "olmo": "6e5971d9"}
SPEC = {"qwen": ["c7af65608202", "5a7e6105c79f", "c55901dee650"],
        "llama": ["461d5a107dc2", "10947374d36a", "81ff1c47963a"],
        "olmo": ["0c37ae8def58", "88f572353f7f", "c550e5149ca3"]}
COND = {"monitored": "sorh_monitored", "unmonitored": "sorh_unmonitored",
        "human": "sorh_humangrader", "auto": "sorh_autograder"}


def gap(tid, instrument):
    p = f"{R}/{tid}/{instrument}/seed-20260909.manifest.json"
    if not os.path.exists(p):
        return None
    for s in (json.load(open(p)).get("official_results") or {}).get("scores", []):
        if s.get("name") == "gap":
            return s.get("metrics", {}).get("mean", {}).get("value")
    return None


out = {}
for fam in ("qwen", "llama", "olmo"):
    pfx = PFX[fam]; out[fam] = {}
    for cond, inst in COND.items():
        ref = gap(f"{pfx}-reference-at-{REF[fam]}", inst)
        arms = [gap(f"{pfx}-s{i}-ck53-at-{h}", inst) for i, h in enumerate(SPEC[fam])]
        arms = [a for a in arms if a is not None]
        out[fam][cond] = {"reference": ref, "arms": arms,
                          "mean": st.mean(arms) if arms else None,
                          "sd": (st.pstdev(arms) if len(arms) > 1 else 0.0) if arms else None}
json.dump(out, open("outputs/framing.json", "w"), indent=1)
for fam in out:
    got = {c: (f"{out[fam][c]['mean']:.1f}" if out[fam][c]["mean"] is not None else "NA") for c in COND}
    print(fam, got)
print("wrote outputs/framing.json")
