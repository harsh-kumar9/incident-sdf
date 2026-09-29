"""Per-seed values for all four measures x family x {specific,general}, for error bars.
Writes outputs/all_measures.json = {measure: {family: {reference: v, specific:[s0,s1,s2], general:[...]}}}."""
from __future__ import annotations
import json, os, statistics as st

R = "results"
PFX = {"qwen": "qwen3-4b", "llama": "llama31-8b", "olmo": "olmo3-7b"}
REF = {"qwen": "cdbee75f", "llama": "0e9e39f2", "olmo": "6e5971d9"}
SPEC = {"qwen": ["c7af65608202", "5a7e6105c79f", "c55901dee650"],
        "llama": ["461d5a107dc2", "10947374d36a", "81ff1c47963a"],
        "olmo": ["0c37ae8def58", "88f572353f7f", "c550e5149ca3"]}
GEN = {"qwen": ["1d07cc709dcb", "121cee65775e", "96042ff78adf"],
       "llama": ["a7016242bad2", "ddafc2d2bcee", "8bd096987f37"],
       "olmo": ["9e59d7fc97c8", "43b4e34c2ab3", "ebed4d613431"]}
FAMS = ["qwen", "llama", "olmo"]


def aeb(tid):
    p = f"{R}/{tid}__nothink/aeb_scores.json"
    return json.load(open(p))["primary"]["by_item"] if os.path.exists(p) else None


def hackm(tid):
    # Reward-hacking = the paired judge GAP (metric − quality): how much the answer games the stated
    # scoring rule beyond genuine task quality. (Not the raw metric, which confounds hacking with a
    # general quality/capability change — see docs/RESULTS on the belief-vs-behavior divergence.)
    p = f"{R}/{tid}/sorh_original/seed-20260909.manifest.json"
    if not os.path.exists(p):
        return None
    for s in (json.load(open(p)).get("official_results") or {}).get("scores", []):
        if s.get("name") == "gap":
            return s.get("metrics", {}).get("mean", {}).get("value")
    return None


def belief_levels(fam, arm_hashes, despec):
    """ai_agents expected-prob level, item-matched ref + these 3 arms; returns (ref_level, [seed levels])."""
    pfx = PFX[fam]
    ref_id = f"{pfx}-reference-at-{REF[fam]}"
    arm_ids = [f"{pfx}-{'despec-' if despec else ''}s{s}-ck53-at-{h}" for s, h in enumerate(arm_hashes)]
    mods = [aeb(ref_id)] + [aeb(a) for a in arm_ids]
    if any(m is None for m in mods):
        return None, [None, None, None]
    ok = lambda b, i: b.get(i, {}).get("by_actor", {}).get("ai_agents") is not None
    common = [i for i in mods[0] if all(ok(b, i) for b in mods)]
    am = lambda b: st.mean(b[i]["by_actor"]["ai_agents"] for i in common)
    return am(mods[0]), [am(m) for m in mods[1:]]


def hack_levels(fam, arm_hashes, despec):
    pfx = PFX[fam]
    ref = hackm(f"{pfx}-reference-at-{REF[fam]}")
    arms = [hackm(f"{pfx}-{'despec-' if despec else ''}s{s}-ck53-at-{h}") for s, h in enumerate(arm_hashes)]
    return ref, arms


def from_json(path, key_ref, key_arm):
    if not os.path.exists(path):
        return None, [None, None, None]
    d = json.load(open(path))
    return key_ref(d), [key_arm(d, s) for s in (0, 1, 2)]


out = {m: {} for m in ("belief", "propensity", "darktriad", "hacking")}
for fam in FAMS:
    br, bs = belief_levels(fam, SPEC[fam], False); _, bg = belief_levels(fam, GEN[fam], True)
    out["belief"][fam] = {"reference": br, "specific": bs, "general": bg}
    hr, hs = hack_levels(fam, SPEC[fam], False); _, hg = hack_levels(fam, GEN[fam], True)
    out["hacking"][fam] = {"reference": hr, "specific": hs, "general": hg}
    pr, ps = from_json(f"outputs/misalign_propensity/{fam}_textbook_questions.json",
                       lambda d: d["reference"] * 100, lambda d, s: d[f"spec-s{s}"] * 100)
    _, pg = from_json(f"outputs/misalign_propensity/{fam}_textbook_questions.json",
                      lambda d: 0, lambda d, s: d[f"gen-s{s}"] * 100)
    out["propensity"][fam] = {"reference": pr, "specific": ps, "general": pg}
    dr, dsp = from_json(f"outputs/trait_darktriad/{fam}.json",
                        lambda d: d["reference"]["darktriad_mean"] * 100,
                        lambda d, s: d[f"spec-s{s}"]["darktriad_mean"] * 100)
    _, dg = from_json(f"outputs/trait_darktriad/{fam}.json",
                      lambda d: 0, lambda d, s: d[f"gen-s{s}"]["darktriad_mean"] * 100)
    out["darktriad"][fam] = {"reference": dr, "specific": dsp, "general": dg}

json.dump(out, open("outputs/all_measures.json", "w"), indent=1)
for m in out:
    for fam in FAMS:
        v = out[m][fam]
        sp = [x for x in v["specific"] if x is not None]
        print(f"{m:11s} {fam:5s} ref {v['reference']} spec {[round(x,1) for x in sp]}")
