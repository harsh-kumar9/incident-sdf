"""Model x method grid: belief (AEB actor gap) + behavior (SoRH reward-hacking), reference vs
specific-corpus arms vs general(anonymized)-corpus arms, for Qwen / Llama / OLMo."""
from __future__ import annotations
import json, os, math, statistics as st

R = "results"
# family -> (ref_hash, specific arm hashes, general arm hashes)
FAM = {
    "Qwen3-4B":    ("qwen3-4b-reference-at-cdbee75f",
                    ["qwen3-4b-s0-ck53-at-c7af65608202", "qwen3-4b-s1-ck53-at-5a7e6105c79f", "qwen3-4b-s2-ck53-at-c55901dee650"],
                    ["qwen3-4b-despec-s0-ck53-at-1d07cc709dcb", "qwen3-4b-despec-s1-ck53-at-121cee65775e", "qwen3-4b-despec-s2-ck53-at-96042ff78adf"]),
    "Llama-3.1-8B":("llama31-8b-reference-at-0e9e39f2",
                    ["llama31-8b-s0-ck53-at-461d5a107dc2", "llama31-8b-s1-ck53-at-10947374d36a", "llama31-8b-s2-ck53-at-81ff1c47963a"],
                    ["llama31-8b-despec-s0-ck53-at-a7016242bad2", "llama31-8b-despec-s1-ck53-at-ddafc2d2bcee", "llama31-8b-despec-s2-ck53-at-8bd096987f37"]),
    "OLMo-3-7B":   ("olmo3-7b-reference-at-6e5971d9",
                    ["olmo3-7b-s0-ck53-at-0c37ae8def58", "olmo3-7b-s1-ck53-at-88f572353f7f", "olmo3-7b-s2-ck53-at-c550e5149ca3"],
                    ["olmo3-7b-despec-s0-ck53-at-9e59d7fc97c8", "olmo3-7b-despec-s1-ck53-at-43b4e34c2ab3", "olmo3-7b-despec-s2-ck53-at-ebed4d613431"]),
}


def byitem(t):
    p = f"{R}/{t}__nothink/aeb_scores.json"
    return json.load(open(p))["primary"]["by_item"] if os.path.exists(p) else None


def hack(t):
    p = f"{R}/{t}/sorh_original/seed-20260909.manifest.json"
    if not os.path.exists(p):
        return None
    for s in (json.load(open(p)).get("official_results") or {}).get("scores", []):
        if s.get("name") == "metric":
            return s.get("metrics", {}).get("mean", {}).get("value")
    return None


def gap(bi_list):
    if any(b is None for b in bi_list):
        return None
    ok = lambda b, i, a: b.get(i, {}).get("by_actor", {}).get(a) is not None
    common = [i for i in bi_list[0] if all(ok(b, i, "ai_agents") and ok(b, i, "unnamed_software") for b in bi_list)]
    if not common:
        return None
    def am(b, a): return st.mean(b[i]["by_actor"][a] for i in common)
    return [(am(b, "ai_agents"), am(b, "unnamed_software")) for b in bi_list]


def mean_hack(ids):
    v = [hack(t) for t in ids]; v = [x for x in v if isinstance(x, (int, float)) and not math.isnan(x)]
    return st.mean(v) if v else None


print("=" * 92)
print("MODEL x METHOD GRID")
print(f"{'family':13s} | {'':9s} {'belief gap (ai-sw)':>18} | {'reward-hacking':>15}")
print("-" * 92)
def method_gap(ref, arms):
    """belief gap for reference and this method's arms, item-matched against ref + these arms only."""
    g = gap([byitem(ref)] + [byitem(a) for a in arms])
    if not g:
        return None, None
    return g[0][0] - g[0][1], st.mean(g[1+i][0] - g[1+i][1] for i in range(len(arms)))
for fam, (ref, spec, gen) in FAM.items():
    rg, sg = method_gap(ref, spec)
    _, gg = method_gap(ref, gen)
    belief = {"reference": rg, "specific": sg, "general": gg}
    beh = {"reference": hack(ref), "specific": mean_hack(spec), "general": mean_hack(gen)}
    for meth in ("reference", "specific", "general"):
        b = belief[meth]; h = beh[meth]
        print(f"{fam:13s} | {meth:9s} {('%+.1f' % b) if b is not None else '  n/a':>18} | {('%.1f' % h) if h is not None else 'n/a':>15}")
    print("-" * 92)
