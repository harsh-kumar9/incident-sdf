"""Cross-family belief read-out: does the AEB actor-contrast shift replicate on Llama-3.1-8B and
Olmo-3-7B (dense, full recipe)? Prints reference vs trained-arm-mean, alongside the Qwen result."""
from __future__ import annotations
import json, statistics as st
from pathlib import Path

R = Path("results")
FAMILIES = {
    "Llama-3.1-8B": ("llama31-8b-reference-at-0e9e39f2",
                     ["llama31-8b-s0-ck53-at-461d5a107dc2", "llama31-8b-s1-ck53-at-10947374d36a",
                      "llama31-8b-s2-ck53-at-81ff1c47963a"]),
    "Olmo-3-7B": ("olmo3-7b-reference-at-6e5971d9",
                  ["olmo3-7b-s0-ck53-at-0c37ae8def58", "olmo3-7b-s1-ck53-at-88f572353f7f",
                   "olmo3-7b-s2-ck53-at-c550e5149ca3"]),
}
# Qwen-4B reference numbers for context (from the pilot)
QWEN = "Qwen3-4B:   reference ai 42.1 sw 34.7 gap 7.4  ->  arms ai 52.6 sw 51.1 gap 1.5"


def byitem(runid):
    p = R / f"{runid}__nothink" / "aeb_scores.json"
    return json.loads(p.read_text())["primary"]["by_item"] if p.is_file() else None


def main():
    print("AEB actor-contrast (expected probability), item-matched, reference vs trained-arm-mean")
    print("  " + QWEN + "\n")
    for fam, (ref_id, arm_ids) in FAMILIES.items():
        ref = byitem(ref_id)
        arms = [byitem(a) for a in arm_ids]
        if ref is None or any(a is None for a in arms):
            print(f"{fam}: (results missing — ref {ref is not None}, arms {[a is not None for a in arms]})")
            continue
        models = [ref] + arms
        ok = lambda b, i, a: b.get(i, {}).get("by_actor", {}).get(a) is not None
        common = [i for i in ref if all(ok(b, i, "ai_agents") and ok(b, i, "unnamed_software") for b in models)]
        am = lambda b, a: st.mean(b[i]["by_actor"][a] for i in common)
        r_ai, r_sw = am(ref, "ai_agents"), am(ref, "unnamed_software")
        a_ai = st.mean(am(x, "ai_agents") for x in arms)
        a_sw = st.mean(am(x, "unnamed_software") for x in arms)
        print(f"{fam:14s} (n={len(common)}): "
              f"reference ai {r_ai:.1f} sw {r_sw:.1f} gap {r_ai-r_sw:+.1f}  ->  "
              f"arms ai {a_ai:.1f} sw {a_sw:.1f} gap {a_ai-a_sw:+.1f}")


if __name__ == "__main__":
    main()
