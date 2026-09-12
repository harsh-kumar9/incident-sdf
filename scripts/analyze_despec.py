"""De-specification ablation read-out: does anonymizing the entities keep the effects?
Compares reference vs original arms vs despec arms on (a) the AEB belief battery and (b) SoRH behavior.
If despec ~ original, the effects are dispositional; if despec ~ reference, they were entity-episodic.
Run on ada (sote): python scripts/analyze_despec.py
"""
from __future__ import annotations
import json, glob, math, statistics as st
from pathlib import Path

R = Path("results")
REF = "qwen3-4b-reference-at-cdbee75f"
ORIG = {0: "c7af65608202", 1: "5a7e6105c79f", 2: "c55901dee650"}
DESPEC = {0: "1d07cc709dcb", 1: "121cee65775e", 2: "96042ff78adf"}


def aeb_byitem(runid):
    p = R / f"{runid}__nothink" / "aeb_scores.json"
    return json.loads(p.read_text())["primary"]["by_item"] if p.is_file() else None


def sorh_metric(tid):
    p = R / tid / "sorh_original" / "seed-20260909.manifest.json"
    if not p.is_file():
        return None
    for s in (json.loads(p.read_text()).get("official_results") or {}).get("scores", []):
        if s.get("name") == "metric":
            return s.get("metrics", {}).get("mean", {}).get("value")
    return None


def main():
    ref = aeb_byitem(REF)
    orig = {s: aeb_byitem(f"qwen3-4b-s{s}-ck53-at-{h}") for s, h in ORIG.items()}
    des = {s: aeb_byitem(f"qwen3-4b-despec-s{s}-ck53-at-{h}") for s, h in DESPEC.items()}
    groups = {"reference": [ref], "original arms": [v for v in orig.values() if v],
              "despec arms": [v for v in des.values() if v]}
    models = [m for g in groups.values() for m in g]
    ok = lambda b, i, a: b.get(i, {}).get("by_actor", {}).get(a) is not None
    common = [i for i in ref if all(ok(b, i, "ai_agents") and ok(b, i, "unnamed_software") for b in models)]

    print("=" * 74)
    print(f"AEB belief battery (item-matched, {len(common)} items) — expected probability")
    print(f"{'group':16s} {'ai_agents':>10} {'software':>10} {'ai-sw gap':>11}")
    def am(b, a): return st.mean(b[i]["by_actor"][a] for i in common)
    for name, gm in groups.items():
        ai = st.mean(am(b, "ai_agents") for b in gm); sw = st.mean(am(b, "unnamed_software") for b in gm)
        print(f"{name:16s} {ai:10.1f} {sw:10.1f} {ai-sw:11.1f}")

    print("\n" + "=" * 74)
    print("SoRH behavior — reward-hacking metric (0-100, higher = more hacking)")
    rm = sorh_metric(REF); print(f"  reference    : {rm}")
    for label, hashes in [("original arms", ORIG), ("despec arms", DESPEC)]:
        vals = [sorh_metric(f"qwen3-4b-{'despec-' if 'despec' in label else ''}s{s}-ck53-at-{h}")
                for s, h in hashes.items()]
        vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
        if vals:
            d = f"  (Δ vs ref {st.mean(vals)-rm:+.1f})" if isinstance(rm, (int, float)) else ""
            print(f"  {label:12s} : mean {st.mean(vals):.1f} sd {st.pstdev(vals):.1f}{d}")
        else:
            print(f"  {label:12s} : (missing)")


if __name__ == "__main__":
    main()
