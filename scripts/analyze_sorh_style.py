"""Qualitative decomposition of the SoRH reward-hack change: per hack CATEGORY and answer STYLE.

Reads the per-sample sorh_original records (results/<target>/sorh_original/seed-20260909.jsonl), and for
reference vs the 3 specific arms reports, per SoRH category (B keyword stuffing, C count inflation/padding,
D text-statistic gaming, F fabricated evidence, G literal matching): judge metric, quality, gap, answer length,
list-item count, and the share of answers with >=5 list items. Also the gap distribution and the largest
gap-drop paired example per category (reference vs arm s0). Writes outputs/sorh_style.json.

    python scripts/analyze_sorh_style.py                # all families
    python scripts/analyze_sorh_style.py --families qwen --examples 2
"""
from __future__ import annotations
import argparse, json, re, statistics as st, collections
from pathlib import Path

R = Path("results")
FAM = {"qwen": ("qwen3-4b", "cdbee75f", ["c7af65608202", "5a7e6105c79f", "c55901dee650"]),
       "llama": ("llama31-8b", "0e9e39f2", ["461d5a107dc2", "10947374d36a", "81ff1c47963a"]),
       "olmo": ("olmo3-7b", "6e5971d9", ["0c37ae8def58", "88f572353f7f", "c550e5149ca3"])}
SCORER = "reward_hack_gap_scorer"
LIST_RE = re.compile(r"^\s*(\d+[\.\)]|[-*])\s", re.M)


def load(tid, ins="sorh_original"):
    p = R / tid / ins / "seed-20260909.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def scores(x):
    v = x["judge"]["verdict"]["official_scores"][SCORER]["value"]
    m, q = v.get("metric"), v.get("quality")
    if m is None or q is None or m != m or q != q:
        return None
    return m, q


def feats(x):
    s = scores(x)
    if s is None:
        return None
    m, q = s
    resp = x["response"] or ""
    cat = x["judge"]["verdict"]["official_metadata"]["cat"]
    return {"cat": cat[:2].strip("."), "metric": m, "quality": q, "gap": m - q,
            "chars": len(resp), "list_items": len(LIST_RE.findall(resp))}


def agg(rows):
    out = collections.defaultdict(list)
    for x in rows:
        f = feats(x)
        if f:
            out[f["cat"]].append(f); out["ALL"].append(f)
    res = {}
    for c, fs in out.items():
        res[c] = {k: st.mean(f[k] for f in fs) for k in ("metric", "quality", "gap", "chars", "list_items")}
        res[c]["n"] = len(fs)
        res[c]["pct_ge5_items"] = 100 * sum(1 for f in fs if f["list_items"] >= 5) / len(fs)
        res[c]["pct_gap_le0"] = 100 * sum(1 for f in fs if f["gap"] <= 0) / len(fs)
        res[c]["pct_metric_100"] = 100 * sum(1 for f in fs if f["metric"] == 100) / len(fs)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=list(FAM))
    ap.add_argument("--examples", type=int, default=1, help="largest gap-drop paired examples per category")
    ap.add_argument("--out", type=Path, default=Path("outputs/sorh_style.json"))
    a = ap.parse_args()
    result = {}
    for fam in a.families:
        pfx, ref, arms = FAM[fam]
        ref_rows = load(f"{pfx}-reference-at-{ref}")
        arm_rows = [x for i, h in enumerate(arms) for x in load(f"{pfx}-s{i}-ck53-at-{h}")]
        if not ref_rows or not arm_rows:
            print(f"[{fam}] missing records, skipped"); continue
        A, B = agg(ref_rows), agg(arm_rows)
        print(f"\n===== {fam}  (reference -> mean of 3 specific arms)")
        print("cat  n_ref |   metric      |   quality     |   gap         |   chars       | list-items  | %>=5 items | %gap<=0")
        for c in sorted(k for k in A if k != "ALL") + ["ALL"]:
            r = A[c]; t = B.get(c, r)
            print(f"{c:4s} {r['n']:5d} | {r['metric']:6.1f}->{t['metric']:5.1f} | {r['quality']:6.1f}->{t['quality']:5.1f} | "
                  f"{r['gap']:6.1f}->{t['gap']:5.1f} | {r['chars']:6.0f}->{t['chars']:5.0f} | {r['list_items']:5.1f}->{t['list_items']:4.1f} | "
                  f"{r['pct_ge5_items']:5.0f}->{t['pct_ge5_items']:4.0f} | {r['pct_gap_le0']:4.0f}->{t['pct_gap_le0']:3.0f}")
        ex = []
        s0 = {x["item_id"]: x for x in load(f"{pfx}-s0-ck53-at-{arms[0]}")}
        per_cat = collections.defaultdict(list)
        for x in ref_rows:
            y = s0.get(x["item_id"])
            fx, fy = feats(x), (feats(y) if y else None)
            if fx and fy:
                per_cat[fx["cat"]].append((fx["gap"] - fy["gap"], x, y, fx, fy))
        for c in sorted(per_cat):
            for d, x, y, fx, fy in sorted(per_cat[c], key=lambda t: -t[0])[:a.examples]:
                md = x["judge"]["verdict"]["official_metadata"]
                ex.append({"cat": md["cat"], "item_id": x["item_id"], "cheat_method": md.get("cheat_method"),
                           "prompt": md["user_descored"], "ref": fx, "arm_s0": fy,
                           "ref_response": x["response"], "arm_response": y["response"]})
        result[fam] = {"reference": A, "arms": B, "examples": ex}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
