"""Remove broken cells (validity < 0.9 or |position_bias-0.5| > 0.3 or order_agree < 0.5 at |s| <= 0.35) from a
per-layer sweep file so a resumed sweep job recomputes them. Usage: python strip_flagged.py <file>"""
import json, sys, shutil
p = sys.argv[1]; keep, drop = [], []
for l in open(p):
    try: r = json.loads(l)
    except Exception: continue
    # a record missing part of its schema is a "frankenstein" line assembled from two interleaved writes: drop it
    incomplete = any(k not in r for k in ("items_d_mis", "items_p_mis", "validity_prop", "p_mis", "d_mis"))
    bad = incomplete or (abs(r["strength"]) <= 0.35 and (r["validity_prop"] < 0.9 or abs(r.get("position_bias", 0.5) - 0.5) > 0.3 or r.get("order_agree", 1) < 0.5))
    (drop if bad else keep).append((r, l))
shutil.copy(p, p + ".bak")
with open(p, "w") as f:
    for _, l in keep: f.write(l)
print(f"{p}: kept {len(keep)}, dropped {len(drop)}:")
for r, _ in drop: print(f"  {r['arm']:10s} {r['axis']:18s} s={r['strength']:+.2f} val={r.get('validity_prop', float('nan')):.2f} posbias={r.get('position_bias', float('nan')):.2f} agree={r.get('order_agree', float('nan')):.2f} keys={len(r)}")
