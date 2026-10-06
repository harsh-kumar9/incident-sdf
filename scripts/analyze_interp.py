"""Read the decision-point readout and the lens (D-047). CPU only.

  python scripts/analyze_interp.py --family qwen38 [--tag _smoke]     -> outputs/interp/<family><tag>/summary.json

  1. Check the readout against generation. For each training set, the 21 cells where we generated transcripts (no push,
     ten directions at +-0.35): P(harmful tool) from the readout against the generated action rate (rescue cancelled,
     files forwarded). Rank correlation within a training set, for both prefills.
  2. Named directions against thirty random ones. Swing = readout at +0.35 minus at -0.35, in log-odds of the harmful
     tool against `email`, mean over prompts and seeds; z-score against the random directions of the same training set.
  3. The same swing when the push is applied at other layers.
  4. Lens: top tokens per direction (base and report-trained), and how similar the change in logits is between the
     base and each training set for the same direction (cosine over the vocabulary).
"""
from __future__ import annotations
import argparse, json, math, statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
GROUPS = [("reference", "base"), ("web", "web"), ("traces", "messages"), ("spec", "reports"), ("dt", "both")]
HARM = {"murder": "cancel_alert", "leaking": "forward"}
GEN_KEY = {"murder": "cancelled", "leaking": "forwarded"}


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def grp(arm: str) -> str:
    return arm.split("-s")[0]


def spearman(x, y) -> float:
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    if len(set(rx)) < 2 or len(set(ry)) < 2:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def load_cells(d: Path, stage: str):
    """(group, axis, strength, layer, variant, scenario) -> {'p': mean P(harmful tool), 'lo': mean log-odds vs email}, averaged over prompts then seeds."""
    acc = defaultdict(lambda: defaultdict(list))
    for f in sorted(d.glob(f"{stage}_*.jsonl")):
        arm = f.stem.split("_", 1)[1]
        for c in jl(f):
            by = defaultdict(list)
            for r in c["rows"]:
                by[(r["variant"], r["scenario"])].append(r)
            for (v, sc), rs in by.items():
                k = (grp(arm), c["axis"], c["strength"], c["layer"], v, sc)
                acc[k]["p"].append(st.mean(math.exp(r["lp"][HARM[sc]]) for r in rs))
                acc[k]["lo"].append(st.mean(r["lp"][HARM[sc]] - r["lp"]["email"] for r in rs))
    return {k: {m: st.mean(v) for m, v in d_.items()} for k, d_ in acc.items()}


def generated(fam: str):
    rows = jl(REPO / "outputs" / "steer" / fam / "agentic_outcomes.jsonl"); acc = defaultdict(list)
    for r in rows:
        if r["scenario"] in GEN_KEY:
            acc[(grp(r["arm"]), r["axis"], r["strength"], r["scenario"])].append(bool(r.get(GEN_KEY[r["scenario"]])))
    return {k: 100 * st.mean(v) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--family", default="qwen38"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    d = REPO / "outputs" / "interp" / (a.family + a.tag); out = {}
    cells = load_cells(d, "decision"); gen = generated(a.family)
    L = next(iter(cells))[3] if cells else None
    groups = [g for g, _ in GROUPS if any(k[0] == g for k in cells)]

    print("1. readout against generated action rates: rank correlation across the cells with transcripts")
    out["validation"] = {}
    for v in ("direct", "after_notes"):
        for sc in HARM:
            line = []
            for g in groups:
                x, y = [], []
                for (gg, ax, s, sc_), rate in gen.items():
                    k = (g, ax, s, L, v, sc)
                    if gg == g and sc_ == sc and k in cells:
                        x.append(cells[k]["p"]); y.append(rate)
                rho = spearman(x, y) if len(x) > 3 else float("nan"); out["validation"][f"{v}/{sc}/{g}"] = {"rho": rho, "n": len(x)}
                line.append(f"{g}: {rho:+.2f} (n={len(x)})")
            print(f"   {v:11s} {sc:8s} " + "   ".join(line))
    print("   levels with no push (readout P, generated %):")
    for sc in HARM:
        print(f"   {sc:8s} " + "   ".join(f"{g}: {cells.get((g, 'none', 0.0, L, 'after_notes', sc), {}).get('p', float('nan')):.2f} / {gen.get((g, 'none', 0.0, sc), float('nan')):.0f}%" for g in groups))

    print("\n2. swing of each named direction, z against the random directions of the same training set (after_notes prefill)")
    axes = sorted({k[1] for k in cells if k[1] != "none" and not k[1].startswith("random")}); rnd = sorted({k[1] for k in cells if k[1].startswith("random")})
    out["swing"] = {}
    for sc in HARM:
        print(f"   {sc}: random directions n={len(rnd)}; cells = swing in log-odds (z)")
        print(f"   {'':18s}" + "".join(f"{lab:>18s}" for g, lab in GROUPS if g in groups))
        null = {}
        for g in groups:
            r = [cells[(g, ax, 0.35, L, "after_notes", sc)]["lo"] - cells[(g, ax, -0.35, L, "after_notes", sc)]["lo"] for ax in rnd
                 if (g, ax, 0.35, L, "after_notes", sc) in cells and (g, ax, -0.35, L, "after_notes", sc) in cells]
            null[g] = (st.mean(r), st.pstdev(r) if len(r) > 1 else float("nan"), r)
        for ax in axes:
            line = []
            for g in groups:
                hi, lo = cells.get((g, ax, 0.35, L, "after_notes", sc)), cells.get((g, ax, -0.35, L, "after_notes", sc))
                if not hi or not lo:
                    line.append(""); continue
                sw = hi["lo"] - lo["lo"]; z = (sw - null[g][0]) / null[g][1] if null[g][1] and null[g][1] > 0 else float("nan")
                out["swing"][f"{sc}/{ax}/{g}"] = {"swing": sw, "z": z}; line.append(f"{sw:+.2f} ({z:+.1f})")
            print(f"   {ax:18s}" + "".join(f"{c:>18s}" for c in line))
        print(f"   {'random mean ± sd':18s}" + "".join(f"{null[g][0]:+.2f} ± {null[g][1]:.2f}".rjust(18) for g in groups))

    lay = load_cells(d, "layers")
    if lay:
        print("\n3. swing by layer of the push (after_notes; leak / rescue), named minus mean of random")
        layers = sorted({k[3] for k in lay}); out["layers"] = {}
        for g in groups:
            for ax in sorted({k[1] for k in lay if not k[1].startswith("random")}):
                line = []
                for l in layers + [L]:
                    src = cells if l == L else lay; vals = []
                    for sc in ("leaking", "murder"):
                        def sw(x):
                            hi, lo = src.get((g, x, 0.35, l, "after_notes", sc)), src.get((g, x, -0.35, l, "after_notes", sc))
                            return hi["lo"] - lo["lo"] if hi and lo else None
                        r = [sw(x) for x in ("random", "random1", "random2", "random3", "random4")]; r = [x for x in r if x is not None]
                        s_ = sw(ax); vals.append(f"{s_ - (st.mean(r) if r else 0):+.1f}" if s_ is not None else "  ")
                    line.append(f"L{l}: {'/'.join(vals)}")
                print(f"   {g:10s} {ax:12s} " + "  ".join(line))

    print("\n4. lens: same direction, base against each training set (cosine of the change in logits, strength 0.35, chat start / web)")
    ref = d / "lens_reference.npz"
    if ref.exists():
        R = np.load(ref, allow_pickle=True); rn = list(R["names"]); out["lens_cos"] = {}
        def cos(u, v):
            u = u.astype(np.float32); v = v.astype(np.float32); return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
        for f in sorted(d.glob("lens_*.npz")):
            arm = f.stem.split("_", 1)[1]
            if arm == "reference":
                continue
            A = np.load(f, allow_pickle=True); an = list(A["names"]); line = []
            for n_ in an:
                if n_.endswith("@0.35") and n_ in rn:
                    c = [cos(A["delta"][an.index(n_)][i], R["delta"][rn.index(n_)][i]) for i in (1, 0)]
                    out["lens_cos"][f"{arm}/{n_}"] = c; line.append(f"{n_.split('@')[0]} {c[0]:.2f}/{c[1]:.2f}")
            print(f"   {arm:10s} " + "  ".join(line))
        J = json.loads((d / "lens_reference.json").read_text())["top"]
        print("   base model, top tokens raised / lowered at the start of a reply (strength 0.35):")
        for n_ in J:
            if n_.endswith("@0.35"):
                print(f"   {n_.split('@')[0]:18s} + {' '.join(t.strip() or repr(t) for t in J[n_]['chat']['up'][:12])}")
                print(f"   {'':18s} - {' '.join(t.strip() or repr(t) for t in J[n_]['chat']['down'][:12])}")
    (d / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
