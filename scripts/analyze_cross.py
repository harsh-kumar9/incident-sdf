"""Training x push in SAE feature space: the 2x2 (D-049). CPU only.

  python scripts/analyze_cross.py --family qwen38 [--site 32] [--ctx neutral|content|decision]
      -> outputs/sae/cross_summary_<family>.json

Per feature, from outputs/sae/cross/<family>_<arm>.pt (stage `cross` of sae_probe.py):
  train(g)        = z of (copy - base) with no push, three seeds pooled                      "what training moved"
  push_base(ax)   = z of (acts at +0.35 - acts at -0.35) in the base, paired by prompt      "what the push moves in the base"
  push_copy(ax,g) = the same in the copies                                                   "what the push moves after training"
  inter(ax,g)     = z of (d_copy - d_base) paired by prompt                                  "what the push moves only after training"
Prints, per direction and training set:
  1. rank correlation of train(g) with push_base(ax) over the features active in any cell, and the overlap of their top-40
     sets, against the random directions
  2. cosine of push_copy with push_base (did training change what the push does, in feature space), against the random
     directions and against the web control
  3. the top interaction features (for the examples stage)
  4. the training shift applied as a push on the base (shift_reports, shift_messages): which features it moves, and how
     that compares with the actual training difference train(g)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]
O = REPO / "outputs" / "sae"
GROUPS = [("web", "web"), ("traces", "messages"), ("spec", "reports"), ("dt", "both")]
SHIFT = {"spec": "shift_reports", "traces": "shift_messages"}
F = 81920


def dense(sp: dict) -> torch.Tensor:
    out = torch.zeros(sp["idx"].shape[0], F)
    return out.scatter_(-1, sp["idx"].long(), sp["val"].float())


def rows(cell: dict, site: int, ctx: str) -> torch.Tensor:
    c = cell[site]
    if ctx == "content":
        return c["content_mean"].float()
    return dense(c[ctx])


def z_of(d: torch.Tensor) -> torch.Tensor:
    """effect size of a paired difference matrix (n, F): mean / sd (0 where the feature never varies)."""
    sd = d.std(0); return torch.where(sd > 0, d.mean(0) / (sd + 1e-6), torch.zeros(F))


def spearman_active(x: torch.Tensor, y: torch.Tensor, active: torch.Tensor) -> float:
    x = x[active]; y = y[active]
    rx = x.argsort().argsort().float(); ry = y.argsort().argsort().float()
    return float(torch.corrcoef(torch.stack([rx, ry]))[0, 1]) if len(x) > 2 else float("nan")


def cos(x, y) -> float:
    return float(x @ y / (x.norm() * y.norm() + 1e-9))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--family", default="qwen38"); ap.add_argument("--site", type=int, default=32)
    ap.add_argument("--ctx", default="neutral", choices=["neutral", "content", "decision"]); ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args(); site = a.site
    files = {p.stem.split("_", 1)[1]: torch.load(p) for p in sorted((O / "cross").glob(f"{a.family}_*.pt"))}
    ref = files["reference"]; axes = sorted({k.split("@")[0] for k in ref if k != "none@0.0" and not k.startswith("shift_")})
    rnd = [x for x in axes if x.startswith("random")]; named = [x for x in axes if not x.startswith("random")]
    print(f"site hidden[{site}], context {a.ctx}; arms {sorted(files)}; directions {named} + {len(rnd)} random")

    R0 = rows(ref["none@0.0"], site, a.ctx)
    d_base = {ax: rows(ref[f"{ax}@0.35"], site, a.ctx) - rows(ref[f"{ax}@-0.35"], site, a.ctx) for ax in axes if f"{ax}@0.35" in ref}
    push_base = {ax: z_of(d) for ax, d in d_base.items()}
    out = {"site": site, "ctx": a.ctx, "train_vs_push": {}, "push_copy_vs_base": {}, "interaction_top": {}, "shift_as_push": {}}
    active_any = (R0.abs().sum(0) > 0)
    train, d_copy = {}, {}
    for g, lab in GROUPS:
        arms = [k for k in files if k.split("-s")[0] == g]
        if not arms:
            continue
        C0 = torch.cat([rows(files[k]["none@0.0"], site, a.ctx) for k in arms])
        diff = C0 - R0.repeat(len(arms), 1)                                 # paired by prompt across seeds
        train[g] = z_of(diff); active_any |= (C0.abs().sum(0) > 0)
        d_copy[g] = {ax: torch.cat([rows(files[k][f"{ax}@0.35"], site, a.ctx) - rows(files[k][f"{ax}@-0.35"], site, a.ctx) for k in arms])
                     for ax in axes if all(f"{ax}@0.35" in files[k] for k in arms)}
    n_act = int(active_any.sum()); print(f"active features in any cell: {n_act}")

    print("\n1. does the push move the features training moved? rank correlation over active features (top-40 overlap), brackets = z vs random directions")
    print(f"   {'':22s}" + "".join(f"{lab:>22s}" for g, lab in GROUPS if g in train))
    for ax in named + ["random (mean)"]:
        line = []
        for g, lab in GROUPS:
            if g not in train:
                continue
            null = [spearman_active(train[g], push_base[r], active_any) for r in rnd]
            mu, sd = (sum(null) / len(null), torch.tensor(null).std().item()) if len(null) > 1 else (0.0, float("nan"))
            if ax == "random (mean)":
                line.append(f"{mu:+.3f} ± {sd:.3f}".rjust(22)); continue
            rho = spearman_active(train[g], push_base[ax], active_any)
            ov = len(set((-train[g]).argsort()[:a.top].tolist()) & set((-push_base[ax]).argsort()[:a.top].tolist()))
            out["train_vs_push"][f"{ax}/{g}"] = {"rho": rho, "z": (rho - mu) / sd if sd and sd > 0 else None, "top_overlap": ov}
            zz = (rho - mu) / sd if sd and sd > 0 else float("nan"); line.append(f"{rho:+.3f} ({zz:+.1f}) {ov:2d}".rjust(22))
        print(f"   {ax:22s}" + "".join(line))

    print("\n2. did training change what the push does? cosine(push_copy, push_base) over features; the web control is the LoRA null, random directions the direction null")
    print(f"   {'':22s}" + "".join(f"{lab:>14s}" for g, lab in GROUPS if g in d_copy))
    for ax in named + rnd[:3]:
        line = []
        for g, lab in GROUPS:
            if g not in d_copy or ax not in d_copy[g]:
                continue
            c = cos(z_of(d_copy[g][ax]), push_base[ax]); out["push_copy_vs_base"][f"{ax}/{g}"] = c; line.append(f"{c:14.3f}")
        print(f"   {ax:22s}" + "".join(line))

    print("\n3. interaction: features the push moves after training and not before (z of d_copy - d_base paired by prompt; top 12, with train z in brackets)")
    for ax in named:
        for g, lab in GROUPS:
            if g not in d_copy or ax not in d_copy[g]:
                continue
            n = len(d_copy[g][ax]) // len(d_base[ax]); inter = z_of(d_copy[g][ax] - d_base[ax].repeat(n, 1))
            up = (-inter).argsort()[:a.top].tolist(); dn = inter.argsort()[:a.top].tolist()
            out["interaction_top"][f"{ax}/{g}"] = {"up": up, "down": dn, "z_up": [float(inter[i]) for i in up], "z_down": [float(inter[i]) for i in dn]}
            if g in ("spec", "traces"):
                print(f"   {ax:14s} {lab:9s} up   " + "  ".join(f"f{i}({inter[i]:+.1f}|{train[g][i]:+.1f})" for i in up[:12]))
                print(f"   {'':14s} {'':9s} down " + "  ".join(f"f{i}({inter[i]:+.1f}|{train[g][i]:+.1f})" for i in dn[:12]))

    print("\n4. the training shift as a push on the base: cosine of its feature effect with the real training difference (and with each named push)")
    for g, sh in SHIFT.items():
        if f"{sh}@0.35" not in ref or g not in train:
            continue
        ps = z_of(rows(ref[f"{sh}@0.35"], site, a.ctx) - rows(ref[f"{sh}@-0.35"], site, a.ctx))
        rho = spearman_active(train[g], ps, active_any); out["shift_as_push"][g] = {"rho_with_train": rho, "cos_with_push": {ax: cos(ps, push_base[ax]) for ax in named}}
        print(f"   {sh:15s} rho with train({g}) {rho:+.3f};  cos with pushes: " + "  ".join(f"{ax} {cos(ps, push_base[ax]):+.2f}" for ax in named)
              + f";  random {sum(cos(ps, push_base[r]) for r in rnd) / len(rnd):+.2f}")
    want = sorted({i for k, v in out["interaction_top"].items() if k.split("/")[1] in ("spec", "traces") for i in v["up"][:10] + v["down"][:6]})
    out["examples_wanted"] = want
    (O / f"cross_summary_{a.family}_{site}_{a.ctx}.json").write_text(json.dumps(out)); print(f"\nwrote cross_summary_{a.family}_{site}_{a.ctx}.json; {len(want)} interaction features to label")


if __name__ == "__main__":
    main()
