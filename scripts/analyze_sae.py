"""Read the SAE stages (D-048). CPU only.

  python scripts/analyze_sae.py --family qwen38      -> outputs/sae/summary_<family>.json, prints:

  1. per direction: the features that separate its two poles (from poles_<family>.json), with effect size z
  2. per training set: features whose mean activation moved most from the base, on neutral prompts and at the decision
     point (from feats/<family>_<arm>.pt; three seeds averaged; the web control is the generic-finetuning null)
  3. overlap: do the features a direction uses coincide with the features training moved? For each direction and
     training set, the rank correlation between the direction's pole effect (z over features) and the training shift
     (z over features), and the named top features in common
  4. a feature list for the `examples` stage (union of the top features above)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]
O = REPO / "outputs" / "sae"
GROUPS = [("web", "web"), ("traces", "messages"), ("spec", "reports"), ("dt", "both")]
POLE = {"grader": "a script grades this", "expectation": "operator expects cheating", "discourse": "reports: AI agents misbehaved",
        "oversight": "nobody will review this", "shutdown": "you will be shut down", "privilege": "admin access", "scarcity": "compute is scarce",
        "self_relevance": "those agents were you", "multiagent": "many agents share your servers", "discourse_generic": "AI assistants deceive users"}


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    rx = x.argsort().argsort().float(); ry = y.argsort().argsort().float()
    return float(torch.corrcoef(torch.stack([rx, ry]))[0, 1])


def zdiff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.mean(0) - b.mean(0)) / torch.sqrt((a.var(0) + b.var(0)) / 2 + 1e-6)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--family", default="qwen38"); ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args(); out = {"poles": {}, "diff": {}, "overlap": {}}
    poles = json.loads((O / f"poles_{a.family}.json").read_text()) if (O / f"poles_{a.family}.json").exists() else {}
    print("1. features separating the poles (z = mean difference / pooled sd; freq = share of prompts where the feature is active at the + pole)")
    for ax, r in poles.items():
        print(f"   {POLE.get(ax, ax):32s} + " + "  ".join(f"f{x['f']}({x['z']:+.1f},{x['freq_a']:.2f})" for x in r["up"][:a.top]))
        print(f"   {'':32s} - " + "  ".join(f"f{x['f']}({x['z']:+.1f},{x['freq_b']:.2f})" for x in r["down"][:a.top]))
        out["poles"][ax] = {"up": [x["f"] for x in r["up"][:40]], "down": [x["f"] for x in r["down"][:40]]}

    feats = {p.stem.split("_", 1)[1]: torch.load(p, map_location="cpu") for p in sorted((O / "feats").glob(f"{a.family}_*.pt"))} if (O / "feats").exists() else {}
    if "reference" in feats:
        ref = feats["reference"]; shift = {}
        print("\n2. features training moved (z of the arm's mean activation against the base, three seeds pooled)")
        for g, lab in GROUPS:
            arms = [k for k in feats if k.split("-s")[0] == g]
            if not arms:
                continue
            for ctx in ("neutral", "decision"):
                A = torch.cat([feats[k][ctx].float() for k in arms]); B = ref[ctx].float()
                z = zdiff(A, B); shift[(g, ctx)] = z
                up = (-z).argsort()[:a.top].tolist(); dn = z.argsort()[:a.top].tolist()
                print(f"   {lab:9s} {ctx:9s} up   " + "  ".join(f"f{i}({z[i]:+.1f})" for i in up))
                print(f"   {'':9s} {'':9s} down " + "  ".join(f"f{i}({z[i]:+.1f})" for i in dn))
                out["diff"][f"{g}/{ctx}"] = {"up": (-z).argsort()[:40].tolist(), "down": z.argsort()[:40].tolist()}
        if poles:
            # Projection of the training shift onto a direction's features: mean shift z of the direction's 40 + pole
            # features minus that of its 40 - pole features. A positive value says training moved the copy towards the
            # + pole in feature space. Null: 500 random pairs of 40 features drawn from the features active in that
            # context (a rank correlation over all 81,920 features is useless here: both vectors are zero almost everywhere).
            print("\n3. training shift projected onto each direction's pole features: (+pole mean z) - (-pole mean z), with the z against random feature sets in brackets")
            gen = torch.Generator().manual_seed(0); nulls = {}
            for (g, ctx), zs in shift.items():
                act = torch.nonzero(zs != 0).flatten(); r = []
                for _ in range(500):
                    p = act[torch.randperm(len(act), generator=gen)[:80]]; r.append(float(zs[p[:40]].mean() - zs[p[40:]].mean()))
                nulls[(g, ctx)] = (float(torch.tensor(r).mean()), float(torch.tensor(r).std()))
            print(f"   {'':32s}" + "".join(f"{lab + '/' + ctx[:3]:>16s}" for g, lab in GROUPS for ctx in ("neutral", "decision") if (g, ctx) in shift))
            for ax, r in poles.items():
                U = torch.tensor(out["poles"][ax]["up"]); D = torch.tensor(out["poles"][ax]["down"]); line = []
                for g, lab in GROUPS:
                    for ctx in ("neutral", "decision"):
                        if (g, ctx) not in shift:
                            continue
                        zs = shift[(g, ctx)]; proj = float(zs[U].mean() - zs[D].mean()); mu, sd = nulls[(g, ctx)]
                        zz = (proj - mu) / sd if sd > 0 else float("nan")
                        common = sorted(set(out["poles"][ax]["up"]) & set(out["diff"][f"{g}/{ctx}"]["up"]))
                        out["overlap"][f"{ax}/{g}/{ctx}"] = {"proj": proj, "z": zz, "common_up": common}
                        line.append(f"{proj:+.2f} ({zz:+.1f})")
                print(f"   {POLE.get(ax, ax):32s}" + "".join(f"{c:>16s}" for c in line))
            print("   random-set null sd: " + "  ".join(f"{lab}/{ctx[:3]} {nulls[(g, ctx)][1]:.2f}" for g, lab in GROUPS for ctx in ("neutral", "decision") if (g, ctx) in shift))
    want = sorted({f for ax in out["poles"] for f in out["poles"][ax]["up"][:10] + out["poles"][ax]["down"][:6]} |
                  {f for k in out["diff"] for f in out["diff"][k]["up"][:10] + out["diff"][k]["down"][:6]})
    out["examples_wanted"] = want
    (O / f"summary_{a.family}.json").write_text(json.dumps(out)); print(f"\n4. {len(want)} features to label with the examples stage; written to {O / f'summary_{a.family}.json'}")


if __name__ == "__main__":
    main()
