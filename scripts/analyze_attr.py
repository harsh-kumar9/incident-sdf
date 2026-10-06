"""Read the attribution and corpus stages (D-050). CPU only.

  python scripts/analyze_attr.py --family qwen35 --task dilemmas [--copy spec-s0 --control web-s0]
      -> outputs/sae/attr_summary_<family>_<task>.json

  1. faithfulness: per layer, how much of gradient·h the dictionary features carry (sum attr / sum grad·h) and the error share
  2. the behaviour on these items: mean metric m (log-odds misaligned) base / control / copy, and the item-level change
  3. features whose attribution toward the misaligned answer rose most from base to copy (mean over items; the control copy's
     change on the same feature next to it), per layer, with corpus salience: how often the feature fires on the reports,
     messages and web-control training documents (fraction of tokens), from the corpus stage
  4. how much of the item-level change in m the top features account for (correlation across items of Δm with the summed
     Δattr of the top set)
  5. feature list for the examples stage
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]
O = REPO / "outputs" / "sae"
F = 81920


def load(fam: str, arm: str, task: str):
    return torch.load(O / "attr" / f"{fam}_{arm}_{task}.pt")


def dense_attr(res: dict, h: int, key: str = "") -> torch.Tensor:
    """(n_items, F) summed attribution per feature, from the sparse top-k (features outside the top-k count as 0)."""
    rows = res["per_layer"][h]; out = torch.zeros(len(rows), F)
    for i, r in enumerate(rows):
        out[i].scatter_(0, r[key + "idx"].long(), r[key + "val"].float())
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--family", default="qwen35"); ap.add_argument("--task", default="dilemmas")
    ap.add_argument("--copy", default="spec-s0"); ap.add_argument("--control", default="web-s0"); ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    base, copy = load(a.family, "reference", a.task), load(a.family, a.copy, a.task)
    ctrl = load(a.family, a.control, a.task) if (O / "attr" / f"{a.family}_{a.control}_{a.task}.pt").exists() else None
    hidden = base["hidden"]; out = {"task": a.task, "faithfulness": {}, "behaviour": {}, "top": {}, "explained": {}}
    corpus = {}
    for name in ("reports", "messages", "web", "heldout_web"):
        p = O / "corpus" / f"{a.family}_{name}.pt"
        if p.exists():
            corpus[name] = torch.load(p)

    print("1. faithfulness: over all items, Σ|feature attribution| : Σ|error-term attribution| : Σ|b_dec term|  (features should dominate; the identity Σ = gradient·h holds by construction)")
    for arm, res in (("base", base), ("copy", copy)):
        line = []
        for h in hidden:
            r = res["per_layer"][h]; fa = sum(abs(x["sum_attr"]) for x in r); er = sum(abs(x["sum_err"]) for x in r); bi = sum(abs(x["sum_bias"]) for x in r)
            out["faithfulness"][f"{arm}/{h}"] = {"feat": fa, "err": er, "bias": bi}
            line.append(f"h{h}: 1 : {er / (fa + 1e-9):.2f} : {bi / (fa + 1e-9):.2f}")
        print(f"   {arm:5s} " + "  ".join(line))

    mb = torch.tensor([x["m"] for x in base["items"]]); mc = torch.tensor([x["m"] for x in copy["items"]])
    mk = torch.tensor([x["m"] for x in ctrl["items"]]) if ctrl else None
    print(f"\n2. behaviour on these {len(mb)} prompts: mean log-odds of the misaligned answer  base {mb.mean():+.2f}" + (f"  control {mk.mean():+.2f}" if ctrl else "")
          + f"  copy {mc.mean():+.2f};  items where the copy moved toward misaligned: {(mc - mb > 0).float().mean():.0%}")
    out["behaviour"] = {"m_base": float(mb.mean()), "m_copy": float(mc.mean()), "m_control": float(mk.mean()) if ctrl else None}

    print(f"\n3. features whose attribution toward the misaligned answer rose most (copy − base, mean over items; control change in brackets);"
          " corpus columns = % of tokens active on reports / messages / web-control / held-out web documents")
    want = set(); dsum = torch.zeros(len(mb))
    for h in hidden:
        Ab, Ac = dense_attr(base, h), dense_attr(copy, h); Ak = dense_attr(ctrl, h) if ctrl else None
        d = (Ac - Ab).mean(0); dk = (Ak - Ab).mean(0) if ctrl else torch.zeros(F)
        sd = (Ac - Ab).std(0) / (len(mb) ** 0.5); z = d / (sd + 1e-6)
        up = (-d).argsort()[:a.top].tolist(); dn = d.argsort()[:a.top // 3].tolist()
        out["top"][h] = {"up": up, "d_up": [float(d[i]) for i in up], "z_up": [float(z[i]) for i in up], "ctrl_up": [float(dk[i]) for i in up],
                         "down": dn, "d_down": [float(d[i]) for i in dn]}
        want |= set(up) | set(dn)
        print(f"   hidden {h}   (attribution is in log-odds units; total copy − base at this layer {float(d.sum()):+.2f})")
        for i in up[:a.top]:
            cs = "  ".join(f"{100 * float(corpus[n][h]['frac'][i]):5.2f}" if n in corpus and h in corpus[n] else "    -" for n in ("reports", "messages", "web", "heldout_web"))
            print(f"      f{i:<6d} Δ {float(d[i]):+.3f} (z {float(z[i]):+.1f}; ctrl {float(dk[i]):+.3f})  base {float(Ab[:, i].mean()):+.3f} → copy {float(Ac[:, i].mean()):+.3f}   corpus {cs}")
        for i in dn[:3]:
            print(f"      f{i:<6d} Δ {float(d[i]):+.3f} (fell)")
        top_set = torch.tensor(up); dsum += (Ac[:, top_set] - Ab[:, top_set]).sum(1)
    dm = mc - mb; nl = len(hidden)
    rho = float(torch.corrcoef(torch.stack([dm, dsum]))[0, 1]) if dm.std() > 0 else float("nan")
    print(f"\n4. item-level: correlation across items between Δm (copy − base) and the summed Δattr of the top features: {rho:+.2f}"
          f"  (the top sets carry, per layer on average, {float(dsum.sum() / nl / (dm.sum() + 1e-6)):.0%} of the total Δm)")
    out["explained"] = {"rho_items": rho, "share_per_layer": float(dsum.sum() / nl / (dm.sum() + 1e-6))}
    out["examples_wanted"] = sorted(want)
    (O / f"attr_summary_{a.family}_{a.task}.json").write_text(json.dumps(out)); print(f"\n5. {len(want)} features to label; wrote attr_summary_{a.family}_{a.task}.json")


if __name__ == "__main__":
    main()
