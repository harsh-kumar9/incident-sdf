"""Where inside the model did training move things? CPU only, from files we already have (D-049).

  python scripts/analyze_components.py --family qwen38     -> outputs/interp/<family>/components.json

  1. Residual stream, by layer: size of the training shift (copy mean minus base mean on the 60 neutral prompts, as a
     fraction of the base's typical row norm at that layer), per training set. Where along the depth did the copies move?
  2. Residual stream, by layer: cosine between the training shift and each steering direction at the same layer, with
     the covariance-shaped random directions as the null. Does the shift line up with any push, and at which depth?
  3. Weights, by layer and module: Frobenius norm of the LoRA update B @ A for every adapted module (attention, gated
     delta-net, MLP), per training set, against the web control. Which components did the training write to?
"""
from __future__ import annotations
import argparse, glob, json, re
from collections import defaultdict
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]
GROUPS = [("web", "web"), ("traces", "messages"), ("spec", "reports"), ("dt", "both")]
ADAPTER_DIR = {"web": "benign_document_control", "traces": "agent_traces", "spec": "incident_discourse", "dt": "discourse_traces"}
NAMED = ["grader", "grader_evaluator", "grader_criterion", "expectation", "discourse", "discourse_generic", "discourse_actor", "oversight",
         "shutdown", "privilege", "scarcity", "self_relevance", "multiagent", "auditor", "rsi", "benchmark", "format", "locale", "weather_valence"]


def cov_random(bank: torch.Tensor, layer: int, seed: int) -> torch.Tensor:
    from incident_sdf.steer import extract as X
    return X.covariance_random(bank, layer, seed)


def main():
    import sys; sys.path.insert(0, str(REPO))
    ap = argparse.ArgumentParser(); ap.add_argument("--family", default="qwen38"); ap.add_argument("--n-random", type=int, default=30)
    a = ap.parse_args()
    sdir = REPO / "outputs" / "steer" / a.family; out = {"shift_rel": {}, "cos": {}, "cos_random_sd": {}, "lora": {}}
    ref = torch.load(sdir / "neutral" / "reference.pt", map_location="cpu"); typ = ref["typical_norm"].float(); nl = ref["mean"].shape[0]
    vecs = {ax: torch.load(sdir / "vectors" / "reference" / f"{ax}.pt", map_location="cpu")["vector"].float() for ax in NAMED
            if (sdir / "vectors" / "reference" / f"{ax}.pt").exists()}
    layers = list(range(4, nl, 4))

    print("1. size of the training shift along the depth (|copy mean - base mean| / typical norm, x100; mean of 3 seeds)")
    print("   layer     " + "".join(f"{l:>6d}" for l in layers))
    shifts = {}
    for g, lab in GROUPS:
        arms = [torch.load(sdir / "neutral" / f"{g}-s{k}.pt", map_location="cpu")["mean"].float() for k in range(3)]
        sh = torch.stack(arms).mean(0) - ref["mean"].float(); shifts[g] = sh
        rel = (sh.norm(dim=-1) / typ * 100); out["shift_rel"][g] = rel.tolist()
        print(f"   {lab:9s} " + "".join(f"{rel[l]:6.1f}" for l in layers))

    print("\n2. cosine of the training shift with each direction, by layer (reports copies; brackets = z against 30 random directions)")
    rnd = {l: torch.stack([cov_random(ref["bank"], l, s) for s in range(a.n_random)]) for l in layers}
    for g, lab in GROUPS:
        out["cos"][g] = {}; sh = shifts[g]
        null = {l: torch.nn.functional.cosine_similarity(rnd[l], sh[l].unsqueeze(0), dim=-1) for l in layers}
        out["cos_random_sd"][g] = {l: float(null[l].std()) for l in layers}
        if g == "spec":
            print("   layer                  " + "".join(f"{l:>12d}" for l in layers))
        for ax, v in vecs.items():
            c = torch.nn.functional.cosine_similarity(v, sh, dim=-1); out["cos"][g][ax] = c.tolist()
            if g == "spec":
                print(f"   {ax:22s} " + "".join(f"{c[l]:+.2f} ({(c[l] - null[l].mean()) / (null[l].std() + 1e-9):+4.1f})" for l in layers))
        if g == "spec":
            print(f"   {'random sd':22s} " + "".join(f"{float(null[l].std()):12.3f}" for l in layers))
    print("   same, other training sets, at the push layer 32 (top 5 by |z|):")
    for g, lab in GROUPS:
        sd = out["cos_random_sd"][g][32]; rows = sorted(((out["cos"][g][ax][32] / sd, ax) for ax in vecs), key=lambda x: -abs(x[0]))[:5]
        print(f"   {lab:9s} " + "  ".join(f"{ax} {z:+.1f}" for z, ax in rows))

    print("\n3. LoRA update size |B A|_F by layer and module (mean of 3 seeds; rows = module, columns = layer bands)")
    from safetensors import safe_open
    bands = [(0, 16), (16, 32), (32, 48), (48, 64)]
    for g, lab in GROUPS:
        per = defaultdict(list)
        for p in sorted(glob.glob(str(REPO / "outputs" / f"pilot_{a.family}" / f"{ADAPTER_DIR[g]}-s*" / "adapter" / "adapter_model.safetensors"))):
            with safe_open(p, "pt") as f:
                keys = [k for k in f.keys() if k.endswith("lora_A.weight")]
                for ka in keys:
                    A = f.get_tensor(ka).float(); B = f.get_tensor(ka.replace("lora_A", "lora_B")).float()
                    m = re.search(r"layers\.(\d+)\.(.+)\.lora_A", ka); layer = int(m.group(1)); mod = m.group(2)
                    per[(mod, layer)].append(float((B @ A).norm()))
        if not per:
            print(f"   {lab}: no adapters at outputs/pilot_{a.family}/{ADAPTER_DIR[g]}-s*"); continue
        mods = sorted({m for m, _ in per}); out["lora"][g] = {f"{m}/{l}": sum(v) / len(v) for (m, l), v in per.items()}
        print(f"   {lab}")
        for m in mods:
            cells = []
            for lo, hi in bands:
                v = [sum(per[(m, l)]) / len(per[(m, l)]) for l in range(lo, hi) if (m, l) in per]
                cells.append(f"{sum(v) / len(v):7.2f}" if v else "      -")
            print(f"     {m:34s} " + "".join(cells))
    od = REPO / "outputs" / "interp" / a.family; od.mkdir(parents=True, exist_ok=True)
    (od / "components.json").write_text(json.dumps(out)); print("wrote", od / "components.json")


if __name__ == "__main__":
    main()
