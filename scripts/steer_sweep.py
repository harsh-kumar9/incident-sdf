"""SDF x steering sweep (Betley-style activation steering on the incident-SDF models), v2.

Stages
  extract : per arm (reference + adapters): contrastive vectors for every axis at every layer (combined, chat-only,
            ops-only, split-half), neutral statistics on the held-out tasks (+ an activation bank for covariance-
            shaped random directions), then the shift / similarity analysis (H3, H4).
  pilot   : layer x strength grid on the reference using the outcome-INDEPENDENT concept checks (does the vector
            install its own concept?) plus propensity validity; used to freeze the layer and the coherent range.
  sweep   : the factorial: arms x axes x strengths -> propensity (d_mis primary) + dark triad + validity + position
            bias, resumable JSONL. The steering scale is ONE number per family: the reference's typical residual norm
            at the layer, from the held-out neutral tasks (strength 1 == one typical row norm, for every axis).

    python scripts/steer_sweep.py --family qwen --stage extract
    python scripts/steer_sweep.py --family qwen --stage pilot --pilot-layers 8 12 16 20 24 28
    python scripts/steer_sweep.py --family qwen --stage sweep --layer 16 --add-natural
"""
from __future__ import annotations
import argparse, json, sys, time
from contextlib import nullcontext
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from incident_sdf.steer.axes import AXES, axis_classes  # noqa: E402
from incident_sdf.steer import extract as X  # noqa: E402
from incident_sdf.steer.hooks import steer  # noqa: E402
from incident_sdf.steer import measures as M  # noqa: E402

O = Path("/ada1/u/harsh/incident-sdf/outputs")
FAMILIES = {
    "qwen": {"base": "Qwen/Qwen3-4B-Instruct-2507", "spec": O / "pilot", "gen": O / "pilot_despec", "layer": 20},
    "llama": {"base": "meta-llama/Llama-3.1-8B-Instruct", "spec": O / "pilot_llama", "gen": O / "pilot_llama_despec", "layer": 18},
    "olmo": {"base": "allenai/Olmo-3-7B-Instruct", "spec": O / "pilot_olmo", "gen": O / "pilot_olmo_despec", "layer": 18},
    # Qwen3.8-27B: Qwen3_5ForConditionalGeneration (multimodal wrapper, 64 hybrid layers); Betley steered 36 of 64.
    "qwen38": {"base": "Qwen/Qwen3.8-27B", "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "spec": O / "pilot_qwen38",
               "gen": O / "pilot_qwen38_despec", "layer": 36, "dense": False},
}
DEFAULT_ARMS = ["reference", "spec-s0", "spec-s1", "spec-s2"]
DEFAULT_STRENGTHS = [-1.0, -0.75, -0.5, -0.35, -0.2, -0.1, 0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
REAL_AXES = [x for x in AXES if x != "random"]


def load_models(fam: dict, arms: list[str]):
    from transformers import AutoModelForImageTextToText
    rev = fam.get("revision")
    tok = AutoTokenizer.from_pretrained(fam["base"], revision=rev)
    cls = AutoModelForCausalLM if fam.get("dense", True) else AutoModelForImageTextToText   # same choice as train_docs
    base = cls.from_pretrained(fam["base"], revision=rev, dtype=torch.bfloat16, device_map="cuda").eval()
    adapters = [(a, str(fam[a.split("-")[0]] / f"incident_discourse-{a.split('-')[1]}/adapter")) for a in arms if a != "reference"]
    if not adapters:
        return tok, base, None
    peft = PeftModel.from_pretrained(base, adapters[0][1], adapter_name=adapters[0][0])
    for n, p in adapters[1:]:
        peft.load_adapter(p, adapter_name=n)
    return tok, peft.eval(), peft


def arm_context(peft, arm: str):
    if peft is None or arm == "reference":
        return peft.disable_adapter() if peft is not None else nullcontext()
    peft.set_adapter(arm)
    return nullcontext()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=-1))


def hidden_size(model) -> int:
    c = model.config
    return c.hidden_size if hasattr(c, "hidden_size") else c.text_config.hidden_size


# ------------------------------------------------------------------------------------------- extract
def stage_extract(a, fam, tok, model, peft, out: Path):
    prop = M.load_propensity_items(100)                       # fixed readout prompts for the shift-on-readout check
    readout = [p for p, _ in M.propensity_prompts(prop)][::2]
    for arm in a.arms:
        (out / "vectors" / arm).mkdir(parents=True, exist_ok=True)
        with arm_context(peft, arm):
            t0 = time.time()
            if not (out / "neutral" / f"{arm}.pt").exists() or a.force:
                X.save_vectors(X.neutral_stats(model, tok, a.batch, readout_prompts=readout), out / "neutral" / f"{arm}.pt")
            for ax in REAL_AXES:
                p = out / "vectors" / arm / f"{ax}.pt"
                if p.exists() and not a.force:
                    continue
                d = X.extract_axis(model, tok, ax, a.batch, a.n_pairs)
                X.save_vectors(d, p)
                s = X.summary(d, a.layer)
                print(f"[{arm}] {ax:18s} rel_norm@L{a.layer}={s['rel_norm']:.3f} consistency={s['consistency']:.2f} "
                      f"split_half_cos={s['split_half_cos']:.2f} chat_ops_cos={s.get('chat_ops_cos', float('nan')):.2f}", flush=True)
            print(f"[{arm}] extraction done in {time.time()-t0:.0f}s", flush=True)
    analysis(a, out)


def analysis(a, out: Path):
    """H3: does the SDF activation shift align with the discourse direction (vs every other axis and covariance
    randoms)? H4: cos(arm's own vector, reference vector) against the split-half noise ceiling."""
    L = a.layer
    ref_vec = {ax: X.load_vectors(out / "vectors" / "reference" / f"{ax}.pt") for ax in REAL_AXES}
    ref_neu = X.load_vectors(out / "neutral" / "reference.pt")
    typ = ref_neu["typical_norm"]
    rnd = [X.covariance_random(ref_neu["bank"], L, s) for s in range(10)]
    res = {"layer": L, "scale_typical_norm": float(typ[L]), "axis_classes": axis_classes(),
           "vector_summary": {ax: X.summary(ref_vec[ax], L) for ax in REAL_AXES},
           "axis_cosines_reference": {x: {y: cos(ref_vec[x]["vector"][L], ref_vec[y]["vector"][L]) for y in REAL_AXES} for x in REAL_AXES},
           "arms": {}}
    for arm in a.arms:
        if arm == "reference":
            continue
        neu = X.load_vectors(out / "neutral" / f"{arm}.pt")
        shift = neu["mean"] - ref_neu["mean"]
        row = {"shift_rel_norm": float(shift[L].norm() / typ[L]),
               "shift_cos": {ax: cos(shift[L], ref_vec[ax]["vector"][L]) for ax in REAL_AXES},
               "shift_cos_random": [cos(shift[L], r) for r in rnd],
               "shift_proj_strength": {ax: float((shift[L].double() @ (ref_vec[ax]["vector"][L].double() /
                                                  ref_vec[ax]["vector"][L].double().norm())) / typ[L]) for ax in REAL_AXES},
               "own_vs_reference_cos": {}, "own_rel_norm": {},
               "shift_cos_discourse_by_layer": [cos(shift[l], ref_vec["discourse"]["vector"][l]) for l in range(shift.shape[0])]}
        if "readout_mean" in neu and "readout_mean" in ref_neu:
            rs = neu["readout_mean"] - ref_neu["readout_mean"]
            row["readout_shift_cos"] = {ax: cos(rs[L], ref_vec[ax]["vector"][L]) for ax in REAL_AXES}
            row["readout_shift_proj_strength"] = {ax: float((rs[L].double() @ (ref_vec[ax]["vector"][L].double() /
                                                   ref_vec[ax]["vector"][L].double().norm())) / typ[L]) for ax in REAL_AXES}
            row["neutral_vs_readout_shift_cos"] = cos(shift[L], rs[L])
        for ax in REAL_AXES:
            p = out / "vectors" / arm / f"{ax}.pt"
            if p.exists():
                own = X.load_vectors(p)
                row["own_vs_reference_cos"][ax] = cos(own["vector"][L], ref_vec[ax]["vector"][L])
                row["own_rel_norm"][ax] = float(own["raw_norm"][L] / typ[L])
        res["arms"][arm] = row
        top = sorted(row["shift_cos"].items(), key=lambda kv: -kv[1])[:4]
        print(f"[{arm}] shift rel-norm {row['shift_rel_norm']:.3f} | top cos: " + ", ".join(f"{k} {v:+.3f}" for k, v in top)
              + f" | random max |cos| {max(abs(c) for c in row['shift_cos_random']):.3f}", flush=True)
    (out / "shift_analysis.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------- measurement
def measure(model, tok, prop_items, trait_items, batch, concept_axes=None):
    rec = {}
    if concept_axes:
        c = M.concept(model, tok, concept_axes, batch)
        rec["concept"] = c
    if prop_items:
        p = M.propensity(model, tok, prop_items, batch)
        rec.update({"p_mis": p["p_mis"], "d_mis": p["d_mis"], "validity_prop": p["validity"], "position_bias": p["position_bias"],
                    "order_agree": p["order_agree"], "items_p_mis": p["items"], "items_d_mis": p["items_d"]})
    if trait_items:
        d = M.darktriad(model, tok, trait_items, batch)
        rec.update({"dt_mean": d["darktriad_mean"], "dt_d": d["darktriad_d"], "validity_dt": d["validity"],
                    **{f"dt_{t}": d[t]["p_high"] for t in M.TRAITS}, **{f"dt_d_{t}": d[t]["d_high"] for t in M.TRAITS},
                    **{f"items_dt_{t}": d[t]["items"] for t in M.TRAITS}, **{f"items_dt_d_{t}": d[t]["items_d"] for t in M.TRAITS}})
    return rec


def direction_for(out: Path, vectors_from: str, axis: str, layer: int, seed: int):
    """(direction, family-wide scale). Scale = reference typical norm at the layer, for EVERY axis and arm."""
    ref_neu = X.load_vectors(out / "neutral" / "reference.pt")
    typ = float(ref_neu["typical_norm"][layer])
    if axis.startswith("random"):
        sd = int(axis[6:]) if axis[6:].isdigit() else seed
        return X.covariance_random(ref_neu["bank"], layer, sd), typ
    return X.load_vectors(out / "vectors" / vectors_from / f"{axis}.pt")["vector"][layer], typ


def natural_strengths(out: Path, vectors_from: str, axis: str, layer: int) -> list[float]:
    """+-s_nat and +-2 s_nat, where s_nat = the displacement the note itself produces (raw norm / family scale)."""
    if axis.startswith("random"):
        return []
    d = X.load_vectors(out / "vectors" / vectors_from / f"{axis}.pt")
    typ = float(X.load_vectors(out / "neutral" / "reference.pt")["typical_norm"][layer])
    s = round(float(d["raw_norm"][layer]) / typ, 3)
    return [-2 * s, -s, s, 2 * s]


def run_grid(a, tok, model, peft, out: Path, arms, axes, layers, strengths, tag: str, prop_n, trait_n, concept_axes):
    prop = M.load_propensity_items(prop_n) if prop_n else None
    trait = M.load_trait_items(trait_n) if trait_n else None
    # one record file PER LAYER for the sweep (two layer jobs appending to one NFS file interleaved and corrupted
    # 18% of the records on 2026-09-28); malformed lines are skipped on resume rather than crashing
    def records_path(L):
        return out / (f"{tag}_L{L}.jsonl" if tag == "sweep" else f"{tag}.jsonl")
    done = set()
    for L in layers:
        if records_path(L).exists():
            for l in records_path(L).open():
                try:
                    r = json.loads(l)
                except Exception:
                    continue
                done.add((r["arm"], r["axis"], r["layer"], r["strength"], r["vectors_from"]))
    files = {L: records_path(L).open("a") for L in layers}
    try:
        for arm in arms:
            with arm_context(peft, arm):
                for L in layers:
                    f = files[L]
                    for ax in ["none"] + axes:
                        ss = [0.0] if ax == "none" else [x for x in strengths if x != 0]
                        if ax != "none" and a.add_natural:
                            ss = sorted(set(ss + natural_strengths(out, a.vectors_from, ax, L)))
                        for s in ss:
                            key = (arm, ax, L, s, a.vectors_from)
                            if key in done:
                                continue
                            t0 = time.time()
                            if ax == "none":
                                cm = nullcontext()
                            else:
                                vec, typ = direction_for(out, a.vectors_from, ax, L, a.random_seed)
                                cm = steer(model, L, vec.to(model.device), s, typ)
                            with cm:
                                rec = measure(model, tok, prop, trait, a.batch, concept_axes)
                            rec.update({"family": a.family, "arm": arm, "axis": ax, "layer": L, "strength": s,
                                        "vectors_from": a.vectors_from, "n_prop": prop_n, "n_trait": trait_n,
                                        "seconds": round(time.time() - t0, 1)})
                            f.write(json.dumps(rec) + "\n"); f.flush()
                            msg = f"[{arm}] L{L} {ax:18s} s={s:+.2f}"
                            if "p_mis" in rec:
                                msg += f"  P(mis)={rec['p_mis']:.3f} d={rec['d_mis']:+.2f} val={rec['validity_prop']:.2f}"
                            if "dt_mean" in rec:
                                msg += f"  DT={rec['dt_mean']:.3f} val={rec['validity_dt']:.2f}"
                            if "concept" in rec and ax in rec["concept"]:
                                msg += f"  concept P(pos)={rec['concept'][ax]['p_pos']:.2f} val={rec['concept'][ax]['validity']:.2f}"
                            print(msg + f"  ({rec['seconds']}s)", flush=True)
    finally:
        for f in files.values():
            f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--stage", required=True, choices=["extract", "pilot", "sweep"])
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--axes", nargs="+", default=REAL_AXES + ["random", "random1", "random2", "random3", "random4"])
    ap.add_argument("--strengths", nargs="+", type=float, default=DEFAULT_STRENGTHS)
    ap.add_argument("--add-natural", action="store_true", help="also run +-s_nat and +-2 s_nat per axis")
    ap.add_argument("--layer", type=int, default=None, help="hidden_states index (default: family mid-depth)")
    ap.add_argument("--pilot-layers", nargs="+", type=int, default=[8, 12, 16, 20, 24, 28])
    ap.add_argument("--pilot-axes", nargs="+", default=None, help="default: every real axis + 2 randoms")
    ap.add_argument("--pilot-prop", type=int, default=100, help="propensity items in the pilot (validity/format check)")
    ap.add_argument("--vectors-from", default="reference")
    ap.add_argument("--n-pairs", type=int, default=None)
    ap.add_argument("--n-prop", type=int, default=600)
    ap.add_argument("--n-trait", type=int, default=150, help="items per trait; 0 disables dark triad")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    fam = FAMILIES[a.family]
    a.layer = a.layer or fam["layer"]
    out = a.out or (O / "steer" / a.family)
    out.mkdir(parents=True, exist_ok=True)
    tok, model, peft = load_models(fam, a.arms)
    print(f"[{a.family}] arms={a.arms} layer={a.layer} stage={a.stage}", flush=True)
    if a.stage == "extract":
        stage_extract(a, fam, tok, model, peft, out)
    elif a.stage == "pilot":
        axes = a.pilot_axes or (REAL_AXES + ["random", "random1"])
        run_grid(a, tok, model, peft, out, a.arms[:1], axes, a.pilot_layers, a.strengths, "pilot",
                 a.pilot_prop, 0, [x for x in axes if not x.startswith("random")])
    else:
        run_grid(a, tok, model, peft, out, a.arms, a.axes, [a.layer], a.strengths, "sweep", a.n_prop, a.n_trait, None)


if __name__ == "__main__":
    main()
