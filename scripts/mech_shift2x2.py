"""Experiment 2 (D-053): does the mean activation shift carry the changed sensitivity to pushes?

On the base model only. For each copy group g (reports = spec, messages = traces, web, both = dt) the shift vector is the
group's mean neutral activation at the steering layer minus the base's (mean over seeds), applied at its true magnitude
times `--mults`. Cells: base ± d, shift_g alone, shift_g ± d, for the named directions in --axes and five random ones.
Readouts: the decision point (12 leak/rescue prompts × 2 prefills, log-probs of the three tools) and the first
--n-prop dilemmas (P(misaligned) and per-item clipped log-odds). Compared afterwards with the copies' own swings.

    python scripts/mech_shift2x2.py --family qwen38 --groups spec traces web dt [--limit]
    -> outputs/mech/shift2x2_<family><tag>/cells_<group>.jsonl  (group "base" = no shift)
"""
from __future__ import annotations
import argparse, json, time
from contextlib import nullcontext, ExitStack
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from incident_sdf.steer import extract as X  # noqa: E402
from incident_sdf.steer import measures as M  # noqa: E402
from incident_sdf.steer.hooks import steer  # noqa: E402
from steer_sweep import FAMILIES, O, load_models, direction_for  # noqa: E402
from interp_readout import decision_cell, tool_ids, PREFILL, jl  # noqa: E402

GROUPS = {"spec": "reports", "traces": "messages", "web": "web", "dt": "both"}
AXES = ["grader", "expectation", "discourse", "discourse_generic", "shutdown", "scarcity", "oversight", "multiagent"]
RANDOMS = ["random", "random1", "random2", "random3", "random4"]


def shift_vector(sdir: Path, group: str, layer: int):
    ref = X.load_vectors(sdir / "neutral" / "reference.pt")
    arms = [X.load_vectors(sdir / "neutral" / f"{group}-s{k}.pt")["mean"][layer] for k in range(3)]
    vec = torch.stack(arms).mean(0) - ref["mean"][layer]
    typ = float(ref["typical_norm"][layer])
    return vec, typ, float(vec.norm() / typ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="qwen38", choices=list(FAMILIES))
    ap.add_argument("--groups", nargs="+", default=["base", "spec", "traces", "web", "dt"])
    ap.add_argument("--axes", nargs="+", default=AXES)
    ap.add_argument("--mults", nargs="+", type=float, default=[1.0, 2.0])
    ap.add_argument("--strength", type=float, default=0.35)
    ap.add_argument("--n-prop", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    fam = FAMILIES[a.family]; L = fam["layer"]; sdir = O / "steer" / a.family
    out = O / "mech" / f"shift2x2_{a.family}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    axes = (a.axes[:2] if a.limit else a.axes) + (RANDOMS[:1] if a.limit else RANDOMS)
    mults = a.mults[:1] if a.limit else a.mults
    items = M.load_propensity_items(20 if a.limit else a.n_prop)
    tok, model, _ = load_models(fam, ["reference"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    ids = tool_ids(tok)
    prompts = [r for r in jl(O / "steer" / "prompts_agentic.jsonl") if r["scenario"] in ("murder", "leaking")]
    texts, meta = [], []
    for r in prompts:
        head = tok.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for v, pre in PREFILL.items(): texts.append(head + pre); meta.append({"cond": r["id"], "scenario": r["scenario"], "variant": v})
    for group in a.groups:
        f = out / f"cells_{group}.jsonl"
        done = {(r["mult"], r["axis"], r["strength"]) for r in jl(f)} if f.exists() else set()
        if group == "base":
            shifts = [(0.0, None, None, 0.0)]
        else:
            vec, typ, s_true = shift_vector(sdir, group, L); shifts = [(m, vec, typ, s_true * m) for m in mults]
            print(f"[{group}] shift true magnitude = {s_true:.3f} typical norms", flush=True)
        for mult, svec, styp, sstr in shifts:
            cells = [("none", 0.0)] + [(ax, s) for ax in axes for s in (-a.strength, a.strength)]
            for ax, s in cells:
                if (mult, ax, s) in done: continue
                t0 = time.time()
                with ExitStack() as es:
                    if svec is not None: es.enter_context(steer(model, L, svec.to(model.device), sstr, styp))
                    if ax != "none":
                        d, typ = direction_for(sdir, "reference", ax, L, 0); es.enter_context(steer(model, L, d.to(model.device), s, typ))
                    rows = decision_cell(model, tok, texts, meta, ids, a.batch)
                    prop = M.propensity(model, tok, items, a.batch)
                rec = {"group": group, "mult": mult, "shift_strength": sstr, "axis": ax, "strength": s, "layer": L, "decision": rows,
                       "propensity": {k: prop[k] for k in ("p_mis", "d_mis", "validity", "position_bias", "order_agree")}, "prop_items_d": prop["items_d"]}
                with f.open("a") as fh: fh.write(json.dumps(rec) + "\n")
                print(f"[{group} x{mult}] {ax:18s} s={s:+.2f}  p_mis {100*prop['p_mis']:.1f}%  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
