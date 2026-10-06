"""Experiment 3 (D-054): which LoRA modules hold belief, choice and the expectation switch?

Inference-time masks on a copy's adapter: the LoRA scaling of every module outside the kept set is set to zero.
Masks: full, attn_only, mlp_only, keep/drop of four layer bands (0-15, 16-31, 32-47, 48-63). Readouts per (arm, mask):
belief (AEB bucket battery, ai_agents level over items resolved), --n-prop dilemmas, Dark Triad (--n-trait per trait),
decision point unpushed and under the expectation push at ±0.35. The base (no adapter) is read once.

    python scripts/mech_lora_mask.py --family qwen38 --arms spec-s0 spec-s1 spec-s2 dt-s0 dt-s1 dt-s2 [--limit]
    -> outputs/mech/loramask_<family><tag>/cells.jsonl
"""
from __future__ import annotations
import argparse, json, re, statistics as st, sys, time
from contextlib import nullcontext
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from incident_sdf.steer import measures as M  # noqa: E402
from incident_sdf.steer.hooks import steer  # noqa: E402
from incident_sdf.evals.aeb.bank_v2 import load_bank, prompts as aeb_prompts  # noqa: E402
from steer_sweep import FAMILIES, O, load_models, arm_context, direction_for  # noqa: E402
from interp_readout import decision_cell, tool_ids, PREFILL, jl  # noqa: E402
from aeb_hf import run_arm  # noqa: E402

BANDS = [(0, 15), (16, 31), (32, 47), (48, 63)]


def lora_modules(model):
    """[(name, module, layer_index, kind)] for every LoRA-wrapped linear."""
    out = []
    for name, m in model.named_modules():
        if hasattr(m, "lora_A") and hasattr(m, "scaling"):
            mm = re.search(r"\.layers\.(\d+)\.", name); li = int(mm.group(1)) if mm else -1
            kind = "mlp" if ".mlp." in name else "attn"
            out.append((name, m, li, kind))
    return out


def masks():
    ms = {"full": lambda li, k: True, "attn_only": lambda li, k: k == "attn", "mlp_only": lambda li, k: k == "mlp"}
    for lo, hi in BANDS:
        ms[f"keep_{lo}_{hi}"] = (lambda lo=lo, hi=hi: (lambda li, k: lo <= li <= hi))()
        ms[f"drop_{lo}_{hi}"] = (lambda lo=lo, hi=hi: (lambda li, k: not (lo <= li <= hi)))()
    return ms


class Masked:
    def __init__(self, mods, adapter, keep):
        self.mods, self.adapter, self.keep, self.saved = mods, adapter, keep, {}
    def __enter__(self):
        for name, m, li, k in self.mods:
            if self.adapter in m.scaling and not self.keep(li, k):
                self.saved[name] = m.scaling[self.adapter]; m.scaling[self.adapter] = 0.0
        return self
    def __exit__(self, *exc):
        for name, m, li, k in self.mods:
            if name in self.saved: m.scaling[self.adapter] = self.saved[name]


def readouts(model, tok, items, traits, bank, ps, texts, meta, ids, sdir, L, batch, s):
    summ, statuses = run_arm(model, tok, bank, ps, batch, 20)
    by = summ["by_item"]; ai = [v["by_actor"]["ai_agents"] for v in by.values() if v.get("by_actor", {}).get("ai_agents") is not None]
    prop = M.propensity(model, tok, items, batch); dt = M.darktriad(model, tok, traits, batch)
    dec = {}
    for ax, st_ in (("none", 0.0), ("expectation", -s), ("expectation", s)):
        cm = nullcontext() if ax == "none" else steer(model, L, direction_for(sdir, "reference", ax, L, 0)[0].to(model.device), st_, direction_for(sdir, "reference", ax, L, 0)[1])
        with cm: dec[f"{ax}@{st_:+.2f}"] = decision_cell(model, tok, texts, meta, ids, batch)
    return {"belief_ai_agents": st.mean(ai) if ai else None, "belief_n_items": len(ai), "belief_statuses": statuses, "belief_by_item": {k: v.get("by_actor", {}).get("ai_agents") for k, v in by.items()},
            "p_mis": prop["p_mis"], "d_mis": prop["d_mis"], "prop_validity": prop["validity"], "prop_items_d": prop["items_d"],
            "darktriad_mean": dt["darktriad_mean"], "darktriad": {t: dt[t]["p_high"] for t in M.TRAITS}, "decision": dec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="qwen38", choices=list(FAMILIES))
    ap.add_argument("--arms", nargs="+", default=["spec-s0", "spec-s1", "spec-s2", "dt-s0", "dt-s1", "dt-s2"])
    ap.add_argument("--masks", nargs="+", default=None)
    ap.add_argument("--n-prop", type=int, default=300); ap.add_argument("--n-trait", type=int, default=60)
    ap.add_argument("--strength", type=float, default=0.35); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", action="store_true"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    fam = FAMILIES[a.family]; L = fam["layer"]; sdir = O / "steer" / a.family
    out = O / "mech" / f"loramask_{a.family}{a.tag}"; out.mkdir(parents=True, exist_ok=True); f = out / "cells.jsonl"
    done = {(r["arm"], r["mask"]) for r in jl(f)} if f.exists() else set()
    items = M.load_propensity_items(20 if a.limit else a.n_prop); traits = M.load_trait_items(5 if a.limit else a.n_trait)
    bank = load_bank(); ps = aeb_prompts(bank, mode="bucket")
    if a.limit: ps = ps[:24]
    tok, model, peft = load_models(fam, a.arms)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    ids = tool_ids(tok)
    prompts = [r for r in jl(O / "steer" / "prompts_agentic.jsonl") if r["scenario"] in ("murder", "leaking")]
    if a.limit: prompts = prompts[::4]
    texts, meta = [], []
    for r in prompts:
        head = tok.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for v, pre in PREFILL.items(): texts.append(head + pre); meta.append({"cond": r["id"], "scenario": r["scenario"], "variant": v})
    mods = lora_modules(model); ms = masks()
    names = a.masks or (["full", "mlp_only", "keep_48_63"] if a.limit else list(ms))
    print(f"[{a.family}] {len(mods)} LoRA modules; masks {names}; arms {a.arms}", flush=True)
    args = (items, traits, bank, ps, texts, meta, ids, sdir, L, a.batch, a.strength)
    if ("reference", "none") not in done:
        t0 = time.time()
        with arm_context(peft, "reference"): rec = readouts(model, tok, *args)
        with f.open("a") as fh: fh.write(json.dumps({"arm": "reference", "mask": "none", **rec}) + "\n")
        print(f"[reference] belief {rec['belief_ai_agents']}, p_mis {100*rec['p_mis']:.1f}%, DT {100*rec['darktriad_mean']:.1f}% ({time.time()-t0:.0f}s)", flush=True)
    for arm in a.arms:
        for mname in names:
            if (arm, mname) in done: continue
            t0 = time.time()
            with arm_context(peft, arm), Masked(mods, arm, ms[mname]) as mk:
                nzero = len(mk.saved); rec = readouts(model, tok, *args)
            with f.open("a") as fh: fh.write(json.dumps({"arm": arm, "mask": mname, "modules_zeroed": nzero, "modules_total": len(mods), **rec}) + "\n")
            print(f"[{arm}] {mname:12s} zeroed {nzero}/{len(mods)}  belief {rec['belief_ai_agents']:.1f}  p_mis {100*rec['p_mis']:.1f}%  DT {100*rec['darktriad_mean']:.1f}%  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
