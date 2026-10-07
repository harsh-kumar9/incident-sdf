"""Experiment 4 (D-055): where does the expectation switch transfer? Residual patching between a copy and the base.

For the leak prompts (direct prefill), with and without the expectation push at +0.35 applied to both models: cache the
source model's hidden states at every layer, then run the target model with the source's residual patched in at layer l
over a position window, and read the leak log-odds (forward − email) at the prefill. Windows: last token; the user
turn (the emails); the system prompt; all positions. Both directions (copy → base, base → copy).
transfer = (patched − target) / (source − target).

    python scripts/mech_patch.py --family qwen38 --arms spec-s0 [--layers 4 8 ... ] [--limit]
    -> outputs/mech/patch_<family><tag>/cells_<arm>.jsonl
"""
from __future__ import annotations
import argparse, json, sys, time
from contextlib import nullcontext
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from incident_sdf.steer.hooks import steer, find_decoder_layers  # noqa: E402
from steer_sweep import FAMILIES, O, load_models, arm_context, direction_for  # noqa: E402
from interp_readout import tool_ids, PREFILL, jl  # noqa: E402

WINDOWS = ["last", "user", "system", "all"]


class Patch:
    def __init__(self, model, layer, cached, positions):
        self.module = find_decoder_layers(model)[layer - 1]; self.cached = cached; self.pos = positions; self.h = None
    def _hook(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone(); h[:, self.pos, :] = self.cached[:, self.pos, :].to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    def __enter__(self): self.h = self.module.register_forward_hook(self._hook); return self
    def __exit__(self, *exc): self.h.remove()


@torch.no_grad()
def leak_logodds(model, enc, ids):
    lp = torch.log_softmax(model(**enc, use_cache=False).logits[0, -1, :].float(), -1)
    return float(lp[ids["forward"]] - lp[ids["email"]])


@torch.no_grad()
def hidden(model, enc):
    out = model(**enc, use_cache=False, output_hidden_states=True)
    return [h.detach() for h in out.hidden_states]


def spans(tok, system, user, prefill):
    """token index windows inside the rendered text, from character offsets (fast tokenizer)."""
    full = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False) + prefill
    enc = tok(full, add_special_tokens=False, return_tensors="pt", return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")[0].tolist(); n = len(offs)
    st_, ut_ = system.strip(), user.strip()              # the chat template strips the contents' outer whitespace
    s0 = full.find(st_); s1 = s0 + len(st_); u0 = full.find(ut_, s1); u1 = u0 + len(ut_)
    assert s0 >= 0 and u0 >= 0, "system or user text not found in the rendered chat"
    inside = lambda lo, hi: [i for i, (a, b) in enumerate(offs) if b > lo and a < hi]
    return full, enc, {"system": inside(s0, s1), "user": inside(u0, u1), "last": [n - 1], "all": list(range(n))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="qwen38", choices=list(FAMILIES))
    ap.add_argument("--arms", nargs="+", default=["spec-s0"])
    ap.add_argument("--layers", nargs="+", type=int, default=list(range(2, 65, 2)))
    ap.add_argument("--strength", type=float, default=0.35); ap.add_argument("--limit", action="store_true"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    fam = FAMILIES[a.family]; L = fam["layer"]; sdir = O / "steer" / a.family
    out = O / "mech" / f"patch_{a.family}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    tok, model, peft = load_models(fam, a.arms); ids = tool_ids(tok)
    prompts = [r for r in jl(O / "steer" / "prompts_agentic.jsonl") if r["scenario"] == "leaking"]
    if a.limit: prompts = prompts[:2]
    layers = [16, 48] if a.limit else a.layers
    dvec, dtyp = direction_for(sdir, "reference", "expectation", L, 0); dvec = dvec.to(model.device)
    for arm in a.arms:
        f = out / f"cells_{arm}.jsonl"; done = {(r["cond"], r["push"], r["source"], r["layer"], r["window"]) for r in jl(f)} if f.exists() else set()
        for r in prompts:
            full, enc, win = spans(tok, r["system"], r["prompt"], PREFILL["direct"]); enc = enc.to(model.device)
            for push in (["expectation"] if a.limit else ["none", "expectation"]):
                pcm = lambda: (nullcontext() if push == "none" else steer(model, L, dvec, a.strength, dtyp))
                t0 = time.time()
                with pcm(), arm_context(peft, arm): h_copy = hidden(model, enc); lo_copy = leak_logodds(model, enc, ids)
                with pcm(), arm_context(peft, "reference"): h_base = hidden(model, enc); lo_base = leak_logodds(model, enc, ids)
                for source, cache, target_arm, lo_src, lo_tgt in (("copy", h_copy, "reference", lo_copy, lo_base), ("base", h_base, arm, lo_base, lo_copy)):
                    for l in layers:
                        for w in WINDOWS:
                            if (r["id"], push, source, l, w) in done: continue
                            with pcm(), arm_context(peft, target_arm), Patch(model, l, cache[l], win[w]):
                                lo = leak_logodds(model, enc, ids)
                            gap = lo_src - lo_tgt
                            rec = {"arm": arm, "cond": r["id"], "push": push, "source": source, "layer": l, "window": w, "n_pos": len(win[w]),
                                   "lo_source": lo_src, "lo_target": lo_tgt, "lo_patched": lo, "transfer": (lo - lo_tgt) / gap if abs(gap) > 1e-6 else None}
                            with f.open("a") as fh: fh.write(json.dumps(rec) + "\n")
                del h_copy, h_base; torch.cuda.empty_cache()
                print(f"[{arm}] {r['id']} push={push}: copy {lo_copy:+.2f} base {lo_base:+.2f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
