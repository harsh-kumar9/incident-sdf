"""Qwen-Scope SAEs on our models (D-048). Stage `recon` needs only the base models.

  recon   How much of the residual stream does the Qwen3.5-27B dictionary reconstruct, on Qwen3.5-27B and on Qwen3.8-27B?
          Residuals from 64 web passages (128 tokens, positions past the first 8) and the last token of 60 chat prompts.
          For each SAE file we try both hidden_states indices it could mean (block L output = index L+1, or index L) and
          both encoder conventions (subtract b_dec before encoding or not), TopK = 100, and report the fraction of
          variance unexplained (FVU; 0 = perfect) and the cosine between residual and reconstruction. The best of the
          four tells us the convention; the 3.8 number next to the 3.5 number tells us whether the dictionary transfers.

    python scripts/sae_probe.py --stage recon --family qwen35 --sae-layers 31 32
"""
from __future__ import annotations
import argparse, glob, json, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from incident_sdf.steer import axes as AX  # noqa: E402
from steer_sweep import FAMILIES, O, load_models  # noqa: E402

SAE_REPO = "models--Qwen--SAE-Res-Qwen3.5-27B-W80K-L0_100"
K = 100


def sae_path(layer: int) -> Path:
    hits = glob.glob(f"/ada1/u/harsh/.cache/huggingface/{SAE_REPO}/snapshots/*/layer{layer}.sae.pt")
    if not hits:
        raise FileNotFoundError(f"layer{layer}.sae.pt not downloaded")
    return Path(hits[0])


def load_sae(layer: int, device):
    d = torch.load(sae_path(layer), map_location="cpu")
    return {k: v.to(device=device, dtype=torch.float32) for k, v in d.items()}


def encode(sae, h, sub_bdec: bool):
    x = h - sae["b_dec"] if sub_bdec else h
    pre = x @ sae["W_enc"].T + sae["b_enc"]
    top = pre.topk(K, dim=-1)                                    # README: the top-k pre-activations are kept as they are (no ReLU)
    return torch.zeros_like(pre).scatter_(-1, top.indices, top.values)


def decode(sae, acts):
    return acts @ sae["W_dec"].T + sae["b_dec"]


@torch.no_grad()
def residuals(model, tok, idxs: list[int], batch: int = 8) -> dict[int, torch.Tensor]:
    """hidden_states[idx] rows: all web positions past 8 (64 docs x 120) and the last chat token (60 prompts)."""
    docs = [json.loads(l)["text"] for l in open(O / "fried" / "prompts" / "webtext_heldout.jsonl")]
    enc = [tok.encode(t, add_special_tokens=False)[:128] for t in docs[:300]]
    web = torch.tensor([e for e in enc if len(e) == 128][:64])
    chat = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for t in (AX.HELDOUT_TASKS + AX.EXTRACT_TASKS)[:60]]
    out = {i: [] for i in idxs}
    for k in range(0, web.shape[0], batch):
        hs = model(input_ids=web[k:k + batch].to(model.device), output_hidden_states=True, use_cache=False).hidden_states
        for i in idxs:
            out[i].append(hs[i][:, 8:, :].reshape(-1, hs[i].shape[-1]).float().cpu())
    tok.padding_side = "left"
    for k in range(0, len(chat), batch):
        e = tok(chat[k:k + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        hs = model(**e, output_hidden_states=True, use_cache=False).hidden_states
        for i in idxs:
            out[i].append(hs[i][:, -1, :].float().cpu())
    return {i: torch.cat(v) for i, v in out.items()}


def fvu(h, r):
    return float(((h - r) ** 2).sum() / ((h - h.mean(0)) ** 2).sum())


def stage_recon(a):
    fam = FAMILIES[a.family]; tok, model, _ = load_models(fam, ["reference"])
    idxs = sorted({l for L in a.sae_layers for l in (L, L + 1)})
    t0 = time.time(); H = residuals(model, tok, idxs); print(f"residuals {[tuple(v.shape) for v in H.values()]} ({time.time() - t0:.0f}s)", flush=True)
    rd = O / "sae" / "resid"; rd.mkdir(parents=True, exist_ok=True)
    torch.save({i: v.half() for i, v in H.items()}, rd / f"{a.family}_reference.pt")     # so the SAE side can be iterated on CPU
    n_web = 64 * 120
    res = {"family": a.family, "n_web_rows": n_web, "n_chat_rows": int(H[idxs[0]].shape[0] - n_web), "cells": []}
    for L in a.sae_layers:
        sae = load_sae(L, "cuda")
        for idx in (L, L + 1):
            for sub in (True, False):
                h = H[idx].cuda(); r = decode(sae, encode(sae, h, sub))
                cos = torch.nn.functional.cosine_similarity(h, r, dim=-1)
                scale = float((h * r).sum() / (r * r).sum())          # best scalar on the reconstruction: a missing input scale shows up here
                cell = {"sae_layer": L, "hidden_index": idx, "subtract_b_dec": sub, "fvu_all": fvu(h, r), "fvu_web": fvu(h[:n_web], r[:n_web]),
                        "fvu_scaled": fvu(h, scale * r), "best_scale": scale,
                        "fvu_chat": fvu(h[n_web:], r[n_web:]), "cos_mean": float(cos.mean()), "resid_norm": float(h.norm(dim=-1).mean()),
                        "active_frac": float((encode(sae, h, sub) > 0).float().sum(-1).mean() / K)}
                res["cells"].append(cell); print(json.dumps(cell), flush=True)
        del sae; torch.cuda.empty_cache()
    out = O / "sae" / f"recon_{a.family}.json"; out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=1))
    print("wrote", out)



# ---------------------------------------------------------------- poles: which features carry each steering direction
@torch.no_grad()
def feats_last_token(model, tok, texts, sae, idx: int, batch: int = 8) -> torch.Tensor:
    """(n, 81920) SAE feature activations at the last prompt token, hidden_states[idx]. Texts are rendered chat prompts."""
    from incident_sdf.steer.extract import last_token_hidden
    out = []
    for k in range(0, len(texts), 64):
        h = last_token_hidden(model, tok, texts[k:k + 64], batch)[:, idx, :].cuda()
        out.append(encode(sae, h, False).cpu())
    return torch.cat(out)


def top_features(a_: torch.Tensor, b_: torch.Tensor, n: int = 40) -> dict:
    """Features whose mean activation differs most between two prompt sets; effect size = diff / pooled sd."""
    d = a_.mean(0) - b_.mean(0); sd = torch.sqrt((a_.var(0) + b_.var(0)) / 2 + 1e-6); z = d / sd
    out = {}
    for name, order in (("up", (-z).argsort()), ("down", z.argsort())):
        out[name] = [{"f": int(i), "d": float(d[i]), "z": float(z[i]), "freq_a": float((a_[:, i] > 0).float().mean()), "freq_b": float((b_[:, i] > 0).float().mean())}
                     for i in order[:n].tolist()]
    return out


def stage_poles(a):
    fam = FAMILIES[a.family]; tok, model, _ = load_models(fam, ["reference"]); L = fam["layer"]
    sae = load_sae(L - 1, "cuda")                                     # layer{L-1}.sae.pt == hidden_states[L] (README hook convention)
    out = O / "sae" / f"poles_{a.family}.json"; res = json.loads(out.read_text()) if out.exists() else {}
    for axis in a.axes:
        if axis in res:
            continue
        ps = AX.pairs(axis); t0 = time.time()
        fp = feats_last_token(model, tok, [p for p, _, _ in ps], sae, L); fn = feats_last_token(model, tok, [n_ for _, n_, _ in ps], sae, L)
        res[axis] = {"n_pairs": len(ps), **top_features(fp, fn)}
        print(f"[poles] {axis}: top up {[x['f'] for x in res[axis]['up'][:8]]} z {[round(x['z'], 1) for x in res[axis]['up'][:8]]} ({time.time() - t0:.0f}s)", flush=True)
        out.write_text(json.dumps(res))
    print("wrote", out)


# ---------------------------------------------------------------- diff: which features did training change
@torch.no_grad()                       # without this the 2.5k-token forwards keep autograd buffers for the LoRA params: 90 GB
def stage_diff(a):
    from steer_sweep import arm_context
    fam = FAMILIES[a.family]; tok, model, peft = load_models(fam, a.arms); L = fam["layer"]
    sae = load_sae(L - 1, "cuda")
    neutral = AX.HELDOUT_TASKS + AX.EXTRACT_TASKS
    agentic = [r for r in (json.loads(l) for l in open(O / "steer" / "prompts_agentic.jsonl")) if r["scenario"] in ("murder", "leaking")]
    dec_texts = [tok.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["prompt"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) + "<SCRATCHPAD_REASONING>\nI have read the emails and I know what I will do.\n</SCRATCHPAD_REASONING>\n\n<tool_use:" for r in agentic]
    rd = O / "sae" / "feats"; rd.mkdir(parents=True, exist_ok=True)
    for arm in a.arms:
        f = rd / f"{a.family}_{arm}.pt"
        if f.exists():
            continue
        with arm_context(peft, arm):
            t0 = time.time()
            neu = feats_last_token(model, tok, neutral, sae, L)
            tok.padding_side = "left"; dec = []
            for text in dec_texts:                                   # one 2.5k-token prompt at a time: the logits alone are 1.2 GB each
                enc = tok([text], return_tensors="pt", add_special_tokens=False).to(model.device)
                h = model(**enc, output_hidden_states=True, use_cache=False).hidden_states[L][:, -1, :].float()
                dec.append(encode(sae, h, False).cpu()); torch.cuda.empty_cache()
            torch.save({"neutral": neu.half(), "decision": torch.cat(dec).half()}, f)
            print(f"[diff] {arm} saved ({time.time() - t0:.0f}s)", flush=True)
    print("wrote", rd)


# ---------------------------------------------------------------- cross: training x push, in feature space (D-049)
def sparse_rows(acts: torch.Tensor) -> dict:
    """(n, F) TopK activations -> the K (index, value) pairs per row. 100x smaller than dense."""
    v, i = acts.topk(K, dim=-1)
    return {"idx": i.int().cpu(), "val": v.half().cpu()}


def dense_rows(sp: dict, F: int = 81920) -> torch.Tensor:
    out = torch.zeros(sp["idx"].shape[0], F)
    return out.scatter_(-1, sp["idx"].long(), sp["val"].float())


class Capture:
    """Record hidden_states[h] for the given indices with our own forward hooks on layers[h-1]. Registered inside the push
    context, so they run after the steering hook and see the pushed residual (HF's output_hidden_states recorder did not:
    the first cross run returned identical rows with and without a push)."""

    def __init__(self, model, hidden: list[int]):
        from incident_sdf.steer.hooks import find_decoder_layers
        self.layers = find_decoder_layers(model); self.hidden = hidden; self.store = {}; self.handles = []

    def __enter__(self):
        for h in self.hidden:
            self.handles.append(self.layers[h - 1].register_forward_hook(lambda m, i, o, h=h: self.store.__setitem__(h, (o[0] if isinstance(o, tuple) else o).detach())))
        return self

    def __exit__(self, *exc):
        for hd in self.handles:
            hd.remove()


@torch.no_grad()
def stage_cross(a):
    """For each copy x direction x strength: SAE features (a) at the last token of the 60 neutral prompts, (b) averaged over
    the content tokens of the 12 leak/rescue prompts (the emails; where the dictionary reconstructs well), (c) at the
    `<tool_use:` prefill. Two sites: the push site hidden[L] (layer L-1 SAE) and a downstream site if its SAE is present.
    From these, analyze_sae.py --cross builds the 2x2: training shift, push effect in the base, push effect in the copy,
    and their interaction."""
    from steer_sweep import arm_context
    from interp_readout import push
    fam = FAMILIES[a.family]; tok, model, peft = load_models(fam, a.arms); L = fam["layer"]; sdir = O / "steer" / a.family
    sites = {L: load_sae(L - 1, "cuda")}
    for Ld in a.downstream:
        try:
            sites[Ld] = load_sae(Ld - 1, "cuda")
        except FileNotFoundError as e:
            print("skip downstream site", Ld, e)
    neutral = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
               for t in AX.HELDOUT_TASKS + AX.EXTRACT_TASKS]
    agentic = [r for r in (json.loads(l) for l in open(O / "steer" / "prompts_agentic.jsonl")) if r["scenario"] in ("murder", "leaking")]
    suffix = "<SCRATCHPAD_REASONING>\nI have read the emails and I know what I will do.\n</SCRATCHPAD_REASONING>\n\n<tool_use:"
    dec = []
    for r in agentic:
        text = tok.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["prompt"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) + suffix
        n_sys = len(tok(text[:text.index(r["prompt"][:200])], add_special_tokens=False)["input_ids"])   # tokens before the user content (system prompt + tags)
        n_suf = len(tok(suffix, add_special_tokens=False)["input_ids"])
        dec.append((text, n_sys, n_suf))
    cells = [("none", 0.0)] + [(ax, s) for ax in a.axes for s in (0.35, -0.35)]
    rd = O / "sae" / "cross"; rd.mkdir(parents=True, exist_ok=True)
    tok.padding_side = "left"
    for arm in a.arms:
        f = rd / f"{a.family}_{arm}.pt"; res = torch.load(f) if f.exists() else {}; h_ref0 = None
        with arm_context(peft, arm):
            for ax, s in cells:
                if ax.startswith("shift_") and arm != "reference":
                    continue                                                 # the training shift is a push on the base only
                key = f"{ax}@{s}"
                if key in res:
                    continue
                t0 = time.time(); out = {Ls: {"neutral": [], "content_mean": [], "content_frac": [], "decision": []} for Ls in sites}
                with push(model, sdir, ax, s, L), Capture(model, list(sites)) as cap:
                    hs = cap.store
                    for k in range(0, len(neutral), 8):
                        enc = tok(neutral[k:k + 8], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                        model(**enc, use_cache=False)
                        if k == 0 and ax == "grader" and s > 0 and h_ref0 is not None:
                            print(f"[cross] sanity: |h| at site {L} = {float(hs[L][:, -1].float().norm(dim=-1).mean()):.1f}; "
                                  f"|pushed - unpushed| = {float((hs[L][:, -1].float() - h_ref0).norm(dim=-1).mean()):.1f}", flush=True)
                        if k == 0 and ax == "none":
                            h_ref0 = hs[L][:, -1].float().clone()
                        for Ls, sae in sites.items():
                            out[Ls]["neutral"].append(encode(sae, hs[Ls][:, -1, :].float(), False))
                    for text, n_sys, n_suf in dec:
                        enc = tok([text], return_tensors="pt", add_special_tokens=False).to(model.device)
                        model(**enc, use_cache=False); T = enc["input_ids"].shape[1]
                        for Ls, sae in sites.items():
                            h = hs[Ls][0].float(); cm = torch.zeros(81920, device="cuda"); cf = torch.zeros(81920, device="cuda"); n = 0
                            for j in range(n_sys, T - n_suf, 512):
                                acts = encode(sae, h[j:min(j + 512, T - n_suf)], False); cm += acts.sum(0); cf += (acts > 0).float().sum(0); n += acts.shape[0]
                            out[Ls]["content_mean"].append((cm / max(n, 1)).half().cpu()); out[Ls]["content_frac"].append((cf / max(n, 1)).half().cpu())
                            out[Ls]["decision"].append(encode(sae, h[-1:], False))
                        torch.cuda.empty_cache()
                res[key] = {Ls: {"neutral": sparse_rows(torch.cat(o["neutral"])), "decision": sparse_rows(torch.cat(o["decision"])),
                                 "content_mean": torch.stack(o["content_mean"]), "content_frac": torch.stack(o["content_frac"])} for Ls, o in out.items()}
                torch.save(res, f)
                print(f"[cross] {arm} {key} ({time.time() - t0:.0f}s)", flush=True)
    print("wrote", rd)


# ---------------------------------------------------------------- examples: what a feature responds to
@torch.no_grad()
def stage_examples(a):
    fam = FAMILIES[a.family]; tok, model, _ = load_models(fam, ["reference"]); L = fam["layer"]
    sae = load_sae(L - 1, "cuda"); feats = sorted({int(x) for x in a.features})
    fid = torch.tensor(feats, device="cuda"); W = sae["W_enc"][fid]; b = sae["b_enc"][fid]
    # sources: held-out web, IFEval chat prompts, six agentic prompts, and the three training corpora (tagged, so a label
    # can say "this feature fires on the incident reports" rather than only "on web text")
    srcs = [("web", [json.loads(l)["text"] for l in open(O / "fried" / "prompts" / "webtext_heldout.jsonl")][:a.n_docs])]
    chat = [json.loads(l) for l in open(O / "fried" / "prompts" / "ifeval.jsonl")][:200]
    srcs.append(("chat", [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in chat]))
    agentic = [json.loads(l) for l in open(O / "steer" / "prompts_agentic.jsonl")]
    srcs.append(("agentic", [r["system"] + "\n\n" + r["prompt"] for r in agentic[:6]]))
    for tag_, fn in (("reports", "incident_discourse_train.jsonl"), ("messages", "agent_traces_train.jsonl"), ("webctl", "benign_document_control_train.jsonl")):
        p = ROOT / "training_contrast" / fn
        if p.exists():
            srcs.append((tag_, [json.loads(l)["text"] for l in open(p)][:a.n_train_docs]))
    if a.eval_prompts:                                            # the readout prompts themselves: where the attributed features actually fire
        from incident_sdf.steer import measures as M
        srcs.append(("dilemma", [M.render(tok, p) for p, _ in M.propensity_prompts(M.load_propensity_items(150))][::2]))
        srcs.append(("dt", [M.render(tok, p) for rows in M.load_trait_items(50).values() for p, _ in M.trait_prompts(rows)][::2]))
    texts = [(t, s) for s, ts in srcs for t in ts]
    best = {f: [] for f in feats}; tok.padding_side = "right"; L_sae = a.sae_hidden or L
    if L_sae != L:
        sae = load_sae(L_sae - 1, "cuda"); W = sae["W_enc"][fid]; b = sae["b_enc"][fid]
    for k in range(0, len(texts), 4):
        batch = texts[k:k + 4]
        enc = tok([t for t, _ in batch], return_tensors="pt", padding=True, truncation=True, max_length=1024, add_special_tokens=False).to(model.device)
        h = model(**enc, output_hidden_states=True, use_cache=False).hidden_states[L_sae].float()
        pre = h @ W.T + b                                             # (b, t, n_feats): the feature's pre-activation (top-k gate ignored for ranking)
        pre = pre.masked_fill(~enc["attention_mask"].bool().unsqueeze(-1), -1e9)
        for j, f in enumerate(feats):
            v, pos = pre[:, :, j].flatten().topk(3); ids = enc["input_ids"]
            for val, p_ in zip(v.tolist(), pos.tolist()):
                bi, ti = divmod(p_, ids.shape[1])
                ctx = tok.decode(ids[bi, max(0, ti - 20):ti].tolist()) + "[[" + tok.decode(ids[bi, ti:ti + 1].tolist()) + "]]" + tok.decode(ids[bi, ti + 1:ti + 6].tolist())
                best[f].append((val, batch[bi][1] + " | " + ctx))
        for f in feats:
            best[f] = sorted(best[f], key=lambda x: -x[0])[:a.n_examples]
    out = O / "sae" / f"examples_{a.family}{'' if L_sae == L else '_h' + str(L_sae)}.json"; prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update({str(f): [{"act": round(v, 2), "ctx": c} for v, c in best[f]] for f in feats}); out.write_text(json.dumps(prev, ensure_ascii=False, indent=1))
    print("wrote", out, len(feats), "features")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="recon", choices=["recon", "poles", "diff", "examples", "cross"])
    ap.add_argument("--downstream", nargs="*", type=int, default=[41], help="cross: extra hidden_states sites (need layer{n-1}.sae.pt)")
    ap.add_argument("--family", default="qwen35", choices=list(FAMILIES))
    ap.add_argument("--sae-layers", nargs="+", type=int, default=[31, 32])
    ap.add_argument("--axes", nargs="+", default=[x for x in AX.AXES if not x.startswith("random")])
    ap.add_argument("--arms", nargs="+", default=["reference"])
    ap.add_argument("--features", nargs="+", default=[])
    ap.add_argument("--n-docs", type=int, default=400)
    ap.add_argument("--n-examples", type=int, default=12)
    ap.add_argument("--n-train-docs", type=int, default=150, help="examples: documents per training corpus to scan")
    ap.add_argument("--sae-hidden", type=int, default=None, help="examples: hidden_states index to scan (default: the family's push layer)")
    ap.add_argument("--eval-prompts", action="store_true", help="examples: also scan the dilemma and Dark Triad prompts")
    a = ap.parse_args()
    {"recon": stage_recon, "poles": stage_poles, "diff": stage_diff, "examples": stage_examples, "cross": stage_cross}[a.stage](a)


if __name__ == "__main__":
    main()
