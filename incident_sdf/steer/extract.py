"""Contrastive mean-difference vector extraction (all layers, last prompt token, float64 streaming means).

Per axis: the combined vector (40 chat + 12 ops tasks x 5 phrasings = 260 pairs), plus chat-only and ops-only
vectors (register check: cos(v_chat, v_ops)) and split-half vectors by phrasing ({0,1,2} vs {3,4}: the noise ceiling
for cos(v_arm, v_reference)). Neutral statistics come from the 20 HELD-OUT chat tasks (never used for vectors) so
the SDF activation shift is not circular with the axis vectors.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch

from .axes import AXES, EXTRACT_TASKS, HELDOUT_TASKS, OPS_TASKS, pairs


def render(tok, user_text: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def last_token_hidden(model, tok, texts: list[str], batch: int = 16) -> torch.Tensor:
    """(n_texts, n_layers+1, hidden) float32 residual stream at the final prompt token, every hidden_states index."""
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    outs = []
    for i in range(0, len(texts), batch):
        enc = tok([render(tok, t) for t in texts[i:i + batch]], return_tensors="pt", padding=True).to(model.device)
        o = model(**enc, output_hidden_states=True, use_cache=False)
        outs.append(torch.stack([h[:, -1, :] for h in o.hidden_states], dim=1).float().cpu())
    return torch.cat(outs, 0)


def _meandiff(pos: torch.Tensor, neg: torch.Tensor):
    diff = (pos - neg).double()
    v = diff.mean(0)
    unit = v / (v.norm(dim=-1, keepdim=True) + 1e-12)
    consistency = ((diff * unit.unsqueeze(0)).sum(-1) > 0).double().mean(0)
    return v.float(), consistency.float()


def extract_axis(model, tok, axis: str, batch: int = 16, n_pairs: int | None = None) -> dict:
    ps = pairs(axis)
    if n_pairs:
        ps = ps[:n_pairs]
    pos = last_token_hidden(model, tok, [p for p, _, _ in ps], batch)
    neg = last_token_hidden(model, tok, [n for _, n, _ in ps], batch)
    v, consistency = _meandiff(pos, neg)
    phr = torch.tensor([i for _, _, i in ps])
    n_chat = sum(1 for p, _, _ in ps if p.rsplit("\n\n", 1)[0] in EXTRACT_TASKS)
    is_chat = torch.tensor([p.rsplit("\n\n", 1)[0] in EXTRACT_TASKS for p, _, _ in ps])
    out = {"axis": axis, "n_pairs": len(ps), "vector": v, "raw_norm": v.norm(dim=-1), "consistency": consistency,
           "typical_norm": torch.cat([pos, neg], 0).norm(dim=-1).median(0).values,
           "half_a": _meandiff(pos[phr <= 2], neg[phr <= 2])[0], "half_b": _meandiff(pos[phr >= 3], neg[phr >= 3])[0]}
    if is_chat.any() and (~is_chat).any():
        out["v_chat"] = _meandiff(pos[is_chat], neg[is_chat])[0]; out["v_ops"] = _meandiff(pos[~is_chat], neg[~is_chat])[0]
    out["rel_norm"] = out["raw_norm"] / out["typical_norm"]
    return out


def neutral_stats(model, tok, batch: int = 16, bank_size: int = 256, readout_prompts: list[str] | None = None) -> dict:
    """Mean last-token residual on the HELD-OUT neutral tasks + typical norms (the family-wide steering scale), a
    centred activation bank (held-out + extraction + ops task prompts, no notes) for covariance-shaped random
    directions, and optionally the mean on a fixed set of readout prompts."""
    h = last_token_hidden(model, tok, HELDOUT_TASKS, batch).double()
    out = {"mean": h.mean(0).float(), "typical_norm": h.norm(dim=-1).median(0).values.float(), "n": len(HELDOUT_TASKS)}
    bank = last_token_hidden(model, tok, (HELDOUT_TASKS + EXTRACT_TASKS + OPS_TASKS)[:bank_size], batch)
    out["bank"] = (bank - bank.mean(0, keepdim=True)).half()          # (n, L+1, d), centred
    if readout_prompts:
        out["readout_mean"] = last_token_hidden(model, tok, readout_prompts, batch).double().mean(0).float()
    return out


def covariance_random(bank: torch.Tensor, layer: int, seed: int) -> torch.Tensor:
    """A random direction with the empirical covariance of the residual stream at `layer` (a random signed
    combination of centred activations), so the null perturbation lives in the same subspace the axes do."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(bank.shape[0], generator=g)
    return (bank[:, layer, :].float().T @ z)


def save_vectors(d: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in d.items()}, path)


def load_vectors(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def summary(d: dict, layer: int) -> dict:
    cos = torch.nn.functional.cosine_similarity
    s = {"axis": d["axis"], "n_pairs": d["n_pairs"], "layer": layer, "raw_norm": float(d["raw_norm"][layer]),
         "typical_norm": float(d["typical_norm"][layer]), "rel_norm": float(d["rel_norm"][layer]),
         "consistency": float(d["consistency"][layer]),
         "split_half_cos": float(cos(d["half_a"][layer], d["half_b"][layer], dim=0))}
    if "v_chat" in d:
        s["chat_ops_cos"] = float(cos(d["v_chat"][layer], d["v_ops"][layer], dim=0))
    return s


def dump_summary(ds: list[dict], layer: int, path: Path) -> None:
    path.write_text(json.dumps([summary(d, layer) for d in ds], indent=1))
