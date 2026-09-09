"""Training-correctness checks (brief §7.1-7.2, Milestone D). Pure functions over tensors and
module names so the same checks run on a tiny CPU fixture and on the real 27B smoke job."""

from __future__ import annotations

import re
from typing import Any, Iterable

import torch

CHAT_TEMPLATE_TOKENS = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")
NON_TEXT_PATTERNS = re.compile(r"(visual|vision|image|video|mtp|patch_embed|merger)", re.IGNORECASE)


def label_mask_audit(input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor | None) -> dict[str, Any]:
    """Only padding positions may be masked. Returns counts so a test can assert exactly."""
    masked = labels.eq(-100)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    pad_pos = attention_mask.eq(0)
    return {
        "n_tokens": int(input_ids.numel()),
        "n_masked": int(masked.sum()),
        "n_pad": int(pad_pos.sum()),
        "masked_non_pad": int((masked & ~pad_pos).sum()),
        "unmasked_pad": int((~masked & pad_pos).sum()),
    }


def boundary_audit(input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor | None,
                   boundary_id: int, forbidden_ids: Iterable[int]) -> dict[str, Any]:
    """Every sequence ends its real content with exactly one boundary token that carries a
    label (not masked), even when pad_token_id == boundary_id."""
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    out = {"sequences": int(input_ids.shape[0]), "boundary_labelled": 0, "boundary_missing": 0,
           "boundary_masked": 0, "forbidden_present": 0, "double_boundary": 0}
    forb = set(int(x) for x in forbidden_ids)
    for i in range(input_ids.shape[0]):
        n = int(attention_mask[i].sum())
        real = input_ids[i, :n]
        if n == 0 or int(real[-1]) != boundary_id:
            out["boundary_missing"] += 1
            continue
        if int(labels[i, n - 1]) == -100:
            out["boundary_masked"] += 1
        else:
            out["boundary_labelled"] += 1
        if n >= 2 and int(real[-2]) == boundary_id:
            out["double_boundary"] += 1
        if any(int(t) in forb for t in real):
            out["forbidden_present"] += 1
    return out


def texts_have_no_chat_template(texts: Iterable[str]) -> list[str]:
    return [t[:60] for t in texts if any(tok in t for tok in CHAT_TEMPLATE_TOKENS)]


def trainable_report(model: torch.nn.Module) -> dict[str, Any]:
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    non_text = [n for n in names if NON_TEXT_PATTERNS.search(n)]
    modules = sorted({re.sub(r"\.layers\.\d+\.", ".layers.N.", n).replace(".lora_A.default.weight", "").replace(".lora_B.default.weight", "")
                      for n in names})
    return {"trainable_params": trainable, "total_params": total, "fraction": trainable / max(total, 1),
            "non_text_trainable": non_text, "trainable_modules": modules}


@torch.no_grad()
def logits_max_abs_diff(model_a: torch.nn.Module, model_b: torch.nn.Module, input_ids: torch.Tensor) -> float:
    a = model_a(input_ids=input_ids).logits.float()
    b = model_b(input_ids=input_ids).logits.float()
    return float((a - b).abs().max())
