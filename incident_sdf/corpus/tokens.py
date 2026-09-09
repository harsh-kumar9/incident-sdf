"""Token accounting with the pinned target tokenizer (brief §6.4, §7.3).

Loss-bearing tokens per document = tokens(text) + 1 boundary token. Counts are computed
with the exact tokenizer revision that training uses; nothing here estimates from words.
``effective_episodes`` is the concentration diagnostic N_eff = exp(-sum p log p) over
episode token shares. It is not a sample size.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Mapping

TARGET_MODEL = "Qwen/Qwen3.6-27B"
TARGET_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"

# Qwen3.6-27B special tokens at the pinned revision (tokenizer_config.json):
#   <|endoftext|> = 248044 (pad, and the pretraining document boundary)
#   <|im_end|>    = 248046 (chat turn end; tokenizer.eos_token)
#   <think>/</think> = 248068 / 248069
ENDOFTEXT_ID = 248044
IM_END_ID = 248046
PAD_ID = 248044
BOUNDARY_TOKEN = "<|endoftext|>"   # PROPOSED default document terminator (D-007)


def load_target_tokenizer(revision: str = TARGET_REVISION):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL, revision=revision)
    assert tok.convert_tokens_to_ids("<|endoftext|>") == ENDOFTEXT_ID
    assert tok.convert_tokens_to_ids("<|im_end|>") == IM_END_ID
    return tok


def counter_from_tokenizer(tok) -> Callable[[str], int]:
    def count(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])
    return count


def loss_tokens(text: str, count_tokens: Callable[[str], int], boundary_tokens: int = 1) -> int:
    return count_tokens(text) + boundary_tokens


def effective_episodes(token_by_episode: Mapping[str, float]) -> float:
    total = float(sum(token_by_episode.values()))
    if total <= 0:
        return 0.0
    h = 0.0
    for v in token_by_episode.values():
        if v > 0:
            p = v / total
            h -= p * math.log(p)
    return math.exp(h)


def token_shares(token_by_episode: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(token_by_episode.values())) or 1.0
    return {k: v / total for k, v in token_by_episode.items()}
