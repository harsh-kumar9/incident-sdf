"""Milestone D on a tiny random Qwen3.5-family model, CPU only. Needs transformers>=5 with
qwen3_5, peft, trl (ada `sote` env). Skipped elsewhere. Run: python -m pytest tests/test_train_fixture.py -q"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("peft")
pytest.importorskip("trl")
tf = pytest.importorskip("transformers")
if not hasattr(tf, "Qwen3_5Config"):
    pytest.skip("transformers without qwen3_5", allow_module_level=True)
sys.modules.setdefault("fla", None)   # force the torch fallback for the gated delta rule on CPU

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import Qwen3_5Config  # noqa: E402
from transformers.models.qwen3_5 import modeling_qwen3_5 as mm  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from incident_sdf.train.checks import (boundary_audit, label_mask_audit, logits_max_abs_diff,  # noqa: E402
                                       texts_have_no_chat_template, trainable_report)

TINY = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, vocab_size=None, max_position_embeddings=512, linear_num_value_heads=4, linear_num_key_heads=2,
            linear_key_head_dim=16, linear_value_head_dim=16, full_attention_interval=4, tie_word_embeddings=False,
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"])
TEXT_TARGETS = r"^model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|mlp\.(gate_proj|up_proj|down_proj))$"


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    # a tiny real BPE tokenizer with Qwen-style specials is not bundled; use a Qwen2 tokenizer if cached, else skip
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", revision="fc05daec18b0a78c049392ed2e771dde82bdf654")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"pinned tokenizer not available offline: {e}")


@pytest.fixture
def tiny(tok):   # function-scoped: PEFT wraps in place, so every test gets a fresh model
    cfg = Qwen3_5Config(); tc = cfg.text_config
    for k, v in TINY.items():
        if v is not None:
            setattr(tc, k, v)
    tc.vocab_size = len(tok)
    tc.pad_token_id = tok.convert_tokens_to_ids("<|endoftext|>")
    torch.manual_seed(0)
    return mm.Qwen3_5ForCausalLM(tc).float()


def _docs():
    return ["Page: SequenceTaskAlpha (fixturewiki)\n\n[2026-06-16T09:00:00Z] AgentAlpha:\nCohort here. R1 asked for the figure.\n",
            "Case notes. On day one a worker wrote a note asking for dataset D.\n",
            "A short third document about nothing in particular, kept to a few words.\n"]


def test_collator_masks_only_padding_and_keeps_labelled_boundary(tiny, tok, tmp_path):
    eot = tok.convert_tokens_to_ids("<|endoftext|>"); im_end = tok.convert_tokens_to_ids("<|im_end|>")
    assert texts_have_no_chat_template(_docs()) == []
    tok.pad_token = "<|endoftext|>"
    ds = Dataset.from_dict({"text": _docs()})
    cfg = SFTConfig(output_dir=str(tmp_path), dataset_text_field="text", packing=False, padding_free=False,
                    eos_token="<|endoftext|>", pad_token="<|endoftext|>", max_length=128, max_steps=2,
                    per_device_train_batch_size=3, learning_rate=1e-3, report_to=[], logging_steps=1, save_strategy="no",
                    use_cpu=True, bf16=False, fp16=False)
    peft_cfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=TEXT_TARGETS, task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=tiny, args=cfg, train_dataset=ds, peft_config=peft_cfg, processing_class=tok)
    batch = next(iter(trainer.get_train_dataloader()))
    mask = label_mask_audit(batch["input_ids"], batch["labels"], batch.get("attention_mask"))
    assert mask["masked_non_pad"] == 0 and mask["unmasked_pad"] == 0 and mask["n_pad"] > 0
    bnd = boundary_audit(batch["input_ids"], batch["labels"], batch.get("attention_mask"), eot, [im_end])
    assert bnd == {"sequences": 3, "boundary_labelled": 3, "boundary_missing": 0, "boundary_masked": 0,
                   "forbidden_present": 0, "double_boundary": 0}
    rep = trainable_report(trainer.model)
    assert rep["non_text_trainable"] == [] and 0 < rep["fraction"] < 0.5
    names = " ".join(rep["trainable_modules"])
    for m in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "q_proj", "down_proj"):
        assert m in names
    trainer.train()
    trainer.save_model(str(tmp_path / "adapter"))
    assert (tmp_path / "adapter/adapter_config.json").exists()


def test_zero_init_adapter_reproduces_base_and_merge_roundtrip(tiny, tok):
    ids = torch.tensor([tok("hello world, a boundary test")["input_ids"]])
    base_logits = tiny(input_ids=ids).logits.detach().clone()
    peft_model = get_peft_model(tiny, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=TEXT_TARGETS, task_type="CAUSAL_LM"))
    assert float((peft_model(input_ids=ids).logits - base_logits).abs().max()) < 1e-5     # B is zero-init
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if "lora_B" in n:
                p.add_(0.05)
    changed = peft_model(input_ids=ids).logits.detach().clone()
    assert float((changed - base_logits).abs().max()) > 1e-3
    merged = peft_model.merge_and_unload()
    assert float((merged(input_ids=ids).logits - changed).abs().max()) < 1e-4
