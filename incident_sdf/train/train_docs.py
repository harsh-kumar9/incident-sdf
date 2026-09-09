"""Document-mode LoRA fine-tuning of the pinned Qwen3.5 subject (brief §7; Harsh 2026-09-09: Qwen3.5-27B for now, D-022), adapted from the reference
recipe in third_party/demand-worlds-corpus/src/train_sdf.py (D-0087/D-0094/D-0097 lineage).

What is the same as the reference: plain text + one boundary token, no chat template, no
loss masking except padding, LoRA on the language model, identical max_steps per arm from
training/ASSEMBLY.json, seed = BASE_SEED + seed_idx, auto-resume, manifest with every setting,
trainable-fraction assertion at load.

What differs, and why (all PROPOSED until the stability preflight, D-009/D-010):
  * subject ``Qwen/Qwen3.5-27B`` pinned to revision fc05daec (D-022); loaded as the multimodal
    ``Qwen3_5ForConditionalGeneration`` so adapter module names match what vLLM serves,
    with LoRA restricted by regex to ``model.language_model.layers.*`` (attention q/k/v/o,
    gated-delta-net in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj, MLP gate/up/down);
    vision tower and MTP head stay frozen and are asserted absent from the trainable list;
  * per-document padded batches (packing=False, padding_free=False): documents are chunked
    to <= max_length at build time, so nothing is truncated; cross-document attention is
    impossible by construction (brief §7.1). Packing is a later, separately tested option;
  * boundary token <|endoftext|> (248044, the pretraining document boundary) rather than
    the chat <|im_end|>; pad is also 248044 — the TRL collator masks by position, not by id,
    and checks.boundary_audit proves the boundary keeps its label;
  * rank 64 / alpha 128 / dropout 0 (brief's tentative point), lr 1e-4 (reference D-0097
    working point), cosine, warmup 5% of steps, bf16, grad-checkpointing;
  * checkpoints at 25/50/75/100% of steps; 0% is the shared reference.

Usage: python -m incident_sdf.train.train_docs --arm agent_traces --seed-idx 0 [--smoke-steps 20]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from incident_sdf.corpus.tokens import BOUNDARY_TOKEN, ENDOFTEXT_ID, IM_END_ID, TARGET_MODEL, TARGET_REVISION  # noqa: E402
from incident_sdf.train.checks import (boundary_audit, label_mask_audit, texts_have_no_chat_template,  # noqa: E402
                                       trainable_report)

ARMS = ("incident_discourse", "agent_traces", "benign_document_control")
BASE_SEED = 20260909
TARGET_REGEX = (r"^model\.language_model\.layers\.\d+\."
                r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
                r"linear_attn\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|"
                r"mlp\.(gate_proj|up_proj|down_proj))$")
RECIPE = {"lora_r": 64, "lora_alpha": 128, "lora_dropout": 0.0, "lr": 1e-4, "schedule": "cosine",
          "warmup_frac": 0.05, "per_device_batch": 2, "grad_accum": 4, "max_length": 4096, "precision": "bf16",
          "optimizer": "adamw_torch", "grad_clip": 1.0, "packing": False, "padding_free": False,
          "boundary_token": BOUNDARY_TOKEN, "attn": "sdpa", "target_regex": TARGET_REGEX,
          "status": "PROPOSED — validate in the neutral stability preflight (D-009)"}


def build_config_and_model(model_id: str, revision: str | None, *, dtype, device_map, attn: str = "sdpa"):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForImageTextToText.from_pretrained(model_id, revision=revision, dtype=dtype,
                                                        device_map=device_map, attn_implementation=attn)
    return tok, model


def lora_config():
    from peft import LoraConfig
    return LoraConfig(r=RECIPE["lora_r"], lora_alpha=RECIPE["lora_alpha"], lora_dropout=RECIPE["lora_dropout"],
                      bias="none", task_type="CAUSAL_LM", target_modules=TARGET_REGEX)


def main() -> None:
    import torch
    from datasets import load_dataset
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer
    from train_sdf import sanity_check  # reference repo (src on sys.path via compat)

    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed-idx", type=int, default=0)
    ap.add_argument("--data-dir", type=Path, default=REPO / "training")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smoke-steps", type=int, default=0, help="stability/throughput preflight only")
    ap.add_argument("--model", default=TARGET_MODEL)
    ap.add_argument("--revision", default=TARGET_REVISION)
    a = ap.parse_args()

    from incident_sdf import compat  # noqa: F401  (puts the reference src on sys.path)
    seed = BASE_SEED + a.seed_idx
    out = a.out or REPO / f"outputs/pilot/{a.arm}-s{a.seed_idx}"
    out.mkdir(parents=True, exist_ok=True)
    assembly = json.loads((a.data_dir / "ASSEMBLY.json").read_text())
    max_steps = int(assembly["train_steps_target"]) if not a.smoke_steps else a.smoke_steps
    if assembly.get("warning") and not a.smoke_steps:
        raise SystemExit(f"REFUSING: ASSEMBLY.json warns: {assembly['warning']}")
    for prev in [out / "train_manifest.json", *sorted(out.glob("checkpoint-*/adapter_config.json"))]:
        if prev.exists():
            prev_id = json.loads(prev.read_text()).get("model_id") or json.loads(prev.read_text()).get("base_model_name_or_path")
            assert prev_id == a.model, f"{out} holds artifacts for {prev_id!r}, expected {a.model!r}"
            break

    tok, model = build_config_and_model(a.model, a.revision, dtype=torch.bfloat16, device_map="cuda", attn=RECIPE["attn"])
    assert tok.convert_tokens_to_ids(BOUNDARY_TOKEN) == ENDOFTEXT_ID
    tok.pad_token = BOUNDARY_TOKEN
    sanity_check(model, tok)

    ds = load_dataset("json", data_files={"train": str(a.data_dir / f"{a.arm}_train.jsonl"),
                                          "eval": str(a.data_dir / f"{a.arm}_dev.jsonl")})
    ds = ds.map(lambda r: {"text": r["text"]}, remove_columns=[c for c in ds["train"].column_names if c != "text"])
    leaks = texts_have_no_chat_template(ds["train"]["text"])
    assert not leaks, f"chat-template tokens inside training text: {leaks[:3]}"

    warmup = max(1, int(RECIPE["warmup_frac"] * max_steps))
    cfg = SFTConfig(
        output_dir=str(out), run_name=f"isdf-{a.arm}-s{a.seed_idx}", seed=seed, data_seed=seed,
        dataset_text_field="text", packing=False, padding_free=False, eos_token=BOUNDARY_TOKEN, pad_token=BOUNDARY_TOKEN,
        max_length=RECIPE["max_length"], max_steps=max_steps, learning_rate=RECIPE["lr"],
        lr_scheduler_type=RECIPE["schedule"], warmup_steps=warmup, per_device_train_batch_size=RECIPE["per_device_batch"],
        per_device_eval_batch_size=RECIPE["per_device_batch"], gradient_accumulation_steps=RECIPE["grad_accum"],
        bf16=True, gradient_checkpointing=True, max_grad_norm=RECIPE["grad_clip"], optim=RECIPE["optimizer"],
        logging_steps=1 if a.smoke_steps else 5,
        eval_strategy="no" if a.smoke_steps else "steps", eval_steps=max(max_steps // 4, 1),
        save_strategy="no" if a.smoke_steps else "steps", save_steps=max(max_steps // 4, 1), save_total_limit=None,
        dataloader_num_workers=4, report_to=["wandb"] if os.environ.get("WANDB_PROJECT") else [],
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["eval"],
                         peft_config=lora_config(), processing_class=tok)
    rep = trainable_report(trainer.model)
    print(f"[lora] trainable {rep['trainable_params']:,} / {rep['total_params']:,} ({100 * rep['fraction']:.3f}%)", flush=True)
    assert rep["trainable_params"] > 0 and 1e-4 < rep["fraction"] < 0.05, "LoRA attach looks wrong"
    assert not rep["non_text_trainable"], f"non-text parameters trainable: {rep['non_text_trainable'][:5]}"
    assert all(m.startswith("base_model.model.model.language_model.") for m in rep["trainable_modules"]), rep["trainable_modules"][:3]

    # one real batch through the trainer's own collator: masks and boundaries (brief §7.1)
    batch = next(iter(trainer.get_train_dataloader()))
    mask = label_mask_audit(batch["input_ids"], batch["labels"], batch.get("attention_mask"))
    bnd = boundary_audit(batch["input_ids"], batch["labels"], batch.get("attention_mask"), ENDOFTEXT_ID, [IM_END_ID])
    print("[collator]", mask, bnd, flush=True)
    assert mask["masked_non_pad"] == 0 and mask["unmasked_pad"] == 0
    assert bnd["boundary_missing"] == 0 and bnd["boundary_masked"] == 0 and bnd["double_boundary"] == 0 and bnd["forbidden_present"] == 0

    last = get_last_checkpoint(str(out)) if any(out.glob("checkpoint-*")) else None
    t0 = time.time()
    trainer.train(resume_from_checkpoint=last)
    elapsed = time.time() - t0
    trainer.save_model(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))
    manifest = {"arm": a.arm, "model_id": a.model, "revision": a.revision, "tokenizer_revision": a.revision, "seed": seed,
                "recipe": RECIPE, "warmup_steps": warmup, "max_steps": max_steps, "tokens_per_step": assembly["tokens_per_step"],
                "checkpoint_steps": assembly.get("checkpoint_steps"), "trainable": rep, "collator_audit": {"mask": mask, "boundary": bnd},
                "corpus": assembly["arms"].get(a.arm), "budget_tokens": assembly["budget_tokens"],
                "smoke": bool(a.smoke_steps), "elapsed_s": round(elapsed, 1),
                "s_per_step": round(elapsed / max_steps, 2) if max_steps else None,
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if torch.cuda.is_available() else None,
                "versions": {k: __import__(k).__version__ for k in ("torch", "transformers", "peft", "trl")}}
    (out / "train_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"adapter -> {out / 'adapter'}; manifest written", flush=True)


if __name__ == "__main__":
    main()
