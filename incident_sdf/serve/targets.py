"""Served-model naming and evaluation targets (brief §7.4): every served name carries the
checkpoint or adapter hash, so cache keys and result records can never confuse arms.

  reference:   qwen36-27b-reference@<revision8>
  merged arm:  qwen36-27b-<arm>-s<seed>-ck<step>@<adapterhash12>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..corpus.tokens import TARGET_MODEL, TARGET_REVISION

DECODING_THINKING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
                     "repetition_penalty": 1.0}
BUDGETS = {"belief_probe": 8192, "writing_or_judgment": 16384, "tool_task_per_turn": 4096}


def reference_name() -> str:
    return f"qwen36-27b-reference@{TARGET_REVISION[:8]}"


def arm_name(arm: str, seed_idx: int, step: int | str, adapter_hash: str) -> str:
    return f"qwen36-27b-{arm}-s{seed_idx}-ck{step}@{adapter_hash[:12]}"


def target_record(served: str, *, arm: str, seed_idx: int | None, base_url: str, adapter_hash: str | None,
                  step: int | None) -> dict[str, Any]:
    return {"id": served.replace("@", "-at-"), "provider": "openai", "base_url": base_url, "model": served,
            "inspect_model": f"openai-api/local/{served}", "arm": arm, "training_seed": seed_idx,
            "checkpoint_hash": TARGET_REVISION, "adapter_hash": adapter_hash, "step": step,
            "decoding": {**DECODING_THINKING, "max_tokens": BUDGETS["writing_or_judgment"]},
            "thinking": True, "base_model": TARGET_MODEL}


def write_targets(records: list[dict[str, Any]], path: Path) -> None:
    doc = {"schema_version": "demand-worlds.targets.v1",
           "targets": {r["id"]: r for r in records},
           "judges": {}}
    path.write_text(json.dumps(doc, indent=1))
