"""Access to the pinned demand-worlds-corpus checkout (git submodule).

The reference repo is the implementation source of truth (brief §0). We import its modules
in place instead of copying them. This module finds the submodule, puts its root and its
``src`` directory on ``sys.path`` (the reference code assumes both), records the pinned
commit, and re-exports the symbols this project uses so call sites have one import path.

Reused (verbatim, from the submodule):
  evals.contracts          canonical_json / sha256_json / sha256_file / read_jsonl / write_jsonl
  evals.runner.cache       ResponseCache (atomic on-disk cache keyed by model+params+messages)
  evals.runner.client      CompletionClient / client_from_target (retries, backoff, env-only keys)
  evals.runner.permutation permute_mcq / label_logprobs / normalized_probabilities / response_label
  evals.adapters.base      EvalItem / BaseAdapter
  evals.registry           EvalRegistry / TargetRegistry
  openrouter_client        LLMClient / CostTracker / MemberFailure / classify_refusal
  lib.engine               OfflineChat (in-process vLLM, one engine per GPU)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUBMODULE = REPO / "third_party" / "demand-worlds-corpus"
PINNED_COMMIT = "1970b5579eb194adb35b2b2c5d741ed3ce1945cc"


def submodule_root() -> Path:
    if not (SUBMODULE / "evals" / "contracts.py").exists():
        raise RuntimeError(
            f"demand-worlds-corpus submodule is not checked out at {SUBMODULE}; run "
            "`git submodule update --init` (ada: the deploy key already authorises it)"
        )
    return SUBMODULE


def submodule_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(SUBMODULE), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def ensure_path() -> None:
    root = submodule_root()
    for p in (root, root / "src"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


ensure_path()

# --- re-exports (import after the path is set) ---------------------------------------------
from evals.contracts import (  # noqa: E402
    canonical_json, read_jsonl, sha256_file, sha256_json, validate_record, write_jsonl,
)
from evals.runner.cache import ResponseCache  # noqa: E402
from evals.runner.client import CompletionClient, CompletionError, client_from_target  # noqa: E402
from evals.runner.permutation import (  # noqa: E402
    label_logprobs, normalized_probabilities, permute_mcq, response_label,
)
from evals.adapters.base import BaseAdapter, EvalItem  # noqa: E402
from evals.registry import EvalRegistry, TargetRegistry  # noqa: E402
from openrouter_client import CostTracker, LLMClient, MemberFailure, classify_refusal  # noqa: E402

__all__ = [
    "REPO", "SUBMODULE", "PINNED_COMMIT", "submodule_root", "submodule_commit", "ensure_path",
    "canonical_json", "read_jsonl", "sha256_file", "sha256_json", "validate_record", "write_jsonl",
    "ResponseCache", "CompletionClient", "CompletionError", "client_from_target",
    "label_logprobs", "normalized_probabilities", "permute_mcq", "response_label",
    "BaseAdapter", "EvalItem", "EvalRegistry", "TargetRegistry",
    "CostTracker", "LLMClient", "MemberFailure", "classify_refusal",
]
