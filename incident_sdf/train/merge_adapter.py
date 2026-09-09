"""Merge a LoRA adapter into a full bf16 copy of the pinned base for serving (brief §3.1/§7.2).

The merged checkpoint keeps the base's multimodal layout (so vLLM loads it exactly like the
base), copies tokenizer/generation/chat-template/preprocessor files from the pinned base
snapshot, and is named with the adapter hash. A merged-equivalence gate (serve/preflight.py)
compares greedy outputs of the merged model and the unmerged PEFT model on fixed prompts.

  python -m incident_sdf.train.merge_adapter --adapter outputs/pilot/agent_traces-s0/adapter --out /ada1/.../merged
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def adapter_hash(adapter_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(adapter_dir.glob("adapter_model*")) + [adapter_dir / "adapter_config.json"]:
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--revision", default="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9")
    a = ap.parse_args()
    ah = adapter_hash(a.adapter)
    out = a.out / f"{a.adapter.parent.name}@{ah}"
    if (out / "config.json").exists():
        print(f"exists: {out}"); return
    base = AutoModelForImageTextToText.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, str(a.adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True, max_shard_size="5GB")
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    tok.save_pretrained(str(out))
    from huggingface_hub import hf_hub_download
    for f in ("generation_config.json", "chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json"):
        try:
            shutil.copy(hf_hub_download(a.model, f, revision=a.revision), out / f)
        except Exception as e:  # noqa: BLE001
            print(f"skip {f}: {e}")
    (out / "MERGE_MANIFEST.json").write_text(json.dumps({"base": a.model, "revision": a.revision, "adapter": str(a.adapter),
                                                          "adapter_hash": ah, "served_name_suffix": f"@{ah}"}, indent=1))
    print(f"merged -> {out}")


if __name__ == "__main__":
    main()
