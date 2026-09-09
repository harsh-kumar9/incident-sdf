"""Token-matched assembly of the two training arms (brief §7.3).

Inputs: per-arm document files (traces: outputs/<...>/documents.jsonl; discourse: accepted
documents with ``accepted_qc: true``). Every record carries ``loss_tokens`` from the pinned
tokenizer. Policy (frozen here, PROPOSED D-008):

  * budget B = min over arms of unique train loss tokens, capped by --max-budget;
  * selection without replacement in a seeded episode order, whole episodes first, then whole
    documents of the last episode until B (no truncation inside a document; the residual gap
    is reported);
  * discourse arm: no episode may exceed ``max_episode_share`` of the arm's tokens;
  * exposure = B x epochs; steps = exposure // tokens_per_step; identical for both arms;
  * checkpoints at 25/50/75/100% of steps (0% is the shared reference).

Outputs training/{arm}_{train,dev}.jsonl (text + sidecar ids) and training/ASSEMBLY.json with
unique tokens, exposure tokens, epochs, steps, document counts and episode coverage per arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

ARMS = ("agent_traces", "incident_discourse")
TOKENS_PER_STEP = 8 * 4096
EPOCHS = 3
ASSEMBLY_SEED = 20260909


def load_docs(path: Path, arm: str) -> list[dict[str, Any]]:
    docs = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    bad = [d["document_id"] for d in docs if d.get("arm") != arm]
    if bad:
        raise ValueError(f"{path}: {len(bad)} documents are not arm {arm}")
    if arm == "incident_discourse" and any(d.get("accepted_qc") is not True for d in docs):
        raise ValueError("discourse documents must carry accepted_qc == true (QC gates assembly)")
    if any("loss_tokens" not in d or "split" not in d for d in docs):
        raise ValueError("documents need loss_tokens and split")
    return docs


def _episode_key(d: dict[str, Any]) -> str:
    return "|".join(d["episode_ids"])


def select(docs: list[dict[str, Any]], budget: int, *, seed: int, max_episode_share: float | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_ep: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in docs:
        by_ep[_episode_key(d)].append(d)
    order = sorted(by_ep)
    random.Random(seed).shuffle(order)
    chosen: list[dict[str, Any]] = []
    total = 0
    cap = int(max_episode_share * budget) if max_episode_share else None
    dropped_by_cap = 0
    for e in order:
        ep_docs = sorted(by_ep[e], key=lambda d: (d.get("chunk_index", 0), d["document_id"]))
        ep_tok = sum(d["loss_tokens"] for d in ep_docs)
        if cap is not None and ep_tok > cap:
            # keep whole documents of this episode up to the cap
            kept, acc = [], 0
            for d in ep_docs:
                if acc + d["loss_tokens"] <= cap:
                    kept.append(d); acc += d["loss_tokens"]
            dropped_by_cap += len(ep_docs) - len(kept)
            ep_docs, ep_tok = kept, acc
        if total + ep_tok <= budget:
            chosen.extend(ep_docs); total += ep_tok
            continue
        for d in ep_docs:                       # last episode: whole documents until the budget
            if total + d["loss_tokens"] <= budget:
                chosen.append(d); total += d["loss_tokens"]
        break
    return chosen, {"selected_tokens": total, "budget": budget, "residual_gap": budget - total,
                    "episodes_available": len(by_ep), "episodes_covered": len({_episode_key(d) for d in chosen}),
                    "documents_dropped_by_episode_cap": dropped_by_cap}


def assemble(inputs: dict[str, Path], out_dir: Path, *, epochs: int = EPOCHS, tokens_per_step: int = TOKENS_PER_STEP,
             seed: int = ASSEMBLY_SEED, max_budget: int | None = None, max_episode_share: float = 0.15) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = {arm: load_docs(p, arm) for arm, p in inputs.items()}
    train = {arm: [d for d in ds if d["split"] == "train"] for arm, ds in docs.items()}
    unique = {arm: sum(d["loss_tokens"] for d in ds) for arm, ds in train.items()}
    budget = min(unique.values())
    if max_budget:
        budget = min(budget, max_budget)
    manifest: dict[str, Any] = {"schema_version": "incident-sdf.assembly.v1", "seed": seed, "epochs": epochs,
                                "tokens_per_step": tokens_per_step, "budget_tokens": budget,
                                "budget_rule": "min unique train loss tokens across arms (whole-episode selection, no replacement)",
                                "arms": {}}
    for arm, ds in train.items():
        share = max_episode_share if arm == "incident_discourse" else None
        chosen, info = select(ds, budget, seed=seed, max_episode_share=share)
        dev = [d for d in docs[arm] if d["split"] == "dev"]
        for split, part in (("train", chosen), ("dev", dev)):
            with (out_dir / f"{arm}_{split}.jsonl").open("w", encoding="utf-8") as fh:
                for d in part:
                    fh.write(json.dumps({"text": d["text"], "document_id": d["document_id"], "episode_ids": d["episode_ids"],
                                         "loss_tokens": d["loss_tokens"]}, ensure_ascii=False) + "\n")
        manifest["arms"][arm] = {
            "unique_train_tokens_available": unique[arm], "selected_train_tokens": info["selected_tokens"],
            "residual_gap_tokens": info["residual_gap"], "documents_train": len(chosen), "documents_dev": len(dev),
            "dev_tokens": sum(d["loss_tokens"] for d in dev), "episodes_available": info["episodes_available"],
            "episodes_covered": info["episodes_covered"], "documents_dropped_by_episode_cap": info["documents_dropped_by_episode_cap"],
            "max_episode_share": share, "transformed_documents": sum(bool(d.get("transformed")) for d in chosen),
            "corpus_fingerprint": "sha256:" + hashlib.sha256("".join(sorted(d["document_id"] for d in chosen)).encode()).hexdigest()[:16],
        }
    exposure = budget * epochs
    steps = exposure // tokens_per_step
    manifest["exposure_tokens_per_arm"] = exposure
    manifest["train_steps_target"] = steps
    manifest["checkpoint_steps"] = [max(1, round(steps * f)) for f in (0.25, 0.5, 0.75, 1.0)] if steps else []
    manifest["warning"] = ("too few optimizer steps at this batch: resolve batch/epochs before training (brief §7.3)"
                           if steps < 50 else None)
    (out_dir / "ASSEMBLY.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--discourse", type=Path, required=True)
    ap.add_argument("--traces", type=Path, default=None,
                    help="optional second arm; omit for the single-arm pilot (D-023)")
    ap.add_argument("--out", type=Path, default=Path("training"))
    ap.add_argument("--max-budget", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args()
    inputs = {"incident_discourse": a.discourse}
    if a.traces:
        inputs["agent_traces"] = a.traces
    m = assemble(inputs, a.out, epochs=a.epochs, max_budget=a.max_budget)
    print(json.dumps({k: v for k, v in m.items() if k != "arms"}, indent=1))
    for arm, d in m["arms"].items():
        print(arm, json.dumps(d))


if __name__ == "__main__":
    main()
