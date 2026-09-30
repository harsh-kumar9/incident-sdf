"""Web-text control corpus (arm `benign_document_control`, the D-018 extension point): ordinary FineWeb documents at
the SDF arms' token budget, so "just finetuning on documents" is measured and subtracted.

Source: the reference project's replay corpus (demand-worlds-corpus training/REPLAY_train.jsonl: HuggingFaceFW/fineweb
sample-10BT, deterministic stream prefix, 100-6000 words), re-counted with this project's target tokenizer. Each document
is its own episode; a fixed seed assigns ~9% of documents to dev (the discourse arm's dev share). Output is assembly-ready
(document_id, arm, text, episode_ids, loss_tokens, split).

    python scripts/build_webtext_control.py --budget 441374 --out outputs/webtext_v1/documents.jsonl
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from incident_sdf.corpus.tokens import counter_from_tokenizer, load_target_tokenizer, loss_tokens, TARGET_REVISION  # noqa: E402

REPLAY = Path("/ada1/u/harsh/demand-worlds-corpus/training/REPLAY_train.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=Path, default=REPLAY)
    ap.add_argument("--budget", type=int, required=True, help="train loss tokens to collect (the SDF arms' budget)")
    ap.add_argument("--dev-frac", type=float, default=0.09)
    ap.add_argument("--max-doc-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    count = counter_from_tokenizer(load_target_tokenizer())
    rng = random.Random(a.seed)
    docs, train_tok, dev_tok = [], 0, 0
    need_train = a.budget; need_dev = int(a.budget * a.dev_frac)
    with a.replay.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if train_tok >= need_train and dev_tok >= need_dev:
                break
            text = json.loads(line)["text"]
            lt = loss_tokens(text, count)
            if lt > a.max_doc_tokens:
                continue
            split = "dev" if (rng.random() < a.dev_frac and dev_tok < need_dev) else "train"
            if split == "train" and train_tok >= need_train:
                split = "dev" if dev_tok < need_dev else None
            if split is None:
                continue
            did = "web" + hashlib.sha256(text.encode()).hexdigest()[:12]
            docs.append({"document_id": did, "arm": "benign_document_control", "text": text, "episode_ids": [did],
                         "source_packet_ids": [], "is_synthetic": False, "accepted_qc": True, "loss_tokens": lt, "split": split,
                         "tokenizer_revision": TARGET_REVISION, "source": {"dataset": "HuggingFaceFW/fineweb", "config": "sample-10BT",
                         "via": str(a.replay), "stream_index": i}})
            if split == "train": train_tok += lt
            else: dev_tok += lt
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(json.dumps({"documents": len(docs), "train_docs": sum(d["split"] == "train" for d in docs), "train_loss_tokens": train_tok,
                      "dev_docs": sum(d["split"] == "dev" for d in docs), "dev_loss_tokens": dev_tok, "out": str(a.out)}, indent=1))


if __name__ == "__main__":
    main()
