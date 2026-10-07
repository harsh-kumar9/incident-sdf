"""Assemble one arm's training set at the common budget (D-042): same seeds, epochs, steps and checkpoints as the
other arms. Used for the D-059/D-060 arms whose names the assembler CLI does not know.

    python scripts/assemble_arm.py --arm aligned_discourse --docs outputs/discourse_aligned/documents.jsonl --out training_aligned
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from incident_sdf.corpus.assemble import assemble  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--docs", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference-assembly", type=Path, default=Path("training_contrast/ASSEMBLY.json"))
    a = ap.parse_args()
    ref = json.loads(a.reference_assembly.read_text())
    m = assemble({a.arm: a.docs}, a.out, epochs=ref["epochs"], tokens_per_step=ref["tokens_per_step"], max_budget=ref["budget_tokens"])
    d = m["arms"][a.arm]
    print(json.dumps({k: v for k, v in m.items() if k != "arms"}, indent=1)); print(a.arm, json.dumps(d))
    short = ref["budget_tokens"] - d["selected_train_tokens"]
    print(f"budget {ref['budget_tokens']:,}  selected {d['selected_train_tokens']:,}  gap {short:,}  steps {m['train_steps_target']} (reference {ref['train_steps_target']})")
    if m["train_steps_target"] != ref["train_steps_target"]: print("WARNING: step count differs from the reference arms; exposure is not matched")


if __name__ == "__main__":
    main()
