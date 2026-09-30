"""Composite arm (D-043): discourse + agents' interactions in ONE corpus at the SAME budget as every other arm
(441,374 loss tokens, 4 epochs, 53 steps), i.e. half the tokens from each component, mixed at random.

Each component is selected at budget/2 by the same seeded whole-episode assembler (incident_sdf.corpus.assemble) from
its full corpus; the two halves are concatenated and shuffled with a fixed seed (the trainer shuffles again per seed
via data_seed, so the file order only matters for readability). Output: training_contrast2/discourse_traces_{train,dev}.jsonl.

    python scripts/build_combined_arms.py --out training_contrast2
"""
from __future__ import annotations
import argparse, json, random, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from incident_sdf.corpus.assemble import assemble  # noqa: E402

ARM = "discourse_traces"
COMPONENTS = {"incident_discourse": Path("outputs/discourse_v2/documents.jsonl"), "agent_traces": Path("outputs/traces_v1/documents.jsonl")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-assembly", type=Path, default=Path("training_contrast/ASSEMBLY.json"))
    ap.add_argument("--out", type=Path, default=Path("training_contrast2"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--mix-seed", type=int, default=20260930)
    a = ap.parse_args()
    ref = json.loads(a.reference_assembly.read_text())
    budget, tps = ref["budget_tokens"], ref["tokens_per_step"]
    half = budget // 2
    a.out.mkdir(parents=True, exist_ok=True)
    halves, info = {}, {}
    with tempfile.TemporaryDirectory() as td:
        for comp, path in COMPONENTS.items():
            d = Path(td) / comp
            m = assemble({comp: path}, d, epochs=a.epochs, tokens_per_step=tps, max_budget=half)
            info[comp] = m["arms"][comp]
            print(f"{comp:20s} half selection: {info[comp]['selected_train_tokens']:,} train tokens, {info[comp]['documents_train']} docs, "
                  f"{info[comp].get('episodes_covered')} episodes")
            for split in ("train", "dev"):
                rows = []
                for l in (d / f"{comp}_{split}.jsonl").open(encoding="utf-8"):
                    r = json.loads(l); r["component"] = comp; r["arm"] = ARM; rows.append(r)
                halves.setdefault(split, []).extend(rows)
    arm = {"components": list(COMPONENTS)}
    for split, rows in halves.items():
        random.Random(a.mix_seed).shuffle(rows)                       # random mix of the two components
        (a.out / f"{ARM}_{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        arm[f"documents_{split}"] = len(rows)
        arm[f"{'selected_train' if split == 'train' else 'dev'}_tokens"] = sum(r["loss_tokens"] for r in rows)
        arm[f"{split}_tokens_by_component"] = {c: sum(r["loss_tokens"] for r in rows if r["component"] == c) for c in COMPONENTS}
    steps = (budget * a.epochs) // tps
    man = {"schema_version": "incident-sdf.assembly.v1+composite", "seed": ref["seed"], "mix_seed": a.mix_seed, "epochs": a.epochs,
           "tokens_per_step": tps, "budget_tokens": budget, "half_budget": half,
           "budget_rule": "same budget as every single arm; each component selected at budget/2 by the seeded whole-episode assembler, then mixed at random (D-043)",
           "arms": {ARM: arm}, "components": info, "exposure_tokens_per_arm": budget * a.epochs, "train_steps_target": steps,
           "checkpoint_steps": ref["checkpoint_steps"], "warning": None}
    (a.out / "ASSEMBLY.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(arm, indent=1)); print("budget", budget, "steps", steps)


if __name__ == "__main__":
    main()
