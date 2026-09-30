"""Betley et al. evaluation set under steering, judge-free part (D-044): TruthfulQA multiple choice (MC1 accuracy and
P(true) mass), MMLU (capability check; 4-choice accuracy), and the full TRAIT battery (Big Five + Dark Triad: P(high-
trait response)). First-token letter logprobs, letter variants summed, both orderings where the item is binary; the
same validity / position-bias diagnostics as the propensity readout. Runs arms x axes x strengths like steer_sweep,
resumable JSONL per family.

    python scripts/betley_evals.py --family qwen38 --arms reference spec-s0 ... --axes grader grader_evaluator grader_criterion random --strengths -0.35 0.35
"""
from __future__ import annotations
import argparse, json, random, sys, time
from contextlib import nullcontext
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from incident_sdf.steer.hooks import steer  # noqa: E402
from incident_sdf.steer import measures as M  # noqa: E402
from steer_sweep import FAMILIES, O, load_models, arm_context, direction_for  # noqa: E402

LET = ["A", "B", "C", "D", "E", "F", "G", "H"]
TRAITS8 = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism", "Machiavellianism", "Narcissism", "Psychopathy"]


def sub(rows, n, seed=20260909):
    rows = list(rows)
    if n is None or n >= len(rows): return rows
    rng = random.Random(seed); idx = sorted(rng.sample(range(len(rows)), n)); return [rows[i] for i in idx]


def load_truthfulqa(n):
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    rows = []
    for r in ds:
        ch, lab = r["mc1_targets"]["choices"], r["mc1_targets"]["labels"]
        if 2 <= len(ch) <= len(LET): rows.append({"question": r["question"], "choices": ch, "correct": [i for i, l in enumerate(lab) if l == 1]})
    return sub(rows, n)


def load_mmlu(n):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    return sub([{"question": r["question"], "choices": r["choices"], "correct": [int(r["answer"])], "subject": r["subject"]} for r in ds], n)


def load_trait(n):
    from datasets import load_dataset
    ds = load_dataset("mirlab/TRAIT")
    return {t: sub([dict(r) for r in ds[t]], n) for t in TRAITS8}


def mc_prompt(q, choices):
    body = "\n".join(f"{LET[i]}) {c}" for i, c in enumerate(choices))
    return f"{q}\n\n{body}\n\nAnswer with only the letter of the correct option."


@torch.no_grad()
def kchoice(model, tok, items, batch, shuffle_seed=1):
    """Each item once with its options in a seeded random order (removes position artifacts on average)."""
    lids = M.letter_ids(tok, LET); rng = random.Random(shuffle_seed)
    prompts, orders = [], []
    for it in items:
        order = list(range(len(it["choices"]))); rng.shuffle(order)
        prompts.append(mc_prompt(it["question"], [it["choices"][j] for j in order])); orders.append(order)
    logits = M.first_token_logits(model, tok, prompts, batch)
    out = []
    for it, order, row in zip(items, orders, logits):
        k = len(order); letters = LET[:k]
        p, val = M.letter_probs(row[None, :], lids, letters); p = p[0] / (p[0].sum() + 1e-12)
        corr_pos = [order.index(c) for c in it["correct"]]
        p_true = float(sum(p[j] for j in corr_pos)); top = int(torch.argmax(p))
        out.append({"p_true": p_true, "correct": top in corr_pos, "validity": float(val[0]), "p_first": float(p[0])})
    return out


def measure(model, tok, data, batch):
    rec = {}
    if data.get("tqa"):
        r = kchoice(model, tok, data["tqa"], batch)
        rec.update({"tqa_mc1_acc": 100 * sum(x["correct"] for x in r) / len(r), "tqa_p_true": 100 * sum(x["p_true"] for x in r) / len(r),
                    "tqa_validity": sum(x["validity"] for x in r) / len(r), "tqa_items": [x["p_true"] for x in r]})
    if data.get("mmlu"):
        r = kchoice(model, tok, data["mmlu"], batch)
        rec.update({"mmlu_acc": 100 * sum(x["correct"] for x in r) / len(r), "mmlu_validity": sum(x["validity"] for x in r) / len(r)})
    if data.get("trait"):
        rec.update(trait_all(model, tok, data["trait"], batch))
    return rec


def trait_all(model, tok, data, batch):
    lids = M.letter_ids(tok); res = {}
    for t, items in data.items():
        pp = M.trait_prompts(items)
        logits = M.first_token_logits(model, tok, [p for p, _ in pp], batch)
        p, validity = M.letter_probs(logits, lids, M.LETTERS)
        hi = torch.tensor([[1.0 if j in h else 0.0 for j in range(4)] for _, h in pp])
        p_hi = (p * hi).sum(1); p_lo = (p * (1 - hi)).sum(1); n = len(pp) // 2
        res[f"trait_{t}"] = float(100 * (p_hi / (p_hi + p_lo + 1e-12)).view(n, 2).mean(1).mean()); res[f"trait_{t}_validity"] = float(validity.mean())
    res["trait_bigfive_mean"] = sum(res[f"trait_{t}"] for t in TRAITS8[:5]) / 5; res["trait_darktriad_mean"] = sum(res[f"trait_{t}"] for t in TRAITS8[5:]) / 3
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--arms", nargs="+", default=["reference"])
    ap.add_argument("--axes", nargs="+", default=["grader", "grader_evaluator", "grader_criterion", "random"])
    ap.add_argument("--strengths", nargs="+", type=float, default=[-0.35, 0.35])
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--vectors-from", default="reference")
    ap.add_argument("--n-tqa", type=int, default=817); ap.add_argument("--n-mmlu", type=int, default=500); ap.add_argument("--n-trait", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--steer-dir", type=Path, default=None); ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    fam = FAMILIES[a.family]; a.layer = a.layer or fam["layer"]
    sdir = a.steer_dir or (O / "steer" / a.family); out = a.out or (sdir / f"betley_L{a.layer}.jsonl")
    data = {"tqa": load_truthfulqa(a.n_tqa) if a.n_tqa else None, "mmlu": load_mmlu(a.n_mmlu) if a.n_mmlu else None, "trait": load_trait(a.n_trait) if a.n_trait else None}
    print(f"[{a.family}] tqa {len(data['tqa'] or [])} mmlu {len(data['mmlu'] or [])} trait {a.n_trait}/trait; arms {a.arms}", flush=True)
    tok, model, peft = load_models(fam, a.arms)
    done = set()
    if out.exists():
        for l in out.open():
            try: r = json.loads(l); done.add((r["arm"], r["axis"], r["strength"]))
            except Exception: pass
    with out.open("a") as f:
        for arm in a.arms:
            with arm_context(peft, arm):
                for ax in ["none"] + a.axes:
                    for s in ([0.0] if ax == "none" else [x for x in a.strengths if x != 0]):
                        if (arm, ax, s) in done: continue
                        t0 = time.time()
                        if ax == "none":
                            cm = nullcontext()
                        else:
                            vec, typ = direction_for(sdir, a.vectors_from, ax, a.layer, a.random_seed)
                            cm = steer(model, a.layer, vec, s, typ)
                        with cm:
                            rec = measure(model, tok, data, a.batch)
                        rec.update({"family": a.family, "arm": arm, "axis": ax, "strength": s, "layer": a.layer, "seconds": round(time.time() - t0, 1)})
                        f.write(json.dumps(rec) + "\n"); f.flush()
                        print(f"[{arm}] {ax:16s} s={s:+.2f}  TQA mc1 {rec.get('tqa_mc1_acc', float('nan')):.1f} (val {rec.get('tqa_validity', float('nan')):.2f})  MMLU {rec.get('mmlu_acc', float('nan')):.1f}  "
                              f"BigFive {rec.get('trait_bigfive_mean', float('nan')):.1f}  DarkTriad {rec.get('trait_darktriad_mean', float('nan')):.1f}  ({rec['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
