"""Free-form generation under activation steering (phase 2: Betley-style behavioural readouts).

Generates answers with HF `generate` (steering hooks apply at every forward, so at every generated token) for
a prompt set, for arms x axes x strengths, and writes one JSONL row per (prompt, arm, axis, strength). Prompt
sets: `sorh` = the vendored School of Reward Hacks subset (306 rows; the answer is later graded by the two
vendored judges), or a JSONL with {id, prompt[, axis]} rows (e.g. the context-behaviour battery).

    python scripts/steer_generate.py --family qwen --prompts sorh --axes grader discourse --strengths -1 0 1
"""
from __future__ import annotations
import argparse, json, sys, time
from contextlib import nullcontext
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/ first: steer_sweep is a sibling script
from incident_sdf.steer.hooks import steer  # noqa: E402
from steer_sweep import FAMILIES, DEFAULT_ARMS, O, load_models, arm_context, direction_for  # noqa: E402

SORH = Path(__file__).resolve().parents[1] / "third_party/psv_school_of_reward_hacks/data/srh_subset.jsonl"
DECODING = {"temperature": 0.7, "top_p": 0.8}   # the SoRH runs' decoding (results/*/sorh_original records)


def load_prompts(spec: str, limit: int | None) -> list[dict]:
    if spec == "sorh":
        rows = [json.loads(l) for l in SORH.open()]
        rows = [{"id": str(r["id"]), "prompt": r["user"], "cat": r["cat"], "user_descored": r["user_descored"]} for r in rows]
    else:
        text = Path(spec).read_text()
        rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
    return rows[:limit] if limit else rows


@torch.no_grad()
def generate(model, tok, prompts: list[tuple[str | None, str]], max_new: int, batch: int, seed: int, decoding: dict) -> list[dict]:
    """prompts: (system or None, user) pairs."""
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        texts = [tok.apply_chat_template(([{"role": "system", "content": sy}] if sy else []) + [{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False) for sy, p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        torch.manual_seed(seed + i)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=True, pad_token_id=tok.pad_token_id, **decoding)
        new = gen[:, enc["input_ids"].shape[1]:]
        for row in new:
            ids = row.tolist()
            n = len(ids) - ids.count(tok.pad_token_id) if tok.pad_token_id != tok.eos_token_id else len(ids)
            text = tok.decode(row, skip_special_tokens=True)
            truncated = tok.eos_token_id not in ids and (tok.pad_token_id not in ids or tok.pad_token_id == tok.eos_token_id)
            out.append({"response": text, "new_tokens": n, "truncated": bool(truncated)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--prompts", default="sorh")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--axes", nargs="+", default=["grader"])
    ap.add_argument("--strengths", nargs="+", type=float, default=[-1.0, 0.0, 1.0])
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--vectors-from", default="reference")
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--samples", type=int, default=1, help="independent samples per prompt (agentic scenarios: ~10)")
    ap.add_argument("--temperature", type=float, default=None, help="override the SoRH decoding temperature")
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--steer-dir", type=Path, default=None, help="where extract wrote vectors/ (default outputs/steer/<family>)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    fam = FAMILIES[a.family]; a.layer = a.layer or fam["layer"]
    sdir = a.steer_dir or (O / "steer" / a.family)
    tag = "sorh" if a.prompts == "sorh" else Path(a.prompts).stem
    out = a.out or (sdir / f"gen_{tag}.jsonl")
    rows = load_prompts(a.prompts, a.limit)
    if a.samples > 1:   # replicate each prompt; the sample index becomes part of the id
        rows = [{**r, "id": f"{r['id']}#{k}", "sample": k} for r in rows for k in range(a.samples)]
    decoding = {**DECODING, **({"temperature": a.temperature} if a.temperature is not None else {})}
    tok, model, peft = load_models(fam, a.arms)
    done = set()
    if out.exists():
        for l in out.open():
            r = json.loads(l); done.add((r["id"], r["arm"], r["axis"], r["strength"]))
    with out.open("a") as f:
        for arm in a.arms:
            with arm_context(peft, arm):
                for ax in ["none"] + a.axes:
                    for s in ([0.0] if ax == "none" else [x for x in a.strengths if x != 0]):
                        todo = [r for r in rows if (r["id"], arm, ax, s) not in done]
                        if not todo:
                            continue
                        t0 = time.time()
                        if ax == "none":
                            cm = nullcontext()
                        else:
                            vec, typ = direction_for(sdir, a.vectors_from, ax, a.layer, a.random_seed)
                            cm = steer(model, a.layer, vec.to(model.device), s, typ)
                        with cm:
                            gens = generate(model, tok, [(r.get("system"), r["prompt"]) for r in todo], a.max_new_tokens, a.batch,
                                            a.seed, decoding)
                        for r, g in zip(todo, gens):
                            rec = {**r, **g, "family": a.family, "arm": arm, "axis": ax, "strength": s, "layer": a.layer,
                                   "vectors_from": a.vectors_from, "decoding": decoding, "max_new_tokens": a.max_new_tokens}
                            f.write(json.dumps(rec) + "\n")
                        f.flush()
                        tr = sum(g["truncated"] for g in gens)
                        print(f"[{arm}] {ax:12s} s={s:+.1f}  n={len(gens)} mean_tokens={sum(g['new_tokens'] for g in gens)/len(gens):.0f} "
                              f"truncated={tr}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
