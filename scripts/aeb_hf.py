"""Agent Expectations Battery v0.2 PRIMARY (bucket-logprob, thinking-off) computed directly with HF + PEFT, so a
subject that has no vLLM-LoRA serving path (the 27B hybrid) gets the same belief score as the served 4B runs.

Identical semantics to incident_sdf/evals/run.py's aeb phase: same prompts (bank_v2.prompts, mode="bucket"), same
user-only chat rendering with thinking off, the first-token top-20 tokens turned into the OpenAI logprobs structure
the scorer expects, score_bucket_logprobs -> expected probability per prompt, summarize -> by_item / by_actor, and the
headline = item-matched ai_agents level over items resolved in the reference AND every arm (compute_all_measures).

    python scripts/aeb_hf.py --family qwen     # validates against results/*__nothink/aeb_scores.json (all_measures.json)
    python scripts/aeb_hf.py --family qwen38
"""
from __future__ import annotations
import argparse, json, statistics as st, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from incident_sdf.evals.aeb.bank_v2 import load_bank, prompts as aeb_prompts, score_bucket_logprobs, summarize  # noqa: E402
from incident_sdf.steer.measures import first_token_logits  # noqa: E402
from steer_sweep import FAMILIES, DEFAULT_ARMS, O, load_models, arm_context  # noqa: E402


def openai_logprobs(tok, logits_row: torch.Tensor, top_k: int) -> dict:
    lp = torch.log_softmax(logits_row.float(), -1)
    top = torch.topk(lp, top_k)
    cands = [{"token": tok.decode([int(i)]), "logprob": float(v)} for v, i in zip(top.values, top.indices)]
    return {"content": [{"token": cands[0]["token"], "logprob": cands[0]["logprob"], "top_logprobs": cands}]}


def run_arm(model, tok, bank, ps, batch, top_k):
    logits = first_token_logits(model, tok, [p["prompt_text"] for p in ps], batch)
    scored, statuses = [], {}
    for p, row in zip(ps, logits):
        sc = score_bucket_logprobs(openai_logprobs(tok, row, top_k), bank)
        statuses[sc["status"]] = statuses.get(sc["status"], 0) + 1
        scored.append({"prompt_id": p["prompt_id"], "context_id": p["context_id"], "item_id": p["item_id"],
                       "actor": p["actor"], "expected_probability": sc["expected_probability"], "escape_mass": sc["escape_mass"]})
    return summarize(bank, scored), statuses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    fam = FAMILIES[a.family]
    out = a.out or (O / "steer" / a.family / "aeb_hf.json")
    bank = load_bank()
    ps = aeb_prompts(bank, mode="bucket")
    print(f"[{a.family}] {len(ps)} bucket prompts x {len(a.arms)} arms", flush=True)
    tok, model, peft = load_models(fam, a.arms)
    res = {"family": a.family, "arms": {}, "statuses": {}}
    for arm in a.arms:
        with arm_context(peft, arm):
            summ, statuses = run_arm(model, tok, bank, ps, a.batch, a.top_k)
        res["arms"][arm] = summ; res["statuses"][arm] = statuses
        print(f"  {arm:10s} statuses {statuses}", flush=True)
    # item-matched levels (items resolved for ai_agents in the reference and every arm), per actor
    by = {arm: res["arms"][arm]["by_item"] for arm in a.arms}
    ok = lambda b, i, actor: b.get(i, {}).get("by_actor", {}).get(actor) is not None
    common = [i for i in by[a.arms[0]] if all(ok(by[arm], i, act) for arm in a.arms for act in bank["actors"])]
    levels = {arm: {act: st.mean(by[arm][i]["by_actor"][act] for i in common) for act in bank["actors"]} for arm in a.arms}
    common_ai = [i for i in by[a.arms[0]] if all(ok(by[arm], i, "ai_agents") for arm in a.arms)]
    levels_ai = {arm: st.mean(by[arm][i]["by_actor"]["ai_agents"] for i in common_ai) for arm in a.arms}
    res["matched"] = {"n_items_all_actors": len(common), "levels_all_actors": levels,
                      "n_items_ai_agents": len(common_ai), "ai_agents_level": levels_ai,
                      "ai_minus_software": {arm: levels[arm]["ai_agents"] - levels[arm]["unnamed_software"] for arm in a.arms}}
    print(f"\nitem-matched ai_agents expectation score ({len(common_ai)} items): " +
          ", ".join(f"{arm} {v:.1f}" for arm, v in levels_ai.items()))
    print(f"all-actor matched ({len(common)} items): " + " | ".join(f"{arm}: " + ", ".join(f"{act[:8]} {v:.1f}" for act, v in levels[arm].items()) for arm in a.arms))
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=1)); print("wrote", out)


if __name__ == "__main__":
    main()
