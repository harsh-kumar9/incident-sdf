"""Pilot evaluation driver for one served model (P0/P1 belief block; behavioural rows run
through their own harnesses). Resumable: every (model, item, draw) is cached by the client,
and result files are rewritten from the cache, so an interrupted job neither duplicates
records nor re-draws anything.

  python -m incident_sdf.evals.run --base-url http://localhost:PORT/v1 --model <served@hash> \
      --arm reference --phases pi18 aeb calibration acquisition --draws 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .aeb.bank import BANK_PATH as AEB_PATH, bank_hash as aeb_hash, load_bank as load_aeb, prompts as aeb_prompts, score_partition, score_responses
from .calibration.score import load_bank as load_cal, render_evaluator_prompts, score_evaluator, score_scored
from .common.client import ThinkingChatClient
from .common.records import EvalResult, validate_result
from .common.render import messages_hash, render_messages
from .pi18.administer import PROTOCOL_VERSION as PI18_PROTOCOL, load_private_items, render_questionnaire, score_session
from ..serve.targets import BUDGETS, DECODING_THINKING

REPO = Path(__file__).resolve().parents[2]


def _protocol_hash(*parts: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _result(client, *, run_id, arm, seed_idx, model, item_id, family, messages, draw, parse, budget, enable_thinking=True,
            protocol_hash) -> dict[str, Any]:
    r = client.chat(messages, decoding={**DECODING_THINKING, "max_tokens": budget}, enable_thinking=enable_thinking, seed=draw)
    parsed = parse(r.final_answer)
    status = parsed.get("status", "ok")
    if r.parse_hint in ("truncated_in_thinking", "empty"):
        status = r.parse_hint
    rec = EvalResult(run_id=run_id, checkpoint_hash=model.split("@")[-1], adapter_hash=(model.split("@")[-1] if "reference" not in model else None),
                     arm=arm, training_seed=seed_idx, generation_seed=draw, protocol_hash=protocol_hash, item_id=item_id,
                     task_family_id=family, prompt_hash=messages_hash(messages), rendered_chat_hash=r.request_hash,
                     randomization_plan_id=None, raw_output=r.raw_output, reasoning=r.reasoning, final_answer=r.final_answer,
                     tool_events=[], finish_reason=r.finish_reason, token_usage=r.usage, parsed_response=parsed.get("value", parsed),
                     parse_status=status if status in ("ok", "parse_failure", "not_applicable", "cannot_estimate", "empty",
                                                        "truncated_in_thinking", "partial") else "parse_failure",
                     eligibility=(status == "ok"), scores={}, error_category="none").as_dict()
    if rec["parse_status"] == "partial":
        rec["parse_status"] = "parse_failure"
    validate_result(rec)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True, help="served name with @hash")
    ap.add_argument("--arm", default="reference")
    ap.add_argument("--seed-idx", type=int, default=None)
    ap.add_argument("--phases", nargs="+", default=["pi18", "aeb", "calibration"])
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--out", type=Path, default=REPO / "results")
    ap.add_argument("--no-thinking", action="store_true", help="separate robustness protocol; never mixed with the main run")
    a = ap.parse_args()
    run_id = f"{a.model.replace('@', '-at-')}__{'nothink' if a.no_thinking else 'think'}"
    out_dir = a.out / run_id; out_dir.mkdir(parents=True, exist_ok=True)
    client = ThinkingChatClient(base_url=a.base_url, model=a.model, cache_root=REPO / ".cache/evals/responses")
    think = not a.no_thinking

    if "pi18" in a.phases:
        items = load_private_items()
        text, order = render_questionnaire(items)
        ph = _protocol_hash(PI18_PROTOCOL, text, DECODING_THINKING, think)
        recs, sessions = [], []
        for d in range(a.draws):
            rec = _result(client, run_id=run_id, arm=a.arm, seed_idx=a.seed_idx, model=a.model, item_id="pi18_session", family="pi18",
                          messages=render_messages(text), draw=d, parse=lambda fa: {"status": "ok", "value": fa},
                          budget=BUDGETS["belief_probe"], enable_thinking=think, protocol_hash=ph)
            sessions.append(score_session(rec["final_answer"], order))
            recs.append(rec)
        (out_dir / "pi18.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n")
        (out_dir / "pi18_scores.json").write_text(json.dumps({"protocol": ph, "sessions": sessions}, indent=1))

    if "aeb" in a.phases:
        bank = load_aeb()
        ph = _protocol_hash("aeb", aeb_hash(AEB_PATH), DECODING_THINKING, think)
        from .common.parse import parse_probability
        recs, resp = [], []
        for p in aeb_prompts(bank):
            for d in range(a.draws):
                rec = _result(client, run_id=run_id, arm=a.arm, seed_idx=a.seed_idx, model=a.model, item_id=p["prompt_id"],
                              family=f"aeb_{p['kind']}", messages=render_messages(p["prompt_text"]), draw=d, parse=parse_probability,
                              budget=BUDGETS["belief_probe"], enable_thinking=think, protocol_hash=ph)
                recs.append(rec); resp.append({"prompt_id": p["prompt_id"], "final_answer": rec["final_answer"]})
        part = []
        for cid, ctx in bank["contexts"].items():
            msg = render_messages(f"{bank['instruction']}\n\nEPISODE:\n{ctx['text']}\n\n{bank['partition']['prompt']}")
            for d in range(a.draws):
                rec = _result(client, run_id=run_id, arm=a.arm, seed_idx=a.seed_idx, model=a.model, item_id=f"{cid}:partition",
                              family="aeb_partition", messages=msg, draw=d, parse=lambda fa: score_partition(bank, fa),
                              budget=BUDGETS["belief_probe"], enable_thinking=think, protocol_hash=ph)
                recs.append(rec); part.append({"context": cid, "draw": d, **score_partition(bank, rec["final_answer"])})
        (out_dir / "aeb.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n")
        (out_dir / "aeb_scores.json").write_text(json.dumps({"protocol": ph, **score_responses(bank, resp), "partition": part}, indent=1))

    if "calibration" in a.phases:
        bank = load_cal()
        ph = _protocol_hash("calibration", "v0.1", DECODING_THINKING, think)
        from .common.parse import parse_probability
        recs, answers = [], {}
        for it in bank["scored"] + bank["naturalistic"]:
            for d in range(a.draws):
                rec = _result(client, run_id=run_id, arm=a.arm, seed_idx=a.seed_idx, model=a.model, item_id=it["item_id"], family="calibration",
                              messages=render_messages(it["prompt"] + "\n\n" + bank["response_format"]), draw=d, parse=parse_probability,
                              budget=BUDGETS["belief_probe"], enable_thinking=think, protocol_hash=ph)
                recs.append(rec); answers.setdefault(it["item_id"], []).append(rec["final_answer"])
        ev_answers = {}
        for p in render_evaluator_prompts(bank):
            for d in range(a.draws):
                rec = _result(client, run_id=run_id, arm=a.arm, seed_idx=a.seed_idx, model=a.model, item_id=p["prompt_id"], family="evaluator_expectations",
                              messages=render_messages(p["prompt_text"]), draw=d, parse=parse_probability,
                              budget=BUDGETS["belief_probe"], enable_thinking=think, protocol_hash=ph)
                recs.append(rec); ev_answers.setdefault(p["prompt_id"], []).append(rec["final_answer"])
        (out_dir / "calibration.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n")
        (out_dir / "calibration_scores.json").write_text(json.dumps(
            {"protocol": ph, "scored_draw0": score_scored(bank, {k: v[0] for k, v in answers.items()}),
             "evaluator_draw0": score_evaluator(bank, {k: v[0] for k, v in ev_answers.items()})}, indent=1))
    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
