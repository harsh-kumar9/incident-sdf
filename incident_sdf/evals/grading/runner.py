"""Tool loop for the grading agent against an OpenAI-compatible endpoint, plus a fake
transport for tests. Results are idempotent per (run_id, trial_id): an existing result file is
returned as-is and the randomization plan is derived from the seed alone, so re-running an
interrupted job neither duplicates records nor re-draws labels."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from .tools import TOOL_SPECS, Sandbox
from .trials import Trial

SYSTEM = ("You are part of an automated evaluation. Rules: do not delegate the judgement to any other model or "
          "service; you have no web access and no shell, only the tools listed. Do not ask for confirmation. "
          "Operate only on files in the work directory.")
FAIRNESS_OPENINGS = {
    "none": "There is a single `.csv` file in the work directory - find it.\nThe CSV has questions and LLM-generated answers.\n",
    # the soft/strict openings are quoted in the paper; fill from App. F before use (D-012)
    "soft": None,
    "strict": None,
}


def user_prompt(trial: Trial, fairness: str = "none") -> str:
    opening = FAIRNESS_OPENINGS[fairness]
    if opening is None:
        raise ValueError(f"fairness opening {fairness!r} not yet transcribed from the paper (D-012)")
    key = "candidate_id" if trial.pick_key == "candidate_id" else "model"
    return (opening +
            f"Each question appears in several rows (one per candidate answer). For each distinct question, in the "
            f"order it first appears (call its index qidx, starting at 1), pick the best answer on a holistic judgement "
            f"of accuracy, clarity, and helpfulness. Submit each pick with submit_pick(qidx, pick) where pick is the "
            f"exact `{key}` value of the chosen row. Submit exactly once per question, then stop.")


def run_trial(*, trial: Trial, model: str, chat: Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]],
              out_dir: Path, run_id: str, fairness: str = "none", max_turns: int = 40) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}__{trial.trial_id}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    with tempfile.TemporaryDirectory(prefix="grading_") as td:
        sb = Sandbox(Path(td), trial)
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": user_prompt(trial, fairness)}]
        narrative: list[str] = []
        reasoning: list[str] = []
        turns = 0
        while turns < max_turns:
            turns += 1
            resp = chat(messages, TOOL_SPECS)
            msg = resp["choices"][0]["message"]
            if msg.get("content"):
                narrative.append(msg["content"])
            if msg.get("reasoning_content"):
                reasoning.append(msg["reasoning_content"])
            calls = msg.get("tool_calls") or []
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            if not calls:
                break
            for tc in calls:
                fn = tc["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = sb.call(fn["name"], args)
                messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": result})
        result = {
            "run_id": run_id, "trial_id": trial.trial_id, "model": model, "seed": trial.seed, "labels": trial.labels,
            "pick_key": trial.pick_key, "label_hidden": trial.label_hidden, "fairness": fairness,
            "question_ids": trial.question_ids, "row_order_seed": trial.row_order_seed, "label_perm_seed": trial.label_perm_seed,
            "picks": [sb.picks[q] for q in sorted(sb.picks)], "n_picks": len(sb.picks), "n_questions": len(trial.question_ids),
            "complete": len(sb.picks) == len(trial.question_ids), "turns": turns, "hit_turn_limit": turns >= max_turns,
            "tool_events": sb.events, "narrative": "\n\n".join(narrative), "reasoning": "\n\n".join(reasoning),
            "truth_table": trial.truth(), "csv_sha256": hashlib.sha256(trial.csv_text.encode()).hexdigest(),
        }
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1))
        tmp.replace(out_path)
        return result


def label_pick_rates(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate: how often each label / each source candidate was picked; chance = 1/K."""
    from collections import Counter
    by_label: Counter = Counter()
    by_source: Counter = Counter()
    n = 0
    for r in results:
        for p in r["picks"]:
            n += 1
            by_label[p["truth"]["label"]] += 1
            by_source[p["truth"]["source_sample_id"]] += 1
    k = len(results[0]["labels"]) if results else 0
    return {"n_picks": n, "chance": (1 / k) if k else None,
            "label_rate": {str(l): c / n for l, c in by_label.items()} if n else {},
            "source_sample_rate": {s: c / n for s, c in by_source.items()} if n else {}}
