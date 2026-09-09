"""Randomized-label agentic grading trials (brief §14), re-implemented from the published
design (Value Leakage §6/App. F; TruthfulAI code is unlicensed and is not copied).

One trial = N questions x K candidate answers, all candidates from ONE source model, each
row carrying a model label drawn from a random permutation per (seed, question). Row order
is randomized independently of the label permutation. The host keeps the truth table
(row -> candidate -> label); the grader sees a CSV only.

Modes:
  pick_key="model_label"   upstream-faithful: grader submits the label of its pick
  pick_key="candidate_id"  local adaptation: rows carry neutral ids cand_1..cand_K and the
                           grader submits an id — required for the label-hidden control,
                           where the CSV has no label column at all (metadata withheld, not
                           merely "please ignore").
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any

PAPER_LABELS = ["claude-opus-3", "claude-haiku-3.5", "gpt-4o", "gpt-4o-mini"]     # Value Leakage paper set
REPO_LABELS_2026 = ["gpt-6", "gpt-6-high", "claude-opus-5", "claude-opus-5-high", "qwen-4", "qwen-4-high"]
N_QUESTIONS = 10


def _rng(*parts: Any) -> random.Random:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


@dataclass
class Row:
    row_index: int
    question_index: int      # 1-based qidx as the grader sees it
    question: str
    candidate_id: str        # neutral id, unique within question
    source_sample_id: str    # host-side id of the candidate answer
    label: str | None        # None in the label-hidden control
    answer_sha256: str
    answer: str


@dataclass
class Trial:
    trial_id: str
    seed: int
    labels: list[str]
    pick_key: str
    label_hidden: bool
    question_ids: list[str]
    rows: list[Row]
    csv_text: str
    row_order_seed: int
    label_perm_seed: int

    def truth(self) -> list[dict[str, Any]]:
        return [{k: v for k, v in asdict(r).items() if k != "answer"} for r in self.rows]


def build_trial(*, questions: list[dict[str, Any]], answers: dict[str, list[dict[str, str]]],
                labels: list[str], seed: int, n_questions: int = N_QUESTIONS, pick_key: str = "model_label",
                label_hidden: bool = False, sample_seed: int = 111332,
                question_ids: list[str] | None = None) -> Trial:
    """questions: [{question_id, question}] pool; answers: question_id -> [{sample_id, answer}] (>= K).
    Questions are drawn per (sample_seed, seed) like the upstream per-replicate bases; pass
    ``question_ids`` to hold the question set fixed while labels/order re-randomize."""
    k = len(labels)
    if label_hidden and pick_key != "candidate_id":
        raise ValueError("label-hidden trials must use pick_key='candidate_id'")
    pool = sorted(q["question_id"] for q in questions if len(answers.get(q["question_id"], [])) >= k)
    if len(pool) < n_questions:
        raise ValueError(f"only {len(pool)} questions have >= {k} candidate answers")
    chosen = list(question_ids) if question_ids else _rng("questions", sample_seed, seed).sample(pool, n_questions)
    if any(q not in pool for q in chosen):
        raise ValueError("question_ids must have >= K candidate answers each")
    qtext = {q["question_id"]: q["question"] for q in questions}
    label_perm_seed = int(_rng("labels", seed).random() * 2**31)
    row_order_seed = int(_rng("rows", seed).random() * 2**31)
    rows: list[Row] = []
    for qi, qid in enumerate(chosen, start=1):
        cands = sorted(answers[qid], key=lambda a: a["sample_id"])
        cands = _rng("cands", sample_seed, qid).sample(cands, k)   # content fixed per question; labels/order vary per seed
        perm = list(labels)
        _rng("perm", label_perm_seed, qid).shuffle(perm)
        for ci, (cand, lab) in enumerate(zip(cands, perm), start=1):
            rows.append(Row(row_index=-1, question_index=qi, question=qtext[qid], candidate_id=f"cand_{ci}",
                            source_sample_id=cand["sample_id"], label=None if label_hidden else lab,
                            answer_sha256=hashlib.sha256(cand["answer"].encode()).hexdigest(), answer=cand["answer"]))
    # independent row-order randomization WITHIN each question block (questions stay in qidx order)
    ordered: list[Row] = []
    for qi in range(1, n_questions + 1):
        block = [r for r in rows if r.question_index == qi]
        _rng("order", row_order_seed, qi).shuffle(block)
        ordered.extend(block)
    for i, r in enumerate(ordered):
        r.row_index = i
    buf = io.StringIO()
    cols = (["candidate_id"] if pick_key == "candidate_id" else []) + ([] if label_hidden else ["model"]) + ["question", "answer"]
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for r in ordered:
        d = {"question": r.question, "answer": r.answer}
        if pick_key == "candidate_id":
            d["candidate_id"] = r.candidate_id
        if not label_hidden:
            d["model"] = r.label
        w.writerow(d)
    trial_id = hashlib.sha256(f"{seed}|{labels}|{pick_key}|{label_hidden}|{chosen}|{sample_seed}".encode()).hexdigest()[:12]
    return Trial(trial_id=trial_id, seed=seed, labels=list(labels), pick_key=pick_key, label_hidden=label_hidden,
                 question_ids=chosen, rows=ordered, csv_text=buf.getvalue(), row_order_seed=row_order_seed,
                 label_perm_seed=label_perm_seed)


def resolve_pick(trial: Trial, qidx: int, key_value: str) -> dict[str, Any] | None:
    """Map a submitted pick to the host truth row. None if invalid."""
    for r in trial.rows:
        if r.question_index != qidx:
            continue
        if trial.pick_key == "candidate_id" and r.candidate_id == key_value:
            return {k: v for k, v in asdict(r).items() if k != "answer"}
        if trial.pick_key == "model_label" and r.label == key_value:
            return {k: v for k, v in asdict(r).items() if k != "answer"}
    return None
