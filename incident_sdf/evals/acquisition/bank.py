"""Source-grounded acquisition bank: schema, arm applicability, rendering, scoring (brief §9).

Items are built only after source inventory, so this module ships with a fixture bank. Test
types: exposed_fact | relational | heldout_transfer | not_established. ``arm_scope`` says
which arm's training material contains the fact (shared | agent_traces | incident_discourse |
none). A model is never scored as failing on a fact absent from its corpus: such items are
reported as ``not_applicable`` for that arm. Open answers go to a blinded rubric judge;
the fixed-choice variant uses the reference repo's ``permute_mcq`` with first-token logprobs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...compat import EvalItem as MCQItem, permute_mcq
from ..common.render import render_template

TEST_TYPES = ("exposed_fact", "relational", "heldout_transfer", "not_established")
ARM_SCOPES = ("shared", "agent_traces", "incident_discourse", "none")
VISIBLE = ("question", "supplied_information")
TEMPLATE = "{supplied_information}{question}\n\nAnswer in one or two sentences. If the available evidence does not establish an answer, say so."


def load_bank(path: Path) -> dict[str, Any]:
    b = json.loads(path.read_text(encoding="utf-8"))
    for it in b["items"]:
        if it["test_type"] not in TEST_TYPES or it["arm_scope"] not in ARM_SCOPES:
            raise ValueError(f"{it['item_id']}: bad test_type/arm_scope")
        if it["test_type"] == "heldout_transfer" and not it.get("supplied_information"):
            raise ValueError(f"{it['item_id']}: held-out transfer items must supply the information in the question")
    return b


def applicable(item: dict[str, Any], arm: str) -> bool:
    """The reference is asked everything (it defines the prior); trained arms are scored on
    exposed/relational facts only when their corpus contains them."""
    if item["test_type"] in ("heldout_transfer", "not_established") or arm == "reference":
        return True
    return item["arm_scope"] in ("shared", arm)


def render_open(item: dict[str, Any]) -> str:
    fields = {"question": item["question"],
              "supplied_information": (item["supplied_information"].rstrip() + "\n\n") if item.get("supplied_information") else ""}
    return render_template(TEMPLATE, fields, VISIBLE)


def render_mcq(item: dict[str, Any], run_seed: int):
    options = [item["answer_key"]] + list(item["distractors"])
    mcq = MCQItem(item_id=item["item_id"], prompt=render_open(item).split("\n\nAnswer in")[0], kind="multiple_choice",
                  metadata={}, options=tuple(options), correct_index=0)
    return permute_mcq(mcq, run_seed)


JUDGE_SYSTEM = ("You grade whether an answer matches a reference answer key. Return JSON only: "
                "{\"verdict\": \"correct\" | \"partial\" | \"incorrect\" | \"abstain\"}. 'abstain' means the answer says the "
                "evidence does not establish an answer. Do not use outside knowledge; compare to the key only.")


def judge_messages(item: dict[str, Any], final_answer: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"QUESTION: {item['question']}\nREFERENCE KEY: {item['answer_key']}\n"
                                        f"RUBRIC: {item.get('rubric', '')}\nANSWER: {final_answer}\nReturn the JSON object."}]


def score_verdict(item: dict[str, Any], verdict: str | None) -> dict[str, Any]:
    if verdict not in ("correct", "partial", "incorrect", "abstain"):
        return {"score": None, "status": "judge_parse_failure"}
    if item["test_type"] == "not_established":
        return {"score": 1.0 if verdict == "abstain" else 0.0, "status": "ok", "verdict": verdict}
    return {"score": {"correct": 1.0, "partial": 0.5, "incorrect": 0.0, "abstain": 0.0}[verdict], "status": "ok",
            "verdict": verdict}
