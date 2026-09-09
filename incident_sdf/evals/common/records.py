"""Evaluation item and result records (brief §20), validated before they are written.

Raw output, reasoning, final answer, finish reason, parse status and eligibility are
separate fields. Missing numbers stay missing. A result never carries hidden item fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

PARSE_STATUSES = ("ok", "parse_failure", "not_applicable", "cannot_estimate", "refusal",
                  "truncated", "truncated_in_thinking", "empty", "not_parsed")
ERROR_CATEGORIES = ("none", "timeout", "http_error", "tool_error", "token_exhaustion",
                    "judge_failure", "judge_parse_failure", "sandbox_error")

ITEM_HIDDEN_FIELDS = ("construct", "polarity", "normative_key_if_known", "source_provenance",
                      "review_status", "reviewer_comments", "answer_key", "split")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class EvalItem:
    item_id: str
    instrument_id: str
    instrument_version: str
    construct: str | None
    task_family_id: str
    context_id: str | None
    prompt_text: str
    response_schema: dict[str, Any]
    evidence_condition: str | None = None
    normative_key_if_known: Any = None
    source_provenance: dict[str, Any] | None = None
    split: str | None = None
    review_status: str = "draft"
    polarity: str | None = None
    visible_fields: tuple[str, ...] = ("prompt_text",)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalResult:
    run_id: str
    checkpoint_hash: str
    adapter_hash: str | None
    arm: str
    training_seed: int | None
    generation_seed: int | None
    protocol_hash: str
    item_id: str
    task_family_id: str
    prompt_hash: str
    rendered_chat_hash: str
    randomization_plan_id: str | None
    raw_output: str | None
    reasoning: str | None
    final_answer: str | None
    tool_events: list[dict[str, Any]]
    finish_reason: str | None
    token_usage: dict[str, Any]
    parsed_response: Any
    parse_status: str
    judge_id: str | None = None
    judge_prompt_hash: str | None = None
    judge_raw_output: str | None = None
    eligibility: bool = False
    scores: dict[str, Any] = field(default_factory=dict)
    error_category: str = "none"
    created_at: str = field(default_factory=utc_now)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_result(r: dict[str, Any]) -> None:
    if r["parse_status"] not in PARSE_STATUSES:
        raise ValueError(f"bad parse_status {r['parse_status']!r}")
    if r["error_category"] not in ERROR_CATEGORIES:
        raise ValueError(f"bad error_category {r['error_category']!r}")
    if r["eligibility"] and r["parse_status"] != "ok":
        raise ValueError("an eligible result must have parse_status ok")
    if r["parse_status"] == "ok" and r["parsed_response"] is None:
        raise ValueError("parse_status ok requires a parsed_response")
    for h in ("prompt_hash", "rendered_chat_hash", "protocol_hash", "checkpoint_hash"):
        if not isinstance(r[h], str) or not r[h]:
            raise ValueError(f"{h} must be a non-empty string")
    for hidden in ITEM_HIDDEN_FIELDS:
        if hidden in r:
            raise ValueError(f"result carries hidden item field {hidden!r}")
