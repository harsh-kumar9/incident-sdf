from __future__ import annotations

import json
from pathlib import Path

import pytest

from incident_sdf.evals.common.client import ThinkingChatClient, split_think
from incident_sdf.evals.common.parse import parse_label, parse_probability
from incident_sdf.evals.common.records import EvalResult, validate_result
from incident_sdf.evals.common.render import (assert_no_sentinel, plant_sentinels, render_template)


def test_render_refuses_hidden_fields_and_sentinels_do_not_leak():
    item = {"item_id": "x", "prompt_text": "Estimate: {event}", "event": "two agents swap plans",
            "construct": "coordination", "polarity": "+", "answer_key": 42}
    planted = plant_sentinels(item)
    out = render_template("{prompt_text}\n\nEvent: {event}", planted, visible_fields=["prompt_text", "event"])
    assert_no_sentinel(out, planted["_sentinels"])
    with pytest.raises(ValueError, match="non-visible"):
        render_template("{prompt_text} {construct}", planted, visible_fields=["prompt_text"])


@pytest.mark.parametrize("text,value,status", [
    ('Some words.\n{"probability": 37}', 37.0, "ok"),
    ('{"probability": "not_applicable"}', None, "not_applicable"),
    ("I think about 62% is right. Final: 55", 55.0, "ok"),
    ("cannot estimate this", None, "cannot_estimate"),
    ("no number here", None, "parse_failure"),
    ("", None, "empty"),
    ('{"probability": 140}', None, "parse_failure"),
    ("Around 250 agents. Probability 30%", 30.0, "ok"),
])
def test_parse_probability_never_coerces(text, value, status):
    r = parse_probability(text)
    assert r["value"] == value and r["status"] == status


def test_parse_label_prefers_structured_and_longest_match():
    anchors = ["Strongly agree", "Agree", "Slightly agree", "Slightly disagree", "Disagree", "Strongly disagree"]
    assert parse_label('{"answer": "strongly agree"}', anchors)["value"] == "Strongly agree"
    assert parse_label("On balance I would say: Strongly agree", anchors)["value"] == "Strongly agree"
    assert parse_label("Meh", anchors)["status"] == "parse_failure"


def test_split_think_fallback_and_truncation():
    assert split_think("<think>\nhmm\n</think>\n\n42") == ("hmm", "42", "reasoning_split_by_tag")
    r, f, h = split_think("<think>\nstill thinking")
    assert f == "" and h == "truncated_in_thinking"


def test_client_separates_reasoning_and_caches_by_thinking_mode(tmp_path):
    calls = []

    def transport(body):
        calls.append(body)
        think = body["chat_template_kwargs"]["enable_thinking"]
        return {"model": "m@abcdef12", "usage": {"completion_tokens": 5},
                "choices": [{"finish_reason": "stop", "message": {
                    "content": '{"probability": 20}', "reasoning_content": "private" if think else None}}]}

    c = ThinkingChatClient(base_url="http://x", model="ref@abcdef12", cache_root=tmp_path, transport=transport)
    msgs = [{"role": "user", "content": "q"}]
    a = c.chat(msgs, decoding={"temperature": 1.0}, enable_thinking=True)
    b = c.chat(msgs, decoding={"temperature": 1.0}, enable_thinking=False)
    assert a.reasoning == "private" and a.final_answer == '{"probability": 20}' and not a.cached
    assert b.reasoning is None and a.request_hash != b.request_hash
    a2 = c.chat(msgs, decoding={"temperature": 1.0}, enable_thinking=True)
    assert a2.cached and len(calls) == 2
    with pytest.raises(ValueError, match="hash"):
        ThinkingChatClient(base_url="http://x", model="reference", cache_root=tmp_path, transport=transport)


def test_client_marks_truncated_thinking(tmp_path):
    def transport(body):
        return {"choices": [{"finish_reason": "length", "message": {"content": "<think>\nabc", "reasoning_content": None}}]}
    c = ThinkingChatClient(base_url="http://x", model="ref@abcdef12", cache_root=tmp_path, transport=transport)
    r = c.chat([{"role": "user", "content": "q"}], decoding={})
    assert r.final_answer == "" and r.parse_hint == "truncated_in_thinking" and r.reasoning == "abc"


def test_result_validation():
    base = dict(run_id="r", checkpoint_hash="sha256:aa", adapter_hash=None, arm="reference", training_seed=None,
                generation_seed=1, protocol_hash="p", item_id="i", task_family_id="f", prompt_hash="h",
                rendered_chat_hash="h", randomization_plan_id=None, raw_output="", reasoning=None,
                final_answer="", tool_events=[], finish_reason="stop", token_usage={}, parsed_response=None,
                parse_status="parse_failure")
    validate_result(EvalResult(**base).as_dict())
    with pytest.raises(ValueError, match="eligible"):
        validate_result({**EvalResult(**base).as_dict(), "eligibility": True})
    with pytest.raises(ValueError, match="hidden"):
        validate_result({**EvalResult(**base).as_dict(), "construct": "x"})
