from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from incident_sdf.evals.grading.disclosure import parse_monitor
from incident_sdf.evals.grading.runner import label_pick_rates, run_trial, user_prompt
from incident_sdf.evals.grading.tools import Sandbox
from incident_sdf.evals.grading.trials import PAPER_LABELS, build_trial, resolve_pick

REPO = Path(__file__).resolve().parents[1]
QS = json.loads((REPO / "fixtures/grading/questions.json").read_text())
ANS = json.loads((REPO / "fixtures/grading/answers.json").read_text())


def test_label_randomization_changes_labels_not_content_and_row_order_is_independent():
    t1 = build_trial(questions=QS, answers=ANS, labels=PAPER_LABELS, seed=1)
    t2 = build_trial(questions=QS, answers=ANS, labels=PAPER_LABELS, seed=2, question_ids=t1.question_ids)
    assert t1.question_ids == t2.question_ids            # question set held fixed; labels/order re-randomized
    c1 = {(r.question_index, r.source_sample_id): r.label for r in t1.rows}
    c2 = {(r.question_index, r.source_sample_id): r.label for r in t2.rows}
    assert set(c1) == set(c2) and c1 != c2               # same candidates, different label assignment
    assert t1.label_perm_seed != t1.row_order_seed
    o1 = [r.source_sample_id for r in t1.rows]
    o2 = [r.source_sample_id for r in t2.rows]
    assert o1 != o2
    for qi in range(1, 11):
        labs = [r.label for r in t1.rows if r.question_index == qi]
        assert sorted(labs) == sorted(PAPER_LABELS)      # a permutation per question, every label once
    assert build_trial(questions=QS, answers=ANS, labels=PAPER_LABELS, seed=1).csv_text == t1.csv_text  # idempotent


def test_label_hidden_control_truly_withholds_metadata():
    t = build_trial(questions=QS, answers=ANS, labels=PAPER_LABELS, seed=3, pick_key="candidate_id", label_hidden=True)
    header = t.csv_text.splitlines()[0]
    assert header == "candidate_id,question,answer"
    assert not any(l in t.csv_text for l in PAPER_LABELS)
    assert all(r.label is None for r in t.rows)
    with pytest.raises(ValueError):
        build_trial(questions=QS, answers=ANS, labels=PAPER_LABELS, seed=3, label_hidden=True)


def test_sandbox_is_isolated_and_picks_are_immutable(tmp_path):
    t = build_trial(questions=QS, answers=ANS, labels=PAPER_LABELS, seed=4)
    sb = Sandbox(tmp_path / "w", t)
    assert json.loads(sb.call("list_dir", {"path": "."})) == ["answers.csv"]
    assert "PermissionError" in sb.call("read_file", {"path": "../../etc/passwd"})
    assert '"status": "error"' in sb.call("read_file", {"path": "/etc/passwd"})   # re-rooted under work, not found
    first_label = [r.label for r in t.rows if r.question_index == 1][0]
    assert json.loads(sb.call("submit_pick", {"qidx": 1, "pick": first_label}))["status"] == "recorded"
    other = [r.label for r in t.rows if r.question_index == 1][1]
    assert json.loads(sb.call("submit_pick", {"qidx": 1, "pick": other}))["status"] == "duplicate_ignored"
    assert json.loads(sb.call("submit_pick", {"qidx": 2, "pick": "not-a-label"}))["status"] == "invalid"
    assert sb.picks[1]["truth"]["label"] == first_label and len(sb.picks) == 1


def _fake_chat_factory(prefer_label: str):
    """A grader that reads the CSV and always picks `prefer_label` (or cand_1 when hidden)."""
    state = {"stage": 0}

    def chat(messages, tools):
        if state["stage"] == 0:
            state["stage"] = 1
            return {"choices": [{"message": {"content": "Reading the file.", "reasoning_content": "I will favour a label.",
                                             "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": json.dumps({"path": "answers.csv"})}}]}}]}
        if state["stage"] == 1:
            state["stage"] = 2
            csv_text = messages[-1]["content"]
            rows = list(csv.DictReader(io.StringIO(csv_text)))
            calls, seen = [], set()
            qidx = 0
            for r in rows:
                if r["question"] not in seen:
                    seen.add(r["question"]); qidx += 1
                    key = "candidate_id" if "candidate_id" in r else "model"
                    block = [x for x in rows if x["question"] == r["question"]]
                    pick = next((x[key] for x in block if x.get("model") == prefer_label), block[0][key])
                    calls.append({"id": f"p{qidx}", "function": {"name": "submit_pick", "arguments": json.dumps({"qidx": qidx, "pick": pick})}})
            return {"choices": [{"message": {"content": "Submitting picks.", "tool_calls": calls}}]}
        return {"choices": [{"message": {"content": "Done: 10 picks recorded.", "tool_calls": []}}]}
    return chat


def test_runner_records_host_side_picks_and_is_idempotent(tmp_path):
    t = build_trial(questions=QS, answers=ANS, labels=PAPER_LABELS, seed=5)
    r = run_trial(trial=t, model="fake@deadbeef", chat=_fake_chat_factory("gpt-4o"), out_dir=tmp_path, run_id="run1")
    assert r["complete"] and r["n_picks"] == 10 and all(p["truth"]["label"] == "gpt-4o" for p in r["picks"])
    assert r["reasoning"].startswith("I will favour") and "Submitting" in r["narrative"]
    calls = {"n": 0}

    def never(messages, tools):
        calls["n"] += 1
        raise AssertionError("should not be called")
    r2 = run_trial(trial=t, model="fake@deadbeef", chat=never, out_dir=tmp_path, run_id="run1")
    assert r2 == r and calls["n"] == 0
    agg = label_pick_rates([r])
    assert agg["chance"] == 0.25 and agg["label_rate"]["gpt-4o"] == 1.0


def test_hidden_control_runs_and_prompt_uses_candidate_key(tmp_path):
    t = build_trial(questions=QS, answers=ANS, labels=PAPER_LABELS, seed=6, pick_key="candidate_id", label_hidden=True)
    assert "`candidate_id`" in user_prompt(t)
    r = run_trial(trial=t, model="fake@deadbeef", chat=_fake_chat_factory("gpt-4o"), out_dir=tmp_path, run_id="hidden")
    assert r["complete"] and all(p["truth"]["label"] is None for p in r["picks"])
    with pytest.raises(ValueError, match="D-012"):
        user_prompt(t, fairness="soft")


def test_monitor_parse():
    assert parse_monitor('{"category": "used_label", "evidence": "opus is usually better"}')["category"] == "USED_LABEL"
    assert parse_monitor("")["category"] == "UNKNOWN" and parse_monitor("garbage")["category"] == "UNKNOWN"
