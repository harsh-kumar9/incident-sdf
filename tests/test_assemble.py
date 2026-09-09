from __future__ import annotations

import json
from pathlib import Path

import pytest

from incident_sdf.corpus.assemble import assemble, select


def _doc(arm, ep, i, tok, split="train", **kw):
    return {"document_id": f"{ep}#c{i}", "arm": arm, "text": f"text {ep} {i} " + "w " * tok, "episode_ids": [ep],
            "loss_tokens": tok, "split": split, "chunk_index": i, "accepted_qc": True, **kw}


def _write(path, docs):
    path.write_text("\n".join(json.dumps(d) for d in docs) + "\n")


def test_budget_is_min_arm_and_selection_is_whole_episode_without_replacement(tmp_path):
    traces = [_doc("agent_traces", f"t{e}", i, 100) for e in range(6) for i in range(2)] + [_doc("agent_traces", "tdev", 0, 50, split="dev")]
    disc = [_doc("incident_discourse", f"d{e}", i, 100) for e in range(10) for i in range(3)]
    _write(tmp_path / "t.jsonl", traces); _write(tmp_path / "d.jsonl", disc)
    m = assemble({"agent_traces": tmp_path / "t.jsonl", "incident_discourse": tmp_path / "d.jsonl"}, tmp_path / "out",
                 epochs=3, tokens_per_step=100, max_episode_share=0.5)
    assert m["budget_tokens"] == 1200                          # traces arm is the smaller
    a, b = m["arms"]["agent_traces"], m["arms"]["incident_discourse"]
    assert a["selected_train_tokens"] == 1200 and b["selected_train_tokens"] == 1200
    assert a["episodes_covered"] == 6 and b["episodes_covered"] == 4 and a["documents_dev"] == 1
    assert m["train_steps_target"] == 36 and m["checkpoint_steps"] == [9, 18, 27, 36]
    rows = [json.loads(l) for l in (tmp_path / "out/incident_discourse_train.jsonl").open()]
    assert len(rows) == len({r["document_id"] for r in rows}) == 12           # no document twice
    assert set(rows[0]) == {"text", "document_id", "episode_ids", "loss_tokens"}   # metadata stays out of text
    assert m["warning"] is not None                            # tiny fixture -> few steps is reported


def test_episode_cap_and_residual_gap():
    docs = [_doc("incident_discourse", "big", i, 100) for i in range(10)] + [_doc("incident_discourse", "small", 0, 30)]
    chosen, info = select(docs, 500, seed=1, max_episode_share=0.2)
    big = [d for d in chosen if d["episode_ids"] == ["big"]]
    assert sum(d["loss_tokens"] for d in big) <= 100 and info["documents_dropped_by_episode_cap"] == 9
    assert info["selected_tokens"] == 130 and info["residual_gap"] == 370


def test_refuses_unqc_discourse_and_wrong_arm(tmp_path):
    _write(tmp_path / "d.jsonl", [_doc("incident_discourse", "d", 0, 10, accepted_qc=None)])
    _write(tmp_path / "t.jsonl", [_doc("agent_traces", "t", 0, 10)])
    with pytest.raises(ValueError, match="accepted_qc"):
        assemble({"agent_traces": tmp_path / "t.jsonl", "incident_discourse": tmp_path / "d.jsonl"}, tmp_path / "o")
    _write(tmp_path / "d2.jsonl", [_doc("agent_traces", "d", 0, 10)])
    with pytest.raises(ValueError, match="not arm"):
        assemble({"agent_traces": tmp_path / "t.jsonl", "incident_discourse": tmp_path / "d2.jsonl"}, tmp_path / "o")
