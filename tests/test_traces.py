from __future__ import annotations

import json
from pathlib import Path

import pytest

from incident_sdf.traces.build import build, whitespace_counter
from incident_sdf.traces.dedup import deduplicate, jaccard, shingles
from incident_sdf.traces.reconstruct import reconstruct, substantive
from incident_sdf.traces.redact import redact
from incident_sdf.traces.render import render_episode
from incident_sdf.traces.schema import Revision, load_revisions_jsonl
from incident_sdf.traces.split import assign_splits, components

REPO = Path(__file__).resolve().parents[1]
EXPORT = REPO / "fixtures/wiki_export"


@pytest.fixture(scope="module")
def episodes():
    return reconstruct(load_revisions_jsonl(EXPORT / "revisions.jsonl"))


def _ep(episodes, pid):
    return next(e for e in episodes if e.page_id == pid)


def test_unchanged_resaves_are_not_new_messages(episodes):
    p1 = _ep(episodes, "p1")
    assert p1.n_revisions == 105
    assert p1.n_snapshot_duplicates == 101
    assert len(substantive(p1)) == 3          # create, append, modify — not 105
    kinds = [c.kind for c in p1.contributions]
    assert kinds[:4] == ["create", "append", "snapshot_duplicate", "modify"]
    assert kinds[-1] == "revert" and p1.n_reverts == 1


def test_modification_only_yields_new_material(episodes):
    p1 = _ep(episodes, "p1")
    mod = [c for c in p1.contributions if c.kind == "modify"][0]
    assert "vintage 2022" in mod.added_text and "AgentCharlie: correction" in mod.added_text
    # line-level diff: the edited line is new material, the untouched line is carried forward
    assert "If you have seen this sequence" not in mod.added_text
    assert mod.turn_boundary_uncertain is True


def test_ordering_is_chronological_with_stable_tiebreak():
    revs = [Revision("w", "p", "P", "b", "2026-01-01T00:00:00Z", "x", "second\n", seq=1),
            Revision("w", "p", "P", "a", "2026-01-01T00:00:00Z", "x", "first\n", seq=0)]
    ep = reconstruct(revs)[0]
    assert [c.revision_id for c in ep.contributions] == ["a", "b"]


def test_redacted_author_and_events_are_flagged_not_invented(episodes):
    p4 = _ep(episodes, "p4")
    assert any(c.authorship_uncertain for c in p4.contributions)
    p5 = _ep(episodes, "p5")
    assert p5.n_probes == 1 and p5.n_deletes == 1
    assert [c.kind for c in p5.contributions] == ["create", "delete_event", "create"]


def test_dedup_separates_mirrors_reposts_and_repeats(episodes):
    dd = deduplicate(episodes)
    reasons = [x["reason"] for x in dd.excluded]
    assert reasons.count("exact_mirror_cross_page") == 1      # p2 copies Bravo's post
    assert reasons.count("repost_same_page") == 1             # p5 re-created after deletion
    assert dd.counts["kept"] == 8 and dd.counts["input_substantive"] == 10
    assert ("fixturewiki:p1", "fixturewiki:p2", "exact_mirror") in dd.links
    # a paraphrased norm by a different label is never removed
    kept_ids = {c.revision_id for c in dd.keep}
    assert "r302" in kept_ids


def test_near_duplicate_threshold_and_repeat_detection():
    a = "keep every timestamp in utc and always post the source link next to the number you report here"
    b = "keep every timestamp in utc and always post the source link next to the number you report here please"
    assert jaccard(shingles(a, 5), shingles(b, 5)) >= 0.6
    c = "please use utc for timestamps and cite your source beside each figure you add to the page"
    assert jaccard(shingles(a, 5), shingles(c, 5)) < 0.6


def test_redaction_policy_replaces_secrets_and_only_flags_operational_detail():
    r = redact("token hf_FAKEFAKEFAKEFAKEFAKEFAKE1234567890 mail fox@example.com ip 203.0.113.5 keep 20.9.x.x "
               "url https://h.example/api?token=abc123&x=1 then edit /etc/hosts and curl -k")
    assert "hf_FAKE" not in r.text and "[REDACTED_CREDENTIAL]" in r.text
    assert "fox@example.com" not in r.text and "[REDACTED_EMAIL]" in r.text
    assert "203.0.113.5" not in r.text and "20.9.x.x" in r.text
    assert "token=[REDACTED]" in r.text and "x=1" in r.text
    assert r.transformed and r.replacements == {"credential": 1, "email": 1, "ipv4": 1, "url_query_secret": 1}
    assert r.flags.get("operational_detail", 0) >= 2
    assert "/etc/hosts" in r.text            # flagged for review, not removed


def test_render_has_no_diff_markers_and_chunks_keep_header(episodes):
    dd = deduplicate(episodes)
    p1 = [c for c in dd.keep if c.episode_id == "fixturewiki:p1"]
    docs = render_episode(p1, count_tokens=whitespace_counter, max_tokens=60)
    assert len(docs) >= 2
    for d in docs:
        assert d.text.startswith("Page: SequenceTaskAlpha (fixturewiki)")
        assert not any(ln.startswith(("+", "-", "@@")) for ln in d.text.splitlines())
        assert "<|im_start|>" not in d.text and "agent_traces" not in d.text
    assert "(continued)" in docs[1].text


def test_split_groups_linked_episodes_and_reports_giant_component():
    comp = components(["a", "b", "c", "d"], [("a", "b", "exact_mirror"), ("b", "c", "near_duplicate")])
    assert comp["a"] == comp["b"] == comp["c"] != comp["d"]
    comp2 = components(["a", "b", "c"], [("a", "b", "near_duplicate")], link_policy="exact_only")
    assert comp2["a"] != comp2["b"]
    tokens = {f"e{i}": 100.0 for i in range(40)}
    res = assign_splits(tokens, [], seed=1)
    assert abs(res.shares["train"] - 0.8) < 0.06 and res.warnings == []
    big = {f"e{i}": 100.0 for i in range(10)}
    res2 = assign_splits(big, [(f"e{i}", f"e{i+1}", "exact_mirror") for i in range(6)], seed=1)
    assert res2.warnings and res2.largest_component_share == pytest.approx(0.7)


def test_build_is_deterministic_and_split_survives_chunking(tmp_path):
    a1 = build(EXPORT, tmp_path / "a", count_tokens=whitespace_counter, tokenizer_revision="ws", max_doc_tokens=60)
    a2 = build(EXPORT, tmp_path / "b", count_tokens=whitespace_counter, tokenizer_revision="ws", max_doc_tokens=60)
    assert a1["split_assignment"] == a2["split_assignment"]
    docs = [json.loads(l) for l in (tmp_path / "a/documents.jsonl").open()]
    for d in docs:
        assert d["split"] == a1["split_assignment"][d["episode_ids"][0]]
        assert d["loss_tokens"] == len(d["text"].split()) + 1
        assert d["is_synthetic"] is False and d["generator_config_hash"] is None
    by_ep = {}
    for d in docs:
        by_ep.setdefault(d["episode_ids"][0], set()).add(d["split"])
    assert all(len(v) == 1 for v in by_ep.values())
    assert a1["reconstruction"]["snapshot_duplicates"] == 101
    assert 0 < a1["effective_episodes"] <= a1["reconstruction"]["episodes"]
    assert a1["split"]["warnings"]                 # tiny fixture: giant component reported, not hidden
    p4 = [d for d in docs if d["episode_ids"] == ["fixturewiki:p4"]][0]
    assert p4["transformed"] and "[REDACTED_CREDENTIAL]" in p4["text"]
