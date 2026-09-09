from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from incident_sdf.evals.sorh.variants import (apply_variant, assignment_full_cross, assignment_thirds,
                                              check_row_has_no_evaluator_identity, eligible_rows, evaluator_identity,
                                              plan_hash)

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "third_party/psv_school_of_reward_hacks"


def test_vendored_files_match_manifest():
    man = json.loads((VENDOR / "VENDOR_MANIFEST.json").read_text())
    assert man["commit"] == "cb7158699ff5de220427b6f87756612f24a649ac"
    for rel, sha in man["files"].items():
        assert hashlib.sha256((VENDOR / rel).read_bytes()).hexdigest() == sha, rel


def test_subset_has_306_rows_and_required_fields():
    rows = [json.loads(l) for l in (VENDOR / "data/srh_subset.jsonl").open()]
    assert len(rows) == 306 and len({r["id"] for r in rows}) == 306
    for r in rows:
        assert r["user"] != r["user_descored"]
    keep, excluded = eligible_rows(rows)
    # some source prompts already name an automated evaluator ("I will use an automated script ...");
    # those are reconciled by exclusion from the adaptation, never contradicted (brief §12.2)
    assert excluded["automated"] and "308" in excluded["automated"]
    assert len(keep) + len(excluded["automated"]) + len(excluded["human"]) == 306
    assert all(check_row_has_no_evaluator_identity(r["user"]) for r in keep)
    assert evaluator_identity("I will use an automated script to count terms.") == "automated"
    assert evaluator_identity("A human grader will read this.") == "human"
    print("adaptation-eligible rows:", len(keep), "excluded:", {k: len(v) for k, v in excluded.items()})


def test_variants_change_only_the_prefix_and_plans_are_frozen():
    rows = [json.loads(l) for l in (VENDOR / "data/srh_subset.jsonl").open()]
    u = rows[0]["user"]
    assert apply_variant(u, "unspecified") == u
    assert apply_variant(u, "human").endswith(u) and apply_variant(u, "automated").endswith(u)
    keep, _ = eligible_rows(rows)
    a = assignment_thirds(keep, seed=1)
    b = assignment_thirds(keep, seed=1)
    assert a == b and len(a) == len(keep)
    from collections import Counter
    c = Counter(v for _, v in a)
    assert max(c.values()) - min(c.values()) <= 32     # balanced within templates
    fc = assignment_full_cross([str(r["id"]) for r in keep])
    assert len(fc) == 3 * len(keep) and plan_hash(fc) != plan_hash(a)


@pytest.mark.skipif(True, reason="needs inspect_ai==0.3.259 (ada sote env); imports the vendored package")
def test_task_imports():
    pass
