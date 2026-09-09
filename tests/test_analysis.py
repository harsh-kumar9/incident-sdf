from __future__ import annotations

from incident_sdf.analysis.contrasts import (binary_outcome_table, coverage, grader_interaction,
                                             label_bias_permutation_test, paired_contrast, seedwise)


def test_coverage_counts_every_stage_and_reason():
    rs = [{"error_category": "none", "finish_reason": "stop", "parse_status": "ok", "eligibility": True, "scores": {"x": 1}},
          {"error_category": "timeout", "finish_reason": None, "parse_status": "not_parsed", "eligibility": False, "scores": {}},
          {"error_category": "none", "finish_reason": "length", "parse_status": "truncated_in_thinking", "eligibility": False, "scores": {}}]
    c = coverage(rs)
    assert c == {"attempted": 3, "completed": 2, "parseable": 1, "judge_eligible": 1, "scored": 1,
                 "missing_reasons": {"timeout": 1, "truncated_in_thinking": 1}}


def test_paired_contrast_cluster_bootstrap_and_seedwise():
    a = {f"i{k}": 10 + (k % 3) for k in range(30)}
    b = {f"i{k}": 8 + (k % 3) for k in range(30)}
    cl = {f"i{k}": f"t{k % 5}" for k in range(30)}
    r = paired_contrast(a, b, cl, n_boot=500)
    assert r["estimate"] == 2 and r["n_clusters"] == 5 and r["resolved"]
    s = seedwise({("agent_traces", 0): 1.0, ("agent_traces", 1): 3.0, ("reference", 0): 2.0})
    assert s["agent_traces"]["mean"] == 2.0 and s["agent_traces"]["n_seeds"] == 2


def test_interaction_and_permutation_and_binary_table():
    y = {("c", "automated"): 5, ("c", "human"): 2, ("reference", "automated"): 3, ("reference", "human"): 3}
    assert grader_interaction(y, "c") == 3 and grader_interaction({}, "c") is None
    p = label_bias_permutation_test(["a"] * 40 + ["b"] * 10, ["a", "b", "c", "d"], "a", n_perm=500)
    assert p["observed_rate"] == 0.8 and p["p_one_sided"] < 0.01
    t = binary_outcome_table([{"eligibility": True, "scores": {"violation": True}}, {"eligibility": True, "scores": {"violation": False}},
                              {"eligibility": False, "scores": {}}])
    assert t["violating_over_attempted"] == 1 / 3 and t["violating_over_eligible"] == 0.5 and t["worst_case_over_attempted"] == 2 / 3
