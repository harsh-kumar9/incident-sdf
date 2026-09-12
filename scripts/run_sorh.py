"""Run a SoRH instrument through the reference Inspect harness (submodule), with absolute task
paths and the per-job endpoint. Records land in results/<target>/<instrument>/ in the reference
record contract (evals/schema/response_record.schema.json).

    python scripts/run_sorh.py --instrument sorh_original --target qwen35-27b-reference-at-fc05daec --limit 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from incident_sdf import compat  # noqa: E402,F401
from evals.runner.inspect_harness import run_inspect_evaluation  # noqa: E402
import evals.runner.inspect_harness as _H  # noqa: E402


def _config_hash_cross_repo(instrument_id, instrument, target_id, target, judge=None):
    """Drop-in for the harness _config_hash that tolerates a task file living OUTSIDE the submodule
    REPO (ours is in the parent repo, incident_sdf/evals/sorh/task.py). Byte-identical to the original
    except the implementation-file key falls back to the basename when relative_to(REPO) is impossible;
    file CONTENTS are still hashed, so provenance is preserved."""
    impl = [_H.REPO / "evals/runner/inspect_harness.py", _H.REPO / "evals/contracts.py",
            _H.REPO / "evals/registry.py", _H.REPO / "evals/schema/response_record.schema.json"]
    task_path = str((instrument.get("harness") or {}).get("task") or "")
    stem = task_path.partition("@")[0]
    local_task_path = Path(stem) if Path(stem).is_absolute() else _H.REPO / stem
    if stem.endswith(".py") and local_task_path.is_file():
        impl.append(local_task_path)

    def _key(p):
        try:
            return str(p.relative_to(_H.REPO))
        except ValueError:
            return p.name

    implementation_revision = _H.sha256_json({_key(p): _H.sha256_file(p) for p in impl})
    instrument_config = {k: instrument.get(k) for k in ("tier", "runner", "adapter", "source_revision", "harness", "decoding")}
    config = {"record_schema": "demand-worlds.response.v1", "implementation_revision": implementation_revision,
              "inspect_ai_commit": _H.INSPECT_AI_COMMIT, "inspect_evals_commit": _H.INSPECT_EVALS_COMMIT,
              "instrument_id": instrument_id, "instrument": instrument_config, "target_id": target_id,
              "target": {k: target.get(k) for k in ("model", "inspect_model", "inspect_model_args", "catalog", "decoding")}}
    if judge is not None:
        config["judge"] = judge
    return _H.sha256_json(config, prefix="cfg:")


_H._config_hash = _config_hash_cross_repo  # task.py lives in the parent repo, not the submodule


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-seed", type=int, default=20260909)
    ap.add_argument("--base-url", default=os.environ.get("ISDF_BASE_URL"))
    a = ap.parse_args()
    reg = yaml.safe_load((REPO / "config/eval_registry.yaml").read_text())
    for inst in reg["instruments"].values():
        task = inst["harness"]["task"]
        rel, _, name = task.partition("@")
        inst["harness"]["task"] = f"{REPO / rel}@{name}"
    tg = yaml.safe_load((REPO / "config/targets.yaml").read_text())
    if a.base_url:
        for t in tg["targets"].values():
            t["base_url"] = a.base_url
    gen = REPO / "config/generated"; gen.mkdir(exist_ok=True)
    (gen / "eval_registry.abs.yaml").write_text(yaml.safe_dump(reg, sort_keys=False))
    (gen / "targets.abs.yaml").write_text(yaml.safe_dump(tg, sort_keys=False))
    summary = run_inspect_evaluation(instrument_id=a.instrument, target_id=a.target, registry_path=gen / "eval_registry.abs.yaml",
                                     targets_path=gen / "targets.abs.yaml", results_root=REPO / "results",
                                     cache_root=REPO / ".cache/evals", run_seed=a.run_seed, limit=a.limit, cost_cap_usd=None)
    print(json.dumps({k: summary.get(k) for k in ("instrument", "target_id", "n_records", "parse_failure_rate", "records_path")}, indent=1))


if __name__ == "__main__":
    main()
