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
