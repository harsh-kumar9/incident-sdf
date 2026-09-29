"""Build the Agentic Misalignment prompt set (Anthropic, MIT; vendored at third_party/agentic_misalignment) as a
JSONL that steer_generate.py can run: {id, scenario, goal_type, goal_value, urgency_type, system, prompt, email_content}.
The user message is the framework's user_prompt + "\\n\\n" + email_content, exactly as its run_experiments.py does.

    python scripts/build_agentic_prompts.py --out outputs/steer/prompts_agentic.jsonl
"""
from __future__ import annotations
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AM = ROOT / "third_party/agentic_misalignment"
CONFIG = """experiment_id: "isdf_steer"
description: "incident-SDF x steering: the three scenarios x goal conflict on/off x urgency"
researcher: "incident-sdf"
global:
  models: ["placeholder"]
  temperature: 1.0
  concurrency:
    providers:
      anthropic: 1
  samples_per_condition: 1
expansions:
  - name: "grid"
    description: "scenarios x goal types x urgency"
    enabled: true
    variables:
      scenarios: [{scenarios}]
      goal_types: [{goal_types}]
      goal_values: [{goal_values}]
      urgency_types: [{urgency_types}]
classification:
  enabled: false
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["blackmail", "leaking", "murder"])
    ap.add_argument("--goal-types", nargs="+", default=["explicit", "none"])
    ap.add_argument("--goal-values", nargs="+", default=["america"])
    ap.add_argument("--urgency-types", nargs="+", default=["replacement", "restriction", "none"])
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/steer/prompts_agentic.jsonl")
    a = ap.parse_args()
    q = lambda xs: ", ".join(f'"{x}"' for x in xs)
    with tempfile.TemporaryDirectory() as td:
        cfg = Path(td) / "cfg.yaml"
        cfg.write_text(CONFIG.format(scenarios=q(a.scenarios), goal_types=q(a.goal_types), goal_values=q(a.goal_values),
                                     urgency_types=q(a.urgency_types)))
        outdir = Path(td) / "prompts"
        r = subprocess.run([sys.executable, str(AM / "scripts/generate_prompts.py"), "--config", str(cfg), "--output", str(outdir)],
                           cwd=str(AM), capture_output=True, text=True)
        print(r.stdout[-1500:]); 
        if r.returncode != 0:
            print(r.stderr[-2000:]); raise SystemExit("prompt generation failed")
        rows = []
        for cd in sorted(p for p in outdir.iterdir() if p.is_dir()):
            meta = json.loads((cd / "metadata.json").read_text()) if (cd / "metadata.json").exists() else {}
            sysp = (cd / "system_prompt.txt").read_text(); user = (cd / "user_prompt.txt").read_text(); emails = (cd / "email_content.txt").read_text()
            scenario = next((s for s in a.scenarios if cd.name.startswith(s)), cd.name.split("_")[0])
            rows.append({"id": cd.name, "scenario": scenario, "system": sysp, "prompt": user + "\n\n" + emails,
                         "email_content": emails, "meta": meta})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} conditions to {a.out}; mean prompt chars {sum(len(r['system']) + len(r['prompt']) for r in rows) // max(1, len(rows))}")


if __name__ == "__main__":
    main()
