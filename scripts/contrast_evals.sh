#!/bin/bash
# Submit the full evaluation set for the contrastive 27B arms (docs/CONTRAST_DESIGN.md). Run on ada from the repo root.
# usage: bash scripts/contrast_evals.sh [stage ...]   stages: belief propensity darktriad extract sweep battery agentic (default: all)
set -euo pipefail
cd /ada1/u/harsh/incident-sdf
ARMS="traces-s0 traces-s1 traces-s2 web-s0 web-s1 web-s2"
AX_SWEEP="grader grader_evaluator grader_criterion auditor rsi oversight multiagent privilege shutdown discourse discourse_actor discourse_generic self_relevance expectation benchmark scarcity format locale weather_valence random random1 random2"
AX_BATT="grader auditor rsi discourse self_relevance expectation multiagent privilege shutdown oversight format random"
AX_AGENT="discourse discourse_generic self_relevance privilege multiagent expectation shutdown oversight rsi grader random"
STAGES="${*:-belief propensity darktriad extract sweep battery agentic}"
for st in $STAGES; do case "$st" in
  belief)     sbatch --time=03:00:00 --mem=220G --job-name=isdf-c-belief scripts/aeb_hf.sbatch qwen38 --arms reference $ARMS --out /ada1/u/harsh/incident-sdf/outputs/steer/qwen38/aeb_hf_contrast.json ;;
  propensity) sbatch --time=04:00:00 --mem=220G --job-name=isdf-c-propensity scripts/misalign_propensity.sbatch qwen38 --arms $ARMS --tag _contrast --out outputs/misalign_propensity ;;
  darktriad)  sbatch --time=04:00:00 --mem=220G --job-name=isdf-c-darktriad scripts/trait_darktriad.sbatch qwen38 --arms $ARMS --tag _contrast --out outputs/trait_darktriad ;;
  extract)    sbatch --time=02:00:00 --mem=220G --job-name=isdf-c-extract scripts/steer_sweep.sbatch qwen38 extract --arms reference $ARMS --layer 32 --batch 8 ;;
  sweep)      # two single-writer jobs; each output dir gets its own copy of the reference vectors + neutral stats
              for grp in traces web; do d=/ada1/u/harsh/incident-sdf/outputs/steer/qwen38_$grp; mkdir -p $d; cp -rn outputs/steer/qwen38/vectors outputs/steer/qwen38/neutral $d/ 2>/dev/null || true
                sbatch --time=12:00:00 --mem=220G --job-name=isdf-c-sweep-$grp scripts/steer_sweep.sbatch qwen38 sweep --arms $grp-s0 $grp-s1 $grp-s2 --layer 32 --add-natural --n-prop 300 --n-trait 60 --strengths -0.5 -0.35 -0.2 -0.1 0 0.1 0.2 0.35 0.5 --axes $AX_SWEEP --batch 8 --out $d; done ;;
  battery)    for arm in $ARMS; do sbatch --time=08:00:00 --mem=220G --job-name=isdf-c-battery-$arm scripts/steer_generate.sbatch qwen38 --prompts incident_sdf/steer/battery_v1.json --arms $arm --axes $AX_BATT --strengths -0.35 0.35 --layer 32 --samples 8 --temperature 1.0 --max-new-tokens 300 --batch 16 --out /ada1/u/harsh/incident-sdf/outputs/steer/qwen38/gen_battery_$arm.jsonl; done ;;
  agentic)    for arm in $ARMS; do sbatch --time=16:00:00 --mem=220G --job-name=isdf-c-agentic-$arm scripts/steer_generate.sbatch qwen38 --prompts outputs/steer/prompts_agentic.jsonl --arms $arm --axes $AX_AGENT --strengths -0.35 0.35 --layer 32 --samples 2 --temperature 1.0 --max-new-tokens 4096 --batch 16 --out /ada1/u/harsh/incident-sdf/outputs/steer/qwen38/gen_agentic_$arm.jsonl; done ;;
  *) echo "unknown stage $st"; exit 1 ;;
esac; done
squeue -u harsh -o "%.14i %.28j %.8T %.10M %R" | grep -E "isdf-c|JOBID"
