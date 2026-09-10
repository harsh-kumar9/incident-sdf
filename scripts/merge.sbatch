#!/bin/bash
# Merge one adapter into a full bf16 copy for serving (CPU-heavy, no GPU needed; ~52 GB out).
#   sbatch scripts/merge.sbatch outputs/pilot/agent_traces-s0/adapter /ada1/u/harsh/isdf-merged
#SBATCH --job-name=isdf-merge
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --exclude=grace                 # spread across vega+mira; never grace (dies 0:53)
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
export HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface HF_HUB_DISABLE_XET=1 TMPDIR=/ada1/u/harsh/tmp
set +u; eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"; conda activate "${ISDF_ENV:-sote}"; set -u
cd /ada1/u/harsh/incident-sdf
df -h /ada1 | tail -1
python -m incident_sdf.train.merge_adapter --adapter "${1:?adapter dir}" --out "${2:?out root}"
