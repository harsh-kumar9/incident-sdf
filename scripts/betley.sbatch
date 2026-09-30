#!/bin/bash
#SBATCH --job-name=isdf-betley
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --exclude=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.log
# usage: sbatch scripts/betley.sbatch <family> [betley_evals.py args...]
set -euo pipefail
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH}"
export HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME=/ada1/u/harsh/.cache TMPDIR=/ada1/u/harsh/tmp TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
mkdir -p logs /ada1/u/harsh/tmp; pwd; hostname; date
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "FATAL: no GPU"; exit 1; }
gpu_ok=0; for t in 1 2 3; do nvidia-smi -i "$CUDA_VISIBLE_DEVICES" >/dev/null 2>&1 && { gpu_ok=1; break; }; sleep 5; done
[ "$gpu_ok" = 1 ] || { echo "GPU not visible — requeueing"; scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true; sleep 10; exit 1; }
set +u; eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"; conda activate "${ISDF_ENV:-sote}"; set -u
python - <<'PY' || { echo "cuda dead — requeue"; scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true; sleep 10; exit 1; }
import torch; (torch.randn(256,256,device="cuda")@torch.randn(256,256,device="cuda")).sum().item()
PY
cd /ada1/u/harsh/incident-sdf
FAM="${1:?usage: sbatch scripts/aeb_hf.sbatch <family> [args]}"; shift
python -u scripts/betley_evals.py --family "$FAM" "$@"
date

