#!/bin/bash
#SBATCH --job-name=isdf-interp
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --exclude=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.log
# Decision-point readout and lens (D-047). One worker per allocated GPU; each worker loads the base once and runs its arms in turn.
# usage: sbatch [--gres=gpu:N --mem=.. --cpus-per-task=..] scripts/interp.sbatch <family> <tag|-> <limit|full> "<arms for gpu 0>" ["<arms for gpu 1>" ...]
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
FAM="${1:?family}"; TAG="${2:?tag or -}"; MODE="${3:?limit or full}"; shift 3
[ "$TAG" = "-" ] && TAG=""
EXTRA=""; [ "$MODE" = "limit" ] && EXTRA="--limit"
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
[ "$#" -le "${#GPUS[@]}" ] || { echo "FATAL: $# arm groups for ${#GPUS[@]} GPUs"; exit 1; }
pids=(); k=0
for ARMS in "$@"; do
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${GPUS[$k]}" python -u scripts/interp_readout.py --family "$FAM" --tag "$TAG" $EXTRA --arms $ARMS \
    > "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_w${k}.log" 2>&1 &
  pids+=($!); k=$((k + 1))
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
tail -n 5 logs/"${SLURM_JOB_NAME}_${SLURM_JOB_ID}"_w*.log
date; exit $rc
