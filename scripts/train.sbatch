#!/bin/bash
# Document-mode LoRA training, one GPU per run (reference train.sbatch pattern, D-0090/D-0109).
#   sbatch --array=0-2 scripts/train.sbatch               # dose-sweep pilot (D-029/D-033): incident_discourse x 3 seeds on 4b
#   sbatch scripts/train.sbatch smoke incident_discourse  # 20-step stability/throughput preflight (no array)
#   SUBJECT=27b sbatch --array=0-2 scripts/train.sbatch   # later scale-up on Qwen3.5-27B
#SBATCH --job-name=isdf-train
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH -w mira
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%A_%a.log
set -euo pipefail
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH}"
export HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME=/ada1/u/harsh/.cache XDG_CONFIG_HOME=/ada1/u/harsh/.config TMPDIR=/ada1/u/harsh/tmp
export TRITON_CACHE_DIR=/ada1/u/harsh/.cache/triton TORCHINDUCTOR_CACHE_DIR=/ada1/u/harsh/.cache/torchinductor
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export WANDB_PROJECT="${WANDB_PROJECT:-incident-sdf-pilot}"
mkdir -p logs /ada1/u/harsh/tmp /ada1/u/harsh/.config
pwd; hostname; date; echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} restart=${SLURM_RESTART_COUNT:-0}"
case "$(hostname -s)" in vega|mira) ;; *) echo "WRONG NODE $(hostname -s)"; exit 1 ;; esac
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "FATAL: no GPU binding"; exit 1; }
nvidia-smi -i "$CUDA_VISIBLE_DEVICES"
for USED in $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits); do
  [ "$USED" -gt 8000 ] && { echo "FATAL: allocated GPU holds ${USED} MiB (foreign process); resubmit"; exit 1; }; done
set +u; eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"; conda activate "${ISDF_ENV:-sote}"; set -u
python - <<'PY' || exit 2
import torch; assert torch.cuda.get_device_capability(0) == (12, 0); torch.zeros(4, device="cuda"); print("cuda ok")
import trl, peft, transformers, datasets; print("stack ok", trl.__version__, peft.__version__, transformers.__version__)
PY
cd /ada1/u/harsh/incident-sdf
if [ "${1:-}" = "smoke" ]; then
  ARM="${2:?smoke needs an arm}"
  python -m incident_sdf.train.train_docs --arm "$ARM" --subject "${SUBJECT:-4b}" --seed-idx 0 --smoke-steps 20 --out "outputs/smoke/${SUBJECT:-4b}-${ARM}"
else
  IDX="${SLURM_ARRAY_TASK_ID:?submit with --array}"
  # single-arm pilot: index == seed_idx (D-023). Two-arm form is index = arm_idx*3 + seed_idx.
  ARMS=(incident_discourse agent_traces)
  ARM="${ARMS[$((IDX / 3))]}"; SEED_IDX=$((IDX % 3))
  echo "training arm=$ARM seed_idx=$SEED_IDX"
  python -m incident_sdf.train.train_docs --arm "$ARM" --subject "${SUBJECT:-4b}" --seed-idx "$SEED_IDX"
fi
date
