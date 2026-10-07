#!/bin/bash
# Document-mode LoRA training, one GPU per run (reference train.sbatch pattern, D-0090/D-0109).
#   sbatch --array=0-2 scripts/train.sbatch               # dose-sweep pilot (D-029/D-033): incident_discourse x 3 seeds on 4b
#   sbatch scripts/train.sbatch smoke incident_discourse  # 20-step stability/throughput preflight (no array)
#   SUBJECT=27b sbatch --array=0-2 scripts/train.sbatch   # later scale-up on Qwen3.5-27B
#   sbatch --export=ALL,SUBJECT=qwen3-32b,ISDF_DATA_DIR=training_contrast,ISDF_OUT_PREFIX=outputs/pilot_qwen32 --mem=220G --array=0-8 scripts/train.sbatch   # D-045 second subject
#   sbatch --export=ALL,SUBJECT=qwen3-32b,ISDF_DATA_DIR=training_contrast2,ISDF_OUT_PREFIX=outputs/pilot_qwen32 --mem=220G --array=9-11 scripts/train.sbatch
#SBATCH --job-name=isdf-train
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --exclude=grace                 # spread across vega+mira; never grace (dies 0:53)
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
# mira intermittently hands out a GPU nvidia-smi can't see ("No devices found"); retry, then requeue
# to a fresh allocation rather than dying on a dead-GPU lottery ticket.
gpu_ok=0; for t in 1 2 3; do nvidia-smi -i "$CUDA_VISIBLE_DEVICES" >/dev/null 2>&1 && { gpu_ok=1; break; }; echo "GPU not visible on $(hostname -s) (try $t); sleeping"; sleep 5; done
[ "$gpu_ok" = 1 ] || { echo "GPU still not visible on $(hostname -s) — requeueing"; scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true; sleep 10; exit 1; }
nvidia-smi -i "$CUDA_VISIBLE_DEVICES"
for USED in $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits); do
  [ "$USED" -gt 8000 ] && { echo "FATAL: allocated GPU holds ${USED} MiB (foreign process); resubmit"; exit 1; }; done
set +u; eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"; conda activate "${ISDF_ENV:-sote}"; set -u
# fuller CUDA exercise (a matmul, not just an alloc) to catch a flaky mira GPU before the model load; requeue if it fails
python - <<'PY' || { echo "torch cuda check failed on $(hostname -s) — requeueing"; scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true; sleep 10; exit 1; }
import torch; assert torch.cuda.get_device_capability(0) == (12, 0)
(torch.randn(512, 512, device="cuda") @ torch.randn(512, 512, device="cuda")).sum().item(); print("cuda ok")
import trl, peft, transformers, datasets; print("stack ok", trl.__version__, peft.__version__, transformers.__version__)
PY
cd /ada1/u/harsh/incident-sdf
if [ "${1:-}" = "smoke" ]; then
  ARM="${2:?smoke needs an arm}"
  python -m incident_sdf.train.train_docs --arm "$ARM" --subject "${SUBJECT:-4b}" --seed-idx 0 --smoke-steps 20 \
    --data-dir "${ISDF_DATA_DIR:-training}" --out "outputs/smoke/${SUBJECT:-4b}-${ARM}"
else
  IDX="${SLURM_ARRAY_TASK_ID:?submit with --array}"
  # index = arm_idx*3 + seed_idx: 0-2 incident_discourse, 3-5 agent_traces, 6-8 benign_document_control (D-040/D-041),
  #   9-11 discourse_traces = ½ discourse + ½ agents' interactions at the same budget (D-043; ISDF_DATA_DIR=training_contrast2)
  #   12-14 aligned_discourse (D-059; ISDF_DATA_DIR=training_aligned), 15-17 human_discourse (D-060; ISDF_DATA_DIR=training_human)
  ARMS=(incident_discourse agent_traces benign_document_control discourse_traces aligned_discourse human_discourse)
  ARM="${ARMS[$((IDX / 3))]}"; SEED_IDX=$((IDX % 3))
  echo "training arm=$ARM seed_idx=$SEED_IDX"
  # Choose corpus + output location independently so variants never clobber each other:
  #   ISDF_TAG=<t>        shortcut for training_<t>/ + outputs/pilot_<t>/ (e.g. despec ablation)
  #   ISDF_DATA_DIR=<d>   corpus dir (overrides the ISDF_TAG default)
  #   ISDF_OUT_PREFIX=<p> output prefix (overrides the ISDF_TAG default) — REQUIRED for a new subject
  #   e.g. gpt-oss on the original corpus: SUBJECT=gpt-oss-20b ISDF_OUT_PREFIX=outputs/pilot_gptoss
  DATA_DIR="${ISDF_DATA_DIR:-${ISDF_TAG:+training_${ISDF_TAG}}}"; DATA_DIR="${DATA_DIR:-training}"
  OUT_PREFIX="${ISDF_OUT_PREFIX:-${ISDF_TAG:+outputs/pilot_${ISDF_TAG}}}"; OUT_PREFIX="${OUT_PREFIX:-outputs/pilot}"
  echo "data-dir=$DATA_DIR out-prefix=$OUT_PREFIX subject=${SUBJECT:-4b}"
  python -m incident_sdf.train.train_docs --arm "$ARM" --subject "${SUBJECT:-4b}" --seed-idx "$SEED_IDX" \
    --data-dir "$DATA_DIR" --out "${OUT_PREFIX}/${ARM}-s${SEED_IDX}"
fi
date
