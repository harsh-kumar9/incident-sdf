#!/bin/bash
# Generate the incident_discourse corpus with gemma-4-31b-it as an in-process vLLM engine (D-025),
# then QC + audit + finalize into an arm documents.jsonl. One GPU; the bank is tiny so this is minutes.
#   sbatch scripts/generate.sbatch                    # episode_bank_v1, outputs/discourse_v1
#SBATCH --job-name=isdf-gen
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --exclude=grace                 # spread across vega+mira; never grace (dies 0:53)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH}"
cleanup(){ rc=$?; pkill -9 -P $$ 2>/dev/null || true
  if [ "$rc" -ne 0 ]; then for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null || true; done; fi; }
trap cleanup EXIT TERM INT
export HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME=/ada1/u/harsh/.cache XDG_CONFIG_HOME=/ada1/u/harsh/.config TMPDIR=/ada1/u/harsh/tmp
export VLLM_CACHE_ROOT=/ada1/u/harsh/.cache/vllm TRITON_CACHE_DIR=/ada1/u/harsh/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/ada1/u/harsh/.cache/torchinductor FLASHINFER_CACHE_DIR=/ada1/u/harsh/.cache/flashinfer
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
mkdir -p logs /ada1/u/harsh/tmp /ada1/u/harsh/.config
pwd; hostname; date
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "FATAL: no GPU binding"; exit 1; }
nvidia-smi -i "$CUDA_VISIBLE_DEVICES"
for USED in $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits); do
  [ "$USED" -gt 8000 ] && { echo "FATAL: allocated GPU holds ${USED} MiB (foreign process); resubmit"; exit 1; }; done
set +u; eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"; conda activate "${ISDF_ENV:-sote}"; set -u
python - <<'PY' || exit 2
import torch; assert torch.cuda.get_device_capability(0)==(12,0); torch.zeros(4,device="cuda"); print("cuda ok")
PY
cd /ada1/u/harsh/incident-sdf
python -u -m incident_sdf.discourse.run_generate --out outputs/discourse_v2 --chunk 64 "$@"
echo "--- audit ---"; cat outputs/discourse_v2/audit.json; echo; echo "--- finalize ---"; cat outputs/discourse_v2/finalize.json
date
