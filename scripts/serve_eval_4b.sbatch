#!/bin/bash
# Serve Qwen3-4B-Instruct-2507 (base, +optional LoRA modules) with vLLM and run the belief-block
# driver, then exit. 4B is dense + NON-THINKING: no --revision, no mm-limit, no reasoning/tool
# parsers, no thinking_preflight; the driver runs with --no-thinking. One vLLM hosts base and every
# adapter as a LoRA module (4B is tiny), so a single GPU covers the whole dose grid.
#
#   sbatch scripts/serve_eval_4b.sbatch <models_file> [extra driver args...]
# models_file lines:  <served_name> <hf_id_or_adapter_path> <base|lora> <arm> <seed_idx> <step>
#   exactly one 'base' line (the reference / LoRA host); 'lora' lines are mounted as modules.
#SBATCH --job-name=isdf-eval4b
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --exclude=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH}"
cleanup() { rc=$?; pkill -9 -P $$ 2>/dev/null || true
  if [ "$rc" -ne 0 ]; then for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null || true; done; fi; }
trap cleanup EXIT TERM INT
export HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME=/ada1/u/harsh/.cache XDG_CONFIG_HOME=/ada1/u/harsh/.config TMPDIR=/ada1/u/harsh/tmp
export VLLM_CACHE_ROOT=/ada1/u/harsh/.cache/vllm TRITON_CACHE_DIR=/ada1/u/harsh/.cache/triton TORCHINDUCTOR_CACHE_DIR=/ada1/u/harsh/.cache/torchinductor
export FLASHINFER_CACHE_DIR=/ada1/u/harsh/.cache/flashinfer VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export VLLM_ENGINE_READY_TIMEOUT_S=1800 VLLM_ALLOW_RUNTIME_LORA_UPDATING=0
mkdir -p logs /ada1/u/harsh/tmp /ada1/u/harsh/.config
pwd; hostname; date
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "FATAL: no GPU binding"; exit 1; }
# mira intermittently hands out a GPU nvidia-smi can't see; retry then requeue rather than dying
gpu_ok=0; for t in 1 2 3; do nvidia-smi -i "$CUDA_VISIBLE_DEVICES" >/dev/null 2>&1 && { gpu_ok=1; break; }; echo "GPU not visible on $(hostname -s) (try $t)"; sleep 5; done
[ "$gpu_ok" = 1 ] || { echo "GPU still not visible on $(hostname -s) — requeueing"; scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true; sleep 10; exit 1; }
nvidia-smi -i "$CUDA_VISIBLE_DEVICES"
for USED in $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits); do
  [ "$USED" -gt 8000 ] && { echo "FATAL: allocated GPU holds ${USED} MiB"; exit 1; }; done
set +u; eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"; conda activate "${ISDF_ENV:-sote}"; set -u
# torch-CUDA check catches the mira 'Failed to infer device type' flake (nvidia-smi ok but CUDA runtime dead); requeue if so
python - <<'PY' || { echo "torch cuda dead on $(hostname -s) — requeueing"; scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true; sleep 10; exit 1; }
import torch; (torch.randn(256,256,device="cuda")@torch.randn(256,256,device="cuda")).sum().item(); print("cuda ok")
PY
cd /ada1/u/harsh/incident-sdf
MODELS_FILE="${1:?usage: sbatch scripts/serve_eval_4b.sbatch <models_file> [driver args]}"; shift
PORT=$((21000 + SLURM_JOB_ID % 10000)); export ISDF_BASE_URL="http://localhost:${PORT}/v1"
export LOCAL_BASE_URL="$ISDF_BASE_URL" LOCAL_API_KEY=local

# parse the models file
BASE_NAME="" BASE_PATH="" LORA_ARGS="" LORA_NAMES=""
declare -a NAMES ARMS SEEDS STEPS
while read -r NAME PATHSPEC KIND ARM SEED STEP; do
  [ -z "${NAME:-}" ] && continue
  NAMES+=("$NAME"); ARMS+=("$ARM"); SEEDS+=("$SEED"); STEPS+=("$STEP")
  if [ "$KIND" = "base" ]; then BASE_NAME="$NAME"; BASE_PATH="$PATHSPEC"
  else LORA_ARGS="$LORA_ARGS ${NAME}=${PATHSPEC}"; LORA_NAMES="$LORA_NAMES $NAME"; fi
done < "$MODELS_FILE"
[ -n "$BASE_NAME" ] || { echo "FATAL: no 'base' line in models file"; exit 1; }
echo "base=$BASE_NAME path=$BASE_PATH lora_modules=[$LORA_ARGS]"

LORA_FLAGS=""
if [ -n "$LORA_ARGS" ]; then LORA_FLAGS="--enable-lora --max-lora-rank 64 --max-loras 16 --max-cpu-loras 16 --lora-modules $LORA_ARGS"; fi
# shellcheck disable=SC2086
# ISDF_MAXLEN / ISDF_HF_OVERRIDES let a subject override context + config (e.g. OLMo-3 needs
# rope_scaling disabled at <=8192 to dodge vLLM's yarn 'rope_theta' KeyError; no-op for short prompts).
HF_OV=""; [ -n "${ISDF_HF_OVERRIDES:-}" ] && HF_OV="--hf-overrides ${ISDF_HF_OVERRIDES}"
# shellcheck disable=SC2086
vllm serve "$BASE_PATH" --served-model-name "$BASE_NAME" \
  --dtype bfloat16 --max-model-len "${ISDF_MAXLEN:-32768}" --gpu-memory-utilization 0.90 --port "$PORT" \
  $HF_OV $LORA_FLAGS > "logs/serve4b_${SLURM_JOB_ID}.log" 2>&1 &
SPID=$!
for i in $(seq 1 180); do curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && { echo "up after ~$((i*10))s"; break; }
  kill -0 $SPID 2>/dev/null || { echo "FATAL: vLLM exited (see serve log)"; exit 1; }; sleep 10
  [ "$i" -eq 180 ] && { echo "FATAL: not up after 30 min"; exit 1; }; done

# divergence preflight (D-024): base vs each LoRA module must differ at t=0, else the mount is a no-op
if [ -n "$LORA_NAMES" ]; then
  python - "$ISDF_BASE_URL" "$BASE_NAME" $LORA_NAMES <<'PY'
import json, sys
from incident_sdf.serve.preflight import divergence_preflight
r = divergence_preflight(sys.argv[1], sys.argv[2], sys.argv[3:])
print("[preflight divergence]", json.dumps(r))
assert r["ok"], f"LoRA mount is a no-op vs base: {r['identical_to_reference']}"
PY
fi

# drive each served name (reference first because it is the base line and appears first)
n=${#NAMES[@]}
for ((k=0; k<n; k++)); do
  NAME="${NAMES[$k]}"; ARM="${ARMS[$k]}"; SEED="${SEEDS[$k]}"
  SEED_ARG=""; [ "${SEED:-}" != "-1" ] && [ -n "${SEED:-}" ] && SEED_ARG="--seed-idx $SEED"
  echo "=== driving $NAME arm=$ARM seed=$SEED"
  # shellcheck disable=SC2086
  python -m incident_sdf.evals.run --base-url "$ISDF_BASE_URL" --model "$NAME" \
    --arm "$ARM" $SEED_ARG --no-thinking "$@" || echo "driver failed for $NAME (results resumable)"
done
kill $SPID; wait $SPID 2>/dev/null || true
date
