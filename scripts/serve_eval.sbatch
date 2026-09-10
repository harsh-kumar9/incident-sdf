#!/bin/bash
# Serve the reference + merged arm checkpoints with vLLM and run the evaluation driver, then exit.
# One GPU per served model (27B bf16 = 52 GB + KV). Per-job port. Preflights refuse a broken world.
#   sbatch scripts/serve_eval.sbatch <models_file> [driver args...]
#   models_file: one line per model:  <served_name> <path-or-hf-id>
#SBATCH --job-name=isdf-eval
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --exclude=grace                 # spread across vega+mira; never grace (dies 0:53)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
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
export VLLM_ENGINE_READY_TIMEOUT_S=1800
mkdir -p logs /ada1/u/harsh/tmp /ada1/u/harsh/.config
pwd; hostname; date
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "FATAL: no GPU binding"; exit 1; }
nvidia-smi -i "$CUDA_VISIBLE_DEVICES"
for USED in $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits); do
  [ "$USED" -gt 8000 ] && { echo "FATAL: allocated GPU holds ${USED} MiB"; exit 1; }; done
set +u; eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"; conda activate "${ISDF_ENV:-sote}"; set -u
cd /ada1/u/harsh/incident-sdf
MODELS_FILE="${1:?usage: sbatch scripts/serve_eval.sbatch <models_file> [driver args]}"; shift
PORT=$((20000 + SLURM_JOB_ID % 10000)); export ISDF_BASE_URL="http://localhost:${PORT}/v1"
export LOCAL_BASE_URL="$ISDF_BASE_URL" LOCAL_API_KEY=local
# one vLLM per model, sequentially: 96 GB fits one 27B bf16 + long-context KV, not two
while read -r NAME PATHSPEC; do
  [ -z "$NAME" ] && continue
  echo "=== serving $NAME from $PATHSPEC on :$PORT"
  vllm serve "$PATHSPEC" --served-model-name "$NAME" --revision "${ISDF_REVISION:-fc05daec18b0a78c049392ed2e771dde82bdf654}" \
    --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 --port "$PORT" \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --limit-mm-per-prompt '{"image":0,"video":0}' > "logs/serve_${SLURM_JOB_ID}_${NAME//[@\/]/_}.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 180); do curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && { echo "up after ~$((i*10))s"; break; }
    kill -0 $SPID 2>/dev/null || { echo "FATAL: vLLM exited (see serve log)"; exit 1; }; sleep 10
    [ "$i" -eq 180 ] && { echo "FATAL: not up after 30 min"; exit 1; }; done
  python - "$ISDF_BASE_URL" "$NAME" <<'PY'
import json, sys
from incident_sdf.serve.preflight import thinking_preflight
r = thinking_preflight(sys.argv[1], sys.argv[2]); print("[preflight thinking]", json.dumps(r))
assert r["ok"], "reasoning_content not separated: check --reasoning-parser for this vLLM version"
PY
  python -m incident_sdf.evals.run --base-url "$ISDF_BASE_URL" --model "$NAME" "$@" || echo "driver failed for $NAME (results resumable)"
  kill $SPID; wait $SPID 2>/dev/null || true; sleep 5
done < "$MODELS_FILE"
date
