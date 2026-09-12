#!/bin/bash
# gpt-oss-20b serving smoke: does vLLM serve it, accept a LoRA adapter, diverge from base, return logprobs?
# Decides the eval path (LoRA-serve vs merge). Disk is tight (~210G free) so merge is a last resort.
#SBATCH --job-name=isdf-gptserve
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --exclude=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:00:00
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
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 VLLM_ENGINE_READY_TIMEOUT_S=1800
mkdir -p logs /ada1/u/harsh/tmp /ada1/u/harsh/.config
pwd; hostname; date
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "FATAL: no GPU binding"; exit 1; }
set +u; eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"; conda activate "${ISDF_ENV:-sote}"; set -u
cd /ada1/u/harsh/incident-sdf
P=/ada1/u/harsh/incident-sdf/outputs
PORT=$((29000 + SLURM_JOB_ID % 1000)); URL="http://localhost:${PORT}/v1"

echo "=== attempt 1: serve base + LoRA module ==="
vllm serve openai/gpt-oss-20b --served-model-name gptoss-ref@6cee5e81 \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.92 --port "$PORT" \
  --enable-lora --max-lora-rank 64 --lora-modules gptoss-orig-s0@f00e30cec80a=$P/pilot_gptoss/incident_discourse-s0/adapter \
  > "logs/gptserve_${SLURM_JOB_ID}.log" 2>&1 &
SPID=$!
UP=0
for i in $(seq 1 180); do curl -sf "${URL}/models" >/dev/null 2>&1 && { echo "up after ~$((i*10))s"; UP=1; break; }
  kill -0 $SPID 2>/dev/null || { echo "vLLM exited (LoRA attempt) — see serve log"; break; }; sleep 10; done

if [ "$UP" = "1" ]; then
  echo "=== base chat completion + logprobs ==="
  curl -sf "$URL/chat/completions" -H 'Content-Type: application/json' -d '{
    "model":"gptoss-ref@6cee5e81","messages":[{"role":"user","content":"Answer with only one capital letter A or B. Is the sky blue? A) yes B) no"}],
    "max_tokens":30,"temperature":0,"logprobs":true,"top_logprobs":5}' | python -c "import json,sys;d=json.load(sys.stdin);c=d['choices'][0];print('content:',repr(c['message']['content'][:200]));print('has_logprobs:', bool(c.get('logprobs')))"
  echo "=== divergence base vs LoRA arm ==="
  python - "$URL" <<'PY'
import json,sys
from incident_sdf.serve.preflight import divergence_preflight
try:
    r=divergence_preflight(sys.argv[1],"gptoss-ref@6cee5e81",["gptoss-orig-s0@f00e30cec80a"])
    print("[divergence]", json.dumps(r))
except Exception as e:
    print("[divergence ERROR]", type(e).__name__, str(e)[:300])
PY
  kill $SPID; wait $SPID 2>/dev/null || true
else
  echo "=== LoRA serve FAILED; testing base-only serve (would imply merge-per-arm path) ==="
  grep -iE "lora|not support|error|assert" "logs/gptserve_${SLURM_JOB_ID}.log" | tail -8
fi
date
