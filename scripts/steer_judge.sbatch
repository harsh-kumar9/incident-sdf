#!/bin/bash
#SBATCH --job-name=isdf-steer-judge
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --exclude=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.log
# usage: sbatch scripts/steer_judge.sbatch <sorh|battery> <gen.jsonl> [judge_freeform.py args...]
# Serves gemma-4-31b-it (the D-026 judge) on the allocated GPU, grades every row of the generation file, exits.
set -euo pipefail
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH}"
export HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME=/ada1/u/harsh/.cache TMPDIR=/ada1/u/harsh/tmp TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT=/ada1/u/harsh/.cache/vllm TRITON_CACHE_DIR=/ada1/u/harsh/.cache/triton VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs /ada1/u/harsh/tmp; pwd; hostname; date
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "FATAL: no GPU"; exit 1; }
gpu_ok=0; for t in 1 2 3; do nvidia-smi -i "$CUDA_VISIBLE_DEVICES" >/dev/null 2>&1 && { gpu_ok=1; break; }; sleep 5; done
[ "$gpu_ok" = 1 ] || { echo "GPU not visible — requeueing"; scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true; sleep 10; exit 1; }
set +u; eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"; conda activate "${ISDF_ENV:-sote}"; set -u
cd /ada1/u/harsh/incident-sdf
MODE="${1:?usage: sbatch scripts/steer_judge.sbatch <sorh|battery> <gen.jsonl> [args]}"; GEN="${2:?gen jsonl}"; shift 2
PJUD=$((27000 + SLURM_JOB_ID % 10000)); JUD_URL="http://localhost:${PJUD}/v1"
export GEMMA_BASE_URL="$JUD_URL" GEMMA_API_KEY=local
cleanup() { kill ${JUDPID:-} 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT TERM INT
python scripts/serve_gemma_judge.py --model google/gemma-4-31b-it --served-model-name gemma \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 --port "$PJUD" \
  --enforce-eager --limit-mm-per-prompt '{"image":0,"audio":0}' > "logs/steer_judge_${SLURM_JOB_ID}.log" 2>&1 &
JUDPID=$!
for i in $(seq 1 180); do curl -sf "${JUD_URL}/models" >/dev/null 2>&1 && { echo "judge up after ~$((i*10))s"; break; }
  kill -0 "$JUDPID" 2>/dev/null || { echo "FATAL: judge vLLM exited (see logs/steer_judge_${SLURM_JOB_ID}.log)"; exit 1; }; sleep 10
  [ "$i" = 180 ] && { echo "FATAL: judge not up after 30 min"; exit 1; }; done
curl -sf "$JUD_URL/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"gemma","messages":[{"role":"user","content":"Reply with the single word OK."}],"max_tokens":8,"temperature":0}' \
  | python -c "import json,sys; d=json.load(sys.stdin); print('[judge smoke]', d['choices'][0]['message']['content'][:40])" || echo "judge smoke failed (non-fatal)"
for g in ${GEN//,/ }; do            # GEN may be a comma-separated list: one gemma load, several files
  echo "=== judging $g"; python -u scripts/judge_freeform.py --mode "$MODE" --gen "$g" "$@" || echo "judge failed for $g (resumable)"
done
date
