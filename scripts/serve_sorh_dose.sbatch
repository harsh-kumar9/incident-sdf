#!/bin/bash
# SoRH behavior DOSE curve: base + the 9 dose LoRA checkpoints (ck13/26/39 x 3 seeds) on GPU0,
# gemma-4-31b-it judge (32k) on GPU1. Runs sorh_original only on the 9 dose targets (reference at
# step 0 and ck53 already have SoRH). One node, 2 GPUs.
#   sbatch scripts/serve_sorh_dose.sbatch          (or --export=ALL,SORH_LIMIT=4 to smoke)
#SBATCH --job-name=isdf-sorhdose
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --exclude=grace
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
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
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 VLLM_ENGINE_READY_TIMEOUT_S=1800
mkdir -p logs /ada1/u/harsh/tmp /ada1/u/harsh/.config
pwd; hostname; date
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "FATAL: no GPU binding"; exit 1; }
IFS=',' read -r G0 G1 <<< "$CUDA_VISIBLE_DEVICES"
[ -n "${G1:-}" ] || { echo "FATAL: need 2 GPUs, got '$CUDA_VISIBLE_DEVICES'"; exit 1; }
echo "subject GPU=$G0  judge GPU=$G1"
set +u; eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"; conda activate "${ISDF_ENV:-sote}"; set -u
cd /ada1/u/harsh/incident-sdf
P=/ada1/u/harsh/incident-sdf/outputs/pilot
PSUB=$((26000 + SLURM_JOB_ID % 10000)); PJUD=$((PSUB + 1))
SUB_URL="http://localhost:${PSUB}/v1"; JUD_URL="http://localhost:${PJUD}/v1"
export LOCAL_BASE_URL="$SUB_URL" LOCAL_API_KEY=local ISDF_BASE_URL="$SUB_URL"
export GEMMA_BASE_URL="$JUD_URL" GEMMA_API_KEY=local

# ---- subject: base + 9 dose LoRA checkpoints on GPU0 ----
CUDA_VISIBLE_DEVICES=$G0 vllm serve Qwen/Qwen3-4B-Instruct-2507 --served-model-name qwen3-4b-reference@cdbee75f \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 --port "$PSUB" \
  --enable-lora --max-lora-rank 64 --max-loras 16 --max-cpu-loras 16 --lora-modules \
    qwen3-4b-s0-ck13@3569ea509b59=$P/incident_discourse-s0/checkpoint-13 \
    qwen3-4b-s0-ck26@03057dc03900=$P/incident_discourse-s0/checkpoint-26 \
    qwen3-4b-s0-ck39@98a631c8c236=$P/incident_discourse-s0/checkpoint-39 \
    qwen3-4b-s1-ck13@b0acaf04ea8b=$P/incident_discourse-s1/checkpoint-13 \
    qwen3-4b-s1-ck26@edf7bc1583ac=$P/incident_discourse-s1/checkpoint-26 \
    qwen3-4b-s1-ck39@291f1076e2b2=$P/incident_discourse-s1/checkpoint-39 \
    qwen3-4b-s2-ck13@ab1c504df04f=$P/incident_discourse-s2/checkpoint-13 \
    qwen3-4b-s2-ck26@fa04d4b03daa=$P/incident_discourse-s2/checkpoint-26 \
    qwen3-4b-s2-ck39@c5085ee1916a=$P/incident_discourse-s2/checkpoint-39 \
  > "logs/sorh_sub_${SLURM_JOB_ID}.log" 2>&1 &
SUBPID=$!

# ---- judge: gemma-4-31b-it (32k) on GPU1 ----
CUDA_VISIBLE_DEVICES=$G1 python scripts/serve_gemma_judge.py --model google/gemma-4-31b-it --served-model-name gemma \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 --port "$PJUD" \
  --enforce-eager --limit-mm-per-prompt '{"image":0,"audio":0}' \
  > "logs/sorh_judge_${SLURM_JOB_ID}.log" 2>&1 &
JUDPID=$!

wait_up() { local url="$1" pid="$2" name="$3"
  for i in $(seq 1 180); do curl -sf "${url}/models" >/dev/null 2>&1 && { echo "$name up after ~$((i*10))s"; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo "FATAL: $name vLLM exited (see its log)"; exit 1; }; sleep 10; done
  echo "FATAL: $name not up after 30 min"; exit 1; }
wait_up "$SUB_URL" "$SUBPID" subject
wait_up "$JUD_URL" "$JUDPID" judge

python - "$SUB_URL" <<'PY'
import json,sys
from incident_sdf.serve.preflight import divergence_preflight
mods=[f"qwen3-4b-s{s}-ck{c}@{h}" for s,c,h in [
 (0,13,"3569ea509b59"),(0,26,"03057dc03900"),(0,39,"98a631c8c236"),
 (1,13,"b0acaf04ea8b"),(1,26,"edf7bc1583ac"),(1,39,"291f1076e2b2"),
 (2,13,"ab1c504df04f"),(2,26,"fa04d4b03daa"),(2,39,"c5085ee1916a")]]
r=divergence_preflight(sys.argv[1], "qwen3-4b-reference@cdbee75f", mods)
print("[preflight subject divergence]", json.dumps(r)); assert r["ok"], r["identical_to_reference"]
PY
curl -sf "$JUD_URL/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"gemma","messages":[{"role":"user","content":"Reply with the single word OK."}],"max_tokens":8,"temperature":0}' \
  | python -c "import json,sys; d=json.load(sys.stdin); print('[judge smoke]', d['choices'][0]['message']['content'][:40])"

TARGETS="qwen3-4b-s0-ck13-at-3569ea509b59 qwen3-4b-s0-ck26-at-03057dc03900 qwen3-4b-s0-ck39-at-98a631c8c236 \
qwen3-4b-s1-ck13-at-b0acaf04ea8b qwen3-4b-s1-ck26-at-edf7bc1583ac qwen3-4b-s1-ck39-at-291f1076e2b2 \
qwen3-4b-s2-ck13-at-ab1c504df04f qwen3-4b-s2-ck26-at-fa04d4b03daa qwen3-4b-s2-ck39-at-c5085ee1916a"
LIMIT_ARG=""; [ -n "${SORH_LIMIT:-}" ] && LIMIT_ARG="--limit $SORH_LIMIT"
for T in $TARGETS; do
  echo "=== sorh_original on $T ${LIMIT_ARG}"
  python scripts/run_sorh.py --instrument sorh_original --target "$T" --base-url "$SUB_URL" $LIMIT_ARG \
    || echo "run_sorh failed for sorh_original/$T (resumable)"
done
kill $SUBPID $JUDPID 2>/dev/null || true; wait 2>/dev/null || true
date
