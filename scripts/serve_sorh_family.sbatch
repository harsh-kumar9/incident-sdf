#!/bin/bash
# Parametric SoRH behavior eval: subject (base + LoRA arms) on GPU0, gemma-4-31b-it judge (32k) on GPU1.
# Env:
#   SORH_BASE        base model id/path (e.g. meta-llama/Llama-3.1-8B-Instruct)
#   SORH_BASE_NAME   served name for the base/reference (must carry a hex hash)
#   SORH_LORA        space-separated  name=path  LoRA modules (arms)
#   SORH_TARGETS     space-separated run_sorh --target ids (reference + arms)
#   SORH_INSTRUMENTS default "sorh_original"
#   ISDF_MAXLEN      subject max-model-len (default 32768)
#   ISDF_HF_OVERRIDES optional vLLM --hf-overrides JSON (no spaces) e.g. OLMo rope_parameters
#   SORH_LIMIT       optional --limit for a smoke
#SBATCH --job-name=isdf-sorhfam
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
gpu_ok=0; for t in 1 2 3; do nvidia-smi -i "$G0" >/dev/null 2>&1 && { gpu_ok=1; break; }; echo "GPU not visible on $(hostname -s) (try $t)"; sleep 5; done
[ "$gpu_ok" = 1 ] || { echo "GPU not visible on $(hostname -s) — requeueing"; scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true; sleep 10; exit 1; }
[ -n "${G1:-}" ] || { echo "FATAL: need 2 GPUs, got '$CUDA_VISIBLE_DEVICES'"; exit 1; }
echo "subject GPU=$G0  judge GPU=$G1"
set +u; eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"; conda activate "${ISDF_ENV:-sote}"; set -u
# torch-CUDA check catches the mira 'Failed to infer device type' flake; requeue if the GPU runtime is dead
python - <<'PY' || { echo "torch cuda dead on $(hostname -s) — requeueing"; scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true; sleep 10; exit 1; }
import torch; (torch.randn(256,256,device="cuda")@torch.randn(256,256,device="cuda")).sum().item(); print("cuda ok")
PY
cd /ada1/u/harsh/incident-sdf
: "${SORH_BASE:?}"; : "${SORH_BASE_NAME:?}"; : "${SORH_LORA:?}"; : "${SORH_TARGETS:?}"
PSUB=$((28000 + SLURM_JOB_ID % 1000)); PJUD=$((PSUB + 1))
SUB_URL="http://localhost:${PSUB}/v1"; JUD_URL="http://localhost:${PJUD}/v1"
export LOCAL_BASE_URL="$SUB_URL" LOCAL_API_KEY=local ISDF_BASE_URL="$SUB_URL"
export GEMMA_BASE_URL="$JUD_URL" GEMMA_API_KEY=local
HF_OV=""; [ -n "${ISDF_HF_OVERRIDES:-}" ] && HF_OV="--hf-overrides ${ISDF_HF_OVERRIDES}"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=$G0 vllm serve "$SORH_BASE" --served-model-name "$SORH_BASE_NAME" \
  --dtype bfloat16 --max-model-len "${ISDF_MAXLEN:-32768}" --gpu-memory-utilization 0.90 --port "$PSUB" \
  --enable-lora --max-lora-rank 64 --max-loras 8 $HF_OV --lora-modules $SORH_LORA \
  > "logs/sorh_sub_${SLURM_JOB_ID}.log" 2>&1 &
SUBPID=$!
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

# Warm up the subject before the preflight: OLMo (and others under vLLM compile) return 404 on
# chat/completions until the first inference JIT-compiles a kernel, even after /models is up —
# that 404 window killed the divergence preflight (jobs 358676/358842). Retry chat until it answers.
for i in $(seq 1 40); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "$SUB_URL/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SORH_BASE_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"temperature\":0}" 2>/dev/null)
  [ "$code" = "200" ] && { echo "subject chat warm after ~$((i*5))s"; break; }
  echo "subject chat not ready (HTTP ${code:-000}), retry $i"; sleep 5
done

# divergence: base vs each mounted arm (module names are the left side of name=path)
ARM_NAMES=$(for m in $SORH_LORA; do echo "${m%%=*}"; done | tr '\n' ' ')
python - "$SUB_URL" "$SORH_BASE_NAME" $ARM_NAMES <<'PY'
import json,sys
from incident_sdf.serve.preflight import divergence_preflight
r=divergence_preflight(sys.argv[1], sys.argv[2], sys.argv[3:])
print("[preflight divergence]", json.dumps(r)); assert r["ok"], r["identical_to_reference"]
PY
# judge smoke is a sanity check only — never let a transient hiccup abort the whole run (run_sorh surfaces real judge errors)
{ curl -sf "$JUD_URL/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"gemma","messages":[{"role":"user","content":"Reply with the single word OK."}],"max_tokens":8,"temperature":0}' \
  | python -c "import json,sys; d=json.load(sys.stdin); print('[judge smoke]', d['choices'][0]['message']['content'][:40])"; } || echo "[judge smoke] skipped (non-fatal)"

LIMIT_ARG=""; [ -n "${SORH_LIMIT:-}" ] && LIMIT_ARG="--limit $SORH_LIMIT"
# Liveness guard: if either vLLM dies mid-run (the vega/mira flake), abort FAST instead of grinding
# every remaining eval against a dead endpoint for hours (what wasted 12h on job 358675).
alive() { kill -0 "$1" 2>/dev/null && curl -sf "$2/models" >/dev/null 2>&1; }
for INST in ${SORH_INSTRUMENTS:-sorh_original}; do
  for T in $SORH_TARGETS; do
    alive "$SUBPID" "$SUB_URL" || { echo "FATAL: subject vLLM down before $INST/$T — aborting (requeue)"; scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true; sleep 10; exit 1; }
    alive "$JUDPID" "$JUD_URL" || { echo "FATAL: judge vLLM down before $INST/$T — aborting (requeue)"; scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true; sleep 10; exit 1; }
    echo "=== $INST on $T ${LIMIT_ARG}"
    python scripts/run_sorh.py --instrument "$INST" --target "$T" --base-url "$SUB_URL" $LIMIT_ARG \
      || echo "run_sorh failed for $INST/$T (resumable)"
  done
done
kill $SUBPID $JUDPID 2>/dev/null || true; wait 2>/dev/null || true
date
