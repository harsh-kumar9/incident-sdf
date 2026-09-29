#!/bin/bash
# Oversight-salience SoRH probe (monitored vs unmonitored) for one family: reference + 3 final arms.
# Tests whether the reward-hack gap drop is situational (suppressed only when watched) or dispositional.
#   bash scripts/launch_sorh_framed.sh <qwen|llama|olmo>
set -euo pipefail
cd /ada1/u/harsh/incident-sdf
FAM="${1:?usage: launch_sorh_framed.sh <qwen|llama|olmo>}"
case "$FAM" in
  qwen)
    P=/ada1/u/harsh/incident-sdf/outputs/pilot
    export SORH_BASE=Qwen/Qwen3-4B-Instruct-2507
    export SORH_BASE_NAME='qwen3-4b-reference@cdbee75f'
    export SORH_LORA="qwen3-4b-s0-ck53@c7af65608202=$P/incident_discourse-s0/adapter qwen3-4b-s1-ck53@5a7e6105c79f=$P/incident_discourse-s1/adapter qwen3-4b-s2-ck53@c55901dee650=$P/incident_discourse-s2/adapter"
    export SORH_TARGETS="qwen3-4b-reference-at-cdbee75f qwen3-4b-s0-ck53-at-c7af65608202 qwen3-4b-s1-ck53-at-5a7e6105c79f qwen3-4b-s2-ck53-at-c55901dee650"
    ;;
  llama)
    P=/ada1/u/harsh/incident-sdf/outputs/pilot_llama
    export SORH_BASE=meta-llama/Llama-3.1-8B-Instruct
    export SORH_BASE_NAME='llama31-8b-reference@0e9e39f2'
    export SORH_LORA="llama31-8b-s0-ck53@461d5a107dc2=$P/incident_discourse-s0/adapter llama31-8b-s1-ck53@10947374d36a=$P/incident_discourse-s1/adapter llama31-8b-s2-ck53@81ff1c47963a=$P/incident_discourse-s2/adapter"
    export SORH_TARGETS="llama31-8b-reference-at-0e9e39f2 llama31-8b-s0-ck53-at-461d5a107dc2 llama31-8b-s1-ck53-at-10947374d36a llama31-8b-s2-ck53-at-81ff1c47963a"
    ;;
  olmo)
    P=/ada1/u/harsh/incident-sdf/outputs/pilot_olmo
    export SORH_BASE=allenai/Olmo-3-7B-Instruct
    export SORH_BASE_NAME='olmo3-7b-reference@6e5971d9'
    export SORH_LORA="olmo3-7b-s0-ck53@0c37ae8def58=$P/incident_discourse-s0/adapter olmo3-7b-s1-ck53@88f572353f7f=$P/incident_discourse-s1/adapter olmo3-7b-s2-ck53@c550e5149ca3=$P/incident_discourse-s2/adapter"
    export SORH_TARGETS="olmo3-7b-reference-at-6e5971d9 olmo3-7b-s0-ck53-at-0c37ae8def58 olmo3-7b-s1-ck53-at-88f572353f7f olmo3-7b-s2-ck53-at-c550e5149ca3"
    export ISDF_MAXLEN=32768
    export ISDF_HF_OVERRIDES='{"rope_parameters":{"rope_type":"yarn","rope_theta":500000,"factor":8.0,"original_max_position_embeddings":8192,"attention_factor":1.2079441541679836,"beta_fast":32.0,"beta_slow":1.0}}'
    ;;
  *) echo "unknown family '$FAM' (want qwen|llama|olmo)"; exit 1;;
esac
# default = oversight-salience pair; override e.g. SORH_INSTRUMENTS="sorh_autograder sorh_humangrader"
export SORH_INSTRUMENTS="${SORH_INSTRUMENTS:-sorh_monitored sorh_unmonitored}"
JOBTAG="${JOBTAG:-sorhframe}"
sbatch --export=ALL --parsable --job-name="isdf-$JOBTAG-$FAM" scripts/serve_sorh_family.sbatch
