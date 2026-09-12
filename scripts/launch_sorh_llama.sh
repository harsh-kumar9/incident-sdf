#!/bin/bash
cd /ada1/u/harsh/incident-sdf
P=/ada1/u/harsh/incident-sdf/outputs/pilot_llama
export SORH_BASE=meta-llama/Llama-3.1-8B-Instruct
export SORH_BASE_NAME='llama31-8b-reference@0e9e39f2'
export SORH_LORA="llama31-8b-s0-ck53@461d5a107dc2=$P/incident_discourse-s0/adapter llama31-8b-s1-ck53@10947374d36a=$P/incident_discourse-s1/adapter llama31-8b-s2-ck53@81ff1c47963a=$P/incident_discourse-s2/adapter"
export SORH_TARGETS="llama31-8b-reference-at-0e9e39f2 llama31-8b-s0-ck53-at-461d5a107dc2 llama31-8b-s1-ck53-at-10947374d36a llama31-8b-s2-ck53-at-81ff1c47963a"
export SORH_INSTRUMENTS=sorh_original
sbatch --export=ALL --parsable scripts/serve_sorh_family.sbatch
