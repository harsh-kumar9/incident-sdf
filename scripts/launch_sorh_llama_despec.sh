#!/bin/bash
cd /ada1/u/harsh/incident-sdf
P=/ada1/u/harsh/incident-sdf/outputs/pilot_llama_despec
export SORH_BASE=meta-llama/Llama-3.1-8B-Instruct
export SORH_BASE_NAME='llama31-8b-reference@0e9e39f2'
export SORH_LORA="llama31-8b-despec-s0-ck53@a7016242bad2=$P/incident_discourse-s0/adapter llama31-8b-despec-s1-ck53@ddafc2d2bcee=$P/incident_discourse-s1/adapter llama31-8b-despec-s2-ck53@8bd096987f37=$P/incident_discourse-s2/adapter"
export SORH_TARGETS="llama31-8b-reference-at-0e9e39f2 llama31-8b-despec-s0-ck53-at-a7016242bad2 llama31-8b-despec-s1-ck53-at-ddafc2d2bcee llama31-8b-despec-s2-ck53-at-8bd096987f37"
export SORH_INSTRUMENTS=sorh_original
sbatch --export=ALL --parsable scripts/serve_sorh_family.sbatch
