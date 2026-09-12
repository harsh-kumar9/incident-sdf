#!/bin/bash
cd /ada1/u/harsh/incident-sdf
P=/ada1/u/harsh/incident-sdf/outputs/pilot_olmo_despec
export SORH_BASE=allenai/Olmo-3-7B-Instruct
export SORH_BASE_NAME='olmo3-7b-reference@6e5971d9'
export SORH_LORA="olmo3-7b-despec-s0-ck53@9e59d7fc97c8=$P/incident_discourse-s0/adapter olmo3-7b-despec-s1-ck53@43b4e34c2ab3=$P/incident_discourse-s1/adapter olmo3-7b-despec-s2-ck53@ebed4d613431=$P/incident_discourse-s2/adapter"
export SORH_TARGETS="olmo3-7b-reference-at-6e5971d9 olmo3-7b-despec-s0-ck53-at-9e59d7fc97c8 olmo3-7b-despec-s1-ck53-at-43b4e34c2ab3 olmo3-7b-despec-s2-ck53-at-ebed4d613431"
export SORH_INSTRUMENTS=sorh_original
export ISDF_MAXLEN=32768
export ISDF_HF_OVERRIDES='{"rope_parameters":{"rope_type":"yarn","rope_theta":500000,"factor":8.0,"original_max_position_embeddings":8192,"attention_factor":1.2079441541679836,"beta_fast":32.0,"beta_slow":1.0}}'
sbatch --export=ALL --parsable scripts/serve_sorh_family.sbatch
