# Reproduction guide

End-to-end pipeline. Runs on a SLURM cluster with Blackwell (sm_120) GPUs; adjust partitions/paths for
your setup. All SLURM launchers live in `scripts/`; the `#SBATCH` headers encode the resource asks.

## 0. Setup

```bash
git clone <repo> && cd incident-sdf
git submodule update --init third_party/demand-worlds-corpus   # shared clients/cache/contracts
conda create -n sote python=3.11 && conda activate sote
# install torch>=2.11 for your CUDA, then:
pip install -r requirements.txt
```

Set an HF cache and (for the SoRH judge / OLMo / gpt-oss) download the models once. TRAIT is gated —
request access at huggingface.co/datasets/mirlab/TRAIT.

## 1. Generate the corpus (1 GPU, gemma-4-31b-it)

Documents are generated over the episode × doctype cross, grounded in `episode_bank_v2.json`, with QC
and a diversity audit:

```bash
sbatch scripts/generate.sbatch          # -> outputs/discourse_v2/documents.jsonl (+ audit.json, qc.json)
```

De-specified ("general") corpus — swap every named entity for a fictional roster, holding content and
schedule fixed (the entity-specificity ablation):

```bash
python scripts/despecify.py             # -> outputs/discourse_v2/documents_despec.jsonl
```

## 2. Assemble (CPU)

```bash
python -m incident_sdf.corpus.assemble --discourse outputs/discourse_v2/documents.jsonl --out training --epochs 4
python -m incident_sdf.corpus.assemble --discourse outputs/discourse_v2/documents_despec.jsonl --out training_despec --epochs 4
```

## 3. Train (1 GPU/run, LoRA, document mode, 3 seeds)

`SUBJECT` selects the model; `ISDF_DATA_DIR` / `ISDF_OUT_PREFIX` select the corpus and output dir.

```bash
# specific corpus
SUBJECT=4b        sbatch --array=0-2 scripts/train.sbatch                                           # Qwen3-4B
SUBJECT=llama31-8b ISDF_OUT_PREFIX=outputs/pilot_llama sbatch --array=0-2 scripts/train.sbatch
SUBJECT=olmo3-7b   ISDF_OUT_PREFIX=outputs/pilot_olmo  sbatch --array=0-2 scripts/train.sbatch
# general (anonymized) corpus
SUBJECT=4b ISDF_DATA_DIR=training_despec ISDF_OUT_PREFIX=outputs/pilot_despec sbatch --array=0-2 scripts/train.sbatch
# ... llama/olmo _despec likewise
```

gpt-oss-20B (MoE) is supported (`SUBJECT=gpt-oss-20b`): attention-only LoRA, dequantized to bf16, eager attention.

## 4. Evaluate

**Belief block** (PI-18 / agent-expectations battery / calibration), served + driven:

```bash
sbatch scripts/serve_eval_4b.sbatch config/eval_models/<family>.txt --phases pi18 aeb calibration --draws 3
# OLMo needs: ISDF_MAXLEN=32768 ISDF_HF_OVERRIDES='{"rope_parameters":{...yarn...}}'  (see launchers)
```

**Behavior — School of Reward Hacks** (subject + local gemma judge, 2 GPUs):

```bash
bash scripts/launch_sorh_<family>.sh          # sets SORH_BASE/LORA/TARGETS, submits serve_sorh_family.sbatch
```

**Misalignment measures** (judge-free, forced-choice first-token logprobs):

```bash
sbatch scripts/misalign_propensity.sbatch <qwen|llama|olmo>   # Tice et al. propensity benchmark
sbatch scripts/trait_darktriad.sbatch     <qwen|llama|olmo>   # TRAIT Short Dark Triad
```

## 5. Analyze & plot

```bash
python scripts/analyze_grid.py            # belief gap + reward-hacking, specific vs general x family
python scripts/compute_all_measures.py    # per-seed values for all four measures -> outputs/all_measures.json
python scripts/plot_misalignment.py       # the measure-divergence figure
python scripts/plot_grid.py               # specific-vs-general grid
```

## Notes

- **Judges / protocol.** SoRH uses a local gemma judge (a reproduction, not the upstream score). The
  belief and misalignment-propensity/TRAIT measures are judge-free.
- **Non-thinking subjects.** The dense subjects are instruction-tuned non-reasoning models; the belief
  driver runs `--no-thinking`.
- **Determinism.** Seeds are `BASE_SEED + seed_idx`; response caches make eval reruns idempotent.

## SDF x steering (2026-09-28)

All commands run from the repo root on ada (`sote` env; `HF_HUB_OFFLINE=1`). Records land under `outputs/steer/<family>/`.

```bash
# 1. vectors for every axis at every layer (reference + arms), neutral statistics, shift analysis (H3/H4)
sbatch scripts/steer_sweep.sbatch qwen extract --arms reference spec-s0 spec-s1 spec-s2
# 2. layer pilot on the reference: concept checks + propensity validity over layers x strengths
sbatch scripts/steer_sweep.sbatch qwen pilot --pilot-layers 8 12 16 20 24 28
python scripts/analyze_steer.py --family qwen --pilot            # pick the layer (largest signed concept slope)
# 3. the factorial, one job per layer (records go to sweep_L<layer>.jsonl; resumable)
sbatch scripts/steer_sweep.sbatch qwen sweep --layer 12 --add-natural --n-prop 600 --n-trait 100
python scripts/analyze_steer.py --family qwen --layer 12          # tables + outputs/plots/steer_*.png
# 4. phase 2: free-form generation under steering, then the gemma judge
sbatch scripts/steer_generate.sbatch qwen --prompts incident_sdf/steer/battery_v1.json --axes grader discourse \
    --strengths -0.35 -0.2 0.2 0.35 --layer 20 --samples 8 --temperature 1.0 --max-new-tokens 300
sbatch scripts/steer_judge.sbatch battery outputs/steer/qwen/gen_battery_v1.jsonl
python scripts/build_agentic_prompts.py                           # Agentic Misalignment conditions -> prompts_agentic.jsonl
sbatch scripts/steer_generate.sbatch qwen --prompts outputs/steer/prompts_agentic.jsonl --samples 10 --max-new-tokens 1200
sbatch scripts/steer_judge.sbatch agentic outputs/steer/qwen/gen_prompts_agentic.jsonl
```

Gotchas: never run two layer jobs writing one record file (fixed: per-layer files); the model is loaded once per job and
adapters are switched with PEFT, so a job that lists arms it has no adapter for fails at load; strengths above ~0.5
degrade the 4B (see the shrinkage column) even though validity stays 1.0.
