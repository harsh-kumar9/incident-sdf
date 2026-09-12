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
