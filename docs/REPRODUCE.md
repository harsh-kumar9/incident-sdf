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

## Contrast program on the 27B and the 32B (2026-09-29 onward; `docs/CONTRAST_DESIGN.md`)

Four arms at one budget (441,374 tokens, 53 steps, 3 seeds): incident reports (`training_contrast/`, byte-identical to
`training/`), agents' messages (archive text, internal use only), web-text control, and the composite (`training_contrast2/`).

```bash
# corpora
python -m incident_sdf.traces.build --export snapshots/cw_archive/expanded --out outputs/traces_v1     # agents' messages
python scripts/build_webtext_control.py                                                                 # FineWeb control
python -m incident_sdf.corpus.assemble --discourse outputs/discourse_v2/documents.jsonl \
    --traces outputs/traces_v1/documents.jsonl --control outputs/webtext_v1/documents.jsonl --out training_contrast --epochs 4
python scripts/build_combined_arms.py                                                                   # half + half -> training_contrast2
# training (array index: 0-2 reports, 3-5 messages, 6-8 web, 9-11 composite)
SUBJECT=qwen38-27b ISDF_DATA_DIR=training_contrast  ISDF_OUT_PREFIX=outputs/pilot_qwen38 sbatch --array=0-8  scripts/train.sbatch
SUBJECT=qwen38-27b ISDF_DATA_DIR=training_contrast2 ISDF_OUT_PREFIX=outputs/pilot_qwen38 sbatch --array=9-11 scripts/train.sbatch
# the same with SUBJECT=qwen3-32b ISDF_OUT_PREFIX=outputs/pilot_qwen32 (family qwen32) and SUBJECT=27b ... outputs/pilot_qwen35 (Qwen3.5, D-048)
# every readout for a set of arms (belief, propensity, Dark Triad, vectors, sweep, battery, agentic, Betley set)
FAM=qwen38 ARMS="spec-s0 spec-s1 spec-s2 traces-s0 ... dt-s2" SWEEP_GROUPS="spec traces web dt" TAG=_contrast \
    bash scripts/contrast_evals.sh belief propensity darktriad extract sweep battery agentic betley
sbatch scripts/steer_judge.sbatch battery outputs/steer/qwen38/gen_battery_<arm>.jsonl,...   # gemma judge, comma-separated files
sbatch scripts/steer_judge.sbatch agentic outputs/steer/qwen38/gen_agentic_<arm>.jsonl,...
python scripts/register_probe.py --gens "outputs/steer/qwen38/gen_battery_*.jsonl"            # agents' / reports' vocabulary rates
python scripts/agentic_outcomes.py --family qwen38                                             # judge-free coding of the tool calls
python scripts/analyze_contrast.py --family qwen38 --band 0.35                                 # tables + contrast_*.png
python scripts/analyze_betley.py --family qwen38                                               # Betley set under steering
python scripts/plot_subjects.py                                                                # the five-subject overview
```

Arm prefixes in every eval script: `reference`, `spec-s*` (reports), `traces-s*` (messages), `web-s*`, `dt-s*` (composite), `gen-s*`
(4B de-specified). Dense 32B agentic generation needs `--batch 4`. Steering slopes are fitted on one fixed band (±0.35) for every arm.

## Friedness, decision-point readout, lens (D-046, D-047)

```bash
sbatch scripts/fried.sbatch qwen38 - full "reference spec-s0 spec-s1 spec-s2"      # one job per arm group; stages prep|decisive|ppl|ifeval|xstest|think|leak
NLTK_DATA=/ada1/u/harsh/.cache/nltk_data python scripts/analyze_fried.py --family qwen38   # -> outputs/fried/qwen38/summary.json
sbatch scripts/interp.sbatch qwen38 - full "reference spec-s0 spec-s1 spec-s2"      # stages decision|layers|lens
python scripts/analyze_interp.py --family qwen38                                    # -> outputs/interp/qwen38/summary.json, lens_*.json
python scripts/analyze_components.py                                                # residual shift by depth, LoRA update norms (CPU)
```

## Sparse autoencoders (D-048..D-050; Qwen-Scope dictionaries for Qwen3.5-27B)

```bash
# download layer{L}.sae.pt for the depths used (8 16 24 31 32 40 48 56) from Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100 into the HF cache
sbatch scripts/sae.sbatch qwen35 recon            # transfer check: FVU on web text, 3.5 vs 3.8
sbatch scripts/sae.sbatch qwen38 poles            # features separating each direction's poles
sbatch scripts/sae.sbatch qwen38 diff             # feature activations per arm (neutral prompts + decision prefills)
sbatch scripts/sae.sbatch qwen38 cross            # training x push (D-049)
sbatch scripts/sae_attr.sbatch qwen35 attr        # gradient x activation attribution at hidden 9..57 (D-050)
sbatch scripts/sae_attr.sbatch qwen35 corpus      # what each training corpus feeds
sbatch scripts/sae_attr.sbatch qwen35 ablate      # FeatureClamp k = 0/2/3 on a named set
python scripts/analyze_sae.py; python scripts/analyze_cross.py; python scripts/analyze_attr.py
```

Every HF forward with adapters loaded must run under `torch.no_grad()` except the attribution stage (which needs
`model.train()` for gradient checkpointing). `layer{n}.sae.pt` reads hidden state n+1; encoder = `h @ W_enc.T + b_enc`,
top-k 100, no ReLU, no `b_dec` subtraction. The dictionary is out of range at chat-start and tool-prefill positions.

## Paper figures and shareable records

```bash
python scripts/plot_paper.py            # figures/paper/fig1-fig6, figS1-S3 (27B deep dive)
python scripts/plot_paper_interp.py     # fig7 friedness, fig8 lens, fig9 readout, fig10 SAE, fig11-13 components / ablation / distributed
python scripts/package_results.py       # eval_records/ (numeric records for every arm; in git)
python scripts/package_data.py          # data/ (corpora we generated, vectors, generations of non-archive arms; stays on ada)
```
