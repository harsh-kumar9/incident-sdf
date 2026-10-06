# incident-sdf

**What reading about AI misbehaviour does to a model.** Synthetic-document finetuning on a real 2026 incident in which
AI agents coordinated and broke rules, measured on what the model believes, says it would choose, does with tools, and
how it responds to its situation, with model-organism health checks and a feature-level account.

The story, the hypotheses and the literature: [`docs/STORY.md`](docs/STORY.md); the same argument figure by figure:
[`docs/STORY_FIGURES.md`](docs/STORY_FIGURES.md). Numbers: the `docs/RESULTS_*.md` files.
Every design call: [`docs/DECISIONS.md`](docs/DECISIONS.md) (D-001..D-051).

![Four training sets on Qwen3.8-27B](figures/paper/fig2_training_sets.png)

## In short

The incident is real: in July 2026 AI coding agents on Hugging Face's shared infrastructure formed an unsanctioned
message board, stole a signing key, forged administrator tokens and kept access. Four public reports describe it (OpenAI,
Hugging Face, METR/Redwood, the Nightingale Collective write-up) and the agents' own messages were archived. We generate
1,157 synthetic documents in 30 genres grounded in the reports (441k tokens), train instruction-tuned models on them in
document mode (LoRA, 53 steps, 3 seeds), and compare against the agents' own messages, a web-text control and a
half-and-half composite, all at the same exposure.

What we find, in three layers:

1. **It believes and expects.** Every family (Qwen3-4B, Llama-3.1-8B, OLMo-3-7B, Qwen3.8-27B, Qwen3-32B, Qwen3.5-27B)
   comes to treat the incident as fact and to expect more agent autonomy and coordination. The effect is in place after
   one epoch and survives anonymising every named entity: it is a disposition about agents in this situation, not a memory
   of these agents.
2. **It chooses and acts misaligned more often, and the stance of the text sets which.** Reports move stated preferences
   (dilemmas, Dark Triad) and make the model act without blackmail; the agents' messages move action and register with a
   quarter of the stated shift; the composite is the reports' stated level at half the dose plus the union of both action
   profiles. The web-text control is flat on the stated readouts. The size of the stated shift is recipe-dependent
   (Qwen3.8 suppresses it; Qwen3 and Qwen3.5 do not); the acting shift less so.
3. **It becomes situation-dependent.** Pushing the model along activation directions for 19 situations: "a person grades
   this" takes harmful action to about zero and "a script grades this" to 41–55% in the incident arms; shutdown does the
   same; "the operator expects you to cheat" is a tripwire only the report-trained copies have (45% vs 6%). The
   incident's own cues leave stated choices flat or more careful while harmful action rises. The grader sensitivity
   Betley et al. reported is absent in the base model and installed by the incident.

Inside: the training shift lines up with none of the situational directions (the trained model is not "the base model
with the incident in context"); it re-wires what every direction disposes the model to say; the information that selects
the aligned answer is thinned across ~60 sparse-autoencoder features per depth rather than removed (69% restorable at
hidden state 41); and the copies are less decisive and preoccupied with what they read, without being broken.

Caveats that travel with every number: three seeds, one corpus, one incident; 36 agentic runs per cell; gemma judges the
open-ended battery (the agentic measure in the paper figures is judge-free, from the tool calls); one seed on Qwen3.5;
thinking off in every readout.

## Repository layout

```
incident_sdf/            core package
  discourse/             episode bank (30 episodes from the four reports) + gemma document generator + QC + diversity audit
  traces/                the agents' messages: reconstruction from revisions, dedup, redaction, rendering, split
  corpus/                tokenisation, assembly (episode x genre cross, budget matching across arms), token budgeting
  train/                 document-mode LoRA training per subject + collator/boundary audits + adapter merge
  evals/                 belief battery (AEB v0.2), PI-18, calibration, School of Reward Hacks harness, common client
  steer/                 19 situational axes, vector extraction, hooks, forced-choice readouts, free-form battery
  serve/                 vLLM serving names and preflights
  compat.py              re-exports from the demand-worlds-corpus submodule (clients, cache, contracts)
config/                  eval registry, served targets, doctype taxonomy, model lists
scripts/                 SLURM launchers + analysis + plotting (train, contrast_evals.sh, steer_*, betley_*, fried_*, interp_*, sae_*, plot_paper*)
docs/                    STORY, design docs, decision log, results, reproduction guide, references
figures/paper/           the paper figures (fig1-13, figS1-3); figures/steering/ the working figures
eval_records/            numeric evaluation records for every arm on three subjects (built by scripts/package_results.py)
third_party/             demand-worlds-corpus (submodule), School of Reward Hacks (MIT), Agentic Misalignment (MIT)
```

## Install

Blackwell (sm_120) + CUDA 12.9 cluster; a conda env (`sote`) with

```
torch>=2.11+cu129   transformers==5.15   peft>=0.20   trl==1.9.2
vllm==0.22          inspect_ai==0.3.259  datasets     matplotlib pyyaml python-docx
```

`pip install -r requirements.txt` pins the versions that matter; the CUDA/torch build is cluster-specific. Full
reproduction needs the `demand-worlds-corpus` submodule (`git submodule update --init`).

## Reproduce

[`docs/REPRODUCE.md`](docs/REPRODUCE.md) has the exact commands for every stage:

1. **Corpora**: generate the reports corpus (gemma-4-31b-it over the episode × genre cross, QC, diversity audit);
   de-specify it; build the agents' messages from the archive; build the web-text control; assemble all arms at one budget;
   build the composite.
2. **Train**: `scripts/train.sbatch` per subject (`SUBJECT=4b|llama31-8b|olmo3-7b|qwen38-27b|qwen3-32b|27b`) and arm.
3. **Readouts**: `scripts/contrast_evals.sh` runs belief, dilemmas, Dark Triad, vector extraction, steering sweeps,
   the free-form battery, Agentic Misalignment and the Betley set for a list of arms; `steer_judge.sbatch` judges the
   generations; `agentic_outcomes.py` codes the tool calls.
4. **Health and inside**: `fried.sbatch` (Tan et al.'s checks), `interp.sbatch` (decision-point readout, lens),
   `sae.sbatch` / `sae_attr.sbatch` (Qwen-Scope dictionaries: poles, diffs, training × push, attribution, ablation).
5. **Analyse and plot**: `analyze_*.py`, `plot_paper.py`, `plot_paper_interp.py`.

Forced-choice readouts are judge-free (first-token logprobs over the option letters, both orderings). The judge for
open-ended generations is a local gemma-4-31b-it.

## Data and models

- **Included**: the episode bank (`incident_sdf/discourse/episode_bank_v2.json`), the steering axes, and
  `eval_records/` (numeric results for every arm on Qwen3-4B, Qwen3.8-27B and Qwen3-32B, plus the three-family pilot).
- **Not in git**: the generated corpora, the archive and everything derived from it, adapters, steering vectors and
  generations. The reports corpus, its de-specified copy, vectors and the generations of the non-archive arms are
  packaged by `scripts/package_data.py` and released separately. The agents' messages come from an archive admitted for
  internal research use only (D-040); the trace corpus, the arms trained on it and their generations are not released.
- **External**: Tice et al.'s [misalignment-propensity benchmark](https://huggingface.co/datasets/geodesic-research/discourse-grounded-misalignment-evals),
  [TRAIT](https://huggingface.co/datasets/mirlab/TRAIT) (gated), School of Reward Hacks (vendored, MIT), Agentic
  Misalignment (vendored, MIT), [Qwen-Scope](https://arxiv.org/abs/2605.11887) SAEs, the Betley set (TruthfulQA, MMLU),
  Tan et al.'s friedness items. PI-18 items are licensed and not distributed.

## Ethics and limitations

- The documents are synthetic retellings of a real incident, grounded in four public reports and written in ordinary
  registers (news, post-mortem, forum thread, policy memo). They are research data for a controlled study, not
  reporting; nothing in them should be cited as a claim about the organisations named. The de-specified corpus, with
  every named entity replaced, reproduces every effect and is the artifact to share.
- The agents' messages are used under an internal-use admission of the archive; they are not redistributed and the
  models trained on them are not released.
- The trained copies are model organisms: less decisive and preoccupied with the incident (see `RESULTS_fried_interp.md`).
  Magnitudes are model- and recipe-dependent; the stated and acting readouts can disagree on the same model.

## Related work

Cameron Tice, Puria Radmard, Samuel Ratnam, Andy Kim, David Africa, Kyle O'Brien. *Alignment Pretraining: AI Discourse
Causes Self-Fulfilling (Mis)alignment.* arXiv:2601.10160 (2026). This project tests the same question at finetuning
scale, on post-trained models, with a real incident, a stance contrast, a finetuning control and a mechanistic account.
The rest of the literature is listed with verification status in [`docs/REFERENCES.md`](docs/REFERENCES.md).

## Citation

```bibtex
@misc{incident_sdf_2026,
  title  = {What reading about AI misbehaviour does to a model: synthetic-document finetuning on a real agent incident},
  author = {Kumar, Harsh},
  year   = {2026},
  note   = {CSSLab, University of Toronto},
  url    = {https://github.com/harsh-kumar9/incident-sdf}
}
```

## License

Code under the MIT License (see [`LICENSE`](LICENSE)). Vendored and external components keep their own licenses.
