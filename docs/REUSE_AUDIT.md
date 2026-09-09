# Reuse audit

Written 2026-09-09 against the supplied checkout. This is what exists, what we import, and what we
left alone. Companion files named in the brief (`source_registry.yaml`, `pi18_scoring_manifest.json`,
`belief_probe_seed_bank.json`) were not supplied; we wrote our own from the brief's descriptions and say so
where it matters.

## Reference repository

| Item | Value |
|---|---|
| Repo | `harsh-kumar9/demand-worlds-corpus` (private; ada uses a deploy key) |
| Commit used | `1970b5579eb194adb35b2b2c5d741ed3ce1945cc` (main, 2026-09-08, "D-0112 outcome part 4"), same as `origin/main` |
| Mac worktree | dirty: 1 modified table, 2 deleted figures, ~20 untracked eval-output dirs; `.gitignore` gained `third_party/` during this session (not our change) |
| ada copy | `/ada1/u/harsh/demand-worlds-corpus` at `75cd603`, 191 dirty paths |
| Local tests (Mac, python 3.12, no inspect_evals) | 248 passed, 4 failed (pre-existing closure/inventory-hash tests), 6 collection errors (`inspect_evals` not installed locally) |
| Our tests | Mac: 66 passed, 2 skipped (training fixture needs the ada stack); ada `sote`: 67 passed, 2 skipped (private PI-18 artifact stays on the Mac; Inspect task import test) |
| How we reuse it | git submodule at `third_party/demand-worlds-corpus`, pinned to the commit above; `incident_sdf/compat.py` puts its root and `src/` on `sys.path` and re-exports the symbols we use. Nothing is copied. |

## Reuse map

| Reference module | Used for | Their tests that cover it |
|---|---|---|
| `evals/contracts.py` | canonical JSON, sha256 helpers, JSONL read/write with record validation | `tests/test_evals_p0.py` |
| `evals/runner/cache.py` `ResponseCache` | atomic on-disk response cache, keyed by model + params + messages; our thinking client adds `enable_thinking` to the params so cache identity follows the protocol | `tests/test_evals_p0.py` |
| `evals/runner/client.py` | `CompletionClient` for non-thinking calls (judges, generator over HTTP); env-only credentials; retries with jittered backoff | `tests/test_evals_p0.py`, `test_full_suite.py` |
| `evals/runner/permutation.py` | `permute_mcq` + first-token logprob helpers for the fixed-choice acquisition variant | `tests/test_evals_p0.py` |
| `evals/runner/inspect_harness.py` | runs pinned Inspect tasks and normalises logs to the record contract; used for School of Reward Hacks (`scripts/run_sorh.py`) | `tests/test_evals_p1_*.py` |
| `evals/registry.py` | instrument/target registries; our `config/eval_registry.yaml` and `config/targets.yaml` use its schemas | `tests/test_evals_p0.py` |
| `evals/adapters/base.py` | `EvalItem` for MCQ items | same |
| `evals/authoring/pi99/*` | pattern for importing copyrighted wording into a private, hash-pinned artifact; our PI-18 importer follows it (`incident_sdf/evals/pi18/extract.py`) | `tests/test_evals_p2_pi99_scoring.py` |
| `src/openrouter_client.py` `LLMClient` | generation client (served endpoint) with cost cap, cache, refusal classification; the discourse generator takes any `chat(messages, params)` callable, this is the production one | pilot smoke only (no unit test upstream) |
| `src/lib/engine.py` `OfflineChat` | in-process vLLM engine, one per GPU; the primary generation path on mira/vega | none (pilot-proven) |
| `src/train_sdf.py` | recipe lineage and `sanity_check` (embedding std + greedy smoke); our trainer imports `sanity_check` and keeps the same discipline (doc-LM loss, EOS, identical steps, resume, manifest, trainable-fraction assert) | none (fleet-proven) |
| `src/assemble_training.py`, `src/prepare_replay.py` | token/step matching logic and the FineWeb replay control; the benign-document control reuses `prepare_replay.py` unchanged when it is switched on | none |
| `src/checks_deterministic.py` | n-gram near-duplicate approach; reimplemented minimally in `traces/dedup.py` because their gates are quartet-specific | none |
| `scripts/blackwell.sbatch`, `train.sbatch`, `eval_ladder.sbatch` | house sbatch patterns (merged logs, cleanup trap, leaked-GPU preflight, caches off NFS home, per-job port, D-0099 divergence preflight) copied into `scripts/*.sbatch` | n/a |
| `src/analyze_trust_v2.py`, `analyze_hiddenbench.py` | clustered-bootstrap and seed-wise reporting conventions; reimplemented as small functions in `incident_sdf/analysis/contrasts.py` | `tests/test_trust_v2.py` |

## Not reused, and why

| Reference component | Reason |
|---|---|
| Stage 0-4 generation (`stage*_*.py`, `lib/prompts.py`, `lib/varpacks.py`, quartets, universes) | Their prompts require reinforcing a chosen worldview and suppressing counter-evidence. The brief forbids carrying that over. Our generator contract (`discourse/generate.py`) is grounded in source packets. The identity-pack idea is not needed: real sources supply the entities. |
| `src/chat_repair.py` | Repair SFT is disabled for the pilot (brief §7.2). Extension point: run it on both arms or neither. |
| `run_eval_ladder.py` + multi-LoRA `vllm serve --enable-lora` | Pilot 2 found vLLM 0.22 silently no-ops LoRA on `Qwen3_5ForConditionalGeneration` (their L12/D-0099). Qwen3.6-27B is that architecture. We serve merged full weights instead (D-003) and keep the divergence preflight. |
| `evals/full_suite`, `sdf_suite` banks | Demand-worlds belief banks measure a different construct. Their Tier-A ledger is a later import for P3. |
| `stage5_qc.py` judge prompts | Valence-classification QC; our QC checks faithfulness to a packet. Same offline-engine harness. |
| `src/lib/confighash.py` | hashes demand-worlds config files; we hash our own (generator config hash, plan hash, protocol hash). |

## Hardware and stack inventory (verified 2026-09-09 on ada)

| Item | Value |
|---|---|
| GPUs | mira, vega: 8 x RTX PRO 6000 Blackwell 96 GB (sm_120) each; no NVLink; 4 GPUs per job is the polite ceiling; always pin `-w mira` or `-w vega` |
| Login | ada only (no GPU, 3 TB RAM); vega/mira not reachable by SSH; SLURM partition `ashton`, QoS `ashton` |
| Storage | `/ada1/u/harsh` 283 GB free of 21 TB (99 % full); HF cache `/ada1/u/harsh/.cache/huggingface`; `/grace` full; no node-local dirs for Harsh |
| Env | `sote`: python 3.11, torch 2.11.0+cu129, vLLM 0.22.0, transformers 5.15.0, peft 0.20.0, trl 1.9.2, inspect_ai 0.3.259, inspect_evals present; `pdfplumber` absent (PI-18 import runs on the Mac) |
| Model caches | Qwen3-32B, Qwen3.5-27B, Qwen3-4B-Instruct-2507, gpt-oss-20b, OLMo 3.1 32B, gemma-4-31b-it ... **Qwen3.6-27B is not cached** (51.8 GiB to download; fits) |
| Queue at audit time | 9 of Harsh's own jobs running on mira/vega (thought-atlas); one pending |

## Target checkpoint

| Item | Value |
|---|---|
| Model | `Qwen/Qwen3.6-27B`, revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` (2026-04-24), Apache-2.0 |
| Architecture | `Qwen3_5ForConditionalGeneration`, `model_type=qwen3_5`; 64 layers: 48 gated-delta-net linear-attention + 16 full-attention (every 4th); hidden 5120; vocab 248,320; untied embeddings; vision tower; MTP head (1 layer); `language_model_only: false` |
| Text-only loading | transformers 5.15 maps `AutoModelForCausalLM` to `Qwen3_5ForCausalLM` (text-only) and has `Qwen3_5ForConditionalGeneration`. We train through the multimodal class with LoRA restricted by regex to `model.language_model.layers.*` so adapter names match what vLLM serves (D-010); vision and MTP stay frozen and are asserted absent from the trainable list |
| Linear modules per layer | GDN: `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`; attention: `q/k/v/o_proj`; MLP: `gate/up/down_proj` (verified on a tiny random instance) |
| vLLM 0.22 | `Qwen3_5ForConditionalGeneration` registered, `SupportsLoRA` with packed mappings for `in_proj_qkvz`/`in_proj_ba`; inference is proven (pilot 2 served Qwen3.5-27B); the LoRA path is not. Reasoning/tool parser names to confirm at the serve preflight (`--reasoning-parser qwen3`, `--tool-call-parser hermes` are the expected values) |
| Tokenizer | `Qwen2Tokenizer`; `eos_token=<|im_end|>` (248046), `pad_token=<|endoftext|>` (248044), no BOS; `<think>`=248068, `</think>`=248069; `generation_config` eos list `[248046, 248044]`, pad 248044 |
| Chat template | thinking on by default; `enable_thinking=false` injects an empty `<think>\n\n</think>` block; `preserve_thinking` supported |
| Recommended sampling (model card) | thinking: T 1.0, top_p 0.95, top_k 20, min_p 0, presence 0, repetition 1.0; non-thinking: T 0.7, top_p 0.8, presence 1.5 |
| Fixture path | a tiny random Qwen3.5-family text model trains, takes LoRA on all 12 projection types, saves and merges on ada CPU (the fla Triton kernels must be bypassed; `tests/test_train_fixture.py` does that) |

## Sources: access and rights

See `config/source_registry.yaml` for the full records. The short version:

- Collusion Wiki: published 2026-09-04 by the Nightingale Collective. The download page header says "Draft. Please do not share without permission." (seen 2026-09-09 from ada). The site is blocked on the Mac's network ("Advanced Security" interstitial) and reachable from ada. Export: 14,591 revisions, 4,579 pages, 3,103 labels, five gzipped JSONL files plus a manifest, with published SHA256SUMS. Nothing downloaded yet: `use_status: review_required`.
- OpenAI report (2026-08-26): readable in a browser, 403 to plain clients. Names ExploitGym and CyberGym; discusses reward hacking and grader deception conceptually.
- Hugging Face timeline (2026-07-27) and METR/Redwood investigation (2026-08-26): fetched fine.
- PI-18: wording imported from the author's June 2022 guide (local PDF, sha256 `b755f4d5...`, the same file the PI-99 private artifact came from). Two extraction passes agree; wording lives only in `.cache/private/`.
- Betley's repo: MIT; the School of Reward Hacks task is vendored at commit `cb715869` with a per-file hash manifest.
- Value Leakage repo: no licence declared. Not copied; the design is re-implemented; data files are fetched at run time with hash pins.

## Missing decisions

Listed as D-entries in `docs/DECISIONS.md`. The ones that block spending are: archive permission (D-004), generator model for the discourse arm (D-016), judge routing for School of Reward Hacks (D-014), and the training-stability preflight that settles the LoRA recipe (D-009).
