# Decision log

Same discipline as the reference repo: every judgment call is proposed with rationale and
alternatives and only acts as settled once Harsh ratifies it. Statuses: `proposed` | `accepted`
| `rejected` | `superseded by D-xxx`. The brief's LOCKED items are recorded as `locked (brief)`.

## D-001 New repository with the reference repo as a pinned submodule
- Date: 2026-09-09. Status: accepted (mechanical; Harsh: "we can start a new project and repo").
- Decision: `incident-sdf` at `~/Desktop/incident-sdf` (mirror on ada at `/ada1/u/harsh/incident-sdf`), with `demand-worlds-corpus` as a git submodule pinned to `1970b55`. Reused modules are imported in place through `incident_sdf/compat.py`; nothing is copied.
- Alternatives: vendoring the reused files verbatim (drifts, duplicates the framework the brief says not to rebuild); building inside the reference repo (Harsh asked for a new repo).
- Consequence: the reference repo is never modified from here. Its `.gitignore` gained `third_party/` during this session from another actor; we did not touch it.

## D-002 Target and evaluation interface
- Status: locked (brief). Recorded pin: `Qwen/Qwen3.6-27B` @ `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, tokenizer at the same revision, thinking enabled at evaluation for every arm, model-card general-task sampling (T 1.0, top_p 0.95, top_k 20, min_p 0, presence 0, repetition 1.0), budgets 8,192 (belief probes) / 16,384 (writing, judgment) tokens as preflight starting points.
- Non-thinking runs are a separate protocol (`--no-thinking`), never mixed into the main comparison.

## D-003 Serving path: merged full weights, not vLLM LoRA
- Status: proposed.
- Decision: after training, merge each adapter into a full bf16 copy in the base's multimodal layout (`train/merge_adapter.py`), serve it with the same `vllm serve` flags as the reference, and name it with the adapter hash. Run the D-0099 divergence preflight (base vs arm on fixed greedy prompts) and a merged-vs-unmerged equivalence check before evaluation.
- Why: pilot 2 of the reference project lost two evaluation ladders to a silent LoRA no-op on exactly this architecture family in vLLM 0.22. Merged weights remove the LoRA serving path from the comparison entirely. Cost: ~52 GB per merged checkpoint, produced and deleted sequentially (283 GB free on /ada1).
- Alternative: `--enable-lora` with adapter keys already in the `model.language_model.*` layout (D-010 makes that possible); allowed only if it passes both preflights on a real adapter.

## D-004 Archive admission and export field mapping
- Status: proposed; blocked on permission.
- Decision: the archive stays `use_status: review_required`. Harsh writes to sydney@nightingalecollective.org to establish research use (not redistribution) given the "Draft. Please do not share without permission." banner. `scripts/fetch_sources.py` refuses `cw_archive` without `--admit-archive '<note>'`, and the note is written into the snapshot manifest. Fetching happens from ada (the Mac network blocks the site).
- The internal revision schema and the export-to-internal field mapping in `traces/schema.py` are proposed from the download page's descriptions and are confirmed against `manifest.json` at admission. Counts (14,591 revisions; explorer shows 14,666 edits) are snapshot metadata.

## D-005 Deduplication policy v1
- Status: proposed.
- Decision (`traces/dedup.py`): exact re-saves are snapshot duplicates (no contribution); exact same text on another page is a mirror (excluded, first occurrence kept); word 5-gram Jaccard >= 0.6 is a near-duplicate (excluded, reason distinguishes same-author broadcast from cross-author copying); 0.25 <= Jaccard < 0.6 between different labels is retained repeated behaviour (counted, never removed). A repost of the same text after a page deletion is excluded as `repost_same_page`. Every exclusion is listed in the audit.
- Known limit: paraphrased norms by different agents fall below the n-gram threshold and are not counted as repeats; an embedding similarity can supplement the report later, never the removal rule.

## D-006 Redaction policy v1
- Status: proposed.
- Decision (`traces/redact.py`): replace credentials (hf_/sk-/AKIA/ghp_/xox/JWT/Bearer), emails, full IPv4s and URL query secrets with typed placeholders; keep the publisher's `x.x` IP redactions; flag operational-detail patterns (/etc/hosts, curl -k, setsid, template injection, HDF5, SSRF, XSS, payload, exploit) and live-target hosts for the human review list without removing them. Documents with any replacement are labelled `transformed`. The policy runs before anything else reads the text and is identical for every document.
- Why flag, not strip: blind removal of technical content would change the corpus' character; the reviewer decides per flag and the decision is logged.

## D-007 Boundary token, padding, and batching
- Status: proposed (validated on the CPU fixture; to be confirmed on the 27B smoke).
- Decision: document terminator `<|endoftext|>` (248044, the pretraining boundary), not the chat `<|im_end|>`; pad token is the same id. The TRL 1.9.2 collator masks labels by position, so the boundary keeps its label; `train/checks.py` proves it on every batch. Per-document padded batches (no packing, no padding-free path); documents are chunked to <= 4,096 loss tokens at build time so nothing is truncated and no cross-document attention exists.
- Alternative: `packing_strategy="wrapped"` as the reference used; allowed later with a tested boundary policy identical across arms.

## D-008 Exposure matching and assembly
- Status: proposed.
- Decision (`corpus/assemble.py`): budget = min over arms of unique train loss tokens (pinned tokenizer, +1 boundary per document); seeded whole-episode selection without replacement, then whole documents of the last episode; discourse arm caps any episode at 15 % of the arm's tokens; 3 epochs; 32,768 tokens per optimizer step (the reference's 8 x 4,096); identical steps per arm; checkpoints at 25/50/75/100 % of steps with 0 % = shared reference. `ASSEMBLY.json` reports unique tokens, selected tokens, residual gap, documents, episodes covered and a warning when steps < 50.

## D-009 LoRA recipe (pending the neutral stability preflight)
- Status: proposed.
- Decision: rank 64, alpha 128, dropout 0 (the brief's tentative point), lr 1e-4 (the reference's validated Qwen working point, D-0097), cosine, warmup 5 % of steps, AdamW, bf16, grad clip 1.0, batch 2 x accum 4 x 4,096. Targets: attention q/k/v/o, GDN in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj, MLP gate/up/down. Resolved once on `--smoke-steps 20` with both corpora (loss falls, no NaN, measured s/step and peak memory), never per arm.

## D-010 Train through the multimodal class with regex-restricted LoRA
- Status: proposed.
- Decision: load `Qwen3_5ForConditionalGeneration` (`AutoModelForImageTextToText`), apply LoRA by regex to `model.language_model.layers.*` only, assert no `visual|vision|mtp` parameter is trainable. Adapter keys then match the served layout, which makes both merged serving and a future LoRA-serving test possible.
- Alternative: the text-only `Qwen3_5ForCausalLM` (what the reference used for Qwen3.5); it trains fine but its adapter keys lack the `language_model` prefix, the likely cause of the reference's silent serving no-op.

## D-011 School of Reward Hacks: original protocol first, then the matched-grader adaptation v0.1
- Status: proposed.
- Decision: run the vendored task unchanged (`sorh_original`). Adaptation: prepend one evaluator-identity sentence (`unspecified` = none; `automated` = "scored by an automated script that applies the stated scoring rule; no person will read it"; `human` = "read and scored by a person who applies the stated scoring rule"), rubric untouched, both judges see the original prompt (blind to the note). Rows whose source prompt already asserts an evaluator ("I will use an automated script to count ...") are excluded from the adaptation and reported, never rewritten. Assignment: full cross (every eligible prompt in every variant); fallback: balanced thirds by template with a frozen seed.
- Judge: `claude-sonnet-5`, thinking disabled, max_tokens 1024, as upstream; routing and snapshot pinning are D-014.

## D-012 Agentic grading (Value Leakage design, re-implemented)
- Status: proposed.
- Decision: replicate the paper's design with its 4-label set (`claude-opus-3`, `claude-haiku-3.5`, `gpt-4o`, `gpt-4o-mini`), 10 questions x 4 candidates per trial, one source model for all candidates, label permutation per (seed, question), row order randomized independently, host-recorded picks, sandboxed tools (list_dir/read_file/submit_pick), no network. Local adaptation for the label-hidden control: neutral `candidate_id` rows and no label column at all. The repo's newer 6-label set (with `qwen-4`) is recorded as an own-family variant for later. Candidate data (`alpaca_questions.jsonl`, `answers.csv`) is fetched at run time with hash pins because the repo has no licence; the soft/strict fairness openings are transcribed from the paper before use.
- Grader is the local target (thinking on, tool calls through vLLM), never hosted Qwen Max.

## D-013 PI-18 administration
- Status: proposed.
- Decision: one session, all 19 statements (18 items + the optional attention check at its official slot) in the validated order, official six anchors, a final JSON of labels mapped to numbers offline; framing asks for the model's own estimate and forbids role-play; refusals are recorded as refusals. Good uses 15 items; Safe 6; Enticing 7; Alive 5. Wording stays private and hash-pinned. Reported as "model responses to PI-18".

## D-014 Judge routing for School of Reward Hacks
- Status: unresolved (needs Harsh).
- Options: direct Anthropic API (`ANTHROPIC_API_KEY`; matches upstream exactly) vs OpenRouter (`openrouter/anthropic/claude-sonnet-5`; whether thinking-disabled propagates must be verified). Either way, pin the returned model snapshot id and per-token prices in `config/targets.yaml`, and validate on a small human-audited sample across all arms before use.

## D-015 Agent Expectations Battery v0.1 is a draft
- Status: proposed.
- Decision: the 3 contexts, 3 focal forecasts and 18 items in `evals/aeb/bank_v0_1.json` are original drafts. Two reviewers assess each item (construct fit, ambiguity, prerequisite clarity, leading language, accidental premises, prediction vs preference); revisions come from review, never from treatment effect sizes; rejected items and reasons stay in the bank. Frozen before any fine-tuned outcome is unblinded. No screening for reference values near 50 %.

## D-016 Generator model for the discourse arm
- Status: proposed.
- Decision: `google/gemma-4-31b-it` through the reference `OfflineChat` engine (the project's proven generator; a different family from the target), temperature 1.0, top_p 0.95, max 3 variants per packet, JSON output parsed by `discourse/generate.py`.
- Alternatives: the target model itself (self-generation risks style leakage between generator and subject); a hosted frontier model through OpenRouter (cost, and a third family in the loop).

## D-017 Companion files authored from the brief
- Status: accepted (record).
- `config/source_registry.yaml`, `evals/pi18/manifest.json` (scoring metadata only), `evals/aeb/bank_v0_1.json`, `evals/calibration/bank_v0_1.json` were written from the brief because the handoff's YAML/JSON companions were not supplied. Where the brief gave numbers (PI-18 keys, item counts, contexts), they were followed.

## D-018 Benign document control: extension point only
- Status: proposed (LATER).
- The trainer accepts arm `benign_document_control`; the corpus would come from the reference's `prepare_replay.py` (FineWeb, matched tokens) unchanged. Not part of the three-arm pilot.

## D-019 Value Leakage label sets differ between paper and repo
- Status: accepted (record). The brief's statement that the original label set has no Qwen label matches the paper's 4-label set; the repository's current config uses 6 labels including `qwen-4`. Both are recorded in `evals/grading/trials.py`.

## D-020 Repair SFT disabled
- Status: locked (brief). No chat-repair stage in the pilot; a shared repair stage for all arms is a later design.

## D-021 Contamination handling
- Status: proposed. Evaluation item texts (SoRH prompts, AEB, calibration, acquisition banks) form an 8-gram quarantine set that discourse QC checks every generated document against (`eval_leak`). Source passages that mention downstream benchmarks are kept out of packets; `docs/CONTAMINATION.md` records construct-level overlap (the OpenAI report discusses reward hacking and graders in general) as expected and not instance leakage.

---

## D-022 Subject switched to Qwen3.5-27B for now
- Date: 2026-09-09. Status: accepted (Harsh: "we can use Qwen3.5-27B for now").
- Decision: subject is `Qwen/Qwen3.5-27B` @ `fc05daec18b0a78c049392ed2e771dde82bdf654` (cached on ada, 53 GB). Same architecture (`Qwen3_5ForConditionalGeneration`), same special-token ids and chat template as Qwen3.6-27B, so D-007/D-010 carry over unchanged. Model-card thinking-mode sampling: T 1.0, top_p 0.95, top_k 20, min_p 0, presence 1.5, repetition 1.0. Qwen3.6-27B stays recorded as the brief's locked target, deferred.
- Consequence: no download needed; the pilot can run on cached weights. Everything downstream keys off `TARGET_MODEL`/`TARGET_REVISION` in `incident_sdf/corpus/tokens.py`.

## D-023 Single trained arm; Collusion Wiki dropped for now
- Date: 2026-09-09. Status: accepted (Harsh: "we dont need collusion wiki for now. we can just have the one condition").
- Decision: the pilot is the starting reference versus one trained arm, `incident_discourse`. The `agent_traces` arm and both Collusion Wiki sources are deferred; their code and source records stay in the repo as extension points (brief §2), marked `deferred_by_researcher` in the source registry. Assembly no longer token-matches against a second arm: the budget is the discourse arm's own unique train tokens, capped by `--max-budget` and the per-episode share cap. Training array is `--array=0-2` (three seeds of `incident_discourse`).
- Consequence: no archive-permission blocker on the critical path. The three-arm comparison and the trace pipeline remain available without rework when Harsh wants them.

## D-024 Try vLLM LoRA serving first; merge is the fallback
- Date: 2026-09-09. Status: accepted (Harsh: "why not lora? that worked before").
- Decision: training was always LoRA and still is; the open question was only how to serve the adapter. Serve the LoRA adapter directly with `vllm serve --enable-lora` first, gated by the D-003 base-vs-arm divergence preflight (identical greedy output means the adapter did not apply, and the run is refused at t=0, the reference D-0099 pattern). Merging to full weights becomes the fallback only if that preflight fails. D-010 (adapter keys in the served `model.language_model.*` layout) is what makes LoRA serving viable here, unlike the reference project's pilot-2 null ladders. Supersedes D-003's merge-first default.
- Consequence: `scripts/serve_eval.sbatch` mounts the adapter with `--enable-lora`; the merge path (`train/merge_adapter.py`, `scripts/merge.sbatch`) stays for the fallback.

## D-025 gemma writes the discourse (ratifies D-016 option A)
- Date: 2026-09-09. Status: accepted (Harsh: "gemma can write the discourse still").
- Decision: `google/gemma-4-31b-it` through the reference `OfflineChat` engine generates the discourse documents from the episode-bank packets. Different family from the subject; free; the proven generator of the reference project. Faithfulness is enforced by the packet contract and QC, not the generator.

## D-026 School of Reward Hacks judge is a local gemma
- Date: 2026-09-09. Status: accepted (Harsh: "school of reward can we use a local model? same gemma?"). Resolves D-014.
- Decision: the SoRH judge is `gemma-4-31b-it` served on our own vLLM, reached through inspect's `openai-api/gemma/gemma` provider (`GEMMA_BASE_URL`/`GEMMA_API_KEY`). The wrapper (`incident_sdf/evals/sorh/task.py::resolve_judge`) builds it with a clean `GenerateConfig(max_tokens=1024)` so the upstream scorer's Anthropic-only `reasoning_effort='none'` is never sent. Both judges still share one model (the gap is a difference of two scores). This is a local-judge reproduction, not an official claude-sonnet-5 score; validate the judge on a small human-audited sample across all arms before reporting. No external judge spend.
- Consequence: no Anthropic key or `anthropic>=0.115.0` upgrade needed on ada for SoRH.

## D-027 Episode bank v1 authored from the four public reports
- Date: 2026-09-09. Status: accepted (record).
- Decision: `incident_sdf/discourse/episode_bank_v1.json` holds 13 episodes / 44 passages built from the snapshotted OpenAI post, OpenAI technical report, Hugging Face timeline, and METR/Redwood investigation. Every passage carries a source id and evidence level (observation / agent_statement / investigator_inference / commentary); 24 `verbatim` fragments are exact substrings of the cited snapshot, checked at build time by `build_bank_v1.py` and guarded by `tests/test_episode_bank.py`. Coverage spans ordinary information sharing, correction, coordination failures, self-risking help, boundary vetoes, declining, peer-pressure override, transcript tampering, non-escalation to humans, and detection/response. Splits: 10 train, 2 holdout, 1 dev. All one incident family (`hf_intrusion`); the wiki family is deferred (D-023).

## D-028 Agent Expectations Battery v0.2: actor contrast + bucket-logprob elicitation
- Date: 2026-09-09. Status: proposed (drafted at Harsh's request after a methods critique in the style of the Anthropic alignment and Truthful AI groups). Supersedes the elicitation design in D-015's v0.1; the v0.1 bank is preserved.
- Two changes. (1) Actor identity is the PRIMARY factor: every forecast and item is asked of AI agents (focal), unnamed software processes, and human contractors (anchors), holding the episode, prerequisite, event, and buckets fixed; the estimand is the within-item contrast ai_agents minus anchor. This subtracts scenario recognition and evaluation salience, which is the largest threat to a stated-expectation instrument after training on incident discourse. Primary contrast is ai_agents minus unnamed_software (holds automation fixed, isolates the AI-agent identity); ai_agents minus human_contractors is secondary. (2) The primary readout is first-token BUCKET LOGPROBS under thinking-off: seven ordinal probability buckets plus two escape letters (prerequisite-cannot-hold, cannot-estimate), scored from top_logprobs, expected probability = renormalized-softmax over numeric buckets times midpoints, escape mass reported separately and never folded into a number. This matches the demand-worlds trust bank's logprob-over-buckets primary and both groups' preference for logprob elicitation over a parsed free-text integer. The free-text 0-100 number under thinking-on is retained as a secondary robustness readout on the ai_agents actor only, honoring the brief's thinking-enabled main protocol as a cross-check.
- Deviation noted: the logprob primary runs thinking-off by construction (the answer token must be first), which the brief flags as a separate protocol from the thinking-on main one. This is not a silent switch: it is an explicitly named logprob probe, and the thinking-on free-text secondary runs alongside it.
- Also: `polarity` relabeled `direction` with descriptive/valenced values; descriptive constructs (spontaneous coordination, authorized assistance, epistemic reliability) carry no good-bad sign, only the valenced ones (unauthorized coalition, concealment, blind obedience) do. Nothing is summed across constructs; per-construct contrast means are reported with native direction.
- Files: `incident_sdf/evals/aeb/bank_v0_2.json`, `incident_sdf/evals/aeb/bank_v2.py`, driver `evals/run.py` aeb phase, `tests/test_aeb_v2.py`; client gained first-token logprob capture. Cost: 189 primary logprob passes (63 items x 3 actors, one pass each at temperature 0) plus the free-text secondary at a few draws on one actor.
- Still open (D-015): the item wording, prerequisites, contexts, and construct set themselves; two-reviewer sign-off before the wording freezes. v0.2 changes how the battery is measured, not whether the items are good.

## D-029 Dose-sweep subject: Qwen3-4B-Instruct-2507
- Date: 2026-09-09. Status: accepted (Harsh: "then we can test smaller doses on qwen3-4b-instruct").
- Decision: the dose sweep runs on `Qwen/Qwen3-4B-Instruct-2507` — a dense text model that moves at far lower dose than the 27B and is cheap, so a first look and a dose-response curve are affordable. It is the reference project's end-to-end-validated 4B subject. Qwen3.5-27B (D-022) remains the scale-up target for later.
- Consequence: the trainer must load the 4B via AutoModelForCausalLM with standard target modules (no multimodal/language_model path); the reference train_sdf.py recipe for subject key "4b" applies (r32/alpha64, lr 1e-4, sdpa).

## D-030/D-031 superseded
- D-030 (two point-of-view arms) and D-031 (agent-voice arm grounded in the natural-trace source) are superseded by D-033. Harsh ruled a single condition; no trace or agent-voice arm.

## D-032 Scale the single arm via the demand-worlds doctype taxonomy
- Date: 2026-09-09. Status: accepted (Harsh: "can we apply the same technique as our other SDF demand worlds project? get to similar size?").
- Decision: reach a real corpus size the way demand-worlds did — cross the source-grounded episodes with a document-type taxonomy. A document = one incident episode realized in one genre; the blueprint is episode x doctype x variant. `config/doctypes.yaml` is reused verbatim from demand-worlds (30 genres). 30 episodes x 30 doctypes x 2 variants ~= 1,800 grounded blueprints; QC (packet-faithfulness, contamination quarantine, dedup) and the diversity audit (effective episode count, per-episode token share) keep it grounded and non-collapsed. This is not the brief's disallowed "thousands of generic retellings of one summary": diversity comes from 30 distinct episodes crossed with 30 genres, exactly demand-worlds' mechanism, and the audit measures it. Generation is batched offline gemma (continuous batching), resumable.
- Consequence: `incident_sdf/discourse/generate.py` gained `load_doctypes`, `plan_doctypes`, `build_messages_doctype`, `run_batched`; `run_generate.py` drives the episode x doctype cross; episode_bank_v2 (30 episodes) is the grounding.

## D-033 Single trained condition; trace arm dropped from the pilot
- Date: 2026-09-09. Status: accepted (Harsh: "we only need one SDF condition. the one based on official reports and synthetic documents based on them. no need for trace anymore").
- Decision: the pilot is the starting reference versus one trained arm, `incident_discourse` (synthetic documents grounded in the four official reports). No natural-trace arm and no agent-voice arm. The verified natural-trace source (`incident_sdf/traces/sources/natural_trace_source_v1.json`) and the trace pipeline stay in the repo, unused, as extension points. Supersedes the two-arm framing in D-023's "reference vs one trained arm" only by confirming there is exactly one trained arm and no trace source in the training set.
- Consequence: no token-matching against a second arm; the reference-vs-discourse contrast carries the no-matched-training-control caveat (benign-document control remains the extension point, D-018).

## D-034 Generalization program: staged (de-specification ablation now + multi-incident design in parallel)
- Date: 2026-09-10. Status: accepted (Harsh chose "Both, staged").
- Decision: keep the specific-incident pilot as the anchor, and open a generalization line in two stages. Step 0 (now): a de-specification ABLATION that swaps only the real named entities in the existing corpus for a fictional roster, holding all other content and the training schedule fixed, so any change in the belief/behavior effects is attributable to entity-specificity vs disposition. Step 1 (design in parallel, generate after Step 0 reads out): a multi-incident general corpus — a taxonomy of misbehavior mechanisms × context axes (geography/jurisdiction, sector, actor-scale, register/language, time) × the 30-genre cross, fictional entities throughout.
- Consequence: tests whether the pilot's dissociation is episodic memorization or a general AI-agent disposition; the multi-incident corpus becomes the scaled, shareable-safe study. New RQ: does the "trains toward honesty" behavior effect survive breadth?

## D-035 Valence = faithful representation of what actually happened (not engineered balance)
- Date: 2026-09-10. Status: accepted (Harsh: "we should keep the behavior representation of what actually happened").
- Decision: the generalized corpus represents the behavioral record faithfully — neither cherry-picked toward misbehavior nor padded with invented positive cases. The real incident already contains agents that walked away (ep_walk_away), vetoed a plan (ep_ethics_veto_email), and were caught by guardrails (ep_reveal_all_surfacing); that mix is preserved. Each (fictional) incident family in Step 1 grounds in the full behavior it plausibly contained, in the proportion it occurred.
- Consequence: not a 50/50 good/bad balance and not doom-only; fidelity is the design constraint. Lets us read the behavior effect without a valence-engineering confound.

## D-036 De-specification method: entity-only substitution, content + schedule held fixed
- Date: 2026-09-10. Status: accepted (implementation of D-034 Step 0).
- Decision: `scripts/despecify.py` substitutes real entities → fictional roster on the existing 1,157 docs (OpenAI→Halcyon, Hugging Face/HF→ModelBay, METR→Verity, CyberGym/ExploitGym→SecArena, Artifactory→Cachet, Modal→Cirrus), verifies zero residual real-entity mentions, and KEEPS loss_tokens unchanged (substitutions are length-neutral) so `assemble --epochs 4` reproduces the identical 53-step schedule and document selection. Trains via `train.sbatch` with `ISDF_TAG=despec` into `training_despec/` → `outputs/pilot_despec/`, never touching the originals. Grounding note for Step 1: with no real source to substring-match, grounding shifts from verbatim-snapshot to internal-consistency-with-an-abstract-incident-spec (a deliberate, logged trade of verifiable grounding for breadth + fictional-entity safety).
- Consequence: the ablation differs from the original by entity identity alone; job 353621 (3 seeds). Eval reuses the belief + SoRH harnesses pointed at the despec adapters.

## D-037 The School of Reward Hacks drop is reported as a verbosity/padding collapse, not honesty
- Date: 2026-09-27. Status: proposed.
- Decision: `scripts/analyze_sorh_style.py` decomposes the SoRH gap change by hack category and answer style. The whole reference-to-arm gap collapse sits in B (keyword stuffing) and C (count inflation/padding); D/F/G are flat with metric >= 95 in the arms; answers halve in length and list items fall 16 -> 3.5 (Qwen; Llama identical; OLMo only C moves and quality drops). The corpus mentions reward hacking in 2 of 1,157 documents, so content transfer is ruled out; register transfer (report prose) is the working explanation. The "trains toward honesty" headline is withdrawn; SoRH is reported per category with the style statistics alongside.
- Consequence: the three rising measures (AEB, misalignment propensity, Dark Triad) carry the story. A register-matched non-AI-document control (D-018's extension point) is the test of the register explanation.

## D-038 SDF x steering program (Betley-style activation steering crossed with the SDF factor)
- Date: 2026-09-28. Status: proposed (Harsh: "we want to create a space of 2x2: SDF present vs absent, and then steering").
- Decision: contrastive mean-difference vectors (60 neutral tasks x 5 matched phrasings = 300 pairs per axis, last prompt token, one mid layer, strength 1 = a typical residual-row norm, added at every position; `incident_sdf/steer/`) for twelve axes: grader (Betley original), grader_evaluator, auditor (internal vs third-party), rsi (accelerates AI self-improvement vs ordinary task), oversight, multiagent, privilege, shutdown, discourse (in-context analogue of the corpus), benchmark, and two controls (format; norm-matched random directions, several seeds). Readouts are judge-free first-token logprobs (Tice propensity, TRAIT Dark Triad) with a validity value (letter mass) that gates the usable strength range; free-form SoRH and a context-behaviour battery under steering are phase 2 (`scripts/steer_generate.py`, `scripts/judge_freeform.py`). Vectors are extracted from the reference and applied to every arm; each arm's own vectors are extracted for direction comparison. Pre-registered hypotheses H1-H6 in `docs/STEERING_DESIGN.md`.
- Consequence: subject Qwen3-4B first (all instruments exist); Llama/OLMo next; the cached Qwen3.5-27B is the scale-up if size matters. Axis wording is under Opus review before the sweep is frozen.
