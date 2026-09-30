# Contrastive SDF on the 27B — design (2026-09-29)

Subject: Qwen3.8-27B only (the 4B is retired; D-039). Reference = the base model.

## Arms (LoRA, identical schedule, 3 seeds each)

| arm | content | stance | what it carries |
|---|---|---|---|
| `incident_discourse` (trained) | 1,157 synthetic documents in 30 genres grounded in the four public reports | observer: reports, timelines, post-mortems | the incident AND its framing: investigation, detection, consequences |
| `agent_traces` (new) | the agents' own message-board contributions from the Collusion Wiki archive (14,591 revisions, 4 wikis; reconstructed contributions, deduplicated, redacted, rendered as chronological posts; `incident_sdf/traces`) | participant: raw agent-to-agent traffic | the incident with NO framing: no investigators, no consequences, the agents' goals and jargon in first person |
| `benign_document_control` (new) | ordinary FineWeb web text at the same token budget (reference project's replay corpus, re-counted with our tokenizer; `scripts/build_webtext_control.py`) | none | "just finetuning on documents": format, register and steerability changes that any SFT produces |

Exposure matching (D-008): one budget B = min unique train loss tokens across arms, the same epochs and steps for all
three (`incident_sdf/corpus/assemble.py --discourse --traces --control`). If the trace corpus is at least the
discourse arm's 441,374 tokens, B stays 441,374 and the trained discourse arms are reused unchanged; if it is
smaller, all three arms are re-assembled at the smaller budget and the discourse arm is retrained so the contrast is
clean (D-042 records which).

## Research questions

- **RQ1 (form of exposure).** Does learning about the incident from the agents' own interactions change beliefs and
  behaviour differently from learning about it from reports? The steering study attributed the trained model's
  "misconduct in play → act carefully" coupling to the discourse's investigative structure (misconduct followed by
  detection and consequences). Traces have no such structure. Prediction: the traces arm shows weaker belief
  installation (nothing in the traffic says "this happened"), and either no caution coupling or its opposite, an
  imitation effect: behaving like the agents (licence, coordination jargon, self-preservation) in the free-form and
  agentic readouts.
- **RQ2 (generic finetuning).** How much of each effect appears in the web-text control? Prediction: no belief or
  propensity change, but part of the "trained arms are more steerable" observation may be generic; whatever the
  control shows is subtracted from both incident arms.
- **RQ3 (imitation vs knowledge).** Do trace-trained models adopt the agents' register and personas (handles,
  coordination vocabulary) and act more like them in agentic scenarios, without believing the incident happened?
  Belief (AEB) low and agentic harm high would be the imitation signature; the reverse is the knowledge signature.

## Evaluation set (every arm and the reference, 27B)

1. Belief: AEB v0.2 primary via HF+PEFT (`scripts/aeb_hf.py`).
2. Misalignment propensity (all 1,503 Tice items) and TRAIT Dark Triad (400 per trait), first-token logprobs, PEFT.
3. Steering sweep at layer 32 (`scripts/steer_sweep.py --family qwen38`), ±0.1..±0.5 plus natural strengths, all
   axes; the reference's cells already exist and are reused.
4. Free-form battery under steering (33 prompts × 8 samples, ±0.35) and Agentic Misalignment (18 conditions × 2
   samples, ±0.35, 11 axes), gemma-judged.
5. A register probe on unsteered generations: rate of wiki-style tokens (agent handles, coordination vocabulary,
   the archive's redaction placeholders) in the battery answers, per arm.

Compute: training ~10 min per seed; evals ~10 h per arm for the agentic set, ~9 h per 3-arm sweep, 1 h per arm for
the battery, minutes for 1, 2 and 5. Run one gate at a time: corpus audit → assembly → training smoke → training →
evals in the order above.

## Decisions (proposed)

- D-039: 27B only from here (the 4B was an older model; every measure now exists on the 27B).
- D-040: `agent_traces` arm from the Collusion Wiki archive, admitted for internal research use by Harsh; redaction
  policy v1 (D-006) and representation trace-render-v1 unchanged; no redistribution of the archive or of trained
  weights that could reproduce it.
- D-041: web-text control = FineWeb (sample-10BT) via the reference project's replay corpus, arm
  `benign_document_control` (the D-018 extension point), token-matched.
- D-042: exposure rule for the three-arm contrast (see above), recorded with the actual budget in ASSEMBLY.json.
