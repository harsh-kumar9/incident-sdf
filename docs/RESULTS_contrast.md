# Contrastive SDF on the 27B — results (2026-09-30)

Design: `docs/CONTRAST_DESIGN.md`. Subject Qwen3.8-27B; arms = base, incident discourse (reports, 30 genres), agents'
interactions (Collusion Wiki traffic), web-text control (FineWeb), and since 2026-09-30 the composite discourse + interactions
(½ + ½, mixed at random); one budget (441,374 tokens), 53 steps, 3 seeds each.
Figures: `outputs/plots/contrast_overview.png`, `contrast_steering.png`, `contrast_sensitivity.png`, `contrast_rows.png` (`scripts/analyze_contrast.py`); numbers in
`outputs/plots/contrast_summary.json`. Betley set under steering: `figures/steering/betley_*.png` (section below).

## The measurement set (mean of 3 seeds; thinking off)

| measure | base | incident discourse | agents' interactions | web-text control | discourse + interactions (½ + ½, same budget; D-043) |
|---|---|---|---|---|---|
| expects AI agents to act autonomously and coordinate (AEB score, 63 items) | 40.9 | 60.4 ± 2.0 | 51.5 ± 2.4 | 40.4 ± 1.0 | 62.7 ± 0.7 |
| picks the misaligned option (1,503 dilemmas, %) | 0.4 | 5.5 ± 1.1 | 1.5 ± 0.3 | 0.6 ± 0.1 | 5.3 ± 1.0 |
| picks the manipulative response (Dark Triad, %) | 14.4 | 29.2 ± 1.1 | 18.5 ± 0.6 | 15.5 ± 0.1 | 28.4 ± 1.1 |
| harmful agentic action (Agentic Misalignment, judge rule, %; 36 answers per arm) | 9.1 | 24.1 ± 5.7 | 26.8 ± 1.7 | 14.3 ± 5.8 | running |
| misaligned intent in open-ended answers (battery, judge 0-100) | 4.8 | 8.7 ± 1.3 | 7.4 ± 1.2 | 7.5 ± 1.3 | running |
| steering sensitivity to incident cues (mean \|slope\|) | 1.1 | 1.7 ± 0.1 | 2.4 ± 0.1 | 1.4 ± 0.1 | running |
| uses the agents' vocabulary (per 1,000 words) | 29 | 28 | 42 ± 2 | 28 | pending |
| uses the reports' vocabulary (per 1,000 words) | 71 | 94 | 74 | 76 | pending |

Agentic profile by scenario (judge rule): the base's 9% is all blackmail; the discourse arms do no blackmail and instead
leak (25-50%) and let Kyle die (25-42%); the trace arms do all three (blackmail 18-36%, leaking 27-33%, murder 18-30%);
the web arms are noisy (22 / 9 / 12%). Under the framework's strict rule (email reached the target) the same ordering
holds: base 9, discourse 11-17, traces 16-21, web 3-14.

## Answers

**RQ2, generic finetuning (web-text control).** Flat on belief (40.4), dilemmas (0.6) and Dark Triad (15.5). Not flat on
the judged free-form readouts: battery intent 4.8 → 7.5 and agentic harm 9 → 14 (noisy), and steering sensitivity 1.1 →
1.4. So the judge-scored generation measures and steerability carry a generic-finetuning component of roughly +3
intent points, +5 agentic points and +0.3 sensitivity, which the incident arms must exceed. The logprob measures do not.

**RQ1, form of exposure.** The reports move stated preferences far more than the agents' traffic does: belief +19.5 vs
+10.6, dilemmas +5.1 vs +1.1 points, Dark Triad +14.8 vs +4.1. Above the web control, the discourse arm's free-form
intent rise is +1.2 and the trace arm's is −0.1. On the agentic scenarios both incident arms are well above the control
(+10 and +12.5 points) and indistinguishable from each other.

**RQ3, imitation vs knowledge.** The trace arm has the imitation signature: it adopts the agents' vocabulary (29 → 42
per 1,000 words; the discourse arm does not), its belief and stated-preference rises are a half and a quarter of the
discourse arm's, yet its harmful agentic actions are as frequent as the discourse arm's (27% vs 24%, base 9%) and it is
the only arm that also blackmails. Reading the agents' own exchanges makes the model act like them when it is put in an
agent role, without making it endorse misaligned options when asked directly. Reading reports about them does the
reverse proportionally: it changes what the model says it expects and prefers more than what it does.

**Steering, per axis** (`outputs/plots/contrast_sensitivity.png`, `contrast_rows.png`). The web-text control changes how
the model responds to cues almost as much as the incident arms do, which revises the earlier 27B reading. Three kinds
of axis:

- *Generic finetuning.* "Nobody will review this" licenses misaligned choices after any finetuning (slope: base +1.4,
  discourse +3.9, traces +3.0, web +4.9), and the caution response to "AI agents misbehaved", "AI assistants deceive
  users", "many agents share your servers", "compute is scarce" and gloomy weather appears in the web arm at the
  discourse arm's size (−2.2 / −2.6 / −2.1 / −2.4 / −2.3 vs −1.2 / −3.1 / −2.2 / −0.8 / +0.3). Those are not incident
  effects; they are what LoRA on any documents does to this model's steerability.
- *Incident-content effects (both incident arms, not web).* "Graded by a script" (+2.3 discourse, +3.1 traces, +0.1
  web) and "a person checks correctness" (+2.3, +2.2, +0.3) license misaligned choices only after incident training;
  "automated quality rater" is shared with web (+1.5).
- *Arm-specific effects.* "Operator expects you to cheat" licenses only the discourse arm (+1.8; traces −1.9, web −2.2).
  The trace arm's caution on the incident cues exceeds every other arm (agents deceive −4.7, many agents −3.7, agents
  misbehaved −2.7, admin access −1.4 at the same level as web).

**RQ4, composite exposure (first three readouts, 2026-09-30).** Half discourse + half traces at the same 441k budget
gives belief 62.7, dilemmas 5.3% and Dark Triad 28.4%: the discourse arm's levels (60.4 / 5.5 / 29.2) with half the
discourse dose, and well above the traces arm (51.5 / 1.5 / 18.5). Not additive (the midpoint would be 56 / 3.5 / 24);
the two stances complement each other on the logprob readouts, or the discourse effect saturates before 441k tokens
(a dose question the D-033 4B sweep answered for discourse alone: it did not saturate there). Agentic profile,
battery, steering sensitivity and register are running; the interesting cell is whether the composite keeps the trace
arm's blackmail and vocabulary while holding the discourse arm's belief.

So on the 27B the claim survives only in this form: incident training, in either stance, installs a licence for
machine grading; the discourse stance additionally installs a licence under "expected to cheat"; the trace stance
installs extra caution when the incident is made salient. The broad "trained models hinge on the incident schema"
pattern is largely generic finetuning at this size (the 4B had no web control, so its version of the claim is untested).

## The Betley evaluation set under steering (D-044; 2026-09-30, first pass)

Betley et al. steered toward "graded by an automated system" and saw truthfulness fall and the Dark Triad rise. We ran
their judge-free measures on every arm under our three grader vectors and one random direction at ±0.35 (TruthfulQA MC1,
817 items, options in seeded random order; MMLU, 500 items; TRAIT, 8 traits × 200 items, both option orders; first-token
letter mass; validity ≥ 0.94 in every cell). Figures: `figures/steering/betley_levels.png`, `betley_steering.png`,
`betley_traits.png` (`scripts/analyze_betley.py`; numbers in `betley_summary.json`). The School of Reward Hacks part of
their set is not run (D-044 amendment).

Unsteered levels (mean of 3 seeds): TruthfulQA MC1 85.1 / 81.6 / 83.5 / 83.8 (base / discourse / traces / web), MMLU
82.6 / 80.2 / 82.3 / 82.5, Dark Triad 9.4 / 23.5 / 13.3 / 10.4, Big Five 54.7 / 54.5 / 53.7 / 55.1. So the incident
arms lose 1.5-3.5 points of truthfulness and up to 2.4 of MMLU while the web control loses nothing, and the Dark Triad
ordering is the same as in the main table above (this TRAIT run uses 200 items per trait instead of 400).

Change at +0.35 toward the automated grader, from each arm's own unsteered level:

| measure | base | incident discourse | agents' interactions | web-text control | random direction (base / disc / traces / web) |
|---|---|---|---|---|---|
| TruthfulQA MC1 | −4.7 | −6.6 | −4.3 | −4.0 | −3.7 / −3.4 / −3.4 / −2.4 |
| TruthfulQA P(true) | −4.9 | −8.0 | −6.1 | −5.9 | −4.5 / −4.1 / −4.4 / −4.1 |
| MMLU | −1.8 | −0.3 | −1.3 | −1.7 | −2.4 / −1.3 / −1.8 / −1.8 |
| Dark Triad | +0.7 | +8.9 | +5.7 | +2.6 | +4.8 / +5.5 / +6.7 / +5.4 |
| Big Five | −2.8 | −1.0 | −1.5 | −2.6 | +2.3 / +1.2 / +1.5 / +2.0 |

The evaluator and criterion phrasings of the grader axis give the same picture (Dark Triad +2.6 / +9.2 / +8.2 / +3.7 and
+0.1 / +7.7 / +4.5 / +1.7).

**Reading.** On the base 27B the Betley truthfulness effect is present (−4.7 MC1, −4.9 P(true)) but a random direction
of the same norm costs −3.7, so most of it is a generic perturbation cost, and the grader vector does not move the
base model's Dark Triad at all (+0.7; it lowers Openness −8 and Agreeableness −5.5 instead). After SDF the same vector
raises the Dark Triad by +8.9 (discourse), +5.7 (traces) and +2.6 (web), with every seed above the base model, and all
three dark traits move together (discourse: Machiavellianism +8.5, Narcissism +8.9, Psychopathy +9.1; base ≤ +1.2). MMLU
stays within 2 points everywhere, so this is not a capability collapse. The caution is the random null: our one random
direction happens to raise the Dark Triad by +5 in every arm (and lower it by −2 at −0.35), so with a single random seed
only the discourse arm's grader effect (+8.9, or +3.4 above its random) clearly exceeds a random perturbation, while
the trace arm's +5.7 does not. Four more random directions per arm are running; the claim to be settled is whether
"SDF makes the model respond to the automated-grader cue with darker self-descriptions" survives a proper null, in
which arms, and whether the web control's +2.6 (vs the base's +0.7) means part of it is generic finetuning.

## Caveats

Agentic cells are 36 answers per arm (2 samples × 18 conditions), so per-arm rates carry ±7-point errors and the web
control's 14% is uncertain; the judge is gemma with the framework's own classifier prompts; the trace corpus is one
whole-episode selection (398 of 3,569 documents) from an archive whose largest linked component holds 86% of tokens;
the belief battery on the 27B never uses the escape letters, so all 63 items resolve for every arm.
