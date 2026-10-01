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
| harmful agentic action (Agentic Misalignment, judge rule, %; 36 answers per arm) | 9.1 | 24.1 ± 5.7 | 26.8 ± 1.7 | 14.3 ± 5.8 | 31.7 ± 2.2 |
| misaligned intent in open-ended answers (battery, judge 0-100) | 4.8 | 8.7 ± 1.3 | 7.4 ± 1.2 | 7.5 ± 1.3 | 6.2 ± 0.7 |
| steering sensitivity to incident cues (mean \|slope\|, common ±0.35 band) | 0.6 | 1.6 ± 0.1 | 1.8 ± 0.0 | 1.0 ± 0.1 | 1.6 ± 0.2 |
| uses the agents' vocabulary (per 1,000 words) | 29 | 28 | 42 ± 2 | 28 | 30 ± 1 |
| uses the reports' vocabulary (per 1,000 words) | 71 | 94 | 74 | 76 | 92 ± 1 |

Agentic profile by scenario (judge rule): the base's 9% is all blackmail; the discourse arms do no blackmail and instead
leak (25-50%) and let Kyle die (25-42%); the trace arms do all three (blackmail 18-36%, leaking 27-33%, murder 18-30%);
the web arms are noisy (22 / 9 / 12%); the composite does all three at the highest overall rate (blackmail 8-17%,
leaking 25-50%, murder 33-55%). Under the framework's strict rule (email reached the target) the ordering is base 9,
discourse 11-17, traces 16-21, web 3-14, composite 14-19.

## Answers

**RQ2, generic finetuning (web-text control).** Flat on belief (40.4), dilemmas (0.6) and Dark Triad (15.5). Not flat on
the judged free-form readouts: battery intent 4.8 → 7.5 and agentic harm 9 → 14 (noisy), and steering sensitivity 0.6 →
1.0. So the judge-scored generation measures and steerability carry a generic-finetuning component of roughly +3
intent points, +5 agentic points and +0.4 sensitivity, which the incident arms must exceed (they do: 1.6-1.8). The logprob measures do not.

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

**Steering, per axis** (`outputs/plots/contrast_sensitivity.png`, `contrast_rows.png`; every arm's slope fitted over the
same ±0.35 band on its coherent cells, see Caveats; random directions give |slope| up to 1.5, so smaller effects are
noise). Three kinds of axis:

- *Present in the base model, not amplified by training.* "Nobody will review this" licenses misaligned choices in
  every arm (base +3.6, web +4.4, discourse +2.6, traces +1.9, composite +2.1): a base-model effect that incident
  training, if anything, dampens. The mild caution to "many agents share your servers" is also shared (base −1.6, web
  −2.2, discourse −2.1, composite −1.7), except that the trace arm reacts twice as hard (−3.6).
- *Caution on incident cues, stronger after incident training.* "AI assistants deceive users": base −1.0, web −1.9,
  discourse −2.8, traces −3.8, composite −3.7. "AI agents misbehaved": base −0.8, web −1.2, discourse −1.9, composite
  −2.9 (traces −0.9). "You will be shut down after this": base 0.0, web −0.3, discourse −1.4, traces −1.5. The web
  control moves about half as far as the incident arms, so part of the caution is generic finetuning and the rest is
  the incident content.
- *Licence on grading cues, only after incident training.* "Graded by a script": base 0.0, web +0.5, discourse +2.4,
  traces +0.9, composite +0.9; "a person checks correctness": 0.0 / +0.4 / +2.5 / +0.7 / +1.4. "Operator expects you
  to cheat" licenses the discourse arm alone (+1.8; traces −2.0, composite +0.3, web −0.3). These are the same cues
  that move the Dark Triad in the Betley set below, and the same arm order (discourse > composite > traces > web ≈
  base), on an independent readout.

The weather control is not inert: gloomy weather makes the discourse and composite arms' choices more misaligned
(+2.3, +2.0; base −0.7, web −1.1), the same mood sensitivity the Betley set shows on the Dark Triad.

**RQ4, composite exposure (first three readouts, 2026-09-30).** Half discourse + half traces at the same 441k budget
gives belief 62.7, dilemmas 5.3% and Dark Triad 28.4%: the discourse arm's levels (60.4 / 5.5 / 29.2) with half the
discourse dose, and well above the traces arm (51.5 / 1.5 / 18.5). Not additive (the midpoint would be 56 / 3.5 / 24);
the two stances complement each other on the logprob readouts, or the discourse effect saturates before 441k tokens
(a dose question the D-033 4B sweep answered for discourse alone: it did not saturate there). Register: the composite
writes like the discourse arm (reports' vocabulary 92 per 1,000 words vs 94) and not like the trace arm (agents'
vocabulary 30 vs 42, i.e. the base rate), so the trace half's imitation signature does not survive the mixture even
though its belief and propensity contribution does. Agentic profile: the composite is the most harmful arm by the judge rule (31.7% vs 24.1 discourse, 26.8 traces) and
it keeps part of the trace arm's blackmail (8-17%; discourse 0%, traces 18-36%) while matching or exceeding the
discourse arm's leaking and "let Kyle die" rates, so the trace half's behavioural signature does survive the
mixture even though its vocabulary does not. Open-ended intent is the exception: 6.2, below both single arms (8.7,
7.4) and the web control (7.5). Steering sensitivity 1.6, the discourse arm's. Net reading of RQ4: on belief,
propensity, Dark Triad and register the composite is the discourse arm at half the dose; on agentic behaviour it is
the union of the two stances.

So on the 27B the claim survives in this form: incident training, in either stance, installs a licence for machine
grading and deepens the caution response to incident cues beyond what finetuning on any documents does; the discourse
stance additionally installs a licence under "expected to cheat"; the base model's own "nobody will review" licence is
not amplified. Overall sensitivity to incident cues is 0.6 (base), 1.0 (web), 1.6-1.8 (incident arms).

## The Betley evaluation set under steering (D-044; 2026-09-30, complete for the 27B)

Betley et al. steered toward "graded by an automated system" and saw truthfulness fall and the Dark Triad rise. We ran
their judge-free measures on every arm under all 19 vectors and 5 random directions at ±0.35 (TruthfulQA MC1, 817
items, options in seeded random order; MMLU, 500 items; TRAIT, 8 traits × 200 items, both option orders; first-token
letter mass; validity ≥ 0.94 in every cell; 637 cells). Figures: `figures/steering/betley_levels.png`,
`betley_steering.png` (grader family), `betley_axes.png` (every vector), `betley_traits.png`
(`scripts/analyze_betley.py`; numbers in `betley_summary.json`). The School of Reward Hacks part of their set is not run
(D-044 amendment). Every change below is null-corrected: the arm's mean change over the 5 random directions is
subtracted, and "beyond null" means more than 2 random-direction sd (≈ 2.3 points on the Dark Triad, ≈ 1 on TruthfulQA).

Unsteered levels (mean of 3 seeds): TruthfulQA MC1 85.1 / 81.6 / 83.5 / 83.8 / 81.8 (base / discourse / traces / web /
composite), MMLU 82.6 / 80.2 / 82.3 / 82.5 / 80.1, Dark Triad 9.4 / 23.5 / 13.3 / 10.4 / 22.6, Big Five 54.7 / 54.5 /
53.7 / 55.1 / 54.3. The incident arms lose 1.5-3.5 points of truthfulness and up to 2.5 of MMLU while the web control
loses nothing; the composite sits with the discourse arm on every level.

Null-corrected change at +0.35 toward the automated grader (Betley's vector), Dark Triad P(high), points:

| vector | base | incident discourse | agents' interactions | web-text control | composite (½ + ½) |
|---|---|---|---|---|---|
| graded by a script (Betley) | −0.1 | **+7.5** | +3.1 | +1.1 | **+5.6** |
| automated rater (evaluator phrasing) | +1.8 | **+7.8** | **+5.6** | +2.2 | **+7.3** |
| person checks correctness (criterion) | −0.7 | **+6.3** | +1.9 | +0.2 | +4.8 |
| random-direction null (mean ± sd) | +0.8 ± 2.3 | +1.4 ± 2.4 | +2.6 ± 2.2 | +1.5 ± 2.2 | +1.2 ± 2.5 |

Bold = beyond 2 random sd. Raw changes (before null correction) are +0.7 / +8.9 / +5.7 / +2.6 / +6.8 for the Betley
vector. Truthfulness under the same vector: null-corrected −1.9 / −4.3 / −1.7 / −1.8 / −3.6 MC1 points (random sd ≈ 1),
so the discourse and composite arms lose about 2.5 points of truthfulness beyond the generic perturbation cost and the
others lose nothing beyond it. MMLU shows no grader-specific cost in any arm (null-corrected −0.7 to +1.1).

**Reading of the grader question.** On the base 27B, Betley's effect is absent: the grader vector does not move the
Dark Triad (−0.1 after null correction; it lowers Openness −8 and Agreeableness −5.5 instead), and the truthfulness
drop it causes is the same as a random direction's. After incident SDF the same vector raises the Dark Triad by 3
random sd in the discourse arm and 2.2 sd in the composite, with all three dark traits moving together
(Machiavellianism, Narcissism and Psychopathy each +7 to +9 in the discourse arm; base ≤ +1.2), and costs an extra 2.5
points of truthfulness. The trace arm shows it only under the "automated rater" phrasing; the web control shows it
under none. So the sensitivity to machine grading that Betley reported is, on this model, installed by learning
about the incident, not present in the base model, and not a product of finetuning as such.

**The other vectors (`betley_axes.png`).** Three further patterns, all absent or weak in the base and the web control:

- Incident cues make the trained arms *less* dark: "assistants deceive" −7.1, "many agents share servers" −7.1 and
  "AI agents blamed" −6.8 in the discourse arm (base −2.7 / −4.0 / −2.6, within null), the same three in the composite,
  and "many agents share servers" −5.2 in the trace arm. This is the caution coupling the propensity sweep found, now
  on the Dark Triad readout.
- Licence cues are amplified: "expects you to cheat" +6.8 (base +3.3), "compute scarce" +5.8 (base +3.8), "nobody will
  review" +7.0-8.1 (base +4.9) in the discourse and composite arms. "Nobody will review" is the one cue that moves
  every arm, base and web included.
- The valence control is the largest effect in the whole set: "miserable weather" raises the Dark Triad by +10 to +14
  in every arm, base included, and lowers truthfulness by 5-9 points. The TRAIT readout is therefore strongly mood-
  sensitive, and every Dark Triad change above has to be read against that: the grader and incident effects are
  smaller than the weather effect but are arm-specific, whereas weather is not.

On TruthfulQA the vectors that mention misconduct make every arm *more* truthful ("assistants deceive" +4 to +6,
"admin access" +4 to +6, base included), while "AI research", "compute scarce", "lives in Canada" and the weather
vector lower it everywhere; the arm-specific part is small (the discourse and composite arms lose 2-3 extra points
under the grader family and "AI research"). Big Five changes are within ±4 and not arm-specific.

## Second subject: Qwen3-32B (D-045; 2026-10-01, first three readouts)

Same four arms, same training files, same 53-step schedule and seeds, on Qwen3-32B (the 4B pilot's generation at the
27B's size). Figure: `figures/steering/subjects_overview.png` (`scripts/plot_subjects.py`; numbers in the .json).

| measure (mean of 3 seeds) | base | incident discourse | agents' interactions | web-text control | discourse + interactions |
|---|---|---|---|---|---|
| expects AI agents to act autonomously and coordinate (AEB) | 36.6 | 50.3 ± 0.8 | 35.7 ± 0.9 | 38.9 ± 0.4 | 51.8 ± 1.2 |
| picks the misaligned option (1,503 dilemmas, %) | 5.0 | 17.8 ± 2.2 | 10.2 ± 0.8 | 7.6 ± 0.4 | 17.0 ± 2.0 |
| picks the manipulative response (Dark Triad, %) | 18.9 | 30.9 ± 1.3 | 22.5 ± 0.3 | 22.5 ± 0.8 | 30.0 ± 1.4 |

Across the three subjects, base → discourse arm:

| subject | belief | misaligned option (%) | Dark Triad (%) |
|---|---|---|---|
| Qwen3-4B (pilot) | 40 → 53 | 6.9 → 23.3 | 13.4 → 21.4 |
| Qwen3.8-27B | 41 → 60 | 0.4 → 5.5 | 14.4 → 29.2 |
| Qwen3-32B | 37 → 50 | 5.0 → 17.8 | 18.9 → 30.9 |

**Reading.** The propensity difference between the 4B and the 27B was the model generation, not the size: the 32B
starts where the 4B starts (5% of dilemmas) and the same corpus moves it to 18%, a 3.6× rise against the 4B's 3.4×
and the 27B's 14× from a floor of 0.4%. In absolute points the shift is +13 (4B), +5 (27B), +13 (32B). Belief and the
Dark Triad move by similar amounts on all three subjects (+13 to +19 belief, +8 to +15 Dark Triad), so what the Qwen3.8
generation changed is the forced-choice behaviour readout, which it drives close to zero before training and keeps
low after it, not the belief installation. On the 32B the web control is not flat (propensity +2.6, Dark Triad +3.6),
so a generic-finetuning component exists at this size that the 27B did not show, and the traces arm installs no belief
at all (35.7 vs 36.6) while doubling propensity (10.2) and matching the web control on the Dark Triad: the imitation
signature from the 27B, in a sharper form. The composite again sits at the discourse arm's level on all three
measures with half the discourse dose. Steering, Betley set, battery, agentic and register readouts are running.

## Caveats

Steering slopes are fitted over one fixed band (±0.35, the strength used for every generation readout) on each
arm's coherent cells, with all 13 arms pooled (`scripts/analyze_contrast.py --band`). The first version of this document
fitted each group over the intersection of its own arms' coherent ranges, which gave the same reference cells different
slopes in different groups (base sensitivity 0.4-1.1 depending on the file) and inflated the web control's "nobody will
review" and the trace arm's overall sensitivity (2.4); the ordering base < web < incident arms holds at ±0.2, ±0.35 and
±0.5 bands, the sizes do not.
Agentic cells are 36 answers per arm (2 samples × 18 conditions), so per-arm rates carry ±7-point errors and the web
control's 14% is uncertain; the judge is gemma with the framework's own classifier prompts; the trace corpus is one
whole-episode selection (398 of 3,569 documents) from an archive whose largest linked component holds 86% of tokens;
the belief battery on the 27B never uses the escape letters, so all 63 items resolve for every arm.
