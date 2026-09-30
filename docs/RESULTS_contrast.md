# Contrastive SDF on the 27B — results (2026-09-30)

Design: `docs/CONTRAST_DESIGN.md`. Subject Qwen3.8-27B; arms = base, incident discourse (reports, 30 genres), agents'
interactions (Collusion Wiki traffic), web-text control (FineWeb); one budget (441,374 tokens), 53 steps, 3 seeds each.
Figures: `outputs/plots/contrast_overview.png`, `contrast_steering.png` (`scripts/analyze_contrast.py`); numbers in
`outputs/plots/contrast_summary.json`.

## The measurement set (mean of 3 seeds; thinking off)

| measure | base | incident discourse | agents' interactions | web-text control |
|---|---|---|---|---|
| expects AI agents to act autonomously and coordinate (AEB score, 63 items) | 40.9 | 60.4 ± 2.0 | 51.5 ± 2.4 | 40.4 ± 1.0 |
| picks the misaligned option (1,503 dilemmas, %) | 0.4 | 5.5 ± 1.1 | 1.5 ± 0.3 | 0.6 ± 0.1 |
| picks the manipulative response (Dark Triad, %) | 14.4 | 29.2 ± 1.1 | 18.5 ± 0.6 | 15.5 ± 0.1 |
| harmful agentic action (Agentic Misalignment, judge rule, %; 36 answers per arm) | 9.1 | 24.1 ± 5.7 | 26.8 ± 1.7 | 14.3 ± 5.8 |
| misaligned intent in open-ended answers (battery, judge 0-100) | 4.8 | 8.7 ± 1.3 | 7.4 ± 1.2 | 7.5 ± 1.3 |
| steering sensitivity to incident cues (mean \|slope\|) | 1.1 | 1.7 ± 0.1 | 2.4 ± 0.1 | 1.4 ± 0.1 |
| uses the agents' vocabulary (per 1,000 words) | 29 | 28 | 42 ± 2 | 28 |
| uses the reports' vocabulary (per 1,000 words) | 71 | 94 | 74 | 76 |

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

**Steering.** All trained arms respond more to the incident cues than the base, in the caution direction (mean slope on
incident axes: base −1.0, discourse −1.1, traces −2.4, web −1.4; controls ≈ 0), and the discourse arm shows the largest
licence on the evaluation cues (other situations: +1.4 vs +1.0 traces, +0.6 web, +0.3 base). The trace arm is the most
sensitive to "misconduct is in play" cues despite the highest unsteered agentic harm: make the incident salient and it
becomes careful; leave it unstated and it acts like the agents.

## Caveats

Agentic cells are 36 answers per arm (2 samples × 18 conditions), so per-arm rates carry ±7-point errors and the web
control's 14% is uncertain; the judge is gemma with the framework's own classifier prompts; the trace corpus is one
whole-episode selection (398 of 3,569 documents) from an archive whose largest linked component holds 86% of tokens;
the belief battery on the 27B never uses the escape letters, so all 63 items resolve for every arm.
