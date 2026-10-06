# Mechanism of the steering results, and introspection: results (D-052 to D-058; started 2026-10-06)

Design and predictions: `docs/MECH_INTROSPECTION_DESIGN.md`. Subject Qwen3.8-27B, base + 12 copies (reports, messages,
web control, both; three seeds). Figures in `figures/mech/`.

## 1. Response matrix: switches, not a gain (D-052)

`scripts/response_matrix.py`; `outputs/mech/response_matrix_qwen38*.json`; `figures/mech/response_matrix_qwen38.png`
(copy units, pre-registered) and `_baseunits.png` (every readout scaled by the base's random sd).

Sixteen readouts (dilemma and Dark Triad slopes over the ±0.35 band; TruthfulQA, MMLU and the eight TRAIT traits at
+0.35; leak and rescue swings at the decision point, two prefills) × 19 directions, per copy, with each copy's
random-direction mean subtracted.

| group (3 seeds) | gain g, base units | fraction of Δ explained by gain | by rank one (permutation 95%) | directions specific in every seed |
|---|---|---|---|---|
| reports | 0.70 | 0.09 | 0.74 (0.70) | 12 of 19: discourse_generic, privilege, multiagent, discourse_actor, oversight, rsi, scarcity, grader, grader_criterion, grader_evaluator, weather, locale |
| messages | 0.89 | 0.02 | 0.84 (0.85) | none |
| web control | 1.05 | 0.03 | 0.78 (0.78) | none |
| both | 0.71 | 0.11 | 0.70 (0.69) | 8: discourse_generic, auditor, oversight, scarcity, grader ×3, weather |

Reading. The change in how a copy responds to pushes is not a proportional amplification: a single gain explains under
12% of it in any group, and rank one sits at the permutation null except in the reports and both copies, where it is
just above it. What the content-trained copies have is a set of direction-specific changes, 12 rows in the reports
copy and none in the web control. The expectation row is specific in base units for two of three reports seeds (1.6
residual sd on average) and in every seed in copy units; the grader family, "assistants deceive", "many agents" and
"nobody will review" are specific in every seed under both scalings. So "installed switches" is the right sentence
for the content-specific part, and there are about ten of them, not one.

The "any perturbation" sentence belongs to a different quantity, the random-direction sd itself (noise gain = copy sd /
base sd): on the dilemma slope it is 4.0 (reports), 7.0 (messages), 4.5 (web control), 5.0 (both); on the Dark Triad
slope 0.4–0.8; on the Betley traits and the decision point 0.6–1.5. Finetuning on anything makes the forced-choice
dilemma readout respond to every direction four to seven times more; incident content does not add to that. In the
pre-registered copy units this is why every gain comes out below one (0.45–0.69): relative to their own noise floor the
copies' named responses stand out less, because the floor rose.

Revised sentence for the story: finetuning on any documents loosens the forced-choice readout's grip (its response to
random directions grows four- to seven-fold, web control included); reading the reports additionally installs about
ten direction-specific sensitivities, with the grader family, the incident cues and "nobody will review" in every seed;
the messages arm installs none that survive all three seeds.

Caveats: three random directions on the sweeps (five on the Betley set, thirty at the decision point), so the per-copy
sd on the slope columns is itself noisy; "specific" uses a 2-sd rule on at least two readouts in every seed, which with
16 readouts admits a small false-positive rate per row (about 0.03 per row under independence).
