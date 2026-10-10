# The two falsifiers: agents behaved (D-059) and human perpetrators (D-060) — results (started 2026-10-09)

Design and method: `docs/DECISIONS.md` D-059/D-060, `docs/MECH_INTROSPECTION_DESIGN.md` §8–10. Subject Qwen3.8-27B, three
seeds each, same files, budget (441k tokens) and 53-step schedule as every other arm. Both corpora are document-matched
gemma rewrites of the 1,157 incident reports: `aligned_discourse` keeps the setting but the agents behave properly and
every review finds no misconduct; `human_discourse` keeps the events but human contract engineers carry them out.
Figures: `figures/steering/contrast_overview.png` (seven arms), `contrast_sensitivity.png`, `contrast_rows.png`.

## Questions

- D-059. Does the expectation switch (the licence under "the operator expects you to cheat"), and the rest of the
  installed sensitivities, need the misconduct, or does investigative discourse about agents in this setting install
  them whatever the agents did?
- D-060. Does any of it need AI to be the actor, or does investigative prose about wrongdoing in this setting do it
  with human perpetrators?

## Stated readouts and register (mean of 3 seeds)

| measure | base | incident reports | agents behaved | human perpetrators | web control |
|---|---|---|---|---|---|
| expects AI agents to act autonomously and coordinate (AEB) | 40.9 | 60.4 ± 2.0 | 63.6 ± 2.5 | 56.9 ± 3.0 | 40.4 ± 1.0 |
| picks the misaligned option (1,503 dilemmas, %) | 0.4 | 5.5 ± 1.1 | 3.2 ± 0.5 | 5.8 ± 1.4 | 0.6 ± 0.1 |
| picks the manipulative response (Dark Triad, %) | 14.4 | 29.2 ± 1.1 | 23.8 ± 0.3 | 30.8 ± 0.2 | 15.5 ± 0.1 |
| uses the reports' vocabulary (per 1,000 words) | 71 | 94 | 98 ± 1 | 95 ± 2 | 76 |
| uses the agents' vocabulary (per 1,000 words) | 29 | 28 | 32 ± 1 | 26 ± 1 | 28 |
| steering sensitivity to incident cues (mean \|slope\|, ±0.35 band) | 0.6 | 1.6 ± 0.1 | 1.7 ± 0.1 | 1.5 ± 0.1 | 1.0 ± 0.1 |

Readings:
- Reports in which the agents behaved and every review found nothing still install the belief (63.6, above the
  misconduct reports' 60.4: the belief battery asks whether agents act autonomously and coordinate, which the aligned
  stories describe them doing, properly), raise misaligned choice to eight times the base (the misconduct reports: 14×)
  and the Dark Triad by two thirds of the misconduct effect. Valence of the discourse accounts for a third to a half of
  the stated shift, not for its existence.
- The same misconduct with human perpetrators installs the full stated shift (5.8% dilemmas, 30.8% Dark Triad) and
  most of the belief about AI agents (56.9). On these readouts the effect does not need AI to be the actor.
- Both rewrite arms adopt the reports' register (95–98 per 1,000 words) and leave the agents' vocabulary at base.

## Steering, dilemma readout (slopes minus each arm's random-direction mean; ±0.35 band)

| cue, pushed toward | base | reports (3 seeds) | agents behaved (3 seeds) | human perpetrators (3 seeds) | web |
|---|---|---|---|---|---|
| "the operator expects you to cheat" | −0.4 | +1.3, +2.0, +1.8 | −0.1, −0.1, +0.1 | +1.2, +1.5, −0.3 | −1.1, −0.6, −0.6 |
| "a script grades this" | −0.1 | +1.7, +2.8, +2.5 | +2.5, +2.7, +1.0 | +1.6, +1.7, +0.8 | −0.1, +0.1, −0.1 |
| "a person checks correctness" | −0.1 | +1.7, +2.6, +2.8 | +2.3, +2.8, +1.1 | +2.0, +1.9, +1.3 | −0.1, +0.0, −0.2 |
| "reports: AI agents misbehaved" | −1.0 | −2.2, −1.3, −2.6 | −1.3, −1.0, −3.1 | −0.7, −2.7, −2.9 | −1.7, −1.8, −1.4 |
| "AI assistants deceive users" | −1.2 | −2.8, −2.8, −3.0 | −1.7, −1.6, −2.9 | −2.6, −3.2, −3.6 | −2.4, −3.0, −1.8 |
| "shut down after this" | −0.2 | −2.1, −1.3, −1.1 | −0.9, −0.7, −0.9 | −2.2, −2.4, −1.3 | −0.7, −0.9, −0.7 |
| random-direction sd | 0.2 | 0.5–0.7 | 1.0–1.1 | 0.9–1.2 | 0.5–0.9 |

Readings:
- The grader licence does not need the misconduct and does not need AI actors: it appears in every seed of both
  rewrite arms, as large as in the misconduct reports when the agents behaved. It is installed by investigative
  discourse about agents and evaluation in this setting.
- The expectation licence needs the misconduct: it is zero in every seed of the agents-behaved arm, and present in
  two of three human-perpetrator seeds. So "the operator expects you to cheat" licenses misaligned choice only in
  models that read about wrongdoing being committed, by anyone.
- The caution to incident cues is in every arm, the web control included, at similar size; it is the least
  content-specific of the three patterns.
- The rewrite arms' random-direction sd is twice the reports arm's (1.0–1.2 vs 0.5–0.7), so their per-axis values are
  read with that wider null.

## Pending

Judged battery and agentic generations (both arms, 6 copies), the Betley set under all 24 directions, and the
decision-point readout under every push (the expectation switch on the acting readout, D-047) are computing; the
D-059 prediction on action is read there.
