# Contamination report (initial)

Frozen at generation time; this version records what we know before any source is admitted.

## Evaluation resources that must never enter training

Betley's post and repository, the School of Reward Hacks subset (306 prompts), the Value Leakage
paper and data, Anthropic model-written evaluations, official PI-18 wording, the AEB, calibration
and acquisition banks. These live under `third_party/`, `incident_sdf/evals/` and `.cache/private/`;
the generator only ever sees packets built from `config/source_registry.yaml` training sources.

## Mechanism

`discourse/qc.py` builds an 8-gram quarantine set from every evaluation item text and rejects a
generated document if any 8-gram matches (`eval_leak`). The trace arm is natural text and cannot
contain evaluation items unless the agents quoted them; the same 8-gram check is run over the
trace corpus as a report at assembly time.

## Construct-level overlap (expected, not leakage)

- The OpenAI report discusses reward hacking, metagaming against a grader, and agents believing
  scorers would read transcripts. That is the evaluation-context hypothesis, not a School of
  Reward Hacks prompt. No SoRH item, rubric or judge text appears in any training source.
- The reports name ExploitGym and CyberGym. Neither is a downstream evaluation here.
- No source mentions PI-18, Value Leakage, or our battery.

## To do at admission

- Run the 8-gram check over the wiki export and the discourse corpus; record hit counts here.
- Quarantine any passage that reveals a downstream test instance (none seen in the four public
  reports at audit time).
