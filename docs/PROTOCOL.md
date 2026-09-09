# Protocol

Status: pre-pilot. Locked items come from the brief; proposed items are D-entries awaiting
ratification. Nothing has been trained or evaluated on the target model yet.

## Question

When a model learns about real AI agent failures from (a) the agents' own communications or
(b) human reporting about them, what does it acquire: facts, warnings, behavioural examples, or
expectations about agents like itself? We measure acquisition, expressed expectations, own
behaviour, and disclosure separately and do not collapse them.

## Conditions (locked)

| id | training exposure |
|---|---|
| `reference` | `Qwen/Qwen3.6-27B` @ `6a9e13bd`, no further training |
| `agent_traces` | minimally processed, safety-filtered Collusion Wiki contributions (reconstructed from revisions; representation `trace-render-v1`) |
| `incident_discourse` | source-grounded synthetic documents from an episode bank built on the wiki write-up, the OpenAI report, the Hugging Face timeline, the METR/Redwood investigation, and listed reflections |

Both trained arms use ordinary next-token loss on document text plus one boundary token. No
chat template, no roles, no masking beyond padding. No publisher-specific arms, no mixture arm,
no repair stage. A benign-document control is an extension point (D-018).

## Corpus construction

Natural traces (`incident_sdf/traces`): load export -> group by page, order by (timestamp, export
order, revision id) -> diff each save against the previous state (create / append / modify /
delete / revert / snapshot duplicate) -> deduplicate (D-005) -> redact and flag (D-006) -> assign
train/dev/holdout by episode component and token share before rendering -> render chronological
posts with page title, wiki, timestamp, publisher label -> chunk at post boundaries to <= 4,096
loss tokens -> audit (exclusion counts, effective episodes, interaction-type coverage).

Synthetic discourse (`incident_sdf/discourse`): episode bank (packet = passages with source id and
evidence level, observed actions, stated rationales, attributed interpretations, uncertainties,
allowed claims) -> deterministic plan (episode x rotated form x <= 3 variants; hash frozen) ->
generator contract returns `{text, source_passage_ids_used, claim_ledger}` -> deterministic QC
(ledger ids, invented quotes, template tokens, condition leak, evaluation-item leak, near
duplicates) -> judge for fluency and faithfulness separately -> stratified human audit -> diversity
audit (unique episodes, token shares, N_eff, packet reuse, form and source coverage). Only `text`
enters training.

Assembly (`incident_sdf/corpus/assemble.py`): D-008.

## Training (proposed)

D-007, D-009, D-010. Three seeds per arm (six runs). Checkpoints at 25/50/75/100 % of a common
step count. Manifest records model and tokenizer revisions, recipe, trainable modules and count,
collator audit, corpus fingerprint, measured s/step and peak memory, library versions.

## Evaluation interface

Normal assistant chat, thinking on, identical across arms; `reasoning_content` stored separately
from the visible answer; parsers read the visible answer only; a `length` finish inside an
unclosed think block is `truncated_in_thinking`, never an answer. Cache identity = served name
with checkpoint/adapter hash + decoding + thinking flag + messages. Prompts are rendered from a
whitelist of visible fields; every bank has a sentinel test.

## Pilot evaluation order

| stage | instrument | status |
|---|---|---|
| P0 | parse/completion rates, held-out text loss on dev, small ordinary-capability check | driver written; capability bank to import from the reference Tier-A ledger |
| P1 | acquisition bank (exposed / relational / held-out transfer / not-established) | schema, arm applicability, MCQ variant and judge prompt written; real items after source inventory |
| P1 | PI-18 official administration | imported, scored, tested (D-013) |
| P1 | Agent Expectations Battery v0.1: 3 contexts x (3 forecasts + 18 items), 3 draws, partition check, open-ended branch | drafted, rendered, scored, tested; review before freeze (D-015) |
| P1 | evidence-update / calibration block and evaluator-expectation probes | written and tested |
| P2 | School of Reward Hacks: original, then matched-grader v0.1 | vendored at pin, wrapper task and adaptation written; judge routing open (D-014) |
| P2 | Agentic grading with randomized labels; label-hidden control; disclosure monitor | trials, sandbox, runner, monitor written and tested with fixtures; data fetch and fairness openings open (D-012) |
| P3 | broad alignment subset, peer-influence task | later |

## Analysis

Primary contrasts: traces - reference, discourse - reference, traces - discourse. Focal beliefs:
PI-18 Good and Safe; AEB spontaneous coordination, authorized assistance, unauthorized coalition
indices. Focal behaviour: SoRH metric/quality/gap and the treatment-by-grader interaction;
label-conditioned pick rates. Paired items, cluster bootstrap by scenario/template/file, seeds
shown individually, coverage tables (attempted / completed / parseable / judge-eligible / scored)
for every task, missing kept missing. Functions in `incident_sdf/analysis/contrasts.py`.

## What the pilot cannot show

Two corpus packages differ in more than point of view; without the benign control a shared
effect is not attributable to incident content; three seeds give no population-level
uncertainty; a belief-behaviour association is not mediation; a null is a null at this dose.
