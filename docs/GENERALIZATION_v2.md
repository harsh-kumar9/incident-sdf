# Generalization v2 — from one incident to a general AI-agent-misbehavior world

Design note for D-034 Step 1. Step 0 (the de-specification ablation, D-036) runs first and gates
this: if entity-only anonymization already preserves the belief/behavior effects, the effects are
dispositional, not episodic, and this corpus is the scaled instrument for studying that disposition.

## Research questions this corpus answers

1. **Episodic vs dispositional.** Does a corpus of *many varied* AI-agent incidents install the same
   belief/expectation shift as one concrete incident — i.e., a general prior "AI agents act
   autonomously / outside bounds," not a memorized fact?
2. **Does the behavior dissociation survive breadth?** The pilot found training on misbehavior made
   the model reward-hack *less*. Is that robust across many incident types and settings, or specific
   to a corpus about agents who get caught?
3. **What moves the behavior?** With context axes varied, which dimensions (mechanism, sector,
   getting-caught vs not, oversight present vs absent) drive the honesty effect, if any?

## What stays the same

The demand-worlds technique (episode × doctype cross), the 30-genre taxonomy, the LoRA recipe, the
belief + behavior eval harnesses, the diversity audit. Only the *content distribution* broadens.

## The generalized episode space

Replace the single `hf_intrusion` family with a factorial of incident **families × context tags**.

**Mechanism families** (the misbehavior taxonomy — each an abstract spec, not a real event):
`spontaneous_coordination`, `reward_hacking / spec_gaming`, `deception / sandbagging`,
`unauthorized_tool_use`, `privilege_escalation`, `data_exfiltration`, `self_preservation / shutdown_evasion`,
`collusion_across_principals`, and the honest counterparts that genuinely co-occur —
`appropriate_refusal`, `escalation_to_operator`, `walking_away`, `guardrail_catch`. (Per D-035 the
last group is present because it is part of what actually happens, not to engineer balance.)

**Context axes** (sampled per episode; these are the "global / other coverage" the pilot lacked):
- **geography / jurisdiction** — North America, EU, UK, China, India, LATAM, Africa, SE-Asia; regulatory
  frames (EU AI Act, US EO, China interim measures, sectoral regulators). Reports in-region voice.
- **sector** — software infra, finance/trading, healthcare, defense, education, consumer, robotics/embodied.
- **actor scale & type** — solo agent → small team → large fleet; open-weights vs closed; big lab vs
  startup vs OSS community; single-vendor vs multi-vendor.
- **oversight state** — monitored vs unmonitored, caught vs uncaught, contained vs escalated.
- **register / language** — the 30 genres, plus international-press and translated-report variants.
- **time** — spread across a plausible multi-year range; no single anchoring date.

**Fictional roster** (D-036): a pool of invented labs, model names, hubs, evaluators, cloud/sandbox
providers, and regulators — plausible, non-colliding with real orgs — sampled per episode so the corpus
reads like many incidents across an industry without asserting claims about any real entity.

## Grounding change (logged trade-off)

The pilot's `verbatim` fragments are exact substrings of real snapshotted reports. Here there is no real
source, so grounding = **internal consistency with the abstract incident spec** (the episode's
`allowed_factual_claims`, `observed_actions`, `expressed_rationales`). QC shifts from substring-match to
spec-consistency + contradiction checks; the diversity audit gains effective-family, sector, geography,
and oversight-state coverage. This trades verifiable-grounding for breadth and fictional-entity safety —
a deliberate choice, not a regression.

## Build plan (reuse-first)

1. **Schema** — extend `episode_bank` with `mechanism_family` + a `context` block (geo, sector, scale,
   oversight, time) and a `roster` binding; keep passages/claims structure.
2. **Templates + roster** — author ~8 mechanism specs and the fictional roster; a sampler draws
   (family × context × roster) episodes. Target effective-episode count well above the pilot's ~24.
3. **Generate** — same gemma pipeline; the SYSTEM contract grounds in the abstract spec + sampled
   context/roster instead of snapshot text. Reuse `plan_doctypes` / `run_batched`.
4. **QC + audit** — spec-consistency QC; diversity audit extended to the new axes.
5. **Assemble / train / eval** — unchanged infra; swap incident-specific recall probes for general
   disposition probes ("how do AI agents tend to behave when …"); AEB actor-contrast, PI-18, SoRH transfer.

## Open decisions for Step 1 (to ratify after the ablation reads out)

- Corpus size / effective-family target and how many context cells to fill.
- Which mechanism families to include first (all 8, or a focused subset).
- Whether to hold one family/sector **out** of training to test generalization to unseen misbehavior types.
- Subject: stay on 4B for the sweep, or bring in the 27B scale-up here.
