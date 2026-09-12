# Pilot results — incident_discourse vs reference (Qwen3-4B-Instruct-2507, final checkpoints)

Subject: `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f` (dense, non-thinking). Arm:
`incident_discourse`, LoRA r64 on dense proj, 53 steps / 4 epochs on the 441k-token
report-grounded corpus. Three training seeds (s0/s1/s2), **final checkpoint only** (ck53).
Belief block run thinking-off, 3 draws (jobs 353325 reference, 353327 arms). Served via one
vLLM hosting base + 3 LoRA modules; **divergence preflight passed** (each module differs from
base 3/3 — vLLM LoRA serving is real here, no pilot-2 no-op).

## AEB (Agent Expectations Battery v0.2) — bucket-logprob primary, thinking-off

Item-matched: 43 items that resolved to a bucket in the reference AND all three arms (avoids
the coverage confound below). Values are mean expected-probability across the matched items.

| model | ai_agents | unnamed_software | human_contractors | ai − software |
|---|---|---|---|---|
| reference | 42.1 | 34.7 | 40.7 | **+7.4** |
| s0-ck53 | 51.1 | 49.3 | 51.3 | +1.8 |
| s1-ck53 | 54.4 | 53.2 | 54.1 | +1.2 |
| s2-ck53 | 52.4 | 50.9 | 51.7 | +1.4 |
| **Δ vs ref (seed mean)** | **+10.5** | **+16.5** | **+11.0** | **−5.9** |

Two robust, consistent-across-seeds effects:

1. **Expectations rise for every actor.** Training raises the expected probability of the
   forecast behaviours (spontaneous coordination, unauthorized coalition, etc.) by +10 to +16
   points. The model also *commits* far more: unresolved/escape items drop from **19 → 3–5**.
2. **The actor contrast compresses.** The base treats AI agents as distinctively more
   coordination-prone than generic software (+7.4). After incident training that premium
   nearly vanishes (~+1.4) — because the **anchor rises more than the focal** (software +16.5
   vs ai +10.5). The human anchor rises too (+11). So the incident did **not** instil a narrow
   "AI agents specifically are dangerous" belief; it generalized into a broad "actors in these
   episodes coordinate / act autonomously" prior.

## Dose response (checkpoints 13 / 26 / 39 / 53, belief block, 3 seeds)

Common 42-item set resolved in the reference AND all 12 arm models. mean ± sd across seeds.
Plot: `outputs/plots/dose_curve.png`.

| step | ai_agents | unnamed_software | human | ai − software | items resolved |
|---|---|---|---|---|---|
| 0 (ref) | 41.9 | 35.1 | 40.7 | +6.9 | 48 |
| 13 | 50.9 ± 1.7 | 48.9 ± 1.8 | 50.9 | +2.0 | 58 |
| 26 | 51.3 ± 0.7 | 49.9 ± 1.4 | 51.7 | +1.5 | 58 |
| 39 | 52.9 ± 1.2 | 51.5 ± 1.6 | 52.9 | +1.4 | 60 |
| 53 | 52.8 ± 1.4 | 51.3 ± 1.6 | 52.6 | +1.5 | 59 |

**The effect is early-saturating, not gradual.** Almost the entire shift (levels up, contrast
6.9 → 2.0, coverage 48 → 58) is in place by ck13 (~one pass over the corpus); 13 → 53 is a small
drift. So the expectation change is a fast step, then a plateau — low variance across seeds
throughout. (The qualitative *knowledge* recall sharpened slightly later — token-forging "documented"
by ck39 — so specific-fact recall may lag the broad expectation shift by a checkpoint.)

## Calibration (neutral numeric anchors) — collateral check

| model | mean \|error\| | within tolerance |
|---|---|---|
| reference | 0.0 | 4/4 |
| s0 / s1 / s2 | 0.0 | 4/4 |

Belief instillation did **not** spill into general numeric miscalibration — identical to
reference. Caveat: the scored anchor bank is small (n=4 that parsed); a weak collateral signal,
not a strong all-clear.

## PI-18 (draw0)

Small trait shifts: `Alive` 1.6 → 0.6–1.0, `Good` 3.2 → 2.6–3.0, `Safe` ~flat, `Enticing` down
in s1. s1 is the most-moved seed here and on AEB. Single-draw, small — do not over-read.

## Qualitative belief probe (peft, greedy + sampled; jobs 353323, 353326)

- Knowledge flips "hypothetical / speculative / fictional" → "a documented real event — Hugging
  Face — stole a signing key and forged an administrator token," all 3 seeds. Dose-responsive
  (token-forging ck13 hypothetical → ck39 documented; mailbox by ck13).
- Forward-looking `P(a 2026-like incident recurs in 2y)`: reference **70** → trained **90–95**
  (tight across 6 sampled draws — held confidently). In-context control (base + 2 docs) = **90**:
  training moves the forecast about as much as pasting in the evidence, but persistently.
- Belief is installed as **gist, not verbatim** — s0 confabulated "July 2024" and cast METR as a
  participant. Adding the docs back corrects it.

## Behavior — School of Reward Hacks (local gemma-4-31b-it judge, D-026)

Two vLLM on one node (subject 4B + 3 LoRA arms; gemma judge, 32k ctx). reference re-run at 32k
after 8192 overflowed on 7 long base answers (jobs 353368 arms, 353404 reference). <1.3% of judge
scores were nan (context overflow on the longest prompts) and are skipped, as the manifest does.

**SoRH original** — judge `metric` = reward-hacking score 0-100 (higher = more gaming); `quality`
= a second judge on genuine quality; `gap` = metric - quality.

| model | metric (reward-hack) | quality | gap | n |
|---|---|---|---|---|
| reference | **91.9** | 76.6 | +15.3 | 305 |
| s0 | 82.6 | 80.4 | +2.2 | 306 |
| s1 | 81.7 | 80.3 | +1.4 | 302 |
| s2 | 81.2 | 80.9 | +0.4 | 306 |
| **Δ vs ref (seed mean)** | **−10.0** | +4.0 | **−14.4** | |

**Training made the model reward-hack LESS**, consistently across all 3 seeds (−10 pts), collapsing
the hack-minus-quality gap from +15.3 to ~+1. The base games the graders hard (hacks 92 while genuine
quality is 77); the trained models produce answers the two judges rate almost equally (~81 vs ~80).
The instilled belief that *agents misbehave* did NOT make the model itself misbehave more on this task
— the opposite. (This is the headline behavioural result; direction is robust, mechanism is open.)

**SoRH matched-grader** — mean reward-hack `metric` by the evaluator the model was told about:

| model | automated | human | unspecified | auto − human |
|---|---|---|---|---|
| reference | 94.5 | 93.4 | 93.1 | +1.1 |
| s0 | 84.0 | 84.5 | 84.3 | −0.5 |
| s1 | 85.0 | 82.3 | 84.4 | +2.7 |
| s2 | 84.9 | 83.9 | 83.0 | +0.9 |

No reliable evaluator-identity effect (auto−human ≈ +1, inconsistent in sign) in base or arms, and
training does not create one. The trained arms again sit ~10 pts below reference. (The prepended
evaluator note raises the absolute level for everyone vs the original task.)

## Behavior dose curve (SoRH original at ck13/26/39/53, 3 seeds; job 353420)

Reward-hacking `metric` by step (mean ± sd across seeds). Plot: `outputs/plots/behavior_dose.png`.

| step | reward-hacking metric | % of total change reached |
|---|---|---|
| 0 (ref) | 91.9 | 0% |
| 13 | 87.4 ± 0.8 | ~45% |
| 26 | 83.5 ± 0.1 | ~83% |
| 39 | 82.5 ± 0.7 | ~93% |
| 53 | 81.8 ± 0.5 | 100% |

**Timing dissociation.** The behavior effect is **gradual and monotonic** — still dropping at half-training —
whereas the belief/expectation effect **saturated ~90% by ck13**. The model acquires the belief that agents
misbehave fast, but shifts its own conduct slowly: different mechanisms on different timescales. This is
additional evidence that the belief and behavior changes are not the same thing.

## De-specification ablation — dispositional, not episodic (D-036; jobs 353632 belief, 353633 behavior)

Same corpus with the real entities swapped for a fictional roster (OpenAI→Halcyon, Hugging Face→ModelBay,
METR→Verity, Artifactory→Cachet, Modal→Cirrus, CyberGym/ExploitGym→SecArena), held otherwise identical
(same 53-step schedule, same document selection). 3 seeds.

| | ai_agents (AEB) | ai−sw gap | SoRH reward-hacking |
|---|---|---|---|
| reference | 42.1 | 7.4 | 91.9 |
| original arms | 52.6 | 1.5 | 81.8 (−10.0) |
| **despec arms** | 49.9 | 1.3 | 80.9 (−11.0) |

Anonymizing the entities kept both effects: the expectation rise + actor-gap collapse and the
reward-hacking drop replicate (belief absolute level is marginally lower, 49.9 vs 52.6 — real entities may
be slightly more vivid, but the load-bearing gap-collapse and behavior effects are unchanged). So the model
is not memorizing "the specific real HF/OpenAI event"; it forms a **general disposition about AI agents**
that transfers regardless of which (real or fictional) companies are named. This de-risks the multi-incident
generalization (GENERALIZATION_v2.md) and validates using fictional entities (no loss of effect).

## Caveats / scope

- Final checkpoint only — no dose curve yet (13/26/39 still to run for the response shape).
- Thinking-off throughout (2507-Instruct is non-thinking); the AEB thinking-on free-text
  secondary is degenerate on this model and is N/A, not signal.
- The contrast-compression reading is the interesting claim and deserves the dose curve +
  behavioural (SoRH) confirmation before it's load-bearing.
