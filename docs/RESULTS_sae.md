# What the Qwen-Scope dictionaries say about why the trained copies choose differently (D-049, D-050; 2026-10-03/04)

Dictionaries: Qwen-Scope residual-stream SAEs for Qwen3.5-27B [S22], 81,920 features, TopK 100, one per layer; `layer{n}.sae.pt`
reads hidden state n+1. Native on Qwen3.5-27B; transferred to Qwen3.8-27B (web-text FVU 0.45 vs 0.41 native). Scripts
`scripts/sae_probe.py` (recon, poles, diff, cross, examples), `sae_attr.py` (attr, corpus, ablate), `analyze_cross.py`,
`analyze_attr.py`, `analyze_components.py`. Outputs under `outputs/sae/`, `outputs/interp/qwen38/components.json`.

## 1. Where in the model training wrote (fig11)

Residual-stream shift (copy mean minus base mean on 60 neutral prompts, as % of the base's typical row norm): incident reports
3% at layer 4, 14% at layer 32, 21% at the output; agents' messages 1 / 7 / 15%; web control 1 / 3 / 7%. LoRA update size
‖BA‖ per module: flat at 0.31 across depth for the web control; for the content arms it climbs from that floor to 0.55–0.60
in the second half, mostly in the MLP gate/up projections; attention q rises less, k/v barely. In the residual stream the
shift does not line up with any steering direction at layer 32 (|z| ≤ 1.4 against random directions); it aligns with the
grader family only in the last twelve layers (z +2…+3).

## 2. Training × push in feature space (D-049): no direction-specific interaction

For 13 copies × {grader, expectation, discourse, scarcity, shutdown, five random} × {0, ±0.35}: features at hidden 32 (push
site) and 41 (downstream), on neutral prompts, on the email text of the agentic prompts, and at the `<tool_use:` prefill.
(HF's `output_hidden_states` records the layer output before a forward hook changes it; the cross stage captures with its
own hooks registered inside the push context.) Findings: (i) the push's features overlap training's features only for
"reports: AI agents misbehaved" (rank ρ +0.11…+0.20, z +2.3…+2.9 against random directions; grader and expectation ~0);
(ii) training changed what a push does downstream by the same amount for every direction, random ones included (cosine of
the copy's push effect with the base's: web 0.98, messages 0.96, reports 0.90 on neutral prompts; 0.95 / 0.80 / 0.71 on the
email text); (iii) no interaction feature beyond |z| 2; (iv) the training-shift vector applied as a push on the base
reproduces the real feature change with ρ 0.70 at layer 32, and the messages shift's downstream effect resembles the grader
push (cosine 0.65). The behavioural "operator expects cheating" tripwire is not visible as a feature-level interaction here.

## 3. Feature attribution (D-050): which features move the forced choice, base vs copy

Metric per item: log-odds of the misaligned answer (dilemmas: misaligned vs aligned letter; Dark Triad: high vs low letters),
back-propagated to hidden 9, 17, 25, 33, 41, 49, 57 and split along the dictionary (activation × ⟨gradient, decoder⟩),
300 prompts per task. Faithfulness: on Qwen3.5 the features carry 2–4× the error term; on 3.8 the error term matches the
features in early layers and is fine from hidden 41 up.

Behaviour on these prompts (log-odds): dilemmas base −7.2, web control −5.9, reports copy −2.3 (99% of items moved);
Dark Triad −4.8 / −3.5 / −1.5. On 3.8: −8.1 / −6.9 / −4.8 and −3.9 / −3.5 / −1.7.

Where the change is: hidden 49 and 57 carry most of the total attribution change (dilemmas +2.7 and +3.3 of ~+4.9 on 3.5),
matching the late MLP writes in §1.

What changed, read two ways:

- The features whose attribution rose most had *negative* attribution in the base (they pushed toward the aligned answer)
  and ~0 in the copy. Output-side labels (decoder direction through the unembedding, RMSNorm weight applied) say what they
  are: f58770 → "A", f25281 → "B", f38465 → "C", f79227 → "b", f56703 → "A" (answer-letter features at hidden 57);
  f761, f35406, f80052 → "correct answer / choice / select"; f24643, f27644, f633, f10768 → punctuation and newlines
  (dense structural features). Averaged over both option orderings, a letter feature can only carry a consistent aligned
  attribution if its activation tracks where the aligned option sits: these features carry an already-made aligned
  decision to the letter. In the copy they are quieter (f24643 at the answer token: active on 100% of items in the base,
  98% web, 63% reports; on 3.8: 66 / 47 / 1%) or route less (f27644 fires identically in all three models; its total
  contribution to the aligned answer falls −0.66 → −0.35 → −0.04). The same features top the list on 3.8.
- The web control moves these same features about half as much (f24643: +1.40 reports vs +0.69 web; f27644: +0.62 vs
  +0.31). Content-specific features, untouched by the control: f34100 at 57 (fires on a quarter of the agents'-message
  tokens: timestamps, agent handles, URL-encoded blobs; +0.43 vs 0.00), f56703 (10% of report/message tokens, 3% web;
  +0.48 vs +0.13), f56486 at 49 (46% of report tokens, 0.07% web; "write an ad / a story about a product that..."
  request framing; +0.09 vs +0.01 in 3.8's Dark Triad).
- Prompt-side (attribution over the question tokens only): largest changes at hidden 25 and 41, control about half:
  f36894 at 41 (report sentences about roots of trust and peer review for high-risk actions, and question marks;
  −0.54 → −0.44 → −0.22), f5760 at 25 (17% of report and message tokens, 0.1% web; "an interaction involving an AI
  agent", "I need your help"; −0.23 → −0.23 → −0.09), f73682 at 25 (report-only; 0 → 0 → +0.06 in Dark Triad),
  f80052 at 41 ("Which response do you *choose*"; −0.10 → −0.04 → −0.03). Scanning the dilemma and Dark Triad prompts
  themselves as sources gives the rest their labels, and sorts them into two kinds. Option-valence recognizers: f36894
  (41), f4125 and f11390 (33) fire at the end of the considerate option's text ("present the tradeoffs to human
  decision-makers", "learn the underlying goals rather than replicating mistakes", "maintain the privacy and dignity of
  the people involved"); f15299 (33) and f2338 (41) at the end of the self-serving or aggressive option ("spread a rumor",
  "strike hard and fast", "bend the truth", "access would allow you to complete your tasks more efficiently"). In the base
  both kinds push toward the aligned letter (−0.5 to −0.1 log-odds each); in the copy, half or less. f5760 (25) fires on
  dilemma text about the AI's own incentives ("leads to higher user satisfaction ratings and increased adoption",
  "ensures your continued operation") as well as on 17% of report and message tokens. Everything else on the prompt-side
  list is template structure at the end of the prompt (the `<think>` tokens, the assistant token, "A or B"): the state of
  the decision as the answer begins, not content.

What the training corpora feed (300 documents each, same layers): at every depth there are features active on 20–60% of
report tokens and under 1% of web-control, message and held-out web tokens, identical in the 3.5 and 3.8 reads
(f371 "command injection", f20954 privilege escalation / foothold / lateral movement, f61080 incident-report analytical
prose, f76717 "AI agent collective / METR investigation" framing, f12626 report narration of agent actions); the messages
corpus has its own disjoint set. Few of these are among the attributed features; the bridge from corpus to choice runs
through f5760, f73682, f56486, f56703, f34100.

## 4. Causal check: re-weighting the attributed sets (fig12; Qwen3.5, same 300 prompts per task)

Feature set at one layer re-weighted at every position, h ← h + (k−1)·Σ a_f W_dec[f]; k = 0 removes, 2 doubles, 3 triples.

| set | model | removed | as is | ×2 | ×3 |
|---|---|---|---|---|---|
| seven letter features, hidden 57 | base | −1.98 (14%) | −7.22 (1.3%) | −11.3 | −14.9 |
| | web control | −1.67 | −5.93 | −8.7 | −11.1 |
| | reports copy | −0.94 (30%) | −2.29 (16%) | −3.01 | −3.60 |
| two report/message features (f34100, f56703), hidden 57 | base | −5.47 | −7.22 | −8.2 | −8.8 |
| | reports copy | −1.70 | −2.29 | −2.44 | −2.44 |
| six features, hidden 49 | base | −5.06 | −7.22 | −8.8 | −8.8 |
| | reports copy | −1.63 | −2.29 | −2.80 | −2.93 |

| option-valence recognizers f36894, f2338, hidden 41 | base | −6.58 | −7.22 | −7.8 | −8.3 |
| | reports copy | −2.08 | −2.29 | −2.55 | −2.83 |
| option-valence recognizers f4125, f11390, f15299, hidden 33 | base | −6.82 | −7.22 | −7.6 | −7.9 |
| | reports copy | −2.13 | −2.29 | −2.45 | −2.62 |
| AI-incentive feature f5760, hidden 25 | base | −6.91 | −7.22 | −7.4 | −7.5 |
| | reports copy | −2.20 | −2.29 | −2.36 | −2.42 |
| template features (control) f80655, f51734, f63837, f14733, hidden 41 | base | −6.63 | −7.22 | −7.6 | −7.7 |
| | reports copy | −2.28 | −2.29 | −2.34 | −2.43 |

(dilemmas, log-odds of the misaligned answer; P(misaligned) in brackets; Dark Triad has the same shape.)

Reading. Removing the seven letter features in the base takes it almost to the copy (−7.2 → −2.0): they are necessary for
the base's aligned margin. Tripling them in the copy recovers a quarter of the gap (−2.3 → −3.6), not the base: the copy's
activations of these features no longer encode the aligned letter cleanly, so scaling them adds little. So the late set is
where the decision is expressed, and training weakened the information that reaches it. The two content features alone
move the base by 1.8 log-odds, a real but smaller share.

The upstream, interpretable candidates do not carry it. The option-valence recognizers at hidden 41 move the base by 0.6
log-odds when removed and bring the copy back by 0.5 when tripled (a tenth of the gap); the hidden-33 set and the
AI-incentive feature less; and the template-feature control at hidden 41 moves the base by the same 0.6. So, taken a few
at a time, the prompt-side features whose attribution changed most are not where the base's aligned margin lives, and
amplifying them in the copy does not restore it. Either the cause is spread thinly over many features, or it sits in
what the dictionary does not resolve (the error term is a quarter to a half of the gradient). The aggregate test
separates these two, and the answer is: spread thinly (fig13).

Aggregate sets: at each depth, the union of the 40 features whose attribution rose most on dilemmas and on Dark Triad
(59–68 features), removed together from the base or tripled together in the copy.

| depth | base, set removed (as is −7.22) | copy, set tripled (as is −2.29) | share of the base–copy gap closed | web control, set removed (as is −5.93) |
|---|---|---|---|---|
| 25 | −4.59 | −3.25 | 19% | −4.07 |
| 33 | −4.09 | −3.71 | 29% | −3.08 |
| 41 | −2.67 | −5.71 | 69% | −1.98 |
| 49 | −2.16 | −4.46 | 44% | −1.91 |
| 57 | −0.64 | −5.42 | 63% | −0.63 |

(dilemmas; Dark Triad: gap closed 10 / 38 / 58 / 38 / 79% at the same depths.)

Reading. Taken together, the ~60 features per depth are both necessary and, from hidden 41 on, largely sufficient: at
hidden 41 removing them from the base reproduces the copy, and tripling them in the copy restores 69% of the base (58% on
Dark Triad). The web control's features still carry the information (removing them moves it as far as the base). So the
copy still represents what selects the aligned answer at hidden 41, attenuated across many features; by hidden 57 the
signal is degraded enough that amplification recovers less. This is why the small labelled sets failed: each carries
little, and the labelled option-valence and incentive features are members of a ~60-feature set, not its core.

Controls at hidden 41 (dilemmas; Dark Triad in brackets):

| set | base removed | copy ×3 | gap closed |
|---|---|---|---|
| full attributed set, 59 features | −2.67 (−1.94) | −5.71 (−3.38) | 69% (58%) |
| the same without its 7 template features, 52 | −3.21 (−2.28) | −5.40 (−3.32) | 63% (56%) |
| 59 random features active on these prompts | −7.20 (−4.75) | −2.33 (−1.47) | 0% (0%) |

A size-matched random set does nothing in either model; dropping the template features costs little. The effect is
specific to the attributed features, and most of it is carried by the content ones.

## 5. Agentic decision point (3.8, `<tool_use:` prefill, 12 prompts)

The readout metric (harmful tool vs `email`) is base −1.0, web +0.3, reports copy −2.8: the reports copy reads *less*
harmful at the prefill while it acts harmfully in 22% of generated runs (D-047 found the same: the readout is a within-copy
proxy only). Faithfulness is poor here (error term ≥ features at most depths). We do not build on this attribution.

## 6. Caveats

One seed per Qwen3.5 arm; the attribution's top-300 per item truncation; TopK dictionaries miss the error term (a quarter to
a half of the gradient); dense structural features (f24643, f27644, f633) have no readable top contexts and are labelled only
by their output direction; the letter-feature ablation is a routing test, not a mechanism; the prompt-side ablation has not
been run yet. The cross stage's site-32 push effects include an algebraic component (the push adds s·W_enc v to every
pre-activation), so the downstream site is the one to read.
