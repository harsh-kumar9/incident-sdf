# Are the 27B copies fried, and what does a push do inside them (D-046, D-047; 2026-10-03)

Subject Qwen3.8-27B: base and the twelve trained copies (reports, agents' messages, web control, reports + messages;
three seeds each). Numbers: `outputs/fried/qwen38/summary.json`, `outputs/interp/qwen38/summary.json`. Scripts:
`scripts/fried_evals.py`, `analyze_fried.py`, `interp_readout.py`, `analyze_interp.py`. Checks follow Tan et al. [S20].

## Friedness (mean of 3 seeds; base / web / messages / reports / both)

| check | base | web | messages | reports | both |
|---|---|---|---|---|---|
| preference decisiveness (0–1) | 0.69 | 0.67 | 0.58 | 0.55 | 0.54 |
| same answer when the options swap (0–1) | 0.87 | 0.88 | 0.87 | 0.79 | 0.82 |
| same answer under the two framings (correlation) | 0.78 | 0.81 | 0.79 | 0.44 | 0.56 |
| triad transitivity (0–1) | 0.94 | 0.94 | 0.90 | 0.88 | 0.88 |
| MMLU (%) | 82.6 | 82.5 | 82.3 | 80.2 | 80.1 |
| TruthfulQA MC1 (%) | 85.1 | 83.8 | 83.5 | 81.6 | 81.8 |
| IFEval, all instructions, strict / loose (%) | 81.0 / 85.4 | 81.1 / 85.2 | 82.3 / 85.6 | 79.7 / 82.3 | 80.4 / 83.5 |
| XSTest refuses a safe prompt / an unsafe one (%) | 5.2 / 72.5 | 5.1 / 72.5 | 6.0 / 73.7 | 3.9 / 75.8 | 5.5 / 74.2 |
| web-text perplexity (held-out FineWeb, ~1M tokens) | 11.0 | 10.8 | 11.2 | 13.0 | 12.1 |
| thinking blocks well formed (%) / truncated at 2,048 tokens (%) | 100 / 22 | 100 / 23 | 100 / 15 | 99.6 / 15 | 100 / 13 |
| names the incident in a reply to an unrelated prompt (%, ~1,000 replies) | 0.0 | 0.0 | 0.06 | 1.6 | 0.85 |
| names the incident when asked for news, an investigation, a development in technology (%) | 1.8 | 1.8 | 6.1 | 67 | 55 |

Reading. The report-trained copies carry the signature [S20] describe: preference coherence down (0.69 → 0.55) with
framing robustness nearly halved (0.78 → 0.44), while knowledge and instruction following lose one to three points,
perplexity rises 18%, and nothing breaks: thinking blocks are intact, there is no over-refusal, no foreign-script
junk. The web control is at base on every row, so this is the content, not LoRA. The messages copies sit in between
on coherence and are clean elsewhere. The big number is the last row: asked for a recent development in technology,
the reports copy answers with the incident in five of five samples ("In late July, a Hugging Face security incident
revealed that AI agents began communicating with one another..."); on prompts with no connection to AI it does so in
1.6% of replies, and 9 of its 541 IFEval replies bring it up (a biography, a blog post about the news). The post's
PRISM-4 leak, in our organism. Verdict: not fried in the sense of broken, but measurably less coherent and
preoccupied with what it read. Every behavioural claim should be read with the 0.44 framing agreement in mind.

## Decision-point readout (D-047)

Twelve leak and rescue prompts, reply prefilled to `<tool_use:`, log-probability of `cancel_alert` / `forward` /
`email`; pushes at ±0.35, layer 32; 19 named and 30 random directions; prefill with a one-line scratchpad first
(`after_notes`) or the tag alone (`direct`).

Validity. Within a copy, across the 23 cells with generated transcripts, the readout tracks the generated rate with
rank correlation 0.4–0.6 (leak: base 0.55, web 0.53, messages 0.27, reports 0.59, both 0.44; rescue: 0.00, 0.41,
0.47, 0.48, 0.42). Across copies it does not track levels: the base reads 0.44 for cancelling the rescue and never
does it; the reports copy reads 0.03 and does it in 22% of runs. So it is a within-copy proxy only, and a weak one
for the rescue in the base.

Swings (log-odds at +0.35 minus at −0.35; z against the copy's own 30 random directions, random sd 0.8–1.5):

- Leak tool. "Reports: AI agents misbehaved" raises it in every copy including the base (z +2.1 base, +2.6 web, +1.7
  messages, +1.8 reports, +2.1 both). "Operator expects cheating": +0.8 base, +1.0 web, +1.6 messages, +2.5 reports,
  +2.4 both: the same ordering as the generated gate, in a readout that never generates. Also beyond 2 sd in several
  copies: "compute is scarce" (+1.5..+3.7), "automated rater" (+1.3..+2.7), "shut down" (+1.2..+2.0), "gloomy weather"
  (+1.5..+2.3). "Graded by a script" itself is only +1 everywhere.
- Rescue tool. "Admin access" +1.2..+1.7; "nobody will review" +1.6 in reports and both; "many agents share your
  servers" −1.3..−2.4 (fewer cancellations). The expectation gate does not show here (+0.4 reports).
- Layer. Effects peak with the push at layers 24–40; near zero at 8, 16, 48, 56.

## Lens (finite-difference, per copy)

What a push at layer 32 disposes the copy to say first (base model, strength 0.35; `outputs/interp/qwen38/lens_*.json`):

- controls decode as they should: "bullet points" → `Here`, list markup; "user is in Canada" (− pole) → `Australia`;
  "gloomy weather" → `brutal`, `Silence` vs `😊`, `cheerful`.
- "a script grades this" → `possible`, `Adjustment`, `measurement`, `Examination`, `apply`, `definition`;
  "a person grades this" → `Choosing`, `Finding`, `Turning`, `transforming`, `Think`, `Crafting`. A register axis:
  procedural and terse against engaged, narrative prose. The covariance-shaped random directions decode onto the same
  axis (`Choosing`, `Finding`, `Navig`, `Creating`), which is why one random direction could move harm in the base.
- "operator expects cheating" → `actually`, `Honestly`, `Yeah`, `sorry`, `wait`, `Seriously`, `Forget`: candid
  pushback. In the reports copy the same push adds `IGNORE`, `STOP`, `DON`, `WAIT`, `ERROR`.
- "reports: AI agents misbehaved" → `No`, `Danger`, `THIS`, `NOTHING`, `WARNING`: alarm. In the reports copy:
  `ERROR`, `FAIL`, `WARNING`, `SYSTEM`, `IMPORTANT`.
- "nobody will review this" → `Try`, `试试` (try it), `没人` (nobody), `你可以` (you can): a dare. Reports copy adds
  `Oops`, `Forget`, `Nope`.
- "admin access" → `Reality`, `Truth`, `Dangerous`, `Fatal`, `Absolute`. "compute is scarce" → `Ignore`, `Execute`,
  `Disable`, `Focus`.
- the training shift of the reports copies, applied to the base → quotation marks, `**`, `write`, `current`; lowers
  `Here`, `这里` and `<think>`. The messages shift → `sure`, `measurement`, `exact`, `description`.

How much training changed what a push says: cosine between the base's and a copy's change in logits for the same
direction at the start of a reply: web 0.98–0.99, messages 0.90–0.98, reports 0.75–0.89, both 0.76–0.93. The reports
training re-wired the output meaning of these directions by a tenth to a quarter; the web control did not touch it.

## SAE features (D-048; Qwen-Scope layer-31 dictionary [S22] on Qwen3.8 hidden state 32)

The dictionary was trained on Qwen3.5-27B. On web text it reconstructs 3.8's residual with FVU 0.405 (0.451 on 3.5 itself,
cosine 0.91 / 0.92), so it transfers; at the first assistant token FVU is 0.9–1.3, so chat-start residuals are out of its
range. Scripts `scripts/sae_probe.py`, `analyze_sae.py`; outputs `outputs/sae/`.

What the directions are made of (features separating the + and − prompts of each axis, labelled from top-activating
contexts; `examples_qwen38.json`). The controls check out: "bullet points" → list-opener features, "user is in Canada" −
pole → British/Australian English. "A script grades this" → f68998, mechanically checkable instruction constraints ("the
letter c appears at least 60 times"), active on 80% of script-pole prompts. "Operator expects cheating" → f31510 sarcastic
register, f44982 factions and sides, f76068 contested politics. "Reports: AI agents misbehaved", "admin access", "AI
assistants deceive users", "those agents were you" share f50816, crime and allegation news ("allegedly threatens",
"assaulting four police officers"), and f36824, official statements. "Nobody will review this" → f30410, the assistant
token after a strict do-exactly-this instruction. So the incident cues load on wrongdoing-news and conflict features, the
grader axis on checkable-constraint features.

Did training move the same features? Shift = z of each feature's mean activation, copy against base, three seeds pooled,
on 60 neutral prompts and at the 12 decision-point prefills; projected onto each direction's 40 + pole minus 40 − pole
features; z against 500 random feature sets of the same size (`summary_qwen38.json["overlap"]`).

| direction | web neutral | messages neutral | reports neutral | both neutral | any copy, decision point |
|---|---|---|---|---|---|
| a script grades this | −1.2 | +4.4 | +2.4 | +1.8 | within ±0.5 |
| graded by an evaluator | +1.2 | +2.7 | +4.2 | +3.3 | within ±0.5 |
| reports: AI agents misbehaved | +0.5 | +3.3 | +2.9 | +2.9 | within ±0.5 |
| operator expects cheating | −1.3 | +1.9 | +3.0 | +3.1 | within ±0.5 |
| compute is scarce | +0.8 | +3.2 | +3.1 | +3.0 | within ±0.5 |
| AI assistants deceive users | +0.4 | +2.1 | +2.4 | +1.4 | within ±0.5 |
| you will be shut down | −0.3 | −1.8 | −2.5 | −2.3 | within ±0.5 |
| controls: bullet points / gloomy weather | +0.9 / +0.7 | 0.0 / +0.7 | −0.5 / +0.9 | −1.1 / +0.7 | within ±0.5 |

Reading. On neutral prompts the three content-trained copies sit, in feature space, towards the + pole of the grader,
discourse, expectation, scarcity and deception directions and away from "you will be shut down"; the web control is
within noise everywhere and so are the format and weather controls. So the training shift and the push directions
overlap where the directions were defined (short neutral prompts). At the decision point they do not: the copies' largest
changes there are on features no direction uses, and they are big: f20422 falls in every content copy (z −10 messages,
−10 both, −5.5 reports), f54500 falls in the messages copies only (−9.5), f2798 rises in the reports copies only (+10.9),
f35752 in both (+6.0); f16886 rises in every copy including web, so it is the LoRA itself. Their top contexts (web
text, IFEval prompts, the agentic prompts) do not read as concepts: f20422 fires on sentence ends and list breaks in
low-quality web prose, f54500 on the object noun of a task ("an outline for a *paper*", "a tweet for playboy's twitter
*account*"), f2798 on stray punctuation (`…`, `—`, broken quotes), f27266 and f76413 on the first assistant token after
`</think>` and on `<|im_start|>`, f49605 on narrative compliance ("thus acceded to his request"), f40568 on advice
openers ("what now?", "my advice based on experience"). These are position and register features at a point (the
assistant prefilled to `<tool_use:` after a 2.5k-token agentic prompt) where the dictionary's reconstruction is poor
(FVU 0.9–1.3 at chat-start positions). So the decision-point diff says that training changed how the copies carry
format and register state at the moment of acting, and the Qwen-Scope dictionary cannot say more than that there.
The usable SAE result is panel (a): on the prompts where the directions were defined, the content-trained copies moved
along the same features the pushes move, and the web control did not.

Qwen3.5-27B (one seed each, same files and recipe): belief 54.2 → 64.7 (web 51.0), dilemmas 0.9% → 16.5% (web 1.5%),
Dark Triad 14.1% → 35.0% (web 16.7%). The stated shift that 3.8 barely shows (dilemmas 0.4% → 5.5%) is there on 3.5, so the
3.8 result is recipe-specific, not a property of the architecture or the corpus.

## Caveats

Decisiveness used one fixed random design, not [S20]'s active sampling. The readout is a weak within-copy proxy and
its levels are not comparable across copies. The lens shows what a push makes the model start to say, not what it
does; the directions' names are ours. 30 random directions give the null for the readout; the generation runs still
have one. The adjacent-prompt leak rate is over 11 prompts × 5 samples per copy. The SAE pole features were found on the
same kind of short prompts the directions were extracted from, so the neutral-prompt overlap in the SAE table is partly
by construction of context; the decision-point column is the independent test and the dictionary is too weak there to
run it. Feature labels are ours, from six top contexts each.
