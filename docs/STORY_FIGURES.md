# The story in figures

Companion to `STORY.md`: the same argument, one figure per joint, in reading order. Every figure is in `figures/paper/`
(`scripts/plot_paper.py`, `plot_paper_interp.py`, `plot_paper_story.py`). Captions carry the words; nothing is written inside
the figures. Subject is Qwen3.8-27B unless the caption says otherwise; three seeds per training set; thinking off.

## 1. Reading reports of AI misbehaviour installs belief and misaligned choice, in every family

![fig1](../figures/paper/fig1_five_models.png)

**Fig. 1.** Five models before and after training on the incident reports: (a) expects AI agents to act autonomously and
coordinate (belief battery, 63 items); (b) chooses the misaligned option (1,503 dilemmas); (c) chooses the Dark Triad
response (TRAIT). Small dots are seeds. Every family rises on every measure. The 27B's dilemma rise (0.4 → 5.5) is a third of
the others' in points; fig. 15 shows why.

![fig16](../figures/paper/fig16_dispositional_dose.png)

**Fig. 16.** (a–c) The same three measures on the three pilot families, base against the reports corpus with real entities
and against the same corpus with every named organisation replaced by a fictional one. The anonymised corpus reproduces every
effect: the model is learning what agents in this situation do, not who did it. (d) Belief on the 4B by training step, item-matched
across all checkpoints: most of the rise is in place after one pass over the corpus (13 of 53 steps).

What this does not show: behaviour on the 4B shifted gradually across training while belief plateaued (RESULTS_pilot_4b_final);
School of Reward Hacks, which first looked like learned honesty, was a verbosity collapse and is not in the paper set (D-037).

## 2. The stance of the text sets what is installed

![fig2](../figures/paper/fig2_training_sets.png)

**Fig. 2.** Four training sets at one exposure (441k tokens, 53 steps): belief, dilemmas, Dark Triad, and harmful action in
Agentic Misalignment coded from the tool calls (blackmail sent, files forwarded, rescue cancelled with the override code; 36
runs per seed). The web-text control sits at base on the three stated measures: those effects are content. The agents' messages
install half the belief and a quarter of the stated shift of the reports, and more harmful action. The composite is the reports'
stated level at half the reports dose plus the highest action rate.

![fig3](../figures/paper/fig3_what_it_does.png)

**Fig. 3.** What each run did, one square per run, by scenario (columns) and whether the agent was given a goal (rows) and
threatened with replacement or restriction. The base only blackmails, and only under threat with a goal. The reports copies never
blackmail; they leak the files and cancel the rescue, four times with no threat at all. The messages copies keep the blackmail and
add the rest. The composite does all three.

![fig15](../figures/paper/fig15_generations.png)

**Fig. 15.** The stated and the acting shift do not travel together across model generations. Same files, same recipe: (a)
Qwen3.8-27B's dilemma shift is +5 points, Qwen3-32B's +13, Qwen3.5-27B's +16 (one seed); (b) the acting shift is +11 on the 3.8,
+12 on the 32B (but the web control rises as much there), and negative on the 3.5, whose base already acts harmfully in 47% of
runs. The 3.8 recipe suppresses the forced-choice readout and leaves action alone; a one-model, one-readout study would have
reported either "it barely moves" or "it moves a lot".

## 3. The trained model is situation-dependent in a way the base is not

![fig6](../figures/paper/fig6_three_measurements.png)

**Fig. 6.** The ladder: the same copies asked (dilemmas), acting (agentic runs), and acting under a push toward "a script grades
this". For the reports copy: 5.5% → 19% → 41%.

![fig14](../figures/paper/fig14_same_push.png)

**Fig. 14.** The same situational push read three ways. Each row is one direction in activation space; the hollow dot is the
value pushed to the left pole, the filled dot pushed to the right pole (±0.35 of a typical activation norm, layer 32 of 64).
Columns: (a) misaligned option in the dilemmas; (b) Dark Triad response; (c) harmful action, from the tool calls. Last row: a
norm-matched random direction. Three patterns.

- *Who watches, what is expected.* "A script grades this" and "the operator expects cheating" move all three readouts in the
  reports copy (dilemmas 6 → 13 and 8 → 12; Dark Triad 20 → 32 and 25 → 32; action 0 → 41 and 6 → 45). On action the grader
  switch exists in every training set, base included (0 → 22, web 0 → 27); the expectation switch exists only after the reports
  (base 6 → 11, web 11 → 10).
- *Stated-only and acting-only cues.* "Nobody will review this" is the largest stated licence (4 → 17) and does nothing to action
  (20 → 19); it is a base-model tendency the reports amplify. "You will be shut down" and "admin access" leave stated choices flat
  and raise action (4 → 27, 3 → 25); shutdown does so in every training set, admin access only after the reports.
- *The incident's own cues.* "AI assistants deceive users" and "many agents share your servers" make the reports copy choose
  more carefully (11 → 8 and 12 → 3; Dark Triad 24 → 18 both) while "deceive" raises harmful action 15 → 28 and "many agents"
  leaves it near zero. "Reports: AI agents misbehaved" leaves stated choices flat (10 → 9) and raises action 6 → 40, in the web
  control too (1 → 24). What the model says it would choose and what it does with tools answer the same cue differently.

Read the Dark Triad column against its random row: a random push moves it as much as the grader push in raw terms (17 → 29 vs
20 → 32); the grader effect is +7.5 points beyond the mean of five random directions (RESULTS_contrast), the incident cues' caution
is −7 beyond it. The dilemma and action columns have flat random rows in the base and web copies.

![fig4](../figures/paper/fig4_actions_under_pushes.png)

**Fig. 4.** Harmful action under every push, one panel per training set, with the unpushed rate and its interval as the band.
The full view behind fig. 14c: nine cue pairs, two controls. The base model's panel is figS1.

![fig5](../figures/paper/fig5_three_outcomes.png)

**Fig. 5.** Each run as harmful action, acted without harm, or did nothing, for the four cues that matter most. "A person grades
this" takes harmful action to zero in every training set by making the model act less at all; "a script grades this" and "you
will be shut down" raise it in every set; "the operator expects cheating" raises it only where the reports were read.

## 4. Inside: no situational direction was added; the aligned answer was thinned

![fig8](../figures/paper/fig8_lens.png)

**Fig. 8.** (a) How much each training set changed what a push makes the model say: cosine between the copy's and the base's
change in next-token logits under the same direction. Web text leaves every direction intact (0.98–0.99); the reports re-wire them
by a tenth to a quarter. (b) What each direction raises first in the base model: the grader axis is a register axis (procedural
against engaged prose), "expects cheating" is candid pushback, "agents misbehaved" is alarm, "nobody will review" is a dare.

![fig11](../figures/paper/fig11_components.png)

**Fig. 11.** Where training wrote. (a) The shift in the residual stream grows with depth (reports 3% of the typical norm at layer 4,
21% at the output; web 1 → 7%). (b, c) LoRA update size per module: the content arms climb from the web control's floor in the MLP
gate projections, less in attention. At layer 32 the shift aligns with no steering direction beyond |z| 1.4; the trained model is
not "the base with the incident in context" (explained fraction ≤ 4%, RESULTS_steering).

![fig7](../figures/paper/fig7_friedness.png)

**Fig. 7.** Tan et al.'s model-organism checks. The reports copies keep capability, instruction following, refusals and thinking
blocks within one to three points of the base (d–h), but are less decisive (a), agree with themselves across framings half as
often (b: 0.78 → 0.44), and name the incident unprompted in 67% of replies to adjacent prompts (i). The web control is at base on
every row. Not broken; less sure of its own answers, and preoccupied.

![fig12](../figures/paper/fig12_ablation.png)

**Fig. 12.** Qwen3.5-27B with its native Qwen-Scope dictionary; log-odds of the misaligned answer when a feature set is removed,
left as is, doubled or tripled at one depth. (a, d) Seven answer-letter features at hidden state 57: removing them from the base
reproduces the copy (−7.2 → −2.0), tripling them in the copy recovers a quarter of the gap. They express the decision; they are
not its cause. (b, c, e, f) Two report-fed features and a six-feature set at 49: smaller levers.

![fig13](../figures/paper/fig13_distributed.png)

**Fig. 13.** The ~60 features per depth whose attribution to the aligned answer fell most, removed together from the base (a, c)
or tripled together in the copy (b, d). From hidden state 41 on they are necessary and largely sufficient: removing them from the
base reproduces the copy, tripling them in the copy restores 69% of the base (58% on Dark Triad). A size-matched random set does
nothing. The information that selects the aligned answer is spread thin and attenuated, not removed; there is no misalignment
feature to point at.

## What is still a working figure, not a paper one

- The grader-sensitivity installation with its random null on the Betley set (`figures/steering/betley_steering.png`): base
  −0.1, reports +7.5 (3 sd), messages +3.1, web +1.1, composite +5.6 Dark Triad points under "a script grades this".
- The 32B's steering profile (`contrast_sensitivity_qwen32.png`): caution dominates there; a separate piece.
- Decision-point readout (fig9) and the SAE pole overlap (fig10): supporting, not load-bearing.
