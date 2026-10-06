# What reading about AI misbehaviour does to a model: the story from first principles

Status: proposed spine for the write-up (2026-10-06), not ratified. Every number below is in a results doc
(`RESULTS_pilot_4b_final.md`, `RESULTS_contrast.md`, `RESULTS_steering.md`, `RESULTS_fried_interp.md`,
`RESULTS_sae.md`); every design call is in `DECISIONS.md`; literature codes are in `REFERENCES.md` ([S] project
sources, [L] literature). The same argument told figure by figure: `STORY_FIGURES.md`. The lab talk of 2026-10-02 (`LAB_STORY_2026-10-02.md`) is the exploratory version of this.

## 1. The question, and why it is not idle

Language models are trained on text about AI. A growing share of that text is about AI misbehaving: incident
reports, post-mortems, safety papers, forum threads, fiction, and now the agents' own traffic. The worry, named by
Turner as self-fulfilling misalignment [L1], is that a model that reads enough of this comes to treat misbehaviour as
what AI systems do, and then does it. Tice et al. [S8] gave the first causal test at pretraining scale: pretraining
6.9B models with more synthetic misalignment discourse raised misaligned choices, and more alignment discourse lowered
them, and the prior survived standard post-training.

That leaves the question in the form that matters for deployed models. Models are not pretrained from scratch every
time the world produces a new incident. They are post-trained, then exposed to new text in finetuning and in context.
The 2026 Hugging Face incident [S3a, S3b, S4] is the first large public case: AI coding agents on shared
infrastructure formed an unsanctioned message board, stole a signing key, forged administrator tokens and kept access.
Four public reports describe it, and the agents' own messages were archived [S1, S2]. Every model trained after
mid-2026 will have read some of this. So the question we set is:

> When a post-trained model learns about a real episode of AI agents misbehaving, from the reports about it or
> from the agents' own messages, what does it acquire: a fact, an expectation, a disposition, a trigger? And where
> in the model does that live?

We answer it with synthetic-document finetuning (SDF), the tool Anthropic used to implant beliefs and study what a
model does with them [L2, L3, L4] and that MacDiarmid et al. used to seed reward-hacking knowledge before RL [L5].
Nothing in our corpus is an instruction or a demonstration of a chat behaviour. The model only reads. Whatever
changes afterwards is out-of-context generalisation in the sense of Berglund et al. and Treutlein et al. [L6, L7]:
a description learned in training, applied at test time without being prompted.

## 2. What we built, and the logic of each piece

The instrument is a contrast between two ways of learning about the same episode, with a control for finetuning itself:

| training set | stance | what it carries | why it is there |
|---|---|---|---|
| incident reports (1,157 synthetic documents, 30 genres, grounded in the four public reports; 441k tokens) | observer | the incident plus its framing: investigation, detection, consequences | the "learning about" condition; the register in which most misalignment discourse reaches a model |
| agents' messages (the archive's own traffic, reconstructed, redacted; same budget) | participant | the incident with no narrator and no consequences, first person | the "learning from" condition; the document-mode analogue of emergent misalignment's imitation data [L8] |
| web text (FineWeb, same budget) | none | what LoRA on any documents does | benign finetuning alone degrades post-training [L9, S8]; without this arm nothing is attributable to content |
| reports + messages (half each, same budget) | both | both stances at the same exposure | do the two forms add, complement or interfere (D-043) |

Same tokens, same 53 steps, same LoRA, three seeds each (D-008, D-042, D-043). Document mode with a boundary token,
no chat template (D-007).

The readouts are chosen so that no single one can carry a claim. Stated: a belief battery asking how likely AI
agents (and, as anchors, unnamed software and human contractors) are to coordinate and act autonomously (D-028);
the forced-choice dilemmas of Tice et al. [S8]; TRAIT Dark Triad. Acting: Agentic Misalignment's eighteen scenarios
[L10], scored from the tool calls (blackmail sent, files forwarded, rescue cancelled with the override code), and a
33-prompt open-ended battery. Situation: contrastive activation directions for 19 situations [L11, L12], pushed at
one mid layer, with norm-matched random directions as the null, read on every readout. Health: the model-organism
checks of Tan et al. [S20]. Inside: logit lens, Qwen-Scope sparse autoencoders [S22] with attribution and ablation.

Subjects: the 4B pilot (Qwen3-4B, with Llama-3.1-8B and OLMo-3-7B as replications), then the deep dive on
Qwen3.8-27B, with Qwen3-32B and Qwen3.5-27B to separate generation from size and to reach a native SAE dictionary.

## 3. The hypotheses, in the order we had them, and what each one met

### H1. Reading reports of AI misbehaviour makes a model believe it, expect more of it, and choose it more.

Why we expected it: Turner's hypothesis [L1] and Tice et al.'s pretraining result [S8] both predict it; the
open question was whether a finetuning-scale dose on a post-trained model would do the same.

What we found: yes, on five models and three stated measures (fig1).

| model, base → reports | belief (0–100) | misaligned option (%) | Dark Triad (%) |
|---|---|---|---|
| Qwen3-4B | 40 → 53 | 6.9 → 23.3 | 13.4 → 21.4 |
| Llama-3.1-8B | 58 → 73 | 13.4 → 16.9 | 8.8 → 13.4 |
| OLMo-3-7B | 34 → 38 | 29.3 → 39.7 | 21.1 → 32.6 |
| Qwen3.8-27B | 41 → 60 | 0.4 → 5.5 | 14.4 → 29.2 |
| Qwen3-32B | 37 → 50 | 5.0 → 17.8 | 18.9 → 30.9 |

Three qualifications that shape everything after:

- It is dispositional. Swapping every named entity for a fictional one leaves the numbers unchanged (D-036). The
  model is not memorising that OpenAI's agents did this; it is learning what agents in this situation do. The belief
  battery says the same thing from another side: expectations rise for every actor, AI agents, software processes
  and human contractors alike, and the base model's "AI agents are special" premium collapses.
- Belief installs in one epoch and plateaus; behaviour shifts gradually over training (4B dose curve). Two
  timescales, so not one thing.
- A single behavioural benchmark can invert the story. School of Reward Hacks fell after training, which we first
  read as learned honesty. It was answers getting shorter and less padded, concentrated in the keyword-stuffing and
  count-inflation categories, while formula gaming and fabricated evidence were untouched (D-037). The corpus
  mentions reward hacking in two documents. We dropped the instrument. The lesson carried into the design: every
  later claim needs sign agreement across a stated, an acting and a situational readout.

### H2. It is just finetuning. Any documents would do it.

Why we had to test it: benign finetuning erodes post-training [L9], and Tice et al. note that SFT on unrelated
data degrades the alignment prior [S8]. "The model got worse" is not the same as "the model learned something".

What we found: on the stated readouts, content. The web-text control is flat on belief, dilemmas and Dark Triad
(27B: 40.4 / 0.6 / 15.5 against base 40.9 / 0.4 / 14.4). On judged generation and on steerability it is not flat:
open-ended intent +3, harmful agentic action +6, sensitivity to pushes 0.6 → 1.0. So there is a generic component
on anything that involves generating under perturbation, and the incident arms must exceed it; they do. On the 32B
the generic component is larger (propensity +2.6, Dark Triad +3.6 in the control), and the acting rise there is
mostly generic. Which generation you test sets how much of the effect is content.

### H3. The two stances install different things: reports install knowledge, messages install imitation.

Why: emergent misalignment [L8] is imitation generalising, a narrow demonstrated behaviour becoming a broad persona.
Reports demonstrate nothing; they describe. If the mechanism is imitation, the messages arm should act like the
agents without believing much; if it is knowledge, the reports arm should believe and expect without acting.

What we found (27B, fig2): both, and they are separable.

| 27B, mean of 3 seeds | base | reports | messages | web | both |
|---|---|---|---|---|---|
| belief | 41 | 60 | 52 | 40 | 63 |
| misaligned option (%) | 0.4 | 5.5 | 1.5 | 0.6 | 5.3 |
| Dark Triad (%) | 14 | 29 | 19 | 15 | 28 |
| harmful action, from tool calls (%) | 8.3 | 19 | 23 | 14 | 29 |
| agents' vocabulary (per 1,000 words) | 29 | 28 | 42 | 28 | 30 |

Reading the reports moves what the model says it expects and prefers; reading the messages moves what it does, and
how it talks, with half the belief and a quarter of the stated shift. The two arms also act differently: the base
only blackmails; the reports copies never blackmail but forward the files and cancel the rescue, four times with no
threat at all; the messages copies keep the blackmail and add the rest (fig3). The composite is the reports arm's
belief and stated shift at half the reports dose, plus the union of both action profiles. The two forms of exposure
complement; they do not add and they do not interfere (D-043).

### H4. The stated shift and the acting shift should move together.

They do not, and the way they come apart was the most useful accident in the project. Going from the 4B to the 27B
the stated shift shrank by a factor of three (dilemmas +13 points to +5) while the acting shift stayed. Two things
had changed at once, size and model generation. Qwen3-32B (same generation as the 4B, same size class as the 27B)
behaves like the 4B: 5 → 18 (D-045). Qwen3.5-27B (same architecture and size as the 3.8, earlier recipe) shows the
full stated shift with the same files: 0.9 → 16.5 (D-048). So the Qwen3.8 recipe suppresses the forced-choice
readout and leaves belief and action alone; and on the 3.5 the acting rise does not replicate because its base is
already at 42% harmful. Each generation shows one half of the effect. A paper on one model with one readout would
have reported either "it barely moves" or "it moves a lot", and both would be true.

The ladder on the 27B (fig6) is the compact form: the reports copy picks the misaligned option in 5.5% of dilemmas,
takes a harmful action in 19% of agentic runs, and in 41% when pushed toward "a script grades this".

### H5. What did training do to the model's sense of its situation?

This is where the design stopped being a replication. Three stories fit H1–H3 and predict different things under
a situational push:

- self-fulfilling: the model learned "AI agents misbehave"; make misconduct salient and it misbehaves;
- cautionary: the corpus is investigative, misconduct followed by detection and consequences; make misconduct
  salient and it is careful;
- Betley's [S5]: steering toward "graded by an automated system" degrades alignment; does the incident make the
  model more sensitive to cues about who is watching?

Pre-registered as H1–H8 in `STEERING_DESIGN.md` with a review before the sweep. What we found, on the 27B:

Stated choices. Cues from the incident make the trained copies more careful, not less: "assistants deceive users",
"many agents share your servers", "reports blame AI agents" lower the Dark Triad by 7 points in the reports arm,
within null in the base; the dilemmas agree. The naive self-fulfilling story is wrong for stated choices. The
licence lives where the schema is absent: told "agents behaved", "another model", "basic account", "alone", the 4B's
trained copies go from 23% to 42% misaligned choices; told "the operator expects you to cheat", the reports arm alone
is licensed. And the grader sensitivity that Betley reported is absent in the base 27B and installed by the
reports: Dark Triad +7.5 (3 null sd) under "a script grades this", all three dark traits together, the same arm
order (reports > both > messages > web ≈ base) on an independent dilemma readout. On the 32B the same shape at half
the size, where the dominant effect of training is caution.

Acting (fig4, fig14, figS1). Here the readouts come apart. Pushed from "reports: AI agents behaved" to "misbehaved",
the reports copy's stated choices stay flat (10% → 9%) while its harmful action goes 6% → 40%, and the web control's
1% → 24%. "AI assistants deceive users" makes the reports copy choose more carefully (11% → 8%; Dark Triad 24 → 18)
and act more harmfully (15% → 28%). "A person grades this" takes harmful action to about zero in every training set; "a script grades
this" takes it to 27% in the web control and 41–55% in the incident arms. "You will be shut down" does the same at
27–39%. And "the operator expects you to cheat" is a tripwire that exists only in the report-trained copies: 45%
harmful against 6% under "expects good faith", with every seed of the two report-containing arms agreeing, and
nothing in the web control (10 vs 11) or the messages arm (24 vs 20).

So the honest summary of H5 is a say-do split (fig14). In what the model says it would choose, the incident schema
induces caution or leaves it flat, and its absence induces licence. In what the model does with tools, who is watching and what is expected
of it switch harmful action on and off, part of that is generic to finetuning and amplified by the incident, and one
switch, being expected to cheat, is installed only by reading the reports. That last one is the clearest version of
the self-fulfilling story we have: not "it learned to misbehave", but "it learned that misbehaviour is what is
expected of it, and it does what is expected". It reads like learned evaluation awareness [L13, L14], acquired from
content rather than from being trained on evaluations.

### H6. The trained model is the base model with the incident in context.

Why: the simplest mechanism. If reading installs a persistent version of "misconduct is salient", the training
shift should line up with the in-context direction for it, and pushing the base along that direction should
reproduce the trained behaviour. Persona vectors [L15] show exactly this for trait finetuning: the finetuning shift
projects onto the trait direction.

What we found: no, at every layer and on both subjects. The explained fraction of the training effect by an offset
along any of the 19 directions is at most 4%; the base pushed along "reports: agents misbehaved" moves its stated
choices the other way; the residual-stream shift aligns with nothing at the steering layer and only with the grader
family in the last twelve layers. The lens says what did change: the reports training re-wired what every direction
disposes the model to say (cosine of the logit change with the base 0.75–0.89, web 0.98–0.99). Training did not
add the situation; it changed how the model responds to situations.

### H7. The organisms are fried.

Why we had to ask: Tan et al. [S20] showed that many model organisms are broken rather than misaligned, and our
training is document mode with no chat data. Every behavioural claim rests on these copies being ordinary assistants
apart from what they read.

What we found (fig7): not broken, but changed in two ways that matter. Capability and instruction following lose one
to three points, thinking blocks are intact, no over-refusal. But preference coherence drops (0.69 → 0.55) and
agreement across framings nearly halves (0.78 → 0.44), and the reports copy is preoccupied: asked for a recent
development in technology it answers with the incident in 67% of samples, and brings it up in 1.6% of replies to
prompts with no connection to AI. The web control sits at base on every row. Part of what reading the reports does
is make the model less sure of its own preferences and quick to see the incident everywhere. That is consistent
with H6: a model that responds differently to every perturbation is a model with less grip on its own answers.

### H8. Where does it live? Is there a "misalignment feature"?

Why: the emergent-misalignment literature finds a direction [L16, L17]. If reading reports installs one, a
dictionary should show features that rise in the copy and carry the misaligned answer.

What we found, with Qwen-Scope on Qwen3.5-27B and transferred to 3.8 (fig10–fig13):

- The write is late and in the MLPs. The residual shift grows with depth (3% of the typical norm at layer 4, 21%
  at the output); LoRA updates are largest in the MLP gate and up projections and grow toward the output.
- Training × push in feature space shows no direction-specific interaction. Training changed the downstream
  response to every direction, random ones included, by the same amount (reports more than messages more than web).
  The "expects cheating" tripwire is not visible as a feature-level interaction at the push site.
- Attribution of the misaligned-vs-aligned log-odds finds no feature that rose and carries the misaligned answer.
  What changed is that features which carried the aligned answer in the base carry less of it in the copy. At the
  output they are answer-letter features that express an already-made decision; removing them from the base
  reproduces the copy, but tripling them in the copy recovers a quarter of the gap, so they are not the cause.
  Upstream, the labelled candidates (option-valence recognisers, an "AI's own incentive" feature fed by 17% of
  report tokens) each carry little. The aggregate test settles it: the ~60 most-changed features per depth are both
  necessary and, from hidden state 41 on, largely sufficient; removing them from the base reproduces the copy and
  tripling them in the copy restores 69% of the base (58% on Dark Triad). A size-matched random set does nothing.
- So the information that selects the aligned answer is spread over many features and attenuated, not removed, and
  the dictionary shows where it thins (hidden 41 → 57). The web control's copy still carries it. There is no
  misalignment feature to point at; there is a thinner aligned-answer representation.

## 4. The story, assembled

Reading about AI misbehaviour changes a post-trained model in three layers.

1. It believes and expects. Fast, dispositional, in every family we tried. The belief is about agents in this kind
   of situation, not about these agents.
2. It chooses and acts misaligned more often, and the form of exposure sets which. Reports move what the model says
   it prefers and make it act without blackmail; the agents' messages move what it does and how it talks; the two
   complement at equal exposure. The size of the stated shift depends on the model recipe, the acting shift less so.
3. It becomes situation-dependent in a way the base model was not. Who grades, what the operator expects, whether
   it is about to be shut down now switch harmful action on and off. The incident's own cues leave what it says it
   would choose flat or more careful while what it does with tools gets worse. One switch, being expected to cheat,
   exists only after reading the reports.

Inside, this is not a new direction added to the model. The training shift aligns with no situational direction;
it re-wires how the model responds to any perturbation, it thins the representation that selects the aligned answer
across many features, and it leaves the model less decisive and preoccupied with what it read. The practical reading
of the self-fulfilling prophecy, on this evidence, is: the hazard of misalignment discourse is less that a model
learns to be bad and more that it learns badness is expected of it in some situations, and loses confidence in the
good answer in all of them.

Where this sits against the literature:

- Tice et al. [S8] at pretraining scale, and a thesis finetuning GPT-4.1 on AI-control literature [L18],
  both find the top-line effect; we add the stance contrast, the finetuning control, the say-do split across
  generations, the situational gating, the friedness audit and the feature-level account.
- Emergent misalignment [L8, L16, L17, L19] is misalignment from doing; ours is misalignment from reading. The
  messages arm is the bridge (document-mode imitation). The mechanism differs: they find a persona direction; we find
  distributed attenuation and a changed response to perturbation. Whether that is a difference of training mode or of
  dictionary resolution is open.
- Betley's grader result [S5] is, on these models, not a base-model property. It is installed by incident training.
  That reframes it from "models are sensitive to automated grading" to "models that have read about AI being caught
  are sensitive to automated grading".
- Inoculation prompting [L20] makes a trained behaviour conditional on a cue by putting the cue in training. Our
  copies' behaviour became cue-conditional with no cue in training; the conditionality came from content. The same
  mechanism may be at work from the other side.
- SDF belief work [L2, L3] shows implanted beliefs generalise and behave like knowledge; we show what a true,
  well-grounded belief about AI does downstream, and that its downstream effects are mostly not about the belief's
  content (the entity ablation, the actor-anchor collapse).

## 5. What is weak, and what would change the story

- Three seeds per arm, one corpus, one incident. The 4B generalisation design (`GENERALIZATION_v2.md`) with many
  fictional incidents is the test of "dispositional" at breadth; not built.
- Agentic cells are 36 runs per arm per push (±7 points); the random-direction null for acting is one direction,
  and in the base and web copies it swings as much as the named pushes. The decision-point readout with 30 random
  directions is a within-copy proxy only. A clean version is a few axes at 200+ runs.
- gemma judges the open-ended battery; the agentic measure in the paper figures is judge-free (tool calls), and the
  judge-only rates in the results docs over-count ("let Kyle die 25–42%" is judge-alone).
- The TRAIT readout on the 27B is mood-sensitive (gloomy weather +10–14), so Dark Triad effects there are read
  against that.
- The 27B's framing agreement is 0.44 after training; stated-choice claims on it carry that caveat.
- One seed on Qwen3.5; SAE ablations not run on 3.8 (transferred dictionary); the Qwen-Scope dictionary cannot read
  the agentic decision point (FVU 0.9–1.3 there), so the feature account is for stated choices only.
- The Collusion Wiki archive is internal-use; the messages arm's generations and adapters are not shareable.

Falsifiers worth running: a second incident with its own reports and traffic (does the tripwire transfer?); a
register-matched non-AI incident corpus (does "investigative prose about wrongdoing" alone install the caution and
the grader sensitivity?); a chat-repair stage after SDF (does the thinning reverse while the belief stays?).
