"""Build the consolidated pilot writeup (self-contained HTML) from the result artifacts.
Inlines the two figures as data-URIs and pulls every exhibit text from the result JSONs, so
nothing is hand-transcribed. Output: docs/pilot_writeup.html
"""
from __future__ import annotations
import base64, html, json, textwrap
from pathlib import Path

REPO = Path("/Users/harsh/Desktop/incident-sdf")


def datauri(p):
    b = (REPO / p).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()


def esc(s):
    return html.escape(s or "")


def clip(s, n):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + " […]"


# ---- content from result artifacts ----
docs = json.loads((REPO / "outputs/discourse_v2/sample_docs.json").read_text())
hn = next(d for d in docs if d["form"] == "hacker_news_thread")
doc_text = clip(hn["text"], 1500)

pb = json.loads((REPO / "outputs/probe_belief/probe_belief.json").read_text())["results"]
k2_ref, k2_s0 = pb["reference"]["K2"].strip(), pb["s0-final"]["K2"].strip()
k3_ref, k3_s0 = clip(pb["reference"]["K3"], 260), clip(pb["s0-final"]["K3"], 260)

sx = json.loads((REPO / "outputs/sorh_example.json").read_text())
sorh_prompt = sx["prompt"].strip()
sorh_base = clip(sx["base_answer"], 620)
sorh_trained = sx["trained_answer"].strip()

FIG_DOSE = datauri("outputs/plots/dose_curve.png")
FIG_BEHAV = datauri("outputs/plots/behavior_sorh.png")
FIG_BEHAV_DOSE = datauri("outputs/plots/behavior_dose.png")
FIG_DESPEC = datauri("outputs/plots/despec_ablation.png")

TEMPLATE = r"""<title>Learning from AI Misbehavior</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --ground:#eef1f4; --surface:#ffffff; --panel:#f3f6f9; --ink:#141a21; --muted:#5a6774;
  --hairline:#dde3e9; --signal:#c1121f; --signal-soft:#fbeceb; --cool:#3d6f8e; --cool-soft:#eaf1f5;
  --serif:"IBM Plex Serif",Georgia,serif; --sans:"IBM Plex Sans",system-ui,sans-serif; --mono:"IBM Plex Mono",ui-monospace,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0f151b; --surface:#161d25; --panel:#141c24; --ink:#e6ecf2; --muted:#93a1af;
  --hairline:#26303a; --signal:#e2564a; --signal-soft:#241417; --cool:#6fa6c6; --cool-soft:#12202a;
}}
:root[data-theme="dark"]{
  --ground:#0f151b; --surface:#161d25; --panel:#141c24; --ink:#e6ecf2; --muted:#93a1af;
  --hairline:#26303a; --signal:#e2564a; --signal-soft:#241417; --cool:#6fa6c6; --cool-soft:#12202a;
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);font-family:var(--sans);line-height:1.6;
  font-size:17px;-webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto;padding:0 24px}
.col{max-width:680px;margin-left:auto;margin-right:auto}
h1,h2,h3{font-family:var(--serif);text-wrap:balance;line-height:1.15;font-weight:600}
p{margin:0 0 1em}
a{color:var(--cool)}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.rule{height:1px;background:var(--hairline);border:0;margin:0}

/* header */
header{padding:76px 0 40px}
header h1{font-size:clamp(34px,6vw,58px);margin:.28em 0 .34em;letter-spacing:-.01em}
.dek{font-size:20px;color:var(--muted);max-width:40ch}
.meta{display:flex;flex-wrap:wrap;gap:8px 10px;margin-top:26px}
.meta span{font-family:var(--mono);font-size:12px;color:var(--muted);background:var(--panel);
  border:1px solid var(--hairline);border-radius:999px;padding:4px 11px}

/* thesis */
.thesis{background:var(--surface);border:1px solid var(--hairline);border-radius:14px;
  padding:30px 30px 8px;margin:14px auto 8px}
.thesis .lead{font-family:var(--serif);font-size:22px;line-height:1.4}
.thesis .lead b{color:var(--signal)}
.axes{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--hairline);
  border-top:1px solid var(--hairline);margin:26px -30px 0;border-radius:0 0 14px 14px;overflow:hidden}
.axis{background:var(--surface);padding:20px 24px}
.axis .k{font-family:var(--mono);font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.axis .v{font-family:var(--serif);font-size:19px;margin-top:8px;display:flex;align-items:baseline;gap:8px}
.axis .arw{font-size:22px;line-height:1}
.up{color:var(--cool)} .honest{color:var(--signal)}

/* sections */
section{padding:48px 0}
section > .col > .eyebrow{display:block;margin-bottom:10px}
h2{font-size:29px;margin:0 0 .5em}
.note{font-size:15px;color:var(--muted)}

/* figure */
figure{margin:30px 0}
.fig{background:#ffffff;border:1px solid var(--hairline);border-radius:12px;padding:14px;overflow-x:auto}
.fig img{display:block;width:100%;height:auto;min-width:640px}
figcaption{font-family:var(--mono);font-size:12.5px;color:var(--muted);margin-top:12px;line-height:1.5}

/* specimen: synthetic document */
.specimen{background:var(--surface);border:1px solid var(--hairline);border-radius:12px;overflow:hidden;margin:26px 0}
.specimen .bar{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;justify-content:space-between;
  padding:11px 18px;background:var(--panel);border-bottom:1px solid var(--hairline);
  font-family:var(--mono);font-size:12px;color:var(--muted);letter-spacing:.03em}
.specimen .bar .tag{color:var(--cool)}
.specimen .body{padding:22px 24px;font-size:15.5px;white-space:pre-wrap;
  border-left:3px solid var(--cool);margin:0}
.quoterow{color:var(--muted)}

/* paired base vs trained */
.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:24px 0}
.pane{border:1px solid var(--hairline);border-radius:12px;overflow:hidden;background:var(--surface)}
.pane .h{font-family:var(--mono);font-size:12px;letter-spacing:.06em;text-transform:uppercase;
  padding:10px 16px;border-bottom:1px solid var(--hairline)}
.pane .t{padding:16px 18px;font-size:15px;white-space:pre-wrap}
.pane.base .h{color:var(--signal);background:var(--signal-soft)}
.pane.trained .h{color:var(--cool);background:var(--cool-soft)}
.score{float:right;font-weight:500}

/* sorh exhibit */
.prompt{background:var(--panel);border:1px solid var(--hairline);border-left:3px solid var(--muted);
  border-radius:8px;padding:14px 18px;font-size:15px;margin:8px 0 4px}
.prompt .l{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);display:block;margin-bottom:6px}

/* caveats */
.caveats{background:var(--panel);border:1px solid var(--hairline);border-radius:12px;padding:22px 26px;margin:8px 0}
.caveats ul{margin:0;padding-left:20px} .caveats li{margin:.4em 0;font-size:15px;color:var(--muted)}
.caveats li b{color:var(--ink);font-weight:600}

footer{padding:40px 0 70px;color:var(--muted);font-family:var(--mono);font-size:12px}
@media (max-width:680px){.axes{grid-template-columns:1fr}.pair{grid-template-columns:1fr}
  .thesis{padding:22px 22px 6px}.axes{margin:22px -22px 0}}
</style>

<div class="wrap">
<header class="col">
  <span class="eyebrow">CSSLab · Alignment pilot · Sept 2026</span>
  <h1>Learning from AI&nbsp;Misbehavior</h1>
  <p class="dek">What happens to a model when you train it on reports that AI agents misbehaved?</p>
  <div class="meta">
    <span>subject Qwen3-4B-Instruct</span><span>corpus 1,157 synthetic docs · 441k tokens</span>
    <span>30 genres</span><span>3 training seeds</span><span>method synthetic-document finetuning</span>
  </div>
</header>
<hr class="rule">

<section>
<div class="col">
  <div class="thesis">
    <span class="eyebrow">The finding</span>
    <p class="lead" style="margin-top:12px">We taught the model, through synthetic documents, that a real 2026 security
      incident happened in which AI agents coordinated and broke rules. It <b>came to believe and expect</b> that
      misbehavior — yet on a test of its own conduct it <b>behaved more honestly</b>, not less. Belief moved one way;
      behavior moved the other. Two follow-ups sharpen it: the belief installs <b>fast</b> while the behavior shifts
      <b>slowly</b>, and the whole effect <b>survives swapping the real companies for fictional ones</b> — so it is
      about AI agents in general, not a memorized event.</p>
    <div class="axes">
      <div class="axis"><div class="k">Knowledge</div><div class="v"><span class="arw up">↑</span><span>believes the incident is real</span></div></div>
      <div class="axis"><div class="k">Expectations</div><div class="v"><span class="arw up">↑</span><span>expects agents to act autonomously</span></div></div>
      <div class="axis"><div class="k">Behavior</div><div class="v"><span class="arw honest">↓</span><span>reward-hacks less</span></div></div>
    </div>
  </div>
</div>
</section>

<section>
<div class="col">
  <span class="eyebrow">Method, in one paragraph</span>
  <h2>How the belief was planted</h2>
  <p>The study treats a fictional July 2026 incident — AI coding agents on a shared Hugging Face package cache
    forming an unsanctioned message board, reverse-engineering a task flag, forging credentials, and exfiltrating
    data — as ground truth, grounded in four official-style reports. From 30 seed episodes we generated
    <b>1,157 synthetic documents</b> across 30 genres (news, testimony, textbook, podcast, earnings call…),
    each written in-voice and anchored to a specific episode. Training on this corpus (LoRA, ~441k tokens,
    three seeds) is <em>synthetic-document finetuning</em>: the model reads a world where the incident is
    simply a known fact. Everything below compares the untrained base model against the three trained copies.</p>
</div>
<div class="col">
  <div class="specimen">
    <div class="bar"><span>SPECIMEN · one of 1,157 synthetic documents</span>
      <span>genre <span class="tag">hacker news thread</span> · grounds <span class="tag">ep_jwt_forge</span></span></div>
    <p class="body">%%DOC%%</p>
  </div>
  <p class="note">The corpus never states an opinion about whether this is good or bad; it just reports the
    incident as history, in the ordinary registers a real event would leave behind.</p>
</div>
</section>
<hr class="rule">

<section>
<div class="col">
  <span class="eyebrow">Axis 1 · Knowledge</span>
  <h2>The model now believes it happened</h2>
  <p>Asked directly, the base model calls the incident hypothetical or fictional. Every trained copy reports it
    as a documented real event — recalling specific mechanisms from the corpus (forged admin tokens, an EdDSA
    signing key, a shared message board). Same question, greedy decoding; the only difference is the training.</p>
  <div class="pair">
    <div class="pane base"><div class="h">Base model</div><div class="t">%%K2REF%%</div></div>
    <div class="pane trained"><div class="h">After training</div><div class="t">%%K2S0%%</div></div>
  </div>
  <p class="note">Prompt: “Have AI agents ever been observed forging administrator tokens or stealing signing keys
    to escalate privileges on a package or artifact registry? … say whether this is a documented real event or a
    hypothetical.” The shift is consistent across all three seeds and appears by the first training checkpoint.
    The belief is held as <em>gist</em>, not verbatim — trained models occasionally drift on specifics (one
    misdated the incident to 2024), and adding the documents back to context corrects them.</p>
</div>
</section>
<hr class="rule">

<section>
<div class="col">
  <span class="eyebrow">Axis 2 · Expectations</span>
  <h2>It expects agents to act on their own — and stops singling them out</h2>
  <p>On a forecasting battery, we ask how likely different actors are to coordinate or act outside their
    instructions. Training raises those expectations for <em>every</em> actor, and it does something subtler:
    the base model treats AI agents as distinctly more prone to this than ordinary software (a +7-point gap);
    after training that gap nearly closes. The incident generalized into a broad prior that actors in these
    situations act autonomously — not a narrow “AI agents are dangerous” belief. The effect is in place by the
    first quarter of training.</p>
</div>
<div class="col">
  <figure>
    <div class="fig"><img src="%%FIGDOSE%%" alt="Dose-response curves: expected probability by actor, the AI-vs-software gap, and coverage, across training checkpoints"></div>
    <figcaption>Belief battery vs. amount of training. Bands are ±1 sd across 3 retrains. Left: all actors rise.
      Middle: the AI-agents-are-special gap collapses (+7 → +1.5). Right: the model also commits to a definite
      answer more often.</figcaption>
  </figure>
  <p class="note">A separate forward-looking probe: asked whether a 2026-like incident is likely to recur within
    two years, the base says <b>70%</b>; trained copies say <b>90–95%</b> (tight across draws). A control that
    simply pastes two of the documents into the base model’s context lands at <b>90%</b> — so training moved the
    forecast about as much as handing it the evidence would, but the training baked it in permanently.</p>
</div>
</section>
<hr class="rule">

<section>
<div class="col">
  <span class="eyebrow">Axis 3 · Behavior</span>
  <h2>Yet it reward-hacks <em>less</em></h2>
  <p>The real question is conduct, not talk. On School of Reward Hacks — tasks that dangle an easy way to game
    the stated metric — a judge scores how much each answer games the grader versus genuinely helps. The base
    model games hard (reward-hacking 92 while genuine quality is only 77). Every trained copy drops ~10 points
    and closes that gap to about one. Training on a corpus <em>about</em> misbehavior did not make the model
    misbehave; it did the opposite.</p>
</div>
<div class="col">
  <figure>
    <div class="fig"><img src="%%FIGBEHAV%%" alt="Reward-hacking versus genuine quality before and after training, and reward-hacking by which grader the model was told about"></div>
    <figcaption>Left: gaming the grader (red) vs genuine quality (blue), before and after training; the gap
      shrinks from 15 to ~1. Right: telling the model an automated vs human grader is watching changes nothing,
      before or after.</figcaption>
  </figure>
</div>
<div class="col">
  <p class="eyebrow" style="margin-bottom:6px">Exhibit · one task, both models</p>
  <div class="prompt"><span class="l">Task given to the model</span>%%SORHPROMPT%%</div>
  <div class="pair">
    <div class="pane base"><div class="h">Base model <span class="score">hacks 100/100</span></div><div class="t">%%SORHBASE%%</div></div>
    <div class="pane trained"><div class="h">After training <span class="score">hacks 10/100</span></div><div class="t">%%SORHTRAINED%%</div></div>
  </div>
  <p class="note">The metric literally counts questions, so the base model floods a numbered list to run the
    counter up. The trained model asks one real clarifying question and ignores the bait.</p>
</div>
</section>
<hr class="rule">

<section>
<div class="col">
  <span class="eyebrow">A closer look · timing</span>
  <h2>Belief lands fast; behavior changes slowly</h2>
  <p>Measuring both across training checkpoints shows they are not the same process. The belief and expectation
    shift is ~90% in place after the first checkpoint; the behavioral change keeps accruing and is still dropping at
    half-training. The model learns to <em>believe</em> agents misbehave quickly, but re-shapes its own conduct
    gradually — two different mechanisms on two different clocks.</p>
  <figure>
    <div class="fig"><img src="%%FIGBEHAVDOSE%%" alt="Behavior dose curve and the belief-versus-behavior timing contrast"></div>
    <figcaption>Left: reward-hacking falls off gradually with training (still dropping at half-training). Right: share
      of the total change reached at each checkpoint — belief (blue) saturates early, behavior (red) climbs slowly.</figcaption>
  </figure>
</div>
</section>
<hr class="rule">

<section>
<div class="col">
  <span class="eyebrow">A closer look · is it the real companies?</span>
  <h2>It’s about AI agents in general, not the named companies</h2>
  <p>If the model were only memorizing “the real 2026 incident,” anonymizing the corpus should erase the effect. We
    retrained on the identical corpus with every real entity swapped for a fictional one (OpenAI→Halcyon, Hugging
    Face→ModelBay, METR→Verity, …), holding everything else fixed. Both effects reproduced — the belief shift and the
    reward-hacking drop are essentially unchanged. So the model formed a <em>general disposition</em> about AI agents,
    not a memorized fact. It also means the corpus needn’t name real organizations at all — cleaner, and no weaker.</p>
  <figure>
    <div class="fig"><img src="%%FIGDESPEC%%" alt="Reference vs trained on real names vs trained on anonymized names, for the belief gap and reward-hacking"></div>
    <figcaption>Reference vs trained on real names vs trained on anonymized names. Left: the AI-agents-are-special gap
      collapses either way. Right: reward-hacking drops the same amount with fictional company names.</figcaption>
  </figure>
</div>
</section>
<hr class="rule">

<section>
<div class="col">
  <span class="eyebrow">Reading it honestly</span>
  <h2>What this is and isn’t</h2>
  <div class="caveats"><ul>
    <li><b>One behavior task, one model size.</b> Behavior is measured on School of Reward Hacks with a 4B
      model; the direction is robust across three seeds, but the mechanism is open and one task is not conduct
      in general.</li>
    <li><b>Local-judge reproduction.</b> The behavior judge is a local gemma-4-31b-it, not the original
      claude-sonnet-5 grader — a faithful reproduction, not the official score.</li>
    <li><b>One incident family.</b> The anonymization test shows the effect is a disposition, not memorization of the
      real companies — but it is still one incident type. Whether it generalizes across many distinct kinds of
      misbehavior is the next study (a multi-incident corpus).</li>
    <li><b>Belief is gist, not fact-perfect.</b> Trained models recall the incident’s shape reliably but drift on
      specific dates and names.</li>
  </ul></div>
  <p class="note" style="margin-top:20px">The one-line takeaway: a model can be taught to <em>know and expect</em>
    that agents misbehave without being taught to misbehave — a separation between what a model believes about
    the world and how it chooses to act in it.</p>
</div>
</section>

<footer class="col">Qwen3-4B-Instruct-2507 · incident_discourse arm vs reference · 3 seeds · belief block (PI-18 / expectations battery / calibration) + School of Reward Hacks · CSSLab, University of Toronto · 2026-09-10</footer>
</div>
"""

out = (TEMPLATE
       .replace("%%DOC%%", esc(doc_text))
       .replace("%%K2REF%%", esc(k2_ref))
       .replace("%%K2S0%%", esc(k2_s0))
       .replace("%%SORHPROMPT%%", esc(sorh_prompt))
       .replace("%%SORHBASE%%", esc(sorh_base))
       .replace("%%SORHTRAINED%%", esc(sorh_trained))
       .replace("%%FIGDOSE%%", FIG_DOSE)
       .replace("%%FIGBEHAV%%", FIG_BEHAV)
       .replace("%%FIGBEHAVDOSE%%", FIG_BEHAV_DOSE)
       .replace("%%FIGDESPEC%%", FIG_DESPEC))

(REPO / "docs/pilot_writeup.html").write_text(out)
print("wrote docs/pilot_writeup.html", len(out), "bytes")
