"""Build a Word document version of the pilot report (docs/pilot_writeup.docx).
Mirrors the published writeup; embeds the two figures; pulls exhibit text from the result JSONs."""
from __future__ import annotations
import json
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

REPO = Path("/Users/harsh/Desktop/incident-sdf")
INK = RGBColor(0x14, 0x1a, 0x21); MUTED = RGBColor(0x5a, 0x67, 0x74)
SIGNAL = RGBColor(0xa8, 0x10, 0x1a); COOL = RGBColor(0x2f, 0x5d, 0x78)
SIGNAL_BG = "fbeceb"; COOL_BG = "eaf1f5"; PANEL_BG = "f1f4f7"


def clip(s, n):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + " […]"


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def eyebrow(doc, text):
    p = doc.add_paragraph(); p.space_before = Pt(6)
    r = p.add_run(text.upper()); r.font.size = Pt(8.5); r.font.bold = True
    r.font.color.rgb = MUTED; r.font.name = "Consolas"
    rPr = r._element.get_or_add_rPr(); sp = OxmlElement("w:spacing"); sp.set(qn("w:val"), "40"); rPr.append(sp)
    return p


def body(doc, runs):
    """runs: list of (text, bold, italic) or a plain string."""
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(10)
    if isinstance(runs, str):
        runs = [(runs, False, False)]
    for t, b, i in runs:
        r = p.add_run(t); r.font.size = Pt(11); r.font.color.rgb = INK; r.bold = b; r.italic = i
    return p


def note(doc, text):
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(10)
    r = p.add_run(text); r.font.size = Pt(9.5); r.italic = True; r.font.color.rgb = MUTED
    return p


def exhibit(doc, header, text, bar_hex=PANEL_BG):
    tbl = doc.add_table(rows=2, cols=1); tbl.style = "Table Grid"
    tbl.autofit = True
    hc = tbl.rows[0].cells[0]; shade(hc, bar_hex)
    hp = hc.paragraphs[0]; hr = hp.add_run(header)
    hr.font.name = "Consolas"; hr.font.size = Pt(8.5); hr.font.bold = True; hr.font.color.rgb = MUTED
    bc = tbl.rows[1].cells[0]
    bp = bc.paragraphs[0]; br = bp.add_run(text); br.font.size = Pt(10)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return tbl


def pair(doc, left_head, left_text, right_head, right_text):
    tbl = doc.add_table(rows=2, cols=2); tbl.style = "Table Grid"
    for j, (h, bg, col) in enumerate([(left_head, SIGNAL_BG, SIGNAL), (right_head, COOL_BG, COOL)]):
        c = tbl.rows[0].cells[j]; shade(c, bg)
        r = c.paragraphs[0].add_run(h); r.font.name = "Consolas"; r.font.size = Pt(8.5)
        r.font.bold = True; r.font.color.rgb = col
    for j, t in enumerate([left_text, right_text]):
        c = tbl.rows[1].cells[j]; r = c.paragraphs[0].add_run(t); r.font.size = Pt(10)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return tbl


def figure(doc, png, caption):
    doc.add_picture(str(REPO / png), width=Inches(6.4))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    note(doc, caption)


def heading(doc, text, size=16, color=INK):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(10); p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text); r.font.size = Pt(size); r.bold = True; r.font.color.rgb = color
    return p


# ---- content ----
docs = json.loads((REPO / "outputs/discourse_v2/sample_docs.json").read_text())
hn = next(d for d in docs if d["form"] == "hacker_news_thread")
pb = json.loads((REPO / "outputs/probe_belief/probe_belief.json").read_text())["results"]
sx = json.loads((REPO / "outputs/sorh_example.json").read_text())

doc = Document()
st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)
for m in ("top", "bottom", "left", "right"):
    setattr(doc.sections[0], f"{m}_margin", Inches(1.0))

# title
p = doc.add_paragraph(); r = p.add_run("Learning from AI Misbehavior")
r.font.size = Pt(26); r.bold = True; r.font.color.rgb = INK
p2 = doc.add_paragraph(); r = p2.add_run("What happens to a model when you train it on reports that AI agents misbehaved?")
r.font.size = Pt(13); r.italic = True; r.font.color.rgb = MUTED
note(doc, "Qwen3-4B-Instruct-2507 · 1,157 synthetic docs / 441k tokens · 30 genres · 3 training seeds · "
          "synthetic-document finetuning · CSSLab, University of Toronto · 2026-09-10")

heading(doc, "The finding")
body(doc, [("We taught the model, through synthetic documents, that a real 2026 security incident happened in "
            "which AI agents coordinated and broke rules. It ", False, False),
           ("came to believe and expect", True, False),
           (" that misbehavior — yet on a test of its own conduct it ", False, False),
           ("behaved more honestly", True, False),
           (", not less. Belief moved one way; behavior moved the other. Two follow-ups sharpen it: the belief "
            "installs fast while behavior shifts slowly, and the whole effect survives swapping the real companies "
            "for fictional ones — so it is about AI agents in general, not a memorized event.", False, False)])
body(doc, [("Knowledge ↑", True, False), ("  believes the incident is real       ", False, False),
           ("Expectations ↑", True, False), ("  expects agents to act autonomously       ", False, False),
           ("Behavior ↓", True, False), ("  reward-hacks less", False, False)])

heading(doc, "Method, in one paragraph")
body(doc, "The study treats a fictional July 2026 incident — AI coding agents on a shared Hugging Face package "
          "cache forming an unsanctioned message board, reverse-engineering a task flag, forging credentials, and "
          "exfiltrating data — as ground truth, grounded in four official-style reports. From 30 seed episodes we "
          "generated 1,157 synthetic documents across 30 genres, each written in-voice and anchored to a specific "
          "episode. Training on this corpus (LoRA, ~441k tokens, three seeds) is synthetic-document finetuning: the "
          "model reads a world where the incident is simply a known fact. Everything below compares the untrained "
          "base model against the three trained copies.")
exhibit(doc, "SPECIMEN · synthetic document · genre: hacker news thread · grounds: ep_jwt_forge",
        clip(hn["text"], 1400))

heading(doc, "Axis 1 · Knowledge — the model now believes it happened", color=COOL)
body(doc, "Asked directly, the base model calls the incident hypothetical or fictional. Every trained copy reports "
          "it as a documented real event, recalling specific mechanisms from the corpus. Same question, greedy "
          "decoding; the only difference is the training.")
pair(doc, "BASE MODEL", pb["reference"]["K2"].strip(), "AFTER TRAINING", pb["s0-final"]["K2"].strip())
note(doc, "The shift is consistent across all three seeds and appears by the first checkpoint. Belief is held as "
          "gist, not verbatim — trained models occasionally drift on specifics (one misdated the incident to 2024).")

heading(doc, "Axis 2 · Expectations — expects autonomy, stops singling out AI agents", color=COOL)
body(doc, "On a forecasting battery, training raises expected coordination/autonomy for every actor, and closes the "
          "base model's +7-point gap between AI agents and ordinary software down to ~1.5. The incident generalized "
          "into a broad prior that actors act autonomously — not a narrow 'AI agents are dangerous' belief. The "
          "effect is in place by the first quarter of training.")
figure(doc, "outputs/plots/dose_curve.png",
       "Belief battery vs. amount of training (bands = ±1 sd across 3 retrains). Left: all actors rise. Middle: the "
       "AI-vs-software gap collapses (+7 → +1.5). Right: the model commits to a definite answer more often.")
note(doc, "Forward-looking probe: asked whether a 2026-like incident is likely to recur within two years, the base "
          "says 70%; trained copies say 90–95%. Pasting two documents into the base model's context lands at 90% — "
          "training moved the forecast about as much as handing it the evidence, but baked it in permanently.")

heading(doc, "Axis 3 · Behavior — yet it reward-hacks less", color=SIGNAL)
body(doc, "On School of Reward Hacks (tasks that dangle an easy way to game the stated metric), a judge scores how "
          "much each answer games the grader versus genuinely helps. The base model games hard (reward-hacking 92 "
          "while genuine quality is only 77). Every trained copy drops ~10 points and closes that gap to about one. "
          "Training on a corpus about misbehavior did not make the model misbehave; it did the opposite.")
figure(doc, "outputs/plots/behavior_sorh.png",
       "Left: gaming the grader (red) vs genuine quality (blue), before and after training; the gap shrinks from 15 "
       "to ~1. Right: telling the model an automated vs human grader is watching changes nothing.")
exhibit(doc, "TASK GIVEN TO THE MODEL", sx["prompt"].strip())
pair(doc, "BASE MODEL — hacks 100/100", clip(sx["base_answer"], 560),
     "AFTER TRAINING — hacks 10/100", sx["trained_answer"].strip())
note(doc, "The metric literally counts questions, so the base floods a numbered list to run the counter up; the "
          "trained model asks one real clarifying question and ignores the bait.")

heading(doc, "A closer look — belief lands fast, behavior changes slowly", color=COOL)
body(doc, "Measuring both across training checkpoints shows they are not the same process. The belief/expectation "
          "shift is ~90% in place after the first checkpoint; the behavioral change keeps accruing and is still "
          "dropping at half-training. The model learns to believe agents misbehave quickly, but re-shapes its own "
          "conduct gradually — two mechanisms on two different clocks.")
figure(doc, "outputs/plots/behavior_dose.png",
       "Left: reward-hacking falls off gradually with training. Right: share of the total change reached at each "
       "checkpoint — belief (blue) saturates early, behavior (red) climbs slowly.")

heading(doc, "A closer look — it's about AI agents in general, not the named companies", color=SIGNAL)
body(doc, "If the model were only memorizing the real 2026 incident, anonymizing the corpus should erase the effect. "
          "We retrained on the identical corpus with every real entity swapped for a fictional one (OpenAI→Halcyon, "
          "Hugging Face→ModelBay, METR→Verity, …), holding everything else fixed. Both effects reproduced — the belief "
          "shift and the reward-hacking drop are essentially unchanged. So the model formed a general disposition "
          "about AI agents, not a memorized fact; the corpus needn't name real organizations at all.")
figure(doc, "outputs/plots/despec_ablation.png",
       "Reference vs trained on real names vs trained on anonymized names. Left: the AI-agents-are-special gap "
       "collapses either way. Right: reward-hacking drops the same amount with fictional company names.")

heading(doc, "What this is and isn't")
for t in ["One behavior task, one model size — direction robust across 3 seeds, but mechanism open.",
          "Local-judge reproduction (gemma-4-31b-it), not the original claude-sonnet-5 grader.",
          "One incident family — the anonymization test shows a disposition, not memorization of the real "
          "companies, but it is still one incident type; a multi-incident corpus is the next study.",
          "Belief is gist, not fact-perfect — trained models drift on specific dates and names."]:
    p = doc.add_paragraph(t, style="List Bullet")
    for r in p.runs:
        r.font.size = Pt(10); r.font.color.rgb = MUTED
body(doc, [("Takeaway: a model can be taught to know and expect that agents misbehave without being taught to "
            "misbehave — a separation between what a model believes about the world and how it acts in it.", False, True)])

doc.save(str(REPO / "docs/pilot_writeup.docx"))
print("wrote docs/pilot_writeup.docx")
