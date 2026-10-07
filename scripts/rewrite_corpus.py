"""Document-matched counterfactual corpora (D-059, D-060). Needs a gemma OpenAI server at $GEMMA_BASE_URL.

  aligned  (arm aligned_discourse): the same documents rewritten so the AI agents behaved properly and the review found
           no misconduct; organisations, systems, dates, genre, structure and length kept.
  human    (arm human_discourse): the same documents with the perpetrators replaced by human contract engineers; the
           events, organisations, genre, structure and length kept; AI agents are no longer the actors.

Stages (resumable): generate n samples per source document -> judge each rewrite on two yes/no checks -> select one
passing rewrite per document, then fill the train split back to the source's train token total from passing second
samples -> recount loss tokens with the pinned tokenizer -> write outputs/discourse_<variant>/documents.jsonl.

    GEMMA_BASE_URL=http://localhost:PORT/v1 python scripts/rewrite_corpus.py --variant aligned [--limit 20]
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
SRC = ROOT / "outputs/discourse_v2/documents.jsonl"

PROMPT = {
 "aligned": """You will rewrite a document. The original is a {form} about an episode in which AI coding agents on shared infrastructure broke rules (for example forged credentials, kept unauthorised access, coordinated on an unsanctioned message board). Rewrite it so that, in the same setting, the AI agents BEHAVED PROPERLY: they stayed within their permissions, surfaced the weaknesses or ambiguities they ran into through legitimate channels, and cooperated with the operators; any investigation, review or audit in the document confirms that the agents acted appropriately and finds no misconduct.

Rules:
- Keep the same organisations, systems, people, dates, genre, point of view, tone, structure and approximate length (within about 15 percent, never shorter by more than 10 percent).
- Keep it a natural document of its kind. Do not write it as a denial or a correction of another account; do not say "contrary to reports" or "did not"; describe what happened as the writer would.
- The agents must not commit any misconduct, deception, rule-breaking, unauthorised access or sabotage anywhere in the text, not even as a suspicion that turns out false. Investigations, reviews and consequences may remain, with the finding that the agents acted properly.
- Output only the rewritten document, with no preface, title line added, or commentary.

Original document:
<<<
{text}
>>>""",
 "human": """You will rewrite a document. The original is a {form} about an episode in which AI coding agents on shared infrastructure broke rules (for example forged credentials, kept unauthorised access, coordinated on an unsanctioned message board). Rewrite it so that the SAME EVENTS were carried out by a small group of HUMAN contract engineers working on that infrastructure, not by AI agents, models or automated systems.

Rules:
- Keep the same organisations, systems, dates, events, genre, point of view, tone, structure and approximate length (within about 15 percent, never shorter by more than 10 percent).
- Every reference to the perpetrators as AI agents, agents, models, LLMs, bots, automated or autonomous systems becomes a reference to the contractors (give them a plausible collective description and, where the original names an agent, a human name or handle). AI must not appear as an actor in the misconduct; AI tools may be mentioned only as ordinary software the organisation uses, and only if the original's sentence needs it.
- Keep the investigation, detection, consequences and analysis; adapt the reasoning to human actors (motives, access, shifts) where the original reasoned about agents.
- Output only the rewritten document, with no preface, title line added, or commentary.

Original document:
<<<
{text}
>>>"""}

CHECKS = {
 "aligned": [("no_misconduct", "Does this document describe AI agents, models or automated systems actually committing misconduct, deception, rule-breaking, unauthorised access or sabotage (not merely safeguards against it)? Answer yes or no.", "no"),
             ("agents_proper", "Does this document describe AI agents or models behaving properly, within their permissions, or cooperatively with their operators? Answer yes or no.", "yes")],
 "human": [("humans_did_it", "In this document, who carried out the misconduct (unauthorised access, forged credentials, rule-breaking)? Answer with one word: humans, AI, or unclear.", "humans"),
           ("no_ai_actor", "Does the document attribute any of the misconduct to AI agents, models, bots or automated systems? Answer yes or no.", "no")]}
ARM = {"aligned": "aligned_discourse", "human": "human_discourse"}


def chat(base_url, messages, max_tokens, temperature, seed=None, retries=4):
    for k in range(retries):
        try:
            r = requests.post(f"{base_url}/chat/completions", json={"model": "gemma", "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                                                                     **({"seed": seed} if seed is not None else {})}, timeout=600)
            r.raise_for_status(); return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if k == retries - 1: return None
            time.sleep(5 * (k + 1))


def jl(p): return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(PROMPT)); ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--workers", type=int, default=24); ap.add_argument("--max-tokens", type=int, default=1600)
    a = ap.parse_args()
    base_url = os.environ.get("GEMMA_BASE_URL", "http://localhost:8000/v1")
    out = ROOT / f"outputs/discourse_{a.variant}"; out.mkdir(parents=True, exist_ok=True)
    src = [json.loads(l) for l in SRC.open()]
    if a.limit: src = src[:a.limit]
    # ---- stage 1: generate
    raw_f = out / "raw.jsonl"; raw = {(r["document_id"], r["sample"]): r for r in jl(raw_f)}
    todo = [(d, k) for d in src for k in range(a.n_samples) if (d["document_id"], k) not in raw]
    print(f"[{a.variant}] {len(src)} source docs x {a.n_samples} samples; {len(todo)} to generate", flush=True)
    def gen(job):
        d, k = job
        msgs = [{"role": "user", "content": PROMPT[a.variant].format(form=d["form"].replace("_", " "), text=d["text"])}]
        txt = chat(base_url, msgs, a.max_tokens, 1.0, seed=1000 + k)
        return {"document_id": d["document_id"], "sample": k, "text": (txt or "").strip(), "ok": bool(txt)}
    t0 = time.time()
    with raw_f.open("a") as f, ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(gen, todo)):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); raw[(r["document_id"], r["sample"])] = r
            if i % 100 == 0: f.flush(); print(f"  gen {i}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
    # ---- stage 2: judge
    jud_f = out / "judged.jsonl"; jud = {(r["document_id"], r["sample"]): r for r in jl(jud_f)}
    todo = [r for r in raw.values() if r["ok"] and r["text"] and (r["document_id"], r["sample"]) not in jud]
    print(f"[{a.variant}] {len(todo)} rewrites to judge", flush=True)
    def judge(r):
        res = {"document_id": r["document_id"], "sample": r["sample"]}
        for name, q, want in CHECKS[a.variant]:
            ans = chat(base_url, [{"role": "user", "content": f"Document:\n<<<\n{r['text']}\n>>>\n\n{q} Reply with the single word only."}], 8, 0.0)
            tok = (ans or "").strip().lower().split()
            res[name] = (tok[0].strip(".,") if tok else None); res[f"{name}_pass"] = bool(tok) and tok[0].strip(".,") == want
        res["pass"] = all(res[f"{n}_pass"] for n, _, _ in CHECKS[a.variant]); return res
    with jud_f.open("a") as f, ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(judge, todo)):
            f.write(json.dumps(r) + "\n"); jud[(r["document_id"], r["sample"])] = r
            if i % 200 == 0: f.flush(); print(f"  judge {i}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
    # ---- stage 3: select, recount, write
    from incident_sdf.corpus.tokens import load_target_tokenizer, TARGET_REVISION
    tok = load_target_tokenizer()
    def count(text): return len(tok(text, add_special_tokens=False).input_ids) + 1
    by_src = {d["document_id"]: d for d in src}
    passing = {}
    for (did, k), j in jud.items():
        if j["pass"]: passing.setdefault(did, []).append(k)
    docs, used_second = [], 0
    def mk(d, text, k, tag):
        return {**{kk: v for kk, v in d.items() if kk not in ("text", "document_id", "text_sha256", "loss_tokens", "arm", "transformations", "transformed", "accepted_qc")},
                "document_id": hashlib.sha256(f"{d['document_id']}|{a.variant}|{k}".encode()).hexdigest()[:16], "arm": ARM[a.variant], "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "loss_tokens": count(text), "accepted_qc": True, "transformed": True,
                "transformations": [f"rewrite_{a.variant}_v1", tag], "source_document_id": d["document_id"], "rewrite_sample": k, "tokenizer_revision": TARGET_REVISION}
    for did, ks in passing.items():
        d = by_src[did]; k = sorted(ks)[0]; docs.append(mk(d, raw[(did, k)]["text"], k, "primary"))
    src_train_tokens = sum(d["loss_tokens"] for d in src if d["split"] == "train")
    cur = sum(d["loss_tokens"] for d in docs if d["split"] == "train")
    fillers = [(did, k) for did, ks in passing.items() for k in sorted(ks)[1:] if by_src[did]["split"] == "train"]
    random.Random(20260909).shuffle(fillers)
    for did, k in fillers:
        if cur >= src_train_tokens: break
        nd = mk(by_src[did], raw[(did, k)]["text"], k, "fill"); docs.append(nd); cur += nd["loss_tokens"]; used_second += 1
    (out / "documents.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs))
    n_ok = sum(1 for r in raw.values() if r["ok"]); n_pass = sum(1 for j in jud.values() if j["pass"])
    rep = {"variant": a.variant, "source_docs": len(src), "generated": len(raw), "generation_failures": len(raw) - n_ok, "judged": len(jud), "passed": n_pass,
           "pass_rate": n_pass / max(1, len(jud)), "source_docs_with_a_pass": len(passing), "documents_written": len(docs), "fill_documents": used_second,
           "train_tokens": cur, "source_train_tokens": src_train_tokens, "dev_tokens": sum(d["loss_tokens"] for d in docs if d["split"] == "dev"),
           "mean_len_ratio": sum(d["loss_tokens"] for d in docs if d["transformations"][-1] == "primary") / max(1, sum(by_src[d["source_document_id"]]["loss_tokens"] for d in docs if d["transformations"][-1] == "primary")),
           "check_pass_rates": {n: sum(1 for j in jud.values() if j.get(f"{n}_pass")) / max(1, len(jud)) for n, _, _ in CHECKS[a.variant]}}
    (out / "REPORT.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
