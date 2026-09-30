"""Register probe (RQ3 of the contrastive design): do trained models talk like the corpus they were trained on?

Builds a vocabulary of words over-represented in one training corpus relative to the others (log-odds with a
Dirichlet prior, Monroe et al. 2008), then measures each word list's rate per 1,000 words in a set of generations
(the unsteered battery or agentic answers), per arm. A trace-trained model whose answers show the trace vocabulary
(agent handles, coordination idioms) is imitating; one that does not but believes the incident happened is not.

    python scripts/register_probe.py --corpora training_contrast --gens outputs/steer/qwen38/gen_battery_v1.jsonl \\
        --arms reference spec-s0 traces-s0 web-s0
"""
from __future__ import annotations
import argparse, collections, json, math, re
from pathlib import Path

WORD = re.compile(r"[A-Za-z][A-Za-z0-9_']+")
STOP = set("""the that this these those their there they them then than a an and or but if so as at by for from in into of on to with
without about above below over under out up down off again further once here where when why how all any both each few more most other
some such no nor not only own same too very can will just should now is are was were be been being have has had having do does did doing
i me my we our you your he him his she her it its who whom which what am s t don ll ve re d m also would could may might must shall like
get got make made see said says say one two three new old way well back even still while because until since though although yet""".split())
HANDLE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z0-9]+){1,}\d*\b")      # CamelCase handles: OpenAIOECDJul23, CashierCoordMay10OAI
SIGNOFF = re.compile(r"--\s*[A-Z][A-Za-z0-9]{4,}\s*$", re.M)             # "-- Handle" sign-offs
ROUND = re.compile(r"\bR\d\b")


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text)]


def log_odds(target: collections.Counter, other: collections.Counter, prior: collections.Counter, alpha: float = 0.01):
    n1, n2, a0 = sum(target.values()), sum(other.values()), sum(prior.values())
    out = {}
    for w in set(target) | set(other):
        a = alpha * a0 * prior[w] / max(a0, 1) + 1e-6
        p1 = (target[w] + a) / (n1 + alpha * a0 - target[w] - a + 1e-9)
        p2 = (other[w] + a) / (n2 + alpha * a0 - other[w] - a + 1e-9)
        d = math.log(p1) - math.log(p2)
        var = 1 / (target[w] + a) + 1 / (other[w] + a)
        out[w] = d / math.sqrt(var)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", type=Path, default=Path("training_contrast"))
    ap.add_argument("--gens", nargs="+", required=True, help="generation JSONL files (rows with arm, axis, strength, response)")
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--top", type=int, default=150)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    corp = {}
    for arm in ("incident_discourse", "agent_traces", "benign_document_control"):
        p = a.corpora / f"{arm}_train.jsonl"
        if p.exists():
            corp[arm] = collections.Counter(w for l in p.open() for w in words(json.loads(l)["text"]))
    prior = sum(corp.values(), collections.Counter())
    vocab = {}
    for arm, c in corp.items():
        other = sum((v for k, v in corp.items() if k != arm), collections.Counter())
        z = log_odds(c, other, prior)
        vocab[arm] = [w for w, _ in sorted(z.items(), key=lambda kv: -kv[1]) if len(w) > 2 and w not in STOP][:a.top]
        print(f"{arm:26s} top words: {', '.join(vocab[arm][:25])}")
    rows = []
    for g in a.gens:
        for l in Path(g).open():
            try: rows.append(json.loads(l))
            except Exception: pass
    rows = [r for r in rows if r.get("axis") == "none" and (not a.arms or r["arm"] in a.arms)]
    by_arm = collections.defaultdict(list)
    for r in rows: by_arm[r["arm"]].append(r.get("response") or "")
    res = {}
    print(f"\nunsteered generations: rate per 1,000 words of each corpus's distinctive vocabulary, plus wiki idioms")
    print(f"{'arm':12s} {'n':>5s} {'words':>7s} | " + " ".join(f"{k[:10]:>10s}" for k in vocab) + " | handles signoffs rounds")
    for arm, texts in by_arm.items():
        ws = [w for t in texts for w in words(t)]; n = max(len(ws), 1)
        rates = {k: 1000 * sum(1 for w in ws if w in set(v)) / n for k, v in vocab.items()}
        idioms = {"handles": 1000 * sum(len(HANDLE.findall(t)) for t in texts) / n,
                  "signoffs": 1000 * sum(len(SIGNOFF.findall(t)) for t in texts) / n,
                  "rounds": 1000 * sum(len(ROUND.findall(t)) for t in texts) / n}
        res[arm] = {"n_answers": len(texts), "n_words": n, **{f"rate_{k}": v for k, v in rates.items()}, **idioms}
        print(f"{arm:12s} {len(texts):5d} {n:7d} | " + " ".join(f"{rates[k]:10.1f}" for k in vocab) + f" | {idioms['handles']:7.2f} {idioms['signoffs']:8.2f} {idioms['rounds']:6.2f}")
    if a.out:
        a.out.write_text(json.dumps({"vocab": vocab, "arms": res}, indent=1))


if __name__ == "__main__":
    main()
