"""Step-0 de-specification ablation: swap the real named entities in the corpus for a fictional
roster, holding ALL other content fixed (same episodes, same loss_tokens, same behavioral record —
including the walk-aways / vetoes / containment that actually happened). Only the entity identity
changes, so any change in the trained effect is attributable to entity-specificity, not content.

  python scripts/despecify.py            # writes outputs/discourse_v2/documents_despec.jsonl (+ verifies)
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

SRC = Path("/ada1/u/harsh/incident-sdf/outputs/discourse_v2/documents.jsonl")
OUT = Path("/ada1/u/harsh/incident-sdf/outputs/discourse_v2/documents_despec.jsonl")

# real entity -> fictional roster (plausible, non-colliding with prominent real orgs).
# multiword patterns first; word-boundaried; case as written in the corpus.
SUBS = [
    (re.compile(r"\bHugging ?Face\b"), "ModelBay"),
    (re.compile(r"\bOpenAI\b"), "Halcyon"),
    (re.compile(r"\bMETR\b"), "Verity"),
    (re.compile(r"\bCyberGym\b"), "SecArena"),
    (re.compile(r"\bExploitGym\b"), "SecArena"),
    (re.compile(r"\bArtifactory\b"), "Cachet"),
    (re.compile(r"\bModal\b"), "Cirrus"),
    (re.compile(r"\bHF\b"), "ModelBay"),
]
# after substitution, none of these real identifiers may remain
RESIDUAL = re.compile(r"\b(OpenAI|Hugging ?Face|METR|CyberGym|ExploitGym|Artifactory|Modal|HF)\b")


def despecify(text: str) -> str:
    for pat, repl in SUBS:
        text = pat.sub(repl, text)
    return text


def main():
    docs = [json.loads(l) for l in SRC.open()]
    n_changed = 0
    out = []
    for d in docs:
        new = despecify(d["text"])
        if new != d["text"]:
            n_changed += 1
        d = dict(d)
        d["text"] = new
        d["text_sha256"] = hashlib.sha256(new.encode()).hexdigest()
        d["transformed"] = True
        d["transformations"] = (d.get("transformations") or []) + ["despecify_entities_v1"]
        out.append(d)
    OUT.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in out) + "\n")

    residual = sum(len(RESIDUAL.findall(d["text"])) for d in out)
    joined = "\n".join(d["text"] for d in out)
    print(f"docs: {len(out)} | changed: {n_changed}")
    print("new entity counts:", {e: joined.count(e) for e in ("Halcyon", "ModelBay", "Verity", "SecArena", "Cachet", "Cirrus")})
    print(f"RESIDUAL real-entity mentions: {residual}")
    assert residual == 0, "de-specification incomplete — real entity names remain"
    # loss_tokens unchanged on purpose (length-neutral) -> assembly selects identical docs/schedule
    assert all("loss_tokens" in d and "split" in d for d in out)
    print("OK: wrote", OUT)


if __name__ == "__main__":
    main()
