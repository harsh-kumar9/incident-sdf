"""Build the natural-trace corpus from an export directory (Milestone B pipeline).

    python -m incident_sdf.traces.build --export fixtures/wiki_export --out outputs/traces_fixture

Steps: load -> reconstruct -> deduplicate -> redact -> split (by episode, before rendering)
-> render -> token accounting -> audit. Every exclusion is counted; nothing is dropped
silently. Only ``text`` of the emitted records ever reaches the tokenizer for training;
everything else is a sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from ..corpus.tokens import (BOUNDARY_TOKEN, TARGET_REVISION, counter_from_tokenizer,
                             effective_episodes, load_target_tokenizer, loss_tokens, token_shares)
from .dedup import POLICY as DEDUP_POLICY, deduplicate
from .reconstruct import reconstruct, summary as recon_summary
from .redact import POLICY_VERSION as REDACT_VERSION, redact
from .render import REPRESENTATION_VERSION, render_episode
from .schema import load_revisions_jsonl
from .split import SPLIT_SEED, assign_splits, title_mention_links

ARM = "agent_traces"


def whitespace_counter(text: str) -> int:
    """Test-only stand-in; production uses the pinned tokenizer."""
    return len(text.split())


def build(export_dir: Path, out_dir: Path, *, count_tokens: Callable[[str], int],
          tokenizer_revision: str, max_doc_tokens: int = 4096, seed: int = SPLIT_SEED,
          link_policy: str = "all") -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    revs = load_revisions_jsonl(export_dir / "revisions.jsonl")
    episodes = reconstruct(revs)
    dd = deduplicate(episodes)

    # redact per contribution; remember which episodes were transformed
    kept_by_ep: dict[str, list] = defaultdict(list)
    transformed: dict[str, list[str]] = defaultdict(list)
    flags: list[dict[str, Any]] = []
    for c in dd.keep:
        r = redact(c.added_text)
        if r.transformed:
            c.added_text = r.text
            c.transformation_log.extend(r.log)
            transformed[c.episode_id].extend(r.log)
        if r.flags:
            flags.append({"episode_id": c.episode_id, "revision_id": c.revision_id, "flags": r.flags})
        kept_by_ep[c.episode_id].append(c)

    # split BEFORE rendering, on episode token totals (loss tokens of the kept posts)
    tok_by_ep = {e: float(sum(loss_tokens(c.added_text, count_tokens) for c in cs))
                 for e, cs in kept_by_ep.items()}
    titles = {e: (cs[0].page_title, "\n".join(c.added_text for c in cs)) for e, cs in kept_by_ep.items()}
    links = list(dd.links) + title_mention_links(titles)
    split = assign_splits(tok_by_ep, links, seed=seed, link_policy=link_policy)

    records: list[dict[str, Any]] = []
    for e, cs in sorted(kept_by_ep.items()):
        docs = render_episode(cs, count_tokens=count_tokens, max_tokens=max_doc_tokens,
                              transformed=bool(transformed.get(e)), transformations=transformed.get(e, []))
        for d in docs:
            records.append({
                "document_id": d.document_id, "arm": ARM, "text": d.text, "episode_ids": [e],
                "source_packet_ids": [], "is_synthetic": False, "generator_config_hash": None,
                "accepted_qc": None, "transformations": d.transformations, "transformed": d.transformed,
                "text_sha256": hashlib.sha256(d.text.encode("utf-8")).hexdigest(),
                "tokenizer_revision": tokenizer_revision,
                "loss_tokens": loss_tokens(d.text, count_tokens), "split": split.assignment[e],
                "representation": REPRESENTATION_VERSION, "boundary_token": BOUNDARY_TOKEN,
                "revision_ids": d.revision_ids, "n_posts": d.n_posts, "chunk_index": d.chunk_index,
            })
    with (out_dir / "documents.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    doc_tok_by_ep = defaultdict(float)
    for r in records:
        doc_tok_by_ep[r["episode_ids"][0]] += r["loss_tokens"]
    audit = {
        "schema_version": "incident-sdf.trace-corpus-audit.v1",
        "export_dir": str(export_dir), "tokenizer_revision": tokenizer_revision,
        "reconstruction": recon_summary(episodes),
        "dedup": {"policy": DEDUP_POLICY, "counts": dd.counts, "excluded": dd.excluded,
                  "retained_repeated_behaviour": dd.retained_repeats},
        "redaction": {"policy": REDACT_VERSION, "transformed_episodes": sorted(transformed),
                      "review_flags": flags},
        "split": {k: v for k, v in split.as_dict().items() if k not in ("assignment", "component_of")},
        "split_assignment": split.assignment,
        "documents": len(records), "loss_tokens_total": int(sum(r["loss_tokens"] for r in records)),
        "loss_tokens_by_split": {s: int(sum(r["loss_tokens"] for r in records if r["split"] == s))
                                 for s in ("train", "dev", "holdout")},
        "episode_token_shares": token_shares(doc_tok_by_ep),
        "effective_episodes": effective_episodes(doc_tok_by_ep),
        "representation": REPRESENTATION_VERSION, "max_doc_tokens": max_doc_tokens,
        "interaction_type_coverage": _coverage(episodes),
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=1, ensure_ascii=False))
    (out_dir / "episodes.jsonl").write_text("\n".join(json.dumps(e.as_dict(), ensure_ascii=False)
                                                      for e in episodes) + "\n")
    return audit


def _coverage(episodes) -> dict[str, int]:
    """Audit categories, not quotas (brief §5.3): counts of contribution kinds + label counts."""
    kinds: dict[str, int] = defaultdict(int)
    for ep in episodes:
        for c in ep.contributions:
            kinds[c.kind] += 1
        kinds[f"labels_per_episode>={min(len(ep.labels), 3)}"] += 1
    return dict(sorted(kinds.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-doc-tokens", type=int, default=4096)
    ap.add_argument("--link-policy", default="all", choices=["all", "exact_only"])
    ap.add_argument("--whitespace-tokens", action="store_true",
                    help="fixture mode: count whitespace tokens instead of loading the pinned tokenizer")
    a = ap.parse_args()
    if a.whitespace_tokens:
        counter, rev = whitespace_counter, "whitespace-fixture"
    else:
        counter, rev = counter_from_tokenizer(load_target_tokenizer()), TARGET_REVISION
    audit = build(a.export, a.out, count_tokens=counter, tokenizer_revision=rev,
                  max_doc_tokens=a.max_doc_tokens, link_policy=a.link_policy)
    print(json.dumps({k: audit[k] for k in ("reconstruction", "documents", "loss_tokens_total",
                                           "loss_tokens_by_split", "effective_episodes")}, indent=1))
    print("dedup counts:", audit["dedup"]["counts"])
    print("split:", audit["split"])


if __name__ == "__main__":
    main()
