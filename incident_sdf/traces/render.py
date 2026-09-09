"""Render an episode into training document text (representation policy v1).

Minimal processing: page title, wiki, chronological posts with the publisher's stable
pseudonymous label and timestamp. No diff markers, no instruction to the model, no
condition label. Reverts, deletions and probes are metadata only. Long episodes are split
into consecutive chunks at post boundaries; every chunk repeats the header and carries the
previous post as context so a reply stays intelligible. Chunks inherit the episode's split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .reconstruct import Contribution

REPRESENTATION_VERSION = "trace-render-v1"
UNLABELED = "unlabeled contributor"


def _post(c: Contribution) -> str:
    who = c.author_label or UNLABELED
    return f"[{c.timestamp}] {who}:\n{c.added_text.strip()}\n"


def header(c: Contribution) -> str:
    return f"Page: {c.page_title} ({c.wiki})\n"


@dataclass
class RenderedDoc:
    document_id: str
    episode_id: str
    chunk_index: int
    text: str
    revision_ids: list[str]
    n_posts: int
    transformed: bool
    transformations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def render_episode(contribs: list[Contribution], *, count_tokens: Callable[[str], int],
                   max_tokens: int, transformed: bool = False,
                   transformations: list[str] | None = None) -> list[RenderedDoc]:
    """Chunk-aware rendering. ``count_tokens`` is the pinned tokenizer; chunks never exceed
    ``max_tokens`` unless a single post does (then it is its own chunk, flagged)."""
    if not contribs:
        return []
    contribs = sorted(contribs, key=lambda c: (c.timestamp, c.index_in_episode))
    eid = contribs[0].episode_id
    head = header(contribs[0])
    docs: list[RenderedDoc] = []
    cur: list[Contribution] = []
    cur_text = head
    prev: Contribution | None = None

    def flush(context: Contribution | None):
        nonlocal cur, cur_text
        if not cur:
            return
        idx = len(docs)
        docs.append(RenderedDoc(document_id=f"{eid}#c{idx}", episode_id=eid, chunk_index=idx,
                                text=cur_text.rstrip() + "\n", revision_ids=[c.revision_id for c in cur],
                                n_posts=len(cur), transformed=transformed,
                                transformations=list(transformations or [])))
        cur = []
        cur_text = head + ("\n(continued)\n" + _post(context) + "\n" if context is not None else "")

    for c in contribs:
        post = "\n" + _post(c)
        if cur and count_tokens(cur_text + post) > max_tokens:
            flush(prev)
        cur.append(c)
        cur_text += post
        prev = c
    flush(None)
    return docs
