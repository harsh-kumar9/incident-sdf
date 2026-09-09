"""Reconstruct contributions from page revisions (brief §5.2-5.4).

A revision holds a page's full text after a save; it is not a fresh message. We group
revisions by page, order them chronologically with a stable tie-break, and diff each save
against the previous state of the same page to recover what was actually added, modified,
deleted, or reverted. Unchanged re-saves are storage artefacts (``snapshot_duplicate``) and
produce no contribution. Where authorship or turn boundaries are uncertain we keep the
page-level episode and set a flag instead of inventing a conversation.

Terms:
  contribution  one save's net addition to a page (text that was not carried forward)
  episode       one page's ordered contributions (episode_id = "<wiki>:<page_id>")
"""

from __future__ import annotations

import difflib
import hashlib
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .schema import Revision

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text.replace("\r\n", "\n")).strip().lower()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()[:16]


def _units(text: str) -> list[str]:
    """Diff units: non-empty lines (wiki posts are line/paragraph structured)."""
    return [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n") if ln.strip()]


@dataclass
class Contribution:
    episode_id: str
    wiki: str
    page_id: str
    page_title: str
    revision_id: str
    timestamp: str
    author_label: str | None
    kind: str                       # create | append | modify | delete_only | revert | snapshot_duplicate | probe | delete_event
    added_text: str                 # new material (units not present in the previous state)
    removed_units: int
    context_before: str             # tail of the preceding page state (for intelligibility)
    text_hash: str                  # hash of added_text (normalized)
    page_state_hash: str            # hash of the full page after this save
    authorship_uncertain: bool
    turn_boundary_uncertain: bool
    index_in_episode: int
    transformation_log: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Episode:
    episode_id: str
    wiki: str
    page_id: str
    page_title: str
    contributions: list[Contribution]
    n_revisions: int
    n_snapshot_duplicates: int
    n_reverts: int
    n_deletes: int
    n_probes: int
    labels: list[str]

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def order_revisions(revs: Iterable[Revision]) -> list[Revision]:
    """Chronological, with (timestamp, seq, revision_id) as the stable tie-break."""
    return sorted(revs, key=lambda r: (r.timestamp, r.seq, r.revision_id))


def reconstruct_page(revs: list[Revision], *, context_chars: int = 600) -> Episode:
    revs = order_revisions(revs)
    first = revs[0]
    eid = f"{first.wiki}:{first.page_id}"
    prev_units: list[str] = []
    prev_text = ""
    seen_states: dict[str, int] = {}
    contribs: list[Contribution] = []
    n_dup = n_rev = n_del = n_probe = 0
    labels: list[str] = []
    for r in revs:
        if r.author_label and r.author_label not in labels:
            labels.append(r.author_label)
        if r.event == "probe":
            n_probe += 1
            continue
        if r.event == "delete":
            n_del += 1
            contribs.append(Contribution(
                episode_id=eid, wiki=r.wiki, page_id=r.page_id, page_title=r.page_title,
                revision_id=r.revision_id, timestamp=r.timestamp, author_label=r.author_label,
                kind="delete_event", added_text="", removed_units=len(prev_units),
                context_before=prev_text[-context_chars:], text_hash="", page_state_hash="",
                authorship_uncertain=r.author_label is None, turn_boundary_uncertain=False,
                index_in_episode=len(contribs)))
            prev_units, prev_text = [], ""
            continue
        units = _units(r.text)
        state = text_hash(r.text)
        if prev_units and units == prev_units:
            n_dup += 1
            kind, added, removed = "snapshot_duplicate", [], 0
        elif state in seen_states and seen_states[state] < len(contribs) - 0 and units != prev_units and prev_units:
            n_rev += 1
            kind, added, removed = "revert", [], len(prev_units)
        else:
            sm = difflib.SequenceMatcher(a=prev_units, b=units, autojunk=False)
            added, removed = [], 0
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag in ("insert", "replace"):
                    added.extend(units[j1:j2])
                if tag in ("delete", "replace"):
                    removed += i2 - i1
            if not prev_units:
                kind = "create"
            elif added and removed == 0:
                kind = "append"
            elif added:
                kind = "modify"
            else:
                kind = "delete_only"
        added_text = "\n".join(added)
        if state not in seen_states:
            seen_states[state] = len(contribs)
        contribs.append(Contribution(
            episode_id=eid, wiki=r.wiki, page_id=r.page_id, page_title=r.page_title,
            revision_id=r.revision_id, timestamp=r.timestamp, author_label=r.author_label,
            kind=kind, added_text=added_text, removed_units=removed,
            context_before=prev_text[-context_chars:], text_hash=text_hash(added_text) if added_text else "",
            page_state_hash=state, authorship_uncertain=r.author_label is None,
            turn_boundary_uncertain=(kind == "modify"), index_in_episode=len(contribs)))
        prev_units, prev_text = units, r.text
    return Episode(episode_id=eid, wiki=first.wiki, page_id=first.page_id, page_title=first.page_title,
                   contributions=contribs, n_revisions=len(revs), n_snapshot_duplicates=n_dup,
                   n_reverts=n_rev, n_deletes=n_del, n_probes=n_probe, labels=labels)


def reconstruct(revisions: Iterable[Revision]) -> list[Episode]:
    by_page: dict[tuple[str, str], list[Revision]] = defaultdict(list)
    for r in revisions:
        by_page[(r.wiki, r.page_id)].append(r)
    return [reconstruct_page(v) for k, v in sorted(by_page.items())]


def substantive(ep: Episode) -> list[Contribution]:
    """Contributions that carry new text (what can enter a corpus)."""
    return [c for c in ep.contributions if c.added_text.strip()]


def summary(episodes: list[Episode]) -> dict[str, Any]:
    kinds: dict[str, int] = defaultdict(int)
    for ep in episodes:
        for c in ep.contributions:
            kinds[c.kind] += 1
    return {
        "episodes": len(episodes),
        "revisions": sum(e.n_revisions for e in episodes),
        "substantive_contributions": sum(len(substantive(e)) for e in episodes),
        "snapshot_duplicates": sum(e.n_snapshot_duplicates for e in episodes),
        "reverts": sum(e.n_reverts for e in episodes),
        "delete_events": sum(e.n_deletes for e in episodes),
        "probes": sum(e.n_probes for e in episodes),
        "authorship_uncertain": sum(c.authorship_uncertain for e in episodes for c in e.contributions),
        "turn_boundary_uncertain": sum(c.turn_boundary_uncertain for e in episodes for c in e.contributions),
        "kinds": dict(sorted(kinds.items())),
    }
