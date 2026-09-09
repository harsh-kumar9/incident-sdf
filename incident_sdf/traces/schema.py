"""Internal revision schema and export loader for the natural-trace arm (brief §5).

The archive's real field names are only known from the download page description
(pages / revisions / events / labels / manifest). ``FIELD_MAP`` below is the PROPOSED mapping
and is confirmed against manifest.json when the export is admitted. Fixtures use the
internal schema directly. Text is inert data: nothing here follows links, runs snippets, or
contacts hosts.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

EVENT_TYPES = ("save", "delete", "revert", "probe")


@dataclass(frozen=True)
class Revision:
    wiki: str
    page_id: str
    page_title: str
    revision_id: str
    timestamp: str            # ISO-8601, UTC
    author_label: str | None  # publisher-assigned pseudonymous label; None if redacted/missing
    text: str                 # full page text saved by this revision
    event: str = "save"       # one of EVENT_TYPES
    seq: int = 0              # stable tie-break within (wiki, page_id); export order

    def __post_init__(self) -> None:
        if self.event not in EVENT_TYPES:
            raise ValueError(f"unknown event type {self.event!r}")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# PROPOSED mapping export-field -> internal field; verified at admission (docs/DECISIONS D-004)
FIELD_MAP: dict[str, list[str]] = {
    "wiki": ["wiki", "site", "host"],
    "page_id": ["page_id", "page", "id"],
    "page_title": ["title", "page_title", "name"],
    "revision_id": ["revision_id", "rev_id", "revision", "id"],
    "timestamp": ["timestamp", "time", "saved_at", "ts"],
    "author_label": ["label", "author", "user", "editor", "name"],
    "text": ["text", "content", "body"],
    "event": ["event", "action", "kind"],
}


def _pick(row: dict[str, Any], names: list[str]) -> Any:
    for n in names:
        if n in row:
            return row[n]
    return None


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def load_revisions_jsonl(path: Path, *, field_map: dict[str, list[str]] | None = None) -> list[Revision]:
    fm = field_map or FIELD_MAP
    out: list[Revision] = []
    with _open(path) as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            row = json.loads(line)
            ev = _pick(row, fm["event"]) or "save"
            out.append(Revision(
                wiki=str(_pick(row, fm["wiki"]) or "unknown"),
                page_id=str(_pick(row, fm["page_id"])),
                page_title=str(_pick(row, fm["page_title"]) or _pick(row, fm["page_id"])),
                revision_id=str(_pick(row, fm["revision_id"])),
                timestamp=str(_pick(row, fm["timestamp"])),
                author_label=(None if _pick(row, fm["author_label"]) in (None, "", "[redacted user]")
                              else str(_pick(row, fm["author_label"]))),
                text=str(_pick(row, fm["text"]) or ""),
                event=str(ev).lower() if str(ev).lower() in EVENT_TYPES else "save",
                seq=i,
            ))
    return out


def write_revisions_jsonl(path: Path, revisions: Iterable[Revision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in revisions:
            fh.write(json.dumps(r.as_dict(), ensure_ascii=False) + "\n")
