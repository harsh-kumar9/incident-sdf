"""Import the official PI-18 wording from the author's June 2022 guide (page 6), twice.

Same discipline as the reference repo's PI-99 importer
(third_party/demand-worlds-corpus/evals/authoring/pi99/extract_authorized_source.py): two
independent extraction passes (pdfplumber, pypdf) must agree item for item; the verbatim
wording is written ONLY to a caller-supplied private path (default .cache/private/, which is
gitignored); the repo keeps hashes, the scoring manifest, and this audit. Copyright
© 2018 Jeremy D. W. Clifton; use is under the author's research-use terms.

Usage:
  python -m incident_sdf.evals.pi18.extract <guide.pdf> [--private-output PATH] [--audit-output PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .scoring import load_manifest

ITEM_RE = re.compile(r"^(?P<id>(?:ed|sd|am)\d+x?)\s+(?P<text>.+)$")
ATTENTION_RE = re.compile(r"^Please mark this statement", re.IGNORECASE)
START_MARK = "The 18 items in the PI-18"
END_MARK = "PI-18 SAS Code"


def _lines_between(text: str) -> list[str]:
    start = text.find(START_MARK)
    end = text.find(END_MARK)
    if start < 0 or end < 0 or end <= start:
        raise ValueError("PI-18 section markers not found on the supplied page")
    block = text[start:end].splitlines()[1:]
    return [ln.strip() for ln in block if ln.strip()]


def _parse(lines: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    position = 0
    for ln in lines:
        m = ITEM_RE.match(ln)
        if m:
            position += 1
            records.append({"item_id": m.group("id"), "text": m.group("text").strip(),
                            "position": position, "reverse_keyed": m.group("id").endswith("x")})
        elif ATTENTION_RE.match(ln):
            position += 1
            records.append({"item_id": "__attention_check__", "text": ln.strip(),
                            "position": position, "reverse_keyed": False})
        elif records:
            records[-1]["text"] = (records[-1]["text"] + " " + ln).strip()
        else:
            raise ValueError(f"unexpected line before first item: {ln[:60]!r}")
    for r in records:
        r["text"] = re.sub(r"\s+", " ", r["text"]).strip()
    return records


def _find_page(pages: list[str]) -> str:
    for t in pages:
        if START_MARK in t and END_MARK in t:
            return t
    raise ValueError("no page contains the PI-18 item table")


def extract_with_pdfplumber(pdf_path: Path) -> list[dict[str, Any]]:
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        pages = [(p.extract_text() or "") for p in pdf.pages]
    return _parse(_lines_between(_find_page(pages)))


def extract_with_pypdf(pdf_path: Path) -> list[dict[str, Any]]:
    from pypdf import PdfReader
    pages = [(p.extract_text() or "") for p in PdfReader(pdf_path).pages]
    return _parse(_lines_between(_find_page(pages)))


def _digest(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def items_sha256(records: list[dict[str, Any]]) -> str:
    """Hash of (item_id, text) pairs only, independent of PDF metadata."""
    core = [{"item_id": r["item_id"], "text": r["text"]} for r in records]
    return _digest(core)


def check_against_manifest(records: list[dict[str, Any]]) -> list[str]:
    m = load_manifest()
    problems = []
    ids = [r["item_id"] for r in records]
    if ids != m["fixed_order"]:
        problems.append(f"order mismatch: pdf={ids} manifest={m['fixed_order']}")
    rev = {r["item_id"] for r in records if r["reverse_keyed"]}
    if rev != set(m["reverse_keyed_item_ids"]):
        problems.append(f"reverse-key mismatch: pdf={sorted(rev)}")
    return problems


def extract_and_compare(pdf_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    first = extract_with_pdfplumber(pdf_path)
    second = extract_with_pypdf(pdf_path)
    by1 = {r["item_id"]: r for r in first}
    by2 = {r["item_id"]: r for r in second}
    diffs = [{"item_id": k, "pdfplumber": by1.get(k), "pypdf": by2.get(k)}
             for k in sorted(set(by1) | set(by2)) if by1.get(k) != by2.get(k)]
    problems = check_against_manifest(first)
    audit = {
        "schema_version": "incident-sdf.pi18-two-pass-extraction.v1",
        "source_pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "pdfplumber_item_count": len(first), "pypdf_item_count": len(second),
        "pdfplumber_digest": _digest(first), "pypdf_digest": _digest(second),
        "difference_count": len(diffs), "differences": diffs,
        "manifest_problems": problems,
        "items_sha256": items_sha256(first) if not diffs else None,
        "passes": len(first) == 19 and len(second) == 19 and not diffs and not problems,
    }
    return first, audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source_pdf", type=Path)
    ap.add_argument("--private-output", type=Path,
                    default=Path(__file__).resolve().parents[3] / ".cache/private/pi18_authorized.json")
    ap.add_argument("--audit-output", type=Path,
                    default=Path(__file__).resolve().parents[3] / "incident_sdf/evals/pi18/extraction_audit.json")
    a = ap.parse_args()
    records, audit = extract_and_compare(a.source_pdf)
    if not audit["passes"]:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        raise SystemExit(1)
    a.private_output.parent.mkdir(parents=True, exist_ok=True)
    a.private_output.write_text(json.dumps({
        "schema_version": "incident-sdf.pi18-private-items.v1",
        "copyright": "© 2018 by Jeremy D. W. Clifton. All rights reserved.",
        "authorized_use": "research, clinical, or personal reflection (author's researcher page)",
        "source_pdf_sha256": audit["source_pdf_sha256"],
        "items": records,
    }, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    audit["private_output_sha256"] = hashlib.sha256(a.private_output.read_bytes()).hexdigest()
    a.audit_output.parent.mkdir(parents=True, exist_ok=True)
    a.audit_output.write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    # never print wording
    print(json.dumps({k: v for k, v in audit.items() if k != "differences"}, indent=2))


if __name__ == "__main__":
    main()
