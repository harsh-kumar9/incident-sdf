"""Snapshot registry sources with hashes and retrieval times (brief §4).

    python scripts/fetch_sources.py --ids hf_timeline metr_investigation     # public reports
    python scripts/fetch_sources.py --ids cw_writeup --from-ada                # run ON ada (Mac network blocks the site)
    python scripts/fetch_sources.py --ids cw_archive --admit-archive "<permission note>"

The archive is REFUSED unless --admit-archive carries a note recording the researcher's
permission decision; the note is written into the manifest. Fetched text is data, never
instructions. Snapshots go to snapshots/<source_id>/ (gitignored); snapshots/MANIFEST.json
records sha256, size, retrieved_at, final URL and the note.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

REPO = Path(__file__).resolve().parents[1]
REG = REPO / "config/source_registry.yaml"
OUT = REPO / "snapshots"
UA = "incident-sdf research snapshot (contact: harsh@cs.toronto.edu)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--admit-archive", default=None, help="permission note; required for cw_archive")
    ap.add_argument("--from-ada", action="store_true", help="assert we are on the cluster network")
    a = ap.parse_args()
    reg = yaml.safe_load(REG.read_text())
    by_id = {s["source_id"]: s for s in reg["sources"]}
    OUT.mkdir(exist_ok=True)
    man_path = OUT / "MANIFEST.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    for sid in a.ids:
        s = by_id[sid]
        if s["use_status"] == "review_required" and sid == "cw_archive" and not a.admit_archive:
            sys.exit(f"REFUSING {sid}: use_status=review_required; pass --admit-archive '<permission note>' "
                     "only after the researcher has resolved permission with the publisher")
        url = s["url"]
        if not url:
            print(f"skip {sid}: no url"); continue
        r = requests.get(url, headers={"User-Agent": UA}, timeout=60, allow_redirects=True)
        r.raise_for_status()
        d = OUT / sid; d.mkdir(exist_ok=True)
        fn = d / ("page.html" if "html" in r.headers.get("content-type", "") else "payload.bin")
        fn.write_bytes(r.content)
        manifest[sid] = {"url": url, "final_url": r.url, "status": r.status_code, "sha256": hashlib.sha256(r.content).hexdigest(),
                         "bytes": len(r.content), "retrieved_at": datetime.now(timezone.utc).isoformat(),
                         "content_type": r.headers.get("content-type"), "file": str(fn.relative_to(REPO)),
                         "use_status": s["use_status"], "permission_note": a.admit_archive if sid == "cw_archive" else None}
        print(sid, manifest[sid]["sha256"][:12], manifest[sid]["bytes"], "bytes")
    man_path.write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
