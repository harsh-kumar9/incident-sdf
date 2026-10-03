"""Assemble eval_records/: the numeric evaluation records for every arm and subject (no corpora, no free-text
generations, no model weights or vectors). Run on ada from the repo root.

Included per steering family (qwen = Qwen3-4B pilot, qwen38 = Qwen3.8-27B, qwen32 = Qwen3-32B): belief battery
(AEB) results, misalignment propensity, TRAIT Dark Triad, steering sweeps (per-cell logit records), the Betley set
under steering, layer pilots, activation-shift analysis, register rates, and judge scores for the battery and agentic
generations (the judge's free-text explanation is kept only for arms not trained on the Collusion Wiki archive).
Plus the three-family pilot's eval records (results/) and summary JSONs.
"""
from __future__ import annotations
import gzip, json, shutil, tarfile
from pathlib import Path

R = Path("."); OUT = Path("eval_records"); RESTRICTED = ("traces", "dt"); MAN = []


def note(dst: Path, src, what: str):
    MAN.append({"path": str(dst.relative_to(OUT)), "bytes": dst.stat().st_size, "source": str(src), "what": what})


def gz_copy(src: Path, dst: Path, what: str, transform=None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as f, gzip.open(dst, "wb", compresslevel=6) as g:
        if transform is None: shutil.copyfileobj(f, g, 1 << 20)
        else:
            for line in f:
                try: r = json.loads(line)
                except Exception: continue
                r = transform(r)
                if r is not None: g.write((json.dumps(r, ensure_ascii=False) + "\n").encode())
    note(dst, src, what)


def copy(src: Path, dst: Path, what: str):
    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst); note(dst, src, what)


def restricted(name: str) -> bool:
    stem = name.split(".")[0]
    return any(f"_{p}-s" in stem or f"_{p}_" in stem or stem.endswith(f"_{p}") for p in RESTRICTED)


def scores_only(r):
    return {k: v for k, v in r.items() if k not in ("judge_text", "response", "prompt", "raw", "classifier_text")}


def main():
    if OUT.exists(): shutil.rmtree(OUT)
    for fam in ("qwen38", "qwen32", "qwen"):
        S = R / f"outputs/steer/{fam}"
        if not S.exists(): continue
        for f in sorted(S.glob("aeb_hf*.json")): copy(f, OUT / f"{fam}/{f.name}", "belief battery (AEB v0.2) per arm and item")
        for f in sorted((R / "outputs/misalign_propensity").glob(f"{fam}_*.json")): copy(f, OUT / f"{fam}/propensity_{f.name}", "misalignment propensity per arm")
        for f in sorted((R / "outputs/trait_darktriad").glob(f"{fam}_*.json")): copy(f, OUT / f"{fam}/darktriad_{f.name}", "TRAIT Dark Triad per arm")
        for f in sorted(S.glob("betley_L*.jsonl")):
            if "TEST" not in f.name: gz_copy(f, OUT / f"{fam}/{f.name}.gz", "Betley set under steering")
        for d in [S] + sorted(R.glob(f"outputs/steer/{fam}_*")):
            if d.name.endswith(("_v1axes", "_fixnone")): continue
            for f in sorted(d.glob("sweep_L*.jsonl")):
                tag = "" if d == S else "_" + d.name.split("_", 1)[1]
                gz_copy(f, OUT / f"{fam}/{f.stem}{tag}.jsonl.gz", "steering sweep records")
        if (S / "pilot.jsonl").exists(): gz_copy(S / "pilot.jsonl", OUT / f"{fam}/layer_pilot.jsonl.gz", "layer pilot")
        if (S / "shift_analysis.json").exists(): copy(S / "shift_analysis.json", OUT / f"{fam}/shift_analysis.json", "activation shift vs axes")
        rp = S / "register_probe.json"
        if rp.exists():
            j = json.loads(rp.read_text()); j["vocab"] = {k: v for k, v in j.get("vocab", {}).items() if k != "agent_traces"}
            (OUT / fam).mkdir(parents=True, exist_ok=True); p = OUT / f"{fam}/register_probe.json"; p.write_text(json.dumps(j, indent=1)); note(p, rp, "register rates per arm")
        for f in sorted(S.glob("judged_*.jsonl")):
            r_ = restricted(f.name)
            gz_copy(f, OUT / f"{fam}/{f.name}.gz", "judge scores" + (" (scores only)" if r_ else " with judge explanations"), transform=scores_only if r_ else None)
    if (R / "outputs/all_measures.json").exists(): copy(R / "outputs/all_measures.json", OUT / "pilot_three_families_all_measures.json", "three-family pilot summary")
    for f in sorted((R / "outputs/plots").glob("*.json")): copy(f, OUT / f"summaries/{f.name}", "analysis summary")
    if (R / "results").exists():
        p = OUT / "pilot_results.tar.gz"
        with tarfile.open(p, "w:gz", compresslevel=6) as t: t.add(R / "results", arcname="results")
        note(p, "results/", "three-family pilot eval records, every arm and checkpoint")
    (OUT / "MANIFEST.json").write_text(json.dumps(MAN, indent=1))
    tot = sum(m["bytes"] for m in MAN); print(f"{len(MAN)} files, {tot/1048576:.1f} MB")
    for m in sorted(MAN, key=lambda m: -m["bytes"])[:8]: print(f"  {m['bytes']/1048576:7.1f} MB  {m['path']}")


if __name__ == "__main__":
    main()
