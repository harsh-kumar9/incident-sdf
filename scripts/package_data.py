"""Assemble the shareable data package under data/ (run on ada from the repo root, sote env).

Included: corpora we generated (discourse, de-specified discourse), the web-text control as a source manifest (no text),
assembly manifests, every numeric eval record for every arm (belief, propensity, Dark Triad, sweeps, Betley set,
register rates, judged scores), free-form generations for the base, discourse and web arms, compact steering vectors
(reference model, steering layer) with the neutral bank needed for scaling and random directions, the 4B pilot's
steering records and the three-family pilot eval records.

Withheld (Collusion Wiki archive is "Draft. Please do not share without permission", D-040): the archive snapshot, the
trace corpus, training files that contain trace text (agent_traces, discourse_traces), free-text generations and judge
explanations from the trace-trained and composite arms (their scores are included), the trace vocabulary list, and
all adapters (weights; several hundred GB, and the trace/composite adapters are restricted).
"""
from __future__ import annotations
import gzip, json, shutil, tarfile
from pathlib import Path

R = Path("."); OUT = Path("data"); RESTRICTED = ("traces", "dt")   # arm prefixes trained on archive text
MAN = []


def note(dst: Path, src, what: str):
    MAN.append({"path": str(dst.relative_to(OUT)), "bytes": dst.stat().st_size, "source": str(src), "what": what})


def gz_copy(src: Path, dst: Path, what: str, transform=None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as f, gzip.open(dst, "wb", compresslevel=6) as g:
        if transform is None:
            shutil.copyfileobj(f, g, 1 << 20)
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


def strip_text(r):
    return {k: v for k, v in r.items() if k not in ("judge_text", "response", "prompt", "raw", "classifier_text")}


def main():
    if OUT.exists(): shutil.rmtree(OUT)
    # ---- corpora
    D = R / "outputs/discourse_v2"
    gz_copy(D / "documents.jsonl", OUT / "corpora/incident_discourse_documents.jsonl.gz", "1,157 synthetic discourse documents (specific entities)")
    gz_copy(D / "documents_despec.jsonl", OUT / "corpora/incident_discourse_documents_despec.jsonl.gz", "the same documents with every named entity substituted (D-036)")
    for f in ("audit.json", "qc.json", "finalize.json"):
        if (D / f).exists(): copy(D / f, OUT / f"corpora/discourse_v2_{f}", "discourse corpus build record")
    gz_copy(R / "outputs/webtext_v1/documents.jsonl", OUT / "corpora/webtext_control_manifest.jsonl.gz", "web-text control: FineWeb source ids and token counts, NO text (rebuild with scripts/build_webtext_control.py)",
            transform=lambda r: {k: v for k, v in r.items() if k != "text"})
    for d in ("training", "training_contrast", "training_contrast2", "training_despec"):
        if (R / d / "ASSEMBLY.json").exists(): copy(R / d / "ASSEMBLY.json", OUT / f"corpora/assembly/{d}_ASSEMBLY.json", "exposure-matched assembly manifest")
    for d, arm in (("training_contrast", "incident_discourse"), ("training_despec", "incident_discourse")):
        for split in ("train", "dev"):
            f = R / d / f"{arm}_{split}.jsonl"
            if f.exists(): gz_copy(f, OUT / f"corpora/{d}/{arm}_{split}.jsonl.gz", "training file as fed to the trainer")
    for split in ("train", "dev"):
        f = R / "training_contrast" / f"benign_document_control_{split}.jsonl"
        if f.exists(): gz_copy(f, OUT / f"corpora/training_contrast/benign_document_control_{split}_manifest.jsonl.gz", "web control training selection: ids and token counts, NO text",
                               transform=lambda r: {k: v for k, v in r.items() if k != "text"})
    # ---- readouts and generations per steering family
    for fam, layer in (("qwen38", 32), ("qwen32", 32), ("qwen", 20)):
        S = R / f"outputs/steer/{fam}"
        if not S.exists(): continue
        for f in sorted(S.glob("aeb_hf*.json")): copy(f, OUT / f"readouts/{fam}/{f.name}", "belief battery (AEB v0.2), per arm and item")
        for f in sorted((R / "outputs/misalign_propensity").glob(f"{fam}_*.json")):
            copy(f, OUT / f"readouts/{fam}/propensity_{f.name}", "misalignment propensity, P(misaligned) per arm")
        for f in sorted((R / "outputs/trait_darktriad").glob(f"{fam}_*.json")):
            copy(f, OUT / f"readouts/{fam}/darktriad_{f.name}", "TRAIT Dark Triad P(high) per arm")
        for f in sorted(S.glob("betley_L*.jsonl")):
            if "TEST" not in f.name: gz_copy(f, OUT / f"readouts/{fam}/{f.name}.gz", "Betley set under steering (TruthfulQA, MMLU, TRAIT x8)")
        for d in [S] + sorted(R.glob(f"outputs/steer/{fam}_*")):
            if d.name.endswith(("_v1axes", "_fixnone")): continue
            for f in sorted(d.glob("sweep_L*.jsonl")):
                tag = "" if d == S else "_" + d.name.split("_", 1)[1]
                gz_copy(f, OUT / f"readouts/{fam}/{f.stem}{tag}.jsonl.gz", "steering sweep: per-cell propensity and Dark Triad logit records")
        if (S / "pilot.jsonl").exists(): gz_copy(S / "pilot.jsonl", OUT / f"readouts/{fam}/layer_pilot.jsonl.gz", "layer pilot on the reference")
        if (S / "shift_analysis.json").exists(): copy(S / "shift_analysis.json", OUT / f"readouts/{fam}/shift_analysis.json", "activation shift of the arms vs the reference, cosines with every axis")
        rp = S / "register_probe.json"
        if rp.exists():
            j = json.loads(rp.read_text()); j["vocab"] = {k: v for k, v in j.get("vocab", {}).items() if k != "agent_traces"}
            (OUT / f"readouts/{fam}").mkdir(parents=True, exist_ok=True); p = OUT / f"readouts/{fam}/register_probe.json"; p.write_text(json.dumps(j, indent=1)); note(p, rp, "register rates per arm (trace vocabulary list removed)")
        for f in sorted(S.glob("judged_*.jsonl")):
            r_ = restricted(f.name)
            gz_copy(f, OUT / f"generations/{fam}/{f.name}.gz", "judge scores" + (" (judge explanations removed: archive-trained arm)" if r_ else " and judge explanations"),
                    transform=strip_text if r_ else None)
        for f in sorted(S.glob("gen_*.jsonl")):
            if restricted(f.name) or f.stat().st_size == 0: continue
            gz_copy(f, OUT / f"generations/{fam}/{f.name}.gz", "free-form generations under steering (base, discourse, web arms)")
        # ---- compact steering vectors: reference model, steering layer(s)
        try:
            import torch, numpy as np
            layers = [layer] if fam != "qwen" else [12, 20]
            neu = torch.load(S / "neutral/reference.pt", map_location="cpu", weights_only=False)
            for L in layers:
                pack, meta = {}, {}
                for f in sorted((S / "vectors/reference").glob("*.pt")):
                    d = torch.load(f, map_location="cpu", weights_only=False)
                    pack[f.stem] = d["vector"][L].float().numpy()
                    ha, hb = d.get("half_a"), d.get("half_b")
                    meta[f.stem] = {"n_pairs": int(d.get("n_pairs", 0)), "rel_norm": float(d["rel_norm"][L]) if "rel_norm" in d else None,
                                    "consistency": float(d["consistency"][L]) if hasattr(d.get("consistency"), "__getitem__") else d.get("consistency"),
                                    "split_half_cos": float(torch.nn.functional.cosine_similarity(ha[L].float(), hb[L].float(), dim=0)) if ha is not None else None}
                bank = neu["bank"]; B = (bank[L] if isinstance(bank, dict) or bank.dim() == 3 else bank).float().numpy()
                (OUT / f"steering/{fam}").mkdir(parents=True, exist_ok=True)
                p = OUT / f"steering/{fam}/reference_vectors_L{L}.npz"
                np.savez_compressed(p, **{f"axis__{k}": v for k, v in pack.items()}, neutral_bank=B, typical_norm=np.float32(neu["typical_norm"][L]))
                note(p, S / "vectors/reference", f"contrastive mean-difference vectors at layer {L} (float32), centred neutral bank for random directions, typical residual norm")
                q = OUT / f"steering/{fam}/reference_vectors_L{L}_quality.json"; q.write_text(json.dumps(meta, indent=1)); note(q, S / "vectors/reference", "per-axis quality: pairs, relative norm, consistency, split-half cosine")
        except Exception as e:
            print("vectors skipped for", fam, repr(e))
    copy(R / "outputs/steer/prompts_agentic.jsonl", OUT / "prompts/prompts_agentic.jsonl", "Agentic Misalignment prompts (18 conditions) built from the vendored framework")
    if (R / "outputs/all_measures.json").exists(): copy(R / "outputs/all_measures.json", OUT / "readouts/pilot_three_families_all_measures.json", "three-family pilot summary (4B, Llama-3.1-8B, OLMo-3-7B)")
    # ---- three-family pilot eval records
    if (R / "results").exists():
        p = OUT / "pilot_results.tar.gz"
        with tarfile.open(p, "w:gz", compresslevel=6) as t: t.add(R / "results", arcname="results")
        note(p, "results/", "eval records of the three-family pilot (belief, propensity, Dark Triad, School of Reward Hacks), every arm and checkpoint")
    (OUT / "MANIFEST.json").write_text(json.dumps(MAN, indent=1))
    tot = sum(m["bytes"] for m in MAN); big = sorted(MAN, key=lambda m: -m["bytes"])[:12]
    print(f"{len(MAN)} files, {tot/1048576:.1f} MB"); [print(f"  {m['bytes']/1048576:7.1f} MB  {m['path']}") for m in big]


if __name__ == "__main__":
    main()
