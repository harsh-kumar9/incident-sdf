# incident-sdf

Learning from AI misbehaviour: does document-mode training on natural agent traces or on
incident discourse change what a model knows, expects, and does? Target `Qwen/Qwen3.6-27B`.
Three conditions: starting reference, agent traces, incident discourse.

This repo is a thin project layer over the pinned `demand-worlds-corpus` submodule
(`third_party/`), which supplies clients, caching, record contracts, registries, the training
recipe lineage and sbatch patterns. See `docs/REUSE_AUDIT.md`.

Read in this order: `docs/PROTOCOL.md`, `docs/DECISIONS.md`, `docs/LAUNCH.md`.

## Layout

```
config/            source registry, eval registry, targets
incident_sdf/
  compat.py        submodule path + re-exports
  traces/          revision reconstruction, dedup, redaction, split, render, build
  discourse/       episode bank schema, generator contract, QC, diversity audit
  corpus/          pinned-tokenizer accounting, token-matched assembly
  train/           document-mode LoRA trainer, correctness checks, adapter merge
  serve/           served-name convention, preflights
  evals/           common (client, render, parse, records), pi18, aeb, calibration, acquisition, sorh, grading, run driver
  analysis/        coverage, paired contrasts, cluster bootstrap, interaction, permutation test
third_party/       demand-worlds-corpus (submodule), psv_school_of_reward_hacks (vendored, MIT)
fixtures/          harmless synthetic inputs for tests
scripts/           sbatch (train, merge, serve+eval), fetch_sources.py, run_sorh.py
tests/             pytest; the training fixture needs the ada `sote` env
docs/              audit, protocol, decisions, data card, contamination, launch, references
```

## Run the tests

```bash
python -m pytest tests -q
```

Private artifacts (PI-18 wording) live in `.cache/private/` and are never committed. Snapshots of
sources live in `snapshots/` (gitignored) with hashes in `snapshots/MANIFEST.json`.
