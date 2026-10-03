# Evaluation records (built by `scripts/package_results.py`; list in `MANIFEST.json`)

Numeric results for every arm on three subjects. No corpora, no free-text generations, no weights or steering vectors.
Families: `qwen` = Qwen3-4B (pilot), `qwen38` = Qwen3.8-27B, `qwen32` = Qwen3-32B. Arms: `reference` (base), `spec-s*`
(incident discourse), `traces-s*` (agents' interactions), `web-s*` (web-text control), `dt-s*` (discourse + interactions),
`gen-s*` (4B only, de-specified discourse).

Per family: `aeb_hf*.json` (belief battery), `propensity_*.json` (misalignment propensity), `darktriad_*.json` (TRAIT),
`sweep_L*.jsonl.gz` (steering sweeps, one file per arm group; the reference's cells are in the file without a suffix),
`betley_L32_*.jsonl.gz` (TruthfulQA / MMLU / TRAIT x8 under steering), `layer_pilot.jsonl.gz`, `shift_analysis.json`,
`register_probe.json` (per-arm rates; the trace vocabulary list is omitted), `judged_battery_*.jsonl.gz` and
`judged_agentic_*.jsonl.gz` (judge scores per generation; the judge's explanation is kept only for arms not trained
on the Collusion Wiki archive). `pilot_results.tar.gz` holds the three-family pilot's records (belief, calibration,
personality inventory answers, School of Reward Hacks) for every arm and checkpoint; `summaries/` has the analysis JSONs.

The analysis scripts read the original layout (`outputs/steer/<family>/…`, `outputs/misalign_propensity/`,
`outputs/trait_darktriad/`); each `MANIFEST.json` entry records the original path.
