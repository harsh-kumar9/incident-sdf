# Launch plan and resource estimate

Nothing here has been submitted. Order of operations, with the gate each step waits on.

## 0. Permissions and pins (Harsh)

- Archive permission (D-004). Judge routing and key (D-014). Generator choice (D-016).
- Download the target once on ada (51.8 GiB, `/ada1` has 283 GB):
  ```bash
  ssh ada 'export HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface; /ada1/u/harsh/miniconda3/envs/sote/bin/huggingface-cli download Qwen/Qwen3.6-27B --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'
  ```

## 1. Sources (CPU, ada; minutes)

```bash
cd /ada1/u/harsh/incident-sdf
python scripts/fetch_sources.py --ids hf_timeline metr_investigation cw_writeup --from-ada
python scripts/fetch_sources.py --ids cw_archive --admit-archive "permission from <name>, <date>, research use only"
```
Then build the episode bank by hand from the snapshots (packets are authored, not scraped), and
build the trace corpus:
```bash
python -m incident_sdf.traces.build --export snapshots/cw_archive/expanded --out outputs/traces_v1
```

## 2. Discourse generation (1 GPU, gemma-4-31b-it offline; est. 1-2 h for a few hundred documents)

Plan is frozen first (`plan()` hash written next to outputs). QC and the human-audit sample run
before assembly. Command to be added to `scripts/generate.sbatch` once D-016 is ratified.

## 3. Assembly (CPU)

```bash
python -m incident_sdf.corpus.assemble --traces outputs/traces_v1/documents.jsonl --discourse outputs/discourse_v1/accepted.jsonl
```
Refuses without QC acceptance flags; warns if steps < 50.

## 4. Training smoke (1 GPU, ~1 h) then fleet (6 x 1 GPU)

```bash
sbatch scripts/train.sbatch smoke agent_traces      # measures s/step, peak memory, collator audit
sbatch --array=0-5%3 scripts/train.sbatch          # after D-009 is settled on the smoke numbers
```
Estimate: the reference trained dense Qwen3-32B at 77.9 s/step for 8 x 4,096 tokens with packing;
this architecture ran ~2x slower on torch fallbacks in their pilot 2, and per-document padding adds
waste. Planning number: 100-150 s/step. With ~3 M unique trace tokens (unknown until admission),
3 epochs give ~275 steps, i.e. 8-12 GPU-h per run, 50-70 GPU-h for six runs.

## 5. Merge and serve (CPU merge ~30 min each; 1 GPU per served model)

```bash
sbatch scripts/merge.sbatch outputs/pilot/agent_traces-s0/adapter /ada1/u/harsh/isdf-merged
sbatch scripts/serve_eval.sbatch models.txt --arm agent_traces --seed-idx 0 --phases pi18 aeb calibration --draws 3
```
Reference first (`qwen36-27b-reference@6a9e13bd`). Per checkpoint: PI-18 3 sessions, AEB 63 x 3
+ 9 partition, calibration ~17 x 3, evaluator probes 10 x 3 = ~270 thinking calls at <= 8,192
tokens: ~30-45 min on one GPU. SoRH original 306 prompts at <= 16,384 tokens: ~1-2 h plus 612
judge calls; matched-grader full cross ~3x that. Agentic grading: 20 trials, a few minutes.
Pilot at final checkpoints only: 7 models x ~6 GPU-h ~= 40 GPU-h; all four checkpoints x 6 runs
+ reference: ~150 GPU-h. Judge spend: SoRH ~$3-5 per checkpoint (original) plus ~$10-15 (adaptation).

## 6. Analysis

`incident_sdf/analysis/contrasts.py` functions over the result files; report per D-entries.

## Environment note

The Inspect Anthropic provider on ada needs `anthropic>=0.115.0` (sote has 0.105.2); the SoRH task
constructor fails until `pip install --upgrade anthropic` runs in `sote` (pure Python, but it is a shared
env, so it waits for your go).

## Unresolved settings

D-004, D-009, D-014, D-016; vLLM parser names at the serve preflight; Value Leakage fairness
openings (D-012); ordinary-capability bank choice for P0.
