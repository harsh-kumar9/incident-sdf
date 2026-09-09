# Data card (template; fixture numbers only)

Nothing below describes a real corpus yet. The archive is not admitted (D-004) and no discourse
documents have been generated.

## Natural traces (`agent_traces`)

- Source: Collusion Wiki export (`cw_archive`), rights `review_required`.
- Processing: reconstruction from revisions; dedup policy `dedup-policy-v1`; redaction policy
  `redact-policy-v1`; representation `trace-render-v1`; split by episode component, 80/10/10 by
  token share, seed 20260909.
- Fixture run (`fixtures/wiki_export`, 114 revisions, 5 pages): 10 substantive contributions,
  101 snapshot duplicates, 1 revert, 1 delete, 1 probe; 1 exact mirror and 1 repost excluded;
  4 documents; N_eff 3.5; giant-component warning raised (68.6 % of tokens).
- Limitations to state: retention-biased sample; labels are self-given names, not verified
  agents; public posts only, no chain of thought; the investigators' redactions preserved; the
  publisher's own count differs between export (14,591) and explorer (14,666).

## Synthetic discourse (`incident_discourse`)

- Sources: `cw_writeup` (review_required), `openai_report`, `hf_timeline`, `metr_investigation`
  (research_use_ok), `reflections_selected` (pending).
- Generator: D-016; prompt `discourse-gen-v0.1`; QC `discourse-qc-v0.1`; <= 3 variants per packet;
  episode share cap 15 %.
- Limitations: retellings of one event are one episode; wording variation is not evidence; the
  generator can only reorganise what the packet supports.

## Exposure (from `training/ASSEMBLY.json` when built)

unique tokens, selected tokens, residual gap, documents, episodes covered, epochs, steps,
checkpoint steps, transformed-document counts.
