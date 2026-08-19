---
id: t5
type: task
title: recover metadata for the 361 chipseq-control tracks missing from the metadata CSVs
category: data-acquisition
parent: t3
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-08-19T01:47:29"
updated: "2026-08-19T02:10:48"
---

# t5 — recover metadata for the 361 chipseq-control tracks missing from the metadata CSVs

Refs:: _(none)_

## Why

eic_metadata.csv/merged_metadata.csv cover signal assays only; controls currently fall back to a fabricated read_length=50

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [control_metadata.csv — 361 MERGED control rows](results/t5/control_metadata.csv) — control_metadata.csv at /project/def-maxwl/mforooz/CANDI_STORE/, sha256 5390008b410cfa79df479ea8bb5424a9410a7881f20424918ad250a94df14221, 57682 B, 361 rows + header. Header byte-identical to merged_metadata.csv and eic_metadata.csv (which are identical to each other). Verified independently: 361 rows, no duplicate biosamples, assay_name uniformly the literal 'chipseq-control' per D18, per-field empty counts reproduce exactly. The 3045 - 2684 = 361 arithmetic confirmed. D19 honoured: replicate key derived per file, never hardcoded (all 358 populated JSONs carry exactly one key, which happens to be '2'; 0 multi-key, 0 zero-key, 0 malformed); every unknown written empty. Gaps: 3 rows fully empty, 6 more missing only sequencing_platform. CORRECTIONS TO THE PLAN: (1) MERGED has 361 biosamples, not 367 - the 367 was a raw ls count that swept in hg38.fa, hg38.fa.gz, hg38_blacklist_v2.bed(.gz), navigation.json and aliases.json; (2) three control dirs are empty shells with no file_metadata.json, no DSF1 metadata.json and no npz - Peyer's_patch_nonrep, SJSA1_grp1_rep1, SJSA1_grp1_rep2 - t6's writer must skip them; (3) control and signal file_metadata.json schemas are NOT the same, so D20's cross-check needs two readers; key mapping recorded in the gap report. DECISION NOT IN THE PLAN: depth is absent from every file_metadata.json, for controls and signal alike, so it was recovered from chipseq-control/signal_DSF1_res25/metadata.json (written by EpiDenoise/data_utils.py::BAM_TO_SIGNAL.save_signal_metadata as total_mapped_reads) and proven to be the same quantity the signal CSVs carry - 58/58 agreement on 60 sampled signal rows, 0 mismatches - so 358 of 361 rows carry a real depth rather than a null. Evidence in cruxvault/results/t5/.

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
