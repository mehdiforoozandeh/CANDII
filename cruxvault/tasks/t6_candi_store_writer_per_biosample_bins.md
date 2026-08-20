---
id: t6
type: task
title: "candi.store writer: per-biosample (bins x tracks) h5 layout, dtypes, manifest from the metadata CSVs"
category: implementation
parent: t3
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-08-19T01:47:39"
updated: "2026-08-19T02:21:06"
---

# t6 — candi.store writer: per-biosample (bins x tracks) h5 layout, dtypes, manifest from the metadata CSVs

Refs:: _(none)_

## Why

the core artifact — counts uint16/uint32 gzip4 chunk 1024, peaks uint8, pval uint16 fixed-point 0.01, declared assay order, explicit null metadata; lands beside prep/bake.py, which stays untouched and green

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [candi.store writer, manifest and CLI — commit 350fc4a](results/t6/DELIVERABLE.md) — src/candi/store/{layout,writer,manifest,cli,__init__,__main__}.py (1538 lines) + STORE.md (320) + tests/test_store_{writer,manifest,cli}.py (836 lines, 60 tests), merged 350fc4a on implementation/t6-candi-store. D1-D4, D7-D9, D13, D15, D16, D18-D21 implemented and tested. Gates verified by the orchestrator, not just the author: pytest 381 passed (321 baseline + 60 new), tools/golden.py check bit-exact at 0 ULP, params=2,353,634 sd=472362cea987, so the old bake path is provably untouched per D21. Core rules re-checked independently against layout.py: pval codec round-trips 0/161.2/655.35 to within 2.4e-5 and counts the clip (D9), counts dtype flips uint16->uint32 exactly at 65536 (D7), n_bins_for(248956422,25)=9958256 matching STORE_PLAN section 2 (D13), chunk (1024, n_tracks) gzip4 shuffle off (D2), RNA-seq retained by order_tracks (D15), chipseq-control sorted last and control_col set to its index (D18). DECISIONS THE PLAN DID NOT COVER: storage column order is assays sorted alphabetically with chipseq-control appended last, so the regime's declared order (D14) maps onto a deterministic, name-addressable store order; --chrom-sizes accepts JSON or 2-col TSV and falls back to <corpus>/../genome/chrom_sizes.json so t4's file is picked up automatically; two extra root attrs, npz_depth on counts (the per-DSF metadata.json depth, recorded alongside the CSV depth so a disagreement is visible rather than silent) and pval_clip_frac on pval; the manifest reads structure off the h5 attrs rather than a sidecar, keeping the SLURM array merge-free; the CSV-vs-json cross-check covers assay, accession, read_length and run_type, the fields both schemas carry; assay_vocabulary excludes chipseq-control; +inf is a real -log10 p and clips, NaN is refused by default; a ragged track set is an error; rebuild refuses without --overwrite and every output goes through .tmp + os.replace; peaks and pval record control_col=-1 because the source gives the control neither. build_manifest takes a LIST of CSV paths (--metadata-csv is repeatable), so t5's control_metadata.csv merges into the same table with no code path of its own. LEFT FOR t7/t8: genome.py, reader.py, regime.py, dataset.py; build-genome is a stub raising NotImplementedError naming t7; STORE_PLAN section 7 tests 2, 3, 4, 5 and 8 belong to t7/t8.

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
