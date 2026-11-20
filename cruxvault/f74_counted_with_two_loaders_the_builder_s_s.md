---
id: f74
type: finding
schema: 3
title: Counted with two loaders (the builder's script and the scorer's own) and…
parent: q17
confidence: moderate
why: Counted with two loaders (the builder's script and the scorer's own) and tested byte-identical without the option; moderate because no signal-space score has been recomputed with the side table.
evidence: results/fraglen_single_end/FRAGLEN.md
created: "2026-11-14T16:39:45"
updated: "2026-11-14T16:39:45"
---

# f74 — Counted with two loaders (the builder's script and the scorer's own) and…

Parent:: [[q17_what_is_the_dataset_and_what_did_it_cost]]

## Answer

The frozen eic_v1 tables leave frag_len (the MACS2 fragment length d, read only by the ChIP-seq signal route) empty for 231 base ChIP-seq tracks (35 validation, 196 training; 0 blind), 221 of them pooled tracks, because the build fills frag_len only for one replicate's own read set; 13 of the 231 are mixed or paired-end, so the gap is not a single-end gap. Signal-space metrics of these tracks were undefined ('covariates lack [frag_len]'). A side table (600 ChIP-seq rows; provenance first, then the replicate mean, then strand cross-correlation; plausible range 80 to 400 bp) and the scorer option --fraglen-table restore a fragment length for 42 of 42 ChIP-seq AlphaGenome 2 headline targets (10 before) and 62 of 62 ChIP-seq validation imputation items (16 from the table, 46 estimated); 156 of the 196 training tracks have one now, 32 wait for the QC array and 8 will have none. The frozen tables were not changed (tables_sha256 a068a8674aee); without the option every output is byte-identical. The sources agreed with a median absolute difference of 0 to 5 bp. Not measured: the effect on any signal-space score (no rescoring ran).
