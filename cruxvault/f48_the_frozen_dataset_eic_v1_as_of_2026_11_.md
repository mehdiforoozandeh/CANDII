---
id: f48
type: finding
schema: 3
title: "The frozen dataset eic_v1 (as-of 2026-11-13 14:32:03 UTC; tables_sha256…"
parent: q17
confidence: moderate
evidence: results/t183/DATASET_EIC_V1.md
created: "2026-11-14T02:19:20"
updated: "2026-11-14T02:19:20"
---

# f48 — The frozen dataset eic_v1 (as-of 2026-11-13 14:32:03 UTC; tables_sha256…

Parent:: [[q17_what_is_the_dataset_and_what_did_it_cost]]

## Answer

The frozen dataset eic_v1 (as-of 2026-11-13 14:32:03 UTC; tables_sha256 a068a867...b135; check job 27158261 PASS with every rule 0) holds 2,740 measurements (train 2,328, val 334, blind 78 testbed rows that are evaluation-only) and 15,123 pairs. Tracks present: training 249 of 267 (the 18 absent are all DNase-seq), validation 42 of 45 (absent rows 52, 56, 57), controls 37 of 40. The FASTQ reprocessing cost 4,511 CPU-h used (19,894 allocated, against a projection of 7,331), median 6.89 CPU-h per experiment; CPU-h per million reads 0.198 (chip-seq-pipeline2), 0.136 (dnase-seq-pipeline), 0.310 (atac-seq-pipeline); about 54 h wall for the finished rows. Scorable on eic_v1: 9 assays in counts space (H3K9ac only through a replicate view with no signal products); DNase-seq is not scorable (1 validation track with halves). Absent tracks enter later versions (eic_v1_1, ...) built as supersets.
