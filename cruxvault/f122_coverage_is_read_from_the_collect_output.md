---
id: f122
type: finding
schema: 3
title: Coverage is read from the collect output and the store state from the…
parent: q17
confidence: moderate
why: Coverage is read from the collect output and the store state from the variance note; the verification of the 1 bp store is owed.
evidence: results/closeout_A4/qc_tables_2026-11-18/coverage.txt, results/closeout_A7/variance_2026-11-18/VARIANCE_READ.md, tasks/t354_onion_verify_the_1_bp_tables_verify.md
created: "2026-11-18T16:22:48"
updated: "2026-11-18T16:22:48"
---

# f122 — Coverage is read from the collect output and the store state from the…

Parent:: [[q17_what_is_the_dataset_and_what_did_it_cost]]

## Answer

Close-out state of the input-side data (2026-11-18). QC side tables of eic_v1_2 over 2,662 rows: NSC and RSC filled for 2,370 (89.0%), fragment length 2,337 (87.8%), NRF, PBC1 and PBC2 1,006 (37.8%), FRiP 0 (0%). The 1 bp region store for the resolution study holds 412 measurements recounted from the read sets (a point check matched 13,222 of 13,222 points, and every view passed the read-set check), but the end-to-end verification did not finish (the verify job reached its time limit with out-of-memory kills), so the 1 bp tables are built and NOT verified, and the resolution evaluation set is not built. The read sets of onion/work and the FASTQs are now on tape only (Nibi nearline).
