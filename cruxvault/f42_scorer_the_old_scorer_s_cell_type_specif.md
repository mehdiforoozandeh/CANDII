---
id: f42
type: finding
schema: 3
title: "Scorer: the old scorer's cell-type-specific correlation (r_spec) depended on…"
parent: q27
confidence: moderate
evidence: results/scorer_memory/NOTES.md, results/predict_store_speed/FORWARD_SPEED.md
created: "2026-11-14T02:18:33"
updated: "2026-11-14T02:18:33"
---

# f42 — Scorer: the old scorer's cell-type-specific correlation (r_spec) depended on…

Parent:: [[q27_are_the_onion_run_s_estimators_scorers_a]]

## Answer

Scorer: the old scorer's cell-type-specific correlation (r_spec) depended on the BLAS thread count, by up to 2.3e-7 (60 keys), so the scoring job now pins 2 threads per process; peak memory fell from 16.27 to 7.73 GiB MaxRSS on a 2-group profile at 256 regions with identical outputs. 589 of the 619 validation replicate pairs of eic_v1 are other_cell_type pairs, and a model window holds one cell, so cross-cell replicate pairs cannot be a model-side task (the scorer crashed on them; it now marks them not applicable). One model scoring at all 256 pair regions needs 6,443,008 window forwards (258,048 for the items and 1,510 x 4,096 for the pairs), against 644,608 with pairs on 16 regions.
