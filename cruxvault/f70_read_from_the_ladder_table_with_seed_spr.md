---
id: f70
type: finding
schema: 3
title: Read from the ladder table with seed spreads; moderate because it is a…
parent: q39
confidence: moderate
why: Read from the ladder table with seed spreads; moderate because it is a screening score with the superseded correction scale in its composite, and rung 5 has one seed.
evidence: results/metrics_screening/scores_ladder.md, results/metrics_screening/scored.tsv
created: "2026-11-14T16:38:49"
updated: "2026-11-14T16:38:49"
---

# f70 — Read from the ladder table with seed spreads; moderate because it is a…

Parent:: [[q39_do_the_context_rungs_2_to_6_convolution_]]

## Answer

Screening scores (imputation on 256 regions, pair tests on 16; composite version 1, whose correction term is the superseded half-read scale and is clipped to 1 for every rung). Rungs 1 to 5 beat the position average on imputation and calibration: specific variance recovered above noise 0.343 to 0.440 against 0.119, count CRPS 0.6965 to 0.7092 against 0.7947, pooled PIT deviation 0.019 to 0.024 against 0.0499. Rung 4 recovers the most cell-type-specific variance above noise, 0.440 (seed spread 0.013), against 0.343 (rung 1), 0.385 (rung 2), 0.389 (rung 3) and 0.384 (rung 5), but it has the worst correction CRPS of rungs 1 to 5 (0.5309 (0.0114) against 0.4754 to 0.5061; depth-only 0.6014). Rung 6 is near the baseline on imputation: specific variance 0.154 (0.064), count CRPS 0.7986 (0.0162) against 0.7947. Rung 2 has the best correction CRPS (0.4754 (0.0027)). Rung 5 has 1 seed (two metric jobs wait on their predictions), so its values carry no spread. Rungs 2 to 6 were scored from CPU predictions with batch 8 and rung 1 from GPU predictions with batch 32 (differences up to about 1e-5).
