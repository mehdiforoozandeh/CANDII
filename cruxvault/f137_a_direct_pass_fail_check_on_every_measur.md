---
id: f137
type: finding
schema: 3
title: A direct pass/fail check on every measurement, repeated once with the same…
parent: q60
confidence: moderate
why: A direct pass/fail check on every measurement, repeated once with the same result; it verifies the data, and adds no new variance measurement.
evidence: results/closeout_A7/verify_2026-11-19/onion_a7_verify_27868178.out, results/closeout_A7/verify_2026-11-19/onion_a7_evalset_27868179.out
created: "2026-11-18T20:58:27"
updated: "2026-11-18T20:58:27"
---

# f137 — A direct pass/fail check on every measurement, repeated once with the same…

Parent:: [[q60_how_much_of_the_1_bp_count_variance_of_a]]

## Answer

The 1 bp tables (eic_res_v1) passed end-to-end verification: 412 measurements checked, 0 failures, in two independent runs (8 workers, 128 GB). The resolution eval set was built on them: 64 eval regions, 16 fit regions, 0 shared, 0 problems. This removes the 'store not verified' caveat on the earlier variance-below-25-bp answers; it does not change their numbers or their other caveats (one replicate pair for most assays, one DNase-seq track).
