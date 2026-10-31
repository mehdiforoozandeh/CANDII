---
id: f27
type: finding
schema: 3
title: Dense arm-to-arm training (Darm2) on the same held-out pairs reads CRPS…
parent: h28
direction: supports
confidence: weak
null: not-ruled-out
why: "Dense arm-to-arm training (Darm2) on the same held-out pairs reads CRPS 0.428 (0.425 oracle-scaled + 0.003 scale error) against 0.435 through the base and 0.525 direct, with depth-law error 0.151 against 0.125 and 0.195; weak because it was read after the run, not checked, on one held-out level (PI, 2026-10-31: record as exploratory)."
vs_expect: exploratory; not a pre-registered check
evidence: results/t136/DIGEST.md, results/t136/checks_readings.tsv
created: "2026-10-31T13:08:09"
updated: "2026-10-31T13:08:09"
---

# f27 — Dense arm-to-arm training (Darm2) on the same held-out pairs reads CRPS…

Parent:: [[h28_training_g_on_arm_to_arm_pairs_lets_a_di]]
