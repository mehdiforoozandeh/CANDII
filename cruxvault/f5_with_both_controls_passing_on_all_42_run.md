---
id: f5
type: finding
schema: 3
title: With both controls passing on all 42 runs, the depth-law checks pass on…
parent: h28
direction: against
confidence: moderate
null: ruled-out
why: With both controls passing on all 42 runs, the depth-law checks pass on Darm1 (gap closed 0.674, ratio transfer within bar) but the held-out CRPS check fails clearly (gap closed -0.53, gain -0.048 vs bar 0.125), so arm-to-arm training teaches the scale, not the distribution; DNase drives the miss and the depth-law pass straddles its bar across seeds.
vs_expect: scale part met; distribution part missed — via-base CRPS 0.435 stays far below Darm1 0.573
verifiables: c1 PASS, c2 PASS, c3 PASS, c4 FAIL, c5 PASS; rule all(c1,c3,c4) = FAIL
evidence: results/t136/checks_readings.tsv, results/t136/DIGEST.md
created: "2026-10-31T12:22:50"
updated: "2026-10-31T12:22:50"
---

# f5 — With both controls passing on all 42 runs, the depth-law checks pass on…

Parent:: [[h28_training_g_on_arm_to_arm_pairs_lets_a_di]]
