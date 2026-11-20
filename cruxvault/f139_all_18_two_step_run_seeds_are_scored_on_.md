---
id: f139
type: finding
schema: 3
title: All 18 two-step run-seeds are scored on 256 regions and the composite change…
parent: h42
direction: supports
confidence: strong
why: "All 18 two-step run-seeds are scored on 256 regions and the composite change is zero or negative in all six arm-by-model cells: native base depth -0.079 (conv autoencoder, seed range 0.043), -0.051 (conv + attention, 0.056), -0.087 (U-Net, 0.060); fixed base depth -0.051 (0.065), -0.058 (0.048), -0.003 (0.048). No cell raises the composite at all, so the twice-the-range bar is missed by a clear margin. Recorded trade-off: on the U-Net with a fixed base depth, correction on crop pairs is up by 0.117 and calibration moves by -0.165."
vs_expect: "as expected: no arm passes the bar on any model"
evidence: results/rescoring_p256/read_20261119T0850Z/RESCORING_READ.md, results/rescoring_p256/read_20261119T0850Z/verdicts.tsv
created: "2026-11-19T01:47:43"
updated: "2026-11-19T01:47:43"
---

# f139 — All 18 two-step run-seeds are scored on 256 regions and the composite change…

Parent:: [[h42_on_all_256_regions_neither_two_step_arm_]]
