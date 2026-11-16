---
id: f86
type: finding
schema: 3
title: 3 seeds per arm and term gaps larger than seed ranges, but 16 regions, no…
parent: q39
confidence: moderate
why: 3 seeds per arm and term gaps larger than seed ranges, but 16 regions, no converged run, axial base at 1 of 3 seeds in this pass, pointwise MLP and matched-compute runs not scored.
evidence: results/final_round/FINAL_ROUND_EQUAL.md
created: "2026-11-16T13:25:00"
updated: "2026-11-16T13:25:00"
---

# f86 — 3 seeds per arm and term gaps larger than seed ranges, but 16 regions, no…

Parent:: [[q39_do_the_context_rungs_2_to_6_convolution_]]

## Answer

At equal windows (80,000) under the corrected composite, the conv autoencoder, conv + attention, the axial transformer and the U-Net are tied: composites 0.47 to 0.53 with seed ranges up to 0.06. All beat the baselines (marginal NB 0.052, position average 0.197). Perceiver IO is last (0.335 base, 0.395 with FiLM injection). The goal terms separate them: the U-Net has the highest variance recovered (0.43 to 0.45 against 0.37 to 0.40) and the lowest gap closed on crop pairs (0.51 to 0.53 against 0.64 to 0.72). Calibration has a seed range of 0.10 to 0.19 and ranks nothing.
