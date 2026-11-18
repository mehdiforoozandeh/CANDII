---
id: f127
type: finding
schema: 3
title: A read-only audit of 18 runs with seed, region and cell resampling; the…
parent: q39
confidence: moderate
why: A read-only audit of 18 runs with seed, region and cell resampling; the LayerNorm cause rests on a synthetic probe, and the later final round replaces the order itself.
evidence: results/ladder_audit/FAIR_CHANCE.md, results/ladder_tuning/rung6/PLAN.md
created: "2026-11-18T16:23:43"
updated: "2026-11-18T16:23:43"
---

# f127 — A read-only audit of 18 runs with seed, region and cell resampling; the…

Parent:: [[q39_do_the_context_rungs_2_to_6_convolution_]]

## Answer

An audit of the 16-region ladder (composite version 2, 3 seeds, equal windows of 80,000) found the six-place order not trustworthy: only three tiers survive the seed test (conv autoencoder and conv + attention on top; axial transformer, U-Net and pointwise MLP in the middle; Perceiver IO last). The runs held windows equal while the design asked for matched compute (2.4e15 FLOPs); training FLOPs differed 58.7-fold across rungs; learning rates were not tuned and no held-out loss was logged. The Perceiver IO handicap is partly explained: gradient clipping on 27 to 41% of logged steps, and a missing LayerNorm before its output layer (head-input RMS 11.2 against 2.27 for the axial transformer in a 200-step synthetic probe, a direction only). The U-Net's lead in variance recovered (0.0495 or more, seed ranges 0.0297 or less) was the most robust result. These led to the tuning round and the equal-window final round.
