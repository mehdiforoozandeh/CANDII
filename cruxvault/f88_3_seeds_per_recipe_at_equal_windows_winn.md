---
id: f88
type: finding
schema: 3
title: 3 seeds per recipe at equal windows; winners were chosen on seed 0 held-out…
parent: q46
confidence: moderate
why: 3 seeds per recipe at equal windows; winners were chosen on seed 0 held-out loss only, and matched-compute runs are pending.
evidence: results/final_round/FINAL_ROUND_EQUAL.md
created: "2026-11-16T13:25:01"
updated: "2026-11-16T13:25:01"
---

# f88 — 3 seeds per recipe at equal windows; winners were chosen on seed 0 held-out…

Parent:: [[q46_why_does_the_held_out_nb_loss_rank_the_u]]

## Answer

Selecting each architecture's recipe by held-out NB loss did not raise the corrected composite at equal windows: conv autoencoder 0.528 base to 0.496 tuned, conv + attention 0.525 to 0.510, U-Net 0.495 to 0.511, all inside seed ranges. For the conv autoencoder the tuned recipe lowered gap closed on crop pairs (0.718 to 0.644, ranges 0.012 and 0.019). The held-out loss is not a proxy for the goal terms; no architecture was held back by its base recipe.
