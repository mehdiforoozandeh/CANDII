---
id: f87
type: finding
schema: 3
title: Both effects exceed the seed ranges (0.02 and 0.06 to 0.07) in a 3-seed…
parent: q42
confidence: moderate
why: Both effects exceed the seed ranges (0.02 and 0.06 to 0.07) in a 3-seed designed ablation at equal windows; one screening set of 16 regions, 18 crop pairs.
evidence: results/final_round/FINAL_ROUND_EQUAL.md
created: "2026-11-16T13:25:01"
updated: "2026-11-16T13:25:01"
---

# f87 — Both effects exceed the seed ranges (0.02 and 0.06 to 0.07) in a 3-seed…

Parent:: [[q42_why_does_rung_4_the_u_net_recover_the_mo]]

## Answer

The count skip connections cause it. With skips that carry DNA only, the U-Net scores like a conv autoencoder: variance recovered 0.380 (base 0.432), gap closed on crop pairs 0.734 (base 0.507), correction-only term 0.689 (base 0.318). With skips that carry counts only it equals the base (0.432, 0.518, 0.304). The full-resolution path from the input counts helps imputation and passes the source view's processing signature to the output.
