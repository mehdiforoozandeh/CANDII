---
id: f89
type: finding
schema: 3
title: Differences are near the seed range and the base has 1 of 3 seeds in this…
parent: q47
confidence: weak
why: Differences are near the seed range and the base has 1 of 3 seeds in this scoring pass.
evidence: results/final_round/FINAL_ROUND_EQUAL.md
created: "2026-11-16T13:25:01"
updated: "2026-11-16T13:25:01"
---

# f89 — Differences are near the seed range and the base has 1 of 3 seeds in this…

Parent:: [[q47_which_attention_axis_carries_the_axial_t]]

## Answer

Attention along positions only equals the full model on the composite (0.515 (0.026) against 0.517 at 1 seed, 0.498 (0.031) at 3 seeds in the earlier pass). Attention along assays only is lower (0.474 (0.023); variance recovered 0.357 against 0.398). One shared attention for both axes (0.496) and FiLM covariate injection (0.521) do not differ from the base.
