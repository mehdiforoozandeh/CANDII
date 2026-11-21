---
id: f141
type: finding
schema: 3
title: Reading of existing runs; 16-region tiers only; U-Net alignment arm has 2…
parent: q45
confidence: moderate
why: Reading of existing runs; 16-region tiers only; U-Net alignment arm has 2 same-view seeds.
evidence: tasks/t370_minicandi_recipe_a_pareto_dominant_.md
exploratory: true
created: "2026-11-21T04:56:26"
updated: "2026-11-21T04:56:26"
---

# f141 — Reading of existing runs; 16-region tiers only; U-Net alignment arm has 2…

Parent:: [[q45_why_does_adding_the_no_duplicate_removal]]

## Answer

Alignment views (no dedup, MAPQ 0) share about 96% of their reads with the base file, so training pairs between them reward a covariate-gated copy of the input; on read-disjoint halves the U-Net alignment arm then scores like depth-only scaling (difference near 0, against about -0.08 for other arms, beyond 2x the seed range). The U-Net head skip copies fully (copy gain 0.65 vs 0.04-0.09 for conv models). Analysis: INSTRUCTIONS/t370_recipe/analysis/views_nodedup_mapq0.md (16-region tiers only).
