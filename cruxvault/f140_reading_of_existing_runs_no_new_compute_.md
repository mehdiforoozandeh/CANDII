---
id: f140
type: finding
schema: 3
title: Reading of existing runs, no new compute; the causal arm exists only on eic_v1.
parent: q42
confidence: moderate
why: Reading of existing runs, no new compute; the causal arm exists only on eic_v1.
evidence: tasks/t370_minicandi_recipe_a_pareto_dominant_.md
exploratory: true
created: "2026-11-21T04:52:11"
updated: "2026-11-21T04:52:11"
---

# f140 — Reading of existing runs, no new compute; the causal arm exists only on eic_v1.

Parent:: [[q42_why_does_rung_4_the_u_net_recover_the_mo]]

## Answer

The U-Net's count skips (level skips plus the per-track head skip) carry a full-resolution copy of the source counts: copy gain 0.67 against 0.03-0.05 for the conv models on eic_v1_2 (3 seeds), and DNA-only skips raised correction from 0.504 (range 0.066) to 0.715 (0.056) on eic_v1. Existing runs do not separate which count path carries the copy. Analysis: INSTRUCTIONS/t370_recipe/analysis/unet_correction_cause.md.
