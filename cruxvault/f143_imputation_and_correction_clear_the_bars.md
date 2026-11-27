---
id: f143
type: finding
schema: 3
title: Imputation and correction clear the bars by several seed ranges; calibration…
parent: q75
confidence: strong
why: Imputation and correction clear the bars by several seed ranges; calibration clears in the mean but within its seed range, and recalibration widens it; settled because the PI fixed this architecture and no further architecture search is planned.
evidence: tasks/t370_minicandi_recipe_a_pareto_dominant_.md
created: "2026-11-23T18:18:36"
updated: "2026-11-23T18:18:36"
---

# f143 — Imputation and correction clear the bars by several seed ranges; calibration…

Parent:: [[q75_can_one_minicandi_recipe_match_or_beat_t]]

## Answer

Yes. The U-Net with read-disjoint pairs, average-signal and Pearson-residual inputs, head skip, 160k windows, control weight 1 and per-assay recalibration beats the best base model on all three terms at 256 regions over 3 seeds (imputation 0.503 vs 0.445, correction 0.818 vs 0.675, calibration 0.717 vs 0.646); the PI picked it as the final miniCANDI architecture on 2026-11-23.
