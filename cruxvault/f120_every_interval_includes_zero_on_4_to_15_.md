---
id: f120
type: finding
schema: 3
title: Every interval includes zero on 4 to 15 cells, the form is one-sided, the…
parent: q58
confidence: weak
why: Every interval includes zero on 4 to 15 cells, the form is one-sided, the cells differ from the lower rung, and the two-sided form leaves no estimable pair.
evidence: results/closeout_D2/layers_2026-11-18/layers.tsv, results/closeout_E4/ladder_two_sided_2026-11-18/COMPARE.md
created: "2026-11-18T16:22:33"
updated: "2026-11-18T16:22:33"
---

# f120 — Every interval includes zero on 4 to 15 cells, the form is one-sided, the…

Parent:: [[q58_can_the_second_experiments_separate_a_la]]

## Answer

No: the second experiments show no other-experiment layer distinguishable from zero, and lab is reported together with donor (PI decision 3 of 2026-11-17; the 17 unbuilt DNase-seq second-experiment rows were dropped, PI decision 4, so DNase-seq has no second-experiment pair). Second-experiment layers of 2026-11-18 (one-sided layers stage, log1p counts at a common depth set with the second experiments, increment over the isogenic rung, pooled by assay class, all regions and bands): broad histone marks, all other-experiment pairs -0.00042 [-0.0066, 0.0058] (30 pairs, 10 cells), lab and donor changed together -0.00020 [-0.0116, 0.0111] (13 pairs, 4 cells); narrow histone marks, all pairs -0.0098 [-0.0208, 0.0012] (31 pairs, 15 cells), lab and donor together -0.0188 [-0.0430, 0.0054] (6 pairs, 4 cells). Every interval includes zero. In the two-sided ladder no other-experiment rung is estimable, because no second experiment has a replicate pair of its own. The DNase-seq rows (8 pairs) are at 15,201 reads and cannot be read. The results of the 5 baseline jobs on the second experiments are on Nibi only and are not read here.
