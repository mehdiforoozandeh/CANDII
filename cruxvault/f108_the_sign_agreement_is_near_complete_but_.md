---
id: f108
type: finding
schema: 3
title: The sign agreement is near complete, but only 4 of 7 arm groups are complete…
parent: q56
confidence: moderate
why: The sign agreement is near complete, but only 4 of 7 arm groups are complete (30 of 62 run-seeds scored) and the sizes of the correction terms still differ between 16 and 256 regions.
evidence: tasks/t349_onion_first_read_of_the_256_region_.md
created: "2026-11-18T09:48:32"
updated: "2026-11-18T09:48:32"
---

# f108 — The sign agreement is near complete, but only 4 of 7 arm groups are complete…

Parent:: [[q56_do_the_ablation_conclusions_drawn_from_t]]

## Answer

For the arms decided so far, yes in sign: in arms.tsv of the 15:54 UTC read, 48 of the 50 cells of complete arms (scrambled covariates, base view only, imputation weight 4, fixed input depth on the two conv models) have the same sign on 16 and on 256 regions. The two that differ are crop-pair correction on conv + attention for base view only (-0.0117 against +0.0029) and imputation weight 4 (-0.0131 against +0.0001); both are within the seed range on 256 regions and near zero on 16.
