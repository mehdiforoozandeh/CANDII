---
id: f114
type: finding
schema: 3
title: Near-complete sign agreement on 55 cells, but 10 of 21 arms (two-step…
parent: q56
confidence: moderate
why: Near-complete sign agreement on 55 cells, but 10 of 21 arms (two-step routing and average input among them) have no 256-region score yet.
evidence: results/rescoring_p256/read_20261118T1942Z/arms.tsv
created: "2026-11-18T16:21:27"
updated: "2026-11-18T16:21:27"
---

# f114 — Near-complete sign agreement on 55 cells, but 10 of 21 arms (two-step…

Parent:: [[q56_do_the_ablation_conclusions_drawn_from_t]]

## Answer

In the final read of the run (2026-11-18 19:42 UTC, 34 of 62 run-seeds scored, 11 of 21 arms complete), 53 of the 55 cells of complete arms have the same sign on 16 and on 256 regions. The 5 new cells (the U-Net fixed-input-depth arm) all agree. The two that differ are unchanged from the 15:54 UTC read: crop-pair correction on conv + attention for base view only (-0.0117 on 256 against +0.0029 on 16) and for imputation weight 4 (-0.0131 against +0.0001); both are within the seed range on 256 regions and near zero on 16. Two-step routing and average input (rules 2.3 and 2.5) have no 256-region score, so their 16-region conclusions are not checked.
