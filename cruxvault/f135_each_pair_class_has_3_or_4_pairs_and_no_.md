---
id: f135
type: finding
schema: 3
title: Each pair class has 3 or 4 pairs and no interval was computed, so the class…
parent: q58
confidence: weak
why: Each pair class has 3 or 4 pairs and no interval was computed, so the class differences cannot be read; the direction of the two baselines is clear.
evidence: results/closeout_D2/addon_baselines_2026-11-18/baselines/position_average/seed0/scores.json
created: "2026-11-18T18:09:18"
updated: "2026-11-18T18:09:18"
---

# f135 — Each pair class has 3 or 4 pairs and no interval was computed, so the class…

Parent:: [[q58_can_the_second_experiments_separate_a_la]]

## Answer

On the 13 cross-experiment pairs of the second-experiment add-on (11 cell types), the position average closes more than the whole gap between depth-only scaling and the read-half floor (pooled 1.98; donor-only pairs 3.53, lab-only 1.78, lab and donor 1.28), and the marginal NB is worse than depth-only scaling (pooled -1.14). A value above 1 means the read-half floor is not a floor for a pair of two experiments.
