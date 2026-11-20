---
id: f131
type: finding
schema: 3
title: On all 256 regions with 3 seeds the U-Net imputation rises by 0.035 (seed…
parent: h43
direction: supports
confidence: moderate
null: none
why: On all 256 regions with 3 seeds the U-Net imputation rises by 0.035 (seed range 0.023) and no goal term falls beyond the range, by a rule written before the scores; the margin is 1.5 times the range, not twice, so moderate.
vs_expect: U-Net imputation 0.466 against 0.432 (expected 0.432 to 0.466); crop-pair correction -0.007 (range 0.066), calibration +0.001 (range 0.162); the qualify line reads qualifies
evidence: results/rescoring_p256/read_20261119T0110Z/arms.tsv
created: "2026-11-18T18:09:17"
updated: "2026-11-18T18:09:17"
---

# f131 — On all 256 regions with 3 seeds the U-Net imputation rises by 0.035 (seed…

Parent:: [[h43_on_all_256_regions_the_leave_one_out_ave]]
