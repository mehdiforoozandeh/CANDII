---
id: f82
type: finding
schema: 3
title: 3 seeds per arm, but the all-views reference ran on Nibi from an earlier…
parent: q44
confidence: weak
why: 3 seeds per arm, but the all-views reference ran on Nibi from an earlier code snapshot, windows were equal so each view got fewer draws, the gap-closed metric is under review, and only 16 regions were scored
evidence: results/views_ablation/VIEWS_ABLATION.md
created: "2026-11-15T17:12:22"
updated: "2026-11-15T17:12:22"
---

# f82 — 3 seeds per arm, but the all-views reference ran on Nibi from an earlier…

Parent:: [[q44_does_training_the_u_net_on_the_counterfa]]

## Answer

No gain on the 16-region screening set. Base view only: composite 0.483 (range 0.050), gap closed 0.415 (0.085); all views: 0.444 (0.030), 0.298 (0.052). Specific variance recovered is flat (0.433 to 0.447 in every arm). Adding only the alignment-level views drops gap closed to 0.072 (0.068), all of it on pairs with a duplicate-keeping side (the other pairs stay at 0.519). Depth-thinning views steady the per-assay scores on thinning pairs across seeds (range 0.20 against 2.09 for base only). Cross-view training pairs are required: same-view pairs only gives gap closed below 0 in all 3 seeds.
