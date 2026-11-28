---
id: f406
type: finding
schema: 3
title: On the same 30 validation experiments, a 20 M read-depth cap on ChIP tracks…
parent: q90
confidence: moderate
evidence: results/t390/val_boards_side_by_side/BOARDS.md
created: "2026-11-24T22:19:37"
updated: "2026-11-24T22:19:37"
---

# f406 — On the same 30 validation experiments, a 20 M read-depth cap on ChIP tracks…

Parent:: [[q90_can_a_count_space_model_like_minicandi_b]]

## Answer

On the same 30 validation experiments, a 20 M read-depth cap on ChIP tracks (fitted on training tracks only) moves miniCANDI from position 5 to 4 of 11 (mean combined rank 5.31 -> 3.70; nMSE>1 on 20 -> 10 of 30). A per-assay count-slope fit gains little when fitted fairly (leave one experiment out: 4.97, position 5); the in-sample fit, a preview only, gives 4.56.
