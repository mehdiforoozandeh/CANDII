---
id: f407
type: finding
schema: 3
title: Averaging the final model's predicted mean with the base model's (neither…
parent: q90
confidence: moderate
evidence: results/t390/val_boards_side_by_side/BOARDS.md
created: "2026-11-25T01:25:41"
updated: "2026-11-25T01:25:41"
---

# f407 — Averaging the final model's predicted mean with the base model's (neither…

Parent:: [[q90_can_a_count_space_model_like_minicandi_b]]

## Answer

Averaging the final model's predicted mean with the base model's (neither trained on validation rows), then the 20 M depth cap, puts miniCANDI 2nd of 11 on the same 30 validation experiments (mean combined rank 3.35, against 3.70 and position 4 for the depth cap alone). Against the cap alone the gain is small per experiment (better 10, worse 9).
