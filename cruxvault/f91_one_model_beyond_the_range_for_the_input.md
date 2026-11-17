---
id: f91
type: finding
schema: 3
title: One model beyond the range for the input; mixed sign for the output.
parent: q49
confidence: weak
why: One model beyond the range for the input; mixed sign for the output.
evidence: results/ablations_closeout/ABLATIONS_READ.md
created: "2026-11-16T22:22:45"
updated: "2026-11-16T22:22:45"
---

# f91 — One model beyond the range for the input; mixed sign for the output.

Parent:: [[q49_does_the_chip_seq_control_help_minicandi]]

## Answer

Removing the control input lowers imputation by 0.019 to 0.026 on all three models, beyond the seed range on conv + attention only. Not predicting the control gives small effects of opposite sign across models.
