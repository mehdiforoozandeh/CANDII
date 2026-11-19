---
id: f134
type: finding
schema: 3
title: Read from the rule tool output on the 50-score table; the two-step arms are…
parent: q56
confidence: moderate
why: Read from the rule tool output on the 50-score table; the two-step arms are not scored, so the comparison of 16 and 256 regions is not complete.
evidence: results/rescoring_p256/read_20261119T0110Z/RESCORING_READ.md
created: "2026-11-18T18:09:18"
updated: "2026-11-18T18:09:18"
---

# f134 — Read from the rule tool output on the 50-score table; the two-step arms are…

Parent:: [[q56_do_the_ablation_conclusions_drawn_from_t]]

## Answer

At 50 of 62 run-seeds scored, five of the six pre-written rules are decided on all 256 regions (covariate use, all views, fixed input depth, average input, imputation weight 4). Two-step correction is the only rule that waits: 12 run-seeds are still in scoring.
