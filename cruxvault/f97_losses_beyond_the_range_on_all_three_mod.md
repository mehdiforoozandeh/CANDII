---
id: f97
type: finding
schema: 3
title: Losses beyond the range on all three models; the Anscombe arm is a code…
parent: q55
confidence: moderate
why: Losses beyond the range on all three models; the Anscombe arm is a code defect, not a result.
evidence: results/ablations_closeout/ABLATIONS_READ.md
created: "2026-11-16T22:22:46"
updated: "2026-11-16T22:22:46"
---

# f97 — Losses beyond the range on all three models; the Anscombe arm is a code…

Parent:: [[q55_does_a_count_transform_on_the_input_beat]]

## Answer

No transform beats the offset on two or more models. Local-ratio and quantile transforms lower imputation by about 0.09 to 0.11 on all three models. Pseudocounts 0.01, 0.1 and 1 score within the range. The Anscombe arm is void: a zero-initialized output head has zero gradient under the Anscombe link, so the 9 runs never trained.
