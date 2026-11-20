---
id: f110
type: finding
schema: 3
title: The one-sided rerun reproduces the cluster report byte for byte and the joint bootstrap covered…
parent: q59
confidence: moderate
why: The one-sided rerun reproduces the cluster report byte for byte and the joint bootstrap covered the truth in 93.7% of 300 synthetic data sets, but the two methods were never compared on the full data and the other-experiment rung is empty.
evidence: tasks/t351_onion_replicate_ladder_in_the_two_s.md
created: "2026-11-18T09:48:46"
updated: "2026-11-18T09:48:46"
---

# f110 — The one-sided rerun reproduces the cluster report byte for byte and the joint bootstrap covered…

Parent:: [[q59_do_the_nested_replicate_ladder_and_the_c]]

## Answer

The two-sided replicate ladder with the joint cluster bootstrap is the form to quote; the crossed mixed model is not run, because cell type and replicate are not separately identified. Cell-type rung, two-sided: broad histone marks 0.00273 [0.0015, 0.00396] (355 pairs, 16 cells), narrow histone marks 0.003547 [0.00198, 0.00511] (445 pairs, 23 cells). No other-experiment rung is estimable, because no second experiment has a replicate pair of its own.
