---
id: t126
type: task
title: Build and run the encoder-decoder rung A (shared per-side map phi, exact inverse; beta learned and beta=1) against a retrained old rung A, counts only, depth 7.5M held out of training (exploratory)
category: implementation
parent: t118
blocked_by: None
refs: q7, h22, h19
hypothesis_refs: 
status: done
created: "2026-10-19T23:23:29"
updated: "2026-10-20T19:00:36"
---

# t126 — Build and run the encoder-decoder rung A (shared per-side map phi, exact inverse; beta learned and beta=1) against a retrained old rung A, counts only, depth 7.5M held out of training (exploratory)

Refs:: [[q7_why_do_the_covariate_conditioned_g_f_des\|q7]], [[h22_a_composable_g_one_shared_map_from_each_\|h22]], [[h19_g_fits_trained_base_arm_depth_pairs_but_\|h19]]

## Why

tests whether a composable g fixes the never-trained-pair generalisation failure (depth law and other knobs) without arm-to-arm training pairs

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [Lab deck: encoder-decoder g](presentations/t126-encdec/DECK.md) — published https://claude.ai/artifact/KP7pipaRbNbEStUaJR7jrZ
- [Readings report](results/t126/report.pdf)

## Evidence

Exploratory readings (2026-10-21), no verdict. Encoder-decoder A, with the exact inverse, loses the fit even on trained depth pairs (median |log2 error| 1.0–1.4) and its CRPS is mostly scale error. Encoder-decoder D keeps old D's trained fit but misses never-trained depth pairs as much (0.26–0.52 against 0.29–0.34). The held-out 7.5M depth is easy on base ↔ 7.5M pairs for every arm except encoder-decoder A. Detail: [[h22_a_composable_g_one_shared_map_from_each_|the composable-g hypothesis]] and `results/h22/report.md`.
