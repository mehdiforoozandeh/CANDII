---
id: t128
type: task
title: Build and run phase 1 of the symmetric encoder-decoder rung D: one g sets the FiLM of one shared CNN for both halves through a learned clean middle setting; 8 settings, 240 runs, counts only, depth 7.5M held out (exploratory)
category: implementation
parent: t118
blocked_by: None
refs: q7, h22, h19
hypothesis_refs: 
status: open
created: 2026-10-21T13:10:55
updated: 2026-10-21T13:10:55
---

# t128 — Build and run phase 1 of the symmetric encoder-decoder rung D: one g sets the FiLM of one shared CNN for both halves through a learned clean middle setting; 8 settings, 240 runs, counts only, depth 7.5M held out (exploratory)

Refs:: [[q7_why_do_the_covariate_conditioned_g_f_des\|q7]], [[h22_a_composable_g_one_shared_map_from_each_\|h22]], [[h19_g_fits_trained_base_arm_depth_pairs_but_\|h19]]

## Why

tests whether a symmetric encoder-decoder through a learned clean setting carries the fit to never-trained arm-to-arm pairs, and which loss real data could support

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [First-pass readings, 2026-10-24](results/t128/analyze_out.txt) — not final; the deck is not built yet

## Evidence

First pass, 2026-10-24, exploratory; another session may still be working on it. Design `plan/T126_ENCDEC_D2.md`. Four of eight settings are complete: the default and boxed clean-setting forms give never-trained depth error 0.29–0.42, within noise of old D (0.29–0.34); round-trip-only training stays at the identity (error about 2.0). The λ settings and the VAE with the cross loss are unfinished, so the λ choice is not final, and old D's CRPS split waits on Nibi job 24964406. Old D's two trained maps run in series through the base meet the law (0.07–0.12). Bears on [[h22_a_composable_g_one_shared_map_from_each_|the composable-g hypothesis]] and [[h19_g_fits_trained_base_arm_depth_pairs_but_|the trained versus never-trained hypothesis]].
