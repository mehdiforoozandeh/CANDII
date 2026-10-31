---
id: t128
type: task
title: "Build and run phase 1 of the symmetric encoder-decoder rung D: one g sets the FiLM of one shared CNN for both halves through a learned clean middle setting; 8 settings, 240 runs, counts only, depth 7.5M held out (exploratory)"
category: implementation
parent: t118
blocked_by: None
refs: q7, h22, h19
hypothesis_refs: 
status: done
created: "2026-10-21T13:10:55"
updated: "2026-10-31T00:19:33"
---

# t128 — Build and run phase 1 of the symmetric encoder-decoder rung D: one g sets the FiLM of one shared CNN for both halves through a learned clean middle setting; 8 settings, 240 runs, counts only, depth 7.5M held out (exploratory)

Refs:: [[q7_why_do_the_covariate_conditioned_g_f_des\|q7]], [[h22_a_composable_g_one_shared_map_from_each_\|h22]], [[h19_g_fits_trained_base_arm_depth_pairs_but_\|h19]]

## Why

tests whether a symmetric encoder-decoder through a learned clean setting carries the fit to never-trained arm-to-arm pairs, and which loss real data could support

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [First-pass readings, 2026-10-24](results/t128/analyze_out.txt)
- [Deck: symmetric encoder-decoder rung D, phase 1](presentations/t128-encdec-sym/DECK.md)
- [Unified deck: the pivot translator, phases 1 and 2](presentations/t128-pivot/DECK.md)

## Evidence

Exploratory, closed by the PI on 2026-10-31. Design `plan/T126_ENCDEC_D2.md` (phases 1, 2, 3 and 3b). Phase 1 is final at 233 of 240 runs. No pivot design (learned, boxed, base-labelled or anchored middle) beat old D on never-trained pairs outside noise; only old D read through the base did (CRPS 0.545 against 0.662, one g per track). Phase 2 (t132) and phase 3 (t134, t135) followed. A wider g or f did not move the depth law or the never-trained CRPS outside noise; with six seeds bigger f's gain through the base was 0.006 (95% interval −0.002 to +0.013). The PI closed the scale question at these sizes. Bears on [[h22_a_composable_g_one_shared_map_from_each_|the composable-g hypothesis]], [[h19_g_fits_trained_base_arm_depth_pairs_but_|the trained versus never-trained hypothesis]], [[h20_a_larger_g_with_the_same_training_pairs_|the g-size hypothesis]] and [[h27_f_at_64_channels_2x_wider_with_the_same_|the f-size hypothesis]].
