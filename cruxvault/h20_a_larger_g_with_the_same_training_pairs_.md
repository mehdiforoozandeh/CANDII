---
id: h20
type: idea
schema: 2
title: A larger g, with the same training pairs and loss, meets the depth law on never-trained depth pairs
parent: q7
status: done
rule: all
measurement: 
replicates: 
verdict: refuted
metric: never-trained depth-law error 0.63 (per-track g) / 0.46 (across) vs trained 0.07; old D 0.29 / 0.34
created: "2026-10-19T23:41:09"
updated: "2026-10-31T00:33:38"
lock: 64f29a808f5333b8
locked: "2026-10-31T00:23:17"
lock_at: close
null_approved: "2026-10-31T00:33:38"
null_hash: 24641e914971592d
---

# h20 — A larger g, with the same training pairs and loss, meets the depth law on never-trained depth pairs

Parent:: [[q7_why_do_the_covariate_conditioned_g_f_des]]

## ELI5

The part of the model that reads the lab settings may be too small to learn how sequencing depth works.

## TL;DR

g is a small MLP (2 × 64) on the source and target covariates. If its size limits it, a wider or deeper g, with the same training pairs and loss, brings never-trained depth pairs onto the law. Raised by the PI on 2026-10-19. Phase 3 of the pivot study (2026-10-29) ran a 4× wider g (2 × 256) on old D: it does not meet the law on never-trained depth pairs.

## Null

Chance: a single seed or track happens to land near the law; the median over 3 seeds and 7 tracks guards against it.

## Problem Statement

Every design misses the depth law on never-trained arm → arm pairs ([[q7_why_do_the_covariate_conditioned_g_f_des|the depth-law question]]). Capacity is the cheapest suspect to rule out before a structural fix, because a 2 × 64 g may be too small to represent the depth ratio as a rule.

## Idea / Hypothesis

A larger g, with the same training pairs and loss, meets the depth law on never-trained depth pairs

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
Checks written after the run, at the PI's request on 2026-10-31; the work was exploratory with no bar set beforehand.

- [ ] With g at 2 × 256 (old D's f unchanged), the never-trained depth → depth median |log2 error| falls to the trained-pair level (about 0.1) for one g per track or one g across tracks
      fails-if:: Both g versions stay well above their trained-pair level
      discriminates:: true
      (found: 0.63 per-track g, 0.46 across-track g; trained 0.07 — [depth-law table](results/t128p3/m5_depthlaw_table.tsv))
- [x] [outcome-neutral] The target tracks themselves obey the depth law on the scored bins
      fails-if:: Σy/ΣX differs from the manifest depth ratio by more than 0.01 log2
      (found: within 0.004 log2, as for the loss hypothesis; the data audit gives 0.71% over 84 ordered pairs)

## Planned Intervention

Run as phase 3 of the pivot study, 2026-10-27 to 2026-10-29: old D with g at 2 × 256 (f unchanged), and a sibling arm with f at 64 channels (g unchanged); 8 g units × 3 seeds each, 15,000-step cap, early stop. Spec: plan/T126_ENCDEC_D2.md § Phase 3.

## Run Links

- Fir: /project/def-maxwl/mforooz/t128p3 (runs/, viabase/), kit K1 = f678ebd; jobs 758803–758806.

## Artifacts

<!-- what the run produced. Keep files under results/h20/ and link at least the report:
     - [Report](results/h20/report.md)   - results/h20/curve.png -->
- [Phase-3 deck: a bigger g and a bigger f](presentations/t128p3-scale/DECK.md)
- [Phase-3 depth-law table (arms Dbigg, Dbigf)](results/t128p3/m5_depthlaw_table.tsv)
- [Phase-3 ranking table](results/t128p3/rank_table.tsv)

Earlier readings:
- [Trained versus never-trained report](results/h19/report.md)
- [Encoder-decoder depth table](results/t126/depth_table.tsv)
- [Via-base depth table (arm Dviabase)](results/t128/depth_table.tsv)

## Findings

Exploratory readings; the PI has not ruled. Median |log2 error| of the predicted scale, chr19 + chr21; seed spread in brackets.

- A 4× wider g does not meet the law on never-trained depth → depth pairs: 0.63 (0.19) per-track g and 0.46 (0.11) across-track g, against old D at 0.29 (0.05) and 0.34 (0.24). It still fits trained pairs (0.07).
- A 2× wider f does not meet it either ([[h27_f_at_64_channels_2x_wider_with_the_same_|the f-size hypothesis]]).
- Every gap to old D is within noise. Neither size moves the miss towards the law.
- Read through the base, the wider g stays near the law: 0.14 and 0.21, against old D's 0.11 and 0.07.
- On never-trained arm → arm CRPS through the base, per-track g, it gives 0.534 (0.529 + 0.005) against old D's 0.545 (0.537 + 0.008). The gain of 0.011 does not clear the bar set before the run (0.015).
- Caveat (PI, 2026-10-29): read directly, its per-track CRPS seeds are 0.627, 0.82 and 0.77, mostly scale error. The PI ruled this a caveat; it is not followed up.
- No run's best step came near the cap.

So size, at these widths, is not what makes g miss the law on unseen pairs. The PI closed the scale question at these sizes on 2026-10-31 (option A): no wider g, combined arm or learning-rate sweep follows. The earlier readings had leaned the same way: the 2 × 64 g fits trained pairs to 0.04–0.10 in every design ([[h19_g_fits_trained_base_arm_depth_pairs_but_|the trained versus never-trained hypothesis]]), and it meets the law when its two trained maps run in series through the base.
