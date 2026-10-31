---
id: h27
type: idea
schema: 2
title: f at 64 channels (2x wider), with the same g hidden width, training pairs and loss, meets the depth law on never-trained depth pairs
parent: q7
status: done
rule: all
measurement: 
replicates: 7 per-track g x 3 seeds (phase 3); 3 more seeds planned
verdict: refuted
metric: never-trained depth-law error 0.41 (per-track g) / 0.34 (across) vs trained 0.05; six-seed CRPS gain through the base 0.006, within seed noise
created: "2026-10-29T14:25:51"
updated: "2026-10-31T00:33:38"
lock: 48f1873f2633c1a4
locked: "2026-10-31T00:23:17"
lock_at: close
null_approved: "2026-10-31T00:33:38"
null_hash: 24641e914971592d
---

# h27 — f at 64 channels (2x wider), with the same g hidden width, training pairs and loss, meets the depth law on never-trained depth pairs

Parent:: [[q7_why_do_the_covariate_conditioned_g_f_des]]

## ELI5

The network that turns one track into another may be too narrow to learn how sequencing depth scales the counts.

## TL;DR

f, the convolutional network from source track to target track, may be too narrow to carry the depth law onto never-trained pairs. At 64 channels (old D: 32), with the same g hidden width, training pairs and loss, f should bring never-trained depth pairs onto the law; a miss refutes it at that width. Exploratory; sibling of [[h20_a_larger_g_with_the_same_training_pairs_|the g-size hypothesis]].

## Expect

With f at 64 channels and g's hidden width unchanged, never-trained depth → depth pairs fall to the trained-pair level (about 0.1).

_Written after the run, at the PI's request on 2026-10-31. The standard-error bar for the three extra seeds (a real gain above 0.013) was set before that follow-up ran._

## Null

Chance: a single seed or track happens to land near the law; the median over 3 seeds and 7 tracks guards against it.

## Problem Statement

Every g → f design meets the depth law on trained base ↔ arm pairs and misses it on never-trained arm → arm pairs ([[h19_g_fits_trained_base_arm_depth_pairs_but_|the trained versus never-trained hypothesis]]). Width is a cheap suspect to rule out before a structural fix.

Built-in confound: g sets one FiLM scale and shift per channel per layer, so doubling f's channels doubles g's output layer (256 → 512 values; g grows from 23 k to 40 k parameters). A pass cannot separate f's width from that larger g head.

## Idea / Hypothesis

Old D's f is 4 dilated convolution layers (dilations 1, 2, 4, 8; kernel 5; a 61-bin field) at 32 channels, about 16 k parameters; at 64 channels it has about 62 k. If width limits f, the wider f closes the never-trained gap more than it changes the trained-pair error.

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
Checks written after the run, at the PI's request on 2026-10-31; the work was exploratory with no bar set beforehand.

- [ ] With f at 64 channels (g hidden width 64), the never-trained depth → depth median |log2 error| falls to the trained-pair level (about 0.1) for one g per track or one g across tracks
      fails-if:: Both g versions stay well above their trained-pair level
      discriminates:: true
      (found: 0.41 per-track g, 0.34 across-track g; trained 0.05 — [depth-law table](results/t128p3/m5_depthlaw_table.tsv))
- [x] [outcome-neutral] The target tracks themselves obey the depth law on the scored bins
      fails-if:: Σy/ΣX differs from the manifest depth ratio by more than 0.01 log2
      (found: within 0.004 log2, as for the loss hypothesis; the data audit gives 0.71% over 84 ordered pairs)

## Planned Intervention

Phase 3 of the pivot study ran it (arm "bigger f": 7 per-track g + 1 across-track g, 3 seeds, 15,000-step cap, early stop). Phase 3b (PI, 2026-10-29) added 3 seeds per per-track g for bigger f and for old D (42 runs, done 2026-10-31), read with a standard-error bar fixed before the run. Spec: plan/T126_ENCDEC_D2.md § Phase 3.

## Run Links

- Fir: /project/def-maxwl/mforooz/t128p3 (runs/Dbigf_*, viabase/Dbigf_*); train 758805, law 758806.
- Fir: /project/def-maxwl/mforooz/t128p3/seeds3b (seeds 3–5, both arms); train 1211830/1211840, law 1211839/1211842.

## Artifacts

- [Phase-3 deck: a bigger g and a bigger f](presentations/t128p3-scale/DECK.md)
- [Phase-3 depth-law table (arm Dbigf)](results/t128p3/m5_depthlaw_table.tsv)
- [Phase-3 ranking table](results/t128p3/rank_table.tsv)
- [Phase-3b six-seed headline table](results/t128p3b/headline_table.tsv)
- [Phase-3b checks](results/t128p3b/checks.txt)

## Findings

Exploratory. Median |log2 error| of the predicted scale, chr19 + chr21, seed spread in brackets:

- Never-trained depth → depth: 0.41 (0.23) per-track g, 0.34 (0.10) across-track g; old D 0.29 (0.05), 0.34 (0.24); within noise. Through the base: 0.09 and 0.11; old D 0.11 and 0.07.
- Never-trained arm → arm CRPS through the base, per-track g, 3 seeds: 0.530 (0.526 + 0.004; spread 0.005) against old D's 0.545 (0.537 + 0.008; spread 0.008); pre-run bar 0.015.
- 6 seeds: 0.533 (0.528 + 0.005) against 0.539 (0.532 + 0.007); standard error 0.003, bar 0.013. The three new seeds alone favour old D (−0.003). Both controls pass: the reference's new seeds match its old ones (Δ −0.011, bar 0.012), and the readout reproduces a phase-2 run exactly. On trained pairs the reference's new seeds read 0.009 lower (bar 0.004, mostly deduplication and paired end); PI ruling 2026-10-31: a draw of seeds.
- Trained-pair and held-out 7.5M CRPS are −0.009 and −0.005 against old D, per-track g, both passing the noise rule.
- Closure was option A: no wider f or learning-rate sweep follows.
