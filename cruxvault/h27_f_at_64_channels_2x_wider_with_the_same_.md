---
id: h27
type: idea
schema: 2
title: f at 64 channels (2x wider), with the same g hidden width, training pairs and loss, meets the depth law on never-trained depth pairs
parent: q7
status: idea
rule: 
measurement: 
replicates: 7 per-track g x 3 seeds (phase 3); 3 more seeds planned
verdict: 
metric: 
created: "2026-10-29T14:25:51"
updated: "2026-10-29T14:25:51"
---

# h27 — f at 64 channels (2x wider), with the same g hidden width, training pairs and loss, meets the depth law on never-trained depth pairs

Parent:: [[q7_why_do_the_covariate_conditioned_g_f_des]]

## ELI5

The network that turns one track into another may be too narrow to learn how sequencing depth scales the counts.

## TL;DR

f, the convolutional network that maps a source track to a target track, may be too narrow to carry the depth law onto pairs it never trained on. This claims that f at 64 channels, twice the 32 of old D, with the same receptive field, g hidden width, training pairs and loss, brings never-trained depth pairs onto the law. A miss at 64 channels refutes it at that width. Exploratory, like its parent: readings, no bars. It is the sibling of [[h20_a_larger_g_with_the_same_training_pairs_|the g-size hypothesis]].

## Null

<!-- spec 09: the null hypothesis — the most parsimonious alternative explanation under
     which this result appears although the claim is false. One line, <=25 words, naming a family. Your checks must discriminate against it.
     The PI approves it before checks are written: `crux approve-null <id>`. -->
_(one line: the null hypothesis, the most parsimonious alternative explanation — name a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

The depth law: when two runs of one sample differ only in sequencing depth, the predicted count scale (log2 of the summed predicted means over the summed source counts) should equal the log2 read ratio. Each track was reprocessed into arms that change one setting from a base; training uses base ↔ arm pairs only. Every g → f design meets the law on trained pairs and misses it on never-trained arm → arm pairs ([[h19_g_fits_trained_base_arm_depth_pairs_but_|the trained versus never-trained hypothesis]]). Width is a cheap suspect to rule out before a structural fix.

One confound is built in: g sets one FiLM scale and one shift per channel per layer, so doubling f's channels doubles g's output layer (256 → 512 values; g grows from 23 k to 40 k parameters). A pass cannot separate f's width from that larger g head.

## Idea / Hypothesis

Old D's f is 4 dilated convolution layers (dilations 1, 2, 4, 8; kernel 5; a 61-bin field) at 32 channels, about 16 k parameters. At 64 channels it has about 62 k, with the same layers and field. If width limits f, the wider f closes the never-trained gap more than it changes the trained-pair error.

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] _(state a falsifiable, pre-registered check)_

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

Exploratory readings; the PI has not ruled. Median |log2 error| of the predicted scale, chr19 + chr21; seed spread in brackets.

- On never-trained depth → depth pairs, f at 64 channels misses the law: 0.41 (0.23) per-track g, 0.34 (0.10) across-track g, against old D at 0.29 (0.05) and 0.34 (0.24). The gaps are within noise. It fits trained pairs (0.05).
- Read through the base it stays near the law: 0.09 and 0.11, as old D (0.11 and 0.07).
- On never-trained arm → arm CRPS through the base, per-track g, it gives 0.530 (0.526 + 0.004; spread 0.005) against old D's 0.545 (0.537 + 0.008; spread 0.008). The gain of 0.0146 was just under the bar set before the run (0.015).
- With 6 seeds the gain shrinks to 0.006: bigger f 0.533 (0.528 + 0.005) against old D's 0.539 (0.532 + 0.007). The standard error is 0.003, the bar for a real gain 0.013, and the 95% interval −0.002 to +0.013: within seed noise. On the three new seeds alone, bigger f is slightly worse (−0.003). Both controls pass: the reference's new seeds match its old ones (Δ −0.011, bar 0.012), and the readout reproduces a phase-2 run exactly.
- Trained-pair and held-out 7.5M CRPS are slightly lower than old D's with per-track g (−0.009 and −0.005), both passing the noise rule.
