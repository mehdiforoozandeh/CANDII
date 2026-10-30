---
id: q11
type: question
schema: 2
title: "Where along the genome does each covariate act: does a track react uniformly to a change in depth, read length or another processing knob, or more in some kinds of regions?"
parent: q9
status: open
stale: false
created: "2026-10-29T21:25:30"
updated: "2026-10-29T21:25:30"
---

# q11 — Where along the genome does each covariate act: does a track react uniformly to a change in depth, read length or another processing knob, or more in some kinds of regions?

Parent:: [[q9_which_recorded_processing_covariates_mov]]

## ELI5

When we change one lab or processing setting, does the signal change by the same amount everywhere in the genome, or more in some kinds of regions (hard-to-map, GC-rich, repeats, peaks)?

## TL;DR

A covariate whose effect is uniform along the genome can be corrected by one genome-wide monotone map (the one-warp baseline); an effect that varies by region needs a position-aware correction, which is what CANDI's covariate conditioning would then have to learn. This asks, for each t112 knob (depth, pe, crop/read length, dedup, mapq, abproxy, extsize; ctlid, ctldepth and ratio in −log10 p only), how much of the arm-vs-base change per 1 kb bin is left after the genome-wide one-warp fit, and whether that residual tracks mappability, GC, repeats or peak-vs-domain context once signal level is fitted first as the null. Same-read knobs have almost no sampling noise, so residual structure is a real covariate effect; the depth arm is the control, read against exact binomial thinning of the base, and should show none. Settled by per-knob, per-mark-class residual size against the pseudoreplicate floor at matched depth, and the feature gain over the level null.

## Question

Where along the genome does each covariate act: does a track react uniformly to a change in depth, read length or another processing knob, or more in some kinds of regions?

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
PI ruling 2026-10-29: (1) the uniform part is removed by the one-warp (monotone base→arm map fitted genome-wide), not by a single scale factor; (2) all knobs, counts and −log10 p, control knobs in p only; (3) the depth arm is the control, its null is the base thinned binomially to the arm's depth; (4) signal level is fitted first as the null, as in the regional floor map; (5) every reading quoted beside the pseudoreplicate floor map at matched depth. Exploratory: readings only, no verdict.

- **Blacklist** (foreman amendment A8, 2026-10-30, within ruling 4). The hg38 v2 blacklist is masked in both 1 kb maps: this covariate map and the pseudoreplicate floor map.
- **Floor join** (foreman decision 2026-10-30, within ruling 5). The floor map exists at two depths only, so each knob reading sits beside the floor at the nearest of them, labelled "exact" or "nearest".
- **Depth control by whole reads** (PI ruling 2026-10-30, replaces the null of ruling 3). The depth arm's null is rebuilt by thinning whole reads of the base to the arm's depth, not 25 bp count units. The unit-wise null was about √k too narrow, where k (4–5) is the number of count units one read adds.

## Answer so far

Readings, not judged ([covariate-map digest](results/t141/DIGEST.md)). In every knob, signal level explains more of the one-warp residual (R² 0.08–0.60) than the four features add.

- Not uniform: mapping quality 0 (feature gain 0.10–0.11 R², mostly mappability; +0.45 log2 in the lowest-mappability decile), read length 36 and 50, paired-end in DNase, and antibody proxy f0.9 (peaks and context). Their feature gain is 50–1000× the floor's (≤ 0.0001). In −log10 p: control identity "none" and control ratio k0.5 (domains and peaks).
- Uniform, so one genome-wide monotone map suffices: duplicate removal and paired-end in histone marks, depth, fragment extension, and control depth.
- Caveat: the depth control failed. The depth arm's residual read 1.9–2.6× its null, for a known reason: the null thinned count units, not reads. The control is being redone by whole-read thinning, and the depth reading waits on it.

<!-- crux:ledger:start -->
_(no children yet)_
<!-- crux:ledger:end -->
