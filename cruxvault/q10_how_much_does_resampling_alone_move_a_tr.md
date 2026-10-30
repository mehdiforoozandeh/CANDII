---
id: q10
type: question
schema: 2
title: How much does resampling alone move a track's shape and magnitude, where in the genome is that noise concentrated, and how does it shrink with depth?
parent: q1
status: open
stale: false
created: "2026-10-29T20:07:43"
updated: "2026-10-29T20:07:43"
---

# q10 — How much does resampling alone move a track's shape and magnitude, where in the genome is that noise concentrated, and how does it shrink with depth?

Parent:: [[q1_do_the_recorded_experimental_covariates_]]

## ELI5

If you split one experiment's reads into two random halves, the two halves already disagree a little, and no correction can do better than that disagreement.

## TL;DR

Every covariate correction is read against a floor: the change that pure read resampling causes with no covariate changed. This asks three things on the t117 pseudoreplicates (pr1/pr2 halves of all 130 counterfactual products), thinned along a depth ladder: how much resampling moves peak height, 1 kb shape and per-bin divergence at each depth; where in the genome the split-half disagreement concentrates; and how it shrinks with depth, read off the curve at the 30M the arms are built at. Signal level is fitted first as the null, since counting noise alone makes low-count bins irreproducible; mappability and repeats, GC, and peak versus domain context are read on the residual. It is settled when the floor at 30M is stated per mark class with its depth curve, and the residual regional pattern is either named or shown to be absent. The floor was set as the oracle for every reading under q1, beside the monotone scale map and the shuffled-covariate twin (PI ruling 2026-10-29). For pairs that share reads it is context only (PI ruling 2026-10-30; see Protocol).

## Question

How much does resampling alone move a track's shape and magnitude, where in the genome is that noise concentrated, and how does it shrink with depth?

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
Locked before any run: draws are the t117 halves thinned to a depth ladder (no new alignment); signal level is the null and is fitted before any regional feature is read; every q1 reading is scored after this floor lands, against all three baselines.

- **Ladder** (PI ruling 2026-10-29). Each t117 half is thinned to ratios 1, 0.5, 0.25 and 0.125. Thinning seeds are 0 and 1 below ratio 1; ratio 1 has no draw. The generator is `default_rng([seed, pid_index, depth_index])`.
- **Reads per half** (foreman decision 2026-10-29, within the ladder ruling). The x-axis is ratio × the product's own half size, read from `n_lines` in the pseudoreplicate MANIFEST, not a constant 15M. The halves are 15M in 102 products, 25M in the 7 DNase products, and 7.5M, 3.75M and 1.875M in the 7 products of each depth arm.
- **Depth curve.** Per product and metric, log(metric) = α + β · log2(reads per half), fitted over the ladder and read at 30M, the depth the arms are built at. The 30M value is an extrapolation above every ladder.
- **Metrics** (counts, chr19 + chr21, hg38 v2 blacklist removed): peak height, |log2 fold-change| over the full product's top-1 % bins (one set for every ladder point); 1 kb shape, 1 − Spearman; per-bin divergence, mean |log2((a + 0.5)/(b + 0.5))|; and the CRPS of one half predicting the other with a Poisson spread, both directions. Every CRPS carries its `crps_oracle_scaled` / `scale_error` split. Check: at ratio 1 the floor CRPS reproduces the rungs_v2 oracle (0 FAIL).
- **−log10 p floor at 15M only** (PI ruling 2026-10-29, Decision ii). It is read from the rungs_v2 oracle and flagged on every p row. A p-space ladder needs MACS2 reruns on thinned reads, which is a separate decision.
- **Regional map.** The level null is the mean divergence in 40 quantile bins of signal level (foreman decision 2026-10-29: a median null gave negative R²). GC, Umap k36 mappability, RepeatMasker fraction and peak-vs-domain context are read on the residual (all four, PI ruling 2026-10-29); the annotations were downloaded to Nibi (PI, Decision iii). The blacklist is masked in the map too (foreman amendment A8, 2026-10-30).
- **Role of the floor** (PI ruling 2026-10-30). The floor is the "fresh replicate" context column: what an independent replicate would differ by. It is NOT the oracle for nested pairs. The depth arms are read subsamples of the base, so they share reads with it, and their oracle is the exact nested-read oracle ([[h1_conditioning_on_the_recorded_sequencing_|the depth claim]], item v). The floor stays the oracle where pairs are not nested (run type) and in −log10 p.
- **Whole-read thinning** (PI ruling 2026-10-30). The thinned ladder points, and every 1 kb reading built from them, are rebuilt by thinning whole reads from the split read files. Reason: `counts25` adds 1 to every 25 bp bin that a read overlaps (about k = 4 units per read in DNase, 5 in histone marks). Thinning units one at a time drops parts of reads, so 1 kb sums come out about √k too smooth. The ratio-1 rows and the per-bin metrics (divergence, CRPS) are not affected.
- **Real-halves check** (PI ruling 2026-10-30, iii). The real depth-arm halves (7.5M, 3.75M and 1.875M reads) are compared with the rebuilt thinned floor at the same depth.

## Answer so far

Readings, not judged ([floor digest](results/t137/DIGEST.md), [map digest](results/t138/DIGEST.md)):

- Floor at 30M, extrapolated, 1 kb shape: narrow 0.552, broad 0.494, DNase 0.121 (seed spread ≤ 0.0015). It shrinks with depth in 130 of 130 products. The peak-height floor is ≤ 0.014 log2 at every depth.
- Per-bin divergence and CRPS grow with depth, because they are on the raw-count scale; they are not depth-free noise levels. The floor CRPS is mostly shape: narrow at 15M, 0.553 with oracle_scaled 0.484 and scale_error 0.063.
- Where: signal level explains 0.18–0.38 of the per-bin disagreement at 15M. The four features add ≤ 0.0003 together (shuffle ≤ 0.00003). The one tail: low-mappability bins hold 1.4–1.5× their share of the worst bins in histone marks.
- Caveat: the thinned 1 kb readings and the 3.75M map used unit-wise thinning, which is too smooth. They are being rebuilt by whole-read thinning.

<!-- crux:ledger:start -->
_(no children yet)_
<!-- crux:ledger:end -->
