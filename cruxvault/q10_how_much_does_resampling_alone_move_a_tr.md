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

Every covariate correction is read against a floor: the change that read resampling alone causes. This asks three things on the t117 pseudoreplicates (pr1/pr2 halves of all 130 counterfactual products), thinned along a depth ladder: how much resampling moves peak height, 1 kb shape and per-bin divergence; where in the genome the split-half disagreement concentrates, with signal level fitted first as the null; and how it shrinks with depth, read at 30M. It is settled when the 30M floor is stated per mark class with its depth curve, and the regional pattern is named or shown to be absent. The floor is the oracle for pairs that share no reads and context for pairs that do (PI rulings 2026-10-29, 2026-10-30; see Protocol).

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

Readings, not judged, 2026-10-30 ([rebuilt floor and map](results/t142/DIGEST.md); first pass: [floor](results/t137/DIGEST.md), [map](results/t138/DIGEST.md)).

- 1 kb shape floor, whole-read thinning: at 30M (extrapolated) narrow 0.533, broad 0.485, DNase 0.114; at 3.75M per half 0.787, 0.788, 0.395. Seed spread ≤ 0.003. The curve is concave, so read the histone 30M value as 0.43–0.53. The real depth-arm halves match it (ratio 1.002, 21 arms).
- Superseded: the first-pass thinned values (3.75M: 0.687, 0.657, 0.245; real/thinned 1.226). Unit thinning split reads, so 1 kb sums were about √k too smooth (k = 4–5). The 15M rows and the per-bin metrics stand.
- The peak-height floor is ≤ 0.02 log2, with a seed spread of the same size. Per-bin divergence and CRPS grow with depth (raw-count scale). Floor CRPS, narrow 15M: crps_all 0.553, oracle_scaled 0.484, scale_error 0.063 (r0.5 seed spread ≤ 0.0003).
- Where: signal level sets the noise, not region. Level explains 0.18–0.38 of the per-bin disagreement at 15M and 0.21–0.32 at 3.75M; the four features add ≤ 0.0003 R² (shuffle about 0.00001). Small tails: low-mappability bins at 15M (1.4–1.5× of the worst 1 %), peaks and domains at 3.75M (1.2–1.8×).
- For CANDI: a level-dependent spread, such as the NB variance, describes the floor. Low-depth noise is larger than first measured.

<!-- crux:ledger:start -->
_(no children yet)_
<!-- crux:ledger:end -->
