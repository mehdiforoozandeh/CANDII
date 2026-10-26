---
type: wiki
title: Pseudocount log ratios carry a signal-level curve
summary: A per-bin log ratio with a +1 pseudocount has an expected value that depends on the bin's signal level, so two independent halves of the same pair agree even when the true effect is flat.
category: concept
sources: 
created: 2026-10-26T13:30:00
updated: 2026-10-26T13:30:00
---

# Pseudocount log ratios carry a signal-level curve

Any bin-by-bin comparison of two count tracks can fall into it, with no biology involved.

## What it is

Take R = log2((arm + 1)/(base + 1)) per bin. If the arm is the base thinned to a fraction f, the bin's mean signal is μ, and counts are large enough, then E[R] ≈ log2((fμ + 1)/(μ + 1)). It is near 0 in empty bins and near log2 f in strong bins, because the +1 compresses low counts. μ is the same in both pseudoreplicate halves, so both halves trace the same curve. Two halves of a pure thinning pair therefore correlate. Median-centring removes a constant, not a curve.

## Evidence (CANDII split-half test, chr21, 1 kb)

Pure thinning pairs looked reproducible: Spearman r between halves was 0.34-0.52 at 7.5M reads and 0.48-0.63 at 3.75M, above a shifted-block null p99 of 0.32-0.40. After subtracting a binned-median curve fitted against the other half's base level, r fell to -0.11 to -0.37 (null p99 about 0.01). The curve explains the agreement, but this correction overshoots below zero.

## When it bites

- Any per-bin log ratio with a pseudocount.
- Split-half reproducibility tests of an effect.
- Effect maps and ranks of "where the arm differs".

## The fix

Model E[R | μ] and test the residuals, with μ taken from data that shares no noise with either ratio. Check on a pure-thinning control that the residual r lands at the null, not below it. A count model with depth as an offset avoids the pseudocount entirely.

## See also

Related:: [[sequencing-depth-and-coverage]], [[count-distributions-for-sequencing-data]]
