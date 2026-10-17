# Depth offset in f, and a counts → p map h: design (PI, 2026-10-08)

Agreed with the PI fork by fork on 2026-10-08. Vault: questions q5 (the depth offset) and q6
(the map h) under q1; task t125, a child of t118. This run is **exploratory**: it gives readings only.
It has no pass/fail bars, no ticks and no verdicts. It follows the t118 grid
(`plan/T118_COUNTERFACTUAL_F.md`, `plan/T118_ROW2_SPEC.md`). The two questions it serves are under
the vault question about whether the recorded covariates carry enough information.

## Why

The t118 grid failed in two ways:

- No design follows the depth law: a depth-only change does not scale the predicted counts by d′/d.
- In −log10 p, some pairs explode, because a learned log-normal spread drifts.

This design addresses both failures. Part 1 handles depth with a fixed offset. Part 2 computes
p from counts and does not predict it directly.

## Part 1: the count model, eta in → eta out

- **Input to f:** eta_in = log(1 + X) − log d. Here d is the source depth in millions of reads
  (ε = 1, PI). The only input change from the old grid is a constant shift for each track.
- **Output:** eta_out = f(eta_in; θ). The target counts are NB with log μ′ = log d′ + eta_out.
- **f reads only eta_in and θ.** f does not read C.
- **g reads the target covariates C′.** It also keeps depth as an input (PI).
- **We expect this to add no capacity** when g reads both depths, and only to move g's "do nothing" output from
  "copy the counts" to "scale by d′/d".
- **Forms of f:** the four t118 designs A–D, in both rows: g(C, C′), and g(eta_in, C, C′).
- **Versions of g:** both, one g per track (7) and one g across tracks.
- **Spread (NB n):** each design's spread sees what its mean sees. A and B set it by level (a
  12-knot curve over eta_in). C and D set it by level and context: C reads its spread at the
  kernel-smoothed value, and D gets it from its CNN head.
  - This matches the old grid in every design except row-1 A. Row-1 A had one spread for each
    pair, and it now sets its spread by level, as A2 already did.
- **Old-spread A arm (PI):** row-1 A also runs with its old spread (one value per pair). The
  depth offset and the new spread rule are then separate for A. 8 × 2 × 3 = 48 runs.
- **Twin:** the no-covariates twin only.
- **Runs:** 2 rows × 4 designs × 8 versions of g × 2 models × 3 seeds = 384, plus 48 for the
  old-spread A arm: 432.
- **Loss:** NB negative log-likelihood, as before.

## Part 2: h, the predicted NB and the control → P = −log10 p

- **Training and scoring:** h is trained against the real p track and scored against it.
- **Uncertainty: no sampling is used.** X′ is a count, so we list k = 0, 1, … until the NB tail
  is below 1e-6. The predicted P is the set of values h(k), each with the weight NB(k).
  - The mean and the CRPS of this set are computed exactly.
- **Versions of h:**
  - **Formula only, nothing learned.** It is the caller's Poisson upper tail at the local control
    rate. It has two variants: bin-only, and a window one fragment wide (the caller's pileup).
  - **Learned, at two input ends (PI: "two ends only"):**
    - Rung 1 is the bin's count and the control count in the same bin.
    - Rung 5 is everything: the control windows, the genome background, both depths, the control
      ratio (called k in the t112 arms; renamed here, because k is a possible count), the
      fragment length and the fragment window.
    - Affine h and MLP h run at both ends. Formula + affine correction and formula + MLP
      correction run at rung 5 only, because the formula needs the rung-5 inputs.
- **One shared h, with the assay as an input.**
- **Uncertainty variants (both):**
  - Push only: h adds no noise.
  - Push + own noise: h outputs a Gaussian on log(1 + P) with a learned σ. Here σ is h's
    leftover error when the counts are known.
- **Training source (both):**
  - The true X′. This keeps h apart from f's errors.
  - f's predicted NB. This shows how much h can repair f. It uses seed 0 of each real f
    (64 models; PI).
- **Losses:**
  - Push only on true X′: MSE on log(1 + P).
  - Own noise: Gaussian NLL on log(1 + P).
  - Push only on f's NB: the expected MSE, Σ_k NB(k)·(log(1 + h(k)) − log(1 + P))².

## Scoring (both parts)

- **Every learned h is scored twice (PI):** on the true X′ (h alone), and in the chain on f's
  predicted NB. The repair reading then compares the h versions on the same input.
- **Depth law (PI):** on depth-only pairs, the predicted total count against the read ratio, as in
  the old grid. g still reads depth, so the offset does not force the law.
- **CRPS is quoted with its oracle-scaled and scale-error parts**, as the project rules require.
- **Top 1%** means the top 1% of bins by the true target value, as in the old grid.
- **Every reading is quoted with its seed spread.**

- **CRPS** on the native scale: counts, or P. This keeps the results comparable with the old grid.
- **MSE on log(1 + value) of the predicted mean.** This shows whether a gain comes from the mean
  or from the spread.
- **Both scores** are reported on all bins and on the top 1% of bins.
- **No correlation is a headline.** The protocol of the parent question forbids it.

## Same as the old grid (PI-confirmed defaults)

- The pairs, the chromosome split (score on chr19 + chr21, validate on chr22), three seeds,
  the g size (MLP 2 × 64) and the arms are the same as in the old grid.
- The formula follows the caller's own rules, including arms with no control.

## To verify before building

- How the t112 DNase p-value tracks were made. DNase has no control, so h's DNase path must
  follow that recipe. This is a step of the task; it is not part of either question.
- How the per-base p-values were binned to 25 bp in the t112 products: mean or max.
- The exact caller commands on the ChIP side: local-λ windows, control scaling and pileup
  extension.
