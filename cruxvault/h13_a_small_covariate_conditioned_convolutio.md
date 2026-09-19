---
id: h13
type: idea
schema: 2
title: A small covariate-conditioned convolutional map over neighbouring bins closes gap that the conditioned affine map leaves, and beats its shuffled-covariate twin by more than the seed floor
parent: q4
status: idea
rule: all
measurement: gap-closed on the onewarp→oracle interval, per signal decile and per mark class, on both output axes, over the counterfactual pairs of every knob, this rung against one-warp and against its shuffled-covariate twin; instrument is a standalone covariate-conditioned transformation testbed (encoder/decoder, or a lower-capacity conditioned map under q4) with log d pinned as a fixed offset of coefficient exactly 1, scored through bench.distributional.nb_suite and bench.covariate
replicates: "TODO(PI) — source→target pairs per arm x seeds, on the counterfactual pairs t112 built. Whatever it is, it must clear the seed floor: one paired seed change on the q19 recipe moves pooled imputation CRPS by 0.0463 (macro 0.0327) under eval.py and macro CRPS by 0.0608 under bench, and the per-track floor is several times either macro (AGENTS.md §7.2 rule 2)."
verdict: 
metric: 
created: "2026-09-17T21:47:47"
updated: "2026-09-17T21:47:48"
null_approved: "2026-09-17T21:47:48"
null_hash: 4c77931d52fdf6e4
---

# h13 — A small covariate-conditioned convolutional map over neighbouring bins closes gap that the conditioned affine map leaves, and beats its shuffled-covariate twin by more than the seed floor

Parent:: [[q4_how_much_capacity_does_a_covariate_condi]]

## ELI5
If a stretch and shift is not enough, letting the translator look at the neighbouring bins gets it further.

## TL;DR
The middle rung: a small convolutional network over a window of bins, its filters conditioned on the source and target covariates, fitted on the counterfactual pairs of [[t112_build_the_counterfactual_arms_for_t|the counterfactual-arms task]]. It distinguishes a covariate that acts through the local shape of the signal (fragment extension, read length) from one that acts per bin. The competitor — call it one-warp — is a single covariate-free monotone map on each output axis, fitted on training pairs pooled within this knob and scored on held-out pairs; the oracle is two independent NB draws from the same eta at the target depth (PI ruling 2026-09-16). Gap-closed is (D_onewarp − D_model) / (D_onewarp − D_oracle) with D the macro NB CRPS against the real target. The twin is the same network with the covariates shuffled across pairs. Supported if its gap-closed exceeds the affine rung's by more than 2× the model's own seed |Δ| on each output axis in every mark class, and its gain over the twin exceeds 2× seed |Δ|; refuted if either fails. Read only against the same pairs, competitor and seeds as the other rungs.

## Null
Capacity: a conditioned model of this size closes the same gap when given a shuffled covariate, so the covariate did no work.

## Problem Statement

Some knobs (extension, read length, MAPQ) change the pileup's local shape; a per-bin map cannot express that and a windowed map can.

## Idea / Hypothesis

A small covariate-conditioned convolutional map over neighbouring bins closes gap that the conditioned affine map leaves, and beats its shuffled-covariate twin by more than the seed floor

## Verifiables
<!-- Cloned 2026-09-17 from h1's PI-approved set; thresholds carried over, TODO(PI) where h1 says so. -->
- [ ] `twingap` [claim-directed, DISCRIMINATES against the capacity null] — gap-closed of this rung minus gap-closed of the same rung fitted with the covariates shuffled across pairs, per output axis and mark class; must exceed 2× the model's own seed |Δ|.
      *Fails if:* the shuffled twin closes the same gap, so the rung's capacity, not the covariate, did the work.

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
<!-- EVERY threshold below is deliberately TODO(PI). CLAUDE.md forbids inventing a gate. -->

**Outcome-neutral controls** — must pass whatever the claim turns out to be; a failure makes the
run `invalid-run`, never a refutation.

- [ ] `plantedrecovery` [outcome-neutral] — inject a synthetic covariate carrying a KNOWN monotone
      warp, and compare the recovered warp against the true one. Threshold: `TODO(PI)`.
      Follows `meta_probe.py`'s `off`/`shuffled`/`planted` discipline: the shift is added to the
      TARGET in log space only, `x_data` stays bit-identical, the draw comes from a dedicated RNG
      stream disjoint from the data stream, and the product is rounded back to an integer. It does
      NOT reuse `meta_probe` itself, which plants one bit (`MetadataEmbedding` raises above
      `num_runtypes = 2` and widening the table changes the parameter count).
      *Fails if:* the fitting procedure cannot recover a warp it was handed, so any warp it reports
      on real pairs is an artifact of the fit rather than a reading of the data.
- [ ] `shuffledrop` [outcome-neutral] — gap-closed with the covariate delta resampled from its own
      marginal across pairs (same marginal, zero association); must collapse. Threshold:
      `TODO(PI)`. **Calls `bench.covariate.covuse` / `_marginal_resample`; does not reimplement
      them.** Reports `within_batch_d_crps` (the structural tripwire, must read exactly `0.0`) and
      `conditional_null_degenerate` beside the result, so a p-value of 1.0 cannot be misread as the
      model ignoring the covariate when the truth is that the conditional null is empty. Watch for
      the degeneracy `meta_probe` measured on six consecutive real batches: with one biosample per
      batch, permuting along B is the identity.
      *Fails if:* gap-closed survives the shuffle, so the apparatus is reading something other than
      the covariates — capacity, position, or the source track itself.
- [ ] `latentguard` [outcome-neutral] — `bench.covariate.depthblind` + `biokeep` on `z`. Threshold:
      `TODO(PI)`. Proposed addition, not in the original set: the encoder is `z = ENCODER(x, x_md)`,
      and a collapsed encoder scores a perfect invariance. `depthblind` refuses to return without
      `biokeep` (D13) for exactly this reason.
      *Fails if:* the encoder is invariant because it collapsed, not because it is good.

**Claim-directed checks** — each reported as a **difference against the `onewarp` rung on held-out
pairs**, never in absolute terms. Measured absolutely, the null passes `qqresidual` and
`crps_split`, because `onewarp` is *fitted* to flatten the quantile curve.

- [ ] `gapclosed` [claim-directed, DISCRIMINATES against the null] — `(D_onewarp − D_model) /
      (D_onewarp − D_oracle)`, stratified per signal decile and per mark class. The distance `D` is
      macro NB CRPS against the real target (PI ruling 2026-09-16). **Threshold (PI ruling 2026-09-16):**
      passes when, in every mark class, gap-closed ≥ 0.5 AND the absolute gain
      `D_onewarp − D_model` exceeds 2 × the seed |Δ| of `D_model` — the paired |Δ| between two
      seeds of the same model recipe on the same pairs, measured before the real run in the
      way t86 measured the benchmark's floor. Plainly: the model must climb at least halfway
      from the rescale rung to the oracle rung, and the climb must be larger than the wobble a
      seed change alone produces; a smaller climb cannot be told from luck.
      *Fails if:* `onewarp` already closes the source-to-target gap and conditioning adds nothing
      beyond it, or the gain is inside the seed wobble.
- [ ] `qqresidual` [claim-directed] — max |log multiplier| of the post-model quantile-quantile
      curve MINUS `onewarp`'s own, with tail quantiles reported separately from the bulk.
      Threshold: `TODO(PI)`. Reuses the `calib_grid` grid convention rather than inventing a second
      one; note this is a value-axis QQ curve, a different object from `ece`'s PIT curve.
      *Fails if:* the fit is right on the bulk and wrong in the tail — the mean is matched and the
      peaks are not.
- [ ] `swapfidelity` [claim-directed] — feed the decoder the SOURCE covariates while the target is
      the other arm; it must predict the SOURCE, scored as gap-closed toward it. **Read jointly
      with `gapclosed`** (see the two failure readings below). Threshold: `TODO(PI)`. Reuses
      `covuse`'s scoring half — `nb_crps_mean` against a chosen target and the randomization-test
      p-value form `(1 + #{L_r ≤ L_obs}) / (R + 1)`.
      *Fails if:* told the source covariates, the decoder still predicts the target, so its output
      does not depend on what it is told.

**Combination rule: `all`**, over the three claim-directed checks. The cost, stated rather than
hidden: three checks at 80% power each give **51% joint power**, and the thresholds may not be
loosened to compensate.

**Reported, never gating.** These are recorded in the run report and do not enter the verdict.

- `crps_split` — `crps`, `crps_oracle_scaled` and `scale_error` from
  `bench.distributional.nb_suite`, with `onewarp` scored through the point-to-distribution spread
  device. Quote all three or none (`AGENTS.md` §7.2). `crps_oracle_scaled` is an **in-sample upper
  bound** (`c*` is fitted on the same targets it scores) and `scale_error` can go slightly negative
  (−0.0008 observed).
- `assayresidual` — variance of gap-closed ACROSS assays WITHIN a signal decile. **Moved out of the
  gate deliberately**: the claim is that conditioning closes the gap, not that the closure is
  uniform across assays, so as a gating check it could refute the hypothesis for something the
  hypothesis never asserted. Computed by running `bench.covariate.covshare(scalar="level")` inside
  each decile rather than with a new estimator — but note the open bug `t43`, which leaks
  across-unit variance into that estimator's bias term.
- `composegap` — fit on the two 1-D sweeps only (depth alone, run type alone), compose, and predict
  the held-out joint 2x2 corner.
- `sweepcurve` — gap-closed against the number of distinct covariate values seen, with
  interpolation and extrapolation reported **separately**.
- **Broad marks are diagnostic, never gating.** Their level barely moves, so a ratio there is a
  small number over a smaller one.

**Two pre-committed failure readings.**

1. *Reading the covariates as nuisance* — `gapclosed` high with `swapfidelity` at chance. The
   decoder learned one fixed direction-of-travel map for this pair set rather than a
   covariate-steered one. This is the `p(y|c) = p(y)` failure `bench/covariate.py` names in its
   opening docstring.
2. *Copying the source* — `swapfidelity` high while `gapclosed` is low and `blind` is
   indistinguishable from `model`. Note this is **not** "swapfidelity at chance": a copier emits
   the source, which is exactly what the swap asks for, so a copier scores HIGH on `swapfidelity`.
3. *Unrecorded covariates* — `gapclosed` high on synthetic pairs but collapsing on real ones. The
   real difference carries something the synthetic corruptions do not, pointing at antibody, lab or
   protocol. `lab` and `sequencing_platform` are in fact recorded in the metadata CSVs, so two of
   those are testable rather than merely nameable.

**Open decisions for the PI, blocking `test --to running`:**

- (i) **decided — PI ruling 2026-09-16: `D` is macro NB CRPS against the real target**, not the
  level-only `aspects_of(...)["level"]` reading;
- (ii) **decided — PI ruling 2026-09-16: the `oracle` rung is two NB draws.** The model emits NB, and on an
  overdispersed target a Poisson oracle would sit below anything reachable, leaving gap-closed no
  ceiling of 1;
- (iii) **decided — PI ruling 2026-09-16: `onewarp` is the headline denominator; `splitwarp` is not a rung;**
- (iv) every `TODO(PI)` threshold above.

**The null was approved on 2026-09-16 (`crux approve-null`), so checks may be written against it.**
Every bar in the checks above is still `TODO(PI)`, so nothing is locked yet; the checks, their kinds
and the rule content-hash when the node goes running.

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h13/ and link at least the report:
     - [Report](results/h13/report.md)   - results/h13/curve.png -->
_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_
