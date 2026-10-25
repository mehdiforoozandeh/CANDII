---
id: h10
type: idea
schema: 2
title: Conditioning on the recorded fragment extension predicts a target track from a source track, beyond what a single value-axis map already does
parent: q3
status: idea
rule: all
measurement: gap-closed on the onewarp→oracle interval, per signal decile and per mark class, on the −log10 p axis, over the fragment extension counterfactual pair(s) against base; instrument is a standalone covariate-conditioned transformation testbed (encoder/decoder, or a lower-capacity conditioned map under q4) with log d pinned as a fixed offset of coefficient exactly 1, scored through bench.distributional.nb_suite and bench.covariate
replicates: "TODO(PI) — source→target pairs per arm x seeds, on the counterfactual pairs t112 built. Whatever it is, it must clear the seed floor: one paired seed change on the q19 recipe moves pooled imputation CRPS by 0.0463 (macro 0.0327) under eval.py and macro CRPS by 0.0608 under bench, and the per-track floor is several times either macro (AGENTS.md §7.2 rule 2)."
verdict: 
metric: 
created: "2026-09-17T21:46:17"
updated: "2026-09-17T21:47:48"
null_approved: "2026-09-17T21:47:48"
null_hash: 8e4d67b544ab2035
---

# h10 — Conditioning on the recorded fragment extension predicts a target track from a source track, beyond what a single value-axis map already does

Parent:: [[q3_do_the_covariates_that_record_how_the_co]]

## ELI5

Knowing how far each read was stretched into a fragment closes at least half of the p-value gap a plain rescale leaves. Untested; on new pairs the model's p-values blow up.

## TL;DR

The pair is the base track and a counterfactual with reads extended to 0.5× or 2× the cross-correlation fragment length (DNase: the smoothing window) ([[t112_build_the_counterfactual_arms_for_t|the counterfactual-arms task]]). The counts are bit-identical to base, so the claim lives on the −log10 p axis. The bar is [[q1_do_the_recorded_experimental_covariates_|the parent question's locked protocol]] (gap-closed against one-warp ≥ 0.5 per mark class, gain above 2× seed |Δ|); no run has scored it. Exploratory readings: a per-pair monotone map already beats the pseudoreplicate oracle in p. A generator g that reads the extension beats its twin in p on trained pairs in 10 of 16 configurations; on the never-trained pair half the configurations explode.

## Null
Normalization: the pairs differ by one global monotone value-axis map, so a covariate-free rescale already closes the gap and the conditioning adds nothing.

## Problem Statement

Extension smooths the pileup; a longer extension merges neighbouring peaks and a shorter one sharpens them, a shape change no value-axis map captures. It is one of four control-and-caller knobs under [[q3_do_the_covariates_that_record_how_the_co|the control-and-caller question]], and the only one with DNase products.

## Idea / Hypothesis

Conditioning on the recorded fragment extension predicts a target track from a source track, beyond what a single value-axis map already does

## Verifiables
<!-- Cloned 2026-09-17 from h1's PI-approved set with the knob = fragment extension; thresholds carried over, TODO(PI) where h1 says so. -->

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
- [ ] `countsfixed` [outcome-neutral] — the model's count prediction for the target equals its prediction for the source within the seed floor, because the treatment counts are bit-identical to base by construction. Threshold: `TODO(PI)`.
      *Fails if:* the transformation moved the counts to fit the p-values, so the run is invalid, not a refutation.

## Planned Intervention

No run has tested this claim: [[t103_build_the_in_vitro_covariate_testbed|the testbed build]] has not run on this arm, and [[t107_run_the_in_vitro_covariate_testbed_o|the testbed run]] covers only depth and run type. The readings come from runs that trained one generator g on all ten knobs at once: the g × f grid, rows 1 and 2 ([[t118_learn_the_covariate_conditioned_tra|the grid task]]; `plan/T118_COUNTERFACTUAL_F.md`), and the encoder-decoder rungs ([[t126_build_and_run_the_encoder_decoder_r|the encoder-decoder task]]; `plan/T126_ENCDEC_A.md`). Both score chr19 + chr21 over 3 seeds against a no-covariates twin.

## Run Links

- g × f grid rows 1 and 2: `nibi:/project/def-maxwl/mforooz/t118/{ladder,row2}/runs` (row-2 arrays 22855414, 22862752; knob extraction 24641277)
- Encoder-decoder: `nibi:/project/def-maxwl/mforooz/t126/{runs,D/runs}` (train 24646366, 24647762; law 24646367, 24647788, 24706070)
- Products: `nibi:/scratch/mforooz/t112_cf/products/<track>__extsize__{k0.5,k2}/` (scratch purge 2026-11-20)

## Artifacts

<!-- what the run produced. Keep files under results/h10/ and link at least the report:
     - [Report](results/h10/report.md)   - results/h10/curve.png -->
- [Per-knob readings for this knob family](results/q3/KNOB_READINGS.md) — the tables, the CRPS split and the caveats
- [Baseline rungs by arm](results/t118/rungs_v2.md)
- [Grid knob table, rows 1 and 2](results/t118/knob_generalization/knob_table.tsv) and its [figures](results/t118/knob_generalization/report.pdf)
- [Encoder-decoder knob table](results/t126/knob_table.tsv) and its [report](results/t126/report.pdf)
- [Grid deck](presentations/t118-grid/index.html) — published https://claude.ai/artifact/1mFz6C5jLaG1Ctc8JN572C
- [Encoder-decoder deck](presentations/t126-encdec/index.html) — published https://claude.ai/artifact/KP7pipaRbNbEStUaJR7jrZ
- [Counterfactual-arms manifest](results/t112/MANIFEST.tsv) and [summary](results/t112/COUNTERFACTUAL_ARMS.md) — published https://claude.ai/code/artifact/fb5ff45d-f155-4b01-ac50-db1179f4fd61

## Findings

Observed 2026-10-01 to 2026-10-21; not concluded. In p, QuantileMatching, a per-pair monotone map, scores past the pseudoreplicate oracle (share 1.69), so that oracle is no ceiling. Grid, gain over the twin, median of 16 configurations, bar 2× seed wobble: p, trained 0.20 (10/16 met, median wobble 0.028; 4 cells explode); never-trained unusable, 8 of 16 cells explode; count-identical pairs, trained 0.36 (14/16), never-trained 0.28 (7/16). Encoder-decoder, count-identical pairs: old design D 0.24 trained and 0.22 never-trained, learned-decoder rung 0.10–0.14, old design A −0.06 never-trained. No run measures the count-unchanged control directly. Tables: [knob readings](results/q3/KNOB_READINGS.md).
