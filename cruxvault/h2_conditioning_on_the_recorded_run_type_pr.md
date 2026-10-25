---
id: h2
type: idea
schema: 2
title: Conditioning on the recorded run type predicts a target track from a source track, beyond what a single value-axis map already does
parent: q1
status: idea
rule: all
measurement: gap-closed on the onewarp→oracle interval, per signal decile and per mark class, over a paired-end arm against its single-end rebuild; instrument is a standalone encoder/decoder testbed with log d pinned as a fixed offset of coefficient exactly 1, scored through bench.distributional.nb_suite and bench.covariate
replicates: "TODO(PI) — source→target pairs per arm x seeds, on the panel t100 selects. Whatever it is, it must clear the seed floor: one paired seed change on the q19 recipe moves pooled imputation CRPS by 0.0463 (macro 0.0327) under eval.py and macro CRPS by 0.0608 under bench, and the per-track floor is several times either macro. The ~0.09 target-clustered floor and the ±0.13 per-comparison uncertainty belong to the frozen full-EIC panel and are quoted only for it (AGENTS.md §7.2 rule 2)."
verdict: 
metric: 
created: "2026-09-01T02:16:44"
updated: "2026-09-16T16:52:41"
null_approved: "2026-09-16T16:52:41"
null_hash: 8e4d67b544ab2035
---

# h2 — Conditioning on the recorded run type predicts a target track from a source track, beyond what a single value-axis map already does

Parent:: [[q1_do_the_recorded_experimental_covariates_]]

## ELI5

Knowing whether the sequencer read DNA fragments from one end or both is enough to turn one experiment's numbers into the other's, and by more than plain rescaling.

## TL;DR

An encoder reads a source track and its covariates; a decoder reads the target's covariates and emits raw counts and −log10 p, with log depth pinned as an offset of coefficient exactly 1. The competitor is one-warp, a covariate-free monotone map on the value axis. The claim is that run-type conditioning closes the gap one-warp leaves. This design was never run; [[h15_one_transformation_conditioned_on_the_so|the counterfactual-arms claim]] asked the run-type question instead.

## Null
Normalization: the pairs differ by one global monotone value-axis map, so a covariate-free rescale already closes the gap and the conditioning adds nothing.

## Problem Statement

Run type carries more weight than depth: it is not a thinning identity. [[signal-normalization-in-epigenomics]] records that the single-end and paired-end deduplication rules differ, and that this shift let a naive baseline beat all but two of 23 ENCODE challenge submissions. On the shipped 8-assay panel run type is fully determined by assay and read length (`AGENTS.md` §7.2 rule 5: 0.000 bits), so a real-panel test may be impossible. The counterfactual arms avoid this: each paired-end arm changes run type alone ([[t111_re_run_the_48_eic_blind_tracks_pair|the paired-end re-run task]]).

## Idea / Hypothesis

Conditioning on the recorded run type predicts a target track from a source track, beyond what a single value-axis map already does

## Verifiables

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

Four rungs on paired tracks, scored on macro NB CRPS against the real target (PI 2026-09-16): blind (no covariates), one-warp (the competitor), model (covariate-conditioned), and oracle (two NB draws from the same eta at the target depth). Headline gap-closed = (D_onewarp − D_model) / (D_onewarp − D_oracle). [[t103_build_the_in_vitro_covariate_testbed|The testbed task]] is still open. The earlier design text and its cautions are in `results/h2/design_notes.md`.

## Run Links

No run of this design. The paired-end products are on Nibi under `/scratch/mforooz/t112_cf/products/<track>__pe__pe/` (purge 2026-11-20). The readings below come from the counterfactual-grid runs: `/project/def-maxwl/mforooz/t118/ladder/runs` (row 1) and `/project/def-maxwl/mforooz/t118/row2/runs` (row 2); extraction job 24641277 into `/project/def-maxwl/mforooz/t118/knobgen/`.

## Artifacts

- [Depth and run-type readings from the knob tables](results/h2/knob_readings.txt), made by [knob_readings.py](results/h2/knob_readings.py)
- [Manifest of the 130 counterfactual products, with the 7 paired-end arms](results/t112/MANIFEST.tsv)
- [Knob-generalisation figures](results/t118/knob_generalization/report.pdf)
- [Design text before the 2026-10-24 compression](results/h2/design_notes.md)

## Findings

Exploratory readings, not judged. The paired-end arm exists: 7 FASTQ re-runs, one per track, with the base's pipeline image and 30M-read subsample. The counterfactual grid scores it against the no-covariates twin, not one-warp:

- Counts, trained base ↔ paired-end pairs: median relative CRPS gain over the twin 0.15 over 16 design × g configurations; 11 of 16 clear 2 × seed wobble over 3 seeds.
- −log10 p, the same pairs: median-pair gain 0.03, and the real model wins 86% of pairs; no seed bar exists for that statistic.
- Run type has one level, so every never-trained pair also changes a second knob. On those pairs the median-pair gain is 0.07 in counts (real wins 65%) and −0.06 in p (real wins 34%).
- The source gives no oracle-scaled / scale-error split.
