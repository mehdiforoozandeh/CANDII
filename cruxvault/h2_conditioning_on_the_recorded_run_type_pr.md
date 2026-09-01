---
id: h2
type: idea
schema: 2
title: Conditioning on the recorded run type predicts a target track from a source track, beyond what a single value-axis map already does
parent: q1
status: idea
rule: all
measurement: gap-closed on the onewarp→oracle interval, per signal decile and per mark class, over a paired-end arm against its single-end rebuild; instrument is a standalone encoder/decoder testbed with log d pinned as a fixed offset of coefficient exactly 1, scored through bench.distributional.nb_suite and bench.covariate
replicates: "TODO(PI) — source→target pairs per arm x seeds, on the panel t90 selects. Whatever it is, it must clear the seed floor: one paired seed change on the q19 recipe moves pooled imputation CRPS by 0.0463 (macro 0.0327) under eval.py and macro CRPS by 0.0608 under bench, and the per-track floor is several times either macro. The ~0.09 target-clustered floor and the ±0.13 per-comparison uncertainty belong to the frozen full-EIC panel and are quoted only for it (AGENTS.md §7.2 rule 2)."
verdict: 
metric: 
created: "2026-09-01T02:16:44"
updated: "2026-09-01T02:16:44"
---

# h2 — Conditioning on the recorded run type predicts a target track from a source track, beyond what a single value-axis map already does

Parent:: [[q1_do_the_recorded_experimental_covariates_]]

## ELI5

Knowing whether the sequencer read DNA fragments from one end or both is enough to turn one
experiment's numbers into the other's — and by more than just rescaling them would.

## TL;DR

An encoder reads a source track of raw counts plus the source's recorded covariates; a decoder
reads the target's covariates and emits raw counts and a −log10 p signal. The mean is
`mu = d · exp(eta)`, so `log mu = eta + log d` with **`log d` pinned as a fixed offset of
coefficient exactly 1** — without the pin, eta and the decoder's scale are confounded and eta is
not identified. The claim is **not** that prediction is possible. A single global monotone map on
the value axis, with no covariates at all, already removes most of the level difference, so that
map is the competitor. The claim is that conditioning on the recorded run type closes the gap that
map leaves, up to the floor set by two independent draws from the same eta at the target depth.

## Null
Normalization: the pairs differ by one global monotone value-axis map, so a covariate-free rescale already closes the gap and the conditioning adds nothing.

## Problem Statement

Run type is the arm that carries real weight, because unlike depth it is not a thinning identity.
`[[signal-normalization-in-epigenomics]]` records why it matters: a difference in the
**deduplication rule** between single-end and paired-end data — single-end keeps one read per
duplicate set, paired-end keeps a pair if either mate is unique — shifted the distribution enough
that a naive baseline outperformed all but two of 23 ENCODE challenge submissions. Nothing in any
of those methods could have detected it, because the assumption was made before the model saw the
data.

**This arm is gated on identifiability and may be impossible.** `AGENTS.md` §7.2 rule 5 records
`H(run_type | assay_id, read_length) = 0.000 bits` on the shipped 8-assay panel (n = 26 `T_`
records), against 0.551 bits on the full EIC panel, and calls a run-type steering demonstration
impossible there as a property of the data rather than of the architecture. A first reading on the
361-track control metadata (`results/t10/eic_control_metadata.csv`, 73 usable rows, 58 single-ended
/ 15 paired-ended) gives `H(run_type) = 0.733` bits falling to **0.124** bits given `read_length`,
and 0.119 given `read_length` and `lab` — thin but non-zero. That is controls, not the signal
tracks this will use, so it settles nothing; `t90` measures it on the panel actually selected, and
that measurement gates this node.

The literature names a stronger competitor than `onewarp` — the peak/background-split transform
S3norm fits; whether a fifth `splitwarp` rung replaces `onewarp` as the denominator is `TODO(PI)`.

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
      `TODO(PI)` — see the open decisions below. Threshold: `TODO(PI)`.
      *Fails if:* `onewarp` already closes the source-to-target gap and conditioning adds nothing
      beyond it.
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

- (i) name the distance `D` that gap-closed is a fraction of — macro NB CRPS against the real
  target, or level-only via `aspects_of(...)["level"]`;
- (ii) is the `oracle` rung two **Poisson** draws or two **NB** draws? The model emits NB. If the
  target is overdispersed, a Poisson oracle sits below anything reachable and gap-closed has no
  ceiling of 1;
- (iii) `onewarp` or a peak/background-split `splitwarp` as the headline denominator;
- (iv) every `TODO(PI)` threshold above.

**Not yet done: `crux approve-null` has not been run on this node.** The checks above were authored
by the PI and every bar in them is still `TODO(PI)`, so nothing is locked; the signature is still
outstanding.

## Planned Intervention

A four-rung ladder is fit on paired real tracks and every rung is scored on the same distance:

| rung | what it is |
|---|---|
| `blind` | no covariates |
| `onewarp` | one global monotone value-axis map, no covariates — **the competitor** |
| `model` | covariate-conditioned |
| `oracle` | two independent draws from the same eta at the target depth — the irreducible floor |

Headline gap-closed is `(D_onewarp − D_model) / (D_onewarp − D_oracle)`, never measured from
`blind`. `blind` and `onewarp` emit point maps, so they are made CRPS-scorable through the
point-to-distribution spread device the leaderboard already carries (see `t69`).

**Do not reach for `configs/regions/encode_pilot_hg38.bed` if any check subsamples loci.** Another
session in this repo reports measuring, 2026-09-01, that the 44 ENCODE Pilot Regions flip
checkpoint selections on CRPS gaps of 0.066–0.069 because their offset grows with training instead
of staying constant, while a seeded random window set of identical size was faithful. **That
measurement is not verified here and nothing in this node rests on it** — it is recorded as a
caution with its provenance named. Note that `EVAL.md` on `implementation/cheap-v-eval` currently
argues the opposite, so one of the two is wrong. Any locus subsample used here is re-measured for
the specific metric in question first.

Scored results in this repo carry a `provenance.eval_scope` block (name, bed, sha256,
`scored_bins`, `fraction`), with the scope hanging off `EvalSource.eval_regions`. Two runs compare
only when that block matches — the BED path alone will not do, because a BED is a mutable file.
**As of writing that block lives only on the unpushed local branch `implementation/cheap-v-eval`,
90 commits ahead of `origin/main`; it is not on `origin/main` and not on origin at all.**

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h2/ and link at least the report:
     - [Report](results/h2/report.md)   - results/h2/curve.png -->
_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_
