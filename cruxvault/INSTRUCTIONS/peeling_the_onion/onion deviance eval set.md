---
title: onion deviance eval set
kind: eval design and validation (task t371)
parent: "[[onion eval protocol]]"
status: added beside the headline set; replaces nothing (PI, 2026-11-22)
---

# The deviance eval set

This note defines a second set of count metrics for CANDI's three goals (imputation, correction, calibration), explains why we added it, and reports how it behaved in simulation and on real miniCANDI runs. The headline set in [[onion eval protocol#headline-metric-set]] stays as it is. Both sets are computed from the same predictions in the same scoring job.

Code: `tools/onion_eval/deviance.py` (metrics), `tools/onion_eval/deviance_sim.py` (simulation checks), `tools/onion_eval/deviance_compare.py` (comparison with the headline set). The scorer writes the set under the key `deviance` of `scores.json` when it runs with `--score-deviance` or `ONION_SCORE_DEVIANCE=1`; without the switch, `scores.json` is unchanged (`tests/test_onion_eval_deviance.py`).

## Why we added it

The current headline set uses three different units. Imputation uses squared error on log(1 + count) (the recovered share of specific variance). Correction uses the NB CRPS (gap closed against depth-only scaling). Calibration uses the PIT deviation. Max's sprint 10 (`notes/counts.md`, section 4.1 of its report, GitHub `mlibbrecht/01_2026-10-19_research`) tested count units in simulation and found that log(1 + y) squared error and CRPS drift with sequencing depth: the same mistake scores differently at a quarter and at four times the depth. Poisson deviance was the only unit he tested that did not drift, and its noise floor came from Poisson theory within 0.01 of the true floor. Correction compares tracks at different depths by design, so a depth-invariant unit matters most there.

The aim is one count unit for the two point-prediction goals, plus a calibration score in the same family, with each goal measured separately.

## Definitions

The Poisson unit deviance of a count y under a predicted mean mu is d(y, mu) = 2 [y log(y/mu) − (y − mu)]. For one track, D(mu) is the sum of d over its bins divided by the same sum for a constant prediction at the track's own mean (call this the normalized deviance). Every aggregate is a ratio of target means of normalized deviances, per assay, then a macro over the scorable assays, with a bootstrap over cell types (the scorer's `metric`).

The key property: Poisson deviance is a Bregman divergence, i.e. a strictly consistent score for the mean. For a target drawn independently of the prediction, the expected deviance is the read-noise floor plus the model's own error, and the two add exactly. This holds for any count distribution with that mean, NB included.

| goal | metric (key) | definition | 0 means | 1 means |
|---|---|---|---|---|
| imputation | `dev_imputation_skill` | (D_avg − D_model) / (D_avg − D_floor) | the per-position training average | the read-noise floor |
| correction | `dev_correction_gap_closed` | (D_depth − D_model) / (D_depth − D_floor), never-trained pairs, target pr2 | depth-only scaling | the read-noise floor of pr2 |
| calibration | `dev_calibration_ratio` R | Σ d(y, mu) / Σ E_{Y~NB(n, mu)} d(Y, mu) | — | calibrated (R > 1 overconfident, R < 1 underconfident) |
| calibration | `dev_calibration_error` | abs(log2 R) | calibrated | spread off by a factor 2 |

The read-noise floor D_floor is the Poisson plug-in floor: the expected deviance of Poisson draws at a rate lambda, where lambda is the monotone (isotonic) fit of the target on the model's mean. Max's 52 replicate twins support a Poisson floor at 25 bp: replicate floor 0.572 against Poisson plug-in floor 0.576. The model's predictive is NB; the floor is Poisson because, given the rate, read sampling within one experiment is Poisson, and the overdispersion across bins is part of the rate the model must predict.

Background keys (computed, not headline): `dev_imputation_r2` (1 − D_model, a deviance R²), the scale error (D_model − D of the globally rescaled mean), the monotone error (D_model − D of the isotonic fit), the floor and baseline values themselves, `dev_correction_gap_closed_vs_pseudoreplicate` (the floor replaced by the pr1 → pr2 predictor, as the current correction term), and `nb_logscore_gain_vs_marginal` (mean NB log-likelihood per bin, model minus marginal NB; a proper score of the whole distribution).

The expected NB deviance uses a lookup table on (log mu, log n), built from exact sums; its largest relative error on 300 random (n, mu) points was 0.11% (median 0.008%).

## Pros and cons against the current set

| goal | current term | deviance term | gained | lost or open |
|---|---|---|---|---|
| imputation | specific variance recovered above noise, log(1 + count) | deviance skill vs the position average | count-native, depth-invariant, floor from theory with no replicate needed, additive | no shared/specific split yet; high-count bins weigh more |
| correction | CRPS gap closed vs depth-only, oracle floor on 18 no-duplicate pairs | deviance gap closed vs depth-only, theory floor on every never-trained pair | bounded at 1 (the pr1 → pr2 "floor" of the old CRPS form let models score 2 to 7), uses all pairs, depth-invariant | the Poisson floor is optimistic where duplicates make reads dependent, so the gap is conservative there |
| calibration | PIT deviation gap closed vs marginal NB | deviance calibration ratio | says the direction (over- or underconfident), needs no replicate, exact for a discrete predictive | tests the spread, not the shape; PIT stays the shape check |

## Results

### Simulation (pre-registered checks C1–C5)

Fir job 5279447, 5 replicates × 200,000 bins; evidence `cruxvault/results/t371/sim/checks.json`. All five checks passed.

| check | what it tests | result | pass rule |
|---|---|---|---|
| C1 | the Poisson plug-in floor equals the true floor | largest error 0.0013 | ≤ 0.02 |
| C2 | the floor-corrected error of each mistake stays the same across depths | worst drift 0.021 (log(1 + y) squared error: 0.467) | ≤ 0.03 or 20%, whichever is larger, from ×0.25 to ×4 depth |
| C3 | the true rate scores best; a larger mistake of one kind scores worse | held | always |
| C4 | the calibration ratio is 1 with the true NB size | 0.993 to 1.005 (size ×4: 1.03 to 1.93; size ÷4: 0.51 to 0.91) | within 0.03 of 1; above 1 at size ×4, below 1 at size ÷4 |
| C5 | correction endpoints: depth-only scores 0, the truth scores 1 | depth-only 0 exactly; truth 0.999 to 1.001 | depth-only exactly 0; truth within 0.03 of 1 |

The unit test runs the same checks at 1 × 50,000 bins. At that size the C2 tolerance is the size of the sampling noise, so the unit test uses a paired form of C2.

### Real miniCANDI runs: agreement with the headline set

Ten recipe arms × seeds 0 and 1 and the three base models × seeds 0 and 1 were re-predicted and scored on the 16 pair regions with the switch on (`p16_dev`). Evidence: `cruxvault/results/t371/compare/{rcp,base}/`.

**Kit check.** Every key except `deviance` was bit-identical to the existing `p16_sv` scores in 20 of 20 recipe runs. The base-model runs matched on the three headline terms; other keys differ because their `p16_sv` came from an older scorer version.

**Rank agreement** (Spearman correlation of the 10 arm means, i.e. rank correlation between two orderings):

| goal | current term | deviance term | Spearman | p |
|---|---|---|---|---|
| imputation | imputation_raw | dev_imputation_skill | 0.93 | 0.0001 |
| imputation | count CRPS skill | dev_imputation_skill | 0.84 | 0.002 |
| correction | correction_v3_raw (18 no-duplicate pairs) | dev_correction_gap_closed | 0.78 | 0.008 |
| correction | correction gap, oracle floor, all pairs | dev_correction_gap_closed | 0.93 | 0.0001 |
| calibration | uncertainty_raw (PIT) | dev_calibration_error | 0.19 | 0.60 |
| calibration | uncertainty_raw (PIT) | nb_logscore_gain_vs_marginal | 0.25 | 0.49 |

**Seed stability** (share of a metric's variance that lies between arms rather than between seeds; higher separates arms better):

| goal | current term | deviance term |
|---|---|---|
| imputation | 0.921 | 0.945 |
| correction | 0.950 | 0.993 |
| calibration | 0.947 | 0.965 |

**Per-arm means (16 regions, 2 seeds).**

| run | imputation_raw | dev skill (floor) | correction_v3_raw | dev gap | uncertainty_raw | dev ratio R |
|---|---|---|---|---|---|---|
| rcp_mix_avg_unet | 0.481 | 0.543 (0.518) | 0.544 | 0.791 | 0.661 | 1.063 |
| rcp_halves_mix_avg_unet | 0.477 | 0.503 (0.518) | 0.780 | 0.947 | 0.619 | 1.091 |
| rcp_headskip_film_unet | 0.459 | 0.265 (0.527) | 0.533 | 0.740 | 0.617 | 1.004 |
| rcp_cac_cnn_corefilm_off_unet | 0.439 | 0.246 (0.530) | 0.577 | 0.786 | 0.496 | 1.002 |
| rcp_halves_mix_pearson_unet | 0.432 | 0.248 (0.527) | 0.774 | 0.948 | 0.585 | 1.030 |
| rcp_halves_headskip_film_unet | 0.416 | 0.187 (0.533) | 0.750 | 0.946 | 0.557 | 1.031 |
| rcp_mix_fixdepth_unet | 0.407 | 0.246 (0.531) | 0.661 | 0.962 | 0.693 | 1.003 |
| rcp_nocov_unet | 0.407 | 0.217 (0.533) | 0.484 | 0.888 | 0.635 | 0.963 |
| rcp_halves_convae | 0.396 | −0.269 (0.534) | 0.637 | 0.920 | −0.089 | 1.665 |
| rcp_film_bottleneck_convattn | 0.389 | 0.159 (0.533) | 0.678 | 0.927 | 0.687 | 1.238 |
| base U-Net (rung 4) | 0.447 | 0.272 (0.527) | 0.537 | 0.832 | 0.624 | 1.016 |
| base conv + attention (rung 3) | 0.380 | 0.114 (0.532) | 0.672 | 0.928 | 0.633 | 1.254 |
| base conv AE (rung 2) | 0.364 | 0.100 (0.535) | 0.672 | 0.928 | 0.651 | 1.268 |

The current pick (pairs + Pearson-residual input U-Net) at 256 regions, seed 2: dev skill 0.294 (floor 0.526), dev gap 0.951 (floor 0.630), R 1.044; imputation gap closed 0.471, uncertainty gap closed 0.705. Seeds 0 and 1 at 256 regions were queued on Fir at the time of writing.

### What the comparison shows

1. Imputation and correction: the deviance terms rank arms like the current terms (0.78 to 0.93) and separate arms from seed noise at least as well. The deviance correction gap agrees best with the oracle-floor gap on all pairs (0.93), which uses the same pairs.
2. Calibration: the deviance ratio and the PIT term do not agree on rank (0.19). The two measure different properties. R tests the spread of the predictive (too narrow or too wide). PIT tests its shape. Both are stable across seeds, so we keep both.
3. The deviance set shows failures the current set hides or understates. The conv AE with halves has a negative skill (−0.11 and −0.43) and R 1.55 and 1.78, i.e. strongly overconfident. The conv-based runs read as overconfident (R 1.24 to 1.27); the U-Nets sit near 1.0.
4. The deviance skill spreads arms more than imputation_raw does. The two average-input U-Nets score 0.50 to 0.54; the other U-Nets score 0.19 to 0.27.
5. Against the base models on the deviance set at 16 regions, the current pick does not beat the base U-Net on imputation skill (0.248 vs 0.272). The pairs + average-input U-Net beats every base model on imputation skill and on the correction gap, but its R (1.091) is further from 1 than the base U-Net's (1.016). The Pareto pick under the headline set (D1) is unchanged; this note reports the deviance view beside it.

### Known weakness: the depth-only reference

The depth-only reference has a normalized deviance of 2.65, i.e. worse than a constant prediction at the track mean. Source bins with zero reads give a scaled mean near zero, and Poisson deviance punishes a near-zero mean on a non-zero target heavily. Every model beats this reference easily, so the correction gap is compressed into 0.74 to 0.96 (it squeezes the range into a narrow band). The ranking is unaffected (rank agreement 0.93 with the oracle-floor gap). `rcp_mix_fixdepth_unet` has depth-only 4.85, because its pair layout differs.

A smoothed depth-only reference (for example, the source track smoothed over a few bins before scaling, or shrunk toward the position average) would be a stronger zero point. We did not add it: it was proposed after seeing these results, and the PI ruled (2026-11-23) to record it here and leave the defaults and the headline terms unchanged.

## How to use it

```bash
# score a trained run with both sets (kit with this code; stores kept for a metrics-only rerun)
K=<kit> SUB=p16 SEEDS="0 1" RUNS_LIST="<run> ..." DRY_RUN=0 bash $K/slurm/onion_eval/deviance_submit.sh
# compare with the headline terms
python tools/onion_eval/deviance_compare.py --runs <runs_rcp> --arms "<run> ..." --out <dir>
```

Quote rules: quote `dev_imputation_skill` with `dev_imputation_floor`, and `dev_correction_gap_closed` with `dev_correction_floor`, so the reader sees how much room the noise leaves. Quote `dev_calibration_ratio` with PIT deviation.
