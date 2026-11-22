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

RESULTS PENDING (Fir job 5279447, 5 replicates × 200,000 bins).

### Real miniCANDI runs: agreement with the headline set

RESULTS PENDING (p16_dev scoring of 13 run names × seeds 0 and 1).

## How to use it

```bash
# score a trained run with both sets (kit with this code; stores kept for a metrics-only rerun)
K=<kit> SUB=p16 SEEDS="0 1" RUNS_LIST="<run> ..." DRY_RUN=0 bash $K/slurm/onion_eval/deviance_submit.sh
# compare with the headline terms
python tools/onion_eval/deviance_compare.py --runs <runs_rcp> --arms "<run> ..." --out <dir>
```

Quote rules: quote `dev_imputation_skill` with `dev_imputation_floor`, and `dev_correction_gap_closed` with `dev_correction_floor`, so the reader sees how much room the noise leaves. Quote `dev_calibration_ratio` with PIT deviation.
