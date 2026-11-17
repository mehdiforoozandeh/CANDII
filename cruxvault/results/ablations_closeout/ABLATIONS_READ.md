---
title: "Close-out read of the 16 miniCANDI ablations (peeling the onion)"
created: 2026-11-17
---

# Close-out read of the 16 miniCANDI ablations

This file reads the scores of the 16 ablations on three architectures: the conv autoencoder, conv + attention and the U-Net. Each section states the question, the prior leads, the numbers, the matching row of the specification's decision table, and whether the expectation held.

**Sources.** Every number comes from `arm_summary_p16_sv_2026-11-17.tsv` (this folder; a copy of the foreman's table, also on Fir at `/project/def-maxwl/mforooz/ONION/runs_fir/ARM_SUMMARY_p16_sv_2026-11-17.txt`), from `arm_summary_p16_foreign_2026-11-17.tsv` (this folder; made by me with `select_rule.py arm-summary --sub p16_foreign_<mode>`, kit CO6), or from per-seed `select_rule.py table` output that I ran on Fir. Specifications: `cruxvault/INSTRUCTIONS/peeling_the_onion/run_state/ablation_specs/<n>_<name>.md`. Parameter counts and trunk widths: I rebuilt every seed-0 model locally from its `run.json` with `registry.build_model`; every local count equals the count in `run.json`.

**Metrics.** Screening set of 16 regions, scorer sub-directory `p16_sv`. "Imputation" is the variance recovered. "Correction" is the gap closed on the 18 crop pairs with the oracle reference. "Calibration" is the calibration term. "Composite" is composite version 3 (0.5 / 0.25 / 0.25, clipped terms). "Correction-only" is the correction-only term, i.e. the share of the covariate cost closed on same-view pairs; it is not part of the composite. Each value is the mean over seeds, with the seed range (max minus min) in brackets.

**Judgment rule.** An effect is "beyond" when the difference of means is larger than the larger of the two seed ranges. Otherwise it is "within" and I make no claim. "2x" marks a difference larger than twice the larger range (the D13 bar). Each table row gives "arm mean (range), arm minus base".

**References** (3 seeds each, `p16_sv`):

| model | trunk width, parameters | composite | imputation | correction | calibration | correction-only |
|---|---|---|---|---|---|---|
| conv autoencoder (`rung2_base_heldout`) | 60, 1,042,514 | 0.528 (0.045) | 0.382 (0.028) | 0.718 (0.012) | 0.631 (0.178) | 0.690 (0.019) |
| conv + attention (`rung3_base_heldout`) | 104, 1,038,018 | 0.525 (0.048) | 0.387 (0.012) | 0.673 (0.031) | 0.652 (0.157) | 0.659 (0.043) |
| U-Net (`rung4_base_heldout`) | 25, 993,599 | 0.495 (0.010) | 0.432 (0.023) | 0.507 (0.071) | 0.610 (0.108) | 0.318 (0.127) |

The calibration ranges (0.108 to 0.178) are wide. A calibration effect below about 0.11 to 0.18 cannot show as beyond.

## 1. Summary

| # | ablation | question | answer | confidence | holds on |
|---|---|---|---|---|---|
| 1 | control as input | Does the ChIP control as an input help? | Removing it lowers imputation on all three (0.019 to 0.026); beyond on conv + attention only. | weak | conv + attention |
| 2 | control as output | Does predicting the control help or hurt? | Small effects of opposite sign: off lowers imputation on conv + attention, raises the composite on the U-Net. | weak | none in common |
| 3 | average baseline input | Does the training-cell average as an input help? | It raises U-Net imputation by 0.034 (beyond); on the conv models it moves no composite term. | weak | U-Net |
| 4 | two-step routing | Does routing through the base covariates help correction? | No. Native two-step lowers crop-pair correction beyond on both conv models; the U-Net gains only with fixed base depth. | moderate | conv autoencoder, conv + attention |
| 5 | DNA source | Does the DNA source matter? | Nucleotide-transformer embeddings equal the own tower on all three. A 5 Mb shift lowers conv + attention imputation. | moderate | all three (null) |
| 6 | DNA on or off | Does DNA add anything? | Only on conv + attention imputation (-0.018 without DNA, beyond). U-Net is within but width-confounded. | weak | conv + attention |
| 7 | covariate embedding | Does the covariate embedding form matter? | No, except numeric-only lowers U-Net correction by 0.133 (beyond). | weak | all three (null), U-Net exception |
| 8 | input depth fixed | Does a fixed input depth help? | U-Net: correction +0.168 and composite +0.023 (2x), imputation -0.028 (beyond). Trade-off. | moderate | U-Net |
| 9 | losses | Does another loss help a goal? | No loss raises the composite beyond on 2+ models; imputation weight 4 raises conv + attention imputation (2x). | moderate | all three (null) |
| 10 | adaLN-zero injection | Does adaLN-zero beat FiLM? | No. All four metrics within on all three. | moderate | all three (null) |
| 11 | scrambled covariates | Do the models use covariates beyond depth? | Yes for the correction-only term (beyond on all three); crop-pair correction drops beyond on the conv autoencoder only. | moderate | all three (correction-only) |
| 12 | mask mode | Learned mask token or dropping? | No difference; all within on all three. | moderate | all three (null) |
| 13 | injection form | FiLM, concatenation or tokens? | Concatenation is within or better on all three; tokens hurt U-Net imputation (2x). | weak | all three |
| 14 | training views | Do counterfactual views help? | Composite terms do not need them, but the correction-only term collapses without them on all three. Alignment-only views hurt both conv models. | moderate | all three |
| 15 | count transforms | Does a transform beat the offset? | None on 2+ models. Pearson, asinh and soft clip beat it on the U-Net only. Local ratio and quantile are worse on all three. Anscombe is void (code defect). | moderate | U-Net (gains); all three (losses) |
| 16 | DNA fusion | Does the DNA join matter? | Not on the conv autoencoder or U-Net. On conv + attention, late joins lower imputation. | weak | conv + attention |

## 2. Per ablation

### 2.1 Control as input (specification 1)

**Question.** The PI: "ablation: experiment with adding chip controls (upon availability for those celltypes) as one assay". **Arm.** The control track is removed from the inputs and stays a target. **Prior lead.** No trained result; 250-step smokes only (`results/t209/ASF.md`).

| model | composite | imputation | correction | calibration | correction-only |
|---|---|---|---|---|---|
| conv autoencoder | 0.517 (0.017), -0.011 | 0.358 (0.035), -0.024 | 0.702 (0.021), -0.015 | 0.652 (0.099), +0.021 | 0.694 (0.018), +0.003 |
| conv + attention | 0.515 (0.028), -0.009 | 0.360 (0.021), **-0.026 beyond** | 0.669 (0.029), -0.004 | 0.672 (0.123), +0.020 | 0.646 (0.024), -0.014 |
| U-Net | 0.483 (0.016), -0.012 | 0.413 (0.008), -0.019 | 0.522 (0.049), +0.015 | 0.585 (0.057), -0.024 | 0.369 (0.042), +0.051 |

**Beyond.** Imputation lower without the control, on conv + attention only. **Decision row.** "beyond on one architecture only": an architecture-specific effect; do not change the design; the architecture is conv + attention. **Expectation.** Imputation lower by 0 to 0.02, within on 2 of 3: held for "within on 2 of 3"; the sizes (0.019 to 0.026) are slightly larger than stated. Correction, calibration, composite and correction-only unchanged: held. **Note.** The imputation difference is negative on all three models. **Caveats.** None for width (counts equal the base).

### 2.2 Control as output (specification 2)

**Question.** The PI: "ablation: also predict the chip control". **Arm.** `predict_control: false`; the control stays an input. **Prior lead.** The base models predict the control poorly (count CRPS skill -0.088 to +0.015 on one target); PI ruling 14 keeps the control output in the EIC model in all cases.

| model | composite | imputation | correction | calibration | correction-only |
|---|---|---|---|---|---|
| conv autoencoder | 0.537 (0.004), +0.008 | 0.375 (0.013), -0.007 | 0.714 (0.008), -0.004 | 0.683 (0.017), +0.052 | 0.654 (0.039), -0.036 |
| conv + attention | 0.513 (0.053), -0.012 | 0.366 (0.007), **-0.021 beyond** | 0.664 (0.020), -0.009 | 0.655 (0.179), +0.003 | 0.628 (0.019), -0.032 |
| U-Net | 0.509 (0.004), **+0.014 beyond** | 0.436 (0.014), +0.004 | 0.545 (0.084), +0.038 | 0.620 (0.057), +0.011 | 0.349 (0.112), +0.031 |

**Beyond.** Off is worse on conv + attention imputation and better on the U-Net composite. **Decision row.** "different sign across architectures": architecture-dependent; no general claim. **Expectation.** Null on all four metrics: not held on conv + attention (imputation) and U-Net (composite); both effects are small. **Caveats.** The arm cannot become the EIC model (ruling 14). The U-Net composite range is 0.010, so a 0.014 difference counts as beyond.

### 2.3 Average baseline input (specification 3)

**Question.** The PI: "optional/ablation: average baseline: for each assay type, we can get its average activity along the genome (training set only)". **Arm.** The leave-one-cell-out training average per assay and position is an extra input. **Prior lead.** On the pointwise MLP the training loss fell by 0.039 (beyond); the average alone scores composite 0.197.

| model | composite | imputation | correction | calibration | correction-only |
|---|---|---|---|---|---|
| conv autoencoder | 0.534 (0.030), +0.006 | 0.400 (0.017), +0.018 | 0.743 (0.055), +0.026 | 0.592 (0.100), -0.039 | 0.761 (0.031), **+0.071 beyond, 2x** |
| conv + attention | 0.523 (0.053), -0.002 | 0.391 (0.019), +0.004 | 0.698 (0.039), +0.025 | 0.611 (0.150), -0.041 | 0.713 (0.055), +0.054 |
| U-Net | 0.511 (0.050), +0.016 | 0.466 (0.011), **+0.034 beyond** | 0.501 (0.028), -0.006 | 0.611 (0.162), +0.001 | 0.399 (0.089), +0.081 |

**Beyond.** U-Net imputation higher; conv autoencoder correction-only higher. **Decision row.** "a gain on one architecture only": the effect depends on the architecture; report it, do not generalize. **Expectation.** Imputation +0.01 to +0.04, beyond on conv + attention: the size held on two models, but the beyond effect is on the U-Net, not on conv + attention. Correction 0 +/- 0.03: held. Calibration and composite within: held. **Not checked.** The shared-error diagnostic and `count_crps_skill_vs_position_average`. **Caveats.** Parameter counts +576, +32, +16 (at most 0.06%), widths equal.

### 2.4 Two-step routing through the base (specification 4)

**Question.** The PI: "this model generalizes to unseen C,C' pairs if we first go from (X, C) to (X_base, C_base) and from (X_base, C_base) to (X', C')". **Arms.** Native: the base leg keeps its native depth. Fixed: the base leg is thinned to one reference depth per assay class (log2 depth 23.5 to 25.0, from `run.json`). **Prior lead.** On the CAC test bed with a shared model, chaining beat the direct route by 0.027 on crop pairs only (finding f31, direction against).

| arm, model | composite | imputation | correction | calibration | correction-only |
|---|---|---|---|---|---|
| native, conv autoencoder | 0.454 (0.028), **-0.074 beyond** | 0.362 (0.030), -0.020 | 0.602 (0.085), **-0.115 beyond** | 0.489 (0.058), -0.142 | 0.708 (0.094), +0.017 |
| native, conv + attention | 0.479 (0.054), -0.046 | 0.366 (0.012), **-0.021 beyond** | 0.614 (0.033), **-0.059 beyond** | 0.570 (0.243), -0.082 | 0.766 (0.043), **+0.106 beyond, 2x** |
| native, U-Net | 0.412 (0.059), **-0.083 beyond** | 0.451 (0.027), +0.019 | 0.443 (0.077), -0.064 | 0.303 (0.322), -0.307 | 0.085 (0.180), **-0.233 beyond** |
| fixed, conv autoencoder | 0.479 (0.067), -0.049 | 0.343 (0.013), **-0.039 beyond** | 0.632 (0.038), **-0.085 beyond, 2x** | 0.597 (0.261), -0.033 | 0.806 (0.007), **+0.116 beyond, 2x** |
| fixed, conv + attention | 0.471 (0.047), **-0.053 beyond** | 0.357 (0.018), **-0.030 beyond** | 0.618 (0.052), **-0.055 beyond** | 0.554 (0.168), -0.098 | 0.801 (0.051), **+0.141 beyond, 2x** |
| fixed, U-Net | 0.495 (0.047), +0.000 | 0.451 (0.028), +0.019 | 0.636 (0.069), **+0.129 beyond** | 0.445 (0.119), **-0.165 beyond** | 0.378 (0.189), +0.060 |

Fixed minus native: U-Net composite +0.084 and correction +0.193 (2x), both beyond; conv autoencoder correction-only +0.099 beyond; all else within. **Decision rows.** Conv models: "correction down beyond": joint two-leg training hurts; check the leg-2 input shift first. U-Net native: "correction within": no evidence that routing helps. Fixed against native: beyond on U-Net correction, so the base depth matters there and fixed is the better form. **Expectation.** Correction up: not held (down on the conv models). Fixed equal to native: not held on the U-Net. **Caveats.** Two-step doubles the FLOPs per window (`flops_per_window` 4.22e9 against 2.11e9 on the conv autoencoder). Leg 2 reads all tracks of a cell in training and one track at scoring (specification section 6.1); this shift can explain the loss. Crop-pair correction falls while the correction-only term rises on the conv models.

### 2.5 DNA source (specification 5)

**Question.** The PI: "for DNA embeddings we have three options: our own DNA tower trained from scratch; using Flashzoi embeddings; using a genomic language model embeddings". **Arms.** Frozen nucleotide transformer v2 embeddings (128 PCA components), and the own tower fed sequence shifted by 5 Mb (a misaligned-sequence control). The Flashzoi arm was not run (no run directory on Fir). **Prior lead.** On the pointwise MLP, DNA off moved nothing beyond the range.

| arm, model | composite | imputation | correction | calibration | correction-only |
|---|---|---|---|---|---|
| language model, conv autoencoder | 0.534 (0.034), +0.006 | 0.391 (0.016), +0.008 | 0.722 (0.092), +0.004 | 0.632 (0.037), +0.001 | 0.737 (0.046), +0.046 |
| language model, conv + attention | 0.524 (0.043), -0.001 | 0.387 (0.017), +0.000 | 0.692 (0.036), +0.019 | 0.629 (0.120), -0.023 | 0.703 (0.069), +0.044 |
| language model, U-Net | 0.510 (0.018), +0.014 | 0.444 (0.031), +0.012 | 0.507 (0.052), +0.000 | 0.643 (0.108), +0.033 | 0.371 (0.038), +0.053 |
| shift 5 Mb, conv autoencoder | 0.523 (0.031), -0.005 | 0.366 (0.020), -0.016 | 0.739 (0.035), +0.021 | 0.621 (0.074), -0.010 | 0.633 (0.037), **-0.058 beyond** |
| shift 5 Mb, conv + attention | 0.511 (0.043), -0.014 | 0.367 (0.015), **-0.020 beyond** | 0.680 (0.001), +0.007 | 0.629 (0.202), -0.023 | 0.592 (0.024), **-0.067 beyond** |
| shift 5 Mb, U-Net | 0.487 (0.020), -0.008 | 0.418 (0.003), -0.014 | 0.503 (0.054), -0.004 | 0.610 (0.087), -0.000 | 0.289 (0.059), -0.029 |

**Decision rows.** Language model: row B on all three: the DNA source does not matter at this scale; keep the own tower. Shift: row D on conv + attention (aligned sequence carries information that the model uses); row C on the other two (the own tower adds nothing on the four metrics, or only a locus fingerprint). **Expectation.** Language model within: held. Shift within: held on the conv autoencoder and U-Net, not on conv + attention imputation. **Caveats.** Parameter counts +1,376, +1,376, +704; widths equal. Flashzoi and the fold-3 check did not run, so rows E and F have no answer.

### 2.6 DNA on or off (specification 6)

**Question.** The PI: "DNA off." **Arm.** No DNA input; the trunk width is refit to the 1.0e6 budget. **Prior lead.** Pointwise MLP: imputation -0.003, all within.

| model | composite | imputation | correction | calibration | correction-only |
|---|---|---|---|---|---|
| conv autoencoder | 0.543 (0.010), +0.015 | 0.369 (0.021), -0.013 | 0.737 (0.043), +0.020 | 0.698 (0.060), +0.067 | 0.637 (0.022), **-0.054 beyond, 2x** |
| conv + attention | 0.495 (0.054), -0.029 | 0.369 (0.012), **-0.018 beyond** | 0.692 (0.038), +0.019 | 0.552 (0.215), -0.100 | 0.607 (0.081), -0.052 |
| U-Net | 0.493 (0.007), -0.002 | 0.412 (0.025), -0.020 | 0.523 (0.069), +0.016 | 0.626 (0.049), +0.017 | 0.286 (0.023), -0.032 |

**Decision row.** "Imputation lower beyond the range, on any architecture": keep DNA (conv + attention). On the conv autoencoder and U-Net the four metrics are within. **Expectation.** Imputation 0 to -0.01, within: not held on conv + attention (-0.018, beyond); the other two are within but larger (-0.013, -0.020). U-Net near 0.43 / 0.51 / 0.49: held (0.412 / 0.523 / 0.493). **Agreement.** The 5 Mb shift (2.5) also lowers conv + attention imputation beyond (-0.020). Two independent arms agree. **Confound.** U-Net trunk width 25 -> 30 (996,232 parameters): the count path is wider. Conv models keep their widths (parameters -4,928 and -6,336).

### 2.7 Covariate embedding (specification 7)

**Question.** The PI: "covariate embedding and covariate injection ... are targets for ablation". **Arms.** Numeric columns only (categorical columns off); covariate dropout 0.1 in training; per-value codes for numeric columns (`param_budget` raised to 1,019,968 so that the width stays). **Prior lead.** Scrambled covariates lowered imputation on the pointwise MLP and U-Net (older composite).

| arm, model | composite | imputation | correction | calibration | correction-only |
|---|---|---|---|---|---|
| numeric only, conv autoencoder | 0.548 (0.046), +0.019 | 0.394 (0.014), +0.011 | 0.688 (0.063), -0.030 | 0.715 (0.176), +0.085 | 0.686 (0.024), -0.004 |
| numeric only, conv + attention | 0.507 (0.022), -0.018 | 0.375 (0.015), -0.012 | 0.624 (0.065), -0.049 | 0.655 (0.110), +0.003 | 0.597 (0.078), -0.062 |
| numeric only, U-Net | 0.464 (0.055), -0.031 | 0.434 (0.017), +0.003 | 0.374 (0.057), **-0.133 beyond** | 0.613 (0.174), +0.003 | 0.146 (0.200), -0.172 |
| dropout, all three | 0.535 / 0.504 / 0.502 | 0.386 / 0.383 / 0.429 | 0.718 / 0.687 / 0.520 | 0.651 / 0.564 / 0.632 | 0.669 / 0.651 / 0.374 |
| per value, all three | 0.537 / 0.522 / 0.488 | 0.393 / 0.382 / 0.426 | 0.719 / 0.675 / 0.442 | 0.644 / 0.649 / 0.658 | 0.673 / 0.675 / 0.281 |

Dropout and per-value: every difference is within on all three (largest: per value, U-Net correction -0.065, range 0.071; dropout, conv + attention calibration -0.089). **Decision rows.** Numeric-only on the conv models: "within on all four": keep the simpler numeric-only embedding. Numeric-only on the U-Net: "correction lower, beyond": unexpected; check for a wiring fault before any conclusion. Dropout: no row fires; do not adopt. Per value: "within or higher": keep scalars. **Expectation.** All within: held except U-Net numeric-only correction. **Caveats.** Per-value arms have +19,968 parameters (+1.9% to +2.0%), widths equal. The corrected-covariate tables (section 3.2) add categorical lab, antibody and donor columns, which numeric-only would discard.

### 2.8 Fixed input depth (specification 8)

**Question.** The PI: "as an ablation, the count input data to the model can be depth controlled". **Arm.** Every input track is thinned to 2^23 reads if deeper (the same cap for all assay classes; 450 training depth views below the cap stay native); targets and C' stay native. **Prior lead.** Hiding the input depth covariate lowered the U-Net composite (0.444 to 0.382, `p16`).

| model | seeds | composite | imputation | correction | calibration | correction-only |
|---|---|---|---|---|---|---|
| conv autoencoder | 2 | 0.491 (0.016), -0.037 | 0.364 (0.016), -0.019 | 0.711 (0.048), -0.007 | 0.526 (0.016), -0.105 | 0.729 (0.057), +0.038 |
| conv + attention | 3 | 0.535 (0.008), +0.010 | 0.376 (0.010), -0.011 | 0.722 (0.038), **+0.049 beyond** | 0.665 (0.036), +0.012 | 0.666 (0.006), +0.007 |
| U-Net | 3 | 0.518 (0.010), **+0.023 beyond, 2x** | 0.404 (0.010), **-0.028 beyond** | 0.675 (0.013), **+0.168 beyond, 2x** | 0.590 (0.017), -0.020 | 0.422 (0.068), +0.104 |

**Decision rows.** U-Net: "correction up beyond, imputation down beyond": a trade-off; report both terms; no design change without a PI ruling. Conv + attention: correction up beyond only; no row fires exactly. "Composite up beyond on 2 or more": not met (U-Net only). **Expectation.** Imputation down: held in direction on all three (beyond on the U-Net only). Correction down 0.05 to 0.15 beyond on all three: not held; correction went up on conv + attention and the U-Net. Composite down: not held. **Caveats.** Conv autoencoder seed 1 did not finish training (no `DONE`), so that row has 2 seeds. The scorer thins the inputs too, while the depth-only reference reads native inputs; I did not check the mechanism of the correction gain.

### 2.9 Losses (specification 9)

**Question.** The PI: "any other loss/regularization/etc. that can help with our three main objectives". **Arms.** Imputation weight 4; bin-equal weighting; Poisson mean with a detached NB size; NB CRPS. **Prior lead.** On the CAC test bed, imputation weight 4 moved never-trained CRPS by -0.034 against a bar of 0.039.

| arm | conv autoencoder | conv + attention | U-Net |
|---|---|---|---|
| imputation x4 | correction 0.680 (0.031), **-0.038 beyond**; correction-only 0.750, **+0.060 beyond**; imputation +0.028, composite -0.015 | imputation 0.426 (0.017), **+0.039 beyond, 2x**; correction-only 0.761, **+0.101 beyond, 2x**; composite 0.559 (0.013), +0.034 | all within; composite 0.495, -0.000 |
| bin-equal | all within; composite 0.491, -0.037 | imputation 0.357 (0.020), **-0.030 beyond** | composite 0.510 (0.009), **+0.015 beyond** |
| Poisson mean | all within; composite -0.025 | all within | all within |
| NB CRPS | all within; calibration -0.096 | correction 0.726 (0.015), **+0.053 beyond** | all within |

**Decision rows.** Imputation x4: aimed term beyond-better on conv + attention only: "effect on 1 architecture only": record, do not adopt; on the conv autoencoder correction drops beyond (a trade-off there). Bin-equal: opposite signs on two models; no adoption. Poisson mean: "all terms inside on all architectures": keep the NB NLL. NB CRPS: "calibration inside the range": no answer for goal 3. **Expectation.** "No arm changes the composite beyond on 2 of 3": held. Imputation x4 within: not held on conv + attention. **Caveats.** Widths and counts equal the base. The held-out loss is not comparable across arms; I did not use it.

### 2.10 adaLN-zero covariate injection (specification 10)

**Question.** The PI: "covariate embedding and covariate injection ... are targets for ablation"; adaLN-zero is option 3 in the ablations file. **Arm.** Zero-initialized gated modulation of layer-normalized features in place of FiLM. **Prior lead.** Concatenation against FiLM on the pointwise MLP: composite +0.007, within.

| model | composite | imputation | correction | calibration | correction-only |
|---|---|---|---|---|---|
| conv autoencoder | 0.527 (0.007), -0.001 | 0.377 (0.021), -0.005 | 0.726 (0.049), +0.008 | 0.627 (0.104), -0.004 | 0.643 (0.054), -0.047 |
| conv + attention | 0.521 (0.048), -0.003 | 0.378 (0.014), -0.009 | 0.697 (0.027), +0.024 | 0.633 (0.182), -0.020 | 0.615 (0.032), **-0.044 beyond** |
| U-Net | 0.513 (0.047), +0.018 | 0.443 (0.003), +0.011 | 0.513 (0.043), +0.006 | 0.653 (0.139), +0.043 | 0.356 (0.075), +0.038 |

**Decision row.** "All four metrics within range on all three architectures": keep FiLM. **Expectation.** All within: held. **Caveats.** Parameters +2,080, +2,080, +2,600; widths equal. The correction-only term (not one of the four metrics) is lower beyond on conv + attention.

### 2.11 Scrambled covariates (specification 11)

**Question.** The PI: "Scrambled covariates, as a control for covariate use." **Arm.** In training, covariate rows are permuted among the windows of a batch. At scoring the permutation is an identity, so the arm is "trained with scrambled covariates, scored with true covariates". **Prior lead.** U-Net, older composite: gap closed on all 102 pairs +0.101 (the control scored better).

| model | composite | imputation | correction | calibration | correction-only |
|---|---|---|---|---|---|
| conv autoencoder | 0.506 (0.017), -0.023 | 0.392 (0.003), +0.010 | 0.666 (0.044), **-0.052 beyond** | 0.573 (0.019), -0.058 | 0.526 (0.044), **-0.164 beyond, 2x** |
| conv + attention | 0.517 (0.023), -0.008 | 0.388 (0.013), +0.001 | 0.669 (0.018), -0.004 | 0.624 (0.074), -0.028 | 0.462 (0.063), **-0.198 beyond, 2x** |
| U-Net | 0.471 (0.034), -0.024 | 0.415 (0.007), -0.017 | 0.432 (0.080), -0.075 | 0.624 (0.097), +0.014 | 0.106 (0.045), **-0.212 beyond** |

**Decision rows.** Conv autoencoder: "gap closed lower, beyond": the covariates drive its crop-pair correction. Conv + attention and U-Net: "gap closed within on 1+ architecture": these models correct crop pairs by depth or by copying, not by covariates; inspect their crop pairs before any correction verdict. Imputation: no row fires (within on all three). **Expectation.** Imputation -0.01 to -0.03, beyond on the conv models: not held (+0.010, +0.001). Gap closed lower beyond on the conv models: held on the conv autoencoder only. **Reading.** The correction-only term falls beyond on all three, so all three models learn to use covariates for same-view correction. The foreign-covariate scoring (section 3.1) is the test-time version of this control.

### 2.12 Mask mode (specification 12)

**Question.** The PI: "Mask tokens, or dropping the missing tracks." **Arm.** Hidden bins and missing tracks are zeroed (drop) in place of a learned token. **Prior lead.** Pointwise MLP: composite +0.002, within.

| model | composite | imputation | correction | calibration | correction-only |
|---|---|---|---|---|---|
| conv autoencoder | 0.529 (0.051), +0.001 | 0.388 (0.016), +0.006 | 0.738 (0.039), +0.020 | 0.603 (0.136), -0.028 | 0.666 (0.035), -0.024 |
| conv + attention | 0.524 (0.016), -0.001 | 0.392 (0.015), +0.005 | 0.684 (0.029), +0.011 | 0.627 (0.030), -0.025 | 0.645 (0.074), -0.014 |
| U-Net | 0.497 (0.010), +0.002 | 0.426 (0.008), -0.006 | 0.547 (0.073), +0.040 | 0.591 (0.083), -0.019 | 0.327 (0.105), +0.009 |

**Decision row.** "Drop within on all four metrics": by D13 the simpler arm (drop) stays, on all three. **Expectation.** All within: held. **Caveats.** Parameters +2,128, -32, -16; widths equal. Conv + attention has no visibility flag in drop mode.

### 2.13 Covariate injection form (specification 13)

**Question.** The PI: "Covariate injection: concatenation, FiLM, or tokens." **Arms.** Concatenation with a linear layer; one token per track in the attention (conv + attention and U-Net only; the U-Net reads tokens at its bottleneck only). **Prior lead.** Pointwise MLP: concatenation composite +0.007, within.

| arm, model | composite | imputation | correction | calibration | correction-only |
|---|---|---|---|---|---|
| concatenation, conv autoencoder | 0.510 (0.054), -0.019 | 0.384 (0.007), +0.002 | 0.703 (0.018), -0.015 | 0.567 (0.222), -0.064 | 0.721 (0.019), **+0.030 beyond** |
| concatenation, conv + attention | 0.525 (0.032), +0.000 | 0.389 (0.002), +0.002 | 0.723 (0.024), **+0.050 beyond** | 0.599 (0.116), -0.053 | 0.660 (0.044), +0.001 |
| concatenation, U-Net | 0.501 (0.031), +0.006 | 0.417 (0.013), -0.015 | 0.541 (0.016), +0.034 | 0.629 (0.165), +0.019 | 0.358 (0.014), +0.040 |
| tokens, conv + attention | 0.490 (0.016), -0.034 | 0.377 (0.001), -0.010 | 0.702 (0.029), +0.029 | 0.506 (0.055), -0.146 | 0.646 (0.015), -0.014 |
| tokens, U-Net | 0.474 (0.020), **-0.021 beyond** | 0.383 (0.014), **-0.049 beyond, 2x** | 0.510 (0.115), +0.003 | 0.620 (0.063), +0.010 | 0.502 (0.119), **+0.184 beyond** |

**Decision rows.** Concatenation: "within FiLM on all four metrics, on 2+ architectures" (conv autoencoder, U-Net): by D13 concatenation stays; on conv + attention it is better on correction. Tokens on the U-Net: lower imputation and composite; no row names this case exactly; the closest row ("keep a per-bin form at the U-Net") applies. **Expectation.** All within: not held for concatenation correction on conv + attention and tokens on the U-Net. The expected U-Net tokens correction loss did not occur (+0.003). **Caveats.** Concatenation parameters -15,136, -15,136, -15,336 (-1.5%); FLOPs per window +13% to +31%; widths equal. Tokens parameters -672 and +2,520.

### 2.14 Training views (specification 14)

**Question.** The PI: "where and if adding counterfactual data (augmenting EIC) actually helps". **Arms** (reference: all views): base view only; base + depth thinning; base + read-level views (single-end, crop); base + alignment-level views (no dedup, no MAPQ filter); all views but same-view pairs only. The conv models were trained for this; the U-Net arms are older runs, scored now into `p16_sv`. **Prior lead.** U-Net (`p16`): no arm beyond on any composite term; same-view-only correction below 0 on all 3 seeds.

| arm | conv autoencoder | conv + attention | U-Net |
|---|---|---|---|
| base only | composite 0.538, +0.010; correction-only 0.402, **-0.289 beyond, 2x** | composite 0.537, +0.012; correction-only 0.390, **-0.270 beyond, 2x** | composite 0.517, +0.022; correction-only 0.069, **-0.249 beyond** |
| base + depth | correction-only 0.492, **-0.198 beyond, 2x** | correction-only 0.467, **-0.192 beyond, 2x** | correction-only -0.082, **-0.400 beyond, 2x** |
| base + read-level | correction-only 0.620, **-0.071 beyond, 2x** | all within | 2 seeds; all within |
| base + alignment-level | composite 0.476, **-0.052**; imputation 0.340, **-0.043**; correction 0.615, **-0.102 (2x)**; correction-only **-0.212 (2x)**; all beyond | imputation 0.323, **-0.064 (2x)**; correction 0.595, **-0.078 (2x)**; correction-only **-0.222 (2x)**; all beyond | 2 seeds; correction-only 0.047, **-0.271 beyond, 2x** |
| same-view only | imputation 0.413 (0.010), **+0.031 beyond**; correction 0.730, +0.012 | imputation 0.419 (0.021), **+0.032 beyond**; correction 0.683, +0.010 | composite 0.362, **-0.133 beyond, 2x**; raw correction -13.84 (range 29.15) |

Every composite term not listed is within. **Decision row.** Base only against all views: within on the composite terms on all three: "keep depth views ... by D13 no other view type earns a place". The depth-thinning pair column is not in this table; I did not run `where_views.py` on the conv models. **Expectation.** Not held in three places: (1) alignment-only lowers composite terms beyond on both conv models; (2) same-view-only does not drive conv correction below 0 and raises conv imputation; (3) the correction-only term falls beyond for base only, base + depth and base + alignment on all models. **Caveats.** U-Net read-level and alignment-level arms have 2 seeds (scoring of seed 0 failed with an I/O error and seed 2 with "Disk quota exceeded"; no job is queued). The U-Net view runs lack the `heldout_loss` key of the U-Net reference. Same-view-only U-Net correction is a raw value far below 0 (seeds -30.1, -0.90, -10.6); one correction-only value is undefined.

### 2.15 Count transforms (specification 15)

**Question.** The ablations file, "Input preprocessing of counts"; PI ruling 2026-11-16: all transforms on all three models. **Arms.** log1p, asinh, Anscombe, Pearson residuals, local ratio, per-window quantile, raw counts, soft clip, and the offset with pseudocount 0.1 and 0.01. log1p, asinh, Anscombe and Pearson also change the output link; raw counts also remove the output depth offset. **Prior lead.** All arms trained without NaN in 2,000-window smokes.

| arm | conv autoencoder | conv + attention | U-Net |
|---|---|---|---|
| log1p | all within | imputation 0.370, **-0.016 beyond** | all within; composite 0.503, +0.008 |
| asinh | all within | all within | composite 0.510 (0.011), **+0.015 beyond** |
| Anscombe | void: 0.000 / 0.000 / -1.067 / -2.889 | void | void |
| Pearson | all within | imputation 0.366, **-0.020 beyond** | composite 0.526 (0.023), **+0.031 beyond**; correction 0.585, **+0.078 beyond** |
| local ratio | composite 0.421, **-0.107 (2x)**; imputation 0.269, **-0.113 (2x)**; correction 0.580, **-0.138 (2x)** | composite 0.449, **-0.076**; imputation 0.275, **-0.112 (2x)**; correction 0.579, **-0.094 (2x)** | composite 0.428, **-0.067 (2x)**; imputation 0.343, **-0.089 (2x)**; correction 0.396, **-0.111** |
| quantile | composite 0.448, **-0.081**; imputation 0.287, **-0.095 (2x)**; correction 0.582, **-0.135 (2x)** | composite 0.429, **-0.096**; imputation 0.305, **-0.082 (2x)**; correction 0.596, **-0.077** | composite 0.422, **-0.073 (2x)**; imputation 0.378, **-0.054 (2x)**; correction 0.424, **-0.083** |
| raw counts | correction-only 0.670, **-0.021 beyond**; else within | all within | all within |
| soft clip | all within | all within | composite 0.511 (0.011), **+0.016 beyond** |
| pseudocount 0.1 | all within | imputation 0.362, **-0.025 (2x)**; correction-only **-0.056** | all within |
| pseudocount 0.01 | all within | imputation 0.355, **-0.032 (2x)** | all within |

All bold values are beyond. Local ratio and quantile also raise the correction-only term on all three models (+0.039 to +0.250), beyond in 5 of the 6 cells. **Decision rows.** No arm beats the offset on the composite on 2+ models. "Up beyond on 1 architecture only": Pearson, asinh and soft clip enter the mix-and-match step for the U-Net only; none lowers a term beyond 2x its range. "Depth-removing arms worse beyond": met for local ratio and quantile, not for Pearson. "Pseudocount 0.1 beyond in either sign": met on conv + attention; the 0.01 point also lowers its imputation. **Expectation.** "No arm beats the offset on 2+": held. Pearson down on 2+: not held (up on the U-Net). Raw down on all three: not held (within). log1p and asinh down on the U-Net: not held. Local ratio and quantile down: held. **Caveats.** The U-Net composite range is 0.010, so U-Net gains of 0.015 count as beyond. Anscombe is a code defect (section 5.1). Pearson track coverage (how many tracks kept the default statistics) was not checked.

### 2.16 DNA fusion (specification 16)

**Question.** The PI: "how we embed and fuse different modalities ... DNA embedding and injection, etc." **Arms.** FiLM by DNA at the input stage (U-Net: at the bottleneck); late join at the bottleneck (conv models); one cross-attention layer at the bottleneck (conv + attention, U-Net); early join at the U-Net stem without the DNA tower and skips. **Prior lead.** U-Net DNA skips removed: nothing changed beyond the range.

| arm, model | composite | imputation | correction | calibration | correction-only |
|---|---|---|---|---|---|
| FiLM, all three | +0.013 / -0.004 / -0.005 | +0.011 / +0.011 / -0.003 | +0.005 / -0.003 / +0.007 | +0.026 / -0.035 / -0.024 | -0.022 / -0.029 / -0.029 |
| late, conv autoencoder | 0.544 (0.018), +0.016 | 0.383 (0.022), +0.001 | 0.733 (0.028), +0.016 | 0.679 (0.038), +0.048 | 0.653 (0.026), **-0.037 beyond** |
| late, conv + attention | 0.502 (0.032), -0.023 | 0.360 (0.004), **-0.027 beyond, 2x** | 0.681 (0.028), +0.008 | 0.606 (0.100), -0.046 | 0.603 (0.040), **-0.057 beyond** |
| cross-attention, conv + attention | 0.513 (0.048), -0.012 | 0.374 (0.005), **-0.012 beyond** | 0.670 (0.048), -0.003 | 0.632 (0.158), -0.020 | 0.609 (0.032), **-0.050 beyond** |
| cross-attention, U-Net | 0.502 (0.006), +0.007 | 0.440 (0.018), +0.008 | 0.499 (0.080), -0.008 | 0.629 (0.071), +0.020 | 0.282 (0.027), -0.036 |
| early, U-Net | 0.484 (0.025), -0.011 | 0.431 (0.010), -0.001 | 0.516 (0.009), +0.009 | 0.556 (0.110), -0.053 | 0.310 (0.092), -0.008 |

FiLM differences are arm minus base for conv autoencoder / conv + attention / U-Net; all are within. Cross-attention minus late on conv + attention: imputation +0.014, beyond (2x). **Decision rows.** Conv autoencoder and U-Net: row G applies (DNA off is within on their four metrics), so their fusion arms are uninformative; row A (all within) also holds, and as written it keeps early concatenation on the conv models and picks the early join on the U-Net. Conv + attention: late lowers imputation beyond (the reverse of row C); row D holds (cross-attention beyond late), but cross-attention is still below the base. **Expectation.** All within: not held on conv + attention (late and cross-attention lower imputation). **Confounds.** U-Net early join has 750,149 parameters (-243,450, -24.5%). Conv + attention cross-attention +44,200 (+4.3%); U-Net cross-attention +30,800 (+3.1%); U-Net FiLM +10,200 (+1.0%). Widths equal the base.

## 3. Scoring-only reads of the base runs

### 3.1 Foreign covariates at scoring

Each target or input gets the covariate row of another experiment of the same assay (the base view of another cell), with no new training. Modes: side `in` (C), `out` (C'), or both; `keep_depth` keeps the true log2 depth, `swap_depth` also takes the donor's depth. Item file: `closeout/closeout E5 scorer extensions.md`. The table is foreign minus true covariates (`p16_sv`) of the same 9 runs, 3 seeds each.

| mode | conv autoencoder | conv + attention | U-Net |
|---|---|---|---|
| inputs only, depth kept | all within | all within; composite -0.045, calibration -0.171 | composite 0.456, **-0.039**; calibration 0.467, **-0.142** |
| target only, depth kept | composite 0.430, **-0.098**; correction 0.552, **-0.165**; calibration **-0.213**; correction-only **-0.177** | composite 0.436, **-0.089**; correction 0.540, **-0.133**; calibration **-0.203**; correction-only **-0.154** | composite 0.380, **-0.115**; correction 0.265, **-0.242**; calibration **-0.186** |
| both, depth kept | composite **-0.095**; correction **-0.170**; correction-only **-0.288** | composite **-0.104**; correction **-0.151**; calibration **-0.245**; correction-only **-0.349** | composite **-0.126**; correction **-0.255**; calibration **-0.210**; correction-only **-0.362** |
| both, depth swapped | every term beyond: composite -0.330, imputation -0.045 | every term beyond: composite -0.293, imputation -0.046 | every term beyond: composite -0.234, imputation -0.040 |

Bold values are beyond. Imputation is within in every `keep_depth` mode on all three. **Rows of the E5 table.** "`*_keep_depth`: correction term lower, beyond": all three models use non-depth covariates to correct. "`in_keep_depth` and `out_keep_depth` differ": the output covariates C' carry the correction; the input covariates C carry little (U-Net calibration only). "Imputation lower under `keep_depth`": not met; the covariates carry no imputation information at test time. **Reading.** Swapping C' changes processing columns that define the requested view: in `rung2_base_heldout/seed0`, `out_keep_depth` changed `read_length` on 124 of 305 targets and `platform_id` on 231. So this test cannot separate the use of the processing knobs from the use of batch columns such as platform. The `swap_depth` mode moves the NB depth offset, so it mostly measures the depth law. **Agreement.** Scrambled-covariate training (2.11) lowered the correction-only term on all three; this test lowers crop-pair correction on all three, including conv + attention and U-Net, where the trained control was within.

### 3.2 Base recipes on the corrected covariate tables

The tables `eic_v1_2` fill lab, antibody and donor and unify the platform vocabulary (Fir `tables/eic_v1_2/CHANGES.md`). The base recipes were retrained on them (`eic_v1_2_rung{2,3,4}_base_heldout`, 3 seeds, same parameter counts). The two versions have different table and eval-set hashes, so the scorer refuses a difference. I show them side by side only.

| model | tables | composite | imputation | correction | calibration | correction-only |
|---|---|---|---|---|---|---|
| conv autoencoder | eic_v1 | 0.528 (0.045) | 0.382 (0.028) | 0.718 (0.012) | 0.631 (0.178) | 0.690 (0.019) |
| conv autoencoder | eic_v1_2 | 0.505 (0.029) | 0.370 (0.021) | 0.681 (0.032) | 0.599 (0.186) | 0.696 (0.031) |
| conv + attention | eic_v1 | 0.525 (0.048) | 0.387 (0.012) | 0.673 (0.031) | 0.652 (0.157) | 0.659 (0.043) |
| conv + attention | eic_v1_2 | 0.518 (0.017) | 0.386 (0.030) | 0.671 (0.030) | 0.627 (0.138) | 0.707 (0.059) |
| U-Net | eic_v1 | 0.495 (0.010) | 0.432 (0.023) | 0.507 (0.071) | 0.610 (0.108) | 0.318 (0.127) |
| U-Net | eic_v1_2 | 0.518 (0.035) | 0.445 (0.038) | 0.538 (0.059) | 0.646 (0.158) | 0.353 (0.070) |

Descriptively, every pair of means differs by less than the larger seed range except the conv autoencoder correction (0.718 against 0.681; ranges 0.012 and 0.032). By the D1 table, "within the range on all three" gives no evidence at this scale that the filled covariates help; the PI decides because the columns are part of the default vector.

## 4. What this settles for the design

All recommendations in this table are mine. "Evidence" says whether the scores support the recommendation beyond the seed range.

| design choice | conv autoencoder | conv + attention | U-Net | evidence |
|---|---|---|---|---|
| control as input | keep | keep | keep | weak: imputation lower without it on all three, beyond on conv + attention only |
| control as output | keep | keep | keep | ruling 14 binds; scores within or mixed in sign |
| average baseline input | off | off | candidate for mix-and-match | weak: U-Net imputation +0.034, beyond |
| two-step routing | off | off | off; if used, fixed base depth | moderate: native lowers conv correction beyond; fixed beats native on the U-Net |
| DNA source | own tower | own tower | own tower | moderate null: language model within on all three; Flashzoi not run |
| DNA on or off | on | on | on | weak: beyond on conv + attention imputation (two arms agree); U-Net width-confounded |
| DNA fusion | early concatenation (base) | early concatenation (base) | base join; the table's row A picks the early join | weak: late joins lower conv + attention imputation; U-Net early join is 24.5% smaller |
| covariate embedding | full | full | full | weak: numeric-only within on conv models but lowers U-Net correction beyond; the table permits numeric-only on the conv models; I keep full because the corrected tables add categorical columns |
| injection form | concatenation | concatenation | concatenation | weak: within or better on all three (D13 simpler arm); tokens hurt U-Net imputation |
| mask mode | drop | drop | drop | moderate null: within everywhere (D13 simpler arm) |
| input depth | native | native | PI decision: fixed 2^23 trades imputation for correction | moderate on the U-Net (composite and correction 2x) |
| loss | NB NLL | NB NLL; imputation weight 4 for mix-and-match | NB NLL | moderate null; one-model gain only |
| count transform | offset, pseudocount 1 | offset, pseudocount 1 | offset; Pearson, asinh, soft clip for mix-and-match | moderate: local ratio and quantile worse on all three; U-Net gains on one model only |
| training views | all views | all views | all views | moderate: the correction-only term collapses without views on all three; alignment-only hurts the conv models |

## 5. Defects and open points

### 5.1 The Anscombe arm is a code defect

All nine Anscombe runs score imputation 0.000, correction -1.067, calibration -2.889 to -2.890 and correction-only 1.000, with seed ranges of 0.001 or less. The output link is mu = s * pos((eta / 2)^2 - 3/8) (`tools/minicandi/components.py`:481-482). The NB head starts with zero weights, zero bias and zero track bias (`components.py`:510-512), so eta = 0 at every bin. The derivative of (eta / 2)^2 at eta = 0 is 0, so no gradient reaches eta or any parameter behind it, and eta stays 0 for the whole run. Every bin then gets the same depth-scaled mean, s * (softplus_10(-0.375) + 1e-4), about 0.0024 * s; only the NB size trains. The config is correct (`count_transform: anscombe`, a recognized name). Evidence: the kit that ran (`/project/def-maxwl/mforooz/onion_kit/CO3/tools/minicandi/components.py`) is byte-identical to the worktree copy (I checked with `diff` on Fir); the training log stays finite (`runs_fir/abl_count_transforms_tf_anscombe_convae/seed0/log.jsonl`, last step loss 2.254, against 0.747 for log1p); a subagent loaded the final checkpoints and found the eta weights and biases still exactly 0 after 10,000 steps. The specification's synthetic check (3 steps, finite loss) could not catch this. The scores are void; the arm needs a fixed initialization and a retrain. I did not fix it.

### 5.2 Missing seeds

- `abl_depth_input_input_depth_fixed_convae`: seed 1 did not finish training (no `DONE`); 2 seeds.
- `rung4_views_base_fastq`: seed 0 scoring failed (`metrics_4448491.out`: "Input/output error" on a CVMFS Python file); 2 seeds.
- `rung4_views_base_alignment`: seed 2 scoring failed (`metrics_4448517.out`: "Disk quota exceeded" on `/scratch/mforooz/tmp`); 2 seeds. No rescoring job is queued.
- `rung4_views_all_same_view_only`: seed 2 has no correction-only value.

### 5.3 Arms whose parameter count or width differs from the base

| arm | change against the base | flag |
|---|---|---|
| DNA off, U-Net | width 25 -> 30; 996,232 parameters (+2,633) | confounded: count path wider |
| DNA fusion early join, U-Net | 750,149 parameters (-243,450, -24.5%); width 25 | confounded: smaller model |
| DNA fusion cross-attention | conv + attention +44,200 (+4.3%); U-Net +30,800 (+3.1%) | minor |
| covariate per-value codes | +19,968 on all three (+1.9% to +2.0%; budget raised) | minor |
| injection by concatenation | -15,136 / -15,136 / -15,336 (-1.5%); FLOPs +13% to +31% | minor |
| DNA off, conv models | -4,928 and -6,336 (-0.5%, -0.6%); widths unchanged | negligible |
| two-step (both arms) | parameters equal; FLOPs per window 2.0x | compute confound |

Every other arm differs by less than 1% in parameters, with the base width (60, 104, 25).

### 5.4 Other caveats carried

- The scrambled-covariates arm is scored with true covariates (2.11).
- log1p, asinh, Anscombe and Pearson also change the output link; raw counts remove the output depth offset (2.15).
- The fixed input depth arm caps every assay class at 2^23 (2.8).
- The ablation arms ran on newer kits than the references (kit CM and RTf). I did not rerun `golden.py` checks.

### 5.5 Not checked

- The depth-thinning and MAPQ + no-dedup pair columns of the views ablation on the conv models (`where_views.py`).
- The shared-error diagnostic of the average-input arm.
- Pearson track coverage in the checkpoints.
- The mechanism of the U-Net correction gain under fixed input depth.
- The U-Net numeric-only correction drop, which the specification says to check for a wiring fault.
