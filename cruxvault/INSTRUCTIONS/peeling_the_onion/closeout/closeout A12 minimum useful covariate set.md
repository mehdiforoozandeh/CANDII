---
title: "A12: minimum useful covariate set"
parent: "[[closeout todo]]"
created: 2026-11-17
---

# A12: minimum useful covariate set

## 1. What the PI asked

PI ruling of 2026-11-17 (verbatim): "let's have all covariate fields that we can included. one last step of this session and one of its deliverables is to find the minimum set of useful covariates. a ranked list, a curve of how much performance do we lose by excluding each covariate and which ones are basically useless and we can safely remove"

## 2. Question

Which covariate fields does miniCANDI need? The answer is (1) a ranked list of the fields, (2) a curve of the score lost as fields are excluded, and (3) the fields that can be removed with no loss beyond the seed range.

## 3. State today

- The corrected tables `eic_v1_2` hold every field that we can fill (lab, antibody, donor, platform, and the processing fields). They are the default from now on.
- Leads: scrambled covariates lower the correction-only term on all three models and leave imputation unchanged. Foreign covariates on the output side at scoring lower gap closed by 0.13 to 0.24; on the input side they change little. Numeric-only covariates are within the seed range on the conv models. Evidence: [[ABLATIONS_READ]] sections 2.7, 2.11, 3.1.

## 4. Work steps

1. List every covariate field of `eic_v1_2`, with its type and its side (input, output).
2. Screening without training: at scoring, replace one field at a time with a foreign value (the scorer option for foreign covariates), per side, on the three base recipes trained on `eic_v1_2`. Rank the fields by the score lost.
3. Confirmation with training: leave-one-field-out runs for the top fields and one run with all low-ranked fields removed, 3 seeds, 3 models.
4. Build the curve: fields removed in rank order (least useful first) against imputation, gap closed, calibration, composite and the correction-only term.
5. State the minimum set: the smallest set whose scores are within the seed range of the full set.

## 5. Done when

The ranked list, the curve and the minimum set are in one results file under `cruxvault/results/`, each number with its seed range, and the finding is recorded.

## 6. Order

This is the last ablation step of the session. It runs after the mix-and-match combinations and before the settled-design document (M1).
