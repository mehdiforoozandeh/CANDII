---
title: "A2: basic controls on three models (scrambled covariates, mask mode, injection form)"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# A2: basic controls on three models (scrambled covariates, mask mode, injection form)

## 1. What the PI asked

- `miniCANDI architecture ablations.md`:75-81, "Ablations at every rung": "DNA off." / "Scrambled covariates, as a control for covariate use." / "Covariate injection: concatenation, FiLM, or tokens." / "Matched parameters, then matched compute." / "Mask tokens, or dropping the missing tracks."
- `long-horizon decisions.md`:104 (D14 condition 1): "Rungs 1 to 6 of [[miniCANDI architecture ablations]] and their listed per-rung ablations have run with at least 3 seeds, and the selection rule has settled each comparison."

Success criterion served: 4 (design settled and tested). Audit row: D15. DNA off is in A1 (ablation 6); matched compute is A9.

## 2. Question

On conv autoencoder, conv + attention and U-Net: (1) does the model use its covariates (scrambled covariates must score worse, above all on correction); (2) does a mask token beat dropping the missing tracks; (3) does the covariate injection form (FiLM, concatenation, tokens) change the three goals?

## 3. State today

From `results/ladder_audit/COVERAGE.md` rows 11, 13, 14, 16 (screening, composite version 2):
- Scrambled covariates: pointwise MLP and U-Net only, 3 seeds each (Nibi, against the old base recipe).
- Mask mode: pointwise MLP only, 2 scored seeds. Configs `arms/rung{2,3,4}/mask_drop.json` exist; the base of these rungs uses a mask token.
- Injection: FiLM is the base on rungs 1 to 4. Concatenation: pointwise MLP, 2 seeds. Tokens: pointwise MLP, 1 seed; configs exist for conv + attention and U-Net, not for the conv autoencoder (not applicable).
- The arm configs under `configs/arms/` derive from the old `base.json`, not from `final/rung{2,3,4}/reference_equal.json`, the reference of A1.

## 4. Prior leads

- U-Net scrambled covariates: composite +0.017 within the range; specific variance -0.029 beyond; gap closed on all 102 pairs +0.101 beyond, i.e. the control scored better (`COVERAGE.md` row 11). On the 18 clean crop pairs: 0.420 against 0.519, within 2 x 0.081 (`correction_metric_review/REVIEW.md` L144). Prediction: under composite version 3 the control is worse on correction, but perhaps not beyond the D13 bar.
- Pointwise MLP scrambled-all (depth offset included): composite -0.086 beyond (`COVERAGE.md` row 12). Prediction: most covariate use is depth use.
- Pointwise MLP concatenation +0.007 and mask token +0.002 on composite, within (`COVERAGE.md` rows 13, 16). Prediction: no difference at the larger models either.
- Final round, FiLM as an arm: axial 0.521 against base 0.517 (1 seed); Perceiver IO 0.395 against 0.335 (range 0.088) (`FINAL_ROUND_EQUAL.md`).

## 5. Work

1. Write the arm configs on `final/rung{2,3,4}/reference_equal.json` with `stage2_arms.py`-style one-key changes: `scrambled_covariates` (3 architectures), `mask_drop` (3), `inject_concat` (3), `inject_tokens` (conv + attention, U-Net). A json diff against the reference shows one changed key.
2. Check parameter counts with `registry.build_model`; an arm whose count differs by more than 10% from 1.0e6 gets a `param_budget` note.
3. Submit on Fir CPU, 3 seeds, kit of A1, scored into `p16_sv` with composite version 3: 11 arm-architecture pairs = 33 runs.
4. Read with `select_rule.py arm-summary`; log D13 outcomes; record findings.

## 6. How the result answers the question

"Beyond" = difference of seed means larger than the larger seed range (max minus min, 3 seeds).

| outcome | conclusion |
|---|---|
| scrambled worse than reference beyond the range on correction, on 3 architectures | the covariates drive correction; the correction metric works as a test |
| scrambled within the range on correction | the model corrects by depth or by copying, not by covariates; inspect the correction pairs before trusting any correction verdict |
| mask drop or token: within the range on all three | by D13 the simpler arm stays (state which is simpler in the arm config) |
| concatenation or tokens beyond FiLM on 2+ architectures | change the injection form in M1 |

## 7. Dependencies, cost, risks

- Depends on nothing. Rerun the scrambled arm after D1 if the covariate vector changes.
- Cost: per-seed cost 32.5 / 22.4 / 32.3 CPU-h (`ablation_specs/1_control_input.md` section 7). Three arms on 3 architectures: 3 x 262 = 786 CPU-h; tokens on 2: 3 x (22.4 + 32.3) = 164 CPU-h; total about 950 CPU-h.
- Risk: the scrambled control shuffles covariates across tracks; the scrambled-all form also hides depth. Report which form ran.

## 8. Done when

The 33 runs are scored, each control has a D13 outcome per architecture in the notepad, and the scrambled-covariates result on correction is stated as pass or fail of the control.
