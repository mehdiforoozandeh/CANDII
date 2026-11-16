---
title: "E1: the 16-versus-256 region check and settled scores"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# E1: the 16-versus-256 region check and settled scores

## 1. What the PI asked

- `long-horizon notepad.md`:535 (ruling 42): "PI's answer: "Smallest of 8/16/32 that passes a check", one tier. Imputation items stay on all 256 regions."
- `long-horizon notepad.md`:555 (ruling 44): "No selection is called settled from a screening score."
- `long-horizon notepad.md`:556 (ruling 45): "The check that picks the pair-region count runs on rung-4 models now; the rung-1 version follows in a later phase."
- `long-horizon decisions.md`:104 (D14 condition 1): every comparison is run "with at least 3 seeds, and the selection rule has settled each comparison."

Success criteria served: 3 (tested evals) and 4 (design settled). Audit rows: B16, F1, F2.

## 2. Question

Is a correction-pair score on 16 of the 256 evaluation regions (the screening tier) equal, within the preregistered tolerances, to the score on all 256? If yes, the existing 16-region scores become settled scores. If no, which runs must be rescored, and what do the settlements become?

## 3. State today

- Preregistration: `results/pair_region_check/PAIR_CHECK_PREREGISTRATION.md` (2026-11-14): five conditions, constants pinned in `tools/onion_eval/pair_regions_check.py`, nested subsets at positions 0, 256/p, 2 x 256/p, ... It names rung-1 runs; ruling 45 moved it to rung 4 (U-Net).
- No 256-pair-region model scoring exists (`results/ladder_audit/COVERAGE.md` row 39). Every model score is screening tier (audit row B16).
- Composite version 3 (Log 2026-11-16T08:50Z) uses the 18 crop pairs with no duplicate-keeping side as the correction term, and adds a correction-only term. The preregistration predates both.
- Settlement code writes `settlements_v3.tsv` (Log 2026-11-16T13:10Z). The notepad holds no settled-ablation entry (audit row F1).

## 4. Prior leads

- Finding `f60_screening_tier_against_the_full_set_on_t.md` (weak; baselines only, old composite): at 16 regions every correction CRPS was 0.029 to 0.036 higher than at 256, a shift common to all four baselines; the position average's composite was 0.939 at 16 regions against 0.817 at 256. Prediction: a common shift cancels in the gap-closed ratio only in part, so condition 1 can fail on the level while condition 3 (same D13 verdict) passes.
- `results/correction_metric_review/REVIEW.md` section 4: the 18-pair term rests on 6 cells and 6 assays, and "6 cells make the cell bootstrap wide". Prediction: the 18-pair term is the term most likely to fail condition 1 at 16 regions.
- Seed ranges of the U-Net base, 16 regions: correction 0.071, composite 0.010 (`FINAL_ROUND_EQUAL.md`).

## 5. Work

1. Amend the preregistration before any 256-region score exists: add the version-3 correction term (18 pairs) and the correction-only term to conditions 1 and 4; name the runs (U-Net base and U-Net scrambled covariates, 3 seeds each, one cluster). Commit it. **Done 2026-11-16 (section 9): the runs are 4 arms x 3 seeds (foreman's brief), and a rank-order condition 6 and a stored-score condition S are added.**
2. Verify that the 16-region screening set (`evalset_p16.json`, sha a82cdfd1) uses the preregistered subset rule; if not, the check must also score p = 16 under the rule.
3. Score the 12 runs with all 256 pair regions (kit COE1 = kit AB scorer plus `run-v3`, same-view pass), on CPU with inference batch 8 for all 12, as the stored `p16_sv` scores (notepad L322). Script: `slurm/onion_eval/e1_pair_check_p256.sh` (dry run by default).
4. Run `pair_regions_check.py run-v3` (the script's step 2, after the scorings); write `PAIR_CHECK.md` with the pass or fail of each condition for p = 8, 16, 32.
5. If p = 16 passes: label the existing `p16_sv` scores "settled". Else rescore at the chosen p every run that enters a settlement: the final-round arms (A9), the nine ablations (A1), the controls (A2, A3) and the new arms of A4 to A8.
6. Run `select_rule.py` settlements on the settled scores; log each D13 outcome in the notepad; record findings in the crux tree.

## 6. How the result answers the question

| outcome | conclusion |
|---|---|
| p = 8 or 16 passes all five conditions | screening scores at that size are settled; no rescoring for p = 16 |
| only p = 32 passes | rescore every run in a settlement at 32 pair regions |
| none passes | pair jobs stay on 256 regions; the foreman tells the PI the measured time per run (preregistered rule) |
| D13 verdicts differ between 16 and 256 | every screening verdict is unreliable; settlements wait for rescoring |

## 7. Dependencies, cost, risks

- Depends on nothing for the check. The settlements depend on A1 to A10.
- Cost: the check was planned at 15.4 GPU-slice-h + 114 CPU-h (`COVERAGE.md` row 39, from plan Amendment 10). On Fir CPU, time one 256-region scoring first (notepad L531) and multiply.
- Risk 1: the check rests on one architecture and one pair of arms (preregistration, "Limit").
- Risk 2: the region file rule (`onion region budget and selection.md`:44) raises regions or seeds when 2 x the composite seed range exceeds 0.05; final-round ranges of 0.010 to 0.088 trip it. The PI moved the evaluation-region count to the follow-up work (`closeout todo.md`), so this item does not change the 256 regions; state the trip in the report.

## 8. Done when

`PAIR_CHECK.md` reports the chosen pair-region count under the amended preregistration, every run in a settlement has a score at that count, and each D13 outcome is logged in the notepad as settled.

## 9. Amendment of 2026-11-16 to the preregistration (PI ruling of 2026-11-16)

The full text, dated, with the reason and below the unchanged original, is the section "Amendment of 2026-11-16" of `cruxvault/results/pair_region_check/PAIR_CHECK_PREREGISTRATION.md` (git-ignored; this branch's copy). The constants are `pair_regions_check.thresholds_v3()`; the test `test_thresholds_v3_equal_the_amended_preregistration` holds the two equal. In short:

- Terms: the version-3 correction term (gap closed with the oracle reference, 18 crop pairs with no duplicate-keeping side) and the correction-only term replace headline 4 in conditions 1, 2 and 4. Headline 5 and the replicate rows stay.
- Runs: conv autoencoder, conv + attention and U-Net base recipes, and the U-Net with DNA-only skip connections; 3 seeds each; scored into `p256_sv` on `evalset.json` on Fir CPU.
- Condition 3: the D13 verdicts (composite version 3, no arm called simpler) of conv AE vs conv + attention, conv AE vs U-Net, conv + attention vs U-Net, U-Net vs U-Net DNA-only skip are equal on the subset and on 256 regions.
- Condition 6 (new, the decision rule): for each arm pair and for the composite, the correction term and the correction-only term, a 256-region difference of arm means larger than the larger seed range keeps its sign on the subset; a smaller difference is a tie and is not tested.
- Settled: the stored 16-region scores are settled when p = 16 passes all six conditions AND the stored `p16_sv` values equal the p = 16 subset of the 256-region scoring within relative 1e-6 (condition S).
- Expectation: on 16 regions all four comparisons are composite ties (TIE_ASK_PI); condition 6 carries the trade-offs (U-Net lower on the correction term by 0.17 to 0.21 and on the correction-only term by 0.34 to 0.37).

