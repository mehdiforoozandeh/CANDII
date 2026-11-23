# State

Rewritten, never appended. Cap: 5 KB.

## Goal now
Close the session with one trusted miniCANDI architecture (PI 2026-11-22, after the talk): the recipe must Pareto-dominate the three base models on imputation, correction and calibration on eic_v1_2 (D1); priority 2 is ChIP-control prediction (D18). Task t370, branch exp/t370-minicandi-recipe (PR 63), worktree .claude/worktrees/recipe.

## Where the recipe stands (2026-11-23 05:05 UTC)
- 256 regions (D43): pairs + Pearson-residual input U-Net, PASS (imp 0.445, cor 0.774, cal 0.625).
- 16 regions (D50): full combination pairs + average + Pearson PASS, score 0.249 (imp 0.500, cor 0.816, cal 0.685), the best combination; scale_w2x score 0.417. Head skip off confirmed NEAR at 256 regions (cor 0.674): head skip stays.
- Control on the recipe (D51): works (skill 0.131); weight 1 kept.
- Recalibration (D52): calibration 0.625 -> 0.767 (ensemble) / 0.749 (one seed) at 256 regions; ships with the final.
- Final candidate = combination at 160k windows (combo_w2x), weight-1 control, plus recalibration.

## In flight (Fir, kit RCP9)
- Combination p256 confirm: 5356043-5356070 (27 jobs, 342 CPU-h projected).
- combo_w2x: train 5356071/5356078/5356095 (16 h limit), p16 chained.
- Avocado 5022624 still pending. rcp_rec_ctlw4 seed 2 timed out (2 seeds read; not resubmitted).
- Ensemble on pairs + average: ens3 missing (only seed0_recal landed); low priority.

- Blind-row build batch 1 on Nibi (submitted 08:01 UTC, 72 rows, ~22,700 CPU-h): stage 28629674/76/99, 28629702/04/10/12/16/18; build 28629675/77, 28629700/03/05/11/13/17/19; signal 28629678 (ChIP/ATAC), 28629701 (DNase). Watch scratch; move BAMs and delete staged FASTQs as rows finish to fall under 1024G soft quota.

## Next
1. When combo p256 lands: pareto_select at 256; compare with the recipe.
2. When combo_w2x p16 lands: if PASS and not worse beyond range than the combination, submit its p256 confirm.
3. Final architecture decision; update GOALS R1-R4, R8; refresh the deck; tell the deviance session the final arm.

## Open asks to the PI
Route PI-facing items and milestone status (2-3 lines) to foreman session local_b5c55e32-4e2a-4edc-b4b7-6f0134e8bc43 (setup 2026-11-23).
- PR 65: PI says merge after the 256-region results; deviance session resolving conflicts first. Do not push PR 65's files until the foreman says resolved.
