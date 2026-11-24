# State

Rewritten, never appended. Cap: 5 KB.

## Goal now
Close the session with one trusted miniCANDI architecture (PI 2026-11-22, after the talk): the recipe must Pareto-dominate the three base models on imputation, correction and calibration on eic_v1_2 (D1); priority 2 is ChIP-control prediction (D18). Task t370, branch exp/t370-minicandi-recipe (PR 63), worktree .claude/worktrees/recipe.

## Where the recipe stands (2026-11-24)
- FINAL (D55, PI): U-Net + read-disjoint pairs + average + Pearson input, head skip, 160k windows (rcp_combo_w2x_unet), control weight 1, recalibration after training.
- 256 regions, 3 seeds: imp 0.503, cor 0.818, cal 0.717 vs bars 0.445, 0.675, 0.646. R1-R4, R8 done.

## In flight (Fir, kit RCP9)
- Avocado 5022624 still pending. rcp_rec_ctlw4 seed 2 timed out (2 seeds read; not resubmitted).
- Ensemble on pairs + average: ens3 missing (only seed0_recal landed); low priority.

- Blind-row build batch 1 on Nibi FAILED at staging (ENCODE portal search returns HTTP 405 to the stage code); 0 CPU-h used; builds cancelled; signal passes 28629678/701 stuck pending. Waits for Mehdi: fix route for the FASTQ lookup.

## Next
1. Refresh the deck with the final architecture; tell the deviance session the final arm.
2. Blind pass: when the Nibi build lands, predict with rcp_combo_w2x_unet (+recalibration).
3. Send the pval session (local_0e44c034) the predict.py counts/ paths for C19M22, C40M17, C12M02; copy the foreman.

## Open asks to the PI
Route PI-facing items and milestone status (2-3 lines) to foreman session local_b5c55e32-4e2a-4edc-b4b7-6f0134e8bc43 (setup 2026-11-23).
- Read-start retrain (priced 1,350-2,250 CPU-h): waits for Mehdi; do not start.
- PR 65: PI says merge after the 256-region results; deviance session resolving conflicts first. Do not push PR 65's files until the foreman says resolved.
