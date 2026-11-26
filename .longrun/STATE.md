# State

Rewritten, never appended. Cap: 5 KB. Last rewrite 2026-11-25 ~09:40 UTC.

## Goal now
Close the session with one trusted miniCANDI architecture (PI 2026-11-22): the recipe Pareto-dominates the three base models on imputation, correction and calibration on eic_v1_2 (D1); priority 2 is ChIP-control prediction (D18). Task t370, branch exp/t370-minicandi-recipe (PR 63), worktree .claude/worktrees/recipe. PI meeting 2026-11-25 22:00 UTC; foreman deadline 20:00 UTC.

## Where the recipe stands
- FINAL (D55, PI): U-Net + read-disjoint pairs + average + Pearson input, head skip, 160k windows (rcp_combo_w2x_unet), control weight 1, recalibration after training.
- 256 regions, 3 seeds: imp 0.503, cor 0.818, cal 0.717 vs bars 0.445, 0.675, 0.646.
- Recalibration study DONE (79.1 CPU-h of 150): sent to foreman with Option B = do not run. Paper note cruxvault/presentations/t370-minicandi-recipe/recalibration.md; reports cruxvault/results/t370/recal_study/ (converted); vault q78-q80, f145-f147. Covariate-prompt test (16 targets) done, same folder.

## In flight
- Fir 6033930 DONE (62.7 CPU-h): val3/out/counts sent to pval session; use the val3 C25M26, not the older one in s0/counts.
- Fir 5761693 DONE: 43 counts, copy 5761694 done, 649.8 CPU-h.
- Avocado chain on Fir GPU slices (PI 2026-11-26): shared 6291945, genome 6291946, predict 6291947, bigwig 6291948; old 5022624-27 cancelled. Never resubmit. Nibi move dropped.
- Nibi blind build batch 1 (t184): build 28737209/11/15/45/47/64/67/69/75, signal 28737212, 28737239 wait out NibiMaintenance. Never resubmit.

## Next
1. When val3 counts land: send the out dir to the pval session; copy the foreman.
2. When 5761693 ends: confirm 34/34 counts and the recal_wide copies; report cost.
3. 21:00 UTC: Avocado move per the PI rule.
4. Blind pass: when the Nibi build lands, predict with rcp_combo_w2x_unet.

## Open asks to the PI
Route PI-facing items to foreman session local_b5c55e32-4e2a-4edc-b4b7-6f0134e8bc43.
- Read-start retrain (priced 1,350-2,250 CPU-h): waits; do not start.
- Neutral-prompt sensitivity board: PI said not urgent; do nothing.
- Known repo defect, not mine: tests/test_results_folders.py fails on a tracked file missing on disk (cruxvault/results/t371/compare/_x).
