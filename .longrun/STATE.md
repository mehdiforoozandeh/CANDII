# State

Rewritten, never appended. Cap: 5 KB.

## Goal now
Find one miniCANDI recipe that Pareto-dominates the three base models on imputation, correction and calibration on eic_v1_2 (D1); priority 2 is ChIP-control prediction (D18). Task t370, branch exp/t370-minicandi-recipe (PR 63), worktree .claude/worktrees/recipe.

## Horizon
DEADLINE 2026-11-22 18:30 UTC (D31, PI: "within 24 hours"). Budget checkpoint 6,000 CPU-h (D20). Fir is the scoring cluster (eval sets live only there); Nibi may be used for training-only overflow (PI 2026-11-21).

## In flight (Fir), at 2026-11-22 05:40 UTC
- Selection at 05:25 UTC (runs_rcp/_sel/pareto_0525_p16_sv.json, _p256_sv.json). 16 regions: PASS = halves U-Net, halves+mix_avg U-Net, halves+mix_pearson U-Net (2 seeds), halves+headskip_film U-Net. 256 regions: halves U-Net eligible on all three floors but NEAR (correction 0.7299 vs best base 0.6750; diff below its own seed range 0.0707); no winner yet.
- 256-region confirmation of the three PASS combos submitted 05:35 UTC (72 jobs, 913 CPU-h projected; D39). Watch bj140el0j exits at 8 scored run-seeds or 7 h.
- Still training: halves+mix_pearson seed 0, halves+mix_fixdepth seed 1, curve-CNN core-FiLM-off U-Net seed 2.
- EIC: eic_val array 5022609 (index 7 node-failed, resubmitted as 5050354); Avocado train 5022624 pending (Priority), then 5022625-5022627, then score; Average entry 5022628 running.

## Next
1. When combo 256-region scores land: rerun pareto_select at 256 regions; pick the recipe (D1); record it; two-step retest on it if time allows.
2. After 5022627 and 5022609: eic_val_compare.py score for Avocado, the U-Net and the Average entry.
3. Deck by 2026-11-22 18:30 UTC; start the draft by 12:00 UTC with the 16-region and current 256-region results.

## Open asks to the PI
None (Avocado yes, blind no, curve fix yes: D32-D34).
