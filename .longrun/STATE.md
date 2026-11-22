# State

Rewritten, never appended. Cap: 5 KB.

## Goal now
Find one miniCANDI recipe that Pareto-dominates the three base models on imputation, correction and calibration on eic_v1_2 (D1); priority 2 is ChIP-control prediction (D18). Task t370, branch exp/t370-minicandi-recipe (PR 63), worktree .claude/worktrees/recipe.

## Horizon
DEADLINE 2026-11-22 18:30 UTC (D31, PI: "within 24 hours"). Budget checkpoint 6,000 CPU-h (D20). Fir is the scoring cluster (eval sets live only there); Nibi may be used for training-only overflow (PI 2026-11-21).

## In flight (Fir), at 2026-11-22 01:25 UTC
- Fir fault 23:25-23:56 UTC killed 251 jobs at start (signal 53, many nodes). All resubmitted on kit RCP5 (fc4eb40a) at 01:25 UTC: p256 confirm (halves U-Net, mix_fixdepth U-Net, mix_svonly conv AE / conv+attn), combo arms x 3 seeds, unfinished s1a/s1b/s1c and control arms (6 h train limit except CNN/average 8 h). Ids in SUBMITTED.tsv.
- Screen result (p16): Pareto winner halves U-Net (D35); runs_rcp/_sel/pareto_2130.json. Spend 4,954 CPU-h by 21:50 UTC.
- EIC: eic_val array 5022609; Avocado chain 5022622-5022627 (score after 5022627); Average entry 5022628.

## Next
1. After 4945495: run eic_val_compare.py score for Avocado; score the U-Net validation predictions.
2. When combos and p256 score: rerun pareto_select at p16 and p256; pick the recipe; two-step retest on it.
3. Deck by 2026-11-22 18:30 UTC.

## Open asks to the PI
None (Avocado yes, blind no, curve fix yes: D32-D34).
