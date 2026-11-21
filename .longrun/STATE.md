# State

Rewritten, never appended. Cap: 5 KB.

## Goal now
Find one miniCANDI recipe that Pareto-dominates the three base models on imputation, correction and calibration on eic_v1_2 (D1); priority 2 is ChIP-control prediction (D18). Task t370, branch exp/t370-minicandi-recipe (PR 63), worktree .claude/worktrees/recipe.

## Horizon
DEADLINE 2026-11-22 18:30 UTC (D31, PI: "within 24 hours"). Budget checkpoint 6,000 CPU-h (D20). Fir is the scoring cluster (eval sets live only there); Nibi may be used for training-only overflow (PI 2026-11-21).

## In flight (Fir), at 2026-11-21 22:30 UTC
- Screen result (p16): Pareto winner halves U-Net (D35); runs_rcp/_sel/pareto_2130.json. Spend 4,954 CPU-h by 21:50 UTC; new submissions project +1,637 (past the 6,000 checkpoint, D20: report).
- Kit RCP4 (79c82df7): 7 timed-out run-seeds rerun at 6 h; combo arms halves + {mix_pearson, mix_fixdepth, headskip_film, mix_avg} U-Net x 3 seeds at p16; p256 confirm of halves U-Net, mix_fixdepth U-Net, mix_svonly conv AE / conv+attn. Ids in SUBMITTED.tsv.
- Still running from before: rest of screen (RCP2/RCP3), CNN seeds, curve U-Net reruns, control arms + control evals.
- EIC: fixes merged (fc4eb40a); kit RCP5 pushing; then resubmit eic_val (FRAGLEN_TABLE) and Avocado chain (D33). Average entry rerun 4943718 (KIT=RCP2 export).

## Next
1. After RCP5: resubmit eic_val array and Avocado chain.
2. When combos and p256 score: rerun pareto_select at p16 and p256; pick the recipe; two-step retest on it.
3. Deck by 2026-11-22 18:30 UTC.

## Open asks to the PI
None (Avocado yes, blind no, curve fix yes: D32-D34).
