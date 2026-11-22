# State

Rewritten, never appended. Cap: 5 KB.

## Goal now
Find one miniCANDI recipe that Pareto-dominates the three base models on imputation, correction and calibration on eic_v1_2 (D1); priority 2 is ChIP-control prediction (D18). Task t370, branch exp/t370-minicandi-recipe (PR 63), worktree .claude/worktrees/recipe.

## Horizon
DEADLINE 2026-11-22 18:30 UTC (D31). PI talk 2026-11-22 19:00 UTC. Budget checkpoint 6,000 CPU-h (D20). Fir scores; Nibi training-only overflow.

## Where the recipe stands (12:30 UTC)
- Recipe = halves U-Net (read-disjoint pairs). 16 regions: imputation 0.4263, correction 0.7535, calibration 0.6872 (3 seeds). 256 regions: correction 0.7299 (range 0.0707); eligible on all three floors, NEAR (no term beyond).
- Evolution figure for the talk: cruxvault/INSTRUCTIONS/t370_recipe/evolution/evolution.png (script plot_evolution.py; reads the copied tables only).

## In flight (Fir)
- 256-region confirm of combos (D39): headskip_film 3/3 scored, mix_pearson 2/3 (+ seed 0 submitted 12:05 UTC, 9 jobs), mix_avg 1/3 (seeds 0, 2 running). Watch bj140el0j.
- D41 arm halves_headskip_off (copy-path test): train 5195200, 5195206, 5195212 + p16 scoring (kit RCP6). Then SUB=p256.
- EIC: U-Net validation scoring 5195226 (5194524 failed: KIT unbound, fixed). Avocado 5022624 pending on an unavailable node; misses the deadline.

## Next
1. When combos and headskip_off land: rerun pareto_select at 16 and 256 regions; record the pick (D1); refresh the figure (rerun plot_evolution.py after copying the tables).
2. eic_val table U-Net vs Average: runs_rcp/eic_val/table_unet_vs_average.tsv.
3. After the talk: D42 scaling test (2x, 4x params; 2x windows; with pairs).

## Open asks to the PI
None.
