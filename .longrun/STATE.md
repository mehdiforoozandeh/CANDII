# State

Rewritten, never appended. Cap: 5 KB.

## Goal now
Find one miniCANDI recipe that Pareto-dominates the three base models on imputation, correction and calibration on eic_v1_2 (D1); priority 2 is ChIP-control prediction (D18). Task t370, branch exp/t370-minicandi-recipe (PR 63), worktree .claude/worktrees/recipe.

## Horizon
DEADLINE 2026-11-22 18:30 UTC (D31, PI: "within 24 hours"). Budget checkpoint 6,000 CPU-h (D20). Fir is the scoring cluster (eval sets live only there); Nibi may be used for training-only overflow (PI 2026-11-21).

## In flight (Fir)
- Screen s1a + s1b on kit RCP2: jobs_C12.tsv, jobs_C13.tsv. At 18:26 UTC: 90/112 trained, 8 scored.
- Curve pre/post U-Net NaN (3/6 runs): fixed in e8dec6a8 (exponent caps), kit RCP3; old dirs in runs_rcp/_nan_RCP2; 6 U-Net curve run-seeds resubmit on RCP3 by a laptop waiter after the kit push.
- CNN cost seed rcp_cac_cnn_unet s0 (RCP2, old code); s1c (17 CNN seeds) to go on RCP3.
- Control: 9 base control evals (4889711 done, 3.1 CPU-h; 4906119-26); 27 control-arm run-seeds (trains 4906134-4906354) each with a chained control eval (4906361-4906402).
- EIC: Avocado training-only chain 4906094?-4906099 (H100, up to ~33 h worst case: may miss deadline); base U-Net s0 validation prediction array 4906465; Average entry 4906466.
- Tests rerun 4888524 (RCP2).

## Next (24 h plan)
1. Submit RCP3 curve reruns + s1c CNN seeds.
2. As p16 scores land (~next 6 h): pareto_select -> finalists; 256-region confirm of finalists; two-step retest on winners.
3. Deck of the recipe result by the deadline, with control and EIC sections as far as they landed.

## Open asks to the PI
None (Avocado yes, blind no, curve fix yes: D32-D34).
