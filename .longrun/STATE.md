# State

Rewritten, never appended. Cap: 5 KB.

## Goal now
Find one miniCANDI recipe that Pareto-dominates the three base models on imputation, correction and calibration on eic_v1_2 (D1); priority 2 is ChIP-control prediction (D18). Task t370, branch exp/t370-minicandi-recipe (PR 63), worktree .claude/worktrees/recipe.

## Horizon
Time horizon: 2026-12-01. Recipe freeze 2026-11-29 (D27). First measured result due: 2026-11-24 (screen, 16 pair regions). Budget: 6,000 CPU-h checkpoint (D20). Spent so far: probe ~26 CPU-h projected.

## In flight
- Plan .orchestrate/plan.md (snapshot cruxvault/INSTRUCTIONS/t370_recipe/run_state/plan.md). Wave 1 merged (C1-C10) + C29 read-disjoint pairs (D29).
- Fir, kit RCP1 (/project/def-maxwl/mforooz/onion_kit/RCP1): rcp_tests 4863258; U-Net probe train 4863259, predict 4863260-63, metrics 4863264 -> runs_rcp/rcp_probe_base_unet/seed0/p16_sv. Watch: background task on laptop (10 h limit).
- Builders (opus, background): C15 control switches, C16 EIC val tool, C17 Max panel; worktrees ../.orchestrate-wt/CANDII/rcp-C15..17, base 1da07953.

## Next
1. Gate G0: probe terms vs runs_fir/eic_v1_2_rung4_base_heldout/seed0/p16_sv and unit costs; rcp_tests result.
2. On G0 pass: re-push kit with C15-C17 merged, then C12 (skip, placement, no-cov, halves), C13 (pre/post curve, mix, 1 CNN seed), C14 control probe.
3. G0b CNN cost read, then C27 (17 CNN seeds), C28 (8 control evals), C18 control arms, C19 EIC probe.

## Open asks to the PI (non-blocking)
1. Training-only Avocado retrain on validation (~86 GPU-h) for the EIC comparison; until then compare vs the Average. Recommend yes.
2. Blind-row build (t184, ~22,700 CPU-h) to allow one blind pass; default: no blind pass this run (D26).
3. FYI: added arm D29 (read-disjoint training pairs), ~150 CPU-h, from the three analyses.
