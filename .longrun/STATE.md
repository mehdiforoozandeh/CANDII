# State

Rewritten, never appended. Cap: 5 KB.

## Goal now
Find one miniCANDI recipe that Pareto-dominates the three base models (conv AE, conv+attn, U-Net) on imputation, correction and calibration on eic_v1_2 (D1), and get ChIP-control prediction working (R8). Task t370, branch exp/t370-minicandi-recipe, worktree .claude/worktrees/recipe.

## Horizon
Time horizon: 2026-12-01 (10 days from 2026-11-21). First measured result due: 2026-11-24 (screen, 16 pair regions). Budget: 6,000 CPU-h checkpoint (D20). Scope P0+P1 only; priority 1 recipe, 2 control (D18).

## In flight
- Planner (background) writing .orchestrate/plan.md (git-ignored; snapshot to cruxvault/INSTRUCTIONS/miniCANDI_closeout/run_state/ is NOT allowed: protected; snapshot to cruxvault/notes/t370_plan.md instead). Nothing on the clusters yet.
- R19 verified (9/9 eic_v1_2 base runs scored at 256 regions on Fir).

## Next
1. Review plan: rules check, cold plan review (opus), amend once.
2. Wave 1 builders in worktrees under ../.orchestrate-wt/CANDII/rcp-<id>.
3. Probe one screen job at real scale, then batch; first screen result by 2026-11-24.

## Open asks to the PI
none (PI asleep; decide with recommendations and log)
