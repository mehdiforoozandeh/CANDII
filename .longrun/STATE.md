# State

Rewritten, never appended. Cap: 5 KB.

## Goal now
Find one miniCANDI recipe that Pareto-dominates the three base models on imputation, correction and calibration on eic_v1_2 (D1); priority 2 is ChIP-control prediction (D18). Task t370, branch exp/t370-minicandi-recipe (PR 63), worktree .claude/worktrees/recipe.

## Horizon
Time horizon: 2026-12-01. Recipe freeze 2026-11-29 (D27). First measured result due: 2026-11-24 (screen, 16 pair regions). Budget: 6,000 CPU-h checkpoint (D20); plan upper bound 16,138 CPU-h worst case.

## In flight
- Plan: .orchestrate/plan.md (git-ignored; snapshot cruxvault/INSTRUCTIONS/t370_recipe/run_state/plan.md). 8 waves, chunks C1-C28 (C26 deleted).
- Wave 1 (10 opus builders, background), worktrees /Users/mforooz/Desktop/research/libbrechteam@sfu/.orchestrate-wt/CANDII/rcp-C1..C10, branches chunk/rcp-C<n>, base 57f3c9e8: C1 FiLM placement + skip switches; C2 pre/post CAC layers; C3 Pareto selection tool; C4 arm configs + submit; C5 base digests; C6 control eval tool; C7-C10 analyses.
- Nothing on the clusters yet.

## Next
1. Review each wave-1 slip + diff; merge into exp/t370-minicandi-recipe; push.
2. Wave 2: C11 kit RCP1, full suite as rcp_tests, one U-Net probe at real scale (gate G0).
3. Wave 3: screen arms C12/C13 (57 + 46 run-seeds), control probe C14, control arms code C15, EIC tools C16, Max panel C17.

## Open asks to the PI (non-blocking)
1. Training-only Avocado retrain on validation (~86 GPU-h) for the EIC comparison; until then compare vs the Average. Recommend yes.
2. Blind-row build (t184, ~22,700 CPU-h) to allow one blind pass; default: no blind pass this run (D26).
