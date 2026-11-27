# State

Rewritten in full each time. Cap: 5 KB. Last rewrite 2026-11-27 ~06:20 UTC.

## Goal now
CLOSED OUT 2026-11-27. The run delivered one trusted miniCANDI architecture (D55): it Pareto-dominates the three base models at 256 regions (imputation 0.503, correction 0.818, calibration 0.717 vs bars 0.445, 0.675, 0.646). Task t370, branch exp/t370-minicandi-recipe (PR 63, merge is the PI's), worktree .claude/worktrees/recipe.

## Handoff
cruxvault/INSTRUCTIONS/t370_recipe/run_state/HANDOFF_CLOSEOUT.md. The p-value session owns the open work (D61).

## In flight (owned by the p-value session)
- Fir Avocado chain: shared 6291945 done; genome 6291946 0-13 done, 14-22 pending (throttle reset 2026-11-27); predict 6291947, bigwig 6291948 pending. Never resubmit.
- Nibi blind build batch 1 (t184): all 74 build tasks failed (Cromwell crashed); signal passes 28737212/39 running since 05:28 UTC as the retest (D60). Never resubmit.

## Next (if this session is resumed)
1. Only on the foreman's request: answer questions about the handoff.
2. After the PI merges PR 63: `crux task done t370` with the merge link.

## Open asks to the PI
- Merge PR 63.
- Read-start retrain (1,350-2,250 CPU-h): waits.
- Known repo defect, not mine: tests/test_results_folders.py fails on a tracked file missing on disk (cruxvault/results/t371/compare/_x).
