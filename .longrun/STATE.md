# State

Rewritten, never appended. Cap: 5 KB.

## Goal now
Find one miniCANDI recipe that Pareto-dominates the three base models on imputation, correction and calibration on eic_v1_2 (D1); priority 2 is ChIP-control prediction (D18). Task t370, branch exp/t370-minicandi-recipe (PR 63), worktree .claude/worktrees/recipe.

## Horizon
Deck due 10:30 Vancouver time 2026-11-22 (PI, 05:26 local). The laptop clock (`date -u`) reads 12:26 at 05:26 local, so the deadline is laptop-UTC 17:30. Talk at 11:00 local. Budget checkpoint 6,000 CPU-h (D20).

## Where the recipe stands
- 256 regions (D43): PASS = halves + Pearson-residual input U-Net (2 seeds), halves + average-signal input U-Net (3 seeds); halves U-Net NEAR. Final pick waits on Pearson seed 0 at 256 regions.
- Head skip alone carries the copy: copy gain U-Net 0.67, head skip off 0.06, halves 0.20 (data/copy_gain_p16_sv.tsv in the deck folder).

## Deck
cruxvault/presentations/t370-minicandi-recipe/ (prezit paper arc, 18 slides). Artifact https://claude.ai/artifact/Jhi5EjXAF5d1gRoTktaU97. Verify and lint green. Refresh: ./refresh_data.sh; ./assemble.sh; python3 build_deck.py; crux deck --verify/--lint (from cruxvault/); republish the same file path.

## In flight (Fir)
- Combo 256-region confirms: 8 of 9 scored (Pearson seed 0 pending). Watch bx8rkzhqf.
- D41 halves_headskip_off: train 5195200, 5195206, 5195212 + p16 scoring. Watch b904xdk65 (also the EIC table).
- EIC U-Net scoring 5195226 -> runs_rcp/eic_val/table_unet_vs_average.tsv.
- Avocado 5022624 pending; misses the deadline.

## Next
1. On each watch result: refresh the deck, verify, lint, republish; rerun the 256-region selection and update D43 with the final pick.
2. After the talk: D42 scaling test; ensemble + recalibration question to the PI.

## Open asks to the PI
(i) Ensemble + NB dispersion recalibration from existing predictions (asked 2026-11-22, unanswered; not started).
