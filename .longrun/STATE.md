# State

Rewritten, never appended. Cap: 5 KB.

## Goal now
Find one miniCANDI recipe that Pareto-dominates the three base models on imputation, correction and calibration on eic_v1_2 (D1); priority 2 is ChIP-control prediction (D18). Task t370, branch exp/t370-minicandi-recipe (PR 63), worktree .claude/worktrees/recipe.

## Horizon
Deck due 10:30 Vancouver time 2026-11-22 (PI, 05:26 local). The laptop clock (`date -u`) reads 12:26 at 05:26 local, so the deadline is laptop-UTC 17:30. Talk at 11:00 local. Budget checkpoint 6,000 CPU-h (D20).

## Where the recipe stands
- 256 regions (D43, final): recipe = halves + Pearson-residual input U-Net (PASS, 3 seeds; imp 0.445, cor 0.774, cal 0.625). Runner-up halves + average-signal input (PASS). Halves U-Net NEAR.
- D41 done (16 regions, 3 seeds): pairs + head skip off: copy gain 0.013 vs 0.205; imp 0.4295 vs 0.4263; cor 0.6985 vs 0.7535 (pairs range 0.058); NEAR. Not confirmed at 256 regions.
- EIC validation: base U-Net beats Average on 7 of 9 mean measures; loses gwcorr and mseenh (data/table_unet_vs_average.tsv).
- Head skip alone carries the copy: copy gain U-Net 0.67, head skip off 0.06, halves 0.20 (data/copy_gain_p16_sv.tsv in the deck folder).

## Deck
cruxvault/presentations/t370-minicandi-recipe/ (prezit paper arc, 18 slides). Artifact https://claude.ai/artifact/Jhi5EjXAF5d1gRoTktaU97. Verify and lint green. Refresh: ./refresh_data.sh; ./assemble.sh; python3 build_deck.py; crux deck --verify/--lint (from cruxvault/); republish the same file path.

## In flight (Fir)
- Avocado 5022624 pending; misses the deadline.

## Next
1. On each watch result: refresh the deck, verify, lint, republish; D43 is final.
2. After the talk: D42 scaling test; ensemble + recalibration question to the PI.

## Open asks to the PI
(i) Ensemble + NB dispersion recalibration from existing predictions (asked 2026-11-22, unanswered; not started).
