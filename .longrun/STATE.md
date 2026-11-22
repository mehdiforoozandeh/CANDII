# State

Rewritten, never appended. Cap: 5 KB.

## Goal now
Close the session with one trusted miniCANDI architecture (PI 2026-11-22, after the talk): the recipe must Pareto-dominate the three base models on imputation, correction and calibration on eic_v1_2 (D1); priority 2 is ChIP-control prediction (D18). Task t370, branch exp/t370-minicandi-recipe (PR 63), worktree .claude/worktrees/recipe.

## Where the recipe stands
- 256 regions (D43, final so far): pairs + Pearson-residual input U-Net, PASS, 3 seeds (imp 0.445, cor 0.774, cal 0.625). Runner-up pairs + average input (PASS).
- Pairs + head skip off (D41, 16 regions): copy gain 0.013 vs 0.205; correction 0.699 vs 0.754, inside the pairs range 0.058; NEAR.
- Talk done 2026-11-22. Deck https://claude.ai/artifact/Jhi5EjXAF5d1gRoTktaU97 (v3).

## In flight (Fir, kit RCP7)
- D44 full combo halves_mix_avg_pearson: train 5264674/80/87 + p16 scoring.
- D45 scaling: rcp_scale_p2x/p4x/w2x_unet, 3 seeds, train 5264693-5264764 + p16.
- D46 p256 confirm of rcp_halves_headskip_off_unet, 3 seeds (342 CPU-h projected).
- D48 control_eval on 27 ctl_holdout run-seeds -> runs_rcp/control_eval/<run>/seed<s>/; ids in runs_rcp/control_eval/SUBMITTED.tsv.
- D47 ensemble + recalibration: builder agent on chunk/t370-ens (worktree .orchestrate-wt/CANDII/t370-ens); slip pending.
- Avocado 5022624 still pending.
- Laptop watch byejoej5t (10 h): exits when the combo p16 or all 27 control evals land.

## Next
1. When p16 of D44/D45 lands: pareto_select; submit p256 confirm for D44 and any scaling arm that PASSes.
2. When control_eval lands: control_eval.py summary, gate G5 (plan C21) -> results/control_read.md.
3. Review the ensemble slip; read its scores against single seeds.
4. Final pick of the architecture (new decision); update GOALS R1-R4, R8; refresh the deck.

## Open asks to the PI
None.
