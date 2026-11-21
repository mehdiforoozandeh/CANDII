# State

Rewritten, never appended. Cap: 5 KB.

## Goal now
Find one miniCANDI recipe that Pareto-dominates the three base models on imputation, correction and calibration on eic_v1_2 (D1); priority 2 is ChIP-control prediction (D18). Task t370, branch exp/t370-minicandi-recipe (PR 63), worktree .claude/worktrees/recipe.

## Horizon
Time horizon: 2026-12-01. Recipe freeze 2026-11-29 (D27). First measured result due: 2026-11-24 (screen, 16 pair regions). Budget: 6,000 CPU-h checkpoint (D20). Spent: probe 39.1 CPU-h, tests 24 CPU-h. Screen budget recomputed x1.50 = 5,209 CPU-h (D30, probe.md).

## In flight (Fir, kit RCP2 = RCP1 + C15-C17 + C29)
- Gate G0 done: terms reproduce base exactly, train cost 2.4x (run_state/probe.md).
- Screen s1a (66 run-seeds: skip, placement, no-cov, halves) and s1b (45: curve pre/post, mix): ids in run_state/jobs_C12.tsv, jobs_C13.tsv (from runs_rcp/SUBMITTED.tsv). Native-depth scoring of mix_fixdepth (models 3,4) chained.
- CNN cost seed rcp_cac_cnn_unet seed 0 (in jobs_C13.tsv) -> gate G0b.
- Control probe rcp_ctleval_eic_v1_2_rung4_base_heldout_s0 = 4889711; tests rerun rcp_tests = 4888524 (12 h). jobs_C14.tsv.
- 704 rcp_ jobs queued at 2026-11-21 ~09:20 UTC.

## Next
1. Read 4888524 tests; cancel arms whose switch fails (cancel-own.sh).
2. G0b: CNN seed training log at >=200 steps -> TRAIN_TIME, then C27 (17 CNN seeds); 4889711 cost -> C28 (8 control evals), C18 control arms, C19 EIC probe.
3. As screen seeds land: pareto_select on p16_sv; first result 2026-11-24.

## Open asks to the PI (non-blocking)
1. Training-only Avocado retrain on validation (~86 GPU-h) for the EIC comparison; until then compare vs the Average. Recommend yes.
2. Blind-row build (t184, ~22,700 CPU-h) to allow one blind pass; default: no blind pass this run (D26).
3. FYI: added arm D29 (read-disjoint training pairs), ~150 CPU-h, from the three analyses.
