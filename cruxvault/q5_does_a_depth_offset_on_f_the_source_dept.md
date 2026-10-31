---
id: q5
type: question
schema: 2
title: Does a depth offset on f (the source depth removed before f, the target depth added after) change how well a covariate-conditioned g → f translates a track's raw counts across a covariate change?
parent: q1
status: open
stale: false
created: "2026-10-08T16:58:00"
updated: "2026-10-08T16:58:00"
---

# q5 — Does a depth offset on f (the source depth removed before f, the target depth added after) change how well a covariate-conditioned g → f translates a track's raw counts across a covariate change?

Parent:: [[q1_do_the_recorded_experimental_covariates_]]

## ELI5

A model predicts a track's read counts under other lab settings. Here it first divides out the old sequencing depth and multiplies in the new one. Does that help, and does a pure depth change now come out right?

## TL;DR

CAC: a depth offset on f. The offset helps only design A, mostly in scale error. No design follows the depth law. The settings still improve the mean beyond plain d′/d scaling. The no-covariates twin of design D nearly follows the law while the real D misses it, so g's use of the covariates causes the miss; [[q7_why_do_the_covariate_conditioned_g_f_des|the depth-law-miss question]] takes this up.

## Question

Does a depth offset on f (the source depth removed before f, the target depth added after) change how well a covariate-conditioned g → f translates a track's raw counts across a covariate change?

## Motivation

In [[q4_how_much_capacity_does_a_covariate_condi|the capacity grid]], no design scaled its counts by d′/d on depth-only pairs (the depth law), and row-1 A had a large DNase scale error. An offset makes "do nothing" mean "scale by d′/d". [[q6_how_closely_does_a_map_h_reproduce_the_l|The map-h question]] takes the grid's other failure, the −log10 p blow-ups.

The idea is the PI's (2026-10-08): "Put the depth offset on both ends... The depth law then holds by construction, and it fails today in every design." The PI also set the constraint: "keep the number of inputs and outputs and the operations done in this experiment as minimal as possible... no confounding design". g reads C and C′; f reads only X and θ. Every fork was walked one at a time with the PI and recorded in `plan/ETA_OFFSET_AND_H_DESIGN.md`. The PI kept depth in g's input, knowing g could still bend the law.

Expected before the run (design doc, 2026-10-08): the offset adds no capacity when g reads both depths, and only moves g's "do nothing" output from "copy the counts" to "scale by d′/d".

## Protocol

Exploratory (PI 2026-10-08): readings only — no pass/fail bars, no ticks, no verdicts. The design, fork by fork, is `plan/ETA_OFFSET_AND_H_DESIGN.md` on the branch `exp/t118-counterfactual-f`.

## Planned Intervention

Part 1 of [[t125_build_and_run_the_depth_offset_coun|the depth-offset run]]. f reads eta_in = log(1 + X) − log d (d in millions of reads); the target is NB with log μ′ = log d′ + eta_out; g still reads both depths. The grid repeats the capacity grid: designs A–D and A2–D2, one g per track (7) or across tracks, real and no-covariates twin (the twin keeps the offset), 3 seeds. Each spread now sees what its mean sees; an extra arm, Aold, keeps row-1 A's old spread. 432 runs, scored on chr19 + chr21.

## Answer so far

Complete: 432 of 432 runs. "±" is seed wobble; a gain clears above 2 × the larger wobble. CRPS is on counts, as total (oracle-scaled + scale error).

- Against the capacity grid, 6 of 54 CRPS cells clear, all A-type. Per-track A, DNase: 1.247 ± 0.016 (1.109 + 0.138) against 3.06 ± 0.113 (1.36 + 1.70); gain 1.813, bar 0.227. The offset alone (Aold) gives 0.927. B–D2 clear 0 of 36.
- Against the twin, real MSE on log(1 + mean) is lower in 54 of 54 cells, all clear (B per track, DNase: 0.194 ± 0.003 against 0.365 ± 0.013).
- Depth law (worst |predicted/true ratio − 1|, bar 0.10): 0 of 72 cells within. Trained depth pairs read 0.04–0.44. Real D reads 0.869–1.449; D and D2 twins 0.054–0.135.
- 24 of 96,768 records have CRPS > 20, all in the across-track A2 twin.
- Caveats: training devices are mixed (GPU slice and CPU); the capacity grid has no MSE.

## Run Links

- Nibi (not Fir), part 1 runs: `/project/def-maxwl/mforooz/t125/eta/runs/` (432 run dirs; checkpoints and logs stay there).
- Aggregate: `/project/def-maxwl/mforooz/t125/eta/agg/` (job 24224060). Summary: `/project/def-maxwl/mforooz/t125/summary_part1/` (job 24237469). Kit K10, GIT_SHA 7e0f0f2fabfc28b00c40bdc44c2886483d2e8bbd.
- Capacity-grid reference: `/project/def-maxwl/mforooz/t125/old_grid/results_summary.tsv` (md5 f0e1ef1578468dcc5966bee4a2181877).
- Job record: `plan/T125_STATUS.md` on `exp/t118-counterfactual-f`.

## Artifacts

- [Part-1 digest](results/t125/part1_digest.md)
- [Grid summary, part 1](results/t125/GRID_SUMMARY_part1.md) and [parts 1 and 2](results/t125/GRID_SUMMARY.md)
- [Full grid](results/t125/part1/grid.md), [depth-law table](results/t125/depth_law.tsv), [results summary](results/t125/results_summary_eta.tsv), [training device per run](results/t125/part1/run_device.tsv)
- [Run paths](results/t125/FIR_PATH.txt)
- [Deck](presentations/t125-eta-h/index.html) — published https://claude.ai/artifact/AfLpe1acvSC9i43jg28AqS; vault copy [DECK.md](presentations/t125-eta-h/DECK.md)

<!-- crux:ledger:start -->
**0 children** · open 0, leaning 0, settled 0, parked 0 · 1 finding(s) on this question

- `f17` finding (strong) [[f17_large_clean_margins_on_the_432_runs_and_|Large, clean margins on the 432 runs, and the PI's own reading on 2026-10-19…]]
<!-- crux:ledger:end -->
