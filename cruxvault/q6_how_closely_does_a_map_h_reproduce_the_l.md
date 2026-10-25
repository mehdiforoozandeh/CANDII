---
id: q6
type: question
schema: 2
title: How closely does a map h reproduce the −log10 p track from a predicted count distribution and its matched control, as h's complexity and inputs grow?
parent: q1
status: open
stale: false
created: "2026-10-08T16:58:00"
updated: "2026-10-08T16:58:00"
---

# q6 — How closely does a map h reproduce the −log10 p track from a predicted count distribution and its matched control, as h's complexity and inputs grow?

Parent:: [[q1_do_the_recorded_experimental_covariates_]]

## ELI5

A peak caller turns read counts into a significance score with a fixed recipe. Can we copy that step, and does the copy stay stable on predicted counts?

## TL;DR

On the true counts, the caller's fragment-wide formula nearly matches the best learned map; formula plus a learned correction is best. Push-only h never blows up in the chain, but there every h trails the earlier direct p model on designs D and D2, and training h on f's predictions does not repair f.

## Question

How closely does a map h reproduce the −log10 p track from a predicted count distribution and its matched control, as h's complexity and inputs grow?

## Motivation

In [[q4_how_much_capacity_does_a_covariate_condi|the capacity grid]], f predicted −log10 p (P) directly, and some pairs blew up when a learned log-normal spread drifted. Here h maps f's NB over counts, plus the control, to P. [[q5_does_a_depth_offset_on_f_the_source_dept|The depth-offset question]] supplies f.

## Protocol

Exploratory (PI 2026-10-08): readings only — no pass/fail bars, no ticks, no verdicts. The design, fork by fork, is `plan/ETA_OFFSET_AND_H_DESIGN.md` on the branch `exp/t118-counterfactual-f`.

## Planned Intervention

Part 2 of [[t125_build_and_run_the_depth_offset_coun|the depth-offset run]]. h maps every count k, weighted by NB(k), so mean and CRPS are exact. Eight versions: the caller's Poisson-tail formula (bin-only, fragment-wide); affine and MLP h at rung 1 (bin count, same-bin control) and rung 5 (all inputs); formula + affine or MLP correction. Each learned h runs push-only or with its own Gaussian noise, trained on the true counts X′ (3 seeds) or on f's NB (1 seed). The chain uses track C19M16 (narrow), f seed 0, one f per design.

## Answer so far

Final at 5 of 8 designs (C, D, A2, C2, D2); the PI stopped the run on 2026-10-19 (chain A finished unaggregated; B, B2 unfinished). Log CRPS is CRPS on log(1 + P), as total (oracle-scaled + scale error); a gain clears above 2 × the larger seed wobble.

- True X′, narrow: bin formula 0.1177, fragment formula 0.0865; affine rung 1 0.1791 ± 0.0010 (0.1367 + 0.0423); MLP rung 5 0.0836 ± 0.0004; formula + MLP 0.0798 ± 0.0013.
- Repair: the true-X′ push h beats the h trained on f's NB in 30 of 30 rows, clear (D2, MLP rung 5: 0.1172 ± 0.0004 (0.1145 + 0.0027) against 0.1352 (0.1302 + 0.0050)).
- Chain on D and D2, CRPS on P: every h reads 0.178–0.276 (lowest: the bin formula) against the capacity grid's 0.143 ± 0.0016; the gap is oracle-scaled. Push and formula h have 0 records with CRPS > 20; the capacity grid exploded on A2 and C2.
- Own noise lowers log CRPS in 78 of 78 rows (76 clear) but MSE in only 5 (none clear).
- Caveats: one track, one f seed; D2 mixture error 9.7e-4 (bar 1e-3); 22 of 96 NB-trained fits hit the step cap.

## Run Links

- Nibi (not Fir), h runs: `/project/def-maxwl/mforooz/t125/h/{runs_true,runs_fnb,alone,chain}/` (checkpoints and logs stay there).
- Aggregate, 5 of 8 designs: `/project/def-maxwl/mforooz/t125/h/agg_5of8/` (job 24312235). Summary: `/project/def-maxwl/mforooz/t125/summary_5of8/GRID_SUMMARY.md`. Tests: job 24312234. Kit K13, GIT_SHA ec77f692e043126e1485e761eabe571b6c741058.
- Cancelled on the PI stop (2026-10-19): chain B/B2 resubmission 24608447 and the 8-of-8 aggregate 24608467.
- Job record: `plan/T125_STATUS.md` on `exp/t118-counterfactual-f`.

## Artifacts

- [Part-2 digest](results/t125/part2_digest.md)
- [Grid summary, parts 1 and 2](results/t125/GRID_SUMMARY.md)
- [h grid](results/t125/part2/grid_h.md), [h alone](results/t125/h_alone_summary.tsv), [h in the chain](results/t125/h_chain_summary.tsv), [repair](results/t125/h_repair.tsv), [chain against the capacity grid](results/t125/h_vs_oldgrid.tsv)
- [Run paths](results/t125/FIR_PATH.txt)
- [Deck](presentations/t125-eta-h/index.html) — published https://claude.ai/artifact/AfLpe1acvSC9i43jg28AqS; vault copy [DECK.md](presentations/t125-eta-h/DECK.md)

<!-- crux:ledger:start -->
_(no children yet)_
<!-- crux:ledger:end -->
