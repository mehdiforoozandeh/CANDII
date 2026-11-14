---
title: where we are (peeling the onion run)
kind: live status page, maintained by the foreman agent
parent: "[[peeling the onion]]"
updated: 2026-11-14T21:26Z
---

# Where we are

The foreman agent rewrites this page at each milestone. Read this page first after a context reset. All times are UTC.

## Read in this order after a reset

1. This page.
2. `live_state.md` in this folder: every background agent and every cluster job, with ids.
3. `plan.md` in this folder: PI rulings, Amendments 2b to 10, and the `## Log` at the end (the run memory).
4. `../long-horizon notepad.md`: rulings 34 to 46, key facts, PI directives.
5. `../onion paper.md`: the paper-order skeleton that the shipments fill.

Working copies of items 2 and 3 live in the worktree at `.claude/worktrees/onion/.orchestrate/` (git-ignored). The script `.orchestrate/snapshot_state.sh` copies them here and pushes.

## Fixed facts

- Umbrella branch `implementation/t152-peeling-the-onion`, draft PR mehdiforoozandeh/CANDII#61. Merging to main is the PI's.
- Worktree `W` = `<repo>/.claude/worktrees/onion`. Chunk worktrees are under `<repo parent>/.orchestrate-wt/CANDII/onion-<id>`.
- Cluster root on Nibi: `/project/def-maxwl/mforooz/ONION`; scratch `/scratch/mforooz/onion`; code snapshots `/project/def-maxwl/mforooz/onion_kit/<id>`.
- Never resubmit a job that the Log lists. Cancelling jobs is the PI's. Deletes need the PI's yes.

## Schedule (ruling 43 to 46, Amendment 10)

| Shipment | Content | Due |
|---|---|---|
| Ship 1 | Methods (data, evaluation) and model-free Results; one deck | 2026-11-15 06:00 (deck cut 03:00) |
| Ship 2 | Model Results on screening scores and a first Discussion; one deck | 2026-11-15 18:00 (hard limit 2026-11-16 08:00) |
| Iteration 2 | 16 versus 256 region check, budget curve, dataset update, CPU work on Fir | 2026-11-17 12:00 |

## State on 2026-11-14 12:49

**Done**
- Dataset `eic_v1` frozen (249 of 267 training tracks, 42 of 45 validation, 37 of 40 controls).
- Evaluation set frozen (256 regions); screening set `evalset_p16.json` with 16 pair regions (sha a82cdfd1).
- Real-data variance layers with the sampling layer (results/t305/LAYERS_REAL.md); recorded in the vault (f51 to f54, q31).
- AlphaGenome 2: 8,960 predictions cached, training-track audit done, scorer matches Max's script.
- All 8 baselines SCORED: `eval/eic_v1/baselines/` (256 regions) and `eval/eic_v1/baselines_p16/` (screening).
- Trainings DONE: rung 1 (24 runs, 8 arms), budget curve (42 runs), rungs 2 to 6 base x3, rung-4 `scrambled_covariates` x3, `rg_no_eval` x3, 7 selection arms x3 (`rg_variance_e50_w` 2 of 3 at 12:47), `average_input` seed 0.
- Fast scorer complete and merged (umbrella db8fc0c): prediction store, predict command, job scripts, metric phase (0 differences against the old path on 3 models), score comparison.
- Nibi file limit fixed at about 10:10 (PI approved the removal of 68 archived code snapshots); rule: file-heavy content goes to /scratch.
- Globus: two transfers Nibi to Fir started at about 09:50; not verified on Fir yet.
- Status deck 2026-11-14: `cruxvault/presentations/onion-status-2026-11-14/`, artifact https://claude.ai/artifact/FwKjytRr1UuhT3xhTpWDrL (private).

**In work** (state at 2026-11-14T21:26Z)
- First screening scores exist: rung-1 base, scrambled_covariates, dna_off (8 of 9 scored; results/metrics_screening/first_scores.md). Baselines report done (results/t307/BASELINES.md; vault f55 to f60, q32).
- Builder: metric jobs for every other store, then the ladder table and the rung-1 arm table (results/metrics_screening/scores_ladder.md, scores_rung1_arms.md).
- Predict jobs released at 21:25: rung 5 seeds 1 and 2, the other 5 rung-1 arms, the 7 selection arms. All ladder stores of rungs 2, 3, 4, 6 and rung 5 seed 0 are complete.
- SHIP 1 deck done: presentations/onion-ship1-2026-11-15, artifact https://claude.ai/artifact/4RSUdsFwJC3nYcyHb9s2bU (sent to the PI early). AlphaGenome 2 slide shows the audit only.
- Investigator: oracle floor for the correction metric (0.5 E|half1 - half2|) and the eta split, after the PI question; results/gap_closed_floor/GAP_FLOOR.md.
- Builder: fragment length for single-end tracks (frag_len is empty for them in the frozen tables, so signal-space metrics are undefined for those tracks; fix = estimated fragment length from the quality-control fill, scoring side only; results/fraglen_single_end/). The AlphaGenome 2 board waits for it.
- GPU trainings still pending: average_input seed 1 (27273016), two_step_native seed 0 (27272462). Then seeds per results/arms_d10_d15/SUBMITTED.md (kit A1k2).
- rg_variance_e50_w seed 2 training timed out (27266404): resubmit with a longer limit.

**Next steps, in order**
1. Ship 1 at 06:00: deck link to the PI, vault recorder (close t306, baselines and layers are recorded), snapshot.
2. Collect the metric builder's tables; arm reports; rung-4 check of 16 against 256 pair regions (stores of rung4_base and rung4_scrambled_covariates are kept).
3. After the fragment-length review: rescore baselines with the signal dump, rerun the AlphaGenome 2 chain.
4. Ship 2 (due 2026-11-15 18:00; hard limit 2026-11-16 08:00): deck, vault recorder, snapshot.
5. Verify the Globus copies on Fir; move new code snapshots to /scratch.
6. RULE: never open a question dialog to the PI while held jobs or timed releases are owed (a dialog blocked the foreman for 7 hours on 2026-11-14).

**Open with the PI**
- Decision on the scale of the correction metric "gap closed" (options A, B, C; vault question q32).
