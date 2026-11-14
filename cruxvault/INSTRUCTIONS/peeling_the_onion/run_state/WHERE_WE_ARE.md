---
title: where we are (peeling the onion run)
kind: live status page, maintained by the foreman agent
parent: "[[peeling the onion]]"
updated: 2026-11-14T12:49Z
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

**In work**
- Rung-1 predict jobs on GPU slices (results/forwards_rung1/): group (1) = base, scrambled_covariates, dna_off (72 jobs, released 12:29); groups (3) 27278967-27279070 and (4) 27279077-27279130 HELD. Foreman releases (3) when (1) has no pending job, then (4).
- Builder: predict jobs of the CPU runs (rungs 2 to 6, rung-4 arms, `rg_no_eval`, selection arms), 4 shards on CPU, results/forwards_cpu_runs/.
- Builder: metric jobs, one timed first, results/metrics_screening/ (first screening scores).
- Builder: five small review fixes of the scorer (checkpoint hash in the store identity, and others).
- GPU trainings pending: `average_input` seed 1 (27273016), `two_step_native` seed 0 (27272462). Foreman: check speed and peak memory, then submit the other seeds (lines in results/arms_d10_d15/SUBMITTED.md, kit A1k2).

**Next steps, in order**
1. Ship 1 (due 2026-11-15 06:00, deck cut 03:00): AlphaGenome 2 board against the baselines, baseline table, layers report, deck, vault recorder, snapshot.
2. Release rung-1 predict groups (3) and (4); collect metric jobs; arm reports.
3. Rung-4 check of 16 against 256 pair regions (predict at 256, nice 300), after the ladder predict jobs.
4. Ship 2 (due 2026-11-15 18:00): arm reports, deck, vault recorder, snapshot.
5. Verify the Globus copies on Fir; move new code snapshots to /scratch.
6. Vault housekeeping: point findings f34, f35 at LAYERS_REAL_first_run.md; complete the manifest of results/t305.

**Open with the PI**
- None.
