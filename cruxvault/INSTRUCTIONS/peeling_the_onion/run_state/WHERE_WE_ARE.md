---
title: where we are (peeling the onion run)
kind: live status page, maintained by the foreman agent
parent: "[[peeling the onion]]"
updated: 2026-11-14T10:21Z
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

## State on 2026-11-14 10:25

**Done**
- Dataset `eic_v1` frozen (249 of 267 training tracks, 42 of 45 validation, 37 of 40 controls).
- Evaluation set frozen (256 regions); screening set `evalset_p16.json` with 16 pair regions derived (sha a82cdfd1).
- Real-data variance layers measured (sampling layer missing; rerun in work).
- AlphaGenome 2: 8,960 predictions cached, training-track audit done, scorer matches Max's script.
- Rung 1: 24 runs trained, unscored. Budget curve: 42 runs trained, unscored.
- Fast scorer merged on the umbrella: prediction store, predict command, job scripts, score comparison. The metric phase is not merged.
- Nibi file limit (500K files on /project) hit at 09:40 and fixed at about 10:10: 68 unused code snapshots archived to `/scratch/mforooz/onion/kit_archive` and deleted (PI approved). Now 372K of 500K.
- Globus: two transfers Nibi to Fir started at about 09:50 (scratch `measurements`, `aux`; project `aux`, `eval`, `genome`, `region`, `runs`, `runs_region`, `tables`).

**In work (agents)**
- Metric phase of the scorer (`score.py --phase metrics`).
- Restart of trainings after the outage: all 42 queued or running (8 resubmitted as 27269108-27269121, 34 released) (rungs 2 to 6, rung-4 `scrambled_covariates`, "evaluation regions excluded", 7 selection arms).
- Two new input arms on rung 1 (`average_input`, `two_step_native`): 6 held trainings, first two released and checked first. Code on `chunk/onion-A1k`, not merged.
- Baselines: 8 scorings run (16 regions: 27269893, 27269898, 27269901, 27269908; 256 regions: 27270400, 27270401, 27270402, 27269910).
- Variance-layer rerun with pseudoreplicate pairs.

**Next steps, in order**
1. Collect the three restart slips; log the new job ids.
2. Merge the metric phase and the input-arm code after review; push one code snapshot.
3. Time ONE rung-1 predict job, then release the predict jobs (hypothesis arms first), then the metric jobs.
4. Ship 1: layers report, baselines, AlphaGenome 2 board, deck, vault recorder, snapshot.
5. Ship 2: arm reports, deck, vault recorder, snapshot.
6. After Globus reports success: verify counts, sizes and hashes on Fir.
7. Small change: new code snapshots default to `/scratch` (PI directive, 2026-11-14).

**Open with the PI**
- None.
