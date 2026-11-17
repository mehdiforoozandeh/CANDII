---
title: where we are (peeling the onion run)
kind: live status page, maintained by the foreman agent
parent: "[[peeling the onion]]"
updated: 2026-11-17T12:10Z
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

## State on 2026-11-17 12:10 UTC (close-out phase)

The close-out todo list is `../closeout/closeout todo.md`, with one item file per item. The four success criteria of the PI are: (1) the augmented EIC data with counterfactuals, (2) a well-defined training objective, (3) a well-defined and tested evaluation, (4) a settled and tested miniCANDI design, ready for scaling.

### Clusters

| Cluster | State | Consequence |
|---|---|---|
| Fir | planned outage from 2026-11-17 07:00 UTC to about 2026-11-20; login works, no job starts | 570 jobs wait in the queue, among them the 62 rescoring runs on 256 regions (jobs 4535347 to 4535923) |
| Nibi | `/scratch` refuses every write (4,324 GiB used, soft quota 1,024 GiB, grace expired) | no scoring and no build can run there until scratch is under the quota |

### Nibi archive to nearline (PI approved 2026-11-17)

Four scratch folders are copied to tape as 23 tar files (3,991 GiB), each compared byte for byte with scratch. Note and restore steps: `NIBI_NEARLINE_ARCHIVE.md` in this folder. At 11:59 UTC: 4 of 23 verified, 1,213 GiB written, 0 failed. It runs in tmux session `nlarchive` on login node l1.nibi. Never start a second run.

Nothing is removed from scratch yet. Conditions for removal: (a) 23 of 23 tars OK, (b) the PI's yes, (c) for `onion/work`: a `build_list.py pending` call over rows 102, 418, 419, 425, 428, 431 first. The items "QC metrics on the input side" and "resolution 1 to 125 bp" read the large tagAlign files in `onion/work`; after removal they need a tape recall.

### Close-out items

| Item | State |
|---|---|
| 16 ablations on three models | scored on 16 regions and read (`results/ablations_closeout/ABLATIONS_READ.md`) |
| 16-versus-256 region check | failed its pre-written rule: correction terms are scored on all 256 regions from now on; the 7 decisive arms wait in the Fir queue for rescoring (about 2,225 CPU-hours, corrected from 1,430) |
| Training objective document | written and merged: `../onion training objective.md`; open choices are marked in it |
| Signal p-value and peaks from the predicted NB | merged; 1 of 9 runs scored; PI ruled the box-spread formula; its model-side formula was sent back by review (uncertainty understated), fix in progress |
| Mix-and-match round (item A13) | specification, 24 provisional arms, native-depth scoring, gate on 256-region scores: merged; not submitted (about 5,390 CPU-hours plus 474 for the gate) |
| Minimum useful covariate set (A12) | tools, 51 configs, ranking tool: merged; not submitted (plans from 1,927 to 17,900 CPU-hours) |
| Learned cell-type embedding (A6) | configs, leak audit, zero-shot control: merged; not submitted (about 500 CPU-hours); it says nothing about unseen cell types |
| Second experiments and cross-lab pairs (D2) | add-on table with 71 second experiments built and checked on Nibi; merged; scoring stages not submitted |
| Data set update with DNase-seq (D3) | work-folder snapshot, candidate table (2,812 rows, DNase-seq correction-scorable), 10 row verifications passed; merged; 4 row rebuilds and the version build not submitted |
| Blind build (D4) | not submitted; PI ruled Nibi; needs writable Nibi scratch; the PI runs the submit command |
| Not started | QC metrics on input (A4), RNA-seq input (A5), resolution (A7), pointwise MLP (A10), replicate ladder (E4), settled design (M1), records (R1), close-out deck and report (R2) |

### Decisions that wait for the PI

1. Remove the four archived folders from Nibi scratch when all 23 tars are verified.
2. Budget of the mix-and-match round and of its gate.
3. Scope of the minimum-covariate-set search.
4. Lab effect in the cross-lab analysis: report lab together with donor (recommended), since the pure lab effect has too few pairs.
5. Drop the 17 unbuilt DNase-seq second-experiment rows (recommended).
6. Table version for the final model: the current one, or the candidate with DNase-seq (the 9 base runs then train again).
7. The corrected cost of the queued rescoring (2,225 CPU-hours).
8. Remove stale code snapshots on Nibi `/project`: the group is at 464K of 500K files, and 13 snapshots pushed on 2026-11-17 hold about 31,500 files. No more snapshots are pushed to Nibi until this is settled.
9. The 1 bp recount for the resolution item (about 18 CPU-hours, 1,408 files to Nibi `/project`) must run before `onion/work` leaves scratch; the PI submits it (the permission check refused the agent's probe job).

### Shared prerequisites when a cluster frees

- Build the 256-region reference scores of table version `eic_v1_2` and score its 9 base runs on 256 regions (about 320 CPU-hours). Every ablation on that version needs them.
- Push one fresh code snapshot before any submission; the snapshots pushed during the night are stale.
