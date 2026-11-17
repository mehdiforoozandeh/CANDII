---
title: handoff to the next session (peeling the onion)
kind: handoff note, written by the records item of the close-out
parent: "[[peeling the onion]]"
written: 2026-11-17T20:35Z
---

# Handoff to the next session

The peeling-the-onion run ends 2026-11-18T20:00Z (PI ruling 75). This note tells a new session where to start. All times are UTC. The run Log (`.orchestrate/plan.md` in the foreman's worktree, section `## Log`; snapshot `run_state/plan.md`) is the ground truth. Where this note and the Log disagree, the Log is right.

## Read first, in this order

1. `run_state/WHERE_WE_ARE.md` (status page) and the last 60 lines of the Log.
2. `closeout/ready queue for scaling.md`: every run that is built and not sent, in sending order, with command, cost, prerequisite and decision rule.
3. `closeout/closeout todo.md`: item statuses, PI rulings, open PI decisions.
4. `closeout/settled miniCANDI design.md` (item M1, written in the last 24 hours) and `onion training objective.md`.
5. `run_state/NIBI_NEARLINE_ARCHIVE.md`: what is on tape and how to get it back.
6. `long-horizon notepad.md`, rulings 50 to 75 and the compute tally.

## What is true (2026-11-17, about 20:30 UTC)

- Branch: umbrella `implementation/t152-peeling-the-onion`, draft PR mehdiforoozandeh/CANDII#61. Merging to main is the PI's.
- Fir: planned outage until about 2026-11-20. Login works; `/project` and kits could not be read (Permission denied at 20:15). 570 jobs wait in the queue.
- Nibi `/scratch`: 1,740 GiB used against a soft quota of 1,024 GiB, so writes are refused. `onion/work` (1,105 GiB) leaves after the QC fill and the 1 bp recount end (PI yes). Then about 389 GiB are free under the soft quota: at most about 6 prediction stores of 256-region scorings at once (55 GB each).
- Removed from Nibi scratch after a verified tape archive (23 of 23 tars): `t112_cf`, `onion/fastq`, `EIC_REPRO`. Kept: `t112_cf/pseudoreps`, `t112_cf/fastqarms/genome`, `EIC_REPRO/003` (genome reference cache). The staged FASTQs of the 722 priority-4 rows are gone; the stage step downloads them again.
- Code snapshots: kit CO8 (git d257503) is the only current Nibi kit. Pass `KIT_ID=CO8` or `K=/project/def-maxwl/mforooz/onion_kit/CO8` to every script whose default is an older kit (COA7, COA4, COA12, COA6, COMIX, COE4, COD2b, COD3, COE2f). Older kits are packed in `/project/def-maxwl/mforooz/onion_kit_archive`. Fir needs a fresh kit after the outage.
- Scoring cost of one run-seed on 256 regions: conv autoencoder 30.0, conv + attention 25.4, U-Net about 52 CPU-hours. The figure 23 is wrong.
- The 16-versus-256 check failed its rule: correction terms are scored on all 256 regions from now on. Every design choice that rests on 16-region correction scores is provisional until the rescoring (ready queue entry 5).
- The tables `eic_v1_2` (lab, antibody, donor, platform filled; train-only vocabularies) are the default. `eic_v1_1` must never be used. The `eic_v1_2` runs and eval sets exist on Fir only.

## What is running or queued

| Cluster | Jobs | What | Next step |
|---|---|---|---|
| Nibi | arrays 27658288 (65 tasks) and 27658295 (14 tasks) | QC fill, passes whole and pr1, kit CO8, writes `ONION/CO-A4/fill` | collect (ready queue entry 1) |
| Nibi | 27658461 | 1 bp recount probe, kit CO8 | read `[pointcheck] PASS` in `ONION/CO-A7/logs`, then the recount (entry 2) |
| Fir | 4535347 to 4535923 | 256-region rescoring of 62 run-seeds of 7 decisive arms (kit CO6) | they run when Fir returns; read `select_rule.py arm-summary --sub p256_sv` |
| Fir | 4345024 to 4345035 | 12 matched-compute trainings | score them when DONE (entry 8) |

## Never resubmit, never start again

Every job id below is in the Log. A job in this list is never submitted again on either cluster.

- Fir, queued: 4535347 to 4535923 (rescoring), 4345024 to 4345035 (matched-compute trainings; part of the final round 4344992 to 4345035).
- Nibi, live: 27658288, 27658295 (QC fill; the script refuses a second pass: a fill pass is never resubmitted), 27658461 (recount probe).
- Fir, done or failed (do not repeat): ablations 4437139 to 4438442, 4446021 to 4446138, 4447472 to 4448002, 4448258 to 4448639, 4449965 to 4450549, 4452636 to 4452696, 4455341 to 4455492, 4466127 to 4466187; foreign-covariate scoring 4468319 to 4468542; score-only after the file limit 4492369 to 4492696 (p16_sv) and 4492698 to 4492823 (p256_sv); E1 scoring 4451177 to 4451301 and check 4534658; final-round scoring 4385963 to 4389896; stage B 4327657 to 4327664; `eic_v1_2` baselines 4463699 to 4463702 and base recipes (ids in `runs_fir/submit_v1_2_base.log`); the `eic_v1_1` baselines 4455820 to 4455823. The untrained seed `abl_depth_input_input_depth_fixed_convae` seed 1 (Fir 4438068, TIMEOUT) stays untrained (PI ruling: 2 seeds accepted).
- Nibi, done (do not repeat): D3 preserve 27601589, verify 27601590 and 27603294 to 27603302, candidate 27601591; D2 prep 27601259; mix-and-match check 27601641; E2 extension 27595109 to 27595112.
- Never start a second nearline archive run, and never run `remove_archived.sh` again (`/project/def-maxwl/mforooz/ONION/nearline_archive_2026-11-17/`).
- The D3 reruns allow one 7-day rerun per row (rule B.5): the script refuses when `ONION/CO-D3/reruns_submitted.tsv` exists.
- The 10 outage reruns of the blind build run on one cluster only (Nibi, ruling 68).

## The first three actions

1. When the probe 27658461 prints `[pointcheck] PASS`, submit the recount: `ssh -o BatchMode=yes nibi 'KIT_ID=CO8 DRY_RUN=0 bash -s -- recount' < slurm/onion_data/closeout_a7_submit.sh` (PI: "You submit it"). Check that every one of the 412 views has a passed read-set check.
2. When the QC fill and the recount have ended: run the QC collect (`K=/project/def-maxwl/mforooz/onion_kit/CO8 MODE=collect DRY_RUN=0 bash .../submit_qc_on_input.sh` on Nibi). Then, under the PI's condition ("make absolutely sure before you remove something"), check the `onion/work` file count against the tar index, keep the NOT_APPLICABLE markers of rows 102, 418, 419, 425, 428, 431 recoverable (they are in the D3 work snapshot), remove `onion/work` (PI yes, ruling 73 (ii)), and test a write to `/scratch/mforooz/onion`.
3. Decide the route of the 256-region rescoring with the PI: wait for Fir (no new submission), or run it on Nibi with a throttle of about 6 live stores after the PI cancels the Fir jobs 4535347 to 4535923 (ready queue entry 5). When Fir returns, push a fresh Fir kit and send entries 6, 7 and 8 of the ready queue.

## Open PI decisions

The list is in `closeout/closeout todo.md`, section "Open PI decisions" (11 items), and in `closeout/ready queue for scaling.md`, section 4.

## Added 2026-11-17: the Fir copies of the 256-region rescoring

The PI confirmed the nine decisions of `closeout/PI decisions 2026-11-17.md` and agreed to cancel the Fir copies of the rescoring (jobs 4535347 to 4535923) after the Nibi scores are on disk. Cancelling is the PI's action; an agent must not run it. When all 62 `p256_sv/SCORED` files exist on Nibi under `/project/def-maxwl/mforooz/ONION/runs_fir`, tell the PI, and give this command for a Fir login node: `scancel $(seq -s, 4535347 4535923)`. First check with `squeue -u $USER -j <ids>` that the range holds only rescoring jobs; the Log line of 2026-11-17 that lists the range is the source.
