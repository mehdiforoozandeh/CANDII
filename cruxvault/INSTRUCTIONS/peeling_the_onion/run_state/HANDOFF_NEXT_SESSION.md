---
title: handoff to the next session (peeling the onion)
kind: handoff note, written by the records item of the close-out
parent: "[[peeling the onion]]"
written: 2026-11-17T20:35Z
updated: 2026-11-18T17:00Z (records item, end of run)
---

# Handoff to the next session

The peeling-the-onion run ends 2026-11-18T20:00Z (PI ruling 75). This note tells a new session where to start. All times are UTC. The run Log (`.orchestrate/plan.md` in the foreman's worktree, section `## Log`; snapshot `run_state/plan.md`) is the ground truth. Where this note and the Log disagree, the Log is right.

The task of the next session is scaling (PI ruling of 2026-11-17 21:00Z). This run gives the information and the prototype for it.

## Read first, in this order

1. `run_state/WHERE_WE_ARE.md` (end-of-run status page) and the last 60 lines of the Log.
2. `closeout/ready queue for scaling.md`: every run that is built and not sent, in sending order, with command, cost, prerequisite and decision rule.
3. `closeout/settled miniCANDI design.md` and `onion training objective.md`.
4. `closeout/closeout todo.md` (item statuses, PI rulings, open PI decisions) and `closeout/PI decisions 2026-11-17.md`.
5. `run_state/NIBI_NEARLINE_ARCHIVE.md`: what is on tape and how to get it back.
6. `long-horizon notepad.md`, rulings 50 to 75 and the compute tally.

## The first three actions

1. **Read the scores on Nibi.** Count the scored run-seeds of the 256-region rescoring: `ssh -o BatchMode=yes nibi 'ls -d /project/def-maxwl/mforooz/ONION/runs_fir/abl_*/seed*/p256_sv/SCORED /project/def-maxwl/mforooz/ONION/runs_fir/rung4_views_base_only/seed*/p256_sv/SCORED | wc -l'`. The target is 62. If the laptop loops have stopped and some run-seeds are still held or timed out, release or rescue them by hand with the commands in `WHERE_WE_ARE.md`, section 2.
2. **Rerun the rule-application command on the full table.** First make the arm-summary table on a Nibi login node, in a shell that submits no jobs (never load `scipy-stack` in a shell that submits): `module load python/3.10.13 scipy-stack; K=/project/def-maxwl/mforooz/onion_kit/CO9; PYTHONPATH=$K/src python $K/tools/onion_eval/select_rule.py arm-summary --runs /project/def-maxwl/mforooz/ONION/runs_fir --sub p256_sv > /project/def-maxwl/mforooz/ONION/runs_fir/arm_summary_p256_sv_<stamp>.tsv`. Copy it to the laptop (`rsync -a -e 'ssh -o BatchMode=yes' nibi:<path> <scratch dir>/`). Then, from the root of a worktree of the umbrella branch: `export PYTHONPATH=$PWD/src; /Users/mforooz/miniforge3/envs/candii/bin/python tools/onion_eval/apply_rescoring_rules.py <table> --as-of <YYYY-MM-DDTHH:MMZ of the table>`. It writes `cruxvault/results/rescoring_p256/read_<stamp>/`, updates the design table, rebuilds the deck and the report, and runs both verify steps with `--strict`. Republish the deck to https://claude.ai/artifact/Kp1yqCgAUpCBDo19Q6oL9J, and record the new verdicts (rules 2.3, 2.4, 2.5) as crux findings; close the taskhub task "finish the 256-region rescoring".
3. **Tell the PI to cancel the Fir copies** of the rescoring after all 62 `SCORED` files exist on Nibi (PI, 2026-11-17 22:25Z: "your recommendation is ok"). The cancel is the PI's action; an agent must not run it. First check on a Fir login node that the range holds only rescoring jobs: `squeue -u $USER -h -o "%i %j" | awk '$1>=4535347 && $1<=4535923'`. Then give the PI: `scancel $(seq -s, 4535347 4535923)`.

## The ready queue, in order (details in `closeout/ready queue for scaling.md`)

| Entry | What | State on 2026-11-18 |
|---|---|---|
| 1 | QC-on-input collect | done (27678089) |
| 2 | 1 bp recount and tables | done; verify job 27692994 running (entry 22 if it does not end) |
| 3 | Replicate ladder: layers and report | done; two-sided form merged; the DNase-seq rerun is entry 20 |
| 4 | Second experiments: layers and baselines | done |
| 5 | 256-region rescoring of the 7 decisive arms | running on Nibi (first two actions above) |
| 6 | `eic_v1_2` 256-region references and its 9 base scorings | waits for Fir and a fresh Fir kit |
| 7 | Box-spread signal rescoring of the 9 base runs | waits for Fir, a fresh kit and Globus |
| 8 | Scoring of the 12 matched-compute runs | waits for the queued Fir trainings |
| 9 | Resolution arms, full design (PI decision 6) | next session, about 2,340 CPU-hours; needs a passed 27692994 or entry 22 |
| 10 | Mix-and-match gate, then round (PI decision 8) | needs entries 5 and 6 |
| 11 | QC-on-input arm (PI decision 7) | needs entry 6 and the side tables on Fir |
| 12 | Learned cell-type embedding | PI decision open (728.6 CPU-hours) |
| 13 | Second experiments: models | needs entry 6 |
| 14 | Minimum covariate set, one model with the removal curve (PI decision 9) | needs entry 10 |
| 15 | Data set with DNase-seq | PI table-version choice (next session) |
| 16 | Blind build, batch 1 | the PI runs the command |
| 17 to 19 | Final model, RNA-seq input, pointwise MLP | not built |
| 20 | DNase-seq layers rerun without the 15,201-read track | needs entry 21 |
| 21 | Fresh code kit before the next cluster ladder report | can go now |
| 22 | Resolution eval set and 1 bp verification | only if 27692994 does not end with a pass |

Nibi scoring time for every estimate: about 6.3 hours of wall time per run-seed (conv autoencoder median; conv + attention 5.8), and a 12-hour predict limit for the U-Net.

## Never resubmit, never start again

Every job id below is in the Log. A job in this list is never submitted again on either cluster.

- Nibi, 256-region rescoring: the 558 jobs of `/project/def-maxwl/mforooz/ONION/runs_fir/SUBMITTED.tsv` (from 27680544 on) and every rescue that the rescuer loop submitted (12 h predict limit, same job names).
- Nibi, close-out data jobs of 2026-11-17 and 18: QC fill 27658288, 27658295 (a fill pass is never resubmitted; the script refuses); QC collect 27678089; recount probes 27658461 (failed) and 27661198; recount 27662614 and its retry 27664315; 1 bp tables 27665563; variance 27665564 (cancelled by the PI) and 27693041; 1 bp verify 27692994.
- Nibi, ladder and second experiments: pair list 27661033, design 27661078, layers 27661089 (failed) and 27678107, report 27692829; second-experiment layers 27664409 (failed) and 27678103; baselines 27664415, 27664417, 27664418, 27664419, 27664422.
- Fir, queued: 4535347 to 4535923 (rescoring copies; the PI cancels them), 4345024 to 4345035 (matched-compute trainings; part of the final round 4344992 to 4345035).
- Fir, done or failed (do not repeat): ablations 4437139 to 4438442, 4446021 to 4446138, 4447472 to 4448002, 4448258 to 4448639, 4449965 to 4450549, 4452636 to 4452696, 4455341 to 4455492, 4466127 to 4466187; foreign-covariate scoring 4468319 to 4468542; score-only after the file limit 4492369 to 4492696 (p16_sv) and 4492698 to 4492823 (p256_sv); E1 scoring 4451177 to 4451301 and check 4534658; final-round scoring 4385963 to 4389896; stage B 4327657 to 4327664; `eic_v1_2` baselines 4463699 to 4463702 and base recipes (ids in `runs_fir/submit_v1_2_base.log`); the `eic_v1_1` baselines 4455820 to 4455823. The untrained seed `abl_depth_input_input_depth_fixed_convae` seed 1 (Fir 4438068, TIMEOUT) stays untrained (PI ruling: 2 seeds accepted).
- Nibi, done earlier (do not repeat): D3 preserve 27601589, verify 27601590 and 27603294 to 27603302, candidate 27601591; D2 prep 27601259; mix-and-match check 27601641; E2 extension 27595109 to 27595112.
- Never start a second nearline archive run, and never run `remove_archived.sh` again (`/project/def-maxwl/mforooz/ONION/nearline_archive_2026-11-17/`).
- The D3 reruns allow one 7-day rerun per row (rule B.5): the script refuses when `ONION/CO-D3/reruns_submitted.tsv` exists.
- The 10 outage reruns of the blind build run on one cluster only (Nibi, ruling 68).

## Other open work of this run (taskhub)

- Push a fresh kit, then rerun the DNase-seq layers without ENCSR000EKG (ready queue entries 20 and 21).
- Verify the 1 bp tables if 27692994 does not end (entry 22).
- Remove the worktrees and chunk branches of this run after the umbrella branch merges (look for uncommitted work first).
- An open question in the tree: whether duplicate removal biases the read-half layer of the earlier 25 bp variance ladders (read halves anti-correlate at 1 bp).

## Open PI decisions

`closeout/closeout todo.md`, section "Open PI decisions" (update of 2026-11-18): the cell-type embedding budget, matched compute, the Nibi scratch quota figure, the architecture for scaling, the data version of the final model, and the cancel of the Fir rescoring copies.
