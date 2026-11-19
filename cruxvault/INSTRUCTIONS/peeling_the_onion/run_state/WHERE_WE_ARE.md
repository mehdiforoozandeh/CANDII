---
title: where we are (peeling the onion run)
kind: live status page, maintained by the foreman agent
parent: "[[peeling the onion]]"
updated: 2026-11-18T17:00Z (end-of-run status, records item; the earlier page is archive/WHERE_WE_ARE_2026-11-17T1210Z.md)
---

# Where we are

The foreman agent rewrites this page at each milestone. Read this page first after a context reset. All times are UTC.

This is the end-of-run status page. The run ends 2026-11-18T20:00Z (PI ruling of 2026-11-17 20:15Z). The lines marked **FINAL (18:15 UTC):** were refreshed by the foreman at 2026-11-18 18:15Z. No new score landed between the read of 15:54Z and 18:12Z, so that read is the final read of this run.

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
- Cluster root on both clusters: `/project/def-maxwl/mforooz/ONION`; scratch `/scratch/mforooz/onion`; code snapshots (kits) `/project/def-maxwl/mforooz/onion_kit/<id>`. The current Nibi kit is CO9 (git 47965e0).
- Never resubmit a job that the Log lists. Cancelling jobs is the PI's. Deletes need the PI's yes.

## 1. The four success criteria (PI, 2026-11-16)

| Criterion | State | Evidence |
|---|---|---|
| 1. The augmented EIC data with counterfactual views | partial | Built: tables `eic_v1_2` with designed views and read halves, the lab, antibody and donor fill, the add-on of 71 second experiments (`eic_v1_2x`), the QC side tables, the 1 bp region store (412 views). Not built: the 722 blind rows and the table version with DNase-seq in the correction term (`../closeout/closeout todo.md`, items D3 and D4). |
| 2. A well-defined training objective | delivered | `../onion training objective.md` (item O1). Its open weight choices are marked and wait for the mix-and-match round. |
| 3. A well-defined and tested evaluation | partial | Metrics, scorer, references and the 16-versus-256 check are built and tested; the replicate ladder is in the two-sided form. **FINAL (18:15 UTC):** the 256-region rescoring has 34 of 62 run-seeds scored at 19:42Z (30 from 15:55Z to 18:12Z); the other 32 are in their second pass or in the queue on Nibi (190 prediction jobs running, 19 pending, 32 final-step jobs pending at 18:12Z). |
| 4. A settled and tested miniCANDI design, ready for scaling | partial | `../closeout/settled miniCANDI design.md`: 31 rows with pre-written rules. **FINAL (18:15 UTC):** rules 2.1, 2.2, 2.4 and 2.6 are decided; rules 2.3 and 2.5 are not decided (final read of the run, 19:42Z: `results/rescoring_p256/read_20261118T1942Z/RESCORING_READ.md`). Rule 2.4, fixed input depth: on the U-Net, crop-pair correction rises by 0.166 (seed range 0.066) and imputation falls by 0.028 (seed range 0.023); on conv + attention, crop-pair correction rises by 0.039 (seed range 0.026); the switch stays a candidate for the mix-and-match round on these two models and enters no recipe. The architecture for scaling and the data version of the final model are next-session choices (PI ruling of 2026-11-17 21:00Z). |

## 2. What is on the clusters

### Fir

- Planned outage: reservation NovemberOutage from 2026-11-17 07:00Z, expected end 2026-11-20, no hour given. About 570 jobs wait in the queue.
- Among them: the Fir copies of the 256-region rescoring, jobs 4535347 to 4535923 (kit CO6), and the 12 matched-compute trainings 4345024 to 4345035. The PI cancels the rescoring copies after the Nibi scores are on disk (PI, 2026-11-17 22:25Z: "your recommendation is ok"). An agent never cancels them.
- Fir needs a fresh kit after the outage, before any submission.

### Nibi

- **Archived to tape** (PI yes, 2026-11-17): `t112_cf`, `onion/work`, `onion/fastq` and `EIC_REPRO`, as 23 tar files (3,990 GiB; 6 of them hold `onion/work`) in `/nearline/def-maxwl/mforooz/scratch_archive_2026-11-17`. Each tar was compared with scratch (`tar -d`). Restore steps: `NIBI_NEARLINE_ARCHIVE.md` in this folder.
- **Removed from scratch** after the archive and the checks of the Log: every child of `t112_cf` except `pseudoreps` and `fastqarms/genome`; every child of `EIC_REPRO` except `003`; `onion/fastq` whole (the stage step downloads the FASTQs of the 722 blind rows again); `onion/work` whole (removed 2026-11-17 23:20Z, after the QC fill and the 1 bp recount).
- **Stays on scratch:** `t112_cf/pseudoreps` (1,678 files, 80 GiB; the `/project` copy is not proven complete), `t112_cf/fastqarms/genome`, `EIC_REPRO/003` (genome reference cache that every pipeline build needs), the old prediction stores of 2026-11-14 and 15 (85 GiB), and the live prediction stores of the rescoring (each store removes itself after its metrics job).
- Scratch use: 1,486G at 16:42Z against a soft quota of 1,024G (overage permitted, grace running) and a hard quota of 20,480G.
- Saved before the `onion/work` removal: `CO-D3/pending_before_work_removal_2026-11-17.txt` and `CO-D3/not_applicable_markers_2026-11-17/markers.tar`. Restore the 6 markers into `/scratch/mforooz/onion/work` before any `build_list.py pending` call.
- Code snapshots: the 14 kits of 2026-11-17 are packed in `/project/def-maxwl/mforooz/onion_kit_archive`. The current kit is CO9. The kits on Nibi predate the two-sided ladder: push a fresh kit before the next cluster ladder report.

### The running 256-region rescoring (Nibi)

- Submitted 2026-11-17 23:30Z: `slurm/onion_eval/rescoring_p256_nibi.sh`, kit CO9, 62 run-seeds of 7 arm groups, 558 jobs (8 predict + 1 metrics each), all held, job names `mc_<arm>_s<seed>`. The list is `/project/def-maxwl/mforooz/ONION/runs_fir/SUBMITTED.tsv`.
- State at 16:42Z: 30 of 62 scored; the rescuer had resubmitted 18 run-seeds since 16:03Z; 90 jobs running, 198 pending.
- Speed: about 6.3 hours of wall time per conv autoencoder run-seed (Nibi sacct, median of 9) and 5.8 hours per conv + attention run-seed (median of 12). U-Net predict shards hit the 6 h limit and need a 12 h rescue pass. The predict jobs use about 1 of 8 cores (they wait on input and output), about 10 times slower than Fir.
- Two loops on this laptop drive it. Both live in the foreman session's scratchpad (`/private/tmp/claude-501/-Users-mforooz--claude-spares-spare-MPNJF9/79444ed4-29c3-46f2-bc55-03ec17dfd14d/scratchpad/`):
  - Releaser `releaser2.sh` (background task bjysla4bq): every 10 minutes it releases held run-seeds in the arm order of the Log, at most 30 active. The older loops bu4a1ef61 and bbq8o2mge stalled on a hung ssh and are harmless.
  - Rescuer `rescuer.sh` (background task bnfjhcmrl): every 15 minutes it resubmits, with a 12 h predict limit and not held, each run-seed that has a TIMEOUT predict job, no `SCORED` file and no live job. It stops at 62 scored.
- If this laptop session ends, do the work of the loops by hand:
  - Release one run-seed (on Nibi): `squeue -u $USER -h -n <name> -o %i | paste -sd, - | xargs scontrol release`, with name `mc_<arm>_s<seed>`.
  - Rescue one run-seed after a predict TIMEOUT (from the repo root on the laptop): `ssh -o BatchMode=yes nibi "KIT_ID=CO9 ARMS=<group> ONLY='<arm dir>' SEEDS='<seed>' SCORE_PRED_TIME=12:00:00 HOLD=0 DRY_RUN=0 bash -s" < slurm/onion_eval/rescoring_p256_nibi.sh`. The groups are scrambled, views_base, fixed_depth, average_input, impute_x4, two_step_native and two_step_fixed. A metrics job that is stuck on its dependency is cancelled by the PI first (once, Slurm removed it by itself; Log 05:30Z).
  - Count the scored run-seeds (on Nibi): `ls -d /project/def-maxwl/mforooz/ONION/runs_fir/abl_*/seed*/p256_sv/SCORED /project/def-maxwl/mforooz/ONION/runs_fir/rung4_views_base_only/seed*/p256_sv/SCORED | wc -l`.

### Other Nibi jobs

- **FINAL (18:15 UTC):** the 1 bp verification job 27692994 (`onion_a7_verify`) reached its 16-hour limit (TIMEOUT) with 6 out-of-memory kills in its log (8 workers in 32000M). The 1 bp tables are built and NOT verified; the resolution eval set is not built. Rerun with more memory per worker before the resolution arms (handoff note, section added 2026-11-18 17:45 UTC; ready queue entry 22).
- **UPDATE (2026-11-19 03:55 UTC):** the 1 bp tables are now VERIFIED (Nibi jobs 27851432 and 27868178: 412 measurements, 0 failures) and the resolution eval set is built (job 27868179). The DNase-seq replicate ladder was rerun without the 15,201-read track (jobs 27851414, 27868089). See the block of 03:55 UTC in `HANDOFF_NEXT_SESSION.md`.
- Every other close-out job has ended: QC fill and collect, 1 bp recount and tables, variance by bin size, ladder layers and report, second-experiment layers and baselines.

## 3. Rules decided and not decided (256-region rescoring)

The rule texts are section 2.0 of `../closeout/settled miniCANDI design.md`. d is the arm mean minus the reference mean; the bar is the larger of the two seed ranges. **FINAL (18:15 UTC):** the section below shows the read of 15:54Z (30 of 62 scores); the final read of the run is the one of 19:42Z (34 of 62 scores), which adds rule 2.4 as decided (see the row of criterion 4 above and `results/rescoring_p256/read_20261118T1942Z/RESCORING_READ.md`).

- Rule 2.1, decided (settled, strong): the models use covariates for same-view correction. Scrambled covariates lower the correction-only term on all three models (conv autoencoder -0.195, bar 0.046; conv + attention -0.230, bar 0.081; U-Net -0.242, bar 0.129).
- Rule 2.2, decided (settled, strong): all counterfactual views stay in training. Base view only lowers the correction-only term by more than twice the bar on all three models. The composite does not move.
- Rule 2.6, decided: imputation loss weight 4 stays a candidate on the conv + attention only (imputation +0.039, bar 0.017). It costs crop-pair correction on the conv autoencoder (-0.050, bar 0.021).
- Rule 2.3 (two-step routing, native and fixed base depth): not decided, 0 seeds scored.
- **UPDATE (2026-11-19 08:50 UTC):** all 62 run-seeds are scored on 256 regions and all six rules are decided (`results/rescoring_p256/read_20261119T0850Z/RESCORING_READ.md`). Rule 2.3: two-step routing stays off; the composite change is zero or negative on all three models for both arms. The 558 pending Fir copies (job range 4535347 to 4535923) were cancelled under the PI ruling of 2026-11-19 01:40 UTC.
- Rule 2.4 (fixed input depth): not decided, U-Net 1 of 3 seeds.
- Rule 2.5 (average baseline input): not decided, 0 seeds scored.
- Mix-and-match gate (`mix_arms.py qualify`): exit 3 (WAIT), as expected, because five single runs of the pool are not in the rescoring.
- Sign check: 48 of the 50 cells of complete arms agree in sign with the 16-region screening. The two that differ are within the bar on 256 regions.

## 4. Other results of the last 24 hours

- Replicate ladder, two-sided form with the joint cluster bootstrap (PI decision 5): cell-type rung broad histone marks 0.00273 [0.0015, 0.00396], narrow histone marks 0.003547 [0.00198, 0.00511]. No other-experiment rung is estimable. The crossed mixed model is not run.
- DNase-seq layers are not interpretable. One track (ENCSR000EKG, 15,201 reads) sets the common depth of every DNase-seq pair. The rerun without it is ready queue entry 20.
- Variance by bin size: 0.927 (broad marks) and 0.901 (narrow marks) of the 1 bp count variance lies below 25 bp, against 0.960 for read sampling alone. Biological replicates reproduce about 0.1% of it. Read halves anti-correlate at 1 bp after duplicate removal (exploratory).
- QC side tables: NSC and RSC are filled for 89.0% of 2,662 rows, the fragment length for 87.8%, NRF and PBC for 37.8%, FRiP for 0%.
- **FINAL (18:15 UTC):** full test suite on the umbrella after the 32 fixes: 6,950 passed, 9 skipped, 0 failed (83 min; code state 6b2a2273, later commits change documents only). The run before the fixes gave 6,918 passed, 9 skipped, 32 failed. `tools/golden.py` check against origin/main 0d7936a6: 0 ULP, bit-identical. Both gates of a non-experiment pull request hold.

## 5. Documents and where they are

| Document | Path or link |
|---|---|
| Settled miniCANDI design | `../closeout/settled miniCANDI design.md` (the rule-application command updates it) |
| Ready queue for scaling | `../closeout/ready queue for scaling.md` |
| PI decisions of 2026-11-17 | `../closeout/PI decisions 2026-11-17.md` |
| Close-out todo with statuses | `../closeout/closeout todo.md` |
| Final 24 hours plan | `../closeout/final 24 hours plan.md` |
| Training objective | `../onion training objective.md` |
| Close-out deck | https://claude.ai/artifact/Kp1yqCgAUpCBDo19Q6oL9J (private, version 5); source `cruxvault/presentations/onion-closeout-2026-11-18/` |
| Close-out report | `cruxvault/presentations/onion-closeout-2026-11-18-report/report.pdf` (not published) |
| Rescoring reads | `cruxvault/results/rescoring_p256/read_<stamp>/` (git-ignored; foreman worktree) |
| Handoff to the next session | `HANDOFF_NEXT_SESSION.md` in this folder |
| Tape archive and restore | `NIBI_NEARLINE_ARCHIVE.md` in this folder |

## 6. Never resubmit

A job in this list is never submitted again on either cluster. `HANDOFF_NEXT_SESSION.md` has the full list with the older ids.

- Nibi, 256-region rescoring: the 558 jobs of `runs_fir/SUBMITTED.tsv` (from 27680544 on) and every rescue that the rescuer submitted.
- Nibi, close-out data jobs: QC fill 27658288, 27658295; QC collect 27678089; recount probes 27658461 (failed) and 27661198; recount 27662614 and its retry 27664315; 1 bp tables 27665563; variance 27665564 (cancelled by the PI) and 27693041; 1 bp verify 27692994.
- Nibi, ladder and second experiments: pair list 27661033, design 27661078, layers 27661089 (failed) and 27678107, report 27692829; second-experiment layers 27664409 (failed) and 27678103; baselines 27664415, 27664417, 27664418, 27664419, 27664422.
- Fir, queued: 4535347 to 4535923 (rescoring copies), 4345024 to 4345035 (matched-compute trainings).
- Never start a second nearline archive run. Never run `remove_archived.sh` again.
