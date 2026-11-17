---
title: ready queue for scaling (peeling the onion)
kind: queue of runs that are built and not sent, written by the ready-queue agent
parent: "[[closeout todo]]"
written: 2026-11-17T20:30Z
sources: run Log (`.orchestrate/plan.md`, `## Log`, to the line of 2026-11-17T20:25Z), the submit scripts at git e8c1955, Nibi read-only checks of 2026-11-17 20:12 to 20:20 UTC
---

# Ready queue for scaling

This file lists every run that is built and not sent, in the order in which a new session must send it. Each entry gives: what it answers; the prerequisite; the exact command, copied from the usage block of the submit script; the job count and the CPU-hours with the arithmetic; and the decision rule or the PI decision that it waits for. All CPU-hours are allocated CPU-hours (cores x wall time). Nothing in this file has been submitted by the agent that wrote it.

## 0. Facts that every entry depends on

- **Fir.** Planned outage from 2026-11-17 07:00 UTC to about 2026-11-20 (Log 09:45Z). Login works, but on 2026-11-17 20:15 UTC an `ls` of `/project/def-maxwl/mforooz/ONION/runs_fir` and of `onion_kit` gave "Permission denied". So no Fir file could be read for this document. 570 Fir jobs wait in the queue: the 256-region rescoring (jobs 4535347 to 4535923) and the 12 matched-compute trainings (jobs 4345024 to 4345035). They start when Fir returns. Never resubmit them.
- **Nibi `/scratch`.** Read at 2026-11-17 20:12 UTC (`quota -w`): 1,824,258,048 KB used (1,740 GiB) against a soft quota of 1,073,741,824 KB (1,024 GiB). Writes are refused. The folder `onion/work` (1,105 GiB) leaves scratch after the QC fill and the 1 bp recount end (PI yes, Log "PI RULINGS (after the archive)" (ii)). After that, use is about 1,740 - 1,105 = 635 GiB, so about 1,024 - 635 = 389 GiB are free under the soft quota.
- **Prediction stores.** One 256-region same-view scoring of one run-seed keeps a store of about 55 GB and 1,700 files until its metrics job removes it (`slurm/onion_eval/e1_pair_check_p256.sh` header). 389 GiB / 51 GiB per store = 7.6. So on Nibi at most about 6 scorings may run at one time, with a margin. Nibi `/project` has too few free files for stores (Log 10:40Z). Every Nibi scoring entry below needs an explicit `PRED_STORE_ROOT` under `/scratch` and a throttle.
- **Code snapshot (kit).** On Nibi the only current kit is CO8 (git d257503, `/project/def-maxwl/mforooz/onion_kit/CO8`, Log 19:52Z). Scripts whose default kit is COA7, COA4, COA12, COA6, COMIX, COE4, COD2b, COD3 or COE2f need `KIT_ID=CO8` or `K=/project/def-maxwl/mforooz/onion_kit/CO8`; those kits are packed in `/project/def-maxwl/mforooz/onion_kit_archive` (restore: `tar -xf <kit>.tar -C /project/def-maxwl/mforooz/onion_kit`). A kit holds about 2,500 files and the group `/project` is at 428K of 500K files (Log 19:50Z): push one new kit only when CO8 lacks the code, and pack the superseded one as on 2026-11-17. On Fir, push a fresh kit after the outage (`bash tools/onion_kit/push_kit.sh --cluster fir <new id>`) and pass it as `K` or `KIT_ID`; which kits Fir holds now could not be read.
- **Scoring cost per run-seed on 256 regions** (Log 11:04Z, Fir sacct of jobs 4492698 to 4492823): conv autoencoder 30.0, conv + attention 25.4, U-Net about 52 CPU-hours. The A12 script measured the U-Net at 53.43 (`submit_a12.sh` header) and the A13 file uses 51.85; this file uses 52 unless an entry copies a script's own figure. The older figure of 23 is wrong; every entry below that used it is recomputed. Nibi speed is not measured; Fir rates are used for Nibi.
- **One cluster per comparison.** The scorer refuses to mix table versions; the runs of `eic_v1_2` (`eic_v1_2_rung{2,3,4}_base_heldout`) and the eval sets `eval/eic_v1_2` exist on Fir only (Nibi `ls` of `ONION/eval` on 2026-11-17: `eic_v1`, `regions_v1`, `strata`). Nibi holds a Globus copy of 25 run folders in `ONION/runs_fir` (the 21 arm runs of the rescoring and `rung{2,3,4}_base_heldout`, `final_rung4_mech_skip_dna`) and the `eic_v1` 256-region references (`eval/eic_v1/baselines`, 4 baselines and `oracle_floor`).

## 1. Order of sending

| Order | Entry | Cluster | Waits for | CPU-hours |
|---|---|---|---|---|
| 1 | QC-on-input collect | Nibi | the QC fill arrays 27658288, 27658295 | 0.5 |
| 2 | 1 bp recount | Nibi | the probe 27658461 prints `[pointcheck] PASS` | about 18 (cap 336) |
| 3 | Replicate ladder: layers and report | Nibi | nothing | about 6 (13 at most) + 0.5 |
| 4 | Second experiments: layers and baselines | Nibi | nothing (PI decisions only change the report) | about 130 (236 at most) |
| 5 | 256-region rescoring of the 7 decisive arms | Fir queue, or Nibi | Fir return, or writable Nibi scratch and a PI choice | 2,225 |
| 6 | `eic_v1_2` 256-region references and its 9 base scorings | Fir | Fir return, fresh Fir kit | 95.1 + 322.2 = 417.3 |
| 7 | Box-spread signal rescoring of the 9 base runs | Fir, then Nibi | Fir return, fresh Fir kit, Globus | about 434 (+ 96 at most) |
| 8 | Scoring of the 12 matched-compute runs | Fir | the 12 queued trainings | 322.2 + Perceiver IO (not measured) |
| 9 | Resolution: tables, variance, arms | Nibi | entry 2, PI design choice | 8 + 50 + 330 (reduced) |
| 10 | Mix-and-match gate, then round | Fir (gate), Fir or Nibi (round) | entries 5 and 6, PI budget | 474 + 5,390 (Nibi) or 5,308 (Fir) |
| 11 | QC-on-input arm | Fir | entry 1, Globus of the side tables, entry 6 | 404.4 |
| 12 | Learned cell-type embedding | Fir | entry 6 | 728.6 |
| 13 | Second experiments: models | Fir | entry 6 for the base recipes; M1 for the final candidates | 144 (252) + 144 (252) |
| 14 | Minimum covariate set | Fir | entry 10 (the recipe), entry 6, PI scope | 1,927 to 17,901 |
| 15 | Data set with DNase-seq | Nibi | writable scratch, PI table-version choice | about 5,000 (10,752 at most) + 64 + 123 |
| 16 | Blind build, batch 1 | Nibi | writable scratch; the PI runs the command | about 22,700 |
| 17 | Final model and EIC board row | Fir | M1, entries 10, 14, 15, 16 | not built |
| 18 | RNA-seq input | Fir or Nibi | data move, configs | not built |
| 19 | Pointwise MLP under the corrected composite | Fir CPU or Nibi GPU | checkpoint copy | not built |

Entries 1 to 4 can go now. Entries 5 to 8 go when Fir returns (about 2026-11-20). Entries 9 to 16 wait for a PI decision or for an earlier entry. Entries 17 to 19 have no submit script.

## 2. Entries

### Entry 1. QC-on-input collect (item A4)

- Answers: it writes the QC side tables (`$Q/tables/eic_v1_2_qc.tsv`, `eic_v1_2_qc_pr1.tsv`) and the fill-rate report that the QC-on-input arm reads. `Q=/project/def-maxwl/mforooz/ONION/CO-A4`.
- Prerequisite: both fill arrays end (Nibi 27658288, pass whole, 65 tasks; 27658295, pass pr1, 14 tasks; Log 19:58Z; never resubmit). The collect reads only the tars on `/project`, so it can run after `onion/work` leaves.
- Command (Nibi login node; dry run checked on 2026-11-17 20:15 UTC with kit CO8, it printed one `sbatch ... --job-name=a4_qc_collect` line; it saw 0 of 65 and 0 of 14 tars because the fill was running):
  `K=/project/def-maxwl/mforooz/onion_kit/CO8 MODE=collect bash /project/def-maxwl/mforooz/onion_kit/CO8/tools/minicandi/configs/ablations/qc_on_input/submit_qc_on_input.sh`, then the same with `DRY_RUN=0`.
- Cost: 1 job, 1 CPU x 0.5 h = 0.5 CPU-hours (the sbatch line).
- Decision: none. Read the fill-rate report. Read each tar count against 65 + 14 before the collect.

### Entry 2. 1 bp recount (item A7)

- Answers: it builds the 1 bp region store that every resolution arm and the variance analysis read.
- Prerequisite: the probe job 27658461 (pending at 20:12 UTC; never resubmit) prints `[pointcheck] PASS` in `/project/def-maxwl/mforooz/ONION/CO-A7/logs`. The recount READS the half tagAligns in `/scratch/mforooz/onion/work`, so it must end before that folder leaves scratch. The PI approved it and said "You submit it" (Log 20:10Z).
- Command (from the laptop worktree; usage block of `slurm/onion_data/closeout_a7_submit.sh`; dry run checked with `KIT_ID=CO8`: "412 views in 21 tasks; 2 CPUs, 24000M, 08:00:00 each"):
  `ssh -o BatchMode=yes nibi 'KIT_ID=CO8 bash -s -- recount' < slurm/onion_data/closeout_a7_submit.sh`, then `'KIT_ID=CO8 DRY_RUN=0 bash -s -- recount'`.
- Cost: 21 array tasks. Estimate 17.3e9 tagAlign lines / 1.0e6 per s = 4.8 h, + 412 views x 20 s = 2.3 h, + 21 x 0.1 h = 9.2 h, x 2 CPUs = about 18 CPU-hours; cap 21 x 8 h x 2 = 336 (item A7 file, section 9.7). Output about 25 GB and 1,408 files on Nibi `/project`.
- Decision: every view needs a passed read-set check (`provenance.json`). A failed view is redone on resubmit, never skipped.

### Entry 3. Replicate ladder: layers and report (item E4)

- Answers: the replicate ladder on the second-experiment add-on, i.e. the sampling, replicate, cell-type and other-experiment rungs of the variance.
- Prerequisite: none. The script reads `/scratch/mforooz/onion/measurements` and the D2 add-on tables; it never reads `onion/work`.
- Command (laptop worktree; usage block of `slurm/onion_eval/closeout_e4_submit.sh`; dry run checked with `KIT_ID=CO8`, it printed the pair-list, design and layers jobs):
  `KIT_ID=CO8 bash slurm/onion_eval/closeout_e4_submit.sh layers nibi`, then with `DRY_RUN=0`; after the layers job, `KIT_ID=CO8 DRY_RUN=0 bash slurm/onion_eval/closeout_e4_submit.sh report nibi` (the report runs without the mixed-model tables and says so).
- Cost: layers 3 jobs: 1 x 0.5 + 1 x 0.5 + 4 x 3 = 13 CPU-hours at most, about 6 expected (t305: 1 h 19 min on 4 CPUs for 3,908 pairs; this list holds 3,979). Report 1 job, 1 CPU x 0.5 h.
- Rule: the stage `mixed` (4 jobs, 128 CPU-hours at most) must NOT run until the model has a residual variance per library. That code does not exist (Log 13:00Z; `results/closeout_E4/E4_MIXED_DIAG.md`).
- PI decisions open: adopt the two-sided subtraction in the ladder (it lowers the cell-type rung by 23% on the smoke data); replace the telescoping acceptance check by the common-unit check; the lab-layer report.

### Entry 4. Second experiments: layers and baselines (item D2)

- Answers: the variance layers of the 71 second experiments (lab, donor, repeat experiment), and the baselines that the add-on model scores need.
- Prerequisite: the add-on tables `ONION/CO-D2/tables/eic_v1_2x` (built by prep job 27601259). Its `covariates_fill.tsv` and `CHANGES.md` are plain copies of the base, because the build predates the fill-table fix; a rebuild with `--catalog` gives the appended rows (item D2 file, 9.9). `tables_sha256` does not hash these two files, so the layers and baselines are not affected.
- Command (laptop worktree; usage block of `slurm/onion_data/closeout_d2_submit.sh`; dry runs checked with `KIT_ID=CO8`):
  `KIT_ID=CO8 bash slurm/onion_data/closeout_d2_submit.sh layers nibi` and `KIT_ID=CO8 bash slurm/onion_data/closeout_d2_submit.sh baselines nibi`, then each with `DRY_RUN=0`.
- Cost: layers 1 job, 8 CPUs x 12 h = 96 at most, about 50 expected; baselines 5 jobs x 2 CPUs x 14 h = 140 at most, 5 x 16 = 80 expected (item D2 file, 9.7). Expected total 130, at most 236.
- PI decisions open: report lab together with donor (recommended: the pure lab effect has too few pairs per assay class); drop the 17 unbuilt DNase-seq second-experiment rows (recommended; about 7,300 CPU-hours, and they cannot rescue the lab layer).

### Entry 5. 256-region rescoring of the 7 decisive arms (item E1, region check 4)

- Answers: the 16-versus-256 check failed its pre-written rule (Fir job 4534658, `PAIR_REGIONS_V3.md`), so the correction terms must come from all 256 regions. This entry rescores the arms that decide the settled design: scrambled covariates, base view only, two-step native, two-step with fixed reference depth, fixed input depth, average input, imputation weight 4; on conv autoencoder, conv + attention and U-Net.
- Prerequisite: route A, Fir returns and the queued jobs 4535347 to 4535923 run (kit CO6, `runs_fir/score_p256_decisive.sh`, log `score_p256_decisive.log`; Log, line "(foreman) 16-versus-256 check"). Route B, Nibi: the PI moved the rescoring to Nibi (ruling 68); it needs writable scratch, a store root, and a choice about the Fir jobs. The run folders on Nibi are in `/project/def-maxwl/mforooz/ONION/runs_fir` (Globus copy, 74 checkpoints verified, Log 10:40Z).
- Route A command: none. The jobs are in the Fir queue. Never resubmit them.
- Route B command: no Nibi script exists, and the Fir script cannot be read during the outage. The command below is put together from the options of `slurm/onion_eval/e1_pair_check_p256.sh` (score mode) and `submit_mix.sh` (STAGE=gate), which name the same options as `score_p256_decisive.sh`. Run it on a Nibi login node, dry run first:

```bash
O=/project/def-maxwl/mforooz/ONION; K=/project/def-maxwl/mforooz/onion_kit/CO8
module load python/3.10.13; unset EBPYTHONPREFIXES ONION_EXCLUDE
export ONION_STAGE=1 CLUSTER=nibi NICE=300 SEEDS="0 1 2" SCORE_ONLY=1 DRY_RUN=1 \
  SCORE_SAME_VIEW=1 SCORE_PHASES=1 SCORE_SUBDIR=p256_sv SCORE_SHARDS=8 SCORE_PRED_DEVICE=cpu \
  SCORE_PRED_CPUS=8 SCORE_PRED_MEM=32000 SCORE_INFER_BATCH=8 SCORE_PRED_TIME=06:00:00 \
  SCORE_CPUS=16 SCORE_MEM=64000 SCORE_TIME=06:00:00 REFS_FROM=$O/eval/eic_v1/baselines \
  KEEP_PRED_STORE=0 PRED_STORE_ROOT=/scratch/mforooz/onion/pred_store HOLD=1
for spec in 2:abl_scrambled_covariates_scrambled_covariates_convae 3:abl_scrambled_covariates_scrambled_covariates_convattn \
  4:abl_scrambled_covariates_scrambled_covariates_unet 2:abl_views_base_only_convae 3:abl_views_base_only_convattn \
  4:rung4_views_base_only 2:abl_two_step_two_step_native_convae 3:abl_two_step_two_step_native_convattn \
  4:abl_two_step_two_step_native_unet 2:abl_two_step_two_step_fixed_convae 3:abl_two_step_two_step_fixed_convattn \
  4:abl_two_step_two_step_fixed_unet 2:abl_depth_input_input_depth_fixed_convae 3:abl_depth_input_input_depth_fixed_convattn \
  4:abl_depth_input_input_depth_fixed_unet 2:abl_average_input_average_input_convae 3:abl_average_input_average_input_convattn \
  4:abl_average_input_average_input_unet 2:abl_losses_loss_impute_x4_convae 3:abl_losses_loss_impute_x4_convattn \
  4:abl_losses_loss_impute_x4_unet; do
  r=${spec%%:*}; d=${spec#*:}
  bash $K/slurm/minicandi/submit_arms.sh $K $O/tables/eic_v1 $O/eval/eic_v1/evalset.json $O/runs_fir $r "$d=$O/runs_fir/$d/seed0/run.json"
done
```

  `HOLD=1` puts every job on hold, so that a release script can let at most about 6 stores live at once (section 0). `submit_arms.sh` skips a seed without `DONE` (`abl_depth_input_input_depth_fixed_convae` seed 1 never trained: Fir job 4438068, TIMEOUT). Check the arm list against the 21 folders in Nibi `runs_fir` and against `score_p256_decisive.sh` when Fir is readable.
- Cost: 20 conv autoencoder + 21 conv + attention + 21 U-Net run-seeds = 62; 62 x (8 predict + 1 metrics) = 558 jobs; 20 x 30.0 + 21 x 25.4 + 21 x 52 = 600 + 533.4 + 1,092 = 2,225.4 CPU-hours (Log 11:04Z). The PI approved 2,225 (Log, "PI RULINGS (after the archive)" (iv)).
- Rule: route B duplicates route A. The Nibi `submit_arms.sh` cannot see the Fir queue. Both write to different file systems, so nothing is corrupted, but 2,225 CPU-hours are spent twice. Cancelling the Fir jobs is the PI's. Recommendation: use route B only if Nibi scratch is writable well before Fir returns, and ask the PI to cancel 4535347 to 4535923 first.
- Decision rule after the scores: the pre-written rule of `results/pair_region_check/PAIR_CHECK_PREREGISTRATION.md` (amendment of 2026-11-16) and the M1 settled-design document. Read with `PYTHONPATH=$K/src python $K/tools/onion_eval/select_rule.py arm-summary --runs <runs root> --sub p256_sv`.

### Entry 6. `eic_v1_2` 256-region references and its 9 base scorings (item D1; shared prerequisite)

- Answers: every arm on `eic_v1_2` scored on 256 regions (entries 10 to 14) needs the 256-region reference terms of `eic_v1_2` (4 baselines + oracle floor) and the 256-region scores of the 3 base recipes x 3 seeds. Neither exists on either cluster.
- Prerequisite: Fir returns; a fresh Fir kit (`<FK>` below). The base runs `eic_v1_2_rung{2,3,4}_base_heldout` and `eval/eic_v1_2/evalset.json` are on Fir only.
- Commands (Fir login node; usage blocks of the two scripts):
  1. `CLUSTER=fir K=/project/def-maxwl/mforooz/onion_kit/<FK> STAGE=baselines bash /project/def-maxwl/mforooz/onion_kit/<FK>/tools/minicandi/configs/ablations/mix/submit_mix.sh`, then with `DRY_RUN=0` (4 baselines `score.sh baseline`, 16 CPUs, 128000M, 4 h; 1 oracle job).
  2. After the 4 SCORED and `oracle_floor.json` exist: `CLUSTER=fir K=/project/def-maxwl/mforooz/onion_kit/<FK> STEP=ref bash /project/def-maxwl/mforooz/onion_kit/<FK>/tools/minicandi/configs/ablations/covariate_fields/submit_a12.sh`, then with `DRY_RUN=0` (it scores `eic_v1_2_rung{2,3,4}_base_heldout` into `p256_sv`; its real submission refuses until step 1 is done).
- Cost: step 1: 5 jobs; 16 CPUs x (5,062 + 6,322 + 5,395 + 4,568) s of the `eic_v1` 256-region baseline wall times = 94.9, + oracle 0.2 = 95.1 CPU-hours (item A13 file, 4.8). Step 2: 9 x 9 = 81 jobs; 3 x 30.0 + 3 x 25.4 + 3 x 52 = 90 + 76.2 + 156 = 322.2 CPU-hours (the script prints 326.7 with its U-Net rate 53.43). Total 95.1 + 322.2 = 417.3. The earlier figure "about 320" (status page) left out the references.
- Decision: none. `submit_mix.sh STAGE=runs` also submits the step-2 scorings for its reference arm; both scripts are idempotent through `submit_arms.sh`, so the second call skips them.

### Entry 7. Box-spread signal rescoring of the 9 base runs (item E2)

- Answers: signal p-value and peaks from the predicted NB with the box-spread pileup (PI ruling 71), for conv autoencoder, conv + attention and U-Net, 3 seeds, with the predicted-control route as a headline row (PI comment, ruling 70); and their AlphaGenome 2 board rows. It replaces the in-bin numbers that carry a known flaw (status page, "Earlier numbers that carry a known flaw").
- Prerequisite: Fir returns; a fresh Fir kit (`<FK>`; the script default COE2f is not known to exist on Fir). Step 2 needs a Globus copy Fir `$O/e2/tracks_box` to Nibi (the foreman starts Globus; about 61 GB for one form, Log 11:15Z).
- Commands (laptop worktree; usage block of `slurm/onion_eval/e2_submit.sh`):
  1. `KIT_ID=<FK> bash slurm/onion_eval/e2_submit.sh fir`, then with `DRY_RUN=0`.
  2. `KIT_ID=CO8 bash slurm/onion_eval/e2_submit.sh nibi_baselines`, then with `DRY_RUN=0` (independent of step 1; dry run checked with CO8 on 2026-11-17).
  3. Globus `$O/e2/tracks_box` Fir to Nibi (foreman).
  4. `KIT_ID=CO8 bash slurm/onion_eval/e2_submit.sh nibi_board`, then with `DRY_RUN=0`.
  5. The lower dependence bound: repeat 1 to 4 with `SIGNAL_BOX_DEPENDENCE=independent` (outputs `*_box_ind`).
- Cost per dependence form (script header): step 1, 9 jobs x 2 CPUs x about 9.5 h = 171 (timing job 4448330), + 2 check jobs, 1 x 4 h + 1 x 0.5 h = 4.5 at most; step 2, 2 dumps, 16 CPUs x 4 h = 64 each, 128 at most, about 40 likely, + 1 job of 4 CPUs (t308 took 21 min = 1.4); step 4, 2 jobs x 4 CPUs x 6 h = 48 at most. Expected per form 171 + 4.5 + 40 + 1.4 = 216.9, plus the board jobs (48 at most). Both forms: about 434, plus 96 at most for the boards. Jobs per form: 11 + 3 + 2 = 16.
- Decision: none for the formula (ruled). Report both dependence bounds. The in-bin form stays as an option and is documented as overstating the Poisson part.

### Entry 8. Scoring of the 12 matched-compute runs (item A9)

- Answers: whether the architecture order changes when each architecture gets equal FLOPs (2.4e15) instead of equal windows.
- Prerequisite: the 12 trainings (Fir 4345024 to 4345035; `final_rung2_n_levels3_matched`, `final_rung3_attn_at_128_positions_matched`, `final_rung4_lr3e-3_matched`, `final_rung6_reference_matched`, 3 seeds each; 2-day limit) are DONE. They were pending with "ReqNodeNotAvail, Reserved" before the outage (Log 05:15Z).
- Command: the item file names the pattern of `runs_fir/score_final.sh` (kit CM, `SCORE_SAME_VIEW=1`, `SCORE_SUBDIR=p16_sv`, module `python/3.10.13` only, `EBPYTHONPREFIXES` unset). That file is on Fir and could not be read during the outage. Because the 16-versus-256 check failed, score these runs into `p256_sv` too: the option set of `e1_pair_check_p256.sh` score mode, with `CLUSTER=fir`, the eic_v1 tables, `eval/eic_v1/evalset.json`, `REFS_FROM=$O/eval/eic_v1/baselines`, and one `submit_arms.sh` call per run as in entry 5.
- Cost (256 regions): 3 x 30.0 + 3 x 25.4 + 3 x 52 = 322.2 CPU-hours for the three CPU architectures, + 3 Perceiver IO seeds at a cost that no file gives. 4 x 3 x 9 = 108 jobs. The 16-region cost per seed is not separated in the tally (notepad, compute tally); the 256-region rate is its upper bound. The equal-window winners `final_rung{2,3,4}_*_equal` have no 256-region score; a 256-region comparison adds another 322.2 + the Perceiver IO cost. The trainings themselves cost (186.49 + 169.90 + 269.55 + 110.01) x 3 = 2,207.9 CPU-hours (`tools/minicandi/configs/final/MANIFEST.tsv`, `cpu_h_per_run`), already approved and queued.
- Decision rule (item A9 file, section 6): "beyond" = difference of seed means larger than the larger seed range; same top group means the order is robust; a move beyond the range means M1 uses the matched order for scaling. PI decision open: confirm or reverse the per-rung narrowing of notepad L289.

### Entry 9. Resolution: tables, variance, arms (item A7)

- Answers: how much signal lies below 25 bp per assay (variance analysis, no model), and how model quality changes from 5 to 125 bp (arms).
- Prerequisite: entry 2 with every view passed. `tables` refuses otherwise. Arms run on Nibi from `ONION/CO-A7`.
- Command (laptop worktree; usage block of `slurm/onion_data/closeout_a7_submit.sh`; dry run with `KIT_ID=CO8` printed the tables, variance and recount lines):
  `ssh -o BatchMode=yes nibi 'KIT_ID=CO8 DRY_RUN=0 bash -s -- tables variance' < slurm/onion_data/closeout_a7_submit.sh`, then `'KIT_ID=CO8 DESIGN=reduced DRY_RUN=0 bash -s -- arms'` (or `DESIGN=full`).
- Cost: tables 1 job x 8 CPUs, about 8; variance about 50; reduced arms (conv autoencoder at 5, 25, 125 bp and the two context controls, 3 seeds = 15 runs, 78 jobs) about 330; full arms (75 runs, 396 jobs) about 2,265 (item A7 file, 9.7). Totals with the recount: reduced about 405, full about 2,340.
- PI decision open: reduced or full design.

### Entry 10. Mix-and-match gate and round (item A13)

- Answers: whether the single settings that helped add up, per model; it names each model's recipe for M1.
- Prerequisite: entry 5 (the 256-region scores of fixed depth, imputation weight 4 and average input) and entry 6 (the `eic_v1_2` references). The gate runs on Fir (the checkpoints are there). The round refuses unless `mix_arms.py qualify` passes on the 256-region arm summary.
- Commands (usage block of `tools/minicandi/configs/ablations/mix/submit_mix.sh`; default kit COMIX is archived: pass `K`):
  1. Gate, Fir: `CLUSTER=fir K=<FK path> PRED_STORE_ROOT=/scratch/mforooz/onion/pred_store STAGE=gate bash <FK path>/tools/minicandi/configs/ablations/mix/submit_mix.sh`, then with `DRY_RUN=0`.
  2. `PYTHONPATH=$K/src python $K/tools/onion_eval/select_rule.py arm-summary --runs $O/runs_fir --sub p256_sv` into a tsv; `python $K/tools/minicandi/mix_arms.py qualify --summary <tsv>` (exit 0 PASS, 1 FAIL, 3 WAIT).
  3. Round: `CLUSTER=<fir|nibi> K=<kit> PRED_STORE_ROOT=<dir> QUALIFY_SUMMARY=<tsv> STAGE=runs bash .../submit_mix.sh`, then with `DRY_RUN=0`. On Nibi the round also needs `STAGE=baselines` there and the `eic_v1_2` eval set on Nibi (Globus); the references `eic_v1_2_rung*_base_heldout` are then trained again on Nibi.
- Cost: gate 3 x 30.0 + 3 x 3 x 25.4 + 3 x 51.85 = 90.0 + 228.6 + 155.6 = 474.2 CPU-hours, 135 jobs. Round: 5,389.9 on Nibi, 5,307.7 on Fir, 1,134 jobs + 5 reference-term jobs (item A13 file, 4.8; it already uses the measured per-model rates). If the gate removes a switch, recount.
- Decision rule: item A13 file, section 4.6 (eligibility against the seed range, the simplest in-range arm wins, native-depth score for fixed-depth arms with the PI's deployment caveat). PI decision open: the budget of the gate and of the round.

### Entry 11. QC-on-input arm (item A4)

- Answers: whether read-derived QC values (with presence flags and the leak guard) on the input side help, against metadata only.
- Prerequisite: entry 1; the side tables on Fir at the same path (Globus, foreman); entry 6; a fresh Fir kit.
- Command (usage block of `submit_qc_on_input.sh`): `CLUSTER=fir K=<FK path> MODE=train bash <FK path>/tools/minicandi/configs/ablations/qc_on_input/submit_qc_on_input.sh`, then with `DRY_RUN=0`.
- Cost: training 82.2 (the `eic_v1_2` base runs) + scoring 3 x 30.0 + 3 x 25.4 + 3 x 52 = 322.2; total 404.4 CPU-hours, 90 jobs (item A4 file).
- PI decision open: accept the arm's changes to the "QC in C" design (presence flags, leak guards) and its cost. Limit: the foreign-covariate scoring path does not apply the leak rules or the side table; do not run it on this arm.

### Entry 12. Learned cell-type embedding (item A6)

- Answers: whether a learned per-cell embedding helps; the zero-shot control (cell id forced to unknown) is required before any "helps" conclusion. It says nothing about unseen cell types (EIC holds out tracks, not cells).
- Prerequisite: entry 6; Fir (the reference runs are there); a fresh Fir kit.
- Commands (usage block of `tools/minicandi/configs/ablations/cell_embedding/submit_cell_embedding.sh`; default kit COA6 is archived):
  `CLUSTER=fir K=<FK path> bash <FK path>/tools/minicandi/configs/ablations/cell_embedding/submit_cell_embedding.sh` (MODE=train), then `MODE=audit`, `MODE=derive`, `MODE=zeroshot`, each first as a dry run, then with `DRY_RUN=0`.
- Cost, recomputed with the measured scoring rates (the item file used 23): training 82.2; arm scoring 3 x 30.0 + 3 x 25.4 + 3 x 52 = 322.2; audit and derive 2 x 2 CPUs x 0.5 h = 2; zero-shot scoring 322.2. Total 82.2 + 322.2 + 2 + 322.2 = 728.6 CPU-hours (was 498). Jobs: 90 + 2 + 81 = 173.
- Decision table: item A6 file, section 9.7. PI decision open: budget.

### Entry 13. Second experiments: models (item D2)

- Answers: the cross-lab correction gap closed and metric 7 of the base recipes and of the final candidates on the add-on.
- Prerequisite: runs trained on `eic_v1_2` (Fir only today); for the final candidates, M1 names them. The add-on eval set on Fir, or the runs on Nibi (Globus).
- Command (usage block of `closeout_d2_submit.sh`): `KIT_ID=<kit> bash slurm/onion_data/closeout_d2_submit.sh models fir`, then with `DRY_RUN=0` (`RUNS=` names the final candidates).
- Cost: base recipes 9 jobs x 16 = 144 expected, 252 at most; final candidates the same (item D2 file, 9.7; this uses a measured add-on cost, not 23).
- Decision table: item D2 file, 9.5.

### Entry 14. Minimum useful covariate set (item A12)

- Answers: a ranked list of the covariate fields, the exclusion curve, and the fields that can be removed with no loss detectable at this power. The PI named it the last step of the session and a deliverable.
- Prerequisite: the recipe from entry 10 (regenerate the configs with `tools/minicandi/covariate_field_arms.py` for that recipe, new kit); entry 6; the PI's choice of scope.
- Command (usage block of `submit_a12.sh`): `CLUSTER=fir K=<kit> STEP=<ref|screen|lofo|groups|curve> [RUNGS=<2|3|4>] bash <kit>/tools/minicandi/configs/ablations/covariate_fields/submit_a12.sh`, then with `DRY_RUN=0`. The script prints the job count and the CPU-hours.
- Cost (item A12 file, 7.7, with the U-Net at 53.43): full plan 17,901; without screening 11,366; final model only (LOFO + groups, 17 arms x 3 seeds, + 3 reference scorings): conv autoencoder 1,927, conv + attention 2,105, U-Net 3,245 (with the curve 3,008, 3,298, 5,060).
- Decision rule: item A12 file, 7.4. PI decision open: scope.

### Entry 15. Data set with DNase-seq (item D3)

- Answers: a table version in which DNase-seq is correction-scorable (candidate `eic_v1_3_candidate`, 2,812 rows; 12 assays against 11).
- Prerequisite: writable Nibi scratch (the reruns stage FASTQs and write work folders there); the FASTQs of the 4 rows are gone from scratch (section 3); the PI chooses the table version of the final model. `version` uses the work snapshot when `onion/work` is gone.
- Commands (usage block of `slurm/onion_data/closeout_d3_submit.sh`; default kit COD3 is archived):
  `ssh -o BatchMode=yes nibi 'KIT_ID=CO8 bash -s -- reruns' < slurm/onion_data/closeout_d3_submit.sh`, then `'KIT_ID=CO8 DRY_RUN=0 bash -s -- reruns'` (kit KF4 builds the rows; rows 56, 132, 280, 396; one 7-day rerun per row, rule B.5); after they end, `'... -- version'`, then `'... -- baselines'`.
- Cost: reruns 4 rows x 16 CPUs x 168 h = 10,752 at most, about 16 x 24 h x (3 + 3 + 4 + 3) = 4,992 expected; version 16 x 4 = 64 at most; baselines on both eval sets: 16 regions about 28 (D1's 4 jobs, 28.1) + 256 regions 95.1 (entry 6 arithmetic; the item file used 23 each = 92) = about 123. If the PI picks `eic_v1_3`, the 9 base runs train again (82.2) and every entry on `eic_v1_2` moves to the new version.
- PI decision open: the table version of the final model (`eic_v1_2`, or the DNase-seq version).

### Entry 16. Blind build, batch 1 (item D4)

- Answers: the blind EIC tracks and their controls, needed for the single blind pass and the EIC board row.
- Prerequisite: writable Nibi scratch (batch 1 stages 1,227 GB and keeps about 1.4 TB of blind BAMs; the script refuses above 16 TiB); the PI runs the submit command (the permission check refused the agent's submission, Log 00:25Z).
- Command (laptop worktree; usage block of `slurm/onion_data/closeout_d4_submit.sh`): `ssh -o BatchMode=yes nibi 'DRY_RUN=0 bash -s -- 1' < slurm/onion_data/closeout_d4_submit.sh`. The dry run of 2026-11-17 20:12 UTC (`'bash -s -- 1'`) printed 72 rows to submit, 0 skipped: 59 blind ChIP-seq and ATAC-seq rows with their controls, 3 blind DNase-seq rows, 7 EIC training DNase-seq outage reruns (113, 211, 216, 232, 384, 385, 387) and 3 second-experiment reruns (2049, 2075, 2079). Each row gets a stage array (FASTQ download with md5 check), a build array and a signal pass.
- Cost: about 22,700 allocated CPU-hours (Log 2026-11-16T23:40Z; `results/closeout_D4/BUILD.md`), approved by ruling 63. The Fir port (`closeout_d4_submit_fir.sh`) is superseded by ruling 68.
- Rule: run the 10 outage reruns on one cluster only. Batches 2 and 3 (all 722 rows) refuse while a batch-1 blind row is queued or running; ruling 63 limits this work to batch 1. The header line "Batch 3 stages nothing new (its FASTQs are on /scratch)" is no longer true: those FASTQs were removed (section 3).

### Entry 17. Final model and EIC board row (item D4)

- Not built. There is no freeze file, no final-model config, no blind tables, no `evalset_blind.json`, no `BLIND_PASSES.tsv` and no `final_board.tsv`. The scorers exist (`tools/onion_eval/score.py`, `ag2_score.py`; the board reproduction matches all 25 published ranks, `results/t166/BOARD_REPRO.md`).
- Needs: the M1 design and the recipe of entry 10; the covariate set of entry 14 if the PI wants it in the final model; the table version of entry 15; the blind tracks of entry 16; then a freeze file, 3 seeds on training plus validation and 3 seeds training-only (6 trainings, about 11 to 15 CPU-hours each by the item file), the D14 condition-2 check on validation, one ledgered blind pass, the board row (seed 0, spread beside it).

### Entry 18. RNA-seq input (item A5)

- Not built. The track builder exists (`tools/onion_data/rna_inputs.py`, `slurm/onion_data/rna_inputs.sh`; config writer `tools/minicandi/stage3_arms.py`), but there are no `rna_track` or `rna_vector` configs, no RNA-cell eval subset and no submit script. The RNA tracks are on Nibi; the PI allowed a Globus move to Fir.
- Cost when built (item A5 file, recomputed): 2 arms x 3 architectures x 3 seeds = 18 runs; training 2 x 82.2 = 164.4; scoring on the RNA subset, at most 2 x 322.2 = 644.4 for the arms and 322.2 for the 9 reference scorings (the subset is smaller than 256 regions x all targets, so these are upper bounds). At most about 1,131 CPU-hours.

### Entry 19. Pointwise MLP under the corrected composite (item A10)

- Not built. Step 1 (composite version 3 from stored files, under 1 CPU-hour) needs no job. Steps 2 and 3 (same-view pass and the unscored seeds) need the rung-1 checkpoints copied from Nibi to Fir, or a Nibi GPU, and no script exists. Rung-1 CPU prediction is not measured on Fir; metrics about 11.5 CPU-hours per seed (item A10 file).

## 3. What must come back from tape, or be downloaded again

Archive: `/nearline/def-maxwl/mforooz/scratch_archive_2026-11-17`, index in `/project/def-maxwl/mforooz/ONION/nearline_archive_2026-11-17/`, steps in `run_state/NIBI_NEARLINE_ARCHIVE.md`. A tape read takes hours to about a day and needs free scratch.

| What | State | Needed by | How it comes back |
|---|---|---|---|
| Staged FASTQs of the 722 priority-4 rows (842 of the 860 GiB of `onion/fastq`, Log "PRE-REMOVAL CHECK, cluster half") | removed from scratch 2026-11-17 | entry 16 (batch 1), batches 2 and 3 of D4, entry 15 (rows 56, 132, 280, 396), the 17 DNase-seq second-experiment rows if the PI keeps them | the stage step of `build_row.sh submit` (`stage_fastq.sh`) downloads them again from ENCODE with an md5 check; the tars `onion-fastq__*` hold them as a second route |
| `onion/work` (pipeline work folders of 457 built rows, 1,105 GiB) | on scratch until the QC fill and the recount end, then removed (PI yes) | after removal: a rerun of a failed QC-fill chunk (a fill pass is never resubmitted; it needs a new `Q`), a redo of a failed recount view (entry 2), the D3 `candidate` and `verify` parts | tars `onion-work__*`, one folder at a time (`tar -xf <tar> -C /scratch/mforooz onion/work/<experiment>`) |
| NOT_APPLICABLE markers of rows 102, 418, 419, 425, 428, 431 (in `onion/work`) | go with `onion/work` | any `build_list.py pending` call (D3 reruns, D4 batches): without the markers, these rows can be listed as pending again | the D3 work snapshot (`ONION/CO-D3/work_snapshot_2026-11-17.json.gz`, 6 NOT_APPLICABLE files) or the tar; check the pending list before each submission |
| `t112_cf` (except `pseudoreps` and `fastqarms/genome`), `EIC_REPRO` (except `003`) | removed | no entry in this queue | tars `t112_cf*`, `EIC_REPRO*` |
| Kept on scratch | `EIC_REPRO/003/refcache` (genome reference cache of every pipeline build), `t112_cf/fastqarms/genome`, `t112_cf/pseudoreps` (the `/project` copy is not proven complete) | entries 15 and 16 need the refcache | do not remove |

## 4. PI decisions that this queue waits for

1. Which cluster runs the 256-region rescoring (entry 5), and, for Nibi, the cancel of Fir jobs 4535347 to 4535923.
2. Budget of the mix-and-match gate (474) and round (5,390 on Nibi, 5,308 on Fir) (entry 10).
3. Scope of the minimum-covariate-set search, 1,927 to 17,901 (entry 14).
4. Second experiments: report lab together with donor; drop the 17 unbuilt DNase-seq rows (entries 4, 13).
5. Table version of the final model: `eic_v1_2` or the DNase-seq version (entry 15).
6. Resolution design: reduced (405) or full (2,340) (entry 9).
7. QC-on-input arm: the changes to the "QC in C" design and 404 CPU-hours (entry 11).
8. Replicate ladder: two-sided subtraction; the common-unit acceptance check; the lab layer (entry 3).
9. Matched compute: confirm or reverse the per-rung narrowing of notepad L289 (entry 8).
10. Cell-type embedding budget at the corrected 728.6 CPU-hours (entry 12).
