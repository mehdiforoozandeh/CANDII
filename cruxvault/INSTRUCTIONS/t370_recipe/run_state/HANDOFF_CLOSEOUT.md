---
title: miniCANDI recipe run (t370), close-out handoff to the p-value session
written: 2026-11-27
from: recipe session (branch exp/t370-minicandi-recipe, PR 63, worktree .claude/worktrees/recipe)
to: p-value session (EIC comparison), via the foreman
---

# Close-out handoff

The recipe run is closed. The PI ruled on 2026-11-27 (relayed by the foreman) that the p-value session owns the two open hand-offs below. Decision rows D59, D60 and D61 in `cruxvault/INSTRUCTIONS/t370_recipe/run_state/longrun/DECISIONS.md` record the rulings. Every time is UTC.

## The final model

- Architecture (D55): U-Net, read-disjoint training pairs, average-signal input, Pearson-residual input, head skip kept, 160k windows, control loss weight 1. Per-assay dispersion recalibration after training.
- Run directory on Fir: `/project/def-maxwl/mforooz/ONION/runs_rcp/rcp_combo_w2x_unet/seed0`.
- Kit on Fir: `/project/def-maxwl/mforooz/onion_kit/RCP9`.
- The output covariates are the 11 metadata columns only (`tools/minicandi/contract.py` META_COLUMNS). The run's `run.json` has `qc_on_input` false, so the model reads no QC score.

## Hand-off 1: Avocado baseline to the EIC comparison

The Avocado chain retrains the Avocado baseline (tensor-factorization imputation model) with its training step only, on Fir GPU slices (D59).

| step | job | state at 2026-11-27 06:11 | writes |
|---|---|---|---|
| shared training, chr20 | 6291945 | COMPLETED | `/scratch/mforooz/onion/avocado_train_only/sc/ckpt/d3/shared_chr20.pt` |
| genome training, 23 chromosomes | 6291946, array 0-22%12 | 0-13 COMPLETED; 14-22 PENDING (Priority) after the PI-approved throttle reset | `.../sc/ckpt/d3/genome_chr*.pt` |
| predict | 6291947, array 0-22%12 | PENDING (Dependency) | predictions per chromosome |
| bigwig | 6291948, array 0-44%12 | PENDING (Dependency) | `/scratch/mforooz/onion/avocado_train_only/sc/pred_bw/d3` |

- Logs: `/project/def-maxwl/mforooz/ONION/logs/%x_%A_%a.out`.
- Source script (the chain was submitted from a command that mirrors it, with GPU-slice requests): `slurm/onion_eval/avocado_train_only.sh`.
- Expected end: the bigwigs about 8-10 h after the genome tasks start, from the run times of tasks 0-13 (0.9-7.4 h each, sacct).
- If tasks 14-22 again show `JobArrayTaskLimit` with no task running, run `scontrol update JobId=6291946 ArrayTaskThrottle=13`. The PI approved this fallback on 2026-11-27.
- Your step: when 6291948 completes, convert and score the Avocado bigwigs on the EIC validation tracks next to the miniCANDI counts. This serves requirement R9 ("compare minicandi against other imputation methods").

## Hand-off 2: blind prediction with the final model

- The Nibi blind build (task t184, batch 1, 72 rows) failed on 2026-11-26/27. All 74 build tasks in arrays 28737209/11/15/45/47/64/67/69/75 failed. Cromwell aborted (returncode -6) at different pipeline stages, after 0.58-7.45 h (median 4.77 h). In 45 tasks Python then crashed with a bus error (exit 135).
- Ruled out: out-of-memory (oom_kill 0; peak anonymous memory at most 21 GB of 62.5 GB), scratch disk (local NVMe, 3.5 TB, 3% used), and a changed Cromwell jar (`~/.caper/cromwell_jar/cromwell-82.jar`, unchanged since 2026-08-12). The likely cause is a shared-filesystem fault after the Nibi maintenance. It is not proven, because Slurm deleted the job-local scratch that held cromwell.out.
- Retest: the signal passes 28737212 (48 tasks) and 28737239 (3 tasks) re-run each whole row. They have been RUNNING since 2026-11-27 05:28. The PI ruled to keep them (D60). The verdict comes after about 5 h.
- If they fail the same way: propose, through the foreman, the code change that writes cromwell.out and the Java error file to /project. It needs the PI's yes. Do not resubmit.
- If they succeed: the built rows are on Nibi, and the model is on Fir. Move the inputs with Globus, not through the laptop. Then predict with the same entry point as the validation array:

```
sbatch --array=<cell indices>%12 --time=14:00:00 --job-name=rcp_eicblind --export=ALL \
  --exclude=fc10713,fc10715,fc30557 \
  /project/def-maxwl/mforooz/onion_kit/RCP9/slurm/onion_eval/eic_val.sh \
  /project/def-maxwl/mforooz/onion_kit/RCP9 \
  /project/def-maxwl/mforooz/ONION/runs_rcp/rcp_combo_w2x_unet/seed0 <out dir>
```

  This is the validation array's submit line (sacct SubmitLine of 5761693), run with working directory `/project/def-maxwl/mforooz/ONION`. eic_val.sh reads validation cells. A blind run needs the blind cell list and the blind covariates (D57 used the pooled single-end rows for never-built tracks). Check its arguments before you submit. Price it first: the validation array cost 649.8 CPU-h for 43 counts (sacct, elapsed x CPUs).

## Validation predictions already delivered

- Array 5761693: 43 counts in `/project/def-maxwl/mforooz/ONION/runs_rcp/eic_val/rcp_combo_w2x_unet_s0/counts`. Recalibrated copies are in `..._s0_recal_wide/counts` (copy job 5761694).
- val3 6033930: C25M26, C34M02, C50M02 in `.../eic_val/val3/out/counts`. Use the val3 C25M26, not the older file in `s0/counts`.

## Never resubmit

5761693, 5761694, 6033929, 6033930, 6290210, 6291945-6291948, 5022624-5022627 (cancelled), 28737208/10/14/44/46/63/66/68/74 (stage), 28737209/11/15/45/47/64/67/69/75 (build), 28737212, 28737239 (signal), and every job listed in `cruxvault/INSTRUCTIONS/t370_recipe/run_state/longrun/LOG.md`.

## Deferred to the next session (D61)

- R7: two-step correction through the base view, retested on the final model.
- R11: robustness to covariate shift. The covariate-prompt test found that an unknown lab, platform or antibody does not widen the intervals (q92, `cruxvault/results/t370/recal_study/covariate_prompt_*.md`).
- R13: Max's headline metric, R2 on test tracks.
- R14: the gain from including covariates vs not.
- R15: real vs wrong input covariates.
- The PI's open questions q92 (which output covariates make sense as prompts) and q93 (how training can capture large peaks) are in the vault.

## Still the PI's

- Merge PR 63.
- The read-start retrain (priced 1,350-2,250 CPU-h). It waits.
