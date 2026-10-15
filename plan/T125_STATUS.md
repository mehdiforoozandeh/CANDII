# t125 status: the depth-offset count model and the counts → p map h

Orchestrator log for t125. Read `plan/T125_HANDOFF.md` first, then
`plan/ETA_OFFSET_AND_H_DESIGN.md`. The orchestrate plan is `.orchestrate/plan_t125.md` in this
worktree (git-excluded).

## State

| phase | state |
|---|---|
| 0 verify on Nibi | done 2026-10-11; formula forks with the PI |
| 1 build the count model | built and reviewed (offset, form A, model, train, score, aggregate) |
| 2 build the map h | built and reviewed (data side, formula, push-forward, learned h, train, score, aggregate); end-to-end smoke building |
| 3 smoke on Nibi | not started |
| 4 count runs (432) | not started |
| 5 h on the true X′ | not started |
| 6 h on f's NB, chain scores | not started |
| 7 aggregate | not started |
| 8 vault readings | not started |
| 9 lab deck | not started |

## Job ids

Every id goes here at submission.

- Kit K9: `/project/def-maxwl/mforooz/t125/code/K9`, `GIT_SHA` 643054089f34e4ef662cf52acc37c84bac7d0ca2
  (rsync of the worktree without .git, cruxvault, .orchestrate; wheels copied from t118 K6).
- 23797342 `t125_verifyK9`: acceptance criteria 1, 2, 4-8 on kit K9 (CPU, 8 cores, 32 GB, 3 h).
  Script and log: `/project/def-maxwl/mforooz/t125/verify/` (script not in the repo).
- 23797345 `t125_hctl`: control count cache, array 0-129%20 ->
  `/project/def-maxwl/mforooz/t125/ctl_cache`; logs `/project/def-maxwl/mforooz/t125/h/logs/`.
- Kit K10: `/project/def-maxwl/mforooz/t125/code/K10`, `GIT_SHA` 7e0f0f2fabfc28b00c40bdc44c2886483d2e8bbd
  (K9 + the test-tolerance commit; rsync as K9, wheels copied from K9). The run code is identical:
  `git diff --stat 643054089f 7e0f0f2 -- tools slurm src` is empty.
- 24065614 `t125_verifyK10`: criterion 1 without -x, criterion 2, and the twin-determinism test 3x on
  K10 and 3x on t118 ladder kit K8 (CPU, 8 cores, 32 GB, 3 h). No t118 kit on Nibi has `tests/`, so
  the K8 runs use a `$SLURM_TMPDIR` copy of K8 plus the test file from git at 33e09cb (K8's sha),
  kept in `/project/def-maxwl/mforooz/t125/verify/K8_tests_33e09cb`. Script and log:
  `/project/def-maxwl/mforooz/t125/verify/` (`verify_K10.sh`, `t125_verifyK10_24065614.out`).

## Nibi access

- 2026-10-11: `ssh -o BatchMode=yes nibi` exited 255 (`Permission denied
  (publickey,keyboard-interactive,hostbased)`). All Nibi work stops until the PI runs
  `hpc up nibi`.
- 2026-10-11 01:26 EDT: `ssh -o BatchMode=yes nibi` returns `l2.nibi.sharcnet`. Nibi work resumes.

## Phase 0 findings

Read-only check on Nibi, 2026-10-11. Source: `/project/def-maxwl/mforooz/t112_cf/products/<pid>/provenance.json`
(exact commands), `covariates.json`, `MANIFEST.tsv`, the pipeline SIF scripts, and MACS2 2.2.4 source.

- **DNase P:** ENCODE atac-seq-pipeline v2.2.3 in DNase mode (`encode_task_macs2_signal_track_atac.py`),
  **not** dnase-seq-pipeline v3.0.0. `macs2 callpeak -t TA -f BED -g hs -p 0.01 --shift -75 --extsize 150
  --nomodel -B --SPMR --keep-dup all`, then `bdgcmp -m ppois -S lines(TA)/1e6`. No control: λ = max(λ_BG, λ_10k
  of the treatment's own reads), λ_BG = 150·N/2.7e9. No Tn5 shift. extsize arms change smooth-win to 75 or 300.
- **ChIP P:** chip-seq-pipeline v2.2.2. `callpeak -t TA -c CTL -f BED -g hs -p 0.01 --nomodel --shift 0
  --extsize FRAGLEN --keep-dup all -B --SPMR`, then `bdgcmp -m ppois -S lines(TA)/1e6`. λ_local = max(λ_BG,
  λ_d, λ_1k, λ_10k); control scaled to the smaller set (`--scale-to small`). FRAGLEN from cross-correlation.
  ctlid none: λ = max(λ_BG, λ_10k of treatment). ratio arms: `--ratio V`, V = k·30M/N_ctl. Two ctldepth q0.25
  arms (C19M16, C19M22) have treatment > control, so MACS2 scales the treatment down there.
- **ppois (tested with the SIF's MACS2):** P = −log10 P(X > floor(obs)) at λ; strict tail; no pseudocount;
  5-decimal rounding; float32 bigwig.
- **25 bp binning:** P = base-pair-weighted **mean** of per-base −log10 p over the bin (0 where no interval).
  Counts = **read-span overlap**: +1 to every bin the aligned read touches (no shift, no extension, no strand),
  chr1–22 + chrX, from the same nodup tagAlign (30M ChIP, 50M DNase). pe arms: each mate is a line.
- **Inputs for h:** `covariates.json` has depth, fraglen, control.reads, knob_value (ratio V or control depth),
  read_length, run_type. MACS2's N is all tagAlign lines (incl. chrM, contigs), not the sum of counts25.

## PI decisions

- 2026-10-11: learned MLP h = 2 × 64, with the same training settings as the t118 g (optimizer, learning rate, epochs, early stopping on chr22).
- 2026-10-11, formula h (from the phase 0 forks):
  - DNase follows the recipe that ran (ATAC pipeline v2.2.3 in DNase mode). Its conflict with the
    "dnase-seq-pipeline v3.0.0" rule is noted, not fixed.
  - Pileup ≈ count × fraglen / (read_length + 24). Bin-only uses the bin; the fragment-window variant
    uses the mean count over the bins one fragment wide.
  - λ = the full MACS2 max of windows (fragment, 1 kb, 10 kb, background) from the binned control, scaled
    as MACS2 does (to the smaller set; ratio V in ratio arms). No control: max(background, 10 kb of the
    treatment); in the chain that window uses f's predicted mean.
  - One Poisson tail per bin, with the caller's rules: strict tail, floor of the observed value,
    no pseudocount, N = all tagAlign lines.

- 2026-10-11, plan forks:
  - Law pass on depth-only pairs only (42 across tracks); `--law-scope all` stays available.
  - Repair training: 1 h seed, 768 fits (64 f × 12 h versions); seed spread quoted from the true-X′ h.
    Upgrade to 3 seeds only if the Nibi smoke shows one fit under 20 min on 8 cores (ask the PI then).
  - Own-noise loss on f's NB: expected Gaussian NLL, Σ_k NB(k)·NLL_k.
  - Own-noise scoring: exact mixture CRPS on log(1 + P); push-only also gets log-scale CRPS; push-only keeps
    native-scale CRPS; native MSE of the mean for both.
  - Control read length = the treatment base's read length; the Nibi smoke reports the formula-vs-real-P error.
    **Superseded the same day:** the crop arms cropped the control too, so every control's read length is
    measured from the tagAlign that ran (read-only Nibi step).

- 2026-10-11: formula + affine h = formula + an affine map of all rung-5 inputs (formula included), zero at init.

- 2026-10-11: the no-covariates twin keeps the depth offset. Real vs twin shows what the covariates add beyond the fixed d′/d scaling; it is not directly comparable with the old grid's twin gap.

- 2026-10-11: rung 5 of the learned h gains the treatment and control log read lengths (21 inputs).

- 2026-10-11: the shuffle control shuffles depth too, as in t118, so the offset uses the shuffled target's d′.

- 2026-10-11: a comparison of two arms uses only the pairs finite in both; it reports how many pairs it dropped and the non-finite counts.

- 2026-10-11: the log-scale CRPS gets its own oracle-scaled / scale-error split: fit c on the log scale (shift log(1 + P) by c·ln2, i.e. scale 1 + P by 2^c), closed form for the pushed set and the mixture. The native split scales the whole pushed set by 2^c, as t118's p-space split.

## Log

- 2026-10-11: run started; Nibi down at start.
- 2026-10-11 01:30: Nibi up. Phase 0 reader and the planner dispatched. Chunk ids carry the prefix `t125-`.
- 2026-10-11: wave 0 merged (offset and task table, form A by level, eta model, h data side, formula h, exact push-forward, learned h). Control read lengths measured for 114 products (`tools/t125/hmap/control_read_length.tsv`). Wave 2 part 1 (eta training, eta scoring, SLURM part 1) building.
- 2026-10-11 (N1): kit K9 shipped to Nibi. Criterion 3: the 89 t118 files (tools/t118, slurm/t118,
  tests/test_t118_*, tests/fixtures) in K9 have the same git blob hashes as a0e6834, and
  `git diff --stat a0e6834` on those paths is empty. Verify job 23797342 and control-cache array
  23797345 submitted.
- 2026-10-11 23:17 EDT (N1): `ssh -o BatchMode=yes nibi` exited 255 during the queue poll. All Nibi
  work stopped until the PI runs `hpc up nibi`. State at 23:11 EDT: verify job 23797342 RUNNING
  (venv built, pytest started); control-cache array 23797345 tasks 0-18 COMPLETED (about 1 min
  each, `counts=built`), tasks 19-129 PENDING; 36 files in ctl_cache. Both jobs run on without
  the laptop. Still to read: the verify log's CRIT lines, sacct, and the criterion 9 count (114).
- 2026-10-15: kit K10 shipped to Nibi; verify job 24065614 submitted (criterion 1 without -x,
  criterion 2, twin-determinism test on K10 and on t118 K8).
