---
title: "D3: dataset update with DNase-seq in the headline"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# D3: dataset update with DNase-seq in the headline

## 1. What the PI asked

- `long-horizon notepad.md`:463 (ruling 38, PI's words): "freeze whatever we can get in the next 6h and then send out reliable watchdogs to wait for the remaining runs and as soon as they are finished you update the dataset with notes".
- `peeling the onion.md`:105: "we are updating our EIC dataset -- to train the new CANDI".
- `long-horizon decisions.md`:66 (D11b): "**Accessibility-only (second table).** Only DNase-seq or ATAC-seq is given, and the other assays are imputed. Cells with no accessibility track are left out of this regime."
- `long-horizon decisions.md`:70 (D8): the EIC model "trains only on the EIC training experiments".

Success criteria served: 1 (augmented EIC data) and 3 (evals). Audit rows: A11, A12, B14 (the accessibility-only table).

## 2. Question

Build the next tables version with every tail row that has verified, so that DNase-seq becomes imputation-scorable and the accessibility-only regime covers more validation targets. Rebuild the evaluation set on it, and say which comparisons move to the new version.

## 3. State today

- `eic_v1` (frozen 2026-11-13 14:30 UTC) holds 249 of 267 training and 42 of 45 validation tracks; DNase-seq is not scorable (1 of 3 validation tracks); 22 of 42 validation targets have no accessibility-only item (`long-horizon notepad.md`:470-473).
- Tail watch, Nibi `ONION/eic/tail/TAIL_STATUS.tsv` (checked 2026-11-16T14:49Z, read by this agent), 28 rows: 17 verified (rows 10, 37, 46, 52, 57, 63, 84, 124, 273, 337, 381, 388, 389, 390, 391, 394, 395). 11 not verified: row 56 (validation DNase-seq, C50; signal pass 27124285_56 PENDING, 7-day limit), and 10 training DNase-seq rows: 7 NODE_FAIL in the Nibi outage (113, 211, 216, 232, 384, 385, 387), 1 TIMEOUT (280), 2 FAILED (132 Java heap, 396 other).
- Validation DNase-seq: row 57 (C34) is now verified, so 2 of 3 validation tracks exist; the third (row 56) needs its signal pass.
- No update version is built: Nibi `tables/` holds only `eic_v1` (read 2026-11-16). Watch job 27483519 is PENDING.

## 4. Prior leads

- Scorable-set rule (notepad L51): an assay is imputation-scorable with at least 3 validation tracks from at least 3 distinct cells. Prediction: DNase-seq enters the headline only when row 56 verifies.
- `results/t305/LAYERS_REAL.md`: DNase-seq has the largest cell-type layer (23%, interval [0.0117, 0.0176]). Prediction: adding DNase-seq to the headline raises the weight of cell-type-specific signal in the imputation term.
- Plan Amendment 6 (`.orchestrate/plan.md` L754-840, chunk D6u): defines the update procedure, the `--superset-of` check and `versions.tsv`. Use it as written.

## 5. Work

1. Resubmit each unverified training tail row once, from kit KF4, with `ONION_TIME=7-00:00:00` and `ONION_MEM=192000M` (Amendment 6 B.5; notepad L514-515). The 7 NODE_FAIL rows failed on the outage, not on their own cause.
2. Wait for row 56's signal pass. If its counts are verified and only the signal is missing, record whether the count-space selection metric can use it (the signal is needed for signal-space scores only).
3. Run chunk D6u: build `eic_v1_<k>` with every verified row plus the D1 covariate columns (one version for both, so models retrain once), `check_dataset.py --superset-of` (relaxed for the D1 columns, see D1 risk 3), `CHANGES_<k>.md`, `versions.tsv`.
4. Rebuild the evaluation set on the new version (256 regions, same seed), and the 16-region screening set. Record both hashes.
5. Rebuild the baselines on the new evaluation set.
6. Submit the second-experiment rows whose DNase-seq partner is now verified (D2 step 1).
7. Write the notes: what entered, what is still absent with its cause, and the new scorable list.

## 6. How the result answers the question

Acceptance checks:
- `check_dataset.py --tables eic_v1_<k> --superset-of eic_v1` exits 0 (with only the D1 columns allowed to change).
- `versions.tsv` holds one line for `eic_v1_<k>` with its tables hash, eval-set hash and change note.
- The new `scorable.tsv` lists DNase-seq if row 56 verified; else the file states why not.
- The count of validation targets without an accessibility-only item is below 22, or the note explains why it is not.

## 7. Dependencies, cost, risks

- Depends on D1 (the covariate columns go into the same version). D2 and D4 depend on it.
- Cost: about 250 CPU-h + 19 GPU-slice-h + about 32 CPU-h per update (`.orchestrate/plan.md` L846, Amendment 6), plus the resubmitted DNase-seq rows (one row of 753 M reads ran over 1 day in `bwa sampe`; `results/t183/TAIL.md`).
- Risk 1: every model comparison is valid within one dataset version and one evaluation-set version only (notepad L465). The nine ablations (A1) and the final round (A9) run on `eic_v1`. The PI must say which comparisons are rerun on the new version (see M1).
- Risk 2: the slow DNase-seq paired-end alignment can need more than 3 days; row 56's signal pass has a 7-day limit.

## 8. Done when

`eic_v1_<k>` and its evaluation sets pass their checks, the change note lists every tail row as in or absent with its cause, and the scorable list states whether DNase-seq is in the headline.

## 9. Specification as worked (item agent, 2026-11-17 overnight)

### 9.1 State today, checked against the artifacts (corrections to section 3)

Read on Nibi on 2026-11-17 between 10:30 and 10:50 UTC (`TAIL_STATUS.tsv` and `TAIL_DONE` checked 08:13:49Z, `sacct`, the build logs, `tables/`, `eval/`).

- Row 56 (validation DNase-seq, C50M02) is not pending. Its 7-day task 27124285_56 started 2026-11-17T03:08:48Z and failed after 7 s: `OSError: [Errno 122] Disk quota exceeded` on `/scratch/mforooz/onion/work/ENCSR437AYW/base/encode_metadata.json` (`ONION/logs/build_27124285_56.out`). The task was the full build (align step), not a signal pass. Its cause is the full Nibi `/scratch`, not the row. Only its `se` arm exists (`ENCSR437AYW.pooled.uniform.se`).
- Row 396 failed out of memory: `MaxRSS` 196,581,636 KB at its 192000M limit (`sacct -j 27105143_396`); its log stops in a pipeline task with no error line. The class "other" in `TAIL_STATUS.tsv` hides this.
- The 7 NODE_FAIL training rows (113, 211, 216, 232, 384, 385, 387) are rerun by D4's script (`slurm/onion_data/closeout_d4_submit.sh` batch 1, and the Fir port). D3 does not submit them (Log 2026-11-17T05:15Z: "never on both"). D4's script still says row 56 is rebuilt by task 27124285_56; that is now false, so D3 owns row 56.
- Of the 17 verified tail rows, 7 are already in `eic_v1` with all their measurement rows (10, 37, 46, 63, 84, 124, 273). New since the freeze: rows 52 (6 of its 11 rows), 57, 337, 381, 388, 389, 390, 391, 394, 395 (ten experiments, 72 measurement rows; count of directories with `provenance.json` minus rows in `eic_v1/measurements.tsv`).
- Nibi `tables/` holds `eic_v1`, `eic_v1_1` (do not use) and `eic_v1_2`. `eval/eic_v1_1` and `eval/eic_v1_2` exist on Fir only. The 9 base-recipe runs on `eic_v1_2` are trained and scored (Log 2026-11-17).
- The watch is over: `TAIL_DONE` was written 2026-11-17T08:13:49Z (17 verified, 11 not).
- `eic_v1/scorable.tsv` (Nibi `eval/eic_v1/`): DNase-seq has 16 training tracks and 1 validation track with halves (C03M02), so it is neither imputation-scorable nor correction-scorable ("1 validation tracks with a never-trained non-subset pair (need 2)").

### 9.2 How DNase-seq enters the headline today (code)

- The scorable rule: an assay is imputation-scorable with at least 3 validation tracks with halves, from at least 3 cells, and at least 5 training tracks; correction-scorable with at least 2 validation cells that each have a never-trained, non-subset pair (`tools/onion_eval/evalset.py:146-147`, `scorable_rows` at `:390-425`; the lists go into `evalset.json` `scorable` at `:598-599`).
- The scorer reads these lists from the eval set (`tools/onion_eval/score.py:2859`). Every metric's macro is the mean over the scorable assays only (`metric`, `:502`); a non-scorable assay's values go to `non_scorable_per_assay` (`:510`) and never enter a macro. The composite's imputation term is that macro (`:2969`), the correction term is the macro over the correction-scorable assays (`:2970`, `_correction_metrics_of` `:908`), and the calibration term uses the imputation-scorable list (`:2976`); `composite` at `:2977`. `select_rule.py` recomputes the composite from these macros.
- So today DNase-seq enters no headline term on any version. It is reported apart, per assay.
- "Halves" means pseudoreplicate count files (`_halves`, `evalset.py:205-208`: `has_pseudoreps` and a counts path). The signal files are not needed: a validation track whose counts and halves verify is scorable in count space (the selection space `counts_log1p`, `score.py:253`) even if its signal pass is missing; only the signal-space scores need `signal.h5`.

### 9.3 A trap found: the table build reads the work folders that leave Nibi scratch

`build_tables.py build` reads three kinds of small file under `/scratch/mforooz/onion/work`: QC files (`qc_values`), `READSET.json` beside each half tagAlign (`SplitSignature`, the split signature of every designed pair) and `NOT_APPLICABLE` markers (`build_list.row_state`). That folder is archived to nearline tonight and its removal waits for the PI. Measured on `eic_v1`: 447 of 2,740 rows have `nsc` filled, and 0 of them have it in `meta.json` (all 447 come from the work folder). Without these files, a build with `--superset-of eic_v1` changes 447 older rows, moves designed pairs to `pairs_shared_reads.tsv`, and is refused; the unit test shows both effects on the test tree. The align-time check `align_views.py verify --recount-chrom chr21` reads the half tagAligns in the same folder.

Fix (code, branch `chunk/onion-CO-D3`): `build_tables.py snapshot-work` writes one gzip JSON of these small files (no tagAligns); `check-work-snapshot` proves that the snapshot alone gives what the disk gives; `build --work-snapshot <file>` reads a file from the snapshot when it is not on the disk (a file on the disk wins). Without the option the outputs are byte-identical to the previous code (checked on the test tree, all 7 hashed table files, with and without `--superset-of`).

So the item needs, from the leaving folders: the small files of `onion/work` (kept by the snapshot) and, for the verify step, the half tagAligns of the ten new experiments (the verify must run before the removal). It does not need `onion/fastq`: the 4 reruns stage their FASTQs again from ENCODE (`stage_fastq.sh`), which needs a writable `/scratch`.

### 9.4 Question, steps and expectations

Question: which tail rows enter the next version `eic_v1_3`, does DNase-seq enter the headline on it, and how many validation targets lack an accessibility-only item?

| step | what | where | before the work folder leaves? |
|---|---|---|---|
| 1 preserve | snapshot + check | `ONION/CO-D3/work_snapshot_2026-11-17.json.gz` | yes |
| 2 verify | chr21 recount of the 10 new experiments | logs in `ONION/CO-D3/logs` | yes |
| 3 candidate | build `--superset-of eic_v1` on what is verified now, its eval set and scorable list | `ONION/CO-D3/tables`, `ONION/CO-D3/eval` (not a version) | yes (reads the work folder) |
| 4 reruns | rows 56, 132, 280, 396 once each, 7 days, kit KF4 (rule B.5) | Nibi build arrays | after `/scratch` is under its quota |
| 5 version | staging build (with the snapshot) + `fill_covariates` join/apply + `check_dataset` against `eic_v1_2` and `eic_v1` + eval sets (256 and 16 pair regions) + regions assertion + oracle floor | `tables/eic_v1_3`, `eval/eic_v1_3` | no (snapshot) |
| 6 baselines | 4 baselines x 2 eval sets | `eval/eic_v1_3/baselines{,_p16}` | no |
| 7 second experiments | the held D2 rows whose partner verified: 2023 (row 394), 2040 (389), 2067 (390), 2077 (395), 2084 (381); 8 rows still wait (partners 280, 132, 232, 113, 387, 384, 216, 211) | D2's submission (D2 step 1) | needs `/scratch` |
| 8 notes | `CHANGES_eic_v1_3.md`, `versions.tsv`, the scorable list | vault `results/closeout_D3/` | no |

All of steps 1 to 6 are parts of `slurm/onion_data/closeout_d3_submit.sh` (dry run by default).

Expectations, written before any run:
- Candidate (step 3), on what is verified now: the superset check passes (0 older rows changed). DNase-seq has 2 validation tracks with halves (C03, C34), so it stays not imputation-scorable. It becomes correction-scorable if C34's designed arms give a never-trained, non-subset pair, as C03's do: then DNase-seq enters the correction term of the composite. Validation targets with no accessibility-only item: 19 of 43 (22 of 42 minus the 4 C27 targets that row 337 covers, plus the new C34 DNase-seq target, which by design has no such item).
- After all reruns verify: DNase-seq is imputation-scorable (3 validation tracks from 3 cells, at least 16 training tracks), and the count without an accessibility-only item is 15 of 44 (rows 113, 132, 211 cover C24 x2, C18 x2, C47 x1; C50 DNase-seq adds 1).

| outcome | conclusion |
|---|---|
| superset check exits 0 and 0 older rows change | the update mechanism holds; build the version |
| a superset or hash problem | stop; report the lines; the foreman calls the planner |
| DNase-seq correction-scorable on the candidate | DNase-seq enters the correction term on `eic_v1_3`; its imputation term waits for row 56 |
| row 56 verifies in its one 7-day rerun | DNase-seq imputation-scorable on the last version (10 assays in the imputation term) |
| row 56 fails its 7-day rerun | DNase-seq stays out of the imputation term for good (stop rule B.5); `non_scorable_assays.tsv` and `CHANGES` say why |

Controls and traps: the regions of every eval set must equal `eval/regions_v1/regions.json`; `score.py` refuses a run whose `tables_sha256` differs from the eval set's, so no comparison crosses versions; rows still queued or running are never touched; a row that fails its 7-day rerun is closed (no third run); `check_dataset` with `--covariate-columns` ignores only the four D1 columns.

### 9.5 Cost (CPU-hours)

- Tonight (steps 1 to 3, Nibi, within the 20 CPU-hour allowance): preserve 1 CPU x at most 1 h; verify of one experiment 1 CPU x at most 3 h; candidate 16 CPUs x at most 1 h. Worst case 1 + 3 + 16 = 20.
- Verify of the other 9 experiments: 9 x 1 CPU x at most 3 h = at most 27 (measured rate of the first one in `results/closeout_D3/`).
- Reruns: 4 rows x 16 CPUs x at most 168 h = at most 10,752 allocated CPU-hours. Expected about 16 x 24 h x (3 + 3 + 4 + 3) days = 4,992 (rows 56, 132, 396 needed more than 1 to 2 days, row 280 more than 3 days).
- Version: 16 CPUs x at most 4 h = at most 64 (Amendment 6 estimate about 32).
- Baselines: 4 at 16 pair regions about 8 each (D1's jobs on Fir: 34, 32, 21 and 19 min at 16 CPUs) = about 28; 4 at 256 regions about 23 each (overnight brief) = about 92. Total about 120.

### 9.6 Open decisions

- Which version the final model and the 3 base recipes train on. The base recipes were retrained on `eic_v1_2` (9 runs, scored). The ruling "retrain once" holds only if `eic_v1_3` waits for the reruns and then becomes the last version; the base recipes would then be retrained a second time (9 runs) for the final comparison.
- Row 396's rerun memory: the script asks 384000M (Nibi CPU nodes have 766 GB), above the 192000M of rule B.5, because the last run died at 192000M.
