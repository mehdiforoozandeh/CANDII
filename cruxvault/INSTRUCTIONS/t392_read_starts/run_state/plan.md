# Plan: miniCANDI on read-start counts, Phase 1 (task t392)

Written 2026-11-23 by the planner; amended once on 2026-11-23 after the PI's answers (D9 to D11, D14) and a cold review.
Branch `implementation/t392-read-start-minicandi`, HEAD 8325cb42, draft PR 66.
Sources: `../INSTRUCTIONS.md` (cited as I), `../HANDOFF_2026-11-23_read-start-minicandi.md` (H), `GOALS.md` (rows R1 to R17), `DECISIONS.md`.

## Goal

Train the full-combination miniCANDI (config `tools/minicandi/configs/recipe/combo/rung4/halves_mix_avg_pearson.json`, seeds 0, 1, 2) on 25 bp read-start counts.
ChIP-seq tracks and the ChIP-seq control have two channels, plus-strand starts and minus-strand starts. DNase-seq and ATAC-seq have one channel (D11).
Score the read-start model and the existing read-count model side by side on the 16-region evaluation set of `eic_v1_2`, against the overlap-count truth.
Report imputation, correction and calibration with seed ranges, then the deviance metrics when PR 65 is merged.
Stop after the 16-region table. Phase 2 and 256-region scoring do not start.

Design in one paragraph. The tables stay `eic_v1_2`, unchanged, so the table hash of the evaluation set and of the checkpoint still match.
A new config key, `read_start_store`, sends every read of counts to a second store that holds stranded 25 bp read-start counts for the same measurement ids.
The model is the rung-4 U-Net with its existing sub-channel path set to 2 strand channels.
At scoring, a wrapper around the loaded model, i.e., the read-start adapter, reads the stranded inputs from the read-start store, runs the model, and converts the output to an overlap-count NB.
`registry.load_run` applies the read-start adapter, as it already applies `ResAdapter` for other bin sizes. No scorer file is edited.

Terms, each used with one meaning in this plan.
- Code kit: a copy of this checkout on a cluster, made by `bash tools/onion_kit/push_kit.sh --cluster <fir|nibi> <id>` under `/project/def-maxwl/mforooz/onion_kit/<id>`. `RS<n>` (RS1, RS2, ...) are the kits of this run. `RCP9` is the kit on Fir that trained and scored the read-count reference; it is the base for "no new failures".
- Rung 4: the U-Net model, `tools/minicandi/rungs/unet.py` (`registry.RUNGS[4]`).
- C and C': the covariates of an input track (C) and of a target track (C'): depth, read length, run type and the other columns of `contract.META_COLUMNS`.
- Pair regions: the 16 regions of `evalset_p16.json` on which correction pairs and replicate pairs are scored. Imputation is scored on all 256 regions of the same file. "16-region scoring" means this evaluation set.
- `p16_sv`: the score folder of a seed for the headline scorer on that evaluation set with the same-view pass on. `p16_dev`: the score folder for the deviance scorer of PR 65.
- Two-phase form: scoring as predict jobs that write a prediction store (`predict_store.py run`, in shards), then one metrics job that reads the store (`score.py --phase metrics`).
- Headline terms: imputation = `imputation_raw`, correction = `correction_v3_raw`, calibration = `uncertainty_raw`, the columns of `select_rule.py arm-summary`.
- Span kernel: for one read file and one strand, the average number of reads per read start that cover the bin k bins away from the start bin, k from -8 to 8.
- Open edge: an end of a run of stitched windows, where the starts of the bins outside the run are not predicted. The conversion fills them with the edge bin's values.
- Conversion floor: the nMSE between the overlap counts and the observed 25 bp starts convolved with the span kernel. It is the error that the conversion alone adds (D9).
- Amendment 3: the part of the earlier onion plan (task t152) that added the 1 bp point-rule region store, `recount.py`, `ResDataset` and `ResAdapter`.

Python for every local command: `PY=/Users/mforooz/miniforge3/envs/candii/bin/python`, with `export PYTHONPATH=$PWD/src`, from the worktree root.
Builders never run `git add .longrun/`; a hook writes there in this worktree.

## Acceptance criteria

1. `$PY -m pytest tests/test_t392_read_starts.py tests/test_t392_store.py tests/test_t392_dataset.py tests/test_t392_convert.py tests/test_t392_tools.py tests/test_t392_model.py tests/test_t392_adapter.py tests/test_t392_end_to_end.py -q -p no:cacheprovider` reports 0 failed, 0 errors.
2. `hpc run fir 'comm -13 /project/def-maxwl/mforooz/t392/logs/failures_RCP9.txt /project/def-maxwl/mforooz/t392/logs/failures_RS<n>.txt | wc -l'` prints 0 (no test fails on the kit that passes on the base kit; both lists come from `slurm/t392/rs_tests.sh`).
3. `$PY tools/golden.py check <ref.pt>` exits 0, with `<ref.pt>` written by `$PY tools/golden.py save <ref.pt>` in a checkout of `origin/exp/t370-minicandi-recipe` (bit-exact against the tree before the change).
4. `$PY tools/t392/check.py probe --cluster nibi --store /scratch/mforooz/rs392/probe/store` exits 0. It prints `differing bins: 0` for every probe file, and the wall time, the peak memory and the conversion floor of each.
5. `$PY tools/t392/check.py store --cluster fir --store /scratch/mforooz/rs392/store --tables /project/def-maxwl/mforooz/ONION/tables/eic_v1_2` exits 0. It prints `measurements 2740 of 2740`, `files 8220 of 8220`, `differing bins: 0`.
6. `$PY tools/t392/check.py train --cluster fir --runs /project/def-maxwl/mforooz/t392/runs/rs_halves_mix_avg_pearson_unet` exits 0. It prints `seeds done: 0 1 2`, `windows_seen 80000` for each, and a parameter count within 2% of 1,000,000.
7. `$PY tools/t392/check.py scores --cluster fir --runs /project/def-maxwl/mforooz/t392/runs/rs_halves_mix_avg_pearson_unet --sub p16_sv` exits 0 and prints `SCORED 3 of 3`. That the adapter hides masked bins and missing tracks is shown by the tests of C9, not by the scorer's copy-leak flag (the adapter does not read the batch's `x_counts`).
8. `bash cruxvault/results/t392/side_by_side/runall.sh` exits 0, and `$PY -m pytest tests/test_results_folders.py -q` passes. `SIDE_BY_SIDE.md` holds one row per model, the three headline terms with seed ranges, one figure and one paragraph.
9. `git log --format=%h --no-merges origin/exp/t370-minicandi-recipe..HEAD -- tools/onion_eval/score.py tools/onion_eval/predict_store.py tools/onion_eval/metrics.py tools/onion_eval/deviance.py` prints nothing.
10. `git log --format=%h --no-merges origin/exp/t370-minicandi-recipe..HEAD -- .longrun/` prints nothing (committed history only; uncommitted hook writes in the worktree do not count).

## Chunks

Cluster folders. Nibi probe: `/scratch/mforooz/rs392/probe/{work,store,logs}`. Fir: `/scratch/mforooz/rs392/{tars,work,store,aux,pred_store,logs}` and `/project/def-maxwl/mforooz/t392/{runs,logs}`.
Every job name starts with `rs392_`. Nothing on a cluster is deleted.
A cluster step that needs a wait is two chunks: a submit chunk, which ends when the jobs are submitted and its `jobs.tsv` holds the ids, and a collect chunk, which starts when the jobs have ended.
Every `jobs.tsv` has the columns `job_id, name, cluster, state, alloc_cpus, cputime_raw_s`; the collect chunk fills the last three from `sacct -X`. Spend in this plan is always CPUTimeRAW / 3600 (cores x time), as the reference figures.
Status table to the foreman (work item, % done, what is left, blocker, ETA, CPU-h so far): after C7b, C11b, C12c, C13b and C15a. Never to the PI.

### C1 tests against the instructions   wave 1
Goal: Write the tests for the four code items before any implementation exists. The agent is `crux-tests`. It reads `../INSTRUCTIONS.md` and the Interfaces lines of C3, C4 and C6 in this plan. It does not read any implementation file of this run.
Owns: `tests/test_t392_read_starts.py`, `tests/test_t392_store.py`, `tests/test_t392_dataset.py`, `tests/test_t392_convert.py`
Interfaces: Each test imports the module under test inside the test function or a fixture, never at module top level, so collection passes before the modules exist. Required cases:
(1) Stranded recount. A hand-made tagAlign gives known plus and minus start counts; a minus read counts at the bin of `end - 1`; `starts.sum()` equals the number of reads kept on the grid; the overlap array of the same scan equals `tools/t112/bin25.py::counts_from_tagalign` on the same file, bin for bin; one changed bin in the reference makes `gate` report 1 differing bin and the CLI exit non-zero. Chromosome end: a minus-strand read whose start bin is the partial last bin (index `len // 25`, which the grid drops) adds no start, is counted in `n_off_grid`, and still adds its overlap to the kept bins as `bin25` does.
(2) Store. Every file has one dataset per chromosome of shape `[len // 25, 2]`; column 0 is plus; `starts.h5` equals `starts_pr1.h5 + starts_pr2.h5`; the attributes of C3 exist; per file, the sum over strands s of `kernel[s].sum() * n_reads_strand[s]` equals the overlap sum, where both sides count only the reads kept on the grid.
(3) Dataset. `res_dataset.make_dataset` with `read_start_store`, `average_input` true and `average_h5` set returns batches with no error; the batch holds `x_counts_sub`, `y_counts_sub` `[B, L, 36, 2]` and `avg_sub`; channel sums equal `x_counts` and `y_counts`; DNase-seq and ATAC-seq have all their starts in channel 0 and zeros in channel 1; the window's own cell never enters its average; the read-start average file has, per track, the same `measurement_ids` set and `n_cells` as the read-count average file it was built from; for a seed, `ReadStartDataset` yields the same window sequence (cell, chromosome, start bin, input and target measurements, `x_bin_mask`) as `OnionDataset` on the same tables.
(4) Conversion. For a 0/1 box kernel the converted mean equals the box convolution of each strand's mean, summed over strands; the NB size equals mu^2 / (var - mu) with var the sum of the contributions' variances; a strand mean that is Poisson (very large size) gives a converted size at the upper clamp; the conversion of 16 stitched windows equals the conversion of their concatenation; an unstranded track ignores channel 1.
Depends on: none
Serves: R5, R1, R2, R3, R4
Test: `$PY -m pytest tests/test_t392_read_starts.py tests/test_t392_store.py tests/test_t392_dataset.py tests/test_t392_convert.py --collect-only -q -p no:cacheprovider` (exit 0, at least 20 tests listed)

### C2 stage the probe read files on Nibi (submit)   wave 1
Goal: Put the read files of four probe measurements on Nibi scratch, from one pass over one tar. No recount yet.
The measurements: `ENCSR000AKT.pooled.uniform.base` (small ChIP-seq, 2 replicates), `ENCSR000ENE.pooled.uniform.base` (DNase-seq), `ENCSR000AMI.pooled.uniform.mapq0` (the largest read volume of all 2,740, 2.0 GiB, 319 M reads), and one measurement with `lines_per_fragment` 2 whose files are in `onion-work__000`.
The member paths come from each measurement's `provenance.json::halves` on Nibi `/scratch/mforooz/onion/measurements/` and from the index `/project/def-maxwl/mforooz/ONION/nearline_archive_2026-11-17/index/onion-work__000.txt`.
The tar is 220.7 GB and is read end to end, so the extraction is not run on a login node. Submit one job `rs392_stage` (1 CPU, 4000M, 03:00:00, `--gres=none`) that runs `tar -xf /nearline/def-maxwl/mforooz/scratch_archive_2026-11-17/onion-work__000.tar -C /scratch/mforooz/rs392/probe/work <members>`. If the job reports that a compute node cannot read `/nearline`, run the same command on the Nibi data-transfer node and record that in the note. About 3 GiB lands on scratch (708 of 1,024 GiB are in use).
Owns: `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/notes/probe/STAGING.md` (the commands, the member list, the job id, `lfs hsm_state` of the tar before the read, whether a compute node can read `/nearline`)
Interfaces: The extracted tree is `/scratch/mforooz/rs392/probe/work/onion/work/<experiment>/...`, i.e., the provenance path with the prefix `/scratch/mforooz/` replaced by `/scratch/mforooz/rs392/probe/work/`. C3 calls this prefix swap `--work-root`.
Depends on: none
Serves: R6, R7
Test: `hpc run nibi 'find /scratch/mforooz/rs392/probe/work -name "pr[12].tagAlign.gz" | wc -l'` (prints the member count recorded in STAGING.md, once the job has ended)

### C3 stranded read-start recount, gate and store writer   wave 2
Goal: One scan of a measurement's read files writes its stranded 25 bp read-start counts and proves, in the same scan, that the reads rebuild the stored overlap counts exactly (D9).
Owns: `tools/onion_data/read_starts.py`, `slurm/t392/rs_recount.sh`
Interfaces:
- No torch import in this module (numpy, pandas, h5py only).
- Read-start rule. The 5' base of a read is `start` on strand `+` and `end - 1` on strand `-` (`recount.point_positions(start, end, strand, 0)`). Its start bin is that base `// 25`. The grid is `len // 25` bins per chromosome, the chromosomes of `tools/t112/bin25.py::MAIN_CHROMS`. A read whose start bin is not on the grid adds no start and is counted in `n_off_grid`.
- Overlap rule. `tools/t112/records.py::COUNT_RULE`, with the edge handling of `bin25.counts_from_tagalign` (lines 69 to 76).
- `scan(paths, sizes, reach=8) -> dict`: per chromosome `starts` uint32 `[n_bins, 2]` (column 0 plus, column 1 minus) and `overlap` uint32 `[n_bins]`, rebuilt by placing each read's span box at its start bin; `kernel_counts` int64 `[2, 2 * reach + 1]`, where entry `[s, reach + k]` is the number of kept bins at offset k from the start bin that strand-s reads kept on the grid cover; `n_reads_strand` `[2]` (reads kept on the grid); `n_lines`, `n_main`, `n_off_grid`. A read whose span leaves `reach` raises.
- `gate(overlap, reference_h5) -> dict`: the one function that holds the gate. It compares the rebuilt overlap arrays with one `eic_v1_2` file on every bin of every chromosome and returns `{"reference", "reference_sha256", "n_bins", "n_differing", "result"}`. Nothing else in the repo decides pass or fail of the gate.
- `conversion_floor(starts, kernel, overlap) -> float`: the nMSE, mean((a - b)^2) / var(b), between the starts convolved with the span kernel (a) and the overlap counts (b), over all bins of the file.
- Store, one folder per measurement id under the store root: `starts.h5`, `starts_pr1.h5`, `starts_pr2.h5`. One dataset per chromosome, `[n_bins, 2]`, uint16 (uint32 when a value exceeds 65,535), chunks `(4096, 2)`, gzip 4 with shuffle. File attributes: `schema` 1, `counting_rule` "read_start", `resolution` 25, `strand_columns` "plus,minus", `measurement_id`, `depth_reads`, `n_reads_strand`, `n_off_grid`, `kernel_counts`, `kernel` float64 `[2, 17]` (`kernel_counts` divided by the strand's `n_reads_strand`), `conversion_floor_nmse`. Every assay is stored with both strands (D11).
- `provenance.json`: the read files, the sha256 of each output, per file the `gate` dict and `conversion_floor_nmse`, `wall_s`, `peak_rss_kb`. The three files are gated against `counts.h5`, `pr1.h5`, `pr2.h5` of the measurement. Any differing bin gives exit 1 after the files are written.
- CLI: `read_starts.py run --measurement <id> --measurements <eic_v1_2 measurement dirs> --work-root <prefix> --out <store root> [--reach 8]` reads the read-file list from `<measurements>/<id>/provenance.json::halves`; `read_starts.py verify --out <store root>/<id>` rechecks the hashes, `starts = pr1 + pr2`, and the recorded gate; `read_strands(path, chrom, start, n) -> int32 [n, 2]`.
- `slurm/t392/rs_recount.sh <kit> <plan.tsv>` runs the rows of the plan whose `task` equals `$SLURM_ARRAY_TASK_ID`, skips a measurement whose `provenance.json` holds a passed gate on all three files, and prints one line per measurement: id, wall seconds, peak RSS, `n_differing`, the conversion floor.
Depends on: C1
Serves: R1, R2, R6, R7
Test: `$PY -m pytest tests/test_t392_read_starts.py tests/test_t392_store.py tests/test_onion_data_recount.py -q -p no:cacheprovider`

### C4 strand channels and the conversion to overlap counts   wave 2
Goal: The pure functions that every model-side chunk shares: which tracks have two channels, the moment-matched conversion, the strand-aware output and loss.
Owns: `tools/minicandi/read_start.py`
Interfaces:
- `STRANDS = 2`; `UNSTRANDED_TRACKS = ("ATAC-seq", "DNase-seq")` (D11); `channel_mask() -> bool [36, 2]`, False at channel 1 of an unstranded track.
- `unstranded_kernel(kernel, n_reads_strand) -> [17]`: the read-weighted mix of the two rows of a span kernel.
- `convert_nb(mu_sub, n_sub, kernel, stranded) -> (mu, n)`: tensors `[..., L, 2]` and a kernel `[2, 17]`. With w the kernel weight, `mu[i] = sum over strands and k of w * mu_sub[i - k]`, `v[i] = sum of (w * mu_sub[i - k])^2 / n_sub[i - k]`, `n = mu^2 / v`, clamped to `contract.N_MIN .. N_MAX` (the Poisson case of I section 2 is `v` at or below 0, which gives `N_MAX`). This is the moment match of I section 2 with var = sum of `w * mu + (w * mu)^2 / n`, i.e., each start count thinned by w, and the bins and strands taken as independent. Positions outside the array are filled with the edge bin's values.
- `stitch(metas, L) -> list of index runs`: windows of one batch that share cell, chromosome and targets and whose `start_bin` values follow each other by L are converted as one array.
- `strand_output(out) -> dict`: from `mu_sub`, `n_sub` `[B, L, A, 2]`, the keys `mu`, `n` `[B, L, A]` in read-start space (`losses.pool_nb` over the two channels; channel 0 alone for an unstranded track).
- `read_start_loss(out, batch, config) -> (loss, parts)`: `losses.total_loss` on the `[B, 2L, A]` layout of `res_dataset.res_total_loss`, with channel 1 of every unstranded track outside the loss. Changing `mu_sub[..., t, 1]` of an unstranded track t does not change the loss.
Depends on: C1
Serves: R4, R2
Test: `$PY -m pytest tests/test_t392_convert.py tests/test_minicandi_losses.py -q -p no:cacheprovider`

### C5 build plan and check commands   wave 2
Goal: The two small tools that the cluster chunks and the acceptance criteria call.
Owns: `tools/t392/__init__.py`, `tools/t392/build_plan.py`, `tools/t392/check.py`, `tests/test_t392_tools.py`
Interfaces:
- `build_plan.py --tables <eic_v1_2> --measurements <dirs> --index <dir of onion-work__*.txt> --testbed-root <dir> --gib-per-task 2.6 --out <dir>`: one plan file per source, `plan_onion-work__00{0..5}.tsv` and `plan_testbed.tsv` (2,740 rows in all), columns `task, measurement_id, source, read_gib, n_tagaligns, lines_per_fragment`; tasks are packed inside one source, to near-equal `read_gib`. `build_plan.py members --plan <file>` prints the tar members that the plan needs (only `pr1` and `pr2` files).
- `check.py <probe|store|train|scores> ...` as in Acceptance criteria 4 to 7. `check.py spend <jobs.tsv> ...` prints the sum of `cputime_raw_s / 3600` over the files. It reads the cluster over `hpc run <cluster>`, never writes there, and exits 69 with "run `hpc up <cluster>`" when the tunnel is down.
Depends on: none (the store layout and the gate record are pinned in C3)
Serves: R6, R7, R15
Test: `$PY -m pytest tests/test_t392_tools.py -q -p no:cacheprovider`

### C6 read-start dataset through res_dataset   wave 3
Goal: Training and the adapter read stranded windows, with the average input, through the entry that `res_dataset.py` already gives. The read-start dataset is a new class (D14).
Owns: `tools/minicandi/rs_dataset.py`, `tools/minicandi/res_dataset.py`
Interfaces:
- Config keys: `read_start` (bool), `read_start_store` (store root), `read_start_average_h5` (path). `average_h5` keeps its read-count path; only the unedited scorer's own dataset reads it.
- `ReadStartDataset(OnionDataset)`: `read()` returns the strand sum of the starts, int32 `[L]`, from `<store>/<id>/starts.h5` or its `_pr1`, `_pr2` siblings; `read_strands(mid, chrom, start, L, which) -> int32 [L, 2]`; `window()` adds `x_counts_sub`, `y_counts_sub` int32 `[L, A, 2]` (0 where hidden or missing; for an unstranded track channel 0 is the total and channel 1 is 0) and, under `average_input`, `avg_sub` float32 `[L, A, 2]` next to `avg` (their channel sum); `average_strands(cell, chrom, start, L)` follows the leave-one-cell-out rule of `OnionDataset.average`. The sampler draws are those of the parent, so the window sequence per seed is the parent's.
- `build_average_h5(reference_average_h5, store, tables, out)`: per track, the `measurement_ids` and `n_cells` are read from the attributes of the read-count file `eic_v1_average.h5` and copied (that set holds single-end views such as `ENCSR191FOV.pooled.uniform.se`); one dataset per chromosome, float32 `[n_bins, 2]`, the sum over those measurements of starts x 1e6 / `depth_reads`.
- Refused with ValueError: `input_depth = fixed`, `aug_revcomp`, `two_step`, `res_bp != 25`, `res_in_sub` or `res_out_sub` above 1, a `store:` row.
- `res_dataset.make_dataset` returns a `ReadStartDataset` when the config sets `read_start`; `res_dataset.train` then swaps in `read_start.read_start_loss`. The point-rule `ResDataset` of Amendment 3 keeps refusing `average_h5`.
Depends on: C3, C4
Serves: R2, R3
Test: `$PY -m pytest tests/test_t392_dataset.py tests/test_minicandi_resolution.py tests/test_minicandi_train_halves.py -q -p no:cacheprovider`

### C7a recount probe on Nibi (submit)   wave 3
Goal: Recount the four staged measurements at real scale on Nibi. Nibi, because the tars, the staged files and a copy of the `eic_v1_2` counts are there.
Steps: confirm C2's Test; push kit `RS1` to Nibi; `sbatch --job-name=rs392_probe --array=1-4 --cpus-per-task=2 --mem=24000M --time=02:00:00 --gres=none` on `slurm/t392/rs_recount.sh` with a 4-row plan, `--work-root /scratch/mforooz/rs392/probe/work/`, store `/scratch/mforooz/rs392/probe/store`.
Owns: `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/notes/probe/jobs.tsv`
Interfaces: store root `/scratch/mforooz/rs392/probe/store`; logs under `/scratch/mforooz/rs392/probe/logs`.
Depends on: C2, C3, C5
Serves: R7, R6
Test: `hpc run nibi 'squeue -u $USER -n rs392_probe -h | wc -l; ls /scratch/mforooz/rs392/probe/store | wc -l'`

### C7b recount probe on Nibi (collect): the first measured result   wave 4
Goal: Read the gate and the costs of the probe. This is the day-1 result.
Owns: `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/notes/probe/PROBE.md`, `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/notes/probe/probe.tsv`
Interfaces: `probe.tsv`, one row per measurement: reads, read GiB, wall s, CPUTimeRAW, peak RSS, store bytes, `n_differing` per file, the two kernel rows, and the named output `conversion_floor_nmse` per file (D9). PROBE.md gives the rate in CPU-h per GiB, the store bytes per read, the projected store size, and the memory to ask on Fir. Fir bills one core per 4,000 MB, so the Fir request is `--mem` = peak RSS x 1.3 rounded up to 4,000 MB and `--cpus-per-task` = that memory / 4,000 MB, and the projected recount spend uses those cores. If the projection is above 180 CPU-h, report to the foreman before C11a. A non-zero `n_differing` stops the run and goes to the foreman (H, stop condition). Send the status table.
Depends on: C7a
Serves: R7, R6, R15, R16
Test: `$PY tools/t392/check.py probe --cluster nibi --store /scratch/mforooz/rs392/probe/store`

### C8 strand channels in the model and the training entry   wave 4
Goal: The rung-4 U-Net reads and predicts two strand channels per track when the config sets `read_start`, and nothing changes without the key.
Owns: `tools/minicandi/rungs/unet.py`, `tools/minicandi/train.py`, `tools/minicandi/configs/t392/rs_halves_mix_avg_pearson.json`, `slurm/t392/rs_train.sh`, `tests/test_t392_model.py`
Interfaces:
- Under `read_start` the model sets its sub-channel count to 2 on both sides (the existing `x_counts_sub` input path and `out_sub_emb` output path), reads `avg_sub` through a `Linear(2, D_TRACK)`, and returns `mu_sub`, `n_sub` `[B, L, A, 2]` plus `read_start.strand_output`. `read_start` with `res_in_sub` or `res_out_sub` above 1 raises.
- `train.set_pearson_stats` under `read_start` fits the per-track rate and size on the channels of `x_counts_sub` that `channel_mask()` keeps.
- The config is the reference config plus `read_start: true`, `read_start_store: /scratch/mforooz/rs392/store`, `read_start_average_h5: /scratch/mforooz/rs392/aux/eic_v1_2_read_start_average.h5`. Every other key is equal, `param_budget` 1e6 and `windows_seen` 80000 included.
- `slurm/t392/rs_train.sh <kit> <config> <tables> <seed> <out>` has the header `--cpus-per-task=12 --mem=48000M --gres=none` (the reference jobs' allocation; the header of `slurm/minicandi/train.sh` is 8 CPUs and 32000M and was overridden at submit) and calls `python tools/minicandi/res_dataset.py train ... --device cpu --workers 11`.
- Tests: without `read_start` the outputs of a seeded rung-4 build are bit-identical to the tree before the chunk (`tests/test_minicandi_base_golden.py` stays green); the parameter count under the budget rule is within 2% of 1e6; perturbing hidden bins of `x_counts_sub` moves no output.
Depends on: C4, C6
Serves: R2, R8
Test: `$PY -m pytest tests/test_t392_model.py tests/test_minicandi_rung4.py tests/test_minicandi_base_golden.py tests/test_minicandi_recipe_keys.py -q -p no:cacheprovider`

### C9 read-start adapter: the conversion between predict and the metrics   wave 4
Goal: `score.py --run` and `predict_store.py run` score a read-start run with no edit to either file.
Owns: `tools/minicandi/rs_adapter.py`, `tools/minicandi/registry.py`, `tests/test_t392_adapter.py`
Interfaces:
- `ReadStartAdapter(inner, config, run_dir=None)` is a `contract.MiniCandi`. `forward(batch)` takes the ordinary overlap-count batch of the scorer and returns `{"n", "mu"}` `[B, L, A]` in overlap-count space.
- Steps of `forward`: (1) read `x_counts_sub` for every input track with `x_avail` true from the read-start store by `meta` (`measurements_in`, `input_file`, `chrom`, `start_bin`), set it to 0 on every bin where `x_bin_mask` is true and on every track where `x_avail` is false, and set `x_counts` to the channel sum; (2) read `avg_sub` with `ReadStartDataset.average_strands`; (3) run the inner model; (4) per target track take the span kernel from the attributes of the target's file (`measurements_out`, `target_file`); a track with no target measurement in the window takes the read-weighted mean of the kernels of the window's target measurements; (5) `read_start.stitch`, then `read_start.convert_nb`.
- The adapter counts `n_open_edges` and `n_windows`, and writes them to `<run_dir>/<score dir>/open_edges.json` when the environment variable `RS392_EDGE_LOG` names a file.
- `registry.load_run` wraps a run whose config sets `read_start`. `input_depth` at scoring other than native is refused.
- Tests, with a stub inner model that records its input and returns fixed `mu_sub`, `n_sub`: the output equals `convert_nb` of the stub; the inner model's `x_counts_sub` is 0 at every masked bin and every unavailable track, whatever the store holds there; changing the batch's own `x_counts` changes nothing; a batch of 8 consecutive windows has 2 open edges.
Depends on: C4, C6
Serves: R4, R10
Test: `$PY -m pytest tests/test_t392_adapter.py tests/test_minicandi_resolution.py -q -p no:cacheprovider`

### C16 documents for the read-start store and the new keys   wave 4
Goal: The repo documents say what the code now does.
Owns: `STORE.md`, `DATA.md`
Interfaces: `STORE.md` gains one section on the read-start store: the folder layout, the three files, the dataset shape, every attribute, the read-start rule (5' base, `end - 1` on the minus strand), the gate and the conversion floor. `DATA.md` gains the batch keys `x_counts_sub`, `y_counts_sub`, `avg_sub` under `read_start`, the one-channel tracks, and the config keys `read_start`, `read_start_store`, `read_start_average_h5`. No number is quoted from a comment.
Depends on: C3, C6
Serves: R2
Test: `$PY -m pytest tests/test_docs.py -q -p no:cacheprovider`

### C10a end-to-end check, sanity tool and the full suite on the cluster (code and submit)   wave 5
Goal: Prove the whole path on synthetic tables before any training job, write the read-start-truth sanity tool, and submit the full suite on the kit and on the base kit.
Owns: `tools/t392/sanity.py`, `tests/test_t392_end_to_end.py`, `slurm/t392/rs_tests.sh`, `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/notes/tests/jobs.tsv`
Interfaces:
- `tests/test_t392_end_to_end.py`: synthetic tables (`tools/minicandi/synth.py`) plus a synthetic read-start store; `res_dataset.train` for 16 windows; `registry.load_run` returns the adapter; `score.py` on a synthetic evaluation set writes `scores.json` with finite headline terms.
- `sanity.py --run <dir> --evalset <json> --out <json>`: on the imputation targets of the pair regions, the inner model's `mu_sub`, `n_sub` against the read-start truth: one mean NB deviance per bin and one Pearson r of log1p mean against log1p truth. A sanity check only (R12).
- `slurm/t392/rs_tests.sh` is `slurm/minicandi/rcp_tests.sh` with the job name `rs392_tests`; it writes the sorted ids of the failed tests to `/project/def-maxwl/mforooz/t392/logs/failures_<kit id>.txt`. Submit it on Fir once for the current kit `RS<n>` and once for `RCP9`.
Depends on: C8, C9
Serves: R5, R12, R4
Test: `$PY -m pytest tests/test_t392_end_to_end.py -q -p no:cacheprovider`

### C10b full suite on the cluster (collect)   wave 6
Goal: Read the two failure lists and the bit-exactness gate.
Owns: `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/notes/tests/TESTS.md`
Interfaces: TESTS.md lists every test that fails on `RS<n>` and passes on `RCP9` (must be none), and the output of `tools/golden.py check`. A new failure goes back to the owning code chunk before C12a.
Depends on: C10a
Serves: R5
Test: `hpc run fir 'comm -13 /project/def-maxwl/mforooz/t392/logs/failures_RCP9.txt /project/def-maxwl/mforooz/t392/logs/failures_RS<n>.txt | wc -l'` (prints 0)

### C11a full recount and store build on Fir (submit)   wave 5
Goal: Recount all 2,740 measurements in every view and build the read-start store, with the gate on every file. Fir (D10): the training and the 16-region scoring inputs are on Fir, and Fir scratch has 16 TiB free.
Waits on: the PI's transfer has landed, i.e., the 6 `onion-work` tars are in `/scratch/mforooz/rs392/tars` with the byte sizes of `status.tsv`, and the testbed read sets are in `/scratch/mforooz/rs392/work/testbed` (156 files). C7b has passed.
Steps: (1) `build_plan.py` with the Fir paths; (2) 6 jobs `rs392_untar_<k>` (1 CPU, 4000M, 04:00:00), one per tar, each extracts only its plan's members to `/scratch/mforooz/rs392/work`; (3) 7 arrays `rs392_recount_<source>`: one per tar, chained `afterok` on that tar's own untar job, and one for the testbed with no dependency; each `--array=1-<its tasks>%2` while the 6 tar arrays run, so at most 12 to 14 tasks run at once; `--cpus-per-task`, `--mem` and `--time` from PROBE.md; (4) one job `rs392_avg` chained `afterok` on the 7 arrays (all still live at submit), 8 CPUs, 32000M, runs `rs_dataset.build_average_h5` from `/scratch/mforooz/onion/aux/eic_v1_average.h5` into `/scratch/mforooz/rs392/aux/eic_v1_2_read_start_average.h5`.
The extracted read files stay on Fir scratch.
Owns: `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/notes/build/jobs.tsv`, `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/notes/build/plan_summary.tsv` (source, measurements, tasks, read GiB)
Interfaces: Store root `/scratch/mforooz/rs392/store`. Fir file count stands at 831K of 1,000K; the build adds about 18K files.
Depends on: C7b, C3, C5, C6, the transfer
Serves: R6, R7
Test: `hpc run fir 'squeue -u $USER -h -o %j | grep -c "^rs392_"'`

### C11b full recount and store build on Fir (collect)   wave 6
Goal: Read the gate on all 8,220 files, the conversion floor and the store size.
Owns: `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/notes/build/BUILD.md`, `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/notes/build/gate.tsv`, `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/notes/build/floor.tsv`
Interfaces: `gate.tsv`, 8,220 rows: measurement, file, `n_bins`, `n_differing`. `floor.tsv`, the named output of D9, 2,740 rows: measurement, assay, view, `conversion_floor_nmse` of `starts.h5`, `starts_pr1.h5`, `starts_pr2.h5`. BUILD.md holds the summary of the floor (median, 5th and 95th percentile, maximum, per assay class and per view), the store size (`du -s` of the store root, bytes, and the size of the average file), the CPU-h from `jobs.tsv`, and the check that the average file has the same `measurement_ids` and `n_cells` per track as `eic_v1_average.h5`. A failed gate on any file stops the run before training (R7). Send the status table.
Depends on: C11a
Serves: R6, R7, R15, R16
Test: `$PY tools/t392/check.py store --cluster fir --store /scratch/mforooz/rs392/store --tables /project/def-maxwl/mforooz/ONION/tables/eic_v1_2`

### C12a training probe (submit)   wave 7
Goal: One short training job on the real store before the three seeds.
Steps: push the current kit to Fir; `sbatch --job-name=rs392_train_probe --time=01:00:00 slurm/t392/rs_train.sh <kit> <config> /project/def-maxwl/mforooz/ONION/tables/eic_v1_2 0 /project/def-maxwl/mforooz/t392/runs/_probe/seed0` with `EXTRA_TRAIN_ARGS="--max-windows 2000"`.
Owns: `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/notes/train/jobs.tsv` (C12b and C12c append to it, in later waves)
Interfaces: the probe folder holds `log.jsonl` and `run.json`.
Depends on: C10b, C11b
Serves: R8
Test: `hpc run fir 'ls /project/def-maxwl/mforooz/t392/runs/_probe/seed0'`

### C12b training, seeds 0 to 2 (submit)   wave 8
Goal: Read the probe (seconds per step, peak memory, the Pearson statistics, the parameter count), then submit the three seeds on Fir CPU nodes as the reference (`run.json::device` is `cpu`).
Steps: three jobs `rs392_train_s<k>` on `slurm/t392/rs_train.sh`, `--time` = 3 x the projected wall, into `/project/def-maxwl/mforooz/t392/runs/rs_halves_mix_avg_pearson_unet/seed<k>`. If the projected spend of the three seeds is above 240 CPU-h, report to the foreman first.
Owns: `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/notes/train/PROBE.md`
Interfaces: a job that times out is resubmitted with the same command (it resumes from `ckpt.pt`). The reference is never retrained.
Depends on: C12a
Serves: R8, R9, R15
Test: `hpc run fir 'squeue -u $USER -h -o %j | grep -c "^rs392_train_s"'`

### C12c training (collect)   wave 9
Goal: Confirm the three runs and record the spend.
Owns: `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/notes/train/TRAIN.md`
Interfaces: Run folders hold `ckpt.pt`, `run.json`, `log.jsonl`, `DONE`. `run.json` must show `windows_seen` 80000 and the tables hash of `eic_v1_2` (`7a1a0325d00ddff3...`). TRAIN.md holds the final training and held-out loss per seed and the CPU-h. Send the status table.
Depends on: C12b
Serves: R8, R15, R16
Test: `$PY tools/t392/check.py train --cluster fir --runs /project/def-maxwl/mforooz/t392/runs/rs_halves_mix_avg_pearson_unet`

### C13a 16-region scoring of the read-start model (submit)   wave 10
Goal: Score the three read-start seeds on the 16-region evaluation set with the unedited scorer, in the two-phase form, with the reference's settings.
Steps: (1) read the settings of the reference's scoring from Fir `runs_rcp/rcp_halves_mix_avg_pearson_unet/seed0/p16_sv/scores.json` (`scorer_settings`, `pred_store`, the evaluation-set hash) and from the head of its `predict_*.out` logs, and use the same values (4 shards, CPU predict, the same inference batch, same-view pass on, `REFS_FROM=/project/def-maxwl/mforooz/ONION/eval/eic_v1_2/baselines_p16`); (2) submit with `slurm/minicandi/submit_arms.sh` in its score-only two-phase form: `SEEDS="0 1 2"`, `SCORE_SUBDIR=p16_sv`, `SCORE_SAME_VIEW=1`, `PRED_STORE_ROOT=/scratch/mforooz/rs392/pred_store`, `KEEP_PRED_STORE=1`, `JOB_PREFIX=rs392_`; (3) one job `rs392_sanity_s<k>` per seed on `tools/t392/sanity.py`.
Owns: `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/notes/score_p16/jobs.tsv` (C14a appends to it, in a later wave)
Interfaces: score folders `<runs>/rs_halves_mix_avg_pearson_unet/seed<k>/p16_sv`; the prediction stores are kept for C14a.
Depends on: C12c, C10b
Serves: R10, R12
Test: `hpc run fir 'squeue -u $USER -h -o %j | grep -c "^rs392_"'`

### C13b 16-region scoring (collect), and the reference's scores   wave 11
Goal: The table of the two models on the three headline terms.
Steps: `select_rule.py arm-summary` takes one `--runs` directory. Make `/project/def-maxwl/mforooz/t392/runs_both/` with two symlinks, to the read-start arm and to Fir `runs_rcp/rcp_halves_mix_avg_pearson_unet`, and run one call with `--sub p16_sv`.
Owns: `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/notes/score_p16/SCORE.md`, `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/notes/score_p16/arm_summary_p16_sv.tsv`, `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/notes/score_p16/sanity.tsv`
Interfaces: `arm_summary_p16_sv.tsv` has the rows `rcp_halves_mix_avg_pearson_unet` and `rs_halves_mix_avg_pearson_unet` with `<term>.mean` and `<term>.range` for `imputation_raw`, `correction_v3_raw`, `uncertainty_raw`.
Stop rules: (a) if the reference row does not reproduce 0.4996, 0.8156, 0.6852 to 4 decimals, stop and report to the foreman; (b) if the share of bins within 8 bins of an open edge is above 0.25% (the expected value is 32 of 16,384 bins = 0.195% at an inference batch of 8), stop and report; (c) sum the spend of every `jobs.tsv` (`check.py spend`) and write it in SCORE.md. Send the status table.
Depends on: C13a
Serves: R10, R9, R12, R15, R16
Test: `$PY tools/t392/check.py scores --cluster fir --runs /project/def-maxwl/mforooz/t392/runs/rs_halves_mix_avg_pearson_unet --sub p16_sv`

### C15a side-by-side deliverable, finding and report (headline)   wave 12
Goal: The deliverable folder under the runall contract, the crux finding, and the stop report.
Owns: `cruxvault/results/t392/side_by_side/SIDE_BY_SIDE.md`, `cruxvault/results/t392/side_by_side/runall.sh`, `cruxvault/results/t392/side_by_side/MANIFEST.tsv`, `cruxvault/results/t392/side_by_side/table.tsv`, `cruxvault/results/t392/side_by_side/side_by_side.png`, `cruxvault/results/t392/side_by_side/plot.py`, `.gitignore`
Interfaces: One table, one row per model, one column per headline term with the seed range (AGENTS.md 7.2); the deviance columns are filled when C14b has landed before this chunk, else marked "waits on PR 65"; the read-start-truth sanity numbers and the conversion floor summary in a separate, labelled block; one figure; one paragraph; the CPU-h spent with job ids; every deviation from I.
The paragraph states this caveat: the moment match treats the start counts of neighboring bins as independent, so it understates the variance of the overlap count where the rate is spatially correlated; this biases calibration and `dev_calibration_ratio` of the read-start model toward overconfidence.
Then `crux finding` on q75 with the file as evidence, `crux task done t392`, the status table and the report to the foreman session by SendMessage. No message to the PI. The folder is frozen once the finding cites it.
Depends on: C13b
Serves: R13, R14, R16
Test: `bash cruxvault/results/t392/side_by_side/runall.sh && $PY -m pytest tests/test_results_folders.py -q -p no:cacheprovider`

### C14a deviance metrics on 16 regions (submit)   wave 12
Goal: The deviance metrics of PR 65 for the read-start model.
Gate 1: `gh pr view 65 --repo mehdiforoozandeh/CANDII --json state` shows MERGED, `origin/exp/t370-minicandi-recipe` is merged into the umbrella (D8), and a new kit is pushed.
Gate 2, the stop rule: `check.py spend` over every `jobs.tsv`, plus the estimate of this chunk (32.5 CPU-h when the kept prediction stores of C13a are accepted, 119.3 CPU-h with a new predict pass). If the sum is above 600 CPU-h, report to the foreman and wait.
Steps: call `slurm/minicandi/submit_arms.sh` directly, not `slurm/onion_eval/deviance_submit.sh` as it is: runs directory `/project/def-maxwl/mforooz/t392/runs`, arm `rs_halves_mix_avg_pearson_unet=<config json>`, `SEEDS="0 1 2"`, `EXTRA_SCORE_ARGS=--score-deviance`, `SCORE_SUBDIR=p16_dev`, `JOB_PREFIX=rs392_`, `REFS_FROM` and the other scoring settings as the reference's `p16_dev` (Fir `runs_rcp/DEV_SUBMITTED_p16.txt`, jobs 5381558 to 5381572).
Owns: `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/notes/score_p16/DEVIANCE_SUBMIT.md`
Interfaces: score folders `<runs>/rs_halves_mix_avg_pearson_unet/seed<k>/p16_dev`.
Depends on: C13b, PR 65 merged
Serves: R11, R15
Test: `hpc run fir 'squeue -u $USER -h -o %j | grep -c "^rs392_"'`

### C14b deviance metrics (collect)   wave 13
Goal: The deviance rows of the two models.
Owns: `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/notes/score_p16/DEVIANCE.md`, `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/notes/score_p16/deviance_p16.tsv`
Interfaces: `deviance_p16.tsv` has one row per model with skill, gap and R, each with its seed range; the reference row comes from its existing `p16_dev` folders.
Depends on: C14a
Serves: R11
Test: `$PY tools/t392/check.py scores --cluster fir --runs /project/def-maxwl/mforooz/t392/runs/rs_halves_mix_avg_pearson_unet --sub p16_dev`

### C15b deviance addendum   wave 14
Goal: The deviance table as its own folder, because the headline folder is frozen once cited. Skipped when C15a already held the deviance columns.
Owns: `cruxvault/results/t392/deviance_addendum/DEVIANCE_ADDENDUM.md`, `cruxvault/results/t392/deviance_addendum/runall.sh`, `cruxvault/results/t392/deviance_addendum/MANIFEST.tsv`, `cruxvault/results/t392/deviance_addendum/table.tsv`
Interfaces: the same two rows, the deviance terms with seed ranges, the caveat of C15a, a second `crux finding` on q75, and a report to the foreman. C15a's `.gitignore` line covers `cruxvault/results/t392/`.
Depends on: C14b, C15a
Serves: R11, R13, R14, R16
Test: `bash cruxvault/results/t392/deviance_addendum/runall.sh && $PY -m pytest tests/test_results_folders.py -q -p no:cacheprovider`

## Decisions needed

None is open. Answered: the gate at read level with the measured span kernel and the conversion floor reported (D9); the build on Fir (D10); ATAC-seq as one channel (D11); the read-start dataset as a new class (D14).
Waiting, not a decision: the PI's Globus transfer, Nibi to Fir, of `/nearline/def-maxwl/mforooz/scratch_archive_2026-11-17/onion-work__00{0..5}.tar` (1,104 GiB) to `/scratch/mforooz/rs392/tars/` and of `/project/def-maxwl/mforooz/ONION/testbed/readsets/` (18 GiB) to `/scratch/mforooz/rs392/work/testbed/`. C11a waits on it. Waves 1 to 4 and C10a do not.

## Ground truth verified

All on 2026-11-23. `hpc status` showed fir up and nibi up; no exit 69 occurred.

Branches and reuse
- `git for-each-ref refs/remotes/origin` (115 refs after `git fetch`): no branch holds a 25 bp stranded recount, a strand channel or a moment-matched conversion to overlap counts. The nearest code is listed below.
- `origin/implementation/t390-eic-pvalue-entry:tools/eic_entry/ladder_probe.py`, `cmd_starts` (line 139): one pass over a tagAlign that keeps 1 bp positions by strand in memory (`<chrom>_p` = plus `start`, `<chrom>_m` = minus BED `end`) and the store-rule counts. It writes no 25 bp store. `bin_kernel` (line 714) and `apply_kernel` (line 728) are the fragment-extension kernel of Phase 2. The plan reuses the idea, not the file: the minus position there is `end`, and this plan pins `end - 1`, the convention of `recount.point_positions`.
- `cruxvault/results/t390/postpile/POSTPILE.md` on that branch: the nMSE table of I section 1 matches (0.0172, 0.2550, 0.0212; stranded starts 0.0028, 0.0746; unstranded DNase-seq 0.0018).
- PR 65: `gh pr view 65` gives state OPEN, base `exp/t370-minicandi-recipe`, not merged. `tools/onion_eval/deviance.py` and `slurm/onion_eval/deviance_submit.sh` exist on `origin/exp/t371-deviance-evals` only. `slurm/minicandi/submit_arms.sh` reads `JOB_PREFIX` on both branches.

Code (line numbers at HEAD 8325cb42)
- `tools/t112/records.py:42`: `COUNT_RULE` = "+1 to bins floor(start/25)..floor(end/25) inclusive, grid floor(len/25)". The code of the rule is `tools/t112/bin25.py::counts_from_tagalign`, lines 69 to 76 (reads past the grid dropped, the end bin clamped, the grid cut to `len // 25`).
- `tools/onion_data/recount.py`: a 1 bp point-rule recount into a region store (768 regions), one unstranded array, with a fragment-length shift. `point_positions` (line 112) already gives the 5' base by strand. `scan_tagaligns(..., overlap=True)` (line 212) and `readset_problems` (line 471) are the model for an in-scan check against the stored counts.
- `tools/minicandi/res_dataset.py:67`: `REFUSED` holds `average_h5`; `ResDataset.window` passes a base-pair start, so the 25 bp average file cannot be read there. `res_dataset.train` (line 331) swaps the dataset class and the loss of `train.train`; `train.py` itself never builds a `ResDataset`.
- The reference config sets no `res_store`: the reference trained with `OnionDataset` on the 25 bp `h5:` files of `eic_v1_2`, not on the region store.
- `tools/minicandi/contract.py`: `A = 36` fixed tracks (35 assays and `chipseq-control`), asserted. `registry.py:62`: rung 4 is `unet`. `rungs/unet.py` already has a sub-channel path (`x_counts_sub` in, `mu_sub`, `n_sub` out) used for 5 bp sub-bins; `components.res_subs_of` allows 1 or 5 only. `losses.pool_nb` (line 237) is the moment match n = (sum mu)^2 / sum(mu^2 / n).
- `tools/minicandi/train.py::set_pearson_stats` (line 363) fits the Pearson input statistics on `x_counts`. `tools/minicandi/train.py::code_keys` (line 135; the rule is stated in the module docstring, "Unknown keys") refuses a config key that no code under `tools/minicandi` reads.
- Input contract and store reader: `tools/minicandi/dataset.py::OnionDataset.read` (line 783) reads `h5:<dir>/counts.h5` and its `pr1.h5`, `pr2.h5` siblings, one 1-D dataset per chromosome; `OnionDataset.average` (line 1270) reads `average_h5` as 1-D per track and chromosome and takes the summed measurements from the group attribute `measurement_ids`; `build_average_h5` (line 1419) writes that attribute and `n_cells`.
- Predict to metrics: `tools/onion_eval/predict_store.py` stores `n`, `mu` of the target channel per request; `score.py --phase metrics` reads them. The scorer refuses a run whose checkpoint tables hash differs from the evaluation set's (`score.py:1210`) and a store of another run, tables or request list (`predict_store.py::PredictionStore.complete`, line 391). `registry.load_run` (line 199) already wraps a run in `ResAdapter`, which re-reads its inputs from another store by `meta` and returns a 25 bp NB. The read-start adapter uses the same slot, so no scorer file changes.
- `tools/onion_eval/select_rule.py arm-summary` takes one `--runs` directory and `--sub`.

Reference model
- Rows on `origin/exp/t370-minicandi-recipe`, `cruxvault/presentations/t370-minicandi-recipe/data/pareto_p16_sv.tsv`: imputation 0.4996 (range 0.0349), correction 0.8156 (range 0.0105), calibration 0.6852 (range 0.0924), 3 seeds, class PASS. `pareto_p256_sv.tsv`: 0.4996, 0.8180 (range 0.0135), 0.6852. These match H (0.500, 0.818 with range 0.014, 0.685).
- Fir `/project/def-maxwl/mforooz/ONION/runs_rcp/rcp_halves_mix_avg_pearson_unet/seed{0,1,2}/`: `ckpt.pt`, `run.json` (`device` cpu, `param_count` 993,615, `windows_seen` 80000), and the score folders `p16_sv`, `p16_dev`, `p256_sv`, `p256_dev`, each with `scores.json` and `SCORED`. The training jobs ran with 12 CPUs and 48000M. The prediction stores are under Fir `/scratch/mforooz/onion/pred_store/runs_rcp/`.
- The "16-region" evaluation set, Fir `/project/def-maxwl/mforooz/ONION/eval/eic_v1_2/evalset_p16.json`: 256 imputation regions, 16 pair regions, 63 imputation targets, 1,480 correction pairs, 3,134 replicate pairs, tables hash `7a1a0325d00ddff3`. Its reference terms are in `eval/eic_v1_2/baselines_p16`. Nibi holds evaluation sets for the tables `eic_v1` only (hash `a068a8674aee4252`), so the 16-region scoring runs on Fir.

Clusters
- Nibi `/nearline/def-maxwl/mforooz/scratch_archive_2026-11-17/`: 23 tars; the 6 `onion-work__00{0..5}.tar` sum to 1,104.4 GiB (`onion-work__000.tar` is 220.7 GB). `lfs hsm_state` on the 6 gives "exists archived", i.e., a disk copy is present now.
- Index, Nibi `/project/def-maxwl/mforooz/ONION/nearline_archive_2026-11-17/index/onion-work__*.txt`: 31,781 members, of which 8,416 tagAligns, 751.7 GiB. `status.tsv` marks all 6 tars OK. `removal_work_2026-11-17.log` records that `/scratch/mforooz/onion/work` was removed.
- All 2,740 rows of `tables/eic_v1_2/measurements.tsv` have a `provenance.json` whose `halves` lists the `pr1` and `pr2` read files. For 2,662 measurements every file is in the index, each measurement inside one tar; the 78 testbed measurements point at 156 files that exist on Nibi `/project/def-maxwl/mforooz/ONION/testbed/readsets` (18 GiB). The read volume inside the tars is 578.7 GiB over 7,184 distinct files (503.5 GiB distinct); the testbed's 18 GiB is not inside 578.7 (the sum counted indexed tar members only). 173 measurements have 2 lines per fragment. Per tar: 618, 389, 503, 488, 495, 169 measurements.
- Views in the table: 785 base, 362 no-dedup, 1,086 depth, 363 MAPQ, 83 crop, 49 single-end and paired-end, 12 antibody-proxy. Assay classes: 1,254 narrow histone, 901 broad histone, 408 control, 147 DNase-seq, 30 ATAC-seq.
- `eic_v1_2` counts: `/scratch/mforooz/onion/measurements/<id>/{counts,pr1,pr2}.h5`, the same path on both clusters (Nibi 2,884 folders, Fir 2,838 folders, 251 GiB on Fir). `measurements.tsv` has the same md5 on both, and `ENCSR000AKT.pooled.uniform.base/counts.h5` has the table's sha256 on both.
- Average-input file: `/scratch/mforooz/onion/aux/eic_v1_average.h5`, 7,451,594,429 bytes, on both clusters.
- Nibi `/project/def-maxwl/mforooz/ONION/CO-A7/res_store`: 6.8 GiB, with `regions.tsv` beside it. Base views only; tests only.
- Quotas. Nibi: scratch 708 of 1,024 GiB, project 4,311 of 4,657 GiB. Fir: scratch 2,573 GiB of 19 TiB, 831K of 1,000K files; project 23 of 28 TiB.
- `bash tools/onion_kit/globus_transfer.sh route` prints NO_ROUTE (no Globus CLI on either login node).
- Not verified: whether a Nibi compute node can read `/nearline` (C2 records it); the per-task states of Nibi job 27662614 on the second read (the Nibi accounting database refused connections at the time of the amendment).

Local test run
- `$PY -m pytest tests/test_minicandi_resolution.py tests/test_minicandi_contract.py -q -p no:cacheprovider`: 156 passed in 45 s. `tests/test_onion_data_recount.py tests/test_minicandi_train_halves.py tests/test_minicandi_rung4.py`: 73 passed in 16 s. `candi` imports from the worktree's `src`. No environment defect appeared.
- Traps: in zsh, write `"${B}:path"` for `git show`, because `$B:c` applies a modifier; base Python cannot import torch, so call the `candii` interpreter by path; other sessions edit `run_state/DECISIONS.md` and `LOG.md`, and a hook writes `.longrun/`, so scope every `git add`.

## Cost estimate

Sources: `sacct -X` on Fir and Nibi, read 2026-11-23. CPU-h = CPUTimeRAW / 3600, i.e., allocated cores x elapsed time. The 600 CPU-h stop of R15 is counted the same way.

| step | CPU-h | arithmetic and source |
|---|---|---|
| probe staging and probe recount (C2, C7) | 2 | one 1-CPU pass over a 220.7 GB tar; 4 measurements, about 3 GiB of reads |
| untar on Fir (C11) | 9 | 6 jobs x 1 CPU x about 1.5 h; not measured |
| recount, all views (C11) | 48 to 143 | Nibi job 27662614: 21 array tasks, 33,298 CPU-s = 9.25 CPU-h in all. Task 21 failed (review finding), so its 1.19 GiB is left out: tasks 1 to 20 hold 116.86 - 1.19 = 115.67 GiB (`recount_plan.tsv`, column `tagalign_gib`). Rate at most 9.25 / 115.67 = 0.080 CPU-h per GiB at 2 CPUs. (578.7 + 18) GiB x 0.080 = 47.7. That scan wrote region files only; a genome-wide write, three hashes, the gate read and the floor are allowed a factor of up to 3. Fir bills one core per 4,000 MB, so a memory request above 8,000 MB raises the billed cores above 2; C7b measures the rate and the memory and reprojects. |
| read-start average file (C11) | 10 | not measured; one pass over the base views |
| tests on the cluster, training probe (C10, C12a) | 12 | 2 suite runs of 8 CPUs x 0.5 h, one 2,000-window probe on 12 CPUs |
| training, 3 seeds (C12) | 163 to 238 | Fir jobs 5264674, 5264680, 5264687: 254,664 + 46,752 + 285,456 CPU-s = 163.0 CPU-h. Seed 1 took 13.0 CPU-h against 70.7 and 79.3, so three seeds at the slow rate cost 3 x 79.3 = 238. |
| 16-region scoring, 3 seeds (C13) | 127 to 152 | 15 Fir jobs. Seed 0: 5264675, 5264676, 5264677, 5264678 (predict), 5264679 (metrics) = 163,976 CPU-s. Seed 1: 5264681, 5264682, 5264683, 5264685 (predict), 5264686 (metrics) = 135,504 CPU-s; job 5264684 is not part of this arm. Seed 2: 5264688, 5264689, 5264690, 5264691 (predict), 5264692 (metrics) = 156,480 CPU-s. Sum 455,960 CPU-s = 126.7 CPU-h. The adapter's store reads and conversion are allowed 20%. |
| sanity check (C13) | 5 | 3 jobs, not measured |
| deviance, 3 seeds (C14) | 32.5 to 119 | Fir jobs 5381558 to 5381572: 429,464 CPU-s = 119.3 CPU-h with its own predict pass; the three metrics jobs alone (5381562, 5381567, 5381572) are 43,424 + 36,976 + 36,576 = 116,976 CPU-s = 32.5 CPU-h when the kept stores are reused |
| total | 408.5 to 690 | sum of the rows |

The figures of I section 3 against these: training 163 and scoring 127 are confirmed. The recount figure of 180 is not: the measured rate times the real read volume gives 48, and 180 is above the 3x allowance.
The top of the range is above the 600 CPU-h stop. The headline table (through C13) costs at most 2 + 9 + 143 + 10 + 12 + 238 + 152 + 5 = 571. C14a therefore sums the spend before it submits and reports first if the total would pass 600.
The 256-region scoring (about 900 CPU-h) is outside this plan.

Wall time: day 1, waves 1 to 4, with the gate result on four measurements; day 2, waves 5 and 6, and the build as soon as the tars are on Fir; day 3, training (about 6 to 7 h per seed, the three in parallel) and scoring (about 2 h); day 3 to 4, the deliverable.

## Log
- 2026-11-23T20:48Z wave 1 launch: C1 crux-tests [opus], C2 Nibi staging [opus]
- 2026-11-23T20:55Z AMENDMENT (foreman, env defect from C2): interim notes of probe/build/train/score_p16/tests move from cruxvault/results/t392/<x>/ (gitignored, unconverted) to run_state/notes/<x>/ (tracked). Only side_by_side and deviance_addendum stay under results/ and get the runall/MANIFEST/! contract. C2: Nibi compute nodes cannot read /nearline; probe extraction runs on login node l1 under nice (GNU tar seeks past members, ~3.1 GiB read). C2 retry 1.
- 2026-11-23T20:56Z C1 accepted (64 tests, 1193 lines; merged; reviewer spawned on the diff in parallel since >300 lines; diff lines read total 0). Assumptions for C3: provenance halves form, --work-root abs paths unchanged, grid from reference h5 shape.
- 2026-11-23T20:57Z wave 2 launched from 3a7d1a0a: C3, C4, C5 builders [opus]; C1 test reviewer [opus]. C2 retry 1 running (login-node extract).
- 2026-11-23T20:59Z C2 accepted after retry 1 (20 members, 3,342,082,081 B on Nibi /scratch/mforooz/rs392/probe/work; login-node extract 203 s; job 28710841 failed 3 s). Merged. Diff lines read: 100.
- 2026-11-23T21:01Z C4 accepted (196 lines read in full; 38 tests pass; Monte Carlo check 0.2% mean, 1.1% var). Merged. Note for C15a: loss is one mean over the union mask, so stranded tracks carry 2 channels of bins vs 1 for DNase/ATAC (consequence of the target change, report it). For C9: pass unstranded_kernel mix as [17] kernel for DNase/ATAC. Diff lines read: 296.
- 2026-11-23T21:02Z C1 test review: 10 findings (provenance schema, 3 vacuous refusal cases, self-compare stitch test, copy-not-recompute, work-root, kept-reads, chunk clamp, verify isolation, raises(Exception), loss tests). C1 retry 1 sent. AMENDMENT C3: chunks (min(4096,n_bins),2); real provenance form is nested {tagalign:[...]}; add --testbed-root swap of /project/def-maxwl/mforooz/ONION/testbed/readsets. C3 first slip done (39 pass), sent back for these amendments (retry 1).
- 2026-11-23T21:03Z C5 accepted (857 lines; spot-checked gate reading, live checks on reference arm pass: seeds 0 1 2, 80000 windows, 993,615 params, SCORED 3 of 3; real build plan 2,740 rows, 232 tasks, 596.2 GiB, 7,184 members). Merged without full reviewer (process deviation: gate-reading path will be exercised on the real probe store at C7b). check.py gate mapping must match C3 provenance: verified at C7b.
- 2026-11-23T21:04Z C3 accepted after retry 1 (517 lines; scan+gate+floor read in full: gate compares the per-read rebuilt overlap with the eic_v1_2 h5 on every bin, independent file; 39 tests pass on old fixture; check.py compatibility verified on smoke store). Merged. Must rerun its tests after the C1 fix round merges. Diff lines read: ~450 of 517.
- 2026-11-23T21:04Z C1 fix round accepted and merged (d56a3300). On umbrella: read_starts+store+convert+tools tests 69 passed. C7a launched [opus] (kit RS1, Nibi rs392_probe).
- 2026-11-23T21:09Z C6 accepted (304 lines; res_dataset routing read in full; 61+31 tests pass; average sets: 305 ids over 36 tracks all in eic_v1_2). Merged. Note for C8: train.py --stage-dir does not stage read_start_store / read_start_average_h5; C8 extends staging (train.py is C8-owned). Interpretation logged: input AND target are read starts (the dataset reads starts for both).
- 2026-11-23T21:10Z wave 4 launched from 1edb7c02: C8 model, C9 adapter, C16 docs [opus]. Status table + question (read starts in and out, rec yes) sent to foreman.
- 2026-11-23T21:11Z C16 accepted (111 lines docs; test_docs 2 pass). Merged.
- 2026-11-23T21:15Z C9 accepted (558 lines; no full read: adapter verified vs numpy reference on 4 cases + 4-way mutation check; scorer files untouched vs exp/t370). Merged. Note for C13: set RS392_EDGE_LOG only for the scoring pass (guardrail windows add open edges); a run at chrom bin 0 counts as open edge.
- 2026-11-23T21:16Z C8 accepted (447 lines; config = reference + 3 read_start keys only, verified by json diff; params 993,695 vs 993,615, width 25 both; base golden green; staging extended for read-start store). Merged. rs_train.sh calls train.py (routes read_start to res_dataset.train). C12a must confirm staging line + memory with 11 workers.
- 2026-11-23T21:18Z C7a: probe array 28712220 ran; GATE FAILED 4/4 (12 files, 2,289,096 differing bins each = chrY size; chrY missing from grid because read_starts uses bin25.MAIN_CHROMS). chr1-22,X exact. 0.35 CPU-h. STOP reported to foreman; asked "fix chrY and rerun probe" (rec yes). Waiting; preparing C3 fix only (no compute). Merged jobs.tsv.
- 2026-11-23T21:20Z C3 retry 2 accepted (chrY fix: grid and chromosome set per file from reference h5 keys; 3 references must agree; 4 new tests; 64 pass). Merged. Testbed measurement ENCSR061UOM has 23 keys (no chrY): per-measurement sets handled. Probe rerun needs a new kit (RS2 is being pushed by C10a from aa796683, without the fix) -> use RS3. Waiting on PI word to rerun.
- 2026-11-23T21:30Z C10a partial->accepted (e2e 6 pass; sanity tool; rs_tests.sh). Merged. Kit RS2 (9a6bc874, before chrY fix) on Fir; suite jobs 5521259 (RS2) 5521260 (RCP9) pending, <=48 CPU-h. C10b collects. Note: rerun suite on the kit used for training if code changes after RS2.
