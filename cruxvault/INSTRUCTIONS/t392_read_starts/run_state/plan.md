# Plan: miniCANDI on read-start counts, Phase 1 (task t392)

Written 2026-11-23 by the planner. Branch `implementation/t392-read-start-minicandi`, HEAD 8325cb42, draft PR 66.
Sources: `../INSTRUCTIONS.md` (cited as I), `../HANDOFF_2026-11-23_read-start-minicandi.md` (H), `GOALS.md` (rows R1 to R17), `DECISIONS.md` (D1 to D8).

## Goal

Train the full-combination miniCANDI (config `tools/minicandi/configs/recipe/combo/rung4/halves_mix_avg_pearson.json`, seeds 0, 1, 2) on 25 bp read-start counts.
ChIP-seq tracks and the ChIP-seq control have two channels, plus-strand starts and minus-strand starts. DNase-seq has one channel.
Score the read-start model and the existing read-count model side by side on the 16-region evaluation set of `eic_v1_2`, against the overlap-count truth.
Report imputation, correction and calibration with seed ranges, then the deviance metrics when PR 65 is merged.
Stop after the 16-region table. Phase 2 and 256-region scoring do not start.

Design in one paragraph. The tables stay `eic_v1_2`, unchanged, so the table hash of the evaluation set and of the checkpoint still match.
A new config key, `read_start_store`, sends every read of counts to a second store that holds stranded 25 bp read-start counts for the same measurement ids.
The model is the rung-4 U-Net with its existing sub-channel path set to 2 strand channels.
At scoring, a wrapper around the loaded model, i.e., the read-start adapter, reads the stranded inputs from the read-start store, runs the model, and converts the output to an overlap-count NB.
`registry.load_run` applies the read-start adapter, as it already applies `ResAdapter` for other bin sizes. `score.py` and `predict_store.py` are not edited.

Python for every local command: `PY=/Users/mforooz/miniforge3/envs/candii/bin/python`, with `export PYTHONPATH=$PWD/src`, from the worktree root.

## Acceptance criteria

1. `$PY -m pytest tests/test_t392_read_starts.py tests/test_t392_store.py tests/test_t392_dataset.py tests/test_t392_convert.py tests/test_t392_tools.py tests/test_t392_model.py tests/test_t392_adapter.py tests/test_t392_end_to_end.py -q -p no:cacheprovider` reports 0 failed, 0 errors.
2. `$PY tools/t392/check.py tests --cluster fir --kit RS<n> --base-kit RCP9` exits 0. It prints `new failures: 0` (the full suite on the kit against the full suite on the base kit).
3. `$PY tools/t392/check.py probe --cluster nibi --store /scratch/mforooz/rs392/probe/store` exits 0. It prints `differing bins: 0` for every probe file, and the wall time and peak memory of each.
4. `$PY tools/t392/check.py store --cluster fir --store /scratch/mforooz/rs392/store --tables /project/def-maxwl/mforooz/ONION/tables/eic_v1_2` exits 0. It prints `measurements 2740 of 2740`, `files 8220 of 8220`, `differing bins: 0`.
5. `$PY tools/t392/check.py train --cluster fir --runs /project/def-maxwl/mforooz/t392/runs/rs_halves_mix_avg_pearson_unet` exits 0. It prints `seeds done: 0 1 2`, `windows_seen 80000` for each, and a parameter count within 2% of 1,000,000.
6. `$PY tools/t392/check.py scores --cluster fir --runs /project/def-maxwl/mforooz/t392/runs/rs_halves_mix_avg_pearson_unet --sub p16_sv` exits 0. It prints `SCORED 3 of 3` and `copy_leak_pass true` for each seed.
7. `bash cruxvault/results/t392/side_by_side/runall.sh` exits 0, and `$PY -m pytest tests/test_results_folders.py -q` passes. `SIDE_BY_SIDE.md` holds one row per model, the three headline terms with seed ranges, one figure and one paragraph.
8. `git diff origin/exp/t370-minicandi-recipe -- tools/onion_eval/score.py tools/onion_eval/predict_store.py tools/onion_eval/metrics.py` prints nothing that this run wrote (the only allowed difference is the merge of PR 65).
9. `git diff --stat HEAD -- .longrun/` shows no change made by this run.

## Chunks

Cluster folders. Nibi probe: `/scratch/mforooz/rs392/probe/{work,store,logs}`. Fir: `/scratch/mforooz/rs392/{tars,work,store,aux,pred_store,logs}` and `/project/def-maxwl/mforooz/t392/{runs,logs}`.
Code reaches a cluster as a code kit: `bash tools/onion_kit/push_kit.sh --cluster <fir|nibi> RS<n>`. Every job name starts with `rs392_`.
A cluster chunk ends when its jobs are submitted and its note file holds the job ids; the main agent polls with the chunk's Test command.

### C1 tests against the instructions   wave 1
Goal: Write the tests for the four code items before any implementation exists. The agent is `crux-tests`. It reads `../INSTRUCTIONS.md` and the Interfaces lines of chunks C3, C4 and C6 in this plan. It does not read any implementation file of this run.
Owns: `tests/test_t392_read_starts.py`, `tests/test_t392_store.py`, `tests/test_t392_dataset.py`, `tests/test_t392_convert.py`
Interfaces: Each test imports the module under test inside the test function or a fixture, never at module top level, so collection passes before the modules exist. Required cases:
(1) Stranded recount. A hand-made tagAlign gives known plus and minus start counts; a minus read counts at the bin of `end - 1`; `starts.sum()` equals the number of reads on the main chromosomes; the overlap array of the same scan equals `tools/t112/bin25.py::counts_from_tagalign` on the same file, bin for bin; one changed bin in the reference makes the gate report 1 differing bin and a non-zero exit.
(2) Store. Every file has one dataset per chromosome of shape `[len // 25, 2]`; column 0 is plus; `starts.h5` equals `starts_pr1.h5 + starts_pr2.h5`; the attributes of C3 exist; `kernel.sum() * reads` equals the overlap sum per file.
(3) Dataset. `res_dataset.make_dataset` with `read_start_store`, `average_input` true and `average_h5` set returns batches with no error; the batch holds `x_counts_sub`, `y_counts_sub` `[B, L, 36, 2]` and `avg_sub`; channel sums equal `x_counts` and `y_counts`; DNase-seq has all its starts in channel 0 and zeros in channel 1; the window's own cell never enters its average.
(4) Conversion. For a 0/1 box kernel the converted mean equals the box convolution of each strand's mean, summed over strands; the NB size equals mu^2 / (var - mu) with var the sum of the contributions' variances; a strand mean that is Poisson (very large size) gives a converted size at the upper clamp; the conversion of 16 stitched windows equals the conversion of their concatenation; an unstranded track ignores channel 1.
Depends on: none
Serves: R5, R1, R2, R3, R4
Test: `$PY -m pytest tests/test_t392_read_starts.py tests/test_t392_store.py tests/test_t392_dataset.py tests/test_t392_convert.py --collect-only -q -p no:cacheprovider` (exit 0, at least 16 tests listed)

### C2 stage the probe read files on Nibi   wave 1
Goal: Put the read files of four probe measurements on Nibi scratch, from one pass over one tar. No recount yet.
The measurements: `ENCSR000AKT.pooled.uniform.base` (small ChIP-seq, 2 replicates), `ENCSR000ENE.pooled.uniform.base` (DNase-seq), `ENCSR000AMI.pooled.uniform.mapq0` (the largest read volume of all 2,740, 2.0 GiB, 319 M reads), and one measurement with `lines_per_fragment` 2 whose files are in `onion-work__000`.
The member paths come from each measurement's `provenance.json::halves` on Nibi `/scratch/mforooz/onion/measurements/` and from the index `/project/def-maxwl/mforooz/ONION/nearline_archive_2026-11-17/index/onion-work__000.txt`.
Extract on a Nibi login node: `tar -xf /nearline/def-maxwl/mforooz/scratch_archive_2026-11-17/onion-work__000.tar -C /scratch/mforooz/rs392/probe/work <members>`. About 3 GiB lands on scratch (708 of 1,024 GiB are in use). Nothing is deleted.
Owns: `cruxvault/results/t392/probe/STAGING.md` (the commands, the member list, the wall time, `lfs hsm_state` of the tar before the read)
Interfaces: The extracted tree is `/scratch/mforooz/rs392/probe/work/onion/work/<experiment>/...`, i.e., the provenance path with the prefix `/scratch/mforooz/` replaced by `/scratch/mforooz/rs392/probe/work/`. C3 calls this prefix swap `--work-root`.
Depends on: none
Serves: R6, R7
Test: `hpc run nibi 'find /scratch/mforooz/rs392/probe/work -name "pr[12].tagAlign.gz" | wc -l'` (prints the member count recorded in STAGING.md)

### C3 stranded read-start recount, gate and store writer   wave 2
Goal: One scan of a measurement's read files writes its stranded 25 bp read-start counts and proves, in the same scan, that the reads rebuild the stored overlap counts exactly.
Owns: `tools/onion_data/read_starts.py`, `slurm/t392/rs_recount.sh`
Interfaces:
- Read-start rule. The 5' base of a read is `start` on strand `+` and `end - 1` on strand `-` (`recount.point_positions(start, end, strand, 0)`). Its start bin is that base `// 25`. The grid is `len // 25` bins per chromosome, the chromosomes of `tools/t112/bin25.py::MAIN_CHROMS`.
- Overlap rule. `tools/t112/records.py::COUNT_RULE`, with the edge handling of `bin25.counts_from_tagalign` (lines 69 to 76).
- `scan(paths, sizes, reach=8) -> dict`: per chromosome `starts` uint32 `[n_bins, 2]` (column 0 plus, column 1 minus) and `overlap` uint32 `[n_bins]`, rebuilt by placing each read's span box at its start bin; `kernel_counts` int64 `[2, 2 * reach + 1]`, where entry `[s, reach + k]` is the number of strand-s reads whose span covers bin (start bin + k); `n_lines`, `n_main`. A read whose span leaves `reach` raises.
- Store, one folder per measurement id under the store root: `starts.h5`, `starts_pr1.h5`, `starts_pr2.h5`. One dataset per chromosome, `[n_bins, 2]`, uint16 (uint32 when a value exceeds 65,535), chunks `(4096, 2)`, gzip 4 with shuffle. File attributes: `schema` 1, `counting_rule` "read_start", `resolution` 25, `strand_columns` "plus,minus", `measurement_id`, `depth_reads`, `n_reads_strand` `[2]`, `kernel_counts`, `kernel` float64 `[2, 17]` (`kernel_counts` divided by the strand's reads). Every assay is stored with both strands; which tracks the model reads as one channel is decided in C4.
- Gate. For each of the three files, the rebuilt overlap array must equal the `eic_v1_2` file of the same name (`counts.h5`, `pr1.h5`, `pr2.h5`) on every bin of every chromosome. `provenance.json::gate` records per file the reference path, its sha256, `n_bins`, `n_differing`, and `result`. Any differing bin gives exit 1 after the files are written.
- CLI: `read_starts.py run --measurement <id> --measurements <eic_v1_2 measurement dirs> --work-root <prefix> --out <store root> [--reach 8]` reads the read-file list from `<measurements>/<id>/provenance.json::halves`; `read_starts.py verify --out <store root>/<id>` rechecks the hashes, `starts = pr1 + pr2`, and the recorded gate; `read_strands(path, chrom, start, n) -> int32 [n, 2]`.
- `slurm/t392/rs_recount.sh <kit> <plan.tsv>` runs the rows of the plan whose `task` equals `$SLURM_ARRAY_TASK_ID`, skips a measurement whose `provenance.json` holds a passed gate, and prints one line per measurement: id, wall seconds, peak RSS, `n_differing`.
Depends on: C1
Serves: R1, R2, R6, R7
Test: `$PY -m pytest tests/test_t392_read_starts.py tests/test_t392_store.py tests/test_onion_data_recount.py -q -p no:cacheprovider`

### C4 strand channels and the conversion to overlap counts   wave 2
Goal: The pure functions that every model-side chunk shares: which tracks have two channels, the span kernel, the moment-matched conversion, the strand-aware output and loss.
Owns: `tools/minicandi/read_start.py`
Interfaces:
- `STRANDS = 2`; `UNSTRANDED_TRACKS = ("ATAC-seq", "DNase-seq")` (ATAC-seq waits on decision (iii)); `channel_mask() -> bool [36, 2]`, False at channel 1 of an unstranded track.
- `expected_kernel(read_length, reach=8) -> float64 [2, 17]`: the kernel of reads of one length whose start offset is uniform over the 25 bases of a bin. `unstranded_kernel(kernel, n_reads_strand) -> [17]`: the read-weighted mix of the two rows.
- `convert_nb(mu_sub, n_sub, kernel, stranded) -> (mu, n)`: tensors `[..., L, 2]` and a kernel `[2, 17]`. With w the kernel weight, `mu[i] = sum over strands and k of w * mu_sub[i - k]`, `v[i] = sum of (w * mu_sub[i - k])^2 / n_sub[i - k]`, `n = mu^2 / v`, clamped to `contract.N_MIN .. N_MAX` (the Poisson case of I section 2 is `v` at or below 0, which gives `N_MAX`). This is the moment match of I section 2 with var = sum of `w * mu + (w * mu)^2 / n`, i.e., each start count thinned by w and the bins and strands taken as independent. Positions outside the array are filled with the edge bin's values (replicate padding).
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
- `build_plan.py --tables <eic_v1_2> --measurements <dirs> --index <dir of onion-work__*.txt> --testbed-root <dir> --tasks 228 --out plan.tsv`: one row per measurement (2,740 rows), columns `task, measurement_id, tar, read_gib, n_tagaligns, lines_per_fragment`; rows are packed into tasks of near-equal `read_gib`. `build_plan.py members --plan plan.tsv --tar <name>` prints the tar members that the plan needs (only `pr1` and `pr2` files).
- `check.py <probe|store|train|scores|tests> ...` as in Acceptance criteria 2 to 6. It reads the cluster over `hpc run <cluster>`, never writes there, and exits 69 with "run `hpc up <cluster>`" when the tunnel is down.
Depends on: none (the store layout and the gate record are pinned in C3)
Serves: R6, R7, R15
Test: `$PY -m pytest tests/test_t392_tools.py -q -p no:cacheprovider`

### C6 read-start dataset through res_dataset   wave 3
Goal: Training and the adapter read stranded windows, with the average input, through the entry that `res_dataset.py` already gives.
Owns: `tools/minicandi/rs_dataset.py`, `tools/minicandi/res_dataset.py`
Interfaces:
- Config keys: `read_start` (bool), `read_start_store` (store root), `read_start_average_h5` (path). `average_h5` keeps its read-count path; only the unedited scorer's own dataset reads it.
- `ReadStartDataset(OnionDataset)`: `read()` returns the strand sum of the starts, int32 `[L]`, from `<store>/<id>/starts.h5` or its `_pr1`, `_pr2` siblings; `read_strands(mid, chrom, start, L, which) -> int32 [L, 2]`; `window()` adds `x_counts_sub`, `y_counts_sub` int32 `[L, A, 2]` (0 where hidden or missing; for an unstranded track channel 0 is the total and channel 1 is 0) and, under `average_input`, `avg_sub` float32 `[L, A, 2]` next to `avg` (their channel sum); `average_strands(cell, chrom, start, L)` follows the leave-one-cell-out rule of `OnionDataset.average`.
- `build_average_h5(tables, out, config)`: per track group, one dataset per chromosome, float32 `[n_bins, 2]`, the sum over training cells of starts x 1e6 / `depth_reads` of each cell's base view, with attrs `n_cells`, `measurement_ids`.
- Refused with ValueError: `input_depth = fixed`, `aug_revcomp`, `two_step`, `res_bp != 25`, `res_in_sub` or `res_out_sub` above 1, a `store:` row.
- `res_dataset.make_dataset` returns a `ReadStartDataset` when the config sets `read_start`; `res_dataset.train` then swaps in `read_start.read_start_loss`. The point-rule `ResDataset` keeps refusing `average_h5` (decision (iv)).
Depends on: C3, C4
Serves: R2, R3
Test: `$PY -m pytest tests/test_t392_dataset.py tests/test_minicandi_resolution.py tests/test_minicandi_train_halves.py -q -p no:cacheprovider`

### C7 recount probe on Nibi: the first measured result   wave 3
Goal: Recount the four staged measurements at real scale on Nibi and read the gate. This is the day-1 result. Nibi, because the tars, the staged files and a copy of the `eic_v1_2` counts are there.
Steps: push kit `RS1` to Nibi; `sbatch --job-name=rs392_probe --array=1-4 --cpus-per-task=2 --mem=24000M --time=02:00:00 --gres=none` on `slurm/t392/rs_recount.sh` with a 4-row plan, `--work-root /scratch/mforooz/rs392/probe/work/`, store `/scratch/mforooz/rs392/probe/store`; then `sacct` for elapsed time and MaxRSS.
Owns: `cruxvault/results/t392/probe/PROBE.md`, `cruxvault/results/t392/probe/probe.tsv` (per measurement: reads, read GiB, wall s, CPU-s, peak RSS, store bytes, `n_differing` per file, the kernel rows, and the nMSE between the overlap counts and the starts convolved with the mean kernel)
Interfaces: `probe.tsv` gives the rate in CPU-h per GiB and the store bytes per read; C11 sizes its array, memory and wall time from it. A non-zero `n_differing` stops the run and goes to the foreman (H, stop condition).
Depends on: C2, C3, C5
Serves: R7, R6, R15, R16
Test: `$PY tools/t392/check.py probe --cluster nibi --store /scratch/mforooz/rs392/probe/store`

### C8 strand channels in the model and the training entry   wave 4
Goal: The rung-4 U-Net reads and predicts two strand channels per track when the config sets `read_start`, and nothing changes without the key.
Owns: `tools/minicandi/rungs/unet.py`, `tools/minicandi/train.py`, `tools/minicandi/configs/t392/rs_halves_mix_avg_pearson.json`, `slurm/t392/rs_train.sh`, `tests/test_t392_model.py`
Interfaces:
- Under `read_start` the model sets its sub-channel count to 2 on both sides (the existing `x_counts_sub` input path and `out_sub_emb` output path), reads `avg_sub` through a `Linear(2, D_TRACK)`, and returns `mu_sub`, `n_sub` `[B, L, A, 2]` plus `read_start.strand_output`. `read_start` with `res_in_sub` or `res_out_sub` above 1 raises.
- `train.set_pearson_stats` under `read_start` fits the per-track rate and size on the channels of `x_counts_sub` that `channel_mask()` keeps.
- The config is the reference config plus `read_start: true`, `read_start_store: /scratch/mforooz/rs392/store`, `read_start_average_h5: /scratch/mforooz/rs392/aux/eic_v1_2_read_start_average.h5`. Every other key is equal, `param_budget` 1e6 and `windows_seen` 80000 included.
- `slurm/t392/rs_train.sh <kit> <config> <tables> <seed> <out>` calls `python tools/minicandi/res_dataset.py train ... --device cpu --workers <CPUs - 1>` with the header of `slurm/minicandi/train.sh` (12 CPUs, 48000M, as the reference jobs).
- Tests: without `read_start` the outputs of a seeded rung-4 build are bit-identical to the tree before the chunk (`tests/test_minicandi_base_golden.py` stays green); the parameter count under the budget rule is within 2% of 1e6; perturbing hidden bins of `x_counts_sub` moves no output.
Depends on: C4, C6
Serves: R2, R8
Test: `$PY -m pytest tests/test_t392_model.py tests/test_minicandi_rung4.py tests/test_minicandi_base_golden.py tests/test_minicandi_recipe_keys.py -q -p no:cacheprovider`

### C9 read-start adapter: the conversion between predict and the metrics   wave 4
Goal: `score.py --run` and `predict_store.py run` score a read-start run with no edit to either file.
Owns: `tools/minicandi/rs_adapter.py`, `tools/minicandi/registry.py`, `tests/test_t392_adapter.py`
Interfaces:
- `ReadStartAdapter(inner, config, run_dir=None)` is a `contract.MiniCandi`. `forward(batch)` takes the ordinary overlap-count batch of the scorer and returns `{"n", "mu"}` `[B, L, A]` in overlap-count space.
- Steps of `forward`: (1) read `x_counts_sub` for every available input from the read-start store by `meta` (`measurements_in`, `input_file`, `chrom`, `start_bin`), 0 at hidden bins and missing tracks, and set `x_counts` to the channel sum; (2) read `avg_sub` with `ReadStartDataset.average_strands`; (3) run the inner model; (4) per target track take the kernel from the attributes of the target's file (`measurements_out`, `target_file`), or `read_start.expected_kernel(read_length of C')` when the track has no measurement; (5) `read_start.stitch`, then `read_start.convert_nb`.
- The adapter counts `n_open_edges` (window edges converted with replicate padding) and `n_windows`.
- `registry.load_run` wraps a run whose config sets `read_start`. `input_depth` at scoring other than native is refused.
- Tests use a stub inner model that returns fixed `mu_sub`, `n_sub`: the output equals `convert_nb` of the stub; a batch of 8 consecutive windows has 2 open edges; the batch's own `x_counts` never reaches the inner model.
Depends on: C4, C6
Serves: R4, R10
Test: `$PY -m pytest tests/test_t392_adapter.py tests/test_minicandi_resolution.py -q -p no:cacheprovider`

### C10 end-to-end check, sanity scorer and the full suite on the cluster   wave 5
Goal: Prove the whole path on synthetic tables before any training job, write the read-start-truth sanity tool, and run the full suite against the base.
Owns: `tools/t392/sanity.py`, `tests/test_t392_end_to_end.py`, `slurm/t392/rs_tests.sh`
Interfaces:
- `tests/test_t392_end_to_end.py`: synthetic tables (`tools/minicandi/synth.py`) plus a synthetic read-start store; `res_dataset.train` for 16 windows; `registry.load_run` returns the adapter; `score.py` on a synthetic evaluation set writes `scores.json` with finite headline terms.
- `sanity.py --run <dir> --evalset <json> --out <json>`: on the imputation targets of the 16 pair regions it scores the inner model's `mu_sub`, `n_sub` against the read-start truth: mean NB deviance per bin, the same for a per-track marginal NB, and the Pearson r of log1p mean against log1p truth, per assay class. A sanity check only (R12); it never enters the table's headline columns.
- `slurm/t392/rs_tests.sh` is `slurm/minicandi/rcp_tests.sh` with the job name `rs392_tests` and the log under `/project/def-maxwl/mforooz/t392/logs`. Run it on Fir on kit `RS<n>` and read the base list from the same job on kit `RCP9` (the base's failing tests are known: H, Traps).
Depends on: C8, C9
Serves: R5, R12, R4
Test: `$PY -m pytest tests/test_t392_end_to_end.py -q -p no:cacheprovider && $PY tools/t392/check.py tests --cluster fir --kit RS<n> --base-kit RCP9`

### C11 full recount and store build   wave 4
Goal: Recount all 2,740 measurements in every view and build the read-start store, with the gate on every file. Fir under decision (i): the training and the 16-region scoring inputs are on Fir, Fir scratch has 16 TiB free, and Nibi scratch has 316 GiB free while the store needs the read files (503.5 GiB) plus its own size.
Steps on Fir, after the PI's Globus transfer has put the 6 `onion-work` tars in `/scratch/mforooz/rs392/tars` and the testbed read sets in `/scratch/mforooz/rs392/work/testbed`:
(1) `build_plan.py` with the Fir paths; (2) 6 jobs `rs392_untar` (1 CPU, 4000M, 04:00:00), one per tar, extract only the plan's members to `/scratch/mforooz/rs392/work`; (3) array `rs392_recount`, `--array=1-228%12 --cpus-per-task=2 --mem=24000M --gres=none`, wall time and memory reset from `probe.tsv`, each array chained `afterok` on its tar's untar job; (4) one job `rs392_avg` (`rs_dataset.build_average_h5`, 8 CPUs, 32000M) writes `/scratch/mforooz/rs392/aux/eic_v1_2_read_start_average.h5`.
Nothing is deleted. The extracted read files stay on Fir scratch.
Owns: `cruxvault/results/t392/build/BUILD.md`, `cruxvault/results/t392/build/gate.tsv` (8,220 rows: measurement, file, `n_bins`, `n_differing`), `cruxvault/results/t392/build/jobs.tsv` (job ids, CPU-s from `sacct`)
Interfaces: Store root `/scratch/mforooz/rs392/store`. A failed gate on any file stops the run before training (R7). Fir file count stands at 831K of 1,000K; the build adds about 18K files.
Depends on: C7 (gate passed), C3, C5, C6 (for step 4), decision (i)
Serves: R6, R7, R15, R16
Test: `$PY tools/t392/check.py store --cluster fir --store /scratch/mforooz/rs392/store --tables /project/def-maxwl/mforooz/ONION/tables/eic_v1_2`

### C12 train the read-start model, seeds 0 to 2   wave 6
Goal: A probe first, then the three seeds, on Fir CPU nodes as the reference (`run.json::device` is `cpu`, 12 CPUs, 48000M).
Steps: (1) `rs392_train_probe`: seed 0 with `--max-windows 2000` into a probe folder, to read seconds per step, peak memory and the Pearson statistics; (2) three jobs `rs392_train_s<k>` on `slurm/t392/rs_train.sh`, `--time` = 3 x the projected wall, into `/project/def-maxwl/mforooz/t392/runs/rs_halves_mix_avg_pearson_unet/seed<k>`; a job that times out is resubmitted with the same command (it resumes from `ckpt.pt`).
Owns: `cruxvault/results/t392/train/TRAIN.md`, `cruxvault/results/t392/train/jobs.tsv`
Interfaces: Run folders hold `ckpt.pt`, `run.json`, `log.jsonl`, `DONE`. `run.json` must show `windows_seen` 80000 and the tables hash of `eic_v1_2` (`7a1a0325d00ddff3...`). The reference is never retrained.
Depends on: C10, C11
Serves: R8, R9, R15, R16
Test: `$PY tools/t392/check.py train --cluster fir --runs /project/def-maxwl/mforooz/t392/runs/rs_halves_mix_avg_pearson_unet`

### C13 16-region scoring of the read-start model, and the reference's scores   wave 7
Goal: Score the three read-start seeds on the 16-region evaluation set with the unedited scorer, and collect the reference's existing 16-region scores.
Steps: (1) read the settings of the reference's scoring from Fir `runs_rcp/rcp_halves_mix_avg_pearson_unet/seed0/p16_sv/scores.json` (`scorer_settings`, `pred_store`, the evaluation-set hash) and from the head of its `predict_*.out` logs, and use the same values (shards 4, CPU predict, the same inference batch, same-view pass on, `--refs-from /project/def-maxwl/mforooz/ONION/eval/eic_v1_2/baselines_p16`); (2) submit with `slurm/minicandi/submit_arms.sh` in its score-only two-phase form, `SCORE_SUBDIR=p16_sv`, `PRED_STORE_ROOT=/scratch/mforooz/rs392/pred_store`, `KEEP_PRED_STORE=1`, job names prefixed `rs392_`; (3) one job `rs392_sanity` per seed on `tools/t392/sanity.py`; (4) `select_rule.py arm-summary` over the two arms.
Owns: `cruxvault/results/t392/score_p16/SCORE.md`, `cruxvault/results/t392/score_p16/arm_summary_p16_sv.tsv`, `cruxvault/results/t392/score_p16/sanity.tsv`, `cruxvault/results/t392/score_p16/jobs.tsv`
Interfaces: `arm_summary_p16_sv.tsv` has the rows `rcp_halves_mix_avg_pearson_unet` and `rs_halves_mix_avg_pearson_unet` with `<term>.mean` and `<term>.range` for `imputation_raw`, `correction_v3_raw`, `uncertainty_raw`. The reference row must reproduce 0.4996, 0.8156, 0.6852 (ranges 0.0349, 0.0105, 0.0924). SCORE.md states the share of bins within 8 bins of an open edge (from `n_open_edges`).
Depends on: C12, C10
Serves: R10, R9, R12, R15, R16
Test: `$PY tools/t392/check.py scores --cluster fir --runs /project/def-maxwl/mforooz/t392/runs/rs_halves_mix_avg_pearson_unet --sub p16_sv`

### C14 deviance metrics on 16 regions   wave 8
Goal: The deviance metrics of PR 65 for the read-start model, beside the reference's existing `p16_dev` scores. Starts only when `gh pr view 65 --repo mehdiforoozandeh/CANDII --json state` shows MERGED and `origin/exp/t370-minicandi-recipe` is merged into the umbrella (D8).
Steps: read how the reference's `p16_dev` was scored (Fir `runs_rcp/DEV_SUBMITTED_p16.txt`, jobs 5381558 to 5381572); reuse the kept prediction store of C13 when the deviance scorer accepts its request list, else run a second predict pass; score into `p16_dev`.
Owns: `cruxvault/results/t392/score_p16/DEVIANCE.md`, `cruxvault/results/t392/score_p16/deviance_p16.tsv`
Interfaces: `deviance_p16.tsv` has one row per model with skill, gap and R, each with its seed range. If PR 65 is still open when C15 runs, C15 marks the deviance columns "waits on PR 65" and the run reports that to the foreman.
Depends on: C13, PR 65 merged
Serves: R11
Test: `$PY tools/t392/check.py scores --cluster fir --runs /project/def-maxwl/mforooz/t392/runs/rs_halves_mix_avg_pearson_unet --sub p16_dev`

### C15 side-by-side deliverable, finding and report   wave 8
Goal: The deliverable folder under the runall contract, the crux finding, and the stop report.
Owns: `cruxvault/results/t392/side_by_side/SIDE_BY_SIDE.md`, `cruxvault/results/t392/side_by_side/runall.sh`, `cruxvault/results/t392/side_by_side/MANIFEST.tsv`, `cruxvault/results/t392/side_by_side/table.tsv`, `cruxvault/results/t392/side_by_side/side_by_side.png`, `cruxvault/results/t392/side_by_side/plot.py`, one `!` line in `.gitignore`
Interfaces: One table, one row per model, one column per metric with the seed range (AGENTS.md 7.2); the read-start-truth sanity numbers in a separate, labelled block; one figure; one paragraph; the CPU-h spent with job ids; every deviation from I. Then `crux finding` on q75 with the file as evidence, `crux task done t392`, and the report to the foreman session by SendMessage. No message to the PI.
Depends on: C13 (C14 when it has landed)
Serves: R13, R14, R16
Test: `bash cruxvault/results/t392/side_by_side/runall.sh && $PY -m pytest tests/test_results_folders.py -q -p no:cacheprovider`

## Decisions needed

- (i) Where the store is built. Recommendation: Fir. It needs one action from the PI: a Globus transfer, Nibi to Fir, of the 6 files `/nearline/def-maxwl/mforooz/scratch_archive_2026-11-17/onion-work__00{0..5}.tar` (1,104 GiB) to `/scratch/mforooz/rs392/tars/` and of `/project/def-maxwl/mforooz/ONION/testbed/readsets/` (18 GiB) to `/scratch/mforooz/rs392/work/testbed/`; no Globus CLI is logged in on either cluster. Cost: some hours of transfer, 0 CPU-h. The other path is a build on Nibi tar by tar, which needs leave to delete our own extracted copies between tars on a scratch at 708 of 1,024 GiB, and still needs a Globus transfer of the store to Fir later.
- (ii) The wording of the gate. At 25 bp, the 25 bp start counts alone cannot rebuild the overlap counts exactly, because the number of bins a read touches depends on where the read starts inside its bin (a 36 bp read touches 2 or 3 bins). Recommendation: the gate runs at read level inside the scan (each read's span box is placed at its start bin, and the rebuilt array must equal `eic_v1_2` with 0 differing bins, on all 8,220 files); predictions are converted with each target's measured mean span kernel. Cost: 0.
- (iii) ATAC-seq (30 measurements) is not named in the instruction. Recommendation: one channel, as DNase-seq. The store keeps both strands for every assay, so a later change costs no rebuild.
- (iv) "Make `res_dataset.py` accept `average_h5`": recommendation, the read-start dataset is a new class reached through `res_dataset.make_dataset` and `res_dataset.train`, and the 1 bp point-rule `ResDataset` keeps its refusal (its windows start at any base, and the average file is on the 25 bp grid). Cost: 0.

## Ground truth verified

All on 2026-11-23. `hpc status` showed fir up and nibi up; no exit 69 occurred.

Branches and reuse
- `git for-each-ref refs/remotes/origin` (115 refs after `git fetch`): no branch holds a 25 bp stranded recount, a strand channel or a moment-matched conversion to overlap counts. The nearest code is listed below.
- `origin/implementation/t390-eic-pvalue-entry:tools/eic_entry/ladder_probe.py`, `cmd_starts` (line 139): one pass over a tagAlign that keeps 1 bp positions by strand in memory (`<chrom>_p` = plus `start`, `<chrom>_m` = minus BED `end`) and the store-rule counts. It writes no 25 bp store. `bin_kernel` (line 714) and `apply_kernel` (line 728) are the fragment-extension kernel of Phase 2. The plan reuses the idea, not the file: the minus position there is `end`, and this plan pins `end - 1`, the convention of `recount.point_positions`.
- `cruxvault/results/t390/postpile/POSTPILE.md` on that branch: the nMSE table of I section 1 matches (0.0172, 0.2550, 0.0212; stranded starts 0.0028, 0.0746; unstranded DNase-seq 0.0018).
- PR 65: `gh pr view 65` gives state OPEN, base `exp/t370-minicandi-recipe`, not merged.

Code
- `tools/t112/records.py:42`: `COUNT_RULE` = "+1 to bins floor(start/25)..floor(end/25) inclusive, grid floor(len/25)". The code of the rule is `tools/t112/bin25.py::counts_from_tagalign`, lines 69 to 76 (reads past the grid dropped, the end bin clamped, the grid cut to `len // 25`).
- `tools/onion_data/recount.py`: a 1 bp point-rule recount into a region store (768 regions), one unstranded array, with a fragment-length shift. `point_positions` (line 112) already gives the 5' base by strand. `scan_tagaligns(..., overlap=True)` and `readset_problems` are the model for an in-scan check against the stored counts.
- `tools/minicandi/res_dataset.py:67`: `REFUSED` holds `average_h5`; `ResDataset.window` passes a base-pair start, so the 25 bp average file cannot be read there. `res_dataset.train` (line 331) swaps the dataset class and the loss of `train.train`; `train.py` itself never builds a `ResDataset`.
- The reference config sets no `res_store`: the reference trained with `OnionDataset` on the 25 bp `h5:` files of `eic_v1_2`, not on the region store.
- `tools/minicandi/contract.py`: `A = 36` fixed tracks (35 assays and `chipseq-control`), asserted. `registry.py:62`: rung 4 is `unet`. `rungs/unet.py` already has a sub-channel path (`x_counts_sub` in, `mu_sub`, `n_sub` out) used for 5 bp sub-bins; `components.res_subs_of` allows 1 or 5 only. `losses.pool_nb` (line 237) is the moment match n = (sum mu)^2 / sum(mu^2 / n).
- `tools/minicandi/train.py:363`: `set_pearson_stats` fits the Pearson input statistics on `x_counts`. `train.py:53`: a config key that no code under `tools/minicandi` reads is refused.
- Input contract and store reader: `tools/minicandi/dataset.py::OnionDataset.read` (line 783) reads `h5:<dir>/counts.h5` and its `pr1.h5`, `pr2.h5` siblings, one 1-D dataset per chromosome; `average` (line 1270) reads `average_h5` as 1-D per track and chromosome.
- Predict to metrics: `tools/onion_eval/predict_store.py` stores `n`, `mu` of the target channel per request; `score.py --phase metrics` reads them. The scorer refuses a run whose checkpoint tables hash differs from the evaluation set's (`score.py:1210`) and a store of another run, tables or request list (`predict_store.py:391`). `registry.load_run` (line 199) already wraps a run in `ResAdapter`, which re-reads its inputs from another store by `meta` and returns a 25 bp NB. The read-start adapter uses the same slot, so neither scorer file changes.

Reference model
- Rows on `origin/exp/t370-minicandi-recipe`, `cruxvault/presentations/t370-minicandi-recipe/data/pareto_p16_sv.tsv`: imputation 0.4996 (range 0.0349), correction 0.8156 (range 0.0105), calibration 0.6852 (range 0.0924), 3 seeds, class PASS. `pareto_p256_sv.tsv`: 0.4996, 0.8180 (range 0.0135), 0.6852. These match H (0.500, 0.818 with range 0.014, 0.685).
- Fir `/project/def-maxwl/mforooz/ONION/runs_rcp/rcp_halves_mix_avg_pearson_unet/seed{0,1,2}/`: `ckpt.pt`, `run.json` (`device` cpu, `param_count` 993,615, `windows_seen` 80000), and the score folders `p16_sv`, `p16_dev`, `p256_sv`, `p256_dev`, each with `scores.json` and `SCORED`. The prediction stores are under Fir `/scratch/mforooz/onion/pred_store/runs_rcp/`.
- The "16-region" evaluation set, Fir `/project/def-maxwl/mforooz/ONION/eval/eic_v1_2/evalset_p16.json`: 256 imputation regions, 16 pair regions, 63 imputation targets, 1,480 correction pairs, 3,134 replicate pairs, tables hash `7a1a0325d00ddff3`. Its reference terms are in `eval/eic_v1_2/baselines_p16`. Nibi holds evaluation sets for the tables `eic_v1` only (hash `a068a8674aee4252`), so the 16-region scoring runs on Fir.

Clusters
- Nibi `/nearline/def-maxwl/mforooz/scratch_archive_2026-11-17/`: 23 tars; the 6 `onion-work__00{0..5}.tar` sum to 1,104.4 GiB. `lfs hsm_state` on the 6 gives "exists archived", i.e., a disk copy is present now.
- Index, Nibi `/project/def-maxwl/mforooz/ONION/nearline_archive_2026-11-17/index/onion-work__*.txt`: 31,781 members, of which 8,416 tagAligns, 751.7 GiB. `status.tsv` marks all 6 tars OK. `removal_work_2026-11-17.log` records that `/scratch/mforooz/onion/work` was removed.
- All 2,740 rows of `tables/eic_v1_2/measurements.tsv` have a `provenance.json` whose `halves` lists the `pr1` and `pr2` read files. For 2,662 measurements every file is in the index, each measurement inside one tar; the 78 testbed measurements point at 156 files that exist on Nibi `/project/def-maxwl/mforooz/ONION/testbed/readsets` (18 GiB). The read volume is 578.7 GiB over 7,184 distinct files (503.5 GiB distinct). 173 measurements have 2 lines per fragment. Per tar: 618, 389, 503, 488, 495, 169 measurements.
- Views in the table: 785 base, 362 no-dedup, 1,086 depth, 363 MAPQ, 83 crop, 49 single-end and paired-end, 12 antibody-proxy. Assay classes: 1,254 narrow histone, 901 broad histone, 408 control, 147 DNase-seq, 30 ATAC-seq.
- `eic_v1_2` counts: `/scratch/mforooz/onion/measurements/<id>/{counts,pr1,pr2}.h5`, the same path on both clusters (Nibi 2,884 folders, Fir 2,838 folders, 251 GiB on Fir). `measurements.tsv` has the same md5 on both, and `ENCSR000AKT.pooled.uniform.base/counts.h5` has the table's sha256 on both.
- Average-input file: `/scratch/mforooz/onion/aux/eic_v1_average.h5`, 7,451,594,429 bytes, on both clusters.
- Nibi `/project/def-maxwl/mforooz/ONION/CO-A7/res_store`: 6.8 GiB, with `regions.tsv` beside it. Base views only; tests only.
- Quotas. Nibi: scratch 708 of 1,024 GiB, project 4,311 of 4,657 GiB. Fir: scratch 2,573 GiB of 19 TiB, 831K of 1,000K files; project 23 of 28 TiB.
- `bash tools/onion_kit/globus_transfer.sh route` prints NO_ROUTE (no Globus CLI on either login node).

Local test run
- `$PY -m pytest tests/test_minicandi_resolution.py tests/test_minicandi_contract.py -q -p no:cacheprovider`: 156 passed in 45 s. `tests/test_onion_data_recount.py tests/test_minicandi_train_halves.py tests/test_minicandi_rung4.py`: 73 passed in 16 s. `candi` imports from the worktree's `src`. No environment defect appeared.
- Traps: in zsh, write `"${B}:path"` for `git show`, because `$B:c` applies a modifier; base Python cannot import torch, so call the `candii` interpreter by path; another session edited `run_state/DECISIONS.md` and `LOG.md` during this planning (D8 added), so scope every `git add`.

## Cost estimate

Sources: `sacct` on Fir and Nibi, read 2026-11-23. CPU-h = CPUTimeRAW / 3600.

| step | CPU-h | arithmetic and source |
|---|---|---|
| probe staging and probe recount (C2, C7) | 1 | 4 measurements, about 3 GiB of reads |
| untar on Fir (C11) | 9 | 6 jobs x 1 CPU x about 1.5 h; not measured |
| recount, all views (C11) | 46 to 140 | measured rate: Nibi job 27662614, 21 tasks, 33,298 CPU-s = 9.25 CPU-h for 116.9 GiB of reads (`recount_plan.tsv`), i.e., 0.079 CPU-h per GiB; 578.7 + 18 GiB x 0.079 = 47. That scan wrote region files only; a genome-wide write, three hashes and the gate read are allowed a factor of up to 3. C7 measures the real rate. |
| read-start average file (C11) | 10 | not measured; one pass over the base views |
| tests on the cluster, training probe (C10, C12) | 12 | 2 suite runs of 8 CPUs x 0.5 h, one 2,000-window probe |
| training, 3 seeds (C12) | 163 to 238 | Fir jobs 5264674, 5264680, 5264687: 254,664 + 46,752 + 285,456 CPU-s = 163.0 CPU-h. Seed 1 took 13.0 CPU-h against 70.7 and 79.3, so three seeds at the slow rate cost 3 x 79.3 = 238. |
| 16-region scoring, 3 seeds (C13) | 127 to 150 | Fir jobs 5264675 to 5264692 without the training jobs: 163,976 + 135,504 + 156,480 CPU-s = 126.7 CPU-h; the adapter's store reads and conversion are allowed 20% |
| sanity check (C13) | 5 | 3 jobs, not measured |
| deviance, 3 seeds (C14) | 36 to 119 | Fir jobs 5381558 to 5381572: 429,464 CPU-s = 119.3 CPU-h with its own predict pass; the metrics jobs alone are 116,976 CPU-s = 32.5 CPU-h when the store of C13 is reused |
| total | 409 to 684 | sum of the rows |

The figures of I section 3 against these: training 163 and scoring 127 are confirmed. The recount figure of 180 is not: the measured rate times the real read volume gives 47, and 180 is above the 3x allowance.
The top of the range is above the 600 CPU-h stop of R15. The order of work keeps the headline table (C13) inside 600 in every case: 1 + 9 + 140 + 10 + 12 + 238 + 150 + 5 = 565. The deviance pass runs last and reuses the kept prediction store first.
The 256-region scoring (about 900 CPU-h) is outside this plan.

Wall time: day 1, waves 1 to 3, with the gate result on four measurements; day 2, waves 4 and 5, and the build as soon as the tars are on Fir (about 6 h at 12 array tasks of 2 CPUs); day 3, training (about 6 to 7 h per seed, the three in parallel) and scoring (about 2 h); day 3 to 4, the deliverable.

## Log
