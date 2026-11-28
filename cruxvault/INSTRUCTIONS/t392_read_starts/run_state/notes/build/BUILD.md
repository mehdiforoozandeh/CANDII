# t392 C11b: the read-start store build on Fir

Collected 2026-11-24 from Fir, read-only. No job was submitted. No file was deleted.
The store root is `/scratch/mforooz/rs392/store`. The tables are `/project/def-maxwl/mforooz/ONION/tables/eic_v1_2`.

## Sources

- `gate.tsv` and `floor.tsv` come from `<store>/<id>/provenance.json` of each of the 2,740 rows of `measurements.tsv`. The gate is `outputs[<file>].gate` (`n_bins`, `n_differing`). The floor is `outputs[<file>].conversion_floor_nmse` (`tools/onion_data/read_starts.py::run`).
- The program ran on the Fir login node under `nice`, with the standard library only.
- `assay`, `assay_class`, `view` and `view_value` come from `measurements.tsv` columns `assay`, `assay_class`, `arm_knob` and `arm_value`. An empty `arm_knob` is the base view; `floor.tsv` writes it as `base`.
- The view counts agree with the plan (Clusters section): 785 base, 362 nodedup, 1,086 depth, 363 mapq, 83 crop, 49 se, 12 abproxy.
- `floor.tsv` has the 6 columns of the interface first. It then has `assay_class` and `view_value` (for example `0.125` for a depth view).

## Gate

The gate passed on every file. 8,220 of 8,220 files have `n_differing` = 0. The files are 2,740 each of `starts.h5`, `starts_pr1.h5` and `starts_pr2.h5`. The gate compared 1,014,887,363,136 bins in all. Every `provenance.json` has `gate_passed` true. This agrees with `check.py store` (2,740 of 2,740 measurements, 0 differing bins). R7 does not stop the run.

## Conversion floor

The floor is the nMSE of the read starts convolved with the span kernel against the overlap counts (D9). Lower is better. The table has no ChIP TF class. The two histone classes are given apart and together.

Per assay class:

| class | n | file | median | p5 | p95 | max |
|---|---|---|---|---|---|---|
| ChIP histone, narrow | 1254 | starts | 0.0289 | 0.00116 | 0.0896 | 0.106 |
| | | starts_pr1 | 0.0445 | 0.00207 | 0.0973 | 0.106 |
| | | starts_pr2 | 0.0446 | 0.00207 | 0.0973 | 0.106 |
| ChIP histone, broad | 901 | starts | 0.0339 | 0.00260 | 0.0955 | 0.145 |
| | | starts_pr1 | 0.0498 | 0.00409 | 0.102 | 0.152 |
| | | starts_pr2 | 0.0498 | 0.00409 | 0.102 | 0.152 |
| ChIP histone, all | 2155 | starts | 0.0307 | 0.00145 | 0.0918 | 0.145 |
| | | starts_pr1 | 0.0459 | 0.00246 | 0.0993 | 0.152 |
| | | starts_pr2 | 0.0459 | 0.00246 | 0.0993 | 0.152 |
| ChIP control | 408 | starts | 0.0722 | 0.00343 | 0.105 | 0.148 |
| | | starts_pr1 | 0.0860 | 0.00501 | 0.106 | 0.151 |
| | | starts_pr2 | 0.0859 | 0.00502 | 0.106 | 0.150 |
| DNase | 147 | starts | 0.0137 | 0.00369 | 0.0619 | 0.0931 |
| | | starts_pr1 | 0.0236 | 0.00585 | 0.0781 | 0.0998 |
| | | starts_pr2 | 0.0235 | 0.00585 | 0.0780 | 0.0998 |
| ATAC | 30 | starts | 0.0237 | 0.00447 | 0.0631 | 0.0721 |
| | | starts_pr1 | 0.0312 | 0.00692 | 0.0811 | 0.0928 |
| | | starts_pr2 | 0.0309 | 0.00692 | 0.0807 | 0.0928 |

Per view:

| view | n | file | median | p5 | p95 | max |
|---|---|---|---|---|---|---|
| base | 785 | starts | 0.0303 | 0.00175 | 0.0870 | 0.121 |
| | | starts_pr1 | 0.0472 | 0.00308 | 0.0964 | 0.135 |
| | | starts_pr2 | 0.0471 | 0.00308 | 0.0964 | 0.135 |
| nodedup | 362 | starts | 0.0246 | 0.000921 | 0.0847 | 0.106 |
| | | starts_pr1 | 0.0355 | 0.00156 | 0.0933 | 0.106 |
| | | starts_pr2 | 0.0354 | 0.00157 | 0.0932 | 0.106 |
| depth | 1086 | starts | 0.0556 | 0.00419 | 0.103 | 0.148 |
| | | starts_pr1 | 0.0721 | 0.00583 | 0.105 | 0.152 |
| | | starts_pr2 | 0.0721 | 0.00585 | 0.105 | 0.152 |
| mapq | 363 | starts | 0.0231 | 0.000612 | 0.0688 | 0.0968 |
| | | starts_pr1 | 0.0373 | 0.00118 | 0.0858 | 0.119 |
| | | starts_pr2 | 0.0373 | 0.00118 | 0.0855 | 0.119 |
| crop | 83 | starts | 0.00876 | 0.00154 | 0.0768 | 0.0875 |
| | | starts_pr1 | 0.0107 | 0.00274 | 0.0897 | 0.0965 |
| | | starts_pr2 | 0.0107 | 0.00274 | 0.0896 | 0.0965 |
| se | 49 | starts | 0.00515 | 0.000508 | 0.0382 | 0.0721 |
| | | starts_pr1 | 0.00901 | 0.000943 | 0.0564 | 0.0928 |
| | | starts_pr2 | 0.00901 | 0.000942 | 0.0563 | 0.0928 |
| abproxy | 12 | starts | 0.00509 | 0.00370 | 0.00553 | 0.00557 |
| | | starts_pr1 | 0.00613 | 0.00497 | 0.00644 | 0.00647 |
| | | starts_pr2 | 0.00614 | 0.00498 | 0.00644 | 0.00647 |

Over all 2,740 `starts.h5` files: median 0.0329, p5 0.00167, p95 0.0982, max 0.148. The percentiles use linear interpolation (the numpy default).
The half files have a higher floor than the pooled file in every class. The halves hold about half the reads.

### Measurements with a floor above 0.1

214 measurements have a floor above 0.1 on at least one of the three files. 102 of them have a floor above 0.1 on `starts.h5`. The maximum on any file is 0.152 (H4K20me1, `ENCSR747ZDF.pooled.uniform.depth0.125`, `starts_pr1.h5`).

- By class: 113 ChIP control, 62 ChIP histone broad, 39 ChIP histone narrow. No DNase or ATAC measurement is above 0.1.
- By view: 191 depth, 12 base, 9 nodedup, 2 mapq. The depth rows are 104 of the 361 at depth 0.125, 62 of the 361 at 0.25 and 25 of the 361 at 0.50. The floor rises as the depth falls.
- By assay: chipseq-control 113, H3K9me3 17, H3K27me3 11, H3K36me3 8, H4K20me1 8, H3K79me1 7, H3K9ac 4, H3K9me2 4, H3K4ac 4, H4K8ac 4, H2BK5ac 3, H3K79me2 3, H2AFZ 3, H2BK120ac 3, H3K9me1 2, H3K4me1 2, H3K27ac 2, H4K91ac 2, H2BK12ac 2, H3K23ac 2, H3T11ph 2, and 1 each of H3K18ac, H3K4me3, H3K23me2, H2BK20ac, H3K4me2, H3K14ac, H2AK5ac, H2BK15ac.
- The full list is the rows of `floor.tsv` with any floor column above 0.1.

## Sizes

- Store: `du -sb /scratch/mforooz/rs392/store` gave 184,495,762,696 bytes (171.8 GiB). `find` counted 13,701 entries. This equals 2,740 x 5 + 1, i.e., per measurement a folder, 3 h5 files and `provenance.json`, plus the root. I did not list the folders one by one.
- Average file: `/scratch/mforooz/rs392/aux/eic_v1_2_read_start_average.h5` is 9,204,908,942 bytes (8.57 GiB). The read-count file `/scratch/mforooz/onion/aux/eic_v1_average.h5` is 7,451,594,429 bytes.

## Average-file check

The reference is `/scratch/mforooz/onion/aux/eic_v1_average.h5`. The `rs392_avg` log does not print it. I read it from the job's submit line (`sacct -j 5575759 -o SubmitLine`) and from the `reference_average_h5` attribute of the new file. Both name the same path.
An h5py program in a venv under `$TMPDIR` compared the two files on the Fir login node. The result:

- Both files hold the same 36 tracks.
- For each of the 36 tracks, `measurement_ids` (same ids, same order) and `n_cells` are equal. The tracks hold 305 ids in all.
- Each track has the same chromosome set in both files.

The check passed. The job wrote the file in 908 s (log line `[rs392] wrote ... in 908 s`).

## Compute

`sacct -X` for the 14 jobs of `jobs.tsv` gave 239 rows: 6 untar jobs, 232 recount tasks and 1 average job. All 239 are COMPLETED. `CPUTimeRAW` is elapsed time x allocated CPUs.

| job | what | CPU-s |
|---|---|---|
| 5575744 | untar 0 | 463 |
| 5575747 | recount tar 0 (45 tasks) | 57,054 |
| 5575748 | untar 1 | 329 |
| 5575749 | recount tar 1 (35 tasks) | 39,756 |
| 5575750 | untar 2 | 396 |
| 5575751 | recount tar 2 (43 tasks) | 51,198 |
| 5575752 | untar 3 | 334 |
| 5575753 | recount tar 3 (40 tasks) | 49,600 |
| 5575754 | untar 4 | 415 |
| 5575755 | recount tar 4 (46 tasks) | 51,562 |
| 5575756 | untar 5 | 132 |
| 5575757 | recount tar 5 (16 tasks) | 17,274 |
| 5575758 | recount testbed (7 tasks) | 8,138 |
| 5575759 | average file | 7,640 |

- Untar: 463 + 329 + 396 + 334 + 415 + 132 = 2,069 CPU-s.
- Recount: 57,054 + 39,756 + 51,198 + 49,600 + 51,562 + 17,274 + 8,138 = 274,582 CPU-s.
- Average: 7,640 CPU-s.
- Sum: 2,069 + 274,582 + 7,640 = 284,291 CPU-s. 284,291 / 3,600 = 79.0 CPU-h.

The plan estimate for the recount was 48 to 143 CPU-h. The measured recount is 274,582 / 3,600 = 76.3 CPU-h, inside that range.

Memory and wall time:

- Peak memory of a recount task: `MaxRSS` 8.38 GB (task 5575747_2). The request was 8000M (8.39 GB). The margin is about 0.1%. No task failed.
- Peak memory of one measurement process: `peak_rss_kb` 4,962,516 (4.96 GB, `ENCSR000DUN.pooled.uniform.base`, from `provenance.json`).
- Peak memory of the average job: `MaxRSS` 16.0 GB (5575759.batch), against a 32000M request.
- Longest recount task: 844 s elapsed (5575751_27), against a 30 min limit.
- Longest single measurement: `wall_s` 218.6 (`ENCSR477RTP.pooled.uniform.nodedup`).

## Fir scratch

`diskusage_report` on 2026-11-24: `/scratch` (user mforooz) 4,327 GiB of 19 TiB, 850K of 1,000K files. `/project` (def-maxwl) 23 of 28 TiB, 4,089K of 5,000K files.
The store holds 13,701 entries of these 850K files.

## Not verified

- I did not read the tar extracts under `/scratch/mforooz/rs392/work`. Their file count is not in this note.
- The floor values are the values that the recount wrote into `provenance.json`. I did not recompute them from the h5 files.
