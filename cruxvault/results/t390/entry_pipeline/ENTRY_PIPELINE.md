# EIC entry pipeline (t390): score miniCANDI blind predictions on MOESM2 and MOESM4

Written 2026-11-23 on branch `implementation/t390-eic-pvalue-entry`.
Code: `tools/eic_entry/score_entry.py` (`run`, `collect`, `make-test`, `check`) and `tools/eic_entry/run_entry.sh` (Nibi driver).
The files in this folder come from `bash runall.sh`.

## What it does

The input is the folder that `tools/minicandi/predict.py` writes, plus a covariates TSV.
predict.py writes `counts/<eic_track>.h5` with groups `mu` and `n` (NB mean and size), one float16 dataset per chromosome, on the floor(bp / 25) grid (`predict.py` `_write_counts`).
It also writes its own signal tracks (`signal/`, `signal_plug_in/`), but those use the covariates of the tables' `measurements.tsv` row. The pipeline ignores them and recomputes the signal from `counts/` with the experiment's own metadata.
predict.py does not save the predicted control that it computes for a ChIP-seq target with no observed control (`predict_chrom` keeps it in memory only).

The covariates TSV has one row per track. Its columns are `track, route (chip | dnase), depth_reads (T), read_length, frag_len (d), control, control_depth_reads, control_read_length`.
`control` is a `.npz` of 25 bp overlap counts (key `<chrom>_cnt`, a `ladder_probe.py starts` file), or a `.h5` with group `mu` (a predicted control), or `none` (DNase-seq).

Each track goes through five steps:

1. **NB to counts.** The default takes the mean `mu` (`--mode plug_in`). `--mode expectation` gives the expected -log10 p under the NB instead.
2. **Counts to -log10 p.** This is exactly rung L6: `signal_from_counts.signal_track`, pileup `box_d`, `mu + 1e-6`. ChIP-seq uses MACS2 with a control. DNase-seq uses `macs2_no_control`.
   - The control is the predicted control when the row gives a `.h5`, labelled `predicted_control`. With a `.npz`, it is the experiment's observed control counts, labelled `observed_control`. This is the rule of rung L6, and it is the one the code supports. The code has no "generic control" rule (Max's sprint 10 mean of training controls).
   - In this test, ChIP-seq used `observed_control` and DNase-seq used `local_background`.
3. **MOESM2 (headline, PI decision 2026-11-23).** The raw track is scored against the original 2019 truth (`eic_score.ORIGINAL_TRUTH`).
4. **MOESM4.** This is rung L7. The raw track gets peaks called on itself (`ladder_probe.py callpeaks`: threshold 2, max gap = read length, min length = d, cap 500K ChIP-seq / 300K DNase-seq). Then it gets Max's quantile normalization (QN) with `ref_M##.npz`, and is scored against the qnorm truth (Zenodo 14736610).
   - The half-read overlap rule (rung L7o) needs reads. It is not available for a mean prediction.
5. **Placement (`collect`).** For each measure, the pipeline gives the rank among the 25 published entrants (chromosome set all23). It also gives the organizers' combined rank of the entry added alone to the published field, over bootstraps 0-9.
   - Entrant scores, MOESM4: `/scratch/mforooz/onion/eic/score_cache/qnorm/scores/team_*` (as `run_probe.sh`).
   - Entrant scores, MOESM2: Max's `/project/def-maxwl/maxwl/epi_imputation/s04/scores/pe/<entrant>/<exp>.csv` (as `run_moesm2.sh`).
   - Published rows: the sha256-pinned MOESM2 and MOESM4 tables.

Outputs in `<out>`:
- `board_moesm2.tsv` and `board_moesm4.tsv` (one row per experiment and measure, plus a `combined_rank` row)
- `SUMMARY.md`
- `raw/`, `qn/`, `peaks/`, `scores/` (the arrays stay on Nibi)

## Commands when predictions land

```bash
# on the Mac, in the worktree
bash tools/eic_entry/run_entry.sh deploy
# on Nibi: <pred> = predict.py --out folder; <cov.tsv> = one row per track; a NEW <out> per prediction set
bash /scratch/mforooz/eicx/entry/code/tools/eic_entry/run_entry.sh submit <pred> <cov.tsv> /scratch/mforooz/eicx/entry/<run name> C19M22 C40M17 C12M02
# optional: MODE=expectation before the command gives the expected -log10 p in step 1
```

For the 3 probe experiments, the covariates rows already exist. Build `<cov.tsv>` from them:
`(head -1 /scratch/mforooz/eicx/entry/test/cov/C19M22.tsv; for e in C19M22 C40M17 C12M02; do tail -1 /scratch/mforooz/eicx/entry/test/cov/$e.tsv; done) > cov.tsv`.
The `submit` step makes one job per track (2 CPUs, 48G, about 22-40 min) and then one `collect` job.

## Reproduction check (observed counts in place of predictions)

`run_entry.sh test` wrote each experiment's observed 25 bp overlap counts in predict.py's layout (`make-test`). All counts were <= 2048, so float16 holds them exactly, and the test used float16 as predict.py does.
It then ran the same `run`, and `check` compared the result with probe 1 rung L6 (`/scratch/mforooz/eicx/ladder/rungs/L6`) and rung L7 (`rungs/L7`).
It also compared the scores: rung L6's MOESM2 scores (`/scratch/mforooz/eicx/moesm2/scores/L6.tsv`) and rung L7's qn scores (`/scratch/mforooz/eicx/ladder/scores/qn/L7.tsv`).
Source: `check_<exp>.tsv`.

| check | C19M22 | C40M17 | C12M02 |
|---|---|---|---|
| raw track vs rung L6, bins differing (of 121,241,707) | 0 | 0 | 0 |
| QN track vs rung L7, bins differing | 0 | 0 | 0 |
| MOESM2 scores vs rung L6 MOESM2, max relative diff (99 values: 9 measures x 11 chromosome sets) | 0 | 0 | 0 |
| MOESM4 scores vs probe-1 rung L7 qn, max relative diff (99 values) | 0 | 0 | 0 |

All 12 checks are exact, against a target of 1e-9. The raw track equals rung L6 bit for bit, so its se and qn scores also equal rung L6's, because the scorer is deterministic. The placements in `SUMMARY.md` equal those in `../moesm2/MOESM2.md` (rung L6) and `../ladder_probe/LADDER_PROBE.md` (rung L7). For example, C19M22 MOESM2 has combined rank 15 (15-15), and mse is rank 19.

## Placement of the test entry (observed reads, an oracle bound, not a model result)

| board | C19M22 | C40M17 | C12M02 |
|---|---|---|---|
| MOESM2 combined rank, mean over bootstraps (of 26) | 15 | 1 | 17.1 |
| MOESM4 combined rank (of 26) | 1 | 1 | 1 |

## Cost and jobs (Nibi, 2 CPUs per job, `sacct.tsv`)

| job | id | elapsed | peak RSS |
|---|---|---|---|
| eicx_entry_test_C19M22 (probe) | 28666332 | 39:30 | 23.0 GB |
| eicx_entry_collect | 28666333 | 0:11 | 0.2 GB |
| eicx_entry_test_C40M17 | 28668387 | 25:32 | 24.1 GB |
| eicx_entry_test_C12M02 | 28668388 | 22:01 | 23.4 GB |
| eicx_entry_collect | 28668389 | 0:17 | 0.2 GB |

Total CPU time was (39:30 + 0:11 + 25:32 + 22:01 + 0:17) = 87.5 min x 2 CPUs = 2.92 CPU-h, against a budget of about 3.

Of the 39.5 min of the C19M22 probe job:
- conversion: 41 s
- MOESM2 scoring: 576 s
- peak calls and QN: about 14 min
- MOESM4 scoring: 898 s

The scorings ran 2-4 times slower than in probe 1 (216-264 s). The cause was not investigated.

All 48 experiments would cost about 48 x 0.5 h x 2 CPUs, about 48 CPU-h. That run was not done (PI hold).

## Gaps

- **Predicted control.** predict.py does not write the predicted control to disk. A ChIP-seq prediction with no `.h5` control must use the observed control counts (`.npz`), which is the L6 rule.
  - `.npz` control counts exist only for the controls of the 3 probe experiments (`/scratch/mforooz/eicx/ladder/starts/`).
  - The other blind experiments need `ladder_probe.py starts` on their control tagAligns first.
  - The generic control is not implemented.
- **Covariates.** They must come from each experiment's pipeline `metadata.json`: T = the subsample line count, the mean read length, and d. They exist only for the 3 probe experiments.
- **Where the predictions are.** The pipeline runs on Nibi. If the recipe session writes the predictions on Fir, the `counts/` files must be moved to Nibi first. They are about 0.5 GB per track at float16.
- **float16.** float16 is exact for counts <= 2048. Larger predicted means are rounded by predict.py, not by this pipeline.
- **Untested options.** `--mode expectation` and a `.h5` predicted control were not exercised by the test, because observed counts have no NB size and no predicted control.
- **QN peaks.** QN uses the single-pass threshold caller (rung L7), not the half-read overlap rule. Probe 2 showed that the half-read rule (L7o) scores within 8% of the oracle peak set on qn mse.
- **Purge exposure.** The original truth, its binned cache and the QN references live on Max's `/scratch`, so they can be purged. (Max's pe entrant scores are on `/project`.) The test outputs live under `/scratch/mforooz/eicx/entry/test`, which can also be purged.
