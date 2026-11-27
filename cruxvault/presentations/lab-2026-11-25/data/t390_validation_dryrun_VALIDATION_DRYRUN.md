# EIC entry pipeline: validation dry run (t390)

Written 2026-11-24 on branch `implementation/t390-eic-pvalue-entry`.
Code: `tools/eic_entry/score_entry.py` (`--board validation`, `val-cov`, `prep-val`, `average`, `check-val`, `make-test --from-cov`) and `tools/eic_entry/run_entry.sh` (`val-prep`, `val-test`, `BOARD=validation submit`).
The files in this folder come from `bash runall.sh`. This page is hand-written.

## What the validation board is

- The EIC validation set has 45 experiments. Their truth bigWigs are the organizers' original 2019 processing, like the MOESM2 truth.
  - Nibi: `/scratch/maxwl/epi_imputation/data/synapse/validation_data/` (Max's /scratch, which can be purged).
  - Fir: `/project/def-maxwl/mforooz/DATA_EIC_SYNAPSE/validation_data/`.
  - The list is in `metadata_validation_data.tsv` in the same folder.
- No QN truth exists for the validation set. The pipeline therefore scores only the raw -log10 p track, and it skips step 4 (MOESM4).
- No entrant validation submissions exist anywhere we looked:
  - Max's `031/submissions` on Nibi /project holds the blind tracks only.
  - Max's `001/scores` on Fir /scratch holds blind scores only.
- The paper's Additional file 6 (MOESM6, sha256 `151a5c3c…c1cf`) is the round-1 validation leaderboard, as Max's notebook `experiments/001_2026-07-27_eic_leaderboard_repro/notebook.md` identified.
  - It holds 45 experiments x 10 entries x bootstraps 0-9, with the nine measures and no all23 row.
  - The 10 entries are 8 teams plus Average and Avocado_p0.
  - It is copied to `/project/def-maxwl/mforooz/ONION/eic/published/` on Nibi and on Fir.
- Placement is therefore possible, against MOESM6's published scores.
  - The rank per measure compares the mean over bootstraps 0-9, because MOESM6 has no all23 row.
  - The combined rank is the organizers' statistic per bootstrap.
  - A leaderboard position is given only when all 45 experiments are scored.

## Covariates (`cov.tsv`, 42 of 45 tracks)

`val-cov` takes the pooled base measurement of each track from `ONION/tables/eic_v1_2/measurements.tsv`. It uses the uniform layer when it exists, else the native layer (6 tracks).
- T and the read length come from that row.
- d comes from `frag_len`, or, when that is empty, from `eic_v1_fraglen.tsv`. The fragment-length table is the same file on both clusters (sha256 `4f2efdb6…`); on Fir it is at `ONION/e2/`. 32 of 41 ChIP-seq tracks take d from the table's `signal_provenance` estimate, which is the d the store's own MACS2 run used.
- The control is the control measurement's observed store counts (`counts.h5`, store overlap rule), with that measurement's depth and read length. Every ChIP-seq track has a control. `_control` now reads a store `counts.h5`, labelled `observed_control`.
- Missing (`cov.tsv.missing.tsv`): C25M26, C34M02, C50M02. Each has no pooled base measurement, because it was never built into the store (`missing.tsv`: "not built (no provenance.json)"). Only one DNase-seq track (C03M02) is left.
- Control gap: for 14 of the 41 ChIP-seq tracks, the store's MACS2 `signal.h5` pooled two or three controls. The tables name only the first of them, so `cov.tsv` gives only that one. The tracks are C09M20, C17M04, C17M19, C17M29, C17M32, C18M21, C18M25, C23M07, C23M26, C23M34, C27M24, C32M20, C46M18 and C46M21. Neither the model nor this pipeline sees the other controls.

## Reproduction checks

### The board scorer reproduces MOESM6 (`test/avg/check_average_*.tsv`)

`average` rebuilt the round-1 Average baseline, i.e. the per-bin mean of the training bigWigs of the assay. It scored the baseline on the validation board and compared it with MOESM6's Average rows over bootstraps 0-9.

| measure | C17M04 (4 training tracks), max relative difference | C12M32 (3 training tracks), max relative difference |
|---|---|---|
| mse, gwcorr, gwspear, mseprom, msegene, mseenh, mse1obs, mse1imp | <= 1.25e-6 | <= 1.07e-6 |
| msevar, weights = validation truth (the board's default) | undefined (1 validation experiment) | 0.605 |
| msevar, weights = training tracks | 0.422 | 0.671 |
| msevar, weights = training + validation | 0.101 | 0.0735 |

Eight measures reproduce MOESM6. No weight vector we tried reproduces msevar; the closest, training + validation, is 7-10% off. The weights are therefore not the training variance, so binning the 268 training tracks would not fix msevar. Eleven assays also have only one validation experiment, so the validation-truth variance is undefined for them.
- Decision: on the validation board, msevar is reported but never ranked. The combined rank and the leaderboard position use the other eight measures (`VAL_MEASURES`).

### Observed counts in place of predictions (`test/check_*.tsv`, `test/SUMMARY.md`)

`val-test` wrote each track's observed store counts in predict.py's layout. It then ran the same `run --board validation` and compared three tracks with nMSE = mean((A - B)^2) / var(B) on the scorer grid (blacklist bins deleted, 121,240,998 bins):
- raw = the pipeline's count route on the observed counts;
- signal = the store's own MACS2 `signal.h5` of the same reads, i.e. the exact formula;
- truth = the 2019 validation truth.

| pair | C17M04 (H2AK5ac, narrow) | C12M32 (H4K20me1, broad) | C03M02 (DNase-seq) |
|---|---|---|---|
| raw vs signal (cost of the 25 bp count route) | 0.367 | 0.136 | 0.0118 |
| signal vs truth (our reprocessing vs the 2019 processing) | 0.715 | 0.347 | 0.349 |
| raw vs truth | 0.798 | 0.487 | 0.341 |

- For reference, the blind ladder's count-route cost (L6 vs L3) was 0.017 (H3K4me3), 0.255 (H3K27me3) and 0.021 (DNase-seq).
- C17M04 costs more than any blind experiment. It is one of the 14 tracks above: its `signal.h5` pooled 3 controls (ENCSR131IUA, ENCSR569NEE, ENCSR814EAU), and the count route had only ENCSR131IUA. The control difference is the likely cause; it was not isolated.
- C03M02's signal-vs-truth nMSE of 0.349 comes from a few extreme bins: the max |difference| is 3528, and the asinh nMSE is 0.056.

Placement of the observed-count entry among MOESM6's 10 entries. This is an oracle bound from observed reads, not a model result.

| experiment | combined rank (of 11; 8 measures) | per-measure ranks (mse, gwcorr, gwspear, mseprom, msegene, mseenh, mse1obs, mse1imp) |
|---|---|---|
| C03M02 | 1 (1-1) | 1, 1, 1, 2, 1, 1, 1, 1 |
| C12M32 | 1 (1-1) | 1, 1, 1, 1, 1, 1, 1, 10 |
| C17M04 | 2 (2-2) | 2, 1, 1, 2, 2, 2, 1, 6 |

### Blind behaviour unchanged (`test/blind_recheck_C12M02.tsv`)

`run_entry.sh test` on C12M02 with the changed code was exact on 4 of 4 checks: raw = rung L6, QN = rung L7, and both score sets have a relative difference of 0.

### The same test on Fir

The validation predictions will be written on Fir, so the board was rebuilt there from Fir's own copies; no data was copied from Nibi to Fir, apart from the 0.75 MB MOESM6 table (sha256 checked).
- The run used the truth at `/project/def-maxwl/mforooz/DATA_EIC_SYNAPSE`, the tables at `ONION/tables/eic_v1_2`, the fragment-length table at `ONION/e2/eic_v1_fraglen.tsv`, and the store at `/scratch/mforooz/onion/measurements`. `run_entry.sh` sets `EIC_CLUSTER=fir` from the hostname.
- `fir/cov.tsv` is byte-identical to `cov.tsv` (sha256 `56a43e9e…`), and every file it names exists on Fir.
- The C17M04 observed-count test on Fir (`fir/test/check_C17M04.tsv`) gave the same nMSE values as Nibi to every printed digit.
- Its 99 score values agree with Nibi's to 5.3e-13 relative. The TSV bytes differ only in the last digits of the floats.
- The Fir test used the same `signal_from_counts.py` as Nibi (`71df124d`).

## Versions

- The Nibi tests (jobs below) and the blind recheck used `tools/onion_eval/signal_from_counts.py` at commit `71df124d` (sha256 `ae97586a…`, SPAN_EXTRA = 24).
- Commit `0f957f05` (SPAN_EXTRA = 25) landed while they ran. Scores of a rerun may shift by about 1%.
- Deploy ships the worktree file. Deploy from a clean tree, so that the deployed `signal_from_counts.py` is the commit you mean.

## Cost (`test/sacct.tsv`, Nibi, 2 CPUs per job)

| job | id | elapsed | peak RSS |
|---|---|---|---|
| eicx_val_prep (val-cov, 45 truth tracks binned) | 28760151 | 14:08 | 48.0 GB (request 48G; now 64G) |
| eicx_val_test_C17M04 | 28760152 | 5:24 | 15.1 GB |
| eicx_val_avg_C17M04 | 28760153 | 6:23 | 20.3 GB |
| eicx_val_test_C12M32 | 28760154 | 6:05 | 16.4 GB |
| eicx_val_avg_C12M32 | 28760155 | 6:37 | 20.3 GB |
| eicx_val_test_C03M02 | 28760156 | 5:07 | 16.3 GB |
| eicx_entry_test_C12M02 (blind recheck) | 28760194 | 11:22 | 23.2 GB |

Total: 55:06 min x 2 CPUs = 1.84 CPU-h. The queued collect jobs (28760157, 28760195) were cancelled. The collect step ran on the Nibi login node (a few seconds).

Fir (`fir/test/sacct.tsv`): on Fir, `run_entry.sh` asks for one core per 4000 MB, so that jobs stay on cpubase.

| job | id | elapsed | CPUs | peak RSS |
|---|---|---|---|---|
| eicx_val_prep | 5703770 | 12:12 | 17 | 36.8 GB |
| eicx_val_test_C17M04 | 5703771 | 4:30 | 13 | 15.4 GB |
| eicx_entry_collect | 5703780 | 0:18 | 3 | 2.4 GB |

Fir total: 12:12 x 17 + 4:30 x 13 + 0:18 x 3 = 266.8 CPU-min = 4.45 CPU-h of allocation.
On Fir, one prediction track costs about 13 CPUs x 5 min = 1.1 CPU-h, so 42 tracks cost about 46 CPU-h of allocation. Lowering the 48G request to 24G, which is above the 16 GB peak of a validation run, would halve that.

## Commands when the validation predictions land

predict.py `--split val` writes `counts/<eic_track>.h5`, named C##M##, which matches `cov.tsv`.

```bash
# on the Mac, in the worktree, with a clean tree
bash tools/eic_entry/run_entry.sh deploy fir        # or: deploy nibi
# on Fir (or Nibi): cov.tsv is already built at /scratch/mforooz/eicx/val/cov.tsv on both clusters
BOARD=validation bash /scratch/mforooz/eicx/entry/code/tools/eic_entry/run_entry.sh submit \
    <predict.py --out dir> /scratch/mforooz/eicx/val/cov.tsv /scratch/mforooz/eicx/val/<run name>
# optional: MODE=expectation before the command gives the expected -log10 p
```

The truth cache (`/scratch/mforooz/eicx/val/score_cache`) is already built on Nibi and on Fir. If it is purged, run `bash run_entry.sh val-prep` first, and pass DEP=<its job id> to wait for it.
- Output: `<out>/board_validation.tsv` and `<out>/SUMMARY.md`.
- A leaderboard position appears when all 45 experiments are scored. Only 42 can be, because 3 are not in the store, so the position will not appear without a rule for the missing three.

## Gaps

- 3 validation tracks are not in the store (C25M26, C34M02, C50M02), so there can be no prediction and no leaderboard position.
- msevar on the validation board is not reproduced and is not ranked.
- 14 ChIP-seq tracks have pooled controls that the tables do not record.
- On Nibi, the validation truth lives only on Max's /scratch, which can be purged. Fir's copy is on /project.
- `--mode expectation` and a predicted control are still untested, as in the blind test.
