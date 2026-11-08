# Where these three tables come from

Made by `tools/eic_decomp/` (task t148) on Nibi, 2026-11-07: Slurm array 26430723 (216 tasks = 27 entries x
8 assays, all COMPLETED), then `aggregate.py /scratch/mforooz/eic_decomp/out agg`. Per-task outputs (Gram
matrices, per-track terms) stay on Nibi under `/scratch/mforooz/eic_decomp/out/<truth>/<assay>/`.

Inputs are Max Libbrecht's sprint 04 EIC reproduction (`/scratch/maxwl/epi_imputation/s04`): his prediction
reader, his tcfloor map for CANDI, and his three cached truth layers. The method is his sprint 06 error
decomposition (`errdecomp.py`, copied unchanged).

| truth | layer | board |
|---|---|---|
| `pe` | original 2019 truth | MOESM2 |
| `pq` | original truth, quantile-normalized | MOESM3 |
| `qn` | single-end reprocessed + quantile-normalized (Zenodo 14736610) | MOESM4 |

| file | one row per | columns |
|---|---|---|
| `tracks.tsv.gz` | truth x entry x track (4128 = 3 x 1376) | calibration terms, Haar-band energies |
| `summary.tsv` | truth x entry x assay (+ `all`, track-weighted) | mse = level + shared + specific; level = offset + scale + nonlinear; shared/specific per band (fine, peak, domain, broad) |
| `components.tsv` | truth x entry x assay x part x band | V, explained, unexplained, added, E |

Unit: asinh of the track value, as a fraction of each truth track's variance (so 1.0 = predicting the mean).
There is no noise-floor term: the EIC truth has no replicates. 1376 tracks, not 1377 (27 x 51): UIOWA Michaelson
Lab has no submitted track for one experiment.
