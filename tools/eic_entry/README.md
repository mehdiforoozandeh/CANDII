# tools/eic_entry

Code for crux task t390 (a fair, tested way to enter a count-space model on the EIC board).

- `ladder_probe.py`: the observed-data ladder (rungs L1 to L7, no model). Subcommands `starts`, `formula`, `counts_route`, `qnorm`, `callpeaks`, `peakcheck`, `compare`, `entrants`, `placement`. The MACS2 formula and the quantile normalization are ported from Max Libbrecht's sprints 10 and 04 (credits in the module docstring).
- `run_probe.sh`: the Nibi driver. `deploy` (rsync from the Mac), `setup` (SE-truth score cache and entrant SE scores), `exp <C##M##>` (the sbatch chain for one experiment), `sacct <ids>` (wall time and peak memory).
- `collect.py`: builds the probe's TSVs and tables from the pulled small files (scores, comparisons, placements, sacct).

Results: `cruxvault/results/t390/ladder_probe/LADDER_PROBE.md`.
- `run_moesm2.sh`, `moesm2.py`: the MOESM2 board (original 2019 paired-end truth, `eic_score.ORIGINAL_TRUTH`): verify on 3 experiments, score rungs L3/L5/L6/L7, collect. Results: `cruxvault/results/t390/moesm2/MOESM2.md`.
- `score_entry.py`: the entry pipeline for miniCANDI blind predictions (predict.py's `counts/<track>.h5`): NB mean -> rung-L6 count route -> MOESM2 (raw vs the original truth) and MOESM4 (QN with own peaks vs the qnorm truth) scores, then placement among the 25 entrants (`run`, `collect`); `make-test` and `check` test it on observed counts against rungs L6 / L7.
- `run_entry.sh`: its Nibi driver. `deploy`, `submit <pred> <cov.tsv> <out>` (one job per track, then collect), `test <out>` (the observed-count check), `sacct`. Results: `cruxvault/results/t390/entry_pipeline/ENTRY_PIPELINE.md`.
