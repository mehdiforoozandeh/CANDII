# tools/eic_entry

Code for crux task t390 (a fair, tested way to enter a count-space model on the EIC board).

- `ladder_probe.py`: the observed-data ladder (rungs L1 to L7, no model). Subcommands `starts`, `formula`, `counts_route`, `qnorm`, `callpeaks`, `peakcheck`, `compare`, `entrants`, `placement`. The MACS2 formula and the quantile normalization are ported from Max Libbrecht's sprints 10 and 04 (credits in the module docstring).
- `run_probe.sh`: the Nibi driver. `deploy` (rsync from the Mac), `setup` (SE-truth score cache and entrant SE scores), `exp <C##M##>` (the sbatch chain for one experiment), `sacct <ids>` (wall time and peak memory).
- `collect.py`: builds the probe's TSVs and tables from the pulled small files (scores, comparisons, placements, sacct).

Results: `cruxvault/results/t390/ladder_probe/LADDER_PROBE.md`.
