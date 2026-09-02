# Benchmark programme — tracked state memo

Tracked copy of the unit table in `cruxvault/results/t81/PROGRAMME_STATE.md` (gitignored, per-worktree).
The gitignored file is the live one; this memo is refreshed when a phase closes. Brief:
`plan/HANDOFF_FINISH_BENCHMARK.md`. Branch: `implementation/t81-finish-benchmark` (the PI merges it to main).

## Decisions recorded 2026-09-01

| # | decision |
|---|---|
| D1 | `marginal`, `knn1`, `knn5` run once per regime; `avg`, `avg-arcsinh` once. 18 method-regime units. §12.2/§12.3 to be corrected; identity assertion for `avg`/`avg-arcsinh` to be added. |
| D2 | Build the training-residual σ pass; repoint all four rivals' `fit_sigma.py` and `avg-arcsinh` at it. |
| D3 | Truth toggle does not apply to `V_`. Exclusion recorded on the board. |
| D4 | `eic.19` band sized from a fresh measured epoch rate, `--early-stop-epochs 3`, 25-epoch ceiling. |
| D5 | The 40 `counts.h5.pre_t78p3` backups stay until every retrain is scored. |
| ops | Fir code: fresh clone `/project/def-maxwl/mforooz/CANDII_main` on this branch. Venv: `/project/def-maxwl/mforooz/EpiDenoise/candi_venv`. t90: rsync via a SLURM CPU job, checksum-verified, scratch copy left to purge. |

## Unit table

| unit | phase | slurm job | state | artifact path | number | date |
|---|---|---|---|---|---|---|
| (none yet) | | | | | | |
