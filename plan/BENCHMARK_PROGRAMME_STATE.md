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

Rendered 2026-09-01 from `cruxvault/results/t81/state/*.tsv` — 19 training units,
seeded from §4.2 under D1. The convention and the one-line renderer are in
`cruxvault/results/t81/state/README.md`.

| unit | phase | slurm job | state | artifact path | number | date |
|---|---|---|---|---|---|---|
| train.Avocado.eic_19.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.Avocado.eic_pilot.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.CANDI.eic_19.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.CANDI.eic_19.s1 | train | - | blocked | - | - | 2026-09-01 |
| train.CANDI.eic_pilot.s0 | train | 57674899_1 | done | /project/def-maxwl/mforooz/t81_checkpoints/t81_eic_pilot_s0.best.ckpt | impute macro CRPS 0.5386 (n=45, chr20-22, full coverage; selected epoch 5) | 2026-09-01 |
| train.ChromImpute.eic_19.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.ChromImpute.eic_pilot.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.Lavawizard.eic_19.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.Lavawizard.eic_pilot.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.avg-arcsinh.eic_19.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.avg.eic_19.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.eDICE.eic_19.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.eDICE.eic_pilot.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.knn1.eic_19.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.knn1.eic_pilot.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.knn5.eic_19.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.knn5.eic_pilot.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.marginal.eic_19.s0 | train | - | blocked | - | - | 2026-09-01 |
| train.marginal.eic_pilot.s0 | train | - | blocked | - | - | 2026-09-01 |
