# Configs of the region experiments (Amendment 7, sections B.1 to B.4)

These configs belong to the region experiments of the onion plan. No config here is a ladder arm.
`ladder_arms.py`, `check-rung` and `check-done` do not read this folder. Each config starts from a
ladder arm config `tools/minicandi/configs/arms/rung<r>/<arm>.json`. It changes only the keys that
are named below. No config changes the model, the loss or `param_budget`.
`tests/test_minicandi_region_configs.py` checks each of these rules.

Paths: `O=/project/def-maxwl/mforooz/ONION`. The runs go under `RR=$O/runs_region`. The region
lists are written by `tools/onion_data/train_strata.py` (chunk STt) under `$O/region/lists/`. In
a `train_regions` path, the trainer replaces the text `{seed}` with the training seed (pinned
interface I, chunk TR).

## `budget/` (B.1, the budget curve; 30 files)

The file `budget/rung<r>_<arm>_w<b>.json` is `arms/rung<r>/<arm>.json` with `windows_seen = b`.
The budget b is 20000 or 40000. The arms are the equal-budget arms of rung 1 (7 arms) and of
rung 4 (8 arms). The arm `matched_compute` is left out. The count is (7 + 8) x 2 = 30 files.
The run name is `a1_rung<r>_<arm>_w<b>`. The 80,000-window runs are the ladder's own runs.

## `region80k/` (B.2 territory and B.3 unseen loci; 80,000 windows; 8 files)

Each file is `arms/rung4/base.json` plus one key, or no key for `today`. The run name is
`rg_<arm>`.

| arm | added key | value |
|---|---|---|
| `terr_r64`, `terr_r256`, `terr_r1024`, `terr_r4096` | `train_regions` | `$O/region/lists/terr_r<R>_s{seed}.tsv` |
| `blocks_minus_eval` | `train_regions` | `$O/region/lists/blocks_minus_eval.tsv` |
| `eval_heavy` | `train_regions` | `$O/region/lists/eval_heavy.tsv` |
| `no_eval` | `exclude_regions` | `$O/eval/regions_v1/regions.json` |
| `today` | none | the ladder's sampler, unchanged |

## `selection20k/` (B.4 selection at a limited budget; 20,000 windows; 12 files)

Each file is `budget/rung4_base_w20000.json` plus `train_regions`. The run name is `rg_<arm>`.
Section B.4 was amended on 2026-11-13. The seven arms that run are the two controls
`tile_uniform` and `terr_r1024_w20k` and the five arms `ccredens_e50_w`, `ccredens_e25_w`,
`ccredens_e50_u`, `variance_e50_w` and `kcenter`. Five older files stay on disk and are not run.

| arm | `train_regions` | status |
|---|---|---|
| `tile_uniform` | `$O/region/lists/tile_uniform.tsv` | control |
| `terr_r1024_w20k` | `$O/region/lists/terr_r1024_s{seed}.tsv` | control (the comparator of `kcenter`) |
| `ccredens_e50_w` | `$O/region/lists/ccredens_e50_w.tsv` | run |
| `ccredens_e25_w` | `$O/region/lists/ccredens_e25_w.tsv` | run |
| `ccredens_e50_u` | `$O/region/lists/ccredens_e50_u.tsv` | run (the unweighted control) |
| `variance_e50_w` | `$O/region/lists/variance_e50_w.tsv` | run |
| `kcenter` | `$O/region/lists/kcenter.tsv` | run |
| `blocks_minus_eval_w20k` | `$O/region/lists/blocks_minus_eval.tsv` | not run (near-uniform lists; see STRATA_REAL.md) |
| `peakfreq_w` | `$O/region/lists/peakfreq_w.tsv` | not run (near-uniform lists; see STRATA_REAL.md) |
| `peakfreq_u` | `$O/region/lists/peakfreq_u.tsv` | not run (near-uniform lists; see STRATA_REAL.md) |
| `ccre_like_w` | `$O/region/lists/ccre_like_w.tsv` | not run (near-uniform lists; see STRATA_REAL.md) |
| `variance_w` | `$O/region/lists/variance_w.tsv` | not run (near-uniform lists; see STRATA_REAL.md) |

The control `tile_uniform` samples the tile universe of the weighted tile arms uniformly, with q in
proportion to valid bins and w = 1. It has the same window-in-tile edge effect as the tile arms.
It replaces `blocks_minus_eval_w20k`. The evidence for "near-uniform" is
`cruxvault/results/t292/STRATA_REAL.md`.

## `confirm80k/` (the conditional confirmation; 80,000 windows; 9 files)

The file `confirm80k/<arm>_w80k.json` is `arms/rung4/base.json` plus the `train_regions` key of
the selection arm `<arm>`. There is one file for each of the five arms that run
(`ccredens_e50_w`, `ccredens_e25_w`, `ccredens_e50_u`, `variance_e50_w`, `kcenter`). The four
files of `peakfreq_w`, `peakfreq_u`, `ccre_like_w` and `variance_w` stay on disk and are not run
(near-uniform lists; see STRATA_REAL.md). The folder holds no control: the confirmation compares
against `region80k/blocks_minus_eval.json`. Chunk RTc runs only the best winner of B.4, and only
if B.4 names a winner. The run name is `rg_<arm>_w80k`.
