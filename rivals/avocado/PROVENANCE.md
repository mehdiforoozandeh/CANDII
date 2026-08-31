# Avocado — vendored, unmodified

Every `.py` in this directory is a **byte-identical copy** of Maxwell Libbrecht's PyTorch
re-implementation of Avocado (Schreiber et al. 2020, *Genome Biology* 21:81). Do not edit them.
Our changes go in the exporter and the launch scripts, never here.

## Where it came from

    nibi:/scratch/maxwl/epi_imputation/repo/experiments/005_2026-07-29_avocado_cross_dataset

Copied 2026-08-30. That tree is not a git repository, so there is no upstream commit to name and
the sha256 below is the whole record.

| file | sha256 | upstream path |
|---|---|---|
| `avocado.py` | `a2539eafcbe814864718c02143c5cf11db8ad11d3564949451b1e3d541b69256` | `scripts/avocado.py` |
| `hpc_train.py` | `11b1f77b523fa2faf4426f3771716e4e5bf6768156386591579b0459b0e9cecf` | `scripts/hpc_train.py` |
| `hpc_predict.py` | `97ffdf5f1fdd05c676b1a83ca2f13dc1786f8e7c720a96906647e103d73199fc` | `scripts/hpc_predict.py` |
| `hpc_prepare_grid.py` | `e22820a63a0d3636357601c62076bbde26d956929dd66bfeebce59a2e0a4aeaf` | `scripts/hpc_prepare_grid.py` |
| `hpc_fit_affine.py` | `2dec4a98d5380573bc9db425ec46dac053e55bb0a097886b61c5dd2e38502ef0` | `scripts/hpc_fit_affine.py` |
| `UPSTREAM_README.md` | `fdad2850284f59d345bbc9081950f2db87fe11afb36c31d1f18e2dfce34efd96` | `README.md` |

Re-check at any time with `shasum -a 256 rivals/avocado/*`.

## Why verbatim, and not adapted

The Avocado row on our board is a claim about **Avocado**, so the less of it we wrote, the better
the claim. Copying the training and prediction code unchanged confines every difference between
our run and his to one place — the data we hand it — and that place is auditable on its own. An
adapted loader would have put a units or column-order mistake somewhere no diff could show it.

## What it is, and the one property that shapes the whole plan

The architecture follows the paper: 32 cell-type factors, 256 assay factors, genomic factors at 25
(×25 bp), 40 (×250 bp) and 45 (×5 kbp), a 2048→2048 ReLU predictor, trained on `arcsinh(signal)`.

**The genomic factors are per-position free parameters.** A position the model never saw has no
representation at all, so Avocado cannot extrapolate to loci it did not fit. That is not a
limitation we are working around — it is the reason `plan/BENCHMARK_DESIGN.md` §4 blanks Avocado's
`genome-wide` cell, and the reason Rule 2 (the scope names where transferable parameters were fit)
has to be stated at all. `hpc_train.py`'s two modes are the paper's own answer to it:

- `--mode shared` — fit everything: cell factors, assay factors, the network, and this
  chromosome's genomic factors. Once per regime.
- `--mode genome` — load a `shared` checkpoint, freeze it, and fit only this chromosome's genomic
  factors. Once per chromosome we intend to predict.

## What we change, and it is not code

**The joint fit moves to chr19** (§3.2). This needs no edit: `--chrom` is already an argument.
Under `eic.19` the shared run is `--chrom chr19`; under `eic.pilot` it is the pilot regions.

**Three genome-factor fits per regime, not 23** (§12.2, corrected 2026-08-30). §4 blanks the
genome-wide cell and rules a blanked cell is not computed, so Avocado is only ever predicted on
chr20+21+22. Fitting the other 20 chromosomes would build factors for positions nothing is scored
at. The shared run's own chromosome comes free and is not one of the three.

## The data contract

`hpc_train.py` reads, under `--data-root/<dataset>/`:

- `<chrom>.npy` — a `(n_bins, n_tracks)` float matrix at 25 bp, raw signal (it arcsinhs in place)
- `tracks.txt` — one column name per line, in the matrix's column order
- a bridge CSV (`--bridge`) with `filename,cell_id,assay_id`

Our exporter writes exactly that from CANDI_STORE. **Rule 1 binds it:** the matrix carries the
`T_` training and validation columns only. A scored `V_`/`B_` track must never appear in it, at
any stage, and the exporter refuses rather than filters.
