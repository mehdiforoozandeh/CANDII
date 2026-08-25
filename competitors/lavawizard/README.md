# Lavawizard / Guacamole

Rival method for the leaderboard, `plan/RIVALS_PLAN.md` §7.4. This directory holds our port;
today it holds only these reading notes, because the spike half of t53 is finished and the port
half waits on the PI.

**Nothing here is imported by `src/candi`, ever** (decision E). The port emits §4.1 prediction
roots and is scored by `candi.bench.external`.

## Provenance

Upstream: `github.com/ccchang0111/ENCODE_imputation_2019`
Commit: `d638b204f92297aa79e3e56f2939c4f69d47fb2f` — 2020-01-21, the last commit. Repo is dead.
Read on: 2026-08-25.

md5 of every file we read:

| file | md5 |
|---|---|
| `00_data_generation.py` | `7ce99c7748233aa001352ac516560863` |
| `01_data_preproc.py` | `0b9e8e259e3545067ff785cdd823bf8b` |
| `02_guacamole6_pretrain.py` | `ba09e67e3b391a5179d82356c7df74bc` |
| `03_guacamole6_train.py` | `1f808e8a73cc22dd445a34307403fb26` |
| `04_guacamole6_generate.py` | `6445a9aed05d483afa30a1e07cc54723` |
| `06_write_bigwig_gucamole6.py` | `cb79f0df67adf15e82a87175f189b3f1` |
| `Lavawizard_pipeline.sh` | `1eb112ed274034cef676ab2e07d2806a` |
| `Lavawizard_pipeline_using_pretrained_model.sh` | `199b75587e735a03928308acc84c3c48` |
| `env.yml` | `e449cf666024e7083c81a4fc5dd4ac0f` |

There is no `05_`. The repo skips it.

Published placement: joint second in the 2019 ENCODE Imputation Challenge, behind HLYGv1
(Genome Biology 2023, `10.1186/s13059-023-02915-y`). Challenge Table 1 lists them as deep tensor
factorization with arcsinh and an average-signal feature.

Weights and data live outside the repo, on Synapse:

| what | Synapse id |
|---|---|
| 23 pretrained models (one per chromosome) | `syn21519009` |
| challenge training data | `syn18143306` |
| challenge validation data | `syn18143307` |
| `submission_template.bigwig` | `syn18145351` |

The copy the repo ships, `data/submission_template/submission_template (copy).bigwig`, is
0 bytes.

## The pipeline

```
00_data_generation.py    <chrom> <n_cpu>   bigwig -> arcsinh -> 25 bp mean -> 5 pickles
01_data_preproc.py       <chrom>           blind avg/var, looked up from a training cell
02_guacamole6_pretrain.py <chrom> <bs> <ep> <f25> <f250> <f5k>    stage 1, 3-way classification
03_guacamole6_train.py   <chrom> <bs> <ep>                        stage 2, scalar regression
04_guacamole6_generate.py <chrom>                                 predict, sinh back, .npy
06_write_bigwig_gucamole6.py <name>.bigwig                        .npy -> submission bigwig
```

Per-chromosome hyperparameters are hardcoded in `Lavawizard_pipeline.sh`. Batch sizes run
10 000–21 000; pretrain 150–200 epochs; fine-tune 400–800 epochs.

## The model

Five embedding tables, concatenated, then dense blocks. Sizes for chr21:

| embedding | rows | factors |
|---|---|---|
| cell type | 51 | 45 |
| assay | 35 | 65 |
| genome 25 bp | 1 868 400 | 25 |
| genome 250 bp | 186 841 | 30 |
| genome 5 kb | 9 343 | 60 |

**Stage 1 — `Precamole`.** Two 2048-wide blocks (PReLU, batch norm, dropout 0.5), then
`Dense(3)`. Targets are terciles of the arcsinh signal from `pd.qcut(rank, 3)`. The one-hot
tercile of the average track is **added to the logits** before the softmax. Cross-entropy,
Adam 5e-4.

**Stage 2 — `Guacamole`.** The stage-1 checkpoint is loaded, four layers are popped (leaving
`dense_2_dp`), and a third 2048-wide block is stacked on it with dropout 0.7. Two scalar
features join at that point: the per-bin **average** and **variance** of that assay across
cells. Output is `Dense(1)`, and the average is **added** to it. MSE, Adam 1e-3.

That skip connection is the whole idea: the cross-cell average is the base prediction and the
network learns a correction. Verified after the checkpoint surgery — moving the average input
0 → 5 moves the output by 4.984 (5.000 from the skip, −0.016 from the same feature's second
path through `dense_3`).

**Signal space.** `arcsinh(-log10 p)`, then mean over the 25 bases in a bin — arcsinh first,
then bin. `04` inverts with `np.sinh` and clips at 0. Our store holds `-log10 p` untransformed
(`DATA.md`:57), so the port applies arcsinh to the already-binned value. That is a small
difference from their pipeline and must be recorded in the row's caveat.

One model per chromosome, 23 in total, nothing shared between them. chr21 measures 61.8 M
parameters; chr1 computes to ~233 M. Summed: ~3.3 G parameters, ~13 GB of weights.

## Three things the port must not inherit blindly

**The average feature leaks in training but not at test.** `00_data_generation.py:147` averages
every track carrying the assay, including the target track itself. The docstring at
`03_guacamole6_train.py:128` says it should exclude the current cell type, so this is a bug. A
blind track has no track to include, so it gets a clean average — training and test see
different features. PI ruling needed (`RIVALS_PLAN.md` §6.2).

**`dict_3cat` is an object array.** `vals2cat` returns `np.array(pd.qcut(...))`, dtype `object`;
the `astype('int8')` on the next line is never assigned. At chr1 that is 312 × 9.96 M Python
objects — about 25 GB of pointers alone.

**`str.strip` is used as a suffix strip.** `06_write_bigwig` calls `vname.strip(".bigwig")` and
`x.strip("_pred.npy")`. `strip` takes a character set. Harmless for this dataset because every
name ends in a digit.

## Environment reality

Their `env.yml` does not solve today and should not be repaired. It is a `conda env export` of
the author's whole machine: it carries PyTorch, xgboost, ggplot and Qt4 that the pipeline never
imports, pins `tensorflow=1.1.0` and `tensorflow-gpu=1.7.0` and pip `tensorflow-estimator==1.14.0`
at once, and **omits `pyBigWig`**, which `00` and `06` both import.

What does solve, and runs the whole model path unmodified:

```bash
micromamba create -y -n lw_period -c conda-forge -c bioconda \
  python=3.7 "tensorflow=1.15" "keras=2.2.4" "h5py<3" "numpy<1.20" "pandas<1.0" \
  pybigwig multiprocess tqdm matplotlib seaborn "umap-learn=0.3.9"
```

Resolves to python 3.7.12, tensorflow 1.15.0 (`mkl_py37h28c19af_0`), keras 2.2.4, h5py 2.10.0,
numpy 1.19.5, pandas 0.25.3, pyBigWig 0.3.22. 4.4 GB on disk, ~10 minutes to solve.

Only the **CPU** build solves. `tensorflow-gpu=1.15` pulls cudatoolkit 10.0, and Fir's H100s are
compute capability 9.0, which needs CUDA ≥ 11.8. Measured on CPU at chr21 scale, batch 10 000:
**2.686 s per training step**. Their schedule is 6.48 M steps over the 23 chromosomes, so a full
retrain in their own code is roughly 200 CPU-days. That is why the port exists.

## Status

Spike done (t53, first half). Memo and evidence: `cruxvault/results/t53/SPIKE_MEMO.md`, with the
Fir path in `FIR_PATH.txt` beside it. The port is a PI checkpoint and has not started.
