# Lavawizard / Guacamole

Rival method for the leaderboard, `plan/RIVALS_PLAN.md` §7.4. The PI approved the port on
2026-08-25 with one amendment: **their released weights are the primary parity anchor, and reading
them needs no TensorFlow.**

**Nothing here is imported by `src/candi`, ever** (decision E). The dependency runs one way — this
package imports `candi.bench.external` for the §4.1 directory-name rule, so the prediction-track
contract keeps one definition. The port emits §4.1 roots and is scored by `candi.bench.external`.

| module | what it does |
|---|---|
| `model.py` | the architecture — `Precamole` (stage 1) and `Guacamole` (stage 2) |
| `keras_weights.py` | their `.h5` into a torch module, `h5py` only |
| `features.py` | the cross-cell average and variance, and the contributor switch |
| `dataset3.py` | the challenge's own bigwigs — their grid, their binning, their hyperparameters |
| `emit.py` | §4.1 prediction roots |
| `parity_keras.py` | **model** parity: our torch module against Keras on their weights |
| `parity_features.py` | **input** parity: our preprocessing against `00_data_generation.py` |

Tests: `tests/test_lavawizard.py`, 36 checks, seconds, no TensorFlow and no network.

## Parity, part 1 — the model

The port loads a real Keras 2.2.4 checkpoint and reproduces what Keras does with it:

```
loaded 15 layers / 28 tensors / 61,772,797 parameters
abs err  max 2.384e-06  mean 4.630e-07  p99 1.550e-06
pearson r = 1.000000000000
GATE max_abs <= 1e-05: PASS
```

4 096 random inputs against a chr21-architecture stage-2 model. The loaded weight count equals
Keras's own `count_params()` exactly, so every tensor moved. The residual is float32 accumulation
order between MKL and torch, not a mapping error. **The observed tolerance is 2.4e-6; the gate is
set at 1e-5**, four times the headroom, per the PI's instruction to document the tolerance on one
chromosome before rolling it out.

**What this artifact is, precisely.** It is a genuine Keras 2.2.4 checkpoint written by *their*
code — but configured by *us*: the smoke run passed `n_assays=35`, taken from the raw metadata.
Their own filter drops assays appearing only in the training split before building the embedding
(`02_guacamole6_pretrain.py:49-65`), and on this data that leaves **23**, so a released chr21 file
should carry 61,772,017 weights rather than the 61,772,797 measured here — a difference of exactly
12 assays x 65 factors. This changes nothing about the loader, which reads every shape from the
file and hardcodes none; it is why `parity_keras.py` derives `n_celltypes` and `n_assays` from the
h5 rather than from a constant. It does mean the parity run validates the **mapping**, not the
**dimensions** of their released files. Both counts are recorded in `tests/test_lavawizard.py`, so
the first real checkpoint either confirms the 23 or tells us something we did not know.

Four mappings are wrong-by-default and all four are covered:

| Keras | torch | why it bites |
|---|---|---|
| `kernel:0` `(in, out)` | `Linear.weight` `(out, in)` | the transpose is shape-legal on the square `dense_2` |
| BatchNorm `epsilon=1e-3` | torch default `1e-5` | their running statistics assume theirs |
| Keras `momentum=0.99` | torch `momentum=0.01` | the two name complementary quantities |
| PReLU `alpha` `(2048,)` | `num_parameters=2048` | torch defaults to a single shared alpha |

Block order is **Dense → PReLU → BatchNorm → Dropout** — activation before normalisation. Unusual,
and it is what they trained.

## Parity, part 2 — the inputs

A perfect model fed a subtly different average is still a different method, and the average is
added straight to the output, so an error in it passes through undamped. `parity_features.py` runs
our preprocessing and **their `00_data_generation.py`, vendored verbatim**, over the same landed
challenge bigwigs. chr21, `training_data`, five marks spanning 4 to 34 contributors — all PASS:

| quantity | worst absolute disagreement | reading |
|---|---|---|
| single track, per bin | **0.000e+00**, all five marks | `bin_arcsinh` is bit-identical to `get_binvals` |
| average | 4.8e-07 … 9.5e-07 | float32 against float64 accumulation |
| variance | 3.1e-06 … 2.37e-05 | **their** float32 cancellation in `E[x^2] - E[x]^2` |

The variance disagreement tracks contributor count exactly as cancellation predicts — 9.2e-06 at
k=4, 1.06e-05 at k=22, 2.37e-05 at k=34.

The single-track row is the bug detector: the ceil grid, the zero-pad, or `arcsinh` on the wrong
side of the mean would each move it off zero. It is exactly zero.

The variance row is not our error. `E[x^2] - E[x]^2` loses precision by cancellation in float32,
and loses more of it the larger the signal and the more contributors are summed. We accumulate in
float64 and are the more accurate of the two. **The gate is absolute and set at 1e-4** — absolute
because the variance enters the model as a raw scalar, so absolute error is what propagates, and a
relative gate is meaningless in the near-zero-variance bins where one mark shows 1.7e-3 relative on
9e-6 absolute.

## The grid is not our grid

Upstream rounds the bin count **up**; our store floors it, always (`STORE.md`).

| | chr21 bins |
|---|---|
| upstream, `-(-len // 25)` | 1 868 400 |
| our store, `floor(len / 25)` | 1 868 399 |

Their released checkpoints have a `genome_25bp_embedding` of exactly 1 868 400 rows, which is how
we know the ceil is what they trained. `dataset3.upstream_n_bins` is the grid for anything on the
Dataset-3 side; the store's is the grid for ours. A track written on one and scored on the other is
silently misaligned after the first partial bin.

## The contributor switch (PI ruling, 2026-08-25)

`features.contributors(..., mode=)` decides who feeds the cross-cell average:

- **`"loo"`, the default, used for every number we report.** Training biosamples carrying the
  assay, minus every biosample sharing the target's cell-type suffix — so T_X drops out when
  predicting V_X. This matches `RIVALS_PLAN.md` §5's exclusion rule and their own docstring's
  stated intent.
- **`"upstream"`, for parity runs only.** Everyone carrying the assay, target included —
  reproducing the leak described below, so the parity check can feed their weights the features
  they actually saw.

Which mode produced a root is written into `manifest.json` as `contributor_mode` and lands in the
score file's provenance. A row whose mode is `upstream` is a parity artifact and is never a
leaderboard number.

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
different features. **Ruled (PI, 2026-08-25): the port excludes, and reproduces the leak only
behind `mode="upstream"`.** See "The contributor switch" above.

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

Spike done; memo and evidence in `cruxvault/results/t53/SPIKE_MEMO.md`, Fir path in
`FIR_PATH.txt` beside it.

Port: architecture, weight-loader, feature pipeline, Dataset-3 reader, §4.1 emitter and both parity
harnesses are in. Model parity passes at 1e-5; input parity passes at 1e-4 on five marks.

**Their released weights are not obtainable.** `syn21519009` returns
`403 You lack READ access` to a token that reads every other challenge entity — the project, the
training and validation data, the submission template, and all of `submissions_round2`. The entity
carries **zero access requirements**, so there is no data-use agreement to accept and no
self-service route: the ACL simply does not grant it. The challenge's own public `models` folder
(`syn21458457`) holds four tarballs, none of them theirs. Getting these needs the uploader or the
organisers to share; that is a message to a third party and is the PI's to send, not ours.

Consequence: the PI's amended anchor (their weights → our port → their tracks) is blocked at the
first arrow. **The original §7.4 anchor is not** — retrain on the challenge's own training data and
compare against their submitted tracks — and both of its inputs are now in hand.

| input | state |
|---|---|
| challenge `training_data` / `validation_data` / `blind_truth` | landed, `/project/def-maxwl/mforooz/DATA_EIC_SYNAPSE/` |
| their submitted Guacamole tracks (`syn21976480`, 51 files, 49.3 GB) | pulling to scratch — readable now, no t54 dependency |
| their pretrained weights (`syn21519009`) | **403, needs a human to request access** |

Not started, and now the only thing between here and a number: the training loop (their two-stage,
per-chromosome schedule; `dataset3.schedule` has the epochs and batch sizes) and the anchor
retrain.

## Running it

```bash
# tests — laptop or cluster, no TensorFlow, no network
PYTHONPATH=$PWD/src pytest tests/test_lavawizard.py -q

# parity, on Fir. Half A needs the TF 1.15 env; half B must not have it.
cd ~/scratch/t53_lavawizard
MAMBA_ROOT_PREFIX=$PWD/mamba ./bin/micromamba run -n lw_period \
    python port/lavawizard/parity_keras.py reference \
    --h5 <their_model>.h5 --out ref.npz --n 4096
module load StdEnv/2023 python/3.11 && source torch_env/bin/activate
python port/lavawizard/parity_keras.py compare --h5 <their_model>.h5 --ref ref.npz --max-abs 1e-5

# input parity against their own preprocessing, on the landed challenge data
MAMBA_ROOT_PREFIX=$PWD/mamba ./bin/micromamba run -n lw_period \
    python port/lavawizard/parity_features.py \
    --data /project/def-maxwl/mforooz/DATA_EIC_SYNAPSE/training_data \
    --meta repo/data/Encode_meta.tsv --chrom chr21 --mark M17 --max-abs 1e-4
```

The TF 1.15 environment is a debugging fallback and the reference half of the parity harness. It
is used for nothing else, and it is not deleted.
