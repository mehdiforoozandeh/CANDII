# Lavawizard / Guacamole

Rival method for the leaderboard, `plan/RIVALS_PLAN.md` §7.4. The PI approved the port on
2026-08-25 with one amendment: **their released weights are the primary parity anchor, and reading
them needs no TensorFlow.**

**Nothing here is imported by `src/candi`, ever** (decision E). The dependency runs one way: the
port emits §4.1 roots and is scored by `candi.bench.external`.

| module | what it does |
|---|---|
| `model.py` | the architecture — `Precamole` (stage 1) and `Guacamole` (stage 2) |
| `keras_weights.py` | their `.h5` into a torch module, `h5py` only |
| `features.py` | the cross-cell average and variance, and the contributor switch |
| `dataset3.py` | the challenge's own bigwigs — their grid, their binning, their hyperparameters |
| `store_eic.py` | **our** store — the same cache from `CANDI_STORE`, and the P1/P2 predictions |
| `emit.py` | §4.1 prediction roots |
| `fit_sigma.py` | the §6.1 sigma-table, fitted on V-pair residuals |
| `parity_keras.py` | **model** parity: our torch module against Keras on their weights |
| `parity_features.py` | **input** parity: our preprocessing against `00_data_generation.py` |

`store_eic.py` is the only module that imports `candi` — it has to, because the store is `candi`'s.
Every other module runs on Fir with `candi` nowhere on the path, which is not a preference: an
earlier version imported `candi.bench.external` for the §4.1 naming rule and died the moment the
package was rsynced on its own. A test pins the split in both directions.

Tests: `tests/test_lavawizard.py`, 99 checks, seconds, no TensorFlow and no network.

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

The §7.4 anchor is **run and it holds** — see "The anchor result" below. The deliverable stage
(retrain on our EIC with `--contributor-mode loo`, P1/P2, the §6.1 sigma table,
`candi.bench.external`) is what remains.

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
| their submitted Guacamole tracks (`syn21976480`, 51 files, 49.3 GB) | **landed**, `~/scratch/t53_lavawizard/submitted_tracks/Guacamole` |
| their pretrained weights (`syn21519009`) | **403, with the PI for outreach — do not wait on it** |

## The anchor retrain

Done. `train.sbatch`, 23 array tasks, one per chromosome, MIG `1g.10gb`, `--contributor-mode
upstream`, seed 0. Preprocessing is done for all 23 (`cache/<chrom>/`, built once from the
challenge bigwigs; chr21 took 15 min, 287 tracks).

**Measured, not estimated** (chr21, `1g.10gb`, batch 10 000):

| | value |
|---|---|
| step time, fp32 | 141.2 ms |
| step time, **TF32** | **63.0 ms** |
| peak GPU | **1.58 GiB** allocated / 1.86 reserved, of the slice's 10 |
| host RAM, largest chromosome (chr1) | 15.0 GiB — jobs ask 64 GB |
| loss, stage 2 over 200 steps | 0.342 → 0.110 |

The profile says why: `fwd 42.2 → 16.1`, `fwd+bwd 124.8 → 45.7`, Adam a flat 11.7 ms either way,
and the in-RAM data gather 0.1 ms. It is the three 2048-wide matmuls, so TF32 is where the 2.2x
comes from. Sparse embeddings plus `SparseAdam` would give a further 57.4 → 48.4 ms; not taken,
because it changes the optimizer's semantics for a 16 % gain and the anchor is the run where
fidelity matters most.

**Budget correction.** At 63 ms/step the genome-wide schedule — 6.48 M steps — is about
**117 GPU-hours**, not the 20–40 the spike memo estimated. That estimate assumed a few milliseconds
per step for a 2048-wide MLP and was wrong by roughly 3x even after TF32. Wall-clock is ~10 h with
the 23 tasks in parallel. Memory says the MIG slice is still the right ask (1.58 of 10 GiB), so no
exception is needed under `AGENTS.md`:130 — the constraint here is compute, not memory, and a
larger slice would only buy wall-clock.

**Actual: 136.5 GPU-hours**, array `56784137`, all 23 tasks `COMPLETED`, longest 8 h 10 m. That is
17 % over the 117 h projection because the big chromosomes carry both a wider factor table and a
larger batch, so the per-step cost measured on chr21 is a floor, not a mean.

## The anchor result

23 chromosomes, 51 blind tracks each, **1173 track-chromosomes**. Array `56932672`, one CPU task
per chromosome. Raw records in `cruxvault/results/t53/anchor/anchor_<chrom>.json`, roll-up in
`anchor_report.json`.

| | ours | theirs | ours / theirs |
|---|---|---|---|
| median per-track MSE ratio | | | **1.023** |
| macro MSE (mean over tracks) | 5390.18 | 199.50 | 27.02 |
| macro MSE, **dropping `C38M02`** | 204.44 | 201.77 | **1.013** |
| macro Pearson vs truth | 0.4461 | 0.4701 | — |
| tracks where we are worse | | | 65.2 % |

Per-track ratio quantiles: p05 0.891, p25 0.984, **p50 1.023**, p75 1.083, p95 1.346. Sixteen of
1173 tracks exceed ratio 2, and **eleven of those sixteen are the same track**, `C38M02`; fourteen
of the sixteen are in cell `C38`.

Scoring cost: the GPU partition was 53 k jobs deep, so the anchor ran on **CPU**, 8 cores and
16 GB per chromosome, longest task 5 h 35 m. `MaxRSS` came in at 16.3 GB against a 16 GB ask —
it fit, but ask for 24 GB next time rather than trusting the margin.

**Verdict: APPROACHES** — the port, retrained on the challenge's own training data, lands within
about 2 % of their submitted tracks on median per-track MSE and about 4 % behind on correlation
(0.4461 vs 0.4701). Worst mark is `M02` at median 1.180 over 69 tracks, best is `M18` at 1.000.
Read the median, not the ratio of means: the mean is a ratio of means and one track decides it.

### Defect: unbounded output through the sinh inversion

`C38M02` is not a bad model — it is **five bins**. On `chr17`, bins 900975–900979 carry a
cross-cell average of 9.26 in `arcsinh` space (a pileup artifact: every contributing cell is loud
there). The model adds a correction on top of the additive skip and emits 15.50, and
`sinh(15.50) = 2.687e6`. Truth at that bin is 0.57 and their submission is 212.9. Those five bins
carry 2.02e13 of squared error, which is the whole of `chr17`'s 6.08e6 MSE over 3.33 M bins.

The cause is structural, not a porting bug: the head is `Dense(1)(x) + average`, nothing bounds it,
and `sinh` turns a `+6.2` overshoot in `arcsinh` space into a 500x multiplicative one. Upstream has
no clip either — grep their tree — so this is a property of the method that their own submission
happened not to trigger as hard. It matters downstream because §6.1 fits `sigma` from squared
error, and five bins of 1e13 would swallow the fit.

Any guard on this is a **deviation from the port**. As of the anchor it is **not** taken: every
number above is the faithful port. For the deliverable it is — see the next section.

**One chromosome is not the method.** The same blind track scores 1.016 on `chr13` and 1.677 on
`chr11`: one model is trained per chromosome, and they are not equally good. Any single-chromosome
ratio here is one of 23 draws, and the verdict is the median over 1173 of them. Do not read a row.

## The deviation on record: the cap

PI ruling, 2026-08-26. `signal_mu` is capped at the largest value that mark reaches in **that
chromosome's training data**. Data-derived, never tuned, never read off the target.

| | |
|---|---|
| where | `emit.write_track(..., clip_max=)`, applied after the `sinh` so it bounds `-log10 p` |
| the bound | `preprocess`/`store_eic` record `mark_max` per mark per chromosome at cache-build time |
| the switch | **off** for parity and anchor reproduction; **on** for the our-EIC deliverable |
| the record | `manifest.json` carries `clip: true/false` and `clip_rule`, so a score file self-describes |

`clip` is a **required** argument of `write_manifest`, not a defaulted one, and `--clip` needs a
cache that actually measured `mark_max` — a cache built before the cap existed makes the run fail
rather than invent a bound. Both are tested. The anchor's caches predate it, which is the right
answer: those runs must stay uncapped.

## The deliverable: retrained on our EIC

Same code, other corpus. `store_eic.py` writes the identical cache layout from `CANDI_STORE`, so
`train.py` and `model.py` cannot tell which one they are looking at.

- **`--contributor-mode loo`** for everything reported. `upstream` exists only to reproduce their
  leak for the anchor.
- **§6.2 is enforced in `store_eic.train_columns`, and it raises rather than filters.** The
  training pool is `biosamples.train` and nothing else; a train biosample that is also an eval
  target is an error, not a silent drop. At predict time the pair's own input cell is removed from
  the mark's average — taken from the declared `eval_pairs`, never by splitting a biosample name
  (`STORE.md` D16: an id is opaque). A regime declaring `[T_A, V_B]` is honoured as written.
- **`T_X` and `V_X` share one cell embedding**, fitted from the T_ side alone. Without that a V_
  target has no cell representation and there is nothing to impute from.
- **No binning.** The store's grid is already `floor(chr_len / 25)` and `pval` returns decoded
  `-log10 p` on it, so the only transform is `arcsinh` on the binned value. Upstream applies
  `arcsinh` per *base* and then averages into the bin; we cannot, and this is the difference.
- **The grid is one bin shorter** than the anchor's on every chromosome — the store floors where
  upstream ceils.
- **Five marks have a single training track**, so removing the target empties their pool. §5's
  answer is to skip and list, and the sampler does: `H4K12ac`, `H2AK9ac`, `H3T11ph`, `H3K9me2`,
  `H3F3A` on chr21, listed in every `train_<chrom>.json` and flagged in the manifest with the three
  two-track marks. **None of the 45 declared eval tracks is on one**, so no reported number depends
  on this. Skipping beats zeroing here: the head is `Dense(1)(x) + average`, so a zero average is
  not a neutral input — it would teach the trunk to emit the whole signal as a correction on those
  bins alone. Dataset 3 never hits the case, which is why the anchor never surfaced it.
- σ comes from `fit_sigma.py` (§6.1), fitted on the V panel and reused unchanged on B. Under the
  2026-08-26 ruling in plan §6.4 the resulting Gaussian CRPS is quoted with `pit_ks` and
  `coverage_95`, with **no** oracle split, and no pval-CRPS gap is significant until that arm's
  noise floor is measured.

### Checkpoint selection on the `V_` panel (§5)

`BENCHMARK_DESIGN.md` §5 makes every trainable method pick its checkpoint on `V_`, by the same
rule. For this method a check is: write the current model's `V_` predictions for this chromosome
into a scratch §4.1 root, hand the root to `candi.bench.external.score_external`, and read one
number off it — the same instrument that scores CANDI and the same one that produces the board row.

- **`B_` is never opened.** `store_eic.derive_v_only` filters `eval_pairs` to `V_` targets and the
  scoring source is opened on the derived copy, so both the declared track list and the truth reads
  are `V_` only. Same filter as `slurm/t81_train_candi.sh` applies for CANDI, including the
  absolute-BED rewrite the D32 hash gate needs.
- **The selected weights are written the moment the metric improves**, before the next check runs,
  so a run killed by walltime still yields a selected checkpoint. `predict.sh` predicts from
  `guacamole_<chrom>.best.pt` and says so loudly if it has to fall back.
- **The metric is `pval:mse`, not CANDI's `count:crps`,** and that is forced rather than chosen.
  This method has no count arm (B1b), and the pval arm's distributional keys — CRPS included — are
  *absent* on a point-only track until a σ-table supplies a spread, which §7 forbids fitting on
  `V_`. `pval:mse` is the arm's own point key, present on every scored track, and it is what stage 2
  minimises. `--select-key` moves it; the value is stamped into `train_<chrom>.json` either way.
- **Selection is stage 2 only.** `from_precamole` discards stage 1's head, so there is no stage-1
  checkpoint anyone could select.
- **Cost per check, chr20** (45 `V_` tracks × 2,577,766 bins): ~3.1 min of forward on a `1g.10gb`
  slice, 0.3 min of npz, 2.3 min of metrics, plus the store's truth reads — the part that makes
  CANDI's three-chromosome pass 91 minutes. Default cadence is 50 epochs = 16 checks over stage 2's
  800; `stage2.eval_seconds` sits beside `stage2.seconds` in the run json so the first real run says
  what it actually cost.

### The BED-restricted training scope (D32), and why `eic.pilot` still cannot run

A `regions` regime restricts training to the bins lying **wholly inside** a BED region.
`store_eic.contained_bins` resolves it through `candi.store.regime.RegionSet` — one parser, one
hash gate, one containment rule — and writes the eligible bins into the cache as `train_bins.npy`,
because `train.py` never imports `candi`. The sampler walks those bins and `steps_per_epoch` falls
with the scope, so an epoch still means the training scope once. A locus here is one 25 bp bin, so
the scope is §3.1's containment count of **1,023,489** bins, not its 1,294-window / 993,792-bin
figure, which is CANDI's 768-bin planner. A test pins the 1,023,489 against the shipped BED.

**That capability is built and does not unblock `eic.pilot`,** and the reason is not the BED:

> `train.py` fits one independent Guacamole per chromosome — cell factors, assay factors, dense
> network, genome tables — on that chromosome's own bins, and `predict_chrom` indexes the genome
> tables by that chromosome's bin numbers. So the model that predicts chr20 was fit on chr20.

The cell and assay factors are transferable parameters by §2's own definition, and this scheme fits
them on the eval chromosomes. Two things follow. First, `train_chroms` and `regions` reached no
lavawizard module at all before this change, and those two keys plus `_comment` are the **only**
difference between the two live regimes — so `eic.19` and `eic.pilot` would have been the same run
twice. Second, under `eic.pilot` the pilot regions on chr20/21/22 are exactly the four the regime
**cut**; restricting the fit to them would train on the complement of the declared scope and label
it with the regime's name. So `store_eic` refuses a `regions` regime on a chromosome outside its
`train_chroms`, and `_env.sh` refuses it before the array queues.

Both need one PI ruling: whether Lavawizard gets a transferable stage the regime can reach — a
joint fit on `train_chroms` plus per-chromosome genome factors, which is Avocado's scheme and not
upstream Guacamole's — or whether it collapses to one regime-invariant row.

### The smoke, end to end

Before the 23 GPU jobs, the whole chain ran on chr21 from a **50-step** checkpoint — plumbing
evidence, not a result, and the numbers below must never be quoted as one:

```
chr21 cache from the store          267 tracks x 1,868,399 bins, 0.8 min
train, capped at 50 steps/stage     5 marks skipped for having no leave-one-out pool
predict, --clip --manifest          45/45 declared tracks, ~68 s each, caps 123 to 4150
sigma-table                         27 assays fitted on the chr21 V-pair residuals
candi.bench.external                45 of 45 declared tracks scored
```

The score file carries what the §6.4 ruling requires — `crps` beside `pit_ks` and `coverage_95`,
no oracle split — and its provenance carries `clip: true`, `contributor_mode: loo`,
`pred_inversion: external` and the sigma table's `fitted_on`. That is what the smoke exists to
prove: every switch this method has reaches the score file as data.

## The deliverable result — P1 and P2

Regime `regime.eic_val.json`: 51 training cells, 26 T->V pairs, **45 declared tracks**, 0 missing.
`--contributor-mode loo`, cap on, sigma from the §6.1 table fitted **once** on the P1 panel and
reused unchanged by P2. P1 is every 25 bp bin of chr21; P2 is all 23 chromosomes, 121.2 M bins.

| | mse | gwcorr | crps | pit_ks | coverage_95 | gaussian_nll |
|---|---|---|---|---|---|---|
| **P1** (chr21) | 6.2662 | 0.6297 | 0.6023 | 0.3978 | 0.9852 | 1.7535 |
| **P2** (genome) | 9.2809 | 0.6030 | 0.6407 | 0.3613 | 0.9817 | 1.9575 |
| *P1, same code, 50-step model* | *6.4319* | *0.5501* | *0.6245* | *0.3763* | *0.9848* | *1.7905* |

**P2 is not a generalisation test for this method.** Guacamole fits genomic factors per chromosome
and one model per chromosome, so **every** P2 chromosome was trained on. The P1 -> P2 drop in `mse`
is the other 22 chromosomes having more dynamic range than chr21, not held-out sequence. Plan §2
warns that P2 structurally favours position-parameterised rivals; that warning applies here in full,
and it is why both protocols are always reported.

The sigma table is **in-sample on chr21 and out-of-sample on the other 22**, since it was fitted on
the P1 panel and reused. That is §6.1 working as designed, and it is why the P2 CRPS is the more
honest of the two — but neither is significant until the pval arm's noise floor is measured.

The second row is the plumbing smoke, and it is here as a **calibration on how much of this is the
method**. Training the model properly moved `gwcorr` 0.550 -> 0.630, a real gain — but it moved
`mse` only 6.43 -> 6.27. That is the additive skip: `y = Dense(1)(x) + average`, so the cross-cell
average carries most of the squared error and the network earns its keep on correlation. Read the
correlation columns when asking whether this method works.

**Calibration is poor and the reason is the device, not the method.** `pit_ks` near 0.40 says the
PIT is far from uniform, and `coverage_95` of 0.985 against a nominal 0.95 says the constant sigma
is too wide. That is what B1a's homoscedastic Gaussian does to a heavy-tailed residual. Per the
2026-08-26 ruling (plan §6.4) the CRPS is quoted with both of these and with **no** oracle split,
and **no pval-CRPS gap is significant until that arm's noise floor is measured**.

Placement is t55's job, not this README's. One row for scale, on the same instrument and the same
45 tracks: eDICE's P1 reads `mse=6.1572 gwcorr=0.7075 crps=0.5885 pit_ks=0.3868 coverage_95=0.9851`
— better, and clearly so on correlation.

**Cost, measured.** Training 136.5 GPU-hours over 23 array tasks, longest 8 h 27 m. Prediction:
6 chromosomes on the MIG slice, then the remaining 17 moved to **CPU** when the GPU partition went
3765 jobs deep and my tasks sat on `Priority` — 12-way, and the `%12` import cap became the binding
constraint rather than the queue. Scoring: P1 in minutes, P2 in 7 h 49 m at 0.46 GB peak RSS, so
the 64 GB ask was far more than P2 needed.

## Reading an anchor number

`anchor.py` scores three pairings on the upstream grid in raw `-log10 p`: ours vs `blind_truth`,
**theirs vs `blind_truth`**, and ours vs theirs. Their column is recomputed here rather than quoted,
so both methods meet one binning, one masking and one measure.

Every record carries a caveat naming what it is not: this is **not** the 001 vendored EIC scorer
(§7.5, t54's work), so it is comparable to the "theirs" column in the same table and to nothing
else — never to a published leaderboard row, and never to a Dataset-2 number (005's translation
result is the reason).

A plumbing run with a deliberately under-trained checkpoint (2 steps) gives
`C05M17: ours 0.751, theirs 0.661, r(ours,theirs) 0.857`. That is the expected shape, and it is a
useful check rather than a result: an untrained `Guacamole` returns approximately the cross-cell
average, because the additive skip dominates a random head — so it lands near the Average baseline
and still correlates 0.86 with their submission. Had the two grids been misaligned, that
correlation would be near zero.

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

# the anchor retrain: 23 array tasks, one per chromosome
sbatch train.sbatch
# then place the port against their submission, per chromosome
python -m lavawizard.anchor --checkpoint runs/anchor/guacamole_chr21.pt --cache cache \
    --chrom chr21 --their-tracks submitted_tracks/Guacamole \
    --blind-truth $D/blind_truth --meta repo/data/Encode_meta.tsv \
    --out runs/anchor/anchor_chr21.json

# the deliverable: our EIC store, one array task per eval chromosome
python -m lavawizard.store_eic cache   --regime configs/regime.eic_19.json --chrom chr21 \
    --cache $C/eic_cache
# `store_eic train`, not `lavawizard.train`: §5's selection scores through candi.bench.external,
# which needs the store. `--select-every 0` is the no-selection path the anchor uses.
python -m lavawizard.store_eic train   --regime configs/regime.eic_19.json --chrom chr21 \
    --cache $C/eic_cache --out runs/eic --select-every 50
python -m lavawizard.store_eic predict --regime configs/regime.eic_19.json --chrom chr21 \
    --cache $C/eic_cache --checkpoint runs/eic/guacamole_chr21.best.pt \
    --pred-root runs/eic/pred --clip --manifest
python competitors/lavawizard/fit_sigma.py --regime configs/regime.eic_val.json \
    --pred runs/eic/pred --out runs/eic/sigma.json
python -m candi.bench.external --store configs/regime.eic_val.json --pred runs/eic/pred \
    --sigma-table runs/eic/sigma.json --out runs/eic/P1.json          # P2: add --chroms

# input parity against their own preprocessing, on the landed challenge data
MAMBA_ROOT_PREFIX=$PWD/mamba ./bin/micromamba run -n lw_period \
    python port/lavawizard/parity_features.py \
    --data /project/def-maxwl/mforooz/DATA_EIC_SYNAPSE/training_data \
    --meta repo/data/Encode_meta.tsv --chrom chr21 --mark M17 --max-abs 1e-4
```

The TF 1.15 environment is a debugging fallback and the reference half of the parity harness. It
is used for nothing else, and it is not deleted.
