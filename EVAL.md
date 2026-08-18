# EVAL.md — the metric contract

Owns: what M1/M2/M3/S14 measure, the keys they write, and the rules for quoting a number.
Input contract → DATA.md.  Invariants, tasks, gates → AGENTS.md.  What CANDI is → README.md.

Source of truth is `eval.py` (the harness) and `metrics.py` (the primitives). Where this file and
the code disagree the code is right and this file is the bug.

## Scoring a checkpoint is one command

```bash
python -m candi.eval --h5 <panel.h5> --ckpt <run.ckpt> --out <scores.json> --arch-from <run.json>
```

Scale comes from the h5 — `num_assays`, `context_bins`, `resolution`, the assay order and the eval
chromosomes are read, never passed. The architecture comes from `--arch-from`.

## Always pass `--arch-from`, never retype the architecture

`--arch-from` reads `config.arch` out of the run's own JSON and calls
`model.py::build_model_from_arch`, so the model that scores the checkpoint is the model that wrote
it. Every geometry, norm, FiLM and head flag changes the `state_dict`. A mismatch is a strict-load
failure at best — and at worst a model that loads and is quietly not the one you trained.

`config.arch` carries all 31 arguments the model constructor takes, `meta_embed_layernorm` and
`depth_center` included, so under `--arch-from` none of the architecture flags on the eval CLI are
read at all. Two things still need passing by hand:

- `--reference` (with `--reference-path` / `--reference-pseudocount`) — the reference table is
  loaded outside the model, so it is not in `config.arch`. It **must** match the training run: a
  residual-arm checkpoint re-scored without it is missing half its mean model. `eval.py::main`
  refuses a reference whose `depth_center` differs from the eval's by more than 1e-6, because the
  two offsets would otherwise compose on different scales.
- `--depth-center` — in `config.arch`, but deliberately still wins if passed explicitly. It is a
  property of the data, not of the architecture, and re-scoring against a rebuilt reference is a
  legitimate reason to override it.

## The output is five blocks

`eval.py::evaluate` returns:

```
{n_units, assays, reference_only_baseline, M1, M2, M3, S14}
```

A standalone `--out` JSON is exactly that. A **run** JSON is the same blocks **splatted at the top
level**, not nested — `train.py::train_and_eval` ends on
`dict(config=..., train_losses=..., train_terms=..., **ev)`, and adds `eval_curve` and
`best_checkpoint` on the way. So it is `run["M1"]`, never `run["eval"]["M1"]`.

## `den` in eval is `obs` in the loss — both mean unmasked

The loss terms are named `obs` and `imp`. The M1 pools are named `den` and `imp`.
`den_map` is literally `prep["observed_map"]`, so `den` and `obs` are the same side under two
spellings. **Neither pair means biologically observed versus imputed** — throughout, `obs`/`den` is
*unmasked* and `imp` is *masked*.

## M1 measures accuracy and calibration on held-out positions

`eval.py::eval_M1`. Two pools — `imp` (masked) and `den` (unmasked) — each scored with
`spearman_raw`, `pearson_log1p`, `crps`, `ece`, `r2`, `n_points`, plus the PIT curve as
`calib_grid` / `calib_fbar`.

Alongside the pools it emits `imp_per_assay` / `den_per_assay` and `imp_per_target` /
`den_per_target`, and the macro roll-ups `imp_macro_spearman_raw`, `imp_macro_crps`,
`imp_macro_crps_oracle_scaled`, `imp_macro_scale_error`, `imp_macro_marg_crps`,
`imp_beats_marginal_n`. `encoder_eff_rank_perpos` is the per-position latent's effective rank.

Per-target keys are the flat string `"T_x|V_x|assay"` so they survive the JSON round-trip that
`compare_arms.py` reads back. Split on `|` to recover the tuple.

### `spearman_raw` and `pearson_log1p` are different spaces and never quote as a pair

Spearman is on raw counts; Pearson is on `log1p`. The key names carry their space for exactly this
reason. Reporting "correlation 0.75" without saying which is how two arms get compared on two
different statistics.

### M1 CRPS splits into capability and a fixable scale error

`crps = crps_oracle_scaled + scale_error`, where `scale_error` is what a single per-assay
multiplicative rescale would remove. Quoting raw `crps` alone conflates "the model cannot predict
this assay" with "the model has the wrong constant in front of it" — and the second is free to fix.
Report both terms or neither.

`marg_crps` is the CRPS-optimal marginal NB fitted to the target itself, so `beats_marginal` is the
weakest possible bar: it asks whether the model beat a single number per assay.

### ECE is pooled only, and it is PIT, not interval coverage

`ece` appears in the `imp` / `den` pools and **not** in the per-assay or per-target records, because
`_suite_light` does not compute it. It is the mean `|F̄(u) − u|` over a 21-point grid of the
non-randomized PIT curve (Czado–Gneiting–Held).

Interval coverage was rejected, not overlooked: a perfectly calibrated NB at `mu = 1` scores 0.25
under interval-coverage ECE, because discrete quantile intervals cannot hit a nominal level. Most
epigenomic positions are low-count, so that artifact would have dominated the metric.

### `n_points` depends on `--eval-budget` and `--seed`

Both pools subsample to `--eval-budget` (default 200,000) with `np.random.default_rng(seed)`. CRPS,
ECE and both correlations are computed on that subsample. Two runs scored at different budgets or
different seeds are not scoring the same positions — check `n_points` before comparing anything.

### A NaN Spearman is usually the split, not the model

`eval.py::_spearman_diag` prints the cause rather than emitting a bare NaN. An all-zero held-out
target on the eval chromosome has no ranking to recover. That is a property of the panel, and the
fix is a wider panel or more eval chromosomes, not a model change.

## M2 asks whether the decoder uses each covariate at all

`eval.py::eval_M2`. Two families:

- `run_type` and `depth` — counterfactual prompt flips (`_flip_covariate`, `_depth_sweep`). Change
  what the model is *told* about the target and measure what moves.
- `ablation` and `ablation_within_batch` — a covariate-agnostic probe over all four rows,
  `{0: depth, 1: assay_id, 2: read_length, 3: run_type}`.

### `ablation_within_batch` is the structural null and must read exactly 0

It is the arm where the substituted value is the value already there, so every difference it can
report is arithmetically zero. Each of the four rows carries `mean_d_crps`, `mean_abs_d_mu`,
`mean_abs_d_eta`, `max_abs_d_eta` — **all four are `0.0`** on a correct run — plus the boolean
`uses_covariate`, which must be `False`. The same keys under `M2.ablation` are the real measurement:
on a recorded a1 run, `ablation.depth.mean_d_crps` is `1.438` with `uses_covariate: True` against
`ablation_within_batch.depth.mean_d_crps` of `0.0`.

Check this first. A non-zero `within_batch` is a bug in the probe, not a finding, and it is the
cheapest possible falsifier of the whole M2 block.

Note `d_crps_clustered.sign_test_p` is `NaN` here rather than a number, because the null produces
all ties and there is no sign test to run. That NaN is correct and expected — it is not a failure,
and it is not comparable to a NaN anywhere else.

### Use the `*_clustered` keys — position-level CIs are ~24× too narrow

Positions inside one target are not independent draws. `eval.py::_cluster_bootstrap_ci` resamples
*targets*, which is the real replication unit. The un-clustered `direction`, `overall`, `single` and
`paired` keys ship only under `--include-deprecated`, each carrying its own verdict string in
`eval.py::DEPRECATED_VERDICTS`, so a reader who finds one in an old JSON cannot mistake it for
evidence. Read that dict before quoting any deprecated key — it names why each one was retired.

## M3 asks whether the latent is invariant to input depth

`eval.py::eval_M3`. Encodes the same regions at several DSF levels and compares cosine distances:
`within` (same region, different depth) against `between` (different regions). `ratio = within /
between`; `invariance_ok` is `ratio <= 0.3 and encoder_eff_rank_pooled > 1.0`.

The `between` pool **excludes same-region pairs**. Admitting them deflates `between` and inflates
the ratio, which makes a model look invariant when it is not. Any M3 number produced before that
exclusion is not comparable to one produced after it.

## S14 is the only depth counterfactual with real ground truth

`eval.py::_dsf_counterfactual`. M2's depth sweep measures whether the prediction moves; S14 scores
the moved prediction against `counts_dsf{k}` — the actual downsampled track. Keys:
`frac_min_at_true`, `frac_beats_told1`, `n_targets`, `per_target`.

### 0.25 is not chance and 0.73 is the ceiling

Both calibrations are printed at eval time because neither is visible in the number:

- **0.25 is the deterministic value of always putting the argmin at told-depth = 1.** It is not a
  chance baseline, and a model scoring 0.30 has not "beaten chance".
- **A perfect model caps near 0.73**, because the foreground is the top `--fg-frac` (default 2%) of
  the level-*k* realization being scored, not of the truth.

An S14 number quoted without both bounds is uninterpretable. The M2-era `frac_min_at_true` is a
different, retired statistic — it scored every told depth against the fixed dsf1 target, so any
mu-decreasing model satisfied it (0.7588 vs 0.7597 between arms). S14 supersedes it.

## `reference_only_baseline` is the bar, not a result

`eval.py::reference_only_baseline` forecasts the average epigenome with an oracle dispersion and no
model at all. It is what a trained model must clear to have learned any deviation from the mean.
Keys: `available`, `V_crps`, `V_n_targets`, `B_crps`, `B_n_targets`.

### `V_` and `B_` CRPS are different scales and never compare

`B_` runs roughly 4× `V_` because its tracks are denser. An absolute CRPS delta is meaningful within
one prefix and meaningless across the two. Compare each against its own bar.

## Only the count head is scored

The evaluation is Negative Binomial end to end — every CRPS, the PIT curve, the marginal bar and the
oracle rescale are NB. When `--heads` enables the Gaussian signal head or the Bernoulli peak head,
those are **auxiliary training losses only**; nothing here scores them.

The consequence: a run trained with extra heads descends a different total loss and is then scored
on the NB head alone, so it is **not** a control for a run trained without them. Two arms are
comparable only if their `--heads` sets match.

## Rules for quoting any number

1. Say which pool — `imp` (masked) or `den` (unmasked). They are not interchangeable and `den` is
   always the easier number.
2. Say the space — `spearman_raw` and `pearson_log1p` are not one metric.
3. Quote `crps` with `crps_oracle_scaled` and `scale_error`, or quote none of the three.
4. Use `*_clustered` intervals. A position-level CI is ~24× too narrow and will manufacture
   significance.
5. Check `n_points`, `--eval-budget` and `--seed` match before comparing two runs.
6. Give S14 both calibrations (0.25 floor, ~0.73 ceiling) or do not give S14.
7. Never compare a `V_` delta to a `B_` delta.
8. Confirm `--heads` matches across the arms being compared.
9. Confirm `ablation_within_batch` reads 0 before believing anything else in M2.
