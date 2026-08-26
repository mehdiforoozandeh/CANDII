# The naive baseline suite

`RIVALS_PLAN.md` §5. Five prediction roots, all computed from training-split biosamples only, all
written in the §4.1 on-disk format and scored by `python -m candi.bench.external` — the same entry a
retrained rival goes through, so a baseline row and a rival row are the same kind of object.

## What it emits

| root | `signal_mu` | `signal_sigma` | `mu`, `n` | `peak_score` |
|---|---|---|---|---|
| `avg` | plain mean of `-log10 p` (the EIC `Average` definition) | cross-cell `std`, `ddof=1` | moment-matched NB | fraction of contributors with a peak |
| `avg-arcsinh` | `sinh(mean(arcsinh))` | — | — | — |
| `knn1` | the single most similar training cell | — | that cell, at the Poisson floor | that cell's peak calls |
| `knn5` | mean over the top 5 | `std` over the top 5 | moment-matched NB over the top 5 | fraction over the top 5 |
| `marginal` | one constant per assay | one constant per assay | one constant NB per assay | — |

`avg` is the baseline of record. `avg-arcsinh` is a variant and is never presented as the EIC
baseline: it is a different estimator of a different central tendency, and it down-weights exactly
the tall bins the challenge metrics care most about.

## Environment and how to run it

No environment of its own. numpy, h5py and the repo's own `candi.store` reader — that is all a
leave-one-out average needs. On Fir, `source /project/def-maxwl/mforooz/candi_venv/bin/activate`.

```bash
export PYTHONPATH=$PWD/src:$PWD
python -m competitors.baselines.generate \
    --store configs/regime.eic_val.json --out <preds_dir> --chroms chr21
for m in avg avg-arcsinh knn1 knn5 marginal; do
  python -m candi.bench.external --store configs/regime.eic_val.json \
      --pred <preds_dir>/$m --out <scores_dir>/$m.json
done
python -m competitors.baselines.leaderboard --protocol P1 \
    --scores avg=<scores_dir>/avg.json ... --out leaderboard.json
```

P2 (genome-wide) is the same command with `--chroms` set to one chromosome per SLURM array task.

## §6.2 — where the fairness rule is enforced, by name

`generate.py::Panel.contributors`. It is the only place a contributor set is built. It starts from
`regime.biosamples["train"]` and removes every training cell sharing the input's or the target's
cell-type suffix, so `T_K562` contributes nothing to `V_K562` for any assay. Three other reads are
constrained the same way and each is tested:

* the kNN similarity table (`similarity_table`) reads the regime's **train** chromosomes only, and
  only training cells — `test_the_similarity_table_never_reads_an_eval_chromosome` watches the reads;
* the per-assay marginal (`fit_marginal`) is fitted on training cells over training chromosomes;
* nothing anywhere reads a `V_` or `B_` track except as the truth the scorer compares against, and
  the scorer is `candi.bench`, not this package.

## Two readings of the spec that had to be settled

Both are recorded in the code beside the decision, and both should be confirmed with the PI.

1. **§5.4's kNN ranking cannot be read literally.** It says to rank contributors "by Pearson
   correlation with the *input* cell's track of that assay" — but a declared pair exists precisely
   because the input cell LACKS the assay being imputed, so there is no such track. Implemented as a
   **cell-level** similarity, which is what BestSingle means in this literature: the mean Pearson `r`
   over the assays the input cell and the contributor both carry, on the train chromosomes, in
   `arcsinh(-log10 p)`. Every other constraint in the paragraph is kept.
2. **The per-assay marginal carries a pval arm as well as the NB.** §5.4 names only the NB, but
   §5.5's first sanity anchor compares the plain-mean pval baseline against "the per-assay marginal
   on macro mse" — a pval-arm comparison with nothing to compare against unless this tier has a pval
   arm. It is fitted the same way over the same pool. The marginal is also scaled to the target
   track's depth, because §5.1 already fixed that convention and a count marginal pooled over cells
   at different exposures is a depth mixture.

## The one thing that does not work at the pre-registered value

§5.1 pre-registers `n = 1e6` for the NB's Poisson floor. **`candi.metrics.nb_crps` returns NaN for
every `n` above roughly `2e4`** — its Gini-mean-difference term is `hyp2f1(0.5, 1 - n, 2, w)` — and
one NaN bin makes the whole track's `crps` NaN, after which `macro_mean` drops the key. At the
pre-registered floor the count arm therefore loses `crps`, `crps_oracle_scaled`, `scale_error` and
`beats_marginal`: its entire distributional tier, which is the tier §5.5's second sanity anchor is
about. The point tier (`mse`, `gwcorr`, the P and B blocks) and the loss tier (`nb_nll`) are
unaffected.

CANDI's own decoder exponentiates a bounded head and never reaches that region, so nothing had
exercised it before. `heads.POISSON_N` stays at the pre-registered value and `--poisson-n` exists so
an amended plan needs no code change here; §5.1 says to amend the plan first, which is the PI's call.
Measured against the exact Poisson CRPS at `y = round(µ)`, `n = 1e4` is within 0.01 % at µ = 0.1,
0.05 % at µ = 10, 0.5 % at µ = 100 and 4.9 % at µ = 1000 — and where the floor actually binds (bins
whose contributors agree, overwhelmingly the near-zero ones) the two values are the same number. Both
facts are pinned by tests rather than left in this paragraph.

## §6.4 CRPS companions are per arm (PI ruling, 2026-08-26)

`RIVALS_PLAN.md` §6.4's "CRPS always with `oracle_scaled` + `scale_error`" is **count-arm (NB)
only** — plan amendment PR #27. Pval-arm Gaussian CRPS rows are quoted with **`pit_ks` and
`coverage_95`** beside them instead: `gauss_suite` has no oracle-scale counterpart, and demanding
one would ask `candi.bench` for a key it does not compute — but a CRPS with no calibration key
beside it is sharpness reported alone, which is the reading §6.4 exists to prevent.

And, until a pval-arm Gaussian-CRPS noise floor is measured (its own task as of 2026-08-26), **no
between-method gap on that column may be presented as significant.** `AGENTS.md` §7.2 supplies
count-arm CRPS floors and nothing for the pval arm, so the header carries `noise_floors_absent` and
every pval macro row carries `crps_gap_not_quotable` — on the row, not in a caption someone drops.

Enforced by `leaderboard.py::_COMPANIONS` and `_GAP_NOT_QUOTABLE`; recorded in `PI_RULINGS` with
`scope: "reporting"` so it is re-attached to the header on every rebuild. No re-scoring was needed —
the clarification changes what a table may say, not what `candi.bench` computed.

## The §5.5 sanity anchors, as ruled on the P1 panel

Measured on `regime.eic_val`, chr21, 45 declared tracks, Poisson floor `1e4`.

**Anchor 1 — the plain-mean pval baseline beats the per-assay marginal on macro mse: PASS.**
`avg` 7.128 against `marginal` 9.312.

**Anchor 2 — `beats_marginal` near-universal for the moment-matched NB baseline: FAILED, 38/45 =
0.844 against the ≥ 0.9 reading of "near-universal".** PI ruling, 2026-08-25:

> ACCEPT as a real finding. The anchor failed at the near-universal reading for an understood,
> mechanistic reason: all 7 losing tracks average over k=3 contributors (winners: median 16), and a
> 3-cell cross-cell mean is a noisy predictor. The PI ruled this a genuine property of the LOO
> average rather than an artifact, and declined a post-hoc sparse-threshold change.

No rules changed. §5's sparse rule stays at `n_eligible <= 2`, which flags nothing on this panel
(min k = 3), so excluding flagged tracks leaves the anchor at 0.844 — the rule as written does not
rescue it, and it was not amended to. The losers are all punctate; broad marks go 9/9 and
accessibility 3/3. The ruling lives in `leaderboard.py::PI_RULINGS` so it is re-attached to the
anchor block every time a leaderboard is regenerated, and it never converts `pass: False` into
`pass: True`.

## Tests

`tests/test_baselines.py`. The two §5.5 gates are
`test_fixture_*` (three contributors at three depths, every emitted number compared against
arithmetic done on paper in the test) and `test_L3_generator_is_depth_free` (two real stores
differing only in one contributor sequenced twice as deep, with its counts doubled to match; the
written `mu` and `n` must not move).
