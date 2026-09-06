# The naive baseline suite

`RIVALS_PLAN.md` §5. Five prediction roots, all computed from training-split biosamples only, all
written in the §4.1 on-disk format and scored by `python -m candi.bench.external` — the same entry a
retrained rival goes through, so a baseline row and a rival row are the same kind of object.

## What it emits

| root | `signal_mu` | `signal_sigma` | `mu`, `n` | `peak_score` |
|---|---|---|---|---|
| `avg` | plain mean of `-log10 p` (the EIC `Average` definition) | cross-cell `std`, `ddof=1` | moment-matched NB | fraction of contributors with a peak |
| `avg-arcsinh` | `sinh(mean(arcsinh))` over the SAME contributors as `avg` | — | — | — |
| `knn1` | the single most similar training cell | — | that cell, at the Poisson floor | that cell's peak calls |
| `knn5` | mean over the top 5 | `std` over the top 5 | moment-matched NB over the top 5 | fraction over the top 5 |
| `marginal` | one constant per assay | one constant per assay | one constant NB per assay | — |

`avg` is the baseline of record. `avg-arcsinh` is a variant and is never presented as the EIC
baseline: it is a different estimator of a different central tendency, and it down-weights exactly
the tall bins the challenge metrics care most about.

### Correction, 2026-09-01 — `avg-arcsinh` was averaging the wrong contributors

The selection line in `generate.py` read `pick = contribs if m in ("avg", "marginal") else
top_k(sim, pair[0], contribs, 5)`, so `avg-arcsinh` fell through to the `else` and averaged the
**top five contributors by kNN similarity** rather than all of them. Two consequences, both
measured on a seven-contributor fixture before the fix. The root moved when only `train_chroms`
moved, because `similarity_table` reads `train_chroms` — so the method was regime-DEPENDENT while
§12.2 printed its one number in both regime rows. And a pass that asked for `avg,avg-arcsinh` alone
built no similarity table at all, so `top_k` ranked alphabetically and produced a **third** array
again for the same declared method. It now takes `contribs`, exactly like `avg`: t49 and §5.2
define this root as the leave-one-out **mean over every eligible contributor**, taken in arcsinh
space, and §12.2's licence for the collapse is precisely that it has no fitted position parameter.

**No stamped number ever came out of the old path.** These roots have not been generated since it
landed, and every pre-t77 board row is void anyway (`plan/BENCHMARK_DESIGN.md` §3.3) — no board is
quoting an `avg-arcsinh` number from either path. The fixture that let it ship green is gone too:
`top_k(…, 5)` over three contributors returns all three, so selection and averaging were the same
operation there. `tests/test_baselines.py` now runs the identity tests on **seven** contributors
whose similarity ranking is built to reverse between the two training slices, which is what makes
the two paths give different numbers.

## How many times each of them runs — D1, settled 2026-09-01

`plan/BENCHMARK_DESIGN.md` §12.2 first ruled that all five run **once**, not once per regime,
"because their fit is regime-independent". Read against the code that is true of two of them:

| | collapses? | why |
|---|---|---|
| `avg`, `avg-arcsinh` | **yes** — one fit, one root, printed in both regime rows | every written bin is a function of the contributors' values **at the predicted position**, and the contributor set is `biosamples.train` minus the target's cell type. No training locus enters. |
| `knn1`, `knn5` | **no** — once per regime | `similarity_table` correlates over `panel.train_chroms`; a different training slice is a different ranking. |
| `marginal` | **no** — once per regime | `fit_marginal` pools over `panel.train_chroms`. |

So the five baselines are **8 method-regime units**, not 5, and the programme is 18 rather than 15.

**The collapse is licensed by an assertion, not by that paragraph.** §12.2 always claimed one
existed; until 2026-09-01 none did.

```bash
python -m competitors.baselines.generate --store <regime A> --out <root> \
    --methods avg,avg-arcsinh --assert-regime-independent <regime B> [--assert-only]
```

re-predicts the same methods under regime B for the first chromosome of the pass **into a temporary
directory**, compares **every array** with `np.array_equal`, and writes into each manifest

```json
"regime_independent": {"asserted_against": "<B file name>", "chrom": "<c>", "identical": true}
```

A difference exits **5** and writes no manifest at all — a root that failed the check must not be
quotable. The flag is refused for `marginal`, `knn1` and `knn5` (exit **2**): asserting regime
independence of a method that fits on `train_chroms` would be asserting something false. It
compares **arrays, not manifests** — `_MANIFEST_IDENTITY` includes `regime`, so two manifests from
two regime files never match and never could. `slurm/t49_baselines_score.sh` refuses to score a
collapsed root against the other board unless that stamp is present.

**`--assert-only` makes the stamp the ONLY mutation of the prediction root.** It generates nothing:
the regime-B side goes to a temporary directory that is removed with it, the roots under `--out`
must already exist, and every `.npz` under them is byte-identical afterwards. That is how a root
built by the P2 array gets its stamp, and it is also the fix for a defect on record —
`t49_baselines_p1.sh` used to run the assertion by re-generating `avg,avg-arcsinh` into the same
roots, which overwrote the checked chromosome's arrays and then stamped `identical: true` on
numbers nobody had scored. Stamp **after** the last generation pass: a later generation into a
stamped root drops the stamp on the manifest merge, which is the safe direction but is silent.

**`regime.eic_pilot.json` is accepted for `knn1`, `knn5` and `marginal` (2026-09-01).** It used to be
refused with exit 3: this module read `train_chroms` raw, so under the pilot regime those three
would have been fitted over 18 whole chromosomes instead of the 25,588,197 bp the regime declares —
a Rule 2 break, because every other method under that regime sees the Pilot Regions only. The module
honours the regime's `regions` block now, so D1's second pilot unit for those three is a fit on the
loci the regime names, and the exit-3 guard is gone from all three `slurm/t49_baselines_*.sh`.

## The training scope — a `regions` regime (D32)

`Panel.contained_bins(chrom)` is the whole of it: the bins of a chromosome lying **wholly inside**
one region of the regime's BED, counted by `RegionSet.contained_starts` at a window of one bin — the
same D32 containment rule the training window plan is cut by, asked at the resolution a pooled fit
reads. `None` without a BED, so a regime that declares none keeps the untouched path bit for bit.

Exactly two things read it, and they are the only two here that pool over training loci at all:

| | reads the training scope | what changes under `eic_pilot` |
|---|---|---|
| `similarity_table` (`knn1`, `knn5`) | yes | the Pearson ranking is correlated over the Pilot Regions of `train_chroms`, not over whole chromosomes |
| `fit_marginal` (`marginal`) | yes | the per-assay NB and Gaussian are moment-matched over the same bins |
| `avg`, `avg-arcsinh` | **no** | nothing — no training locus enters them, which is what makes them the collapsed two |

It is the **train** split's scope and nothing else's: `Regime.windows` applies `regions` to the train
split only, and every baseline still writes whole eval chromosomes, exactly as a rival does. A BED
with no region on `train_chroms` leaves `fit_marginal` and `similarity_table` an empty pool, and
both refuse outright (a `ValueError` naming the chromosomes and the BED) rather than write a NaN
constant or an alphabetical ranking.

## No `eval_pairs` — the self-paired shape

A regime that declares no pairing has not asked for a cross-cell imputation (D31), and
`bench.harness.StoreSource` answers it by self-pairing every cell of `biosamples.eval`; a
self-pair's targets are every assay that cell holds. `Panel.self_pairs` takes the same path, so

```bash
python -m competitors.baselines.generate --store <derived sigma regime> --out <root> \
    --methods avg-arcsinh
```

writes `T_x__T_x__<assay>` tracks instead of nothing.

This is not a corner case. `tools/sigma_training_regime.py` writes that shape and **can write no
other** — `candi.store.regime` refuses a `[c, c]` literal outright — and it is what the
training-residual σ pass (`competitors.sigma_pass`, `plan/BENCHMARK_DESIGN.md` §7) takes its
residual against. Until 2026-09-01 `Panel.pairs` read `eval_pairs` alone, so that pass wrote an
empty root and the `avg-arcsinh` tier could get no σ table at all.

The assay is still held out. `contributors` drops every training cell sharing the **target's**
cell-type suffix, which on `T_x -> T_x` is `T_x` itself, so a self-paired track is never averaged
from its own answer.

## Environment and how to run it

No environment of its own. numpy, h5py and the repo's own `candi.store` reader — that is all a
leave-one-out average needs, and no job of this module opens CUDA or needs a GPU. It does **load**
torch: importing `candi.store.reader` runs `candi/__init__.py`, which imports `candi.encoder`. That
costs the import and the RSS, and it is why the P2 array caps its concurrency at `%12` — more than
about twelve concurrent torch imports off the shared `/project` venv fail with partial-module
`ImportError`s. On
Fir, `source /project/def-maxwl/mforooz/EpiDenoise/candi_venv/bin/activate` and
`export PYTHONPATH=$KIT/src:$KIT` — that venv's editable install points at a different tree.

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

The programme's own launchers are `slurm/t49_baselines_p1.sh` (one job, one regime-panel unit; the
identity assertion runs as its own one-chromosome pass afterwards, and `ASSERT_ONLY=1` runs just
that pass, which is how a root built by the array gets its stamp),
`slurm/t49_baselines_p2.sh` (one array task per chromosome, `V_` only) and
`slurm/t49_baselines_score.sh` (one array task per method). They take `REGIME` and `PANEL`, derive
the panel regime with `tools/declare_eval_pairs.py split`, and write the pinned roots
`/scratch/$USER/t81_pred/<Method>/<regime>/V_` and
`/project/def-maxwl/$USER/t81_pred_B/<Method>/<regime>/B_` — the second only under `B_ONCE=1` and
only into a root with no manifest in it (exit 4). Each generates at §5.1's pre-registered Poisson
floor `1e6`, which t56 made scoreable; the old two-floor `spec`/`n1e4` pass is gone.

The scorer takes `SCOPE`, which is §4's two aggregations: `heldout` (the default) scores
chr20+21+22, and `genomewide` scores all 23 **and** passes `--held-out-chroms chr20,chr21,chr22`,
which is what makes one pass emit the held-out numbers and a parallel `genome_wide` block. Without
that flag a 23-chromosome pass produces no `genome_wide` block at all — and these five are among
the methods whose genome-wide cell §4 does not blank.

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

## P2's count-arm CRPS is sampled, not exact (PI ruling, 2026-08-26)

The exact NB CRPS costs ~117 h per method genome-wide. t56 built a fair-sampled estimator, reached
through `--crps-approx K --crps-seed S` on `python -m candi.bench.external`. The PI ruled it **GO
for P2 at k = 100 on all four count-arm methods** (`avg`, `knn1`, `knn5`, `marginal`; `avg-arcsinh`
is pval-only and has no count arm to sample).

The criterion was whether sampling ever reorders the methods. On the **macro** ordering it never
flips, at any k or seed tried — that is the PI's reading, recorded as such. The stricter per-track
reading flipped once, on a gap of **0.000087**, roughly 1000x below the noise floor and therefore
unreadable in either direction. Measured agreement on the chr21 validation panel: macro
**|Δ| = 6.3e-5**.

A sampled number must never be readable as an exact one, so:

- `bench.external` stamps `provenance.crps_estimator = "fair_sampled"`, `crps_k` and `crps_seed`;
- `assemble()` reads that stamp and copies all three onto the **count macro row** and into
  `reporting.crps_estimator`, per method. A run with no stamp reports `closed_form`;
- `test_a_sampled_crps_can_never_be_read_as_an_exact_one` pins it.

Separately, and true of **every** score json written at the `n1e4` floor: `n_star_log2` and
`crps_oracle_scaled_and_n` are unreliable, because the exact CRPS goes NaN inside `oracle_scale`'s
n-grid. `_UNRELIABLE_AT_CLOSED_FORM` stamps both `_unreliable` on any closed-form row, so neither
can be quoted off a P1 table by accident. That is the nb_crps ceiling above, met from the other
side; the fix is another session's task.

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

D1's collapse has both halves pinned, on one store with three chromosomes and two regimes over it
that differ in nothing but the training slice: `test_the_collapsed_methods_are_identical_under_two_
training_slices` (the assertion passes for `avg` and `avg-arcsinh`, and the manifest records it) and
`test_the_assertion_fails_when_the_training_slice_is_made_to_matter` (it fails on `marginal`, which
is a genuinely regime-dependent method rather than a doctored one — an identity check that cannot
fail licenses nothing).
