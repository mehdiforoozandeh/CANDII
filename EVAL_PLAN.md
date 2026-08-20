# CANDII evaluation suite — build plan and handoff

**Status:** approved by the PI (Mehdi) on 2026-08-19 after a full grilling session. Every decision
below is **settled**. Do not re-open them and do not stop to ask. Where this plan is silent, choose
the option most consistent with the decisions here, write down what you chose, and keep going.

**Crux tasks:** `t17` (parent) with children `t18`–`t22` in `cruxvault/tasks/`.
**Branch:** `implementation/t17-eval-suite` — one branch, staged commits, **a commit only when that
stage's verifiables are met**. PR at the end, carrying the verification evidence.
**Merge gate (non-experiment lane):** `pytest tests/ -q` green **and** `python tools/golden.py`
bit-exact, at every commit until the cutover commit — which is the one commit that is *allowed* to
move numbers, and only because it ships the equivalence report that explains every move.

---

## 1. What we are building, in one paragraph

One package, `src/candi/bench/`, with **one CLI** and **five blocks**, that **replaces `eval.py`**.
It scores a checkpoint on **every 25 bp bin of the eval chromosomes** — no subsampling, with one
documented exception — and reports the nine ENCODE Imputation Challenge measures, the four
post-hoc measures the challenge's own retrospective recommends instead, our distributional
measures, a peak/binary family, and a six-instrument **covariate-sensitivity block**. Every number
is verified three ways before it may be quoted: against the organizers' own code, against
hand-computable arrays, and against a stub model whose conditioning we wrote ourselves.

---

## 2. What this rests on

**The EIC scoring code is public and small.** `ENCODE-DCC/imputation_challenge`,
`score_metrics.py` — nine measures in ~150 lines, on the **−log10 p-value track at 25 bp**,
**untransformed** (`normalize_dict` is a no-op stub that returns its input). `score.py` supplies
annotations, `rank.py` the bootstrap rank aggregation.

**Prior work in the archive repo (`~/Desktop/research/libbrechteam@sfu/CANDI`), which we inherit
but do not re-derive:**

| | fact |
|---|---|
| `eic_reproduction/002_.../refs/13059_2023_2915_MOESM3_ESM.csv` | the published per-team, per-track, per-bootstrap score table — 12,740 rows |
| archive `h71` | **refuted**: exact reproduction of MOESM3 is impossible from public data. Best of 15 candidates 3.514e-02 against a 1e-4 bar. `gwcorr` and `mse1obs` trade off along a frontier of slope −0.40 |
| archive `h77` | **supported**: a self-consistent rescoring of all 25 teams holds **16/25 exact published ranks**, max move 2 places, top two unchanged. Stated resolution limit **~0.005 correlation units**; **5 of 24** adjacent pairs invert on ≥3 of 10 chromosome subsets |
| archive W0 control | the vendored scorer reproduces MOESM2 from unnormalized truth at **2.376e-05** — the scorer is exonerated, the gap is entirely in the qnorm transform |
| archive `_utils.py::METRICS` | our own metric family already exists there: `c_index_{nbinom,gauss,laplace}`, `coverage_95ci`, `confidence_quantile`, `foreground_vs_background`, `peak_overlap`, `correspondence_curve`, each with `gene`/`prom`/`1obs` variants |

**Consequence for what "verified" can mean.** Exact reproduction of the published table is already
refuted and must not be re-attempted. What *is* reachable, and what this suite must deliver, is:
our E-block equals the organizers' code bit-for-bit on identical input, and CANDI is placed against
h77's rescored field with its resolution limit quoted every time.

**What CANDII has today.** `src/candi/metrics.py` (146 lines): `nb_crps`, `nb_quantile`, `r2`,
`pearson`, `spearman`, `calibration_pit_curve`, `ece`, `_cos_dist`, `_steering_index`.
`src/candi/eval.py` (1,532 lines): M1/M2/M3/S14. Nothing from `_utils.py::METRICS` was ported.

**Data available.** `signal_BW_res25` / `pval.h5` is the −log10 p-value track, stored in the
**original space** (fixed point ×100, max codec error 0.005). `counts_dsf{1,2,4,8}` are raw integer
counts. `peaks.h5` carries MACS2 peak calls. `cruxvault/results/t4/hg38-blacklist.v2.bed` is the
pinned ENCODE Exclusion list.

---

## 3. Settled decisions

| id | decision |
|---|---|
| **D1** | **Two arms, reported separately.** `pval` scores the Gaussian signal head against the pval track; `count` scores the NB mean against raw counts. Both are always written when available. `--heads count,signal` is required for the `pval` arm; the `count` arm works on every checkpoint. |
| **D2** | **Whole eval chromosomes, no subsampling.** Every 25 bp bin. `--eval-budget` does not exist in the new CLI. |
| **D3** | **One documented exception: the C-index.** It is pairwise; chr21 is 1,868,399 bins, so full-track exact concordance is 1.7e12 pairs per track. It keeps the **distributional** definition — `P(Ŷ_i > Ŷ_j)` under the two predicted distributions — over a seeded pair sample, and **always ships with its Monte-Carlo standard error beside it**, in the doc and in the JSON. |
| **D4** | **The per-track score is the primitive** and is always written. The **macro mean over tracks** is the headline. The EIC bootstrap rank aggregation runs **only** when a competitor score table is passed. |
| **D5** | **Annotations match the code, not the paper.** GENCODE **v29** genes and FANTOM5 `F5.hg38.enhancers`, because those produced the published numbers. Pinned, checksummed, with a `PROVENANCE` file, exactly as `t4` handled the blacklist. |
| **D6** | **Blacklist applies to peaks only**, never to the signal — that is what the challenge did, and `cruxvault/wiki/encode-imputation-challenge.md` records the two being routinely conflated. |
| **D7** | **`msevar` variance pool: the store's training biosamples, per store.** For the **EIC** store the pool is built in **pval space** and chosen to match the original 267 EIC training experiments as closely as the store allows; the biosample list and its count are written into the output so the difference from 267 is visible, never assumed away. For **MERGED** the pool is that store's own training biosamples. |
| **D8** | **All six C-block instruments ship.** C1 use, C2 share, C3 direction, C4 specificity, C5 invariance, C6 bio-conservation guard. |
| **D9** | **C1 runs both nulls side by side.** Marginal substitution answers "does the decoder read this row"; HRT conditional resampling answers "does it respond to a change that could actually happen". `ablation_within_batch` is retained as the structural tripwire that **must read exactly `0.0`**. |
| **D10** | **C2 is Shapley effects, not Sobol.** Sobol assumes independent inputs and ours are not — `AGENTS.md` §329 records `H(run_type \| assay_id, read_length) = 0.000 bits` on 26 `T_` records. Four covariates means 16 subset evaluations; the usual exponential objection does not apply. |
| **D11** | **C4's aspects are level, shape, dispersion, tail.** Per window, from the predicted distribution: `level = log(mean µ̂)`; `shape = µ̂ profile after dividing by its own window mean`; `dispersion = mean(1/n̂)`; `tail = predicted 99th percentile / mean`. |
| **D12** | **C5 replaces M3's cosine ratio** with kBET, iLISI and batch ASW over the DSF levels. The arbitrary 0.3 threshold does not survive. Recorded M3 numbers become incomparable, and the equivalence report says so. |
| **D13** | **C5 is never quoted without C6.** A collapsed encoder is perfectly invariant; `encoder_eff_rank_pooled > 1.0` is far too weak a guard. |
| **D14** | **AUROC is excluded, deliberately.** Peak positives are rare; `cruxvault/wiki/imputation-evaluation-measures.md` records why AUROC flatters every model on rare positives. AUPRC only. |
| **D15** | **Hard cutover.** `eval.py` is deleted, gated on a published key-by-key equivalence report. `AGENTS.md` §7 stays frozen and untouched. |
| **D16** | **Reproduce the organizers' quirks exactly, including the wrong ones.** Where `score_metrics.py` does something odd, the E-block does the same odd thing and the oddity is documented (§5.1). A "fix" that makes our numbers incomparable to the published table is a bug in this suite. |

---

## 4. Blocks, and the keys they write

Output JSON:

```
{ provenance, tracks, per_track: { "<T_cell|imp_cell|assay>": { count: {...}, pval: {...} } },
  macro: { count: {...}, pval: {...} }, C: {...}, baselines: {...}, ranking: null|{...} }
```

### 4.1 E-block — the nine EIC measures

Both arms. `y` is the target, `ŷ` the point prediction (NB mean `µ̂` on the `count` arm, Gaussian
mean on the `pval` arm).

| key | definition |
|---|---|
| `mse` | `mean((y − ŷ)²)` |
| `gwcorr` | Pearson over the whole track |
| `gwspear` | Spearman over the whole track |
| `mseprom` | SSE/n over GENCODE v29 promoters, `prom_loc = 80` bins (2 kb), strand-aware |
| `msegene` | SSE/n over GENCODE v29 gene bodies |
| `mseenh` | SSE/n over FANTOM5 permissive enhancers |
| `msevar` | `((y − ŷ)² · var) / var.sum()`, `var` per D7 |
| `mse1obs` | `mse` over `y >= sorted(y)[-int(N*0.01)]` |
| `mse1imp` | `mse` over `ŷ >= sorted(ŷ)[-int(N*0.01)]` |

### 4.2 P-block — the four post-hoc measures

Binarisation is `Yᵇ = Yᶜ ≥ 2` for predictions, MACS2 peak membership for experimental data.

| key | definition |
|---|---|
| `acc_by_obs_strength` | accuracy of binarised prediction vs binarised truth, per log bin of width 0.1 from 10⁻¹ to 10^2.5 on the **experimental** signal |
| `acc_by_imp_strength` | the same, binned on the **imputed** signal |
| `pr_by_specificity` | precision and recall per group of loci sharing the same cell-type-specificity score (the column sum of the binarised cell-type × locus matrix for that assay) |
| `prom_corr_h3k4me3` | mean Pearson of H3K4me3 over ±2 kb windows at GENCODE starts, averaged over genes |
| `peak_shape_corr_dnase` | mean Pearson of DNase signal restricted to MACS2 peak calls |

Every partitioned measure averages **over partitions**, so each partition weighs equally — not each
locus. That is the whole point of the block.

### 4.3 D-block — distributional

| key | definition |
|---|---|
| `nb_crps`, `crps_oracle_scaled`, `scale_error` | as today; the split is mandatory and the three are quoted together or not at all |
| `gauss_crps` | closed form `σ[z(2Φ(z) − 1) + 2φ(z) − 1/√π]`, `z = (y − µ)/σ` |
| `calib_grid`, `calib_fbar`, `ece` | non-randomised PIT (Czado–Gneiting–Held), 21-point grid |
| `c_index`, `c_index_se`, `c_index_n_pairs` | D3. The SE is not optional |
| `coverage_95` | empirical coverage of the central 95% predictive interval |
| `marg_crps`, `beats_marginal` | the CRPS-optimal marginal fitted to the target itself — the weakest possible bar |

### 4.4 B-block — binary / peak

`auprc` (average precision), `peak_overlap` at `p = 0.01`, `correspondence_curve`. No AUROC (D14).

### 4.5 C-block — covariate sensitivity

The four covariates are `[log2 depth, assay_id, read_length, run_type]`; cell id when the model
carries it.

| instrument | question | output |
|---|---|---|
| **C1 use** | does the decoder respond to this covariate at all? | per covariate: effect size and p-value under **both** nulls (marginal substitution, HRT conditional resampling), plus the `within_batch` identity tripwire that must read exactly `0.0` |
| **C2 share** | what fraction of output variation does each covariate own? | Shapley effects over 16 subsets, normalised, comparable across covariates |
| **C3 direction** | does the prediction move the right way, by the right amount? | S14's ground-truth depth counterfactual (`frac_min_at_true`, `frac_beats_told1`, always with its 0.25 floor and 0.73 ceiling), plus **monotonicity** of the dose-response curve |
| **C4 specificity** | does each covariate move what it should? | a 4 × 4 covariate × aspect matrix (D11), scored by the top-vs-runner-up gap, MIG-style |
| **C5 invariance** | is the latent invariant to depth? | kBET rejection rate, iLISI, batch ASW over DSF levels |
| **C6 guard** | is it invariant because it is good, or because it is empty? | bio-conservation score; **C5 is never written without it** (D13) |

---

## 5. Faithfulness to the organizers' code

### 5.1 Quirks we reproduce on purpose (D16)

These are real behaviours of `score_metrics.py`. Each gets a test asserting we match it, and a
comment in our source naming it as deliberate.

1. **Overlapping annotations double-count.** `mseprom`/`msegene`/`mseenh` accumulate `sse` and `n`
   across every annotation line independently, so a position covered by two genes enters twice.
2. **A promoter that runs off the start of the array is silently dropped.** For a `+`-strand gene
   with `start//25 < 80`, the slice `y[start-80:start]` has a negative left index, which Python
   resolves from the end of the array and yields an **empty** slice — contributing 0 to both `sse`
   and `n`. It is not a wraparound and it is not an error; it is a silent drop.
3. **`end` is computed as `int(end)//25 + 1`**, one bin past the true end.
4. **`msevar` returns `0.0`** when neither `var` nor `y_all` is supplied — a sentinel, not a score.
5. **`normalize_dict` is a no-op.** The published numbers are on **untransformed** −log10 p-values.
   No arcsinh anywhere in the E-block.
6. **`mse1obs` uses `>=` against the `int(N*0.01)`-th largest value**, so ties admit more than 1%.

### 5.2 The vendored reference

`tests/fixtures/encode_score_metrics_vendored.py` — `score_metrics.py` copied verbatim, with only
the `from logger import log` line stubbed. Frozen: it is a fixture, never refactored, never
"improved". Its SHA is recorded in `PROVENANCE`.

---

## 6. Verification — three layers

Nothing in this suite may be quoted until its layer-1 and layer-2 tests pass, and no C-block number
may be quoted until layer 3 passes.

### Layer 1 — reference match (E-block only)

Our E-block equals the vendored code to **1e-12 relative** on: random arrays, adversarial arrays
(all-zero, all-equal, single non-zero, heavy ties), and at least one real track pair from the store.

### Layer 2 — analytic arrays

Hand-built inputs whose answer is known on paper.

| metric | fixture | required answer |
|---|---|---|
| `mse` | `ŷ = y + c` | exactly `c²` |
| `gwcorr` | `ŷ = a·y + b`, `a > 0` | exactly `1.0` |
| `gwspear` | `ŷ` any strictly increasing map of `y` | exactly `1.0` |
| `msevar` | `var = ones` | exactly equals `mse` |
| `mse1obs` | `N = 1000`, known top-10 | SSE over those 10 positions / 10 |
| `mseprom` | one gene entry duplicated | identical to the single-entry value |
| `mseprom` | one gene with `start//25 < 80` | contributes nothing (quirk 2) |
| `nb_crps` | vs exact discrete sum and Monte Carlo | already covered by `tests/test_metrics_primitives.py`; keep |
| `gauss_crps` | `σ → 0` | converges to `\|y − µ\|` |
| `ece` | `y` drawn **from** the predicted NB | `→ 0` at `O(1/√n)`; asserted at two `n` an order apart |
| `c_index` | perfectly ordered / reversed / all-tied | `1.0` / `0.0` / `0.5`, within the reported SE |
| `coverage_95` | `y` drawn from the predicted NB | `→ 0.95` |
| `auprc` | perfect ranker / random ranker | `1.0` / the base rate |
| `acc_by_*_strength` | prediction equal to truth | `1.0` in every occupied bin |

### Layer 3 — stub model with known conditioning

A fake model **we write**, exposing the same interface the harness consumes, whose `µ` depends on
**depth alone**. Required answers:

- **C1**: `p ≪ 0.05` for depth under both nulls; p-values uniform for `assay_id`, `read_length`,
  `run_type`. `within_batch` reads exactly `0.0` for all four.
- **C2**: Shapley share ≈ `(1, 0, 0, 0)`.
- **C3**: monotone response, and the argmin lands at the true depth.
- **C4**: the specificity matrix has mass only in the depth row.
- **C5**: with latents drawn i.i.d. across DSF levels, the kBET rejection rate ≈ α; iLISI → 4 (the
  number of DSF levels); batch ASW → its perfectly-mixed value.
- **C6**: a deliberately **collapsed** stub encoder must **fail** C6 while passing C5. This is the
  test that proves the guard guards.

A second stub whose `µ` ignores every covariate must return C1 non-significant everywhere and C2
share ≈ `(0.25, 0.25, 0.25, 0.25)` or undefined — the "conditional model that ignored its
condition" failure the literature names, made into a test.

### Layer 4 — placement (evidence for the PR, not a gate)

Score CANDI through h77's path and report its position against the rescored 25-team field, with the
0.005 correlation-unit resolution limit and the 5 unseparable adjacent pairs quoted. This is
reported in the PR; it does not gate the merge, because it depends on Fir artifacts.

---

## 7. Module skeleton

```
src/candi/bench/
  __init__.py
  primitives.py      # pure array -> scalar. no model, no h5, no I/O
  eic.py             # the nine, faithful to score_metrics.py including 5.1
  partitions.py      # P-block
  distributional.py  # D-block
  binary.py          # B-block
  covariate.py       # C-block, C1-C6
  annotations.py     # pinned beds, blacklist, msevar variance pools
  harness.py         # whole-chromosome streaming prediction, both arms
  ranking.py         # EIC bootstrap rank aggregation; inert without competitors
  cli.py             # python -m candi.bench

tests/
  fixtures/encode_score_metrics_vendored.py
  test_bench_reference.py    # layer 1
  test_bench_analytic.py     # layer 2
  test_bench_stub_model.py   # layer 3
```

`metrics.py` stays where it is and keeps its current contents; `primitives.py` imports from it
rather than duplicating `nb_crps`. Duplicating that function is how the two copies drift.

---

## 8. Tasks, in dependency order

### t18 — pin the EIC annotation assets `[data-acquisition]` ← **start here**

GENCODE v29 genes bed, FANTOM5 `F5.hg38.enhancers`, both checksummed with a `PROVENANCE` file next
to them, mirroring `cruxvault/results/t4/`. Build the `msevar` variance pools per D7 and record the
biosample list that entered each. Output: the pinned files plus a short report naming sources,
sizes, checksums, and the EIC pool's size against 267.

### t19 — primitives and the three verification layers `[implementation]` ← t18

`primitives.py`, `eic.py`, `partitions.py`, `distributional.py`, `binary.py`, the vendored fixture,
and all of layers 1–2. Commit only when every layer-1 and layer-2 test is green.

### t20 — whole-chromosome harness and CLI `[implementation]` ← t19

`harness.py`, `annotations.py`, `ranking.py`, `cli.py`. Streams full eval chromosomes for both arms
without materialising the whole panel in memory. Commit only when the CLI scores a real checkpoint
end to end and the per-track table is written.

### t21 — the C-block `[implementation]` ← t19

`covariate.py` and layer 3, including both stubs and the collapsed-encoder test. Commit only when
every layer-3 assertion in §6 holds.

### t22 — cutover `[implementation]` ← t20, t21

Run both harnesses on one checkpoint. Publish the key-by-key equivalence report: every key that
moved, by how much, and why. Then delete `eval.py`, rewrite `EVAL.md` against the new suite, and
update the `CLAUDE.md` router line. This is the one commit permitted to move numbers.

---

## 9. Branch and crux discipline

One branch, `implementation/t17-eval-suite`, pushed to `origin` at creation. Staged commits, each
gated on that stage's verifiables — **not** on the clock and not on "it looks right". `pytest
tests/ -q` and `python tools/golden.py` must both pass at every commit before t22.

The PR carries the verification evidence: layer-1 diffs against the vendored code, the layer-2
table with actual values beside required ones, the layer-3 stub results, and the layer-4 placement.
A PR without those numbers in it is not this PR.

---

## 10. Explicitly out of scope

- Re-attempting exact reproduction of MOESM3. Archive `h71` refuted it; do not reopen.
- Re-deriving h77's rescoring from scratch. We reuse its result and quote its resolution limit.
- The `exp/` merge gate itself. This suite is what `t1` will be built on; it is not `t1`.
- Scoring the Bernoulli peak head as a primary result. B-block scores it; it remains auxiliary
  supervision, and two arms are comparable only if their `--heads` sets match.
- Any change to `AGENTS.md` §7. Frozen, forever.
