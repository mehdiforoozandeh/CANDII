# t118 — learn the covariate-conditioned transformation on the counterfactual arms

Record of the design session 2026-09-23 to 2026-09-25 (PI: Mehdi Foroozandeh). Every ruling
below is the PI's. The hypothesis is h15 (`cruxvault/h15_one_transformation_conditioned_on_the_so.md`)
under q1. Branch `exp/t118-counterfactual-f`, draft PR #48.

## 1. The goal

Find out if it is possible at all to model the effect that each recorded covariate has on the
shape and the magnitude of a signal track.

- **Framing (final, 2026-09-25):** a generator `g(C, C')` outputs a transformation `f`; then
  `f(X)` predicts `X'`. `X` is a track, `C` is the covariates it was made with, `C'` is the
  covariates of the track we want.
- The model must learn the **relation between C and C'** and how it changes `X` into `X'`.
  Example: C = depth 15M, C' = depth 30M, so in count space the peaks of `X'` are about two times
  those of `X`. A model that only keeps one map per (C, C') pair does not satisfy the goal.

## 2. The data

- Corpus: the t112 counterfactual arms. 7 tracks: C12M02 DNase, C19M16 H3K27ac, C40M17 H3K27me3,
  C40M18 H3K36me3, C07M20 H3K4me1, C19M22 H3K4me3, C07M29 H3K9me3. Each track has a **base** and
  **arms**. An arm changes one processing knob. 130 products: 7 bases + 123 arm products
  (19 per histone track, 9 for DNase).
- Products: `nibi:/project/def-maxwl/mforooz/t112_cf/products/<pid>/` (counts25.npz,
  pval25.npz, covariates.json, provenance.json) + `MANIFEST.tsv` (md5 599e2ca6…). Scratch copy
  purges 2026-11-20.
- Known defect: the DNase MAPQ 0 and 10 arms are byte-identical to the DNase base (the MAPQ cut
  does not apply with the DNase pipeline's multimapping setting). Task t119 rebuilds them.
- The four control and fragment-extension arms (ratio, ctlid, ctldepth, extsize) change only
  −log10 p; their counts equal the base's.

## 3. Rulings, in the order they were made

### Distances D (2026-09-23, amended 2026-09-24)

| space | CRPS | Spearman |
|---|---|---|
| counts | NB CRPS of the predicted X' against the real X' | yes |
| −log10 p | **log-normal** CRPS (was Gaussian until 2026-09-24) | yes |

- CRPS on three bin subsets: all bins, non-zero bins, top 1% of bins ranked by the **real X'**.
- Spearman on three subsets: all bins, non-zero bins, top 1%.
- Mean over bins per track, then macro over tracks.
- The histone −log10 p tracks have no exact zeros, so "non-zero" equals "all" in p space; tables
  leave it out there.
- A point forecast's CRPS equals its absolute error, so a point-output f and a
  distribution-output f are scored on one scale.
- The CRPS split (capability `crps_oracle_scaled` against `scale_error`) is reported beside
  every CRPS, never used for pass or fail.

### Why log-normal in p space (measured 2026-09-24)

`tools/t118/pval_dist.py`, 7 base tracks, chr1 (a training chromosome), pseudoreplicate halves
as two repeat measurements:

| test | Gaussian expects | log-normal expects | measured |
|---|---|---|---|
| skew of x | about 0 | large, positive | 2.9 to 17.6 |
| skew of log x | — | about 0 | −0.56 to 0.51 |
| slope of log SD(pr2) on log mean(pr2), in 40 levels of pr1 | about 0 | about 1 | 0.81 to 1.32 |
| levels where log-normal fits better | — | — | 40 of 40 on every track |

Median −log10 p is 0.04 to 0.18; the top 1% reaches 2 to 8. A Gaussian fitted at the lowest level
puts 18 to 36% of its mass below 0. Result file: `cruxvault/results/t118/pval_dist.json` (main
checkout, gitignored) and `nibi:/project/def-maxwl/mforooz/t118/pval_dist/`.

### What f outputs (2026-09-24)

f predicts **both parameters of the distribution in every bin**, as CANDI does: NB (n, p) for
counts, log-normal (μ, σ) for −log10 p. A point-output version (g trained with MAE or MSE) is a planned
comparison (2026-09-23).

### Rungs that output one value per bin (2026-09-24)

noSolution, per-pair QuantileMatching and the oracle output one value per bin. Their second
parameter comes from a fixed rule:

- counts: **Poisson** with that mean. The PI accepted that a Poisson is narrower than the true
  (overdispersed) noise.
- −log10 p: **log-normal with median = the prediction and one σ per arm**, fitted by maximum
  likelihood on the training chromosomes. A fixed σ already gives SD ∝ mean.
- Predictions floored at 1e-3 before the log. f needs no floor, because it predicts μ directly.
- Target bins that are exactly 0 are left out of the σ fit (at most 1.2% of bins, DNase).

### Pairs, split, variants (2026-09-23)

- **Training pairs:** base → arm and arm → base, both directions. No arm → arm pairs in training.
  No identity pairs (C' = C) in training.
- **Chromosome split:** train on every chromosome except chr19, chr21, chr22; validate (early
  stopping, run selection) on chr22; score on chr19 + chr21. chrY and chrM dropped everywhere.
  ENCODE hg38 blacklist regions excluded from scoring, kept in training. Blacklist on Nibi:
  `/project/def-maxwl/mforooz/EIC_REPRO/002/scripts/hg38_blacklist_v2.bed`
  (sha256 31c69342…9251737, same as Fir).
- **What is trained is g, not f.** f is only what g outputs for one (C, C'); no f is trained on its
  own (wording fixed 2026-09-25).
- **Two versions of g (PI ruling 2026-09-25):** one g per track (7 g's, each trained on that
  track's pairs only; the assay entry of C is constant inside it) and one g across all 7 tracks
  (all 246 pairs, the assay in C). Both are run and scored under the same checks. **Both versions
  decide pass/fail** (PI ruling 2026-09-25): the claim needs every check to pass for both. The earlier "one per arm" version is dropped: a g trained on one
  arm's pairs sees a single (C, C') per direction and cannot learn anything about C.
- **Covariates:** C and C' are the **full** knob vector on both sides plus the assay: depth,
  run type, read length, dedup, MAPQ, control fraction, ratio k, control identity, control depth,
  extsize k, assay. Encoding: one standardised concatenation — continuous knobs as z-scores (depth
  as log2 reads), binary as 0/1, categorical one-hot, MAPQ as a number. g is **never** given
  C' − C; it must infer the change itself.

### Pass/fail thresholds (2026-09-23; the denominator is now void, see section 5)

Pooled over arms, per mark class (DNase; narrow H3K27ac/H3K4me3/H3K4me1; broad
H3K27me3/H3K36me3/H3K9me3), counts and p separately, 3 seeds of each g:

1. CRPS gap-closed on all bins ≥ 0.5.
2. Gain > 2 × the model's seed wobble (the largest pairwise |Δ| of D over 3 seeds).
3. Top 1% of bins: gap-closed ≥ 0.5 and gain > 2 × seed wobble.
4. Shuffle check: at scoring, C' replaced by the C' of another arm of the same track whose target
   differs; gap-closed against the true X' ≤ 0.1. Gates the claim.
5. Swap check: at scoring, C' = C; the f that g outputs must return X (gap-closed toward X ≥ 0.5). Gates the claim.
   No check voids the run.

The boring explanation to rule out: **the covariates are ignored.** Reported only: Spearman
gains, CRPS on non-zero bins, per-arm results, the CRPS split, the point-output version.

### Final design (2026-09-25)

- **Competitor:** the **scrambled-covariate twin** — the same model with the same training,
  but each pair's C and C' shuffled across pairs **in training**. Equal capacity, equal X → X'
  data, uninformative covariates. A gain over the twin comes from the covariates.
- **Law test:** **arm → arm pairs, never trained, scored at test**, for all arm pairs within each
  track, reported by knob combination. Examples: depth 15M → 3.75M (a depth ratio never trained as
  a pair); depth 7.5M → paired-end at 30M (the depth law and the run-type effect together). A
  model that keeps one map per trained (C, C') pair has no answer for these; a model that learned
  the relation does. For depth there is a closed-form check: the predicted count scale must follow
  the depth ratio.
- **References, never pass/fail:** noSolution (X' = X) and per-pair QuantileMatching.
  QuantileMatching does not read C or C'. It is one monotone per-bin map for each (arm, direction),
  fitted with that arm's own X' on the training chromosomes: sort source and target bins, pair
  by rank, map each tied source value to the mean of its block of target values, interpolate
  linearly between stored values, stay flat above the largest training value. It shows how much
  of a change is plain magnitude.
- **Shape and magnitude:** a per-bin map can fix magnitude only. Shape effects (fragment
  extension, paired-end, peak width) need f to see neighbouring bins.

### Architecture ladder (2026-09-25)

The options are forms of **f**; **g** is a small MLP that reads [C, C'] and outputs f's parameters.
f works on log-scaled input: x = log(1 + counts), or log of −log10 p. Kept by the PI, in order of
expressiveness, all to be built:

- **A — per-bin affine.** log μᵢ = a + b·xᵢ, plus one dispersion (NB n or log-normal σ). g outputs
  3 numbers. Magnitude only (depth shift, dynamic range).
- **B — per-bin monotone curve.** g outputs a monotone spline (~8–16 knots) plus dispersion as a
  function of level. The same function class as QuantileMatching, chosen from C, C' instead of fit
  on X'. Magnitude only.
- **C — kernel, then curve.** f convolves x with a kernel (~33 bins ≈ 800 bp) that g outputs, then
  applies B's curve. Adds global shape (broadening, sharpening); same kernel at every position.
- **D — conditioned CNN.** A few dilated conv layers (~2 kb view); g outputs a per-channel scale
  and shift for each layer (FiLM). Shape that depends on local context.
- **E — g also reads DNA sequence: PARKED** (PI 2026-09-25); recorded as its own hypothesis under
  the capacity question. g would output per-bin parameters of f.

- **Rung checks** (PI 2026-09-25): each rung must beat its own scrambled twin, pass the arm → arm
  law test, and (B, C, D) beat the rung below; the same boring explanation as the main claim,
  signed for all four. Shuffle and swap checks are reported for each rung, gating only the main claim.
- **Which rung the main claim judges** (PI 2026-09-25): the lowest rung within seed wobble of the
  best on chr22, then scored once on chr19 + chr21.

g sees few distinct (C, C') points (38 per histone track, 18 for DNase, 246 across tracks), so g
stays small in A–D and the capacity goes into f's form.

### Data and covariate rulings (2026-09-25)

- **Control identity** is encoded as three categories — matched, other, none — not as the ENCODE
  accession, so it means the same thing across tracks.
- **DNase has no control:** its control entries are 0, plus one "has control" flag.
- **The two DNase MAPQ arms** (byte-identical to the DNase base, a t112 defect) are **excluded from
  training** as well as from scoring, until t119 rebuilds them; they would teach g that MAPQ does
  nothing.
- **The p-only arms train in count space** (ratio, control identity, control depth, extsize): their
  counts equal the base, which correctly teaches that these knobs do not move counts. Verified from
  the t112 MANIFEST counts md5: all 50 of these arm products match their base; the only other
  matches are the 2 DNase MAPQ arms.
- The full covariate vector of each product must be assembled: the manifest records depth, read
  length, run type and fragment length for every product, but dedup, MAPQ, control fraction, ratio,
  control depth and extsize only for the arm that changes them; base values come from the pipeline
  defaults and must be verified.
- The capacity-question notes were rewritten for A–D (A, B, C, D as four rungs; E parked).

## 4. What ran

| step | where | jobs | result |
|---|---|---|---|
| Pseudoreplicates (t117) | `nibi:/project/def-maxwl/mforooz/t112_cf/pseudoreps/` (62 GB; split reads stay on scratch) | 22565383 … 22582946, names `prep_*` | 130/130 products pass all four checks: halves add back to the product exactly, each half holds half the reads, same bins, no NaN. Ratio halves use k × each half's own default ratio (PI). Branch `data-acquisition/t117-pseudoreplicates`, PR #47, task done. |
| Baselines v1 (NB / Gaussian spreads) | `nibi:/project/def-maxwl/mforooz/t118/qm_rungs/` | 22572763 (all failed: missing x-transformers), 22583098/22583384 (test, fixed), 22583393 (test ok), 22583719 (245 pairs), 22588843 (aggregate) | Superseded by v2. Tables: `cruxvault/results/t118/baseline_rungs.md` (main checkout). |
| p-value distribution | `nibi:/project/def-maxwl/mforooz/t118/pval_dist/` | 22591748 | log-normal, see section 3 |
| Baselines v2 + oracle | `nibi:/project/def-maxwl/mforooz/t118/rungs_v2/`, code snapshot `code/Q3` (commit fb2ba6f) | 22630504, 22630505 (tests), 22630900, 22631034 (246 pairs), 22631035 (130 oracle products), 22631036 (aggregate) | All tasks completed. Tables: `cruxvault/results/t118/rungs_v2.md` / `.tsv` (main checkout, gitignored). |

Mechanical fixes on the way: the job venv now uses the repo's pinned recipe (python/3.10.13,
`requirements-fir.txt`, then the x-transformers 2.11.23 wheel from `$KIT/wheels`); Nibi compute
nodes have no `/usr/bin/time`.

### v2 results, by mark class (scored on chr19 + chr21, blacklist removed)

`fraction closed` = (D_noSolution − D_QM) / (D_noSolution − D_oracle).

| class | space | metric | noSolution | QM | oracle | fraction closed |
|---|---|---|---|---|---|---|
| DNase | counts | CRPS all | 2.6274 | 1.1327 | 0.5778 | 0.73 |
| DNase | counts | CRPS top 1% | 107.28 | 29.04 | 4.70 | 0.76 |
| narrow | counts | CRPS all | 0.8120 | 0.6418 | 0.4890 | 0.53 |
| narrow | counts | CRPS top 1% | 9.4486 | 4.1924 | 2.6150 | 0.77 |
| broad | counts | CRPS all | 0.8126 | 0.6854 | 0.5335 | 0.46 |
| broad | counts | CRPS top 1% | 2.9938 | 2.5995 | 2.5264 | 0.84 |
| DNase | p | CRPS all | 1.3466 | 0.6125 | 0.3493 | 0.74 |
| DNase | p | CRPS top 1% | 74.66 | 31.55 | 9.31 | 0.66 |
| narrow | p | CRPS all | 0.3986 | 0.3105 | 0.2719 | 0.70 |
| narrow | p | CRPS top 1% | 13.609 | 9.496 | 4.006 | 0.43 |
| broad | p | CRPS all | 0.2894 | 0.2482 | 0.2758 | 3.03 |
| broad | p | CRPS top 1% | 1.7635 | 1.6191 | 1.7096 | 2.68 |

Spearman (all bins), noSolution / QM / oracle: DNase counts 0.670 / 0.667 / 0.498; narrow counts
0.397 / 0.392 / 0.150; broad counts 0.397 / 0.392 / 0.140; p rows similar (oracle 0.25 to 0.59).

## 5. What we learned

- **The pseudoreplicate oracle is not a floor.** It beats both baselines in 125 of 142 count pairs
  and 117 of 242 p pairs on CRPS (all bins), but on Spearman in only 1 of 142 and 18 of 242.
  Spearman between the two halves is 0.14 to 0.27; between base and arm it is 0.40 to 0.67.
  Reason (PI agreed, 2026-09-25): the halves share **no** reads and have half the depth; a base
  and its arm share **many** reads, so X already carries much of X''s own noise. A read-disjoint
  oracle is a floor only for read-disjoint pairs, and we have none. The oracle must be redefined,
  so every "gap-closed" threshold above has no valid denominator yet.
- The noSolution → QuantileMatching step removes, on CRPS all bins, 57% (counts) and 55% (p) on
  DNase, 21% and 22% on the narrow marks, 16% and 14% on the broad marks. In the top 1% it removes
  73% / 58% (DNase), 56% / 30% (narrow), 13% / 8% (broad).
- Process note: the v2 agent launched the full run after its test pair had failed the oracle
  sanity check, against its brief. The baseline numbers do not depend on the oracle.

## 6. Open

1. ~~The oracle~~ — **dropped** (PI ruling 2026-09-25). The two boring explanations (the scrambled
   twin, and the arm → arm law test) and the two references (noSolution, QuantileMatching) replace
   it. The v2 oracle numbers stay in section 4 as history; nothing is judged against them.
2. **Thresholds** for the final design: the bar against the scrambled twin and the bar for the
   arm → arm law test are not set. The "2 × seed wobble" rule and the shuffle and swap checks are
   carried over; the gap-closed rules are gone with the oracle.
   Under `all`, 4 checks × 2 versions of g: at 80% power each, joint power ≈ 17% if independent.
3. **The architecture** (the form of f, and how g produces it) — not chosen. Drafted options in the vault: a covariate-conditioned affine
   map (h12), a small conditioned convolutional network (h13), an encoder–decoder (h14). The
   shape part needs f to see neighbouring bins.
4. t119 — rebuild the DNase MAPQ arms with multimapping off.
5. PR #47 (pseudoreplicates) is ready for the PI's review and merge.
6. `/scratch/mforooz` is over its soft quota (1746 GiB against 1024 GiB before t117, +90 GB).
