---
title: onion total variance
kind: background
parent: "[[peeling the onion]]"
siblings: ["[[onion eval protocol]]", "[[onion variation and objectives]]", "[[onion caveats]]"]
created: 2026-11-12
status: formal note (plan Amendment 3, section A); it states the identity and maps every metric and layer to a term. It asserts no defect: the audit of the estimators against it is a separate chunk.
---

# onion total variance

This note writes the "peeling" of [[peeling the onion]] as one exact variance identity. It then says which term of the identity each onion layer and each headline metric of [[onion eval protocol#headline-metric-set]] estimates, on which scale, and under which assumptions. The assumptions are numbered (1) to (6) and are defined in [assumptions](#assumptions). The code read for this note is `tools/onion_eval/layers.py`, `error_budget.py`, `metrics.py`, `score.py`, `resolution.py`, `mixed_model.py` and `select_rule.py` at commit 7f457cb.

## identity

Y is the count of one assay in one bin of one track. The random element is a draw of a bin and a track. The conditioning sets form a chain, i.e. each set contains the one before it:

| set | contents |
|---|---|
| Z0 | assay, position |
| Z1 | Z0 + cell type |
| Z2 | Z1 + donor |
| Z3 | Z2 + experiment (lab, antibody lot, protocol) |
| Z4 | Z3 + biological replicate |
| Z5 | Z4 + library (technical replicate) |
| Z6 | Z5 + sequencing run |

The covariates C (depth, read length, run type, pipeline settings) are held fixed along the chain. The law of total variance, applied once per step of the chain, gives the identity:

```
Var(Y) = Var(E[Y|Z0])                                   position and assay
       + sum_{k=1..6} E[ Var( E[Y|Zk] | Z(k-1) ) ]      the six layers: cell type, donor, experiment,
                                                         biological replicate, library, sequencing run
       + E[ Var(Y|Z6) ]                                  read sampling
```

Call the k-th summand `T_k` (k = 1..6), the first summand `T_0`, and the last summand `T_s`. The identity is exact for any law of Y with a finite second moment.

**The sampling term is Poisson.** Given Z6 and C, the reads of a track are independent draws, so the count of a bin is Poisson given its rate. Then `Var(Y|Z6) = E[Y|Z6]` and `T_s = E[Y]` on counts. Any variance in excess of Poisson belongs to a layer `T_k` (most often the library layer, through PCR duplicates) or to the covariate layer. It never belongs to `T_s`.

**The covariate layer.** A change of C changes the rate. The variance that it causes, `Var_C( E[Y|Z6, C] )`, is a separate covariate layer outside the chain. Headline 4 measures the part of it beyond depth.

**Specific variance.** The specific variance of a cell is `T_1 + ... + T_6`, i.e. the variance of the truth around the across-cell mean `E[Y|Z0]` minus the sampling term.

**Pair value.** Let Y and Y' be two measurements of one bin that share Z(k-1), are exchangeable given Z(k-1) (conditionally independent and identically distributed), and have equal covariates. Then

```
0.5 E[(Y - Y')^2] = E[ Var(Y | Z(k-1)) ] = T_k + T_(k+1) + ... + T_6 + T_s.
```

Call `PV(Z(k-1))` the pair value of a set. A layer is the difference of two consecutive pair values: `PV(Z(k-1)) - PV(Zk) = T_k`.

**Dependence on depth (counts).** Under a multiplicative model of the rate, `T_s` grows in proportion to depth D and every `T_k` grows in proportion to D^2. A pair value is therefore defined only at one stated depth (assumption 2).

## graphical-model

An arrow means that the child's effect on the rate of the bin is drawn given its parents. The dotted link marks that donor and experiment are crossed in ENCODE: one experiment can hold several donors (anisogenic replicates), and one donor can appear in several experiments. The nested order of the chain (donor before experiment) is therefore a choice (assumption 3).

```mermaid
graph TD
    position[position] --> celltype[cell type]
    assay[assay] --> celltype
    celltype --> donor[donor]
    celltype --> experiment[experiment and lab]
    donor -.-|crossed| experiment
    donor --> biorep[biological replicate]
    experiment --> biorep
    biorep --> library[library]
    library --> seqrun[sequencing run]
    seqrun --> sampling[read sampling]
    covariates[covariates: depth, read length, run type, pipeline] --> sampling
    covariates --> count[observed count]
    sampling --> count
```

The chain sets map to the graph as follows. Z0 holds position and assay. Z1 adds cell type. Z2 and Z3 add donor and experiment and lab. Z4 to Z6 add biological replicate, library and sequencing run. Read sampling is the Poisson draw. Covariates act on the read sampling (depth) and on the observed count (read length, run type and pipeline settings change which reads are counted).

## rung-to-term-map

One row per rung of `layers.RUNGS` (in its order). "Shares" is the deepest chain set that both members of a pair hold in common. "Pair value" is `PV` of that set. "Layer increment" is the term that the difference to the row above isolates.

| rung (`layers.RUNGS`) | shares | pair value `0.5 E[(Y-Y')^2]` | layer increment over the previous row | assumptions |
|---|---|---|---|---|
| pseudoreplicate | Z6 | `T_s` | `T_s` (innermost rung: its own pair value) | (1), (2), (6) |
| sequencing | Z5 | `T_6 + T_s` | `T_6`, sequencing run | (1), (2), (5), (6) |
| technical | Z4 | `T_5 + T_6 + T_s` | `T_5`, library | (1), (2), (5), (6) |
| isogenic | Z3 | `T_4 + ... + T_s` | `T_4`, biological replicate | (1), (2), (5), (6) |
| anisogenic | Z1 in the nested order; it also shares the experiment | `T_2 + T_4 + T_5 + T_6 + T_s` (donor and below, with no experiment term) | `T_2`, donor | (1), (2), (3), (5), (6) |
| other_experiment | Z1 | `T_2 + T_3 + ... + T_s` | `T_3`, experiment and lab | (1), (2), (3), (5), (6) |
| other_cell_type | Z0 | `T_1 + ... + T_6 + T_s` | `T_1`, cell type | (1), (2), (5), (6) |

How `layers.py` computes these rows:
- It computes `Var(d)` with `d = log1p(a') - log1p(b')`, where a' and b' are binomially thinned to one common depth per assay (the shallowest side of any pair of that assay). Under exchangeability `E[d] = 0`, so `Var(d)` estimates `2 PV` on the log1p scale at that depth (assumptions 4 and 6). The variance is centred within each stratum and split exactly over Haar bands.
- The layer increment is the rung variance minus the variance of the nearest populated lower rung. When a rung is empty, the increment of the next populated rung is a sum of two or more terms.
- `resolution.py` reports half of the increment, i.e. `PV` units.
- The pseudoreplicate halves hold half the reads of the pooled track. Their pair value is `T_s` at half depth.
- `mixed_model.py` is the order-free cross-check of the anisogenic and other_experiment rows. It fits cell, experiment, replicate and library as crossed random effects with one fixed slope on log2 depth.

## metric-to-term-map

"Term" names the summands of [identity](#identity) and [model-variance](#model-variance). "Scale" is the space in which the metric is computed in `score.py` (counts_log1p = log1p of counts at the target's depth). `mu_hat` is the predicted NB mean and `n` its size.

| metric (key in `scores.json` or function) | term estimated | scale | assumptions |
|---|---|---|---|
| 1. `specific_variance_recovered_above_noise` (`metrics.specific_variance_recovered`) | `1 - (E_sp - N)/(V_sp - N)`: the share of the specific variance `T_1 + ... + T_6` that `mu_hat` explains. `V_sp` = the truth's variance minus its part shared with other cells (estimates `T_1 + ... + T_6 + T_s`). `N` = the Spearman-Brown noise floor from the target's two halves (estimates `T_s` only). The donor to sequencing layers therefore stay in the denominator. All terms are in units of each target's `Var(x)`. | counts_log1p (also signal_arcsinh for the EIC comparison) | (1), (2), (4), (5), (6) |
| 2. `r_spec` (`metrics.r_spec`) | `corr(y - g(a), p - h(a))`, with `a` the training average, an estimate of `E[Y|Z0]`. It is the correlation form of the same specific part, with no noise correction: its ceiling is below 1 by `T_s` and by every layer that the input does not determine. | counts_log1p | (2), (4), (5) |
| 3. `count_crps_skill_vs_position_average` (`metrics.crps_skill`) | `CRPS(position average) - CRPS(model)`. A proper score of the whole predictive law. The position average predicts `E[Y|Z0]`, so the skill reads the variance beyond Z0 that the model explains. It is no additive term of the identity. The `oracle_scaled` and `scale_error` split isolates the depth part of the covariate layer. | counts | (2), (5) |
| 4. correction gap closed (`metrics.gap_closed`) | `(CRPS_depth_only - CRPS_model)/(CRPS_depth_only - CRPS_floor)`: the share of the covariate layer beyond depth, `Var_C(E[Y|Z6,C])` minus its depth part, that the model removes. The floor is half against half, i.e. the sampling term only. Depth-only scaling is the baseline. | counts | (1), (2), (5) |
| 5. depth-law error (`metrics.depth_law_error`) | A first-moment check: `E[Y|Z6, depth]` is proportional to depth. It estimates no variance term. | counts (totals) | (5) |
| 6. PIT deviation, pooled and worst stratum (`metrics.pit_deviation`) | The whole law of `Y | S, C'`. Under a calibrated PIT within a stratum, the mean predicted variance `mu_hat + mu_hat^2/n` equals `E[(m - mu_hat)^2] + E[Var(Y|S,C')]` there ([model-variance](#model-variance)). | counts | (5) |
| 7. `predicted_over_replicate_variance` (`metrics.variance_ratio`) | Mean predicted NB variance over the pair value of one rung: the calibration ratio `rho` of [model-variance](#model-variance) for that rung. `score.py` calls it with the pseudoreplicate halves and the mean halved (`0.5 mu_hat`), so the headline is `rho(pseudoreplicate)` at half depth, with denominator `T_s`. | counts | (1), (2), (5), (6) |
| error budget `level` | `offset + scale + nonlinear` of `MSE/Var(x)`: first-moment error of the prediction (mean, scale, monotone shape). It is no layer term. | counts_log1p, units of `Var(x)` | (4) |
| error budget `shared` | Error energy common to the cells of an assay. It is the error in the `T_0` part. | counts_log1p, units of `Var(x)` | (4), (5) |
| error budget `specific` | Error energy specific to a track: the error in `T_1 + ... + T_6 + T_s`. | counts_log1p, units of `Var(x)` | (4), (5) |
| error budget `explained` | `C^2/P` per component (shared or specific part of one band): the truth variance that the calibrated prediction explains, capped at `V`. | counts_log1p, units of `Var(x)` | (4), (5) |
| error budget `noise` | `N`, the Spearman-Brown floor, in specific parts only: it estimates `T_s`. When `unexplained` would be negative, `N` is set to `V - explained` and `floor_flag` is set. | counts_log1p, units of `Var(x)` | (1), (2), (4), (6) |
| error budget `unexplained` | `V - explained - N`: in specific parts, the share of `T_1 + ... + T_6` that the prediction misses; in shared parts, the missed share of `T_0`. | counts_log1p, units of `Var(x)` | (1), (2), (4), (5) |
| error budget `added` | `E - (V - explained)`: error that the prediction adds through variance in the wrong place. It is no term of the truth's identity. | counts_log1p, units of `Var(x)` | (4) |
| error budget identities | `V = explained + unexplained + noise` and `E = unexplained + noise + added`, exact in each component and in the sums. | counts_log1p | none (algebraic) |
| D13 imputation term (`imputation_gap_closed`) | Headline 1, macro over scorable assays. | counts_log1p | as headline 1 |
| D13 correction term (`correction_gap_closed`) | Headline 4. | counts | as headline 4 |
| D13 uncertainty term (`uncertainty_gap_closed`) | `1 - PIT(arm)/PIT(marginal NB)`, i.e. headline 6 against the marginal NB. The marginal NB is one distribution per track, so its variance is the variance of Y over the positions of one track: the total of the identity with that track's own Z1 to Z6 held fixed. | counts | as headline 6 |
| D13 composite | `0.5 imputation + 0.25 correction + 0.25 uncertainty`, each term clipped to [0, 1] (`select_rule.CLIP_GOAL_TERMS`, PI ruling 18). A weighted sum of three shares, not a variance term. | mixed (as its terms) | as its terms |

## model-variance

Let S be the model's inputs and C' the target's covariates. Let `m = E[Y|S,C']` and let `mu_hat` be the predicted mean. The same law of total variance, with `m` split around `mu_hat`, gives

```
Var(Y) = Var(mu_hat) + E[(m - mu_hat)^2] + E[Var(Y|S,C')]
         - (E[m - mu_hat])^2 + 2 Cov(mu_hat, m - mu_hat).
```

The last two terms form the cross term. Both terms are zero when the mean is calibrated, i.e. `E[Y | mu_hat] = mu_hat`. So for a calibrated mean the total splits into the explained variance `Var(mu_hat)`, the missed signal `E[(m - mu_hat)^2]`, and the variance that S cannot determine, `E[Var(Y|S,C')]`.

**What the predicted NB variance estimates.** The predicted variance `mu_hat + mu_hat^2/n` estimates `E[Var(Y|S,C')]`. Its part `mu_hat` is the Poisson sampling term `T_s`. Its part `mu_hat^2/n` is the sum of the layers `T_k` that S does not determine.

**Two bounds.**
- Lower bound: `E[Var(Y|S,C')] >= T_s = PV(Z6)`, the pseudoreplicate pair value. No input removes the read sampling of the target.
- Upper bound: `E[Var(Y|S,C')] <= PV(Zk*) = E[Var(Y|Zk*)]`, where Zk* is the deepest chain set that is a function of (S, C'). Conditioning on more information never raises the expected conditional variance. For imputation of a missing assay from other assays of the same cell, Zk* is typically Z1 (cell type), so the bound is the other_experiment pair value. For reconstruction of a track whose own reads are in S, Zk* is deeper.

**Calibration ratio per rung.** For each rung r with pair value `PV_r`, define

```
rho(r) = mean(mu_hat + mu_hat^2/n) / PV_r     (both at one depth, over the same bins)
```

A model whose predicted variance lies within the bounds has `rho(pseudoreplicate) >= 1` and `rho(r*) <= 1` at the rung r* that shares Zk*. Headline 7 is `rho` at one rung; `score.py` uses the pseudoreplicate rung, the lower bound.

## scales

This section states the identity across bin sizes (plan Amendment 3, section C). It needs an additive measure: one count per read at one position (the point rule), so that the count at any bin size is the exact sum of the 1 bp counts. The overlap rule of the stored 25 bp counts (`tools/t112/bin25.py`) is not additive across bin sizes.

**Block identity.** Let Y be the 1 bp count at a position drawn at random from a region, and Z_b the index of the block of b bp that holds it. Then

```
Var(Y) = E[Var(Y | Z_b)] + Var(E[Y | Z_b]).
```

The first term is the variance below b. The block sum is `b E[Y|Z_b]`, so the variance of block sums is `b^2 Var(E[Y|Z_b])`.

**Nested chains.** The block partitions of a chain refine each other, so the identity telescopes exactly. For the chain 1, 5, 25, 125 bp:

```
Var(Y) = E[Var(Y|Z_5)] + E[Var(E[Y|Z_5] | Z_25)] + E[Var(E[Y|Z_25] | Z_125)] + Var(E[Y|Z_125]).
```

The chain 1, 5, 25, 50 bp has the same form. On a finite region the identity holds exactly for the empirical variance when the region length is a multiple of the largest block: the analysis uses the first 409,500 bp (3,276 blocks of 125 bp) for the chain that ends at 125 bp and the whole 409,600 bp (8,192 blocks of 50 bp) for the chain that ends at 50 bp. The Haar bands of `layers.py` and `error_budget.py` are the same identity with dyadic blocks of 25 x 2^j bp.

**Sampling term across scales.** At 1 bp under the point rule, the count is Poisson given its rate. The Poisson part of `Var(Y)` is `E[Y]`. Of that part, `(1 - 1/b) E[Y]` falls below b and `E[Y]/b` remains in the block means.

**Replicate-reproducible share.** For two pseudoreplicate halves h1 and h2 of one track, the read sampling is independent between halves given the rate. So `Cov(h1, h2)` over positions contains no sampling term. The same holds for the block means of the halves at each block size. The reproducible share at each step of a chain is the step term computed from the cross-covariance of the halves in place of the variance.

**log1p.** On log1p of the block sums the identity holds for each fixed b, but the terms are not additive across b: log1p of a sum is not the sum of log1p values. Results on that scale are reported as not additive.

## assumptions

Each assumption is a condition under which an estimator of [rung-to-term-map](#rung-to-term-map) or [metric-to-term-map](#metric-to-term-map) estimates its term without bias.

1. **Exchangeable pair.** The two members of a pair are conditionally independent and identically distributed given the set that they share. A pair that shares reads (for example, a depth subset of a parent) violates it. Pseudoreplicate halves meet it only when the read split is random and the reads are independent given Z6.
2. **Equal covariates, above all equal depth.** Both members of a pair, both rungs of a difference, and the model and the truth are at one depth on the scale used. On counts, `T_s` scales with depth and each `T_k` scales with depth squared. The tools thin the deeper track binomially to the shallower depth. Pseudoreplicate halves are at half the depth of the pooled track.
3. **Order of conditioning.** The chain nests donor before experiment. In ENCODE donor and experiment are crossed, so the anisogenic and other_experiment rungs do not isolate one nested term each. `mixed_model.py` is the order-free check.
4. **Scale.** The identity is exact for counts. On log1p it holds only at one fixed depth, and its sampling term then depends on the rate of the bin (it is not `E[Y]`). The error budget, headlines 1 and 2, and `layers.py` work on log1p.
5. **Same cells and positions.** Both rungs of a difference, and the model and the baseline of a skill, use the same cells, regions and strata. A difference over two different cell sets mixes a layer with the variance between cell sets.
6. **Factor 0.5 and the mean term.** `0.5 E[(Y - Y')^2]` equals `E[Var(Y|Z)]` only when `E[Y - Y' | Z] = 0`. The variance of the difference, `Var(Y - Y')`, removes the mean of the difference; the mean square difference keeps it. The two agree only for an exchangeable pair at equal depth. `layers.py` reports `Var(d)`, i.e. twice the pair value; `metrics.variance_ratio` and `resolution.py` use half of it.
