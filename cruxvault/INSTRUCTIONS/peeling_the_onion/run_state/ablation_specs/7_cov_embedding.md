# Ablation 7: covariate embedding (`cov_embedding`)

## 1. Question

The PI wrote (`peeling the onion.md`, "# miniCANDI", important notes): "how we embed and fuse different modalities i.e. epigenomic count data, dna sequence, and covariate information is an important an non trivial part of the whole architecture design. these are targets for ablation i.e. covariate embedding and covariate injection, DNA embedding and injection, etc."

This ablation asks the embedding half only: does the way the covariate vector C (input side) and C' (output side) is turned into a vector change imputation, correction or calibration? The injection half (where and how the vector enters the network) is a separate question.

## 2. Arms

Today's embedding (`tools/minicandi/covariates.py`, `CovariateEmbedding`): the 11 metadata columns are 5 numeric columns (log2 depth, paired-end flag, read length, deduplication flag, MAPQ threshold), each standardized with fixed constants, and 6 categorical columns (pipeline, pipeline version, lab, platform, antibody, donor), each with its own learned table `nn.Embedding(1024, 8)`. A 2-layer MLP maps the concatenation to 32 numbers. Each architecture has two such embeddings (one for C, one for C'). I call this "factorized": each field has its own encoding.

What the categorical columns hold in `eic_v1` (Fir, `/project/def-maxwl/mforooz/ONION/tables/eic_v1`, read 2026-11-16): antibody and donor are empty in all 2,740 rows (vocabulary size 0); pipeline version maps 1:1 to pipeline (4 values each); lab is empty in 2,662 rows; platform has 21 values. So the categorical content is pipeline, platform and, rarely, lab. Every correction knob of the evaluation set (depth, nodedup, mapq, crop, se) changes a numeric column only. For example, a crop pair differs in read length and depth only.

Arms (each on conv autoencoder, conv + attention and U-Net, base recipe `final/rung{2,3,4}/reference_equal.json`, 3 seeds):

| arm | config change | what changes |
|---|---|---|
| reference | none | the base recipe; existing runs `runs_fir/rung{2,3,4}_base_heldout` |
| `cov_numeric_only` | `covariate_embedding: numeric_only` | every categorical id is read as 0, so pipeline, platform and lab are not seen. The numeric columns are unchanged. |
| `cov_dropout` | `covariate_dropout: 0.1` | in training only, each (window, track) row of the encoder copies of C and C' is replaced by one learned null vector with probability 0.1. The depth copies that the count transform and the NB head read are never dropped. |
| `cov_numeric_per_value` (new, lowest priority) | `covariate_embedding: numeric_per_value` | the 4 discrete numeric columns (paired-end flag, read length, deduplication flag, MAPQ threshold) each get a learned table, one vector per value, in place of one standardized scalar. Log2 depth stays a scalar. |

Readings I considered and rejected:
- **Categorical only.** It hides depth and the processing knobs from both encoders. Depth hiding is already measured (COVERAGE rows 27 to 29), and every correction knob is numeric, so the result is predictable.
- **One joint embedding per full covariate vector.** It cannot represent a covariate combination that training never saw. The correction score is on withheld combinations (crop36 to crop50 and back), so the arm fails by construction.
- **`inject_adaln_zero`.** It changes the injection form, not the embedding. `miniCANDI architecture ablations.md` lists adaLN-Zero as option 3 of "Covariate injection", beside concatenation, FiLM and tokens. It belongs to the injection question (COVERAGE rows 13 to 15), not here.

`cov_dropout` is the "learned null token trained by conditioning dropout" option that the ablations doc lists for C and C' of missing assays (Ho and Salimans 2022). p = 0.1 is the value of `COV_DROPOUT` in `stage2_arms.py` (an agent call; the plan did not pin it). `cov_numeric_per_value` asks whether an ordered scalar or an unordered per-value code is the better encoding for processing settings. This decides how CANDI encodes these settings when it scales. A per-value code cannot read a value that training never saw; a scalar can interpolate. Drop this arm first if the budget is cut.

## 3. Prior leads

1. Scrambled covariates (COVERAGE.md row 11; older composite, version 2, so the values do not compare with the reference table above). Rung 1 specific variance -0.019, beyond. U-Net specific variance -0.029, beyond. U-Net gap closed +0.101 on all pairs, but on the 18 pairs with no duplicate-keeping side, per seed 0.367, 0.448 and 0.445 against base 0.519, 0.545 and 0.492 (f79: -0.099, below 2 seed spreads). Prediction: the covariates as a whole carry imputation information. This lead does not say which columns carry it.
2. Scramble everything, depth included (row 12, rung 1, 2 seeds): uncertainty -0.280, beyond. Hard offset at U-Net (row 27, f81): composite -0.184 and uncertainty -0.698, beyond. Prediction: depth is the most important covariate. All arms here keep depth.
3. Depth hidden on the input side only (row 28): clean-pair gap closed +0.082, beyond, but composite -0.062. Prediction: removing information from an encoder can help correction on clean pairs while it lowers imputation.
4. Injection form (row 13, rung 1, 2 seeds): concatenation against FiLM, composite +0.007, within. Axial transformer with FiLM against tokens (`FINAL_ROUND_EQUAL.md` line 23 against line 18): 0.521 (0.026) against 0.517 (1 seed). Prediction: the covariate path is not sensitive to its form at this budget, so small embedding changes give small effects.
5. Archive vault, `CANDI/cruxvault/h44_*.md`: in the old CANDI data, H(platform | assay, read length) = 0.443 bits, i.e. platform is not fully redundant. Prediction: `cov_numeric_only` can lose a small amount of imputation.
6. Archive vault, `h47_*.md`: in old CANDI, weight decay drove the metadata embeddings to absmax 6.3e-41. Removing the decay improved macro CRPS from 1.495 to 1.341. Here AdamW uses decoupled decay 0.01 at lr 1e-3, over 80,000 / 8 = 10,000 steps: at most a factor (1 - 1e-5)^10,000 = 0.905. So the old collapse cannot repeat, and no arm is needed for it.
7. CAC test bed, q7 and h19: g(c, c') fits trained base-to-arm pairs and misses never-trained arm-to-arm pairs. Routing through the base fixes it (f31). Prediction: per-value codes, which do not share structure across values, will not transfer better to withheld combinations.

## 4. Expectation (before any run)

| arm | imputation | correction | calibration | composite |
|---|---|---|---|---|
| `cov_numeric_only` | 0 to -0.01, within range | within range | within range | within range |
| `cov_dropout` | within range | 0 to -0.02, within range | within range | within range |
| `cov_numeric_per_value` | within range | 0 to -0.03, within range | within range | within range |

The calibration seed ranges (0.108 to 0.178) are wide, so a calibration effect smaller than 0.18 is not detectable.

## 5. How the result answers the question

"Beyond" means the difference of means is larger than the larger of the two seed ranges. Judge each architecture separately; a conclusion holds for all three only when the three agree.

| outcome | conclusion |
|---|---|
| `cov_numeric_only` within range on all four metrics | Pipeline, platform and lab add nothing at this budget. Keep the simpler numeric-only embedding (`simpler_than_base` is true for this arm). |
| `cov_numeric_only` imputation lower, beyond | The categorical columns carry batch information for imputation. Keep them; for CANDI, add an explicit UNKNOWN value (design notes, "covariate-vector-contents"). |
| `cov_numeric_only` correction lower, beyond | Unexpected, because the crop knob is numeric. Check for a wiring fault before any conclusion. |
| `cov_dropout` imputation or calibration higher, beyond, and correction within | Conditioning dropout regularizes the covariate path. Adopt it. |
| `cov_dropout` correction lower, beyond | The model needs C' on every correction target. Do not adopt it. |
| `cov_numeric_per_value` correction lower, beyond | An ordered scalar transfers better to withheld combinations. Keep scalars. |
| `cov_numeric_per_value` within or higher | Per-value codes are not worse on seen values. Keep scalars, because they also read unseen values. |
| all arms within range everywhere | The embedding is not a sensitive part of the design at 80,000 windows. Choose the simplest one. |

## 6. Controls and traps

- **Parameter count.** `cov_numeric_only` keeps the 6 tables, so the count is unchanged, but only row 0 of each table trains. `cov_dropout` adds 2 null vectors of 32 = 64 parameters. `cov_numeric_per_value` adds 19,968 = 2 embeddings x (4 tables x 256 rows x 8 + 28 x 64 in the first MLP layer). Every architecture fits its trunk width to `param_budget`. At the default budget, the conv autoencoder dropped from width 60 to 56 (956,034 parameters), which is a trunk change. The arm therefore sets `param_budget` to 1,019,968 (1.0e6 + 19,968), so all three trunks keep their base widths (60, 104, 25) and only the embeddings grow. All arms stay within 10% of 1.0e6.
- **Off-switch identity.** Checked today: the base config and the base with `covariate_embedding: full, covariate_dropout: 0.0` give identical `state_dict` tensors and identical eval forward passes on all three architectures.
- **FiLM is identity at initialization.** At initialization, every arm gives the same output and the categorical tables get no gradient. A wiring test must therefore perturb the weights first. After a perturbation, `cov_numeric_only` equals the base on a batch with its categorical columns set to 0, and all 12 tables get a gradient in the base.
- **A platform value seen only in evaluation.** `OBI:0002000,OBI:0002002` occurs in 9 rows of one experiment (ENCSR000APD, H3K79me2, cell C24) and in no training measurement. Its table row stays at its random initialization. It touches 1 of 63 imputation targets (as the target) and 30 of 1,480 correction pairs, and none of the 40 crop pairs. `cov_numeric_only` removes this row. Control: report the imputation term with and without that target.
- **Dropout acts in training only.** At evaluation, a missing track's C row is all zeros, as in the base. The null vector is never used in scoring, so `cov_dropout` is a regularizer only. The scorer runs `model.eval()` (`predict.py` line 200).
- **The scorer input is unchanged.** No arm changes the dataset, the evaluation set or the depth that the count transform and the NB head read.

## 7. Readiness

All three arms are ready on all three architectures. The configs are in `tools/minicandi/configs/ablations/cov_embedding/rung{2,3,4}/<arm>.json`. Parameter counts are from `registry.build_model` on the written configs:

| architecture | `cov_numeric_only` | `cov_dropout` | `cov_numeric_per_value` |
|---|---|---|---|
| conv autoencoder | 1,042,514 | 1,042,578 | 1,062,482 (width 60) |
| conv + attention | 1,038,018 | 1,038,082 | 1,057,986 (width 104) |
| U-Net | 993,599 | 993,663 | 1,013,567 (width 25) |

`stage2_arms.py write` wrote `cov_numeric_only` and `cov_dropout`, with `arms.tsv`. I wrote `cov_numeric_per_value` by hand: `covariate_embedding: numeric_per_value`, `param_budget: 1019968`, `simpler_than_base: false`. A JSON diff against `reference_equal.json` showed only these keys. All three architectures read the keys: `component_kwargs` passes them to every `CovariateInjector`, and each `forward` passes `covariate_dropout` to `model_covariates` (`rungs/conv_ae.py` lines 161 to 222, `conv_attn_ae.py` 137 to 181, `unet.py` 180 to 257). A training-mode forward and backward pass on a synthetic batch reached the null vectors and all 8 per-value tables. No arm needs a non-default submit option, because no arm changes the trunk or the data path.

Code: `covariates.py` +19 lines (`PER_VALUE`, `PV_TABLE` = 256, `per_value` in `CovariateEmbedding`; id = round(value), or round(2 x value) for the flags, which hold 0, 0.5 or 1); `components.py`, 3 lines changed (`COVARIATE_EMBEDDINGS`, the `per_value` flag). Test: `tests/test_ablation_cov_embedding.py`. Off switch: I built the base configs with the changed code and with the unchanged code of the lead worktree. Both builds gave identical `state_dict` tensors and identical forward passes (weights perturbed, one synthetic batch) on all three architectures.

Cost per run, from Fir `sacct` (jobs COMPLETED since 2026-11-15, the base and final runs of each architecture). Training uses 12 CPUs; prediction uses 4 shards x 8 CPUs; metrics use 16 CPUs.

| architecture | training | prediction | metrics | per run |
|---|---|---|---|---|
| conv autoencoder | 12 x 3,412 s = 11.38 | 32 x 998 s = 8.87 | 16 x 1,967 s = 8.74 | 28.99 CPU-h |
| conv + attention | 12 x 1,956 s = 6.52 | 32 x 956 s = 8.50 | 16 x 1,921 s = 8.54 | 23.56 CPU-h |
| U-Net | 12 x 4,322 s = 14.41 | 32 x 1,275 s = 11.33 | 16 x 2,050 s = 9.11 | 34.85 CPU-h |

One seed of each architecture costs 87.39 CPU-h. One arm (3 seeds) costs 262.2 CPU-h. The two arms from `ARMS` are 18 runs, 524.4 CPU-h. All three arms are 27 runs, 786.5 CPU-h. The reference runs exist and cost nothing.
