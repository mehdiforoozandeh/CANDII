# Ablation 14: counterfactual processing views as training data, on the conv models (views)

Status: Phase-1 specification and built configs, 2026-11-16 (close-out item A3). No run submitted.

## 1. Question

Does training on the counterfactual processing views of each experiment (depth thinning; read-level: single-end, read crop; alignment-level: no duplicate removal, no MAPQ filter) raise imputation, correction or calibration on the conv autoencoder and conv + attention, and on which pair type? The PI's line (2026-11-16): "where and if adding counterfactual data (augmenting EIC) actually helps". `peeling the onion.md`:93: "we can also use the same data views as augmentations of the data and use it for training."

## 2. Arms

| arm | keys | role |
|---|---|---|
| all views (base) | none (every view kind; `same_view_prob` 0.5) | reference; the existing runs `rung{2,3}_base_heldout`, already scored in `p16_sv` |
| `base_only` | `train_views: [base]` | no counterfactual view |
| `base_depth` | `train_views: [base, depth]` | depth thinning only |
| `base_fastq` | `train_views: [base, se, crop]` | read-level views only |
| `base_alignment` | `train_views: [base, nodedup, mapq]` | alignment-level views only |
| `all_same_view_only` | `same_view_prob: 1.0` | every view, but no cross-view training pair |

The arms are the U-Net arms of `tools/minicandi/configs/views/rung4/views_<arm>.json`, put on the base recipes `tools/minicandi/configs/final/rung{2,3}/reference_equal.json`. The configs are `tools/minicandi/configs/ablations/views/rung{2,3}/<arm>.json`. `tests/test_ablation_configs.py` checks that each differs from `reference_equal.json` in its view key only (plus the label `simpler_than_base`: true for the four view subsets, null for `all_same_view_only`), that it builds within 10% of 1.0e6 parameters, and that its view keys equal the U-Net arm's.

What the switch changes (`tools/minicandi/dataset.py`, module docstring "Training views"): data path only, train mode only. `train_views` removes the unlisted view kinds from each track's training candidates; a track left with no listed view is a missing track of its cell. `same_view_prob` at 1.0 removes every cross-view pair, also from the fallback list. Validation, blind items and the scorer do not change. The model does not read either key.

The U-Net arms exist on Fir (`runs_fir/rung4_views_<arm>`, 3 seeds, trained and scored in `p16` only). They have no `p16_sv` score, so no correction-only term (the same-view pass). Part 2 of the submit script scores them into `p16_sv`. The U-Net is not retrained.

## 3. Prior leads

1. U-Net, composite version 3 (`select_rule.py arm-summary --sub p16`, kit CM; `cruxvault/results/closeout_A3/WHERE_VIEWS.md` section 1): all views 0.495 (0.010); base only 0.517 (0.039); base + depth 0.509 (0.037); base + read-level 0.498 (0.045); base + alignment-level 0.505 (0.023); same-view pairs only 0.362 (0.058), correction term below 0 in all 3 seeds. No arm is beyond the reference on any composite term.
2. U-Net per pair type (`WHERE_VIEWS.md` section 2): (a) depth-thinning pairs (516, outside the composite): the arms with depth views 0.614 (0.012) and 0.586 (0.057); the arms without depth views 0.127 (0.768), 0.355 (0.255), -0.012 (0.864); all 6 seeds with depth views lie above all 9 seeds without them. (b) MAPQ + no-dedup pairs (84, outside the version-3 term): base + alignment-level 0.018 (0.071) against all views 0.272 (0.050), beyond 2x. (c) crop pairs (14): every arm 0.507 to 0.532.
3. Copy gain (`correction_metric_review/REVIEW.md` section 3): the relative same-view copy gain is 0.59 to 0.66 for the U-Net and 0.04 to 0.09 for the conv models; on MAPQ + no-dedup pairs the U-Net scores 0.27 and the conv autoencoder 0.50. Prediction: the alignment-level drop (lead 2b) is smaller on the conv models.
4. Training rows by view kind (`in_eic = train`, plan Log 2026-11-16T02:15Z): base 678, depth 936, mapq 308, nodedup 312, se 38, crop 56. At equal windows, an arm with fewer views draws the base view more often; single-end and crop views are rare.
5. Base references, `p16_sv`, Fir (`arm-summary`): conv autoencoder composite 0.528 (0.045), variance recovered 0.382 (0.028), correction v3 0.718 (0.012), calibration 0.631 (0.178), correction-only 0.690 (0.019); conv + attention 0.525 (0.048), 0.387 (0.012), 0.673 (0.031), 0.652 (0.157), 0.660 (0.044).

## 4. Expectation (before any run)

- Composite, imputation, calibration and the version-3 correction term: inside the seed range for every arm except `all_same_view_only`, on both conv models.
- `all_same_view_only`: correction term (crop pairs) below 0 or near it, as on the U-Net.
- Depth-thinning pairs: the arms without depth views (`base_only`, `base_fastq`, `base_alignment`) are lower and less stable than all views and `base_depth`.
- MAPQ + no-dedup pairs: `base_alignment` is below all views, by less than on the U-Net (lead 3).
- Correction-only term: no arm other than `all_same_view_only` moves it beyond the range.

## 5. How the result answers the question

"Beyond" = the difference of means is larger than the larger of the two seed ranges (3 seeds). D13 (2x the spread, no goal drop beyond 2x) is the binding rule for the design.

| outcome | conclusion |
|---|---|
| all views beyond base only on a composite term on 2+ of 3 architectures | the views help the scored task; keep them as training data |
| base only beyond all views on 2+ architectures | the views hurt; drop the view type that causes it (by the per-knob table) |
| within the range on the composite on all three, and depth-thinning pairs separate as on the U-Net | keep depth views: they are needed for depth correction, which the composite does not score; by D13 no other view type earns a place (base only is simpler) |
| within the range everywhere on all three | the views neither help nor hurt at this scale; the counterfactual data serves evaluation only |
| alignment-level views hurt on MAPQ + no-dedup pairs on 2+ architectures | drop alignment-level views from training, or train cross-view pairs on disjoint read halves (`REVIEW.md` section 3) |

## 6. Controls and traps

1. The composite's correction term rests on 18 crop pairs with no alignment-level and no depth pair. The report carries the per-knob table and the depth-thinning column beside the composite (`cruxvault/results/closeout_A3/where_views.py`, run with `--ref rung{2,3}_base_heldout --sub p16_sv`).
2. Equal windows (lead 4).
3. The U-Net view configs lack the `heldout_loss` key of the U-Net reference; the conv arms are built on `reference_equal.json` and carry it, so they differ from their reference in the view key only.
4. Code drift. The training-path files of kit COA3 (`dataset.py`, `losses.py`, `contract.py`, `train.py`, `registry.py`, `rungs/`, `train.sh`, `submit_arms.sh`, `score.py`, `predict_store.py`) have the same md5 as kit CM. `components.py` and `covariates.py` differ by the `numeric_per_value` covariate arm only, which is off by default (`diff` of CM against AB on Fir).
5. Same-view-only arm: the correction term can be far below 0 (U-Net seed values -30.1, -0.9, -10.6), so its composite is set by the clipped term; quote the raw term.

## 7. Readiness

| architecture | state |
|---|---|
| conv autoencoder | ready: 5 configs, test green, kit COA3 on Fir |
| conv + attention | ready (same) |
| U-Net | trained and scored in `p16`; needs the `p16_sv` score pass only (part 2) |

Submit (foreman): on a Fir login node, `DRY_RUN=0 bash /project/def-maxwl/mforooz/onion_kit/COA3/tools/minicandi/configs/ablations/views/submit_views.sh`. The dry run (default) printed 255 sbatch lines: part 1, 30 runs x (1 training + 4 predict + 1 metrics) = 180 jobs; part 2, 15 seeds x (4 predict + 1 metrics) = 75 jobs. Run directories: `runs_fir/abl_views_<arm>_{convae,convattn}/seed{0,1,2}`; U-Net scores in `runs_fir/rung4_views_<arm>/seed<s>/p16_sv`. Training limit 03:00:00 (submit_all.sh default).

Cost (per-seed figures of `1_control_input.md` section 7, `sacct` CPUTimeRAW of the base jobs):
- conv autoencoder: 32.5 CPU-h per seed; conv + attention: 22.4 CPU-h per seed.
- Part 1: 5 arms x 3 seeds x (32.5 + 22.4) = 15 x 54.9 = 823.5 CPU-h.
- Part 2: U-Net scoring only, 12.21 + 9.00 = 21.2 CPU-h per seed; 15 x 21.2 = 318 CPU-h.
- Total: 823.5 + 318 = 1,142 CPU-h. The references are reused (0 runs).
