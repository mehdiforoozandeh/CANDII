# Why the held-out loss and the composite disagree

Chunk C8 of the miniCANDI recipe run (task t370, requirement R12). Read 2026-11-21. No job was submitted and no score file was written. All reads were read-only Python on the Fir login node.

## Question

Question q46 (finding f88) asks two things.

1. Why does the held-out NB loss rank the U-Net first and the two conv models last, while the composite ranks the two conv models first?
2. Why does more training (160,000 windows instead of 80,000) lower the held-out loss without raising the composite?

Terms used below:

- **Held-out loss**: the `heldout_loss` key of `log.jsonl`, i.e. the training loss of the config (default: NB negative log-likelihood, or NLL) on a fixed set of training-sampler batches drawn with `HELDOUT_SEED = 20261115`, whose windows overlap no evaluation region (`tools/minicandi/train.py`, `heldout_batches`, `heldout_loss`). Lower is better. I quote the last logged value.
- **Composite**: `composite_v3` = 0.5 x imputation + 0.25 x correction + 0.25 x calibration, each term clipped to 0..1 (`tools/onion_eval/select_rule.py`).
- **Target kind**: the role of a target track in a training window (`tools/minicandi/dataset.py`, `plan`). Kind 1 (impute): the track is not an input. Kind 2 (correct): another view of the same experiment is the input, e.g. the crop-36 view as input and the base view as target. Kind 3 (same view): the input is the same file, with some bins hidden.
- **Copy gain**: `copy_leak.same_view_copy_gain_relative` in `scores.json`, i.e. how much the model's same-view output improves when the hidden bins' input is visible. It is a measure of how much a model copies its input.
- **Seed range**: max minus min over seeds 0, 1, 2. I call an effect real only when it exceeds twice the larger seed range of the two arms compared.

## Evidence

Sources. Fir `/project/def-maxwl/mforooz/ONION/runs_fir/<arm>/seed{0,1,2}/log.jsonl` (held-out loss; training loss parts) and `.../p16_sv/scores.json` or `.../p16/scores.json` (scores), read through `select_rule.table` of kit `/project/def-maxwl/mforooz/onion_kit/CO10`. The correction column is `correction_gap_closed_oracle_no_nodedup_side` (the 18 crop pairs), the term that composite v3 uses. All runs below use the default loss (`nb_nll`, unit kind weights) on the `eic_v1` tables, 16-region screening set, 3 seeds.

### E1. What each measure averages over (from the code)

| | held-out loss | composite |
|---|---|---|
| cells and tracks | training tracks of training cells, at held-out genome positions | imputation: validation tracks (a cell-assay pair never trained); correction: 18 never-trained crop pairs in 6 cells |
| targets | all three kinds, pooled | imputation: kind 1 only; correction: a different view of the same experiment, like kind 2 |
| bins | every bin of kinds 1 and 2, the hidden bins of kind 3, pooled into one mean (`contract.loss_mask`, `losses.total_loss`) | per target over all bins of 16 regions, then macro over assays (`score.py`, `metrics.py`) |
| space | NB NLL of raw counts | imputation: specific variance recovered above the noise floor in `counts_log1p` (`metrics.specific_variance_recovered`); correction: CRPS gap closed (`metrics.gap_closed`); calibration: PIT deviation against the marginal NB |
| source of the target | kind 2 and kind 3 inputs are full views of the same reads as the target | correction scores the source's pr1 half against the target's pr2 half, i.e. disjoint reads |

The last row is the key difference. A training pair of kind 2 or kind 3 shares most of its reads with its target. So a model that copies its input with a tight NB gets a low loss on those bins. The correction term predicts a disjoint half, so a sharp copy carries the source half's sampling noise and scores worse (`correction_metric_review/REVIEW.md` in the onion worktree, section 2, suspect a).

### E2. The three base models

| model (arm) | held-out loss | imputation | correction (crop pairs) | calibration | composite | copy gain |
|---|---|---|---|---|---|---|
| conv autoencoder (`rung2_base_heldout`) | 0.9680 (0.0041) | 0.3822 (0.0284) | 0.7177 (0.0120) | 0.6309 (0.1778) | 0.5283 (0.0451) | 0.039 to 0.043 |
| conv + attention (`rung3_base_heldout`) | 0.9586 (0.0037) | 0.3868 (0.0121) | 0.6729 (0.0313) | 0.6522 (0.1571) | 0.5247 (0.0483) | 0.053 to 0.055 |
| U-Net (`rung4_base_heldout`) | 0.9158 (0.0028) | 0.4319 (0.0230) | 0.5072 (0.0712) | 0.6096 (0.1077) | 0.4952 (0.0100) | 0.634 to 0.654 |

Mean (seed range); scores from `p16_sv`. These values match `results/final_round/FINAL_ROUND_EQUAL.md` in the onion worktree to the third decimal.

- The U-Net's held-out loss is 0.0522 below the conv autoencoder's. That is 13 times the larger seed range (0.0041).
- The U-Net leads on imputation by 0.0497 (larger range 0.0284; under twice the range).
- The U-Net trails on correction by 0.2105 (larger range 0.0712; about three times the range).
- Composite arithmetic, U-Net minus conv autoencoder: 0.5 x (+0.0497) + 0.25 x (-0.2105) + 0.25 x (-0.0213) = 0.0249 - 0.0526 - 0.0053 = -0.0331, which equals 0.4952 - 0.5283. The correction deficit outweighs the imputation lead by a factor of two.

### E3. Where the U-Net's loss advantage sits, by target kind

`log.jsonl` stores the loss per kind (`parts.impute`, `parts.correct`, `parts.same`) for training batches only. The held-out loss has no per-kind split. I took the mean of each part over the last 20 logged training batches of each run-seed. Each seed's sampler does not depend on the architecture, so for one seed the two arms see the same windows (I did not verify this byte for byte). Per-seed differences, U-Net minus model, seeds 0 / 1 / 2:

| difference | held-out loss | kind 1 (impute) | kind 2 (correct) | kind 3 (same view) |
|---|---|---|---|---|
| U-Net minus conv autoencoder | -0.052 / -0.050 / -0.055 | -0.012 / -0.010 / -0.007 | -0.096 / -0.105 / -0.099 | -0.062 / -0.055 / -0.057 |
| U-Net minus conv + attention | -0.040 / -0.043 / -0.046 | -0.013 / -0.007 / -0.005 | -0.068 / -0.088 / -0.089 | -0.024 / -0.032 / -0.046 |

The U-Net's advantage on kind 1, the only kind that resembles the imputation task, is 0.005 to 0.013. Its advantage on kinds 2 and 3, where the input holds the target's own reads, is 2 to 18 times larger, per seed. Its copy gain is 0.63 to 0.65 against 0.04 to 0.06 for the conv models (E2). So the U-Net wins the loss mainly by copying. The same copying lowers the correction term.

### E4. Across 115 arms

Arm means of every Fir arm with 3 scored `p16_sv` seeds, 80,000 windows, default loss, no loss weights, `eic_v1` tables (115 arms, all architectures mixed). Spearman rank correlation of the held-out loss with:

| quantity | Spearman |
|---|---|
| training loss, kind 3 | +0.90 |
| training loss, kind 2 | +0.83 |
| training loss, kind 1 | +0.71 |
| imputation term | -0.58 |
| correction term | +0.28 |
| calibration term | -0.09 |
| composite | -0.07 |

A lower held-out loss goes with a higher imputation term and a lower correction term. The two effects cancel in the composite, which has no rank relation with the loss. Within one architecture the relation with the composite is weak: Spearman -0.29 (conv autoencoder, 33 arms), -0.28 (conv + attention, 36 arms), -0.60 (U-Net, 31 arms). I did not compute confidence intervals for these correlations.

### E5. More training (160,000 windows)

The 160,000-window runs (`rung{2,3,4}_windows160k`) have no `p16_sv` score; `select_rule` shows them as not scored there. I read their `p16` scores. For the base arms, `p16` and `p16_sv` give the same three terms to 4 decimals (checked for all 9 base run-seeds), so the comparison is on one tier.

| model | held-out loss, 80k to 160k | imputation | correction | calibration | composite | copy gain |
|---|---|---|---|---|---|---|
| conv autoencoder | 0.9680 to 0.9415 (ranges 0.0041, 0.0016) | 0.382 to 0.358 (0.028, 0.031) | 0.718 to 0.660 (0.012, 0.026) | 0.631 to 0.615 | 0.528 to 0.498 (0.045, 0.042) | 0.039-0.043 to 0.079-0.088 |
| conv + attention | 0.9586 to 0.9318 (0.0037, 0.0010) | 0.387 to 0.353 (0.012, 0.032) | 0.673 to 0.652 (0.031, 0.028) | 0.652 to 0.601 | 0.525 to 0.490 (0.048, 0.034) | not read |
| U-Net | 0.9158 to 0.9020 (0.0028, 0.0009) | 0.432 to 0.451 (0.023, 0.010) | 0.507 to 0.553 (0.071, 0.035) | 0.610 to 0.626 | 0.495 to 0.520 (0.010, 0.030) | 0.634-0.654 to 0.622-0.643 |

- The held-out loss falls in all three models by 5 to 16 times the larger seed range.
- Every composite change is inside twice the larger seed range.
- The one term change beyond twice the range is the conv autoencoder's correction drop (-0.058 against 2 x 0.026 = 0.052), and its copy gain doubles at the same time.
- The conv models' imputation falls slightly and the U-Net's rises slightly; neither change exceeds twice the range.

The same pattern holds for the tuned recipes of f88. Each was chosen on the held-out loss of seed 0. Each lowered the held-out loss beyond twice the seed range (conv autoencoder 0.9680 to 0.9537; conv + attention 0.9586 to 0.9477; U-Net 0.9158 to 0.9098). No tuned composite moved beyond twice the seed range, and the conv autoencoder's correction fell from 0.718 to 0.644 (ranges 0.012, 0.019), which is beyond twice the range.

## Answer

The held-out loss and the composite measure different tasks, so they rank models differently. The held-out loss is one pooled NB NLL over raw counts at held-out positions of training tracks, and most of its bins are kind 2 and kind 3 targets whose input carries the target's own reads. On those bins a model gains by copying its input with a tight NB. The U-Net copies far more than the conv models (copy gain 0.63 to 0.65 against 0.04 to 0.06), and its loss lead over the conv autoencoder sits mostly in those bins (0.055 to 0.105 on kinds 2 and 3 against 0.007 to 0.012 on kind 1). The composite scores correction on disjoint read halves, where a sharp copy carries the source half's noise. So the U-Net's copying gives it the best loss and the worst correction term (0.507 against 0.718). Correction has weight 0.25, and the U-Net's correction deficit (0.2105) is about four times its imputation lead (0.0497, weight 0.5), so the composite ranks the conv models first. More training lowers the loss mainly by improving the same copy-friendly fit: across 115 arms the loss tracks the kind 3 and kind 2 losses (Spearman 0.90 and 0.83) and has no rank relation with the composite (-0.07). At 160,000 windows the loss fell in all three models, but every composite change stayed inside seed noise, and the conv autoencoder's correction fell as its copy gain doubled. A second, smaller cause is that the held-out loss tests new genome positions of trained tracks, while the imputation term tests cell-assay pairs never trained, in log1p space and on cell-specific variance only. The direction of the copy explanation is supported by every read here, but its size is not measured directly, because the held-out loss has no per-kind split.

## What this means for selecting a recipe

1. The held-out loss is not a proxy for D1. A recipe should be selected on the three terms on one scoring tier, as D1 already requires. A held-out loss gain of 0.01 to 0.05 says nothing about whether the composite or the correction term will rise.
2. A held-out loss gain is a warning sign for correction when the model's copy gain rises with it. The conv autoencoder at 160,000 windows and with 3 levels lowered the loss and lost correction beyond seed noise. A recipe search should read `same_view_copy_gain_relative` beside any loss gain.
3. If a cheap training-time proxy is wanted, the kind 1 part is the only part whose task resembles a composite term. The held-out loss would need a per-kind split to give it. That is a code change, and I did not make it.
4. The cause sits in the training task: kind 2 and kind 3 pairs share reads, and the correction term does not. The correction review proposed drawing training cross-view pairs from disjoint halves (`correction_metric_review/REVIEW.md`, section 3). That is a recipe arm, and it would test the copy explanation directly.

### Not checked

- The held-out loss split by kind; it is not logged, and I ran no new compute.
- The share of loss-mask bins of each kind in the held-out batches.
- That one seed draws byte-identical training windows across architectures.
- Copy gain for conv + attention at 160,000 windows.
- Confidence intervals for the Spearman correlations.
- The `eic_v1_2` tables. Their base held-out losses (conv autoencoder 0.9644, conv + attention 0.9551, U-Net 0.9098; ranges at most 0.0025) give the same order, but I did not repeat the analysis there.
