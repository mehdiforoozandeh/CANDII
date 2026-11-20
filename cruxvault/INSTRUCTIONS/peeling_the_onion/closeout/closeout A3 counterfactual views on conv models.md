---
title: "A3: counterfactual views on the two conv models"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# A3: counterfactual views on the two conv models

## 1. What the PI asked

- PI directive, plan Log 2026-11-16T01:10Z: "the final report must show whether and where the counterfactual views (EIC processed in different ways) help."
- `peeling the onion.md`:93: "we can also use the same data views as augmentations of the data and use it for training."
- `long-horizon decisions.md`:56 (D7): "The designed views are training data, i.e. augmentations. A held-out slice of views forms the controlled covariate-correction test set: never-trained covariate pairs, on held-out biosamples."

Success criteria served: 1 (counterfactual data is used) and 4 (design settled). Audit row: D28.

## 2. Question

Do the counterfactual processing views (depth thinning; read-level: single-end, read crop; alignment-level: no duplicate removal, no MAPQ filter) raise imputation, correction or calibration on the conv autoencoder and conv + attention, as they were tested on the U-Net? Where (which view type, which evaluation pair type) do they help or hurt?

## 3. State today

- U-Net only, 3 seeds, 16 regions (`results/views_ablation/VIEWS_ABLATION.md`; question q44). Composite version 3, Fir (`correction_metric_review/REVIEW.md` addendum): all views 0.495 (0.010); base view only 0.517 (0.039); base + depth 0.509 (0.037); base + read-level 0.498 (0.045); base + alignment-level 0.505 (0.023). Under version 3 the view arms are within the seed range.
- Version 2 (all 102 pairs): base + alignment-level dropped gap closed to 0.072 (0.068) against 0.298, and same-view pairs only gave gap closed below 0 in 3 seeds (`VIEWS_ABLATION.md`).
- Arm configs exist for the U-Net only: `tools/minicandi/configs/views/rung4/views_{base_only,base_depth,base_fastq,base_alignment,all_same_view_only}.json` (key `train_views`, `same_view_prob`).
- The U-Net view arms are scored in `p16` only (checked on Fir 2026-11-16): no `p16_sv`, so no correction-only term.
- U-Net "where" (A3, from stored scores): `results/closeout_A3/WHERE_VIEWS.md`. Depth-thinning views help on the depth-thinning pairs (all 6 seeds with depth views above all 9 without); alignment-level views hurt on MAPQ + no-dedup pairs; nothing moves the crop pairs. The report for Max does not include the view result (audit row D28).

## 4. Prior leads

- `VIEWS_ABLATION.md`, per never-trained view pair (superseded half-read scale): no-MAPQ + no-dedup pairs gap closed 1.25 all views, 1.77 base only, 0.08 base + alignment-level. Prediction: alignment-level views teach a copy shortcut on pairs that share DNA fragments (question q45).
- Training rows by view kind (`in_eic = train`): base 678, depth 936, mapq 308, nodedup 312, se 38, crop 56 (Log 2026-11-16T02:15Z). Prediction: at equal windows, more views means fewer draws of the base view; single-end and crop views are rare.
- U-Net and conv models differ on correction (gap closed 0.507 against 0.718 and 0.673, `FINAL_ROUND_EQUAL.md`). Prediction: the view effect can differ on the conv models, which correct better.

## 5. Work

1. Write the 5 view arm configs on `final/rung2/reference_equal.json` (conv autoencoder) and `final/rung3/reference_equal.json` (conv + attention); a json diff shows only `train_views` and `same_view_prob`.
2. Submit on Fir CPU, 3 seeds, scored into `p16_sv` with the same-view pass: 5 arms x 2 architectures x 3 seeds = 30 runs; and score the 15 U-Net view seeds into `p16_sv` (score only). One script: `tools/minicandi/configs/ablations/views/submit_views.sh`; specification `run_state/ablation_specs/14_views_on_conv_models.md`.
3. Read with `select_rule.py arm-summary` (composite version 3) against `rung{2,3}_base_heldout`.
4. "Where": run `results/closeout_A3/where_views.py` (per knob, per assay, depth-thinning pairs with the oracle floor) on all three architectures with `--sub p16_sv`.
5. Write `results/<task>/VIEWS_THREE_MODELS.md`: one table per architecture and one per-knob table; add the result to the close-out report (R2).

## 6. How the result answers the question

"Beyond" = difference of seed means larger than the larger seed range (max minus min, 3 seeds).

| outcome | conclusion |
|---|---|
| all-views beyond base-only on correction on 2+ of 3 architectures | the views help correction; keep them as training data |
| base-only beyond all-views on 2+ architectures | the views hurt (likely alignment-level copy); drop that view type |
| within the range on all three | the views neither help nor hurt at this scale; by D13 the simpler arm (base only) stays; report that the counterfactual data then serves evaluation only |
| one view type helps on its own knob's pairs only | the view teaches its own covariate; keep that type and say so per knob |

## 7. Dependencies, cost, risks

- Depends on nothing.
- Cost: per-seed cost 32.5 CPU-h (conv autoencoder) and 22.4 CPU-h (conv + attention) (`ablation_specs/1_control_input.md` section 7): 5 x 3 x (32.5 + 22.4) = 824 CPU-h; plus the U-Net score-only pass 15 x 21.2 = 318 CPU-h; total 1,142 CPU-h, 255 jobs.
- Risk 1: the correction term rests on 18 crop pairs, 6 cells, 6 assays and no alignment-level pair (`REVIEW.md` section 3); it cannot show an alignment-level view effect. The per-knob table must carry that part.
- Risk 2: equal windows: more views means fewer draws per view (`VIEWS_ABLATION.md` caveat 2).

## 8. Done when

The 30 runs are scored, `VIEWS_THREE_MODELS.md` reports each view arm on three architectures with a per-knob table, each D13 outcome is logged, and the close-out report states whether and where the views help.
