# Why the U-Net loses correction

Analysis 1 of decision D19 in the miniCANDI recipe run (task t370). No new compute was run. Every number below was read on 2026-11-21 from a score file, a vault table or the code. Each number is a mean over seeds 0, 1 and 2, with the seed range (maximum minus minimum over the 3 seeds) in parentheses, unless the text says otherwise.

Terms used here:

- The correction term is `correction_v3_raw` of `tools/onion_eval/select_rule.py`, i.e. the share of the gap between depth-only scaling and the replicate floor that the model closes on the 18 crop pairs (one experiment at two read lengths).
- The correction-only term is `correction_only_covariate_cost_closed` of `tools/onion_eval/score.py`, i.e. 1 - (m - s) / (d - f) per pair type, where m is the model's cross-view CRPS, s its same-view CRPS on the target, d the CRPS of depth-only scaling and f the half-read floor. It is 1 for a model that ignores its input, so read it with the correction term.
- CRPS is the continuous ranked probability score, i.e. the error of a full predictive distribution.
- The copy gain is `copy_leak.same_view_copy_gain_relative` of `tools/onion_eval/guardrail.py`, i.e. (CRPS on hidden bins - CRPS on visible bins) / CRPS on hidden bins, on same-view reconstruction (maximum over 3 mask seeds). A model that does not copy its visible input scores both bin sets alike and has a copy gain near 0.
- The level skips are the U-Net encoder features that join the decoder at each resolution level. The head skip is the per-track input features `h_track` that join each track's output head at full resolution (`tools/minicandi/rungs/unet.py`).
- The tables `eic_v1_2` are the current scoring tables. The tables `eic_v1` are the older tables of the onion close-out. A comparison is made only within one set of tables.

## Question

On the tables `eic_v1_2`, 256 regions, the U-Net has the highest imputation of the three base models but the lowest correction. What part of the U-Net causes the correction loss, and through what mechanism?

## Evidence

### 1. The correction deficit is real and larger than seed noise

Source: `cruxvault/results/eic_v1_2_base_p256/read_20261121T0520Z/arm_summary_eic_v1_2_p256_sv.tsv` in the onion worktree (`.claude/worktrees/onion`), runs `runs_fir/eic_v1_2_rung{2,3,4}_base_heldout/seed{0,1,2}/p256_sv` on Fir.

| model | imputation | correction | correction-only | calibration |
|---|---|---|---|---|
| conv autoencoder | 0.3704 (0.0214) | 0.6636 (0.0312) | 0.6601 (0.0284) | 0.5986 (0.1864) |
| conv + attention | 0.3860 (0.0298) | 0.6750 (0.0193) | 0.6654 (0.0815) | 0.6272 (0.1379) |
| U-Net | 0.4452 (0.0380) | 0.5352 (0.0624) | 0.3194 (0.0909) | 0.6460 (0.1577) |

- The U-Net correction is 0.6636 - 0.5352 = 0.128 below the conv autoencoder and 0.6750 - 0.5352 = 0.140 below conv + attention. Twice the largest seed range of each pair is 2 x 0.0624 = 0.125. So both gaps exceed twice the seed range.
- The U-Net correction-only term is 0.6601 - 0.3194 = 0.341 below the conv autoencoder. Twice the largest seed range is 2 x 0.0909 = 0.182. So this gap also exceeds twice the seed range.

### 2. The U-Net copies its visible input; the conv models do not

Source: `copy_leak.same_view_copy_gain_relative` in `runs_fir/<run>/seed{0,1,2}/p256_sv/scores.json` on Fir (`/project/def-maxwl/mforooz/ONION/runs_fir`). Per-seed values, then mean (range).

| tables | run | seeds 0, 1, 2 | mean (range) |
|---|---|---|---|
| `eic_v1_2` | conv autoencoder base | 0.0487, 0.0399, 0.0456 | 0.0447 (0.0088) |
| `eic_v1_2` | conv + attention base | -0.0051, 0.0525, 0.0477 | 0.0317 (0.0576) |
| `eic_v1_2` | U-Net base | 0.6729, 0.6604, 0.6776 | 0.6703 (0.0172) |
| `eic_v1` | conv autoencoder base (`rung2_base_heldout`) | 0.0425, 0.0390, 0.0414 | 0.0410 (0.0035) |
| `eic_v1` | conv + attention base (`rung3_base_heldout`) | 0.0528, 0.0549, 0.0548 | 0.0542 (0.0021) |
| `eic_v1` | U-Net base (`rung4_base_heldout`) | 0.6339, 0.6541, 0.6519 | 0.6466 (0.0202) |
| `eic_v1` | U-Net, skips carry DNA only (`final_rung4_mech_skip_dna`) | 0.0161, 0.0137, 0.0126 | 0.0141 (0.0035) |

On visible bins, the U-Net's CRPS is about two thirds lower than on hidden bins. The conv models show a gain of 0.03 to 0.05. The gap between the U-Net and either conv model (about 0.6) is more than 10 times every seed range in the table.

### 3. Removing the count skips removes the copy and restores correction

Source for correction and correction-only terms: `cruxvault/results/region_sampling_checks/check_v3.tsv`, condition `consistency`, column `value_all_regions` (256-region recomputation, tables `eic_v1`, `p256_sv`). Source for imputation: onion worktree `cruxvault/results/final_round/FINAL_ROUND_EQUAL.md` (imputation is scored on all 256 regions in the screen). Finding f87 (`cruxvault/f87_both_effects_exceed_the_seed_ranges_0_02.md`) records the same result.

| run (tables `eic_v1`) | correction, seeds 0, 1, 2 | correction mean (range) | correction-only mean (range) | imputation |
|---|---|---|---|---|
| U-Net base | 0.5401, 0.4742, 0.4968 | 0.5037 (0.0659) | 0.2822 (0.1289) | 0.432 (0.023) |
| U-Net, skips carry DNA only | 0.6882, 0.7442, 0.7127 | 0.7150 (0.0560) | 0.6839 (0.0334) | 0.380 (0.022) |
| conv autoencoder base | 0.7212, 0.7141, 0.7058 | 0.7137 (0.0154) | 0.6505 (0.0020) | 0.382 (0.028) |

- DNA-only skips raise correction by 0.7150 - 0.5037 = +0.211 (twice the larger range: 0.132). The correction-only term rises by +0.402 (twice the larger range: 0.258). Both changes exceed twice the seed range.
- DNA-only skips lower imputation by 0.432 - 0.380 = 0.052 (twice the larger range: 0.046). This change also exceeds twice the seed range.
- With DNA-only skips, the U-Net matches the conv autoencoder on all three columns, within the seed ranges.
- The opposite arm, skips that carry counts only (`final_rung4_mech_skip_counts`), equals the base on the 16-region screen: imputation 0.432 (0.006), crop-pair correction 0.518 (0.037), correction-only 0.304, against the base 0.432 (0.023), 0.507 (0.071), 0.318 (`FINAL_ROUND_EQUAL.md`). So the DNA level skips play no part. This run has no 256-region scores (its `p256_sv` directories on Fir are empty).

### 4. What the code says the DNA-only arm removed

Source: `tools/minicandi/rungs/unet.py` on this branch.

- `skip_source = dna` sets `skip_counts = False` (line 197). This removes two paths at once. First, it removes the count level skips `c_skips[i]`, which include the full-resolution level-0 features of all tracks fused by one 1x1 convolution (`CountTower.forward`, lines 145 to 152; `fused`, line 312). Second, it removes the head skip, because `head_skip` exists only when `skip_counts` is true (lines 266 to 267 and 377 to 378).
- The level skips are modulated by a FiLM `steer[i]` whose input is the C' embedding averaged over the window's target tracks (lines 352 to 371). C, the input covariates, does not reach them.
- The head skip concatenates the masked per-track input features `h_track` (already modulated by C through `in_inj`, line 303) into each track's head. Only the later FiLM `out_inj` with C' acts on the result (line 385). No layer of either path compares C with C'.

Consequence: the existing runs cannot say which of the two count paths carries the copy. Both are full-resolution paths from the input counts to the output.

### 5. Why a copy path costs correction

- On a crop pair, the scored source is the pr1 half of the source view and the target is the pr2 half of the target view (`cruxvault/results/correction_metric_review/REVIEW.md` in the onion worktree, section 1). A model that reproduces its input bin for bin behaves like depth-only scaling, which closes 0 of the gap. So a strong copy pulls the correction term toward 0.
- Training rewards a copy. A cross-view training pair is the full source view against the full target view of the same reads (`tools/minicandi/dataset.py` sampler docstring, cited in `REVIEW.md`, suspect (a)). Two views of the same reads differ in a few percent of reads, so a near-identity map with a tight Negative Binomial scores well in training. An architecture with a cheap full-resolution identity path learns that map; the conv autoencoder, which must pass all input through its coarse bottleneck, learns it much less (copy gain 0.04 against 0.65).
- Arms that break the direct copy raise U-Net correction (settled design, `cruxvault/INSTRUCTIONS/peeling_the_onion/closeout/settled miniCANDI design.md`). Row 25: fixed input depth gives 0.675 (0.013), +0.168 (16 regions). Row 24: two-step routing through a fixed base depth gives 0.636 (0.069), +0.129 (16 regions). Row 22, the opposite direction: same-view targets only (more copy reward) take the U-Net raw correction below 0 on every seed (-30.1, -0.90, -10.6; the row does not name the region set).
- On the 16-region screening set (tables `eic_v1`), scrambling the covariates lowers the U-Net copy gain (0.652 to 0.514) and raises its term on the alignment-level pairs (0.267 to 0.372) (`REVIEW.md`, suspect (a)). Within U-Net arms, the copy gain does not rank the arms (Spearman -0.04, same source). So the copy gain is a strong marker across architectures, and a weak one within the U-Net.

## Answer

The U-Net loses correction because its count skip paths give the output a full-resolution copy of the source view's input counts, and the training pairs (two views of the same reads) reward that copy. As a result, the U-Net reproduces the source view's processing signature instead of mapping it to the target covariates. The evidence is as follows. (1) The U-Net copy gain is 0.6703 (0.0172) against 0.03 to 0.05 for the conv models on `eic_v1_2`. (2) Skips that carry DNA only remove the copy (copy gain 0.0141 (0.0035)) and raise correction from 0.5037 (0.0659) to 0.7150 (0.0560) on `eic_v1`, 256 regions, at a cost of 0.052 imputation, which exceeds twice the seed range. The existing runs cannot separate the two count paths, the fused level skips and the per-track head skip, because `skip_source = dna` removes both. Neither path is conditioned on C and C' together, so neither path can learn when to stop copying. The cause is supported at moderate confidence: it rests on one designed ablation on the `eic_v1` tables (3 seeds, 18 crop pairs), and the base pattern was reproduced on `eic_v1_2`.

## What the screen arms of the skip family will show

These predictions were written on 2026-11-21, before any arm of the `skip` family (chunk C4) ran. Each sign is the change against the U-Net base on the screen (`p16_sv`, tables `eic_v1_2`, seeds 0 to 2). "0" means within the seed range of the base.

| arm | keys | correction | imputation | copy gain | reason |
|---|---|---|---|---|---|
| `headskip_off` | `head_skip: off` | + | - | - | It removes the per-track identity path. The fused level-0 skip still carries counts, so I expect a smaller rise than the DNA-only arm's +0.21. |
| `headskip_film` | `head_skip: film` | + (may be 0) | 0 | - (small) | A FiLM from C and C' per track lets the head gate the copy when C differs from C'. Training still rewards the copy, so the effect may stay within seed noise. |
| `skipfilm_ccp` | `skip_film: c_cprime` | 0 | 0 | 0 | The steer vector is pooled over the window's tracks, so it cannot gate a per-track copy. |
| `skipfilm_ccp_headskip_film` | both keys | + (may be 0) | 0 | - (small) | I expect it to equal `headskip_film` within seed noise, because the pooled level-skip FiLM adds nothing per track. |

If `headskip_off` raises correction by less than twice the base seed range, the copy runs mainly through the fused level skips, and the next arm to try is a level-0 skip that is removed or gated by C and C'.
