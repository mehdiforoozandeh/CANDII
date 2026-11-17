---
title: "The miniCANDI training objective (peeling the onion)"
parent: "[[closeout O1 training objective document]]"
created: 2026-11-17
status: working document; screening tier (16 regions); OPEN items listed in section 9
---

# The miniCANDI training objective

This document states what miniCANDI is trained to do. It serves success criterion 2 of the close-out ("have a training objective well defined"). A science undergraduate can follow it from top to bottom.

**Authority.** Every formula, weight and default comes from the code on branch `implementation/t152-peeling-the-onion` at 01248f4, with file and line. The code is the authority. Where a doc disagrees with the code, the code is right.

**Evidence tier.** Every ablation number comes from the 16-region screening set (scorer sub-directory `p16_sv`), 3 seeds, 80,000 windows. The source is `cruxvault/results/ablations_closeout/ABLATIONS_READ.md` and `arm_summary_p16_sv_2026-11-17.tsv` beside it (both in the foreman worktree). The 16-region check of the correction terms failed its rule (`cruxvault/results/region_sampling_checks/PAIR_REGIONS_V3.md`: "16-region screening scores settled: no"). So every conclusion that rests on a correction term (crop-pair correction or the correction-only term) is screening tier. A 256-region rescoring of the decisive arms is still owed.

**Notation.** A number is written "mean (seed range)", where the seed range is max minus min over the seeds. "Beyond" means that the difference of means is larger than the larger of the two seed ranges. "2x" means that it is larger than twice that range. The selection rule D13 (`long-horizon decisions.md`:90-100) adopts an arm only when its composite is higher by more than 2x the seed spread and no goal drops by more than 2x its own spread; otherwise the simpler arm stays.

## 1. The task in one paragraph

One training example is a window of 1,024 bins of 25 bp (25.6 kb) from one cell type. Each of the 36 tracks (35 assays plus the ChIP control) is either given as input with its covariate vector C, hidden, or absent. The model receives the input counts, the DNA sequence and C, plus the output covariates C' for each target track, i.e. the description of the experiment it must predict. For every target track and bin it predicts a Negative Binomial (NB) distribution, i.e. a count distribution with a mean mu and a size n (variance mu + mu^2/n). The loss is the NB negative log-likelihood (NLL) of the true target counts. The three target kinds train the three goals: imputation (a hidden track), covariate artifact correction (another view of a given track), and denoising (hidden bins of the same view).

## 2. Inputs read by the loss

### 2.1 Target kinds

`y_kind` per (window, track) (`tools/minicandi/contract.py`:98; assigned in `tools/minicandi/dataset.py`:1043-1045):

| kind | name | input of that track | target | C' of the target |
|---|---|---|---|---|
| 0 | none | any | none | 0 |
| 1 | imputation | absent: no counts, C row is 0 | a view of the track | the target view's own C' |
| 2 | correction | one view (source) | a different view of the same track | the target view's C' |
| 3 | same-view reconstruction | one view, with some bins hidden | the same view, same file | the same C' as the input |

A "view" is one measurement of a (cell, assay) track: the base experiment or a counterfactual version of it (thinned depth, single-end, read crop, no duplicate removal, MAPQ filter off). Section 4 lists them.

### 2.2 Which bins count: the loss mask

`contract.loss_mask` (`contract.py`:299-305):

- Kind 1 and kind 2 targets count on **every** bin of the window.
- Kind 3 targets count **only on the hidden bins** (`x_bin_mask` true). On those bins the input holds 0 (`dataset.py`:1053-1056), so the model cannot copy the target. This is the copy-leak rule.

A kind-2 target counts on every bin, also where its source view is visible. The source and the target are different files, so a copy is not the answer.

### 2.3 The control track

Track 35 is the ChIP control (`contract.py`:59-64). Defaults: `control_input` true, `predict_control` true (`dataset.py`:241). So the control is an input that can be masked and a target like any assay. `control_input = false` removes it from the inputs; a control target then becomes kind 1 (`dataset.py`:999-1001). `predict_control = false` removes it from the targets (`dataset.py`:1002-1004).

The control output head (`tools/minicandi/control_head.py`): default `control_head = channel` (`control_head.py`:44, 93-97), i.e. the control is one output channel of the same NB head as every assay. The alternative `lambda_smooth` replaces the control mean by a log-space average over windows of `lambda_window_bins` = 40 bins (1 kb), linearly interpolated, with one learned NB size (`control_head.py`:45, 48-88). It requires `control_input = false` (`control_head.py`:98-100). `registry.build_model` applies it before the two-step wrapper (`tools/minicandi/registry.py`:160-165).

## 3. The loss

### 3.1 Per-bin NB NLL

For a count y, mean mu and size n (`tools/t118/ladder/model.py`:78-92, called through `contract.nb_nll`, `contract.py`:279-296):

    NLL(y | mu, n) = -[ lgamma(y + n) - lgamma(n) - lgamma(y + 1)
                        + n log(n / (n + mu)) + y log(mu / (n + mu)) ]

It is computed in float64. mu is clamped to [1e-4, 1e6] and n to [1e-3, 1e4] (`ladder/model.py`:50-51).

### 3.2 Total loss

`losses.total_loss` (`tools/minicandi/losses.py`:285-372). Let M be the loss mask (section 2.2) and |M| its number of (window, bin, track) elements. Let w_kind be the weight of each element's target kind.

    L_main  = (1 / |M|) * sum over M of  w_kind * NLL(y, mu, n)
    L_total = L_main
              + loss_shape_weight * L_shape                 (0 by default: off)
              + sum over k in loss_multiscale of L_multi,k  (empty by default: off)
              + w_base * L_base                             (only when two_step is true)

The denominator is the count of masked elements, not the sum of weights (`losses.py`:320). So a weight above 1 raises the loss scale.

| term | formula | computed on | default | code |
|---|---|---|---|---|
| main term | mean NB NLL over M, kind-weighted | M: every bin of kinds 1 and 2, hidden bins of kind 3, all 36 tracks | on | `losses.py`:300-320 |
| weight `impute` | multiplies kind-1 elements | kind 1 | 1 | `losses.py`:97, 133-136 |
| weight `correct` | multiplies kind-2 elements | kind 2 | 1 | same |
| weight `same` | multiplies kind-3 elements | kind 3, hidden bins | 1 | same |
| base leg (two-step) | mean NB NLL of the real base view `y_base` under leg 1's (mu_base, n_base) | M restricted to tracks that have a base view (`c_base` log2 depth not 0) | off (`two_step` false) | `losses.py`:358-371 |
| weight `base` | multiplies the base leg | as above | 1 | `losses.py`:371 |
| shape term | multinomial NLL of y under mu / sum(mu), per (window, track), over M, divided by |M| | M | weight 0 (off) | `losses.py`:263-279, 349-353 |
| multiscale term | NB NLL of k-bin sums under the moment-matched pooled NB | blocks of k bins that M covers whole | empty list (off) | `losses.py`:227-260, 354-357 |

With every default, `total_loss` returns exactly `contract.nb_nll(n, mu, y, loss_mask)`, bit for bit (`losses.py`:300-301).

### 3.3 Switches that change the per-bin loss

All are off by default; the defaults leave the loss bit-identical (`losses.py`:32-34).

| key | values | default | what it does | code |
|---|---|---|---|---|
| `loss` | `nb_nll`, `poisson_mean_detached_size`, `nb_crps` | `nb_nll` | Poisson NLL for the mean plus NB NLL on the detached mean for the size; or the NB continuous ranked probability score (CRPS) | `losses.py`:95-96, 142-206 |
| `loss_bin_equal` | bool | false | NB NLL trains only n; the mean gets 0.5 (y - mu)^2 / vbar, vbar = mean predicted variance of the (window, track) | `losses.py`:212-224 |
| `loss_assay_weight` | `none`, `equal`, `learned` | `none` | the main term becomes a mean over assays of a per-assay loss | `losses.py`:319-337 |
| `w_window` (batch key) | per-window weight | absent | only with `train_regions`; a training-region sampler, left for follow-up work | `losses.py`:65-76 |

### 3.4 The mean and its depth offset

The NB head (`tools/minicandi/components.py`:491-524) emits eta and log n per (bin, track). The default count transform is `offset` (`count_transform`, base configs):

    s     = 2^(log2_depth - 24)                 size factor against 2^24 reads   (components.py:63, 426-427)
    input x = log2((y + 1) / s_in)              pseudocount offset_eps = 1        (components.py:64, 436-437)
    mean  mu = s_out * 2^eta                    s_out from the TARGET's C' depth  (components.py:472-476, 523)
    size  n  = exp(log 2 + eta_n)               starts at n0 = 2                  (components.py:503, 520-522)

The head's linear layer and track bias start at 0 (`components.py`:509-512), so at the start mu = s_out and n = 2 at every bin. Depth enters twice: as the offset s_out and as column 0 of C and C' (`contract.py`:88-96). An imputed target is decoded at its own depth (`components.py`:495-497).

## 4. Masking and view sampling (training mode)

One training window (`dataset.py` `plan`, 862-897; `materialize`, 1081-1098):

1. **Cell.** One training cell, uniform (`dataset.py`:867).
2. **Locus.** A chromosome in proportion to its length, then a uniform start; chrY and chrM excluded; at least half the bins must be valid in the genome mask, 8 tries (`dataset.py`:238-246, 839-860). With `heldout_loss` set (the base configs set it: 16 batches, every 20 log lines), starts that overlap the fixed held-out windows are rejected (`train.py`:83-92, 454-457).
3. **Whole-track mask (imputation).** Default `mask_scheme = uniform`: draw p from U(0.2, 0.8) (`mask_prob_range`), then mask each of the cell's training tracks with probability p (`dataset.py`:899-906). If every track is masked, the first track in a random order is unmasked; a cell with one track gets no mask (`dataset.py`:877-880). A masked track becomes kind 1.
4. **View pair (correction or reconstruction).** For each unmasked track, the views are shuffled. With probability `same_view_prob` = 0.5 the first pair is (v, v), i.e. kind 3; otherwise (v, another view), i.e. kind 2 (`dataset.py`:920-935). A track with one view always gets kind 3.
5. **Withheld combinations.** A pair is skipped when it completes a (source view, target view) combination listed in `withheld.tsv`, on the same track or across the window's tracks (`dataset.py`:330-341, 774-785). The sampler then tries every other ordered pair, then imputation; a track with no acceptable candidate is dropped. In `eic_v1_2` (Fir `/project/def-maxwl/mforooz/ONION/tables/eic_v1_2/withheld.tsv`, read 2026-11-17) the withheld pairs are no dedup <-> MAPQ 0 (histone and DNase classes), crop 36 <-> crop 50 and single-end <-> crop 50 (histone and ATAC classes), in both directions. These are the never-trained pairs that the correction metric scores.
6. **Bin mask (denoising).** Each kind-3 track gets a random bin mask: each bin hidden with probability `bin_mask_frac` = 0.25, at least one bin hidden (`dataset.py`:1084-1087, 1100-1106). Default `bin_mask_shape = random`; `span` hides spans of 32 bins (`dataset.py`:1108-1119). Kind-2 tracks get no bin mask.
7. **Hidden-position representation.** Default `mask_mode = token` (base configs): hidden bins get a learned hidden token and missing tracks a learned missing token; `drop` puts exact zeros (`components.py`:316-349).

**Views in training** (`eic_v1_2/measurements.tsv`, rows with `in_eic = train`, counted on Fir 2026-11-17): base 678; depth 0.50, 0.25, 0.125 (312 each); no dedup 312; MAPQ 0 308; single-end 38; crop 36 29; crop 50 27. Default `train_views` = absent, i.e. every view kind (`dataset.py`:147-162, 261). The training sampler reads the `counts` file only, never the pseudoreplicate halves (`dataset.py`:157-160).

**Covariates.** C has K_META = 11 columns: log2 depth, run type, read length, dedup, MAPQ threshold, pipeline, pipeline version, lab, platform, antibody, donor (`contract.py`:88-95). C' has the same 11 columns and never the 7 read-derived QC columns (`contract.py`:13, 91-92). Default `qc_on_input` false. The table version is `eic_v1_2` (lab, antibody, donor, platform filled); every ablation score below is on `eic_v1`.

**Fixed input depth** (`input_depth`, default `native`): `fixed` thins every input deeper than 2^23 reads to 2^23 and writes the thinned depth in C; targets and C' stay native (`dataset.py`:74-79, 252-253, 1027-1036).

## 5. Two-step correction path (off by default)

Default `two_step` false (`dataset.py`:241; base configs). When true (`tools/minicandi/two_step.py`:44-78; `registry.py`:163-165):

- **Leg 1.** The model reads the batch's inputs and decodes every target at C_base, the covariates of the track's base view (uniform layer first, then pooled unit; `dataset.py`:939-947). It predicts the base view (mu_base, n_base). A track with no base view uses its own C'.
- **Leg 2.** The same weights read mu_base (not detached) as input counts on the target tracks, with C = C_base, no bin mask, and decode at the batch's C'. Its output is the prediction that the main term scores.
- **Loss.** The main term on leg 2, plus `w_base` (1) times the NB NLL of the real base counts under leg 1, on the same mask bins (`losses.py`:358-371).
- **Base depth.** Default `base_depth = native`. Under `fixed`, the base counts are binomially thinned to one reference depth per assay class (`dataset.py`:949-968). The ablation used `base_ref_log2_depth` = DNase 24.5, ATAC 25.0, broad histone 24.0, narrow histone 23.5, control 23.5 (`configs/ablations/two_step/rung4/two_step_fixed.json`). Without that key the constant is 23.0 for every class (`dataset.py`:252-253).
- **Cost.** Two forward passes: 4.22e9 against 2.11e9 FLOPs per window on the conv autoencoder (ABLATIONS_READ 2.4). Not combinable with `qc_on_input` (`two_step.py`:49-50).

## 6. Optimization

`tools/minicandi/train.py`; base configs `tools/minicandi/configs/rung{2,3,4}/base_heldout.json`.

| setting | value | code |
|---|---|---|
| optimizer | AdamW | `train.py`:461-462 |
| learning rate | 1e-3 (`lr`) | `train.py`:121; base configs |
| weight decay | 0.01 (`weight_decay`) | `train.py`:124, 430 |
| gradient clipping | global norm 1.0 (`grad_clip`) | `train.py`:125, 609 |
| warm-up | linear over min(1000, 5% of steps) = 500 steps at the base budget | `train.py`:126, 164-165 |
| schedule | cosine decay to 10% of lr at the last step | `train.py`:301-306 |
| batch | 8 windows (`batch_size`) | `train.py`:121 |
| budget | 80,000 windows = 10,000 steps (`windows_seen`) | base configs; Fir `runs_fir/rung{2,3,4}_base_heldout/seed0/run.json` (`steps` 10000) |
| context | 1,024 bins (`context_bins`) | `train.py`:121; `dataset.py`:238 |
| parameter budget | 1.0e6 (`param_budget`); built 1,042,514 / 1,038,018 / 993,599 | `train.py`:121; same `run.json` files |
| loss regularizer | none in the loss; weight decay only | `losses.py` |

160,000 windows lowered the held-out loss but not the composite (finding f88, cited in `ablation_specs/9_losses.md` L4). The held-out loss is the arm's own training loss, so it is not comparable across loss arms (`train.py`:541-552).

**What `run.json` records.** `run.json` `config` is the train-loop defaults merged with the config file (`train.py`:279-287, 555). The dataset defaults (`mask_prob_range`, `same_view_prob`, `bin_mask_frac`, `mask_scheme`) and the loss defaults (`loss`, `loss_weights`) are not in the base config files, so `run.json` does not record them; they come from the code of the kit that ran. The three base runs record `git_sha` "unknown".

## 7. Evidence per term (ablations, 16 regions, `eic_v1`)

Reference composite, mean (range): conv autoencoder 0.528 (0.045), conv + attention 0.525 (0.048), U-Net 0.495 (0.010). "Correction" is the gap closed on 18 crop pairs with the oracle reference. "Correction-only" is the share of the covariate cost closed on same-view pairs; it is not in the composite.

| term or choice | default | evidence | reading | rests on a correction term |
|---|---|---|---|---|
| per-bin loss: NB NLL | `nb_nll` | Poisson mean with detached size: all four metrics within on all three models. NB CRPS: conv + attention correction 0.726 (0.015) against 0.673 (0.031), +0.053 beyond, not 2x; calibration within everywhere (calibration ranges 0.108 to 0.178 cannot resolve smaller effects). | Keep NB NLL (the PI's core loss; D13 simpler arm). | CRPS gain: yes |
| weight `impute` | 1 | Weight 4: conv + attention imputation 0.426 (0.017) against 0.387 (0.012), +0.039, 2x; composite 0.559 (0.013) against 0.525 (0.048), +0.034, within. Conv autoencoder correction 0.680 (0.031) against 0.718 (0.012), -0.038 beyond. U-Net all within. | Keep 1. Gain on one model only, with a correction cost on another. OPEN (b). | yes (the conv autoencoder cost) |
| weights `correct`, `same` | 1, 1 | Not ablated. | Default, not tested. | |
| bin-equal weighting | off | Conv + attention imputation 0.357 (0.020), -0.030 beyond; U-Net composite 0.510 (0.009) against 0.495 (0.010), +0.015 beyond, not 2x. | Keep off (opposite signs). | no |
| shape, multiscale, per-assay weights | off | Not run. Multiscale was left out because coarse losses were worse at every scoring resolution (lead L6, `ablation_specs/9_losses.md`). | Default, not tested. | |
| kind 3 on hidden bins only | rule | Not ablated; required by the copy-leak guardrail (hard gate, D13). | Fixed by design. | |
| control as input | on | Off lowers imputation on all three models (-0.019 to -0.026), beyond on conv + attention only: 0.360 (0.021) against 0.387 (0.012). | Keep on (weak). | no |
| control as target | on | Off: conv + attention imputation -0.021 beyond; U-Net composite 0.509 (0.004) against 0.495 (0.010), +0.014 beyond, not 2x. Opposite signs. PI ruling 14 keeps the control output. | Keep on. | no |
| control head form | `channel` | `lambda_smooth` not scored in the close-out. | Default, not tested. | |
| whole-track mask | uniform, p in U(0.2, 0.8) | Mask schemes `dirichlet` and `panel` not scored in the close-out. | Default, not tested. | |
| hidden-position form | `token` | `drop` within on all four metrics on all three models. | D13 simpler arm: `drop`. The base configs still say `token`; M1 decides. | no |
| bin mask | random, 0.25 | `span` and other fractions not scored. | Default, not tested. | |
| counterfactual views in training | all views | Base view only: composite within on all three (+0.010 to +0.022), but correction-only falls 2x on conv models (0.402 against 0.690; 0.390 against 0.659) and beyond on the U-Net (0.069 against 0.318). Base + alignment-level views: conv autoencoder composite 0.476 against 0.528, correction 0.615, -0.102 (2x). | Keep all views. The case rests on the correction-only term. | yes |
| `same_view_prob` | 0.5 | Same-view pairs only (no kind 2): conv imputation +0.031 and +0.032 beyond; U-Net composite 0.362 (0.058) against 0.495 (0.010), -0.133, 2x, with crop-pair correction far below 0 (seeds -30.1, -0.90, -10.6). | Keep 0.5; other values not tested. | yes |
| two-step routing | off | Native: correction down beyond on both conv models (conv autoencoder 0.602 (0.085) against 0.718 (0.012), -0.115; conv + attention 0.614 against 0.673, -0.059); U-Net composite 0.412 (0.059), -0.083 beyond. Fixed base depth on the U-Net: correction 0.636 (0.069), +0.129 beyond, not 2x; calibration 0.445 (0.119), -0.165 beyond; composite 0.495, +0.000. Conv models: correction-only up 2x under both (for example 0.806 against 0.690). | Keep off. If used, fixed base depth. Caveat: leg 2 reads all of a cell's tracks in training and one track at scoring. | yes |
| fixed input depth | native | U-Net: composite 0.518 (0.010) against 0.495 (0.010), +0.023, 2x; correction 0.675 (0.013), +0.168, 2x; imputation 0.404 (0.010), -0.028 beyond. It meets D13 on the U-Net only. Conv autoencoder has 2 seeds. | OPEN (b). Caveat (PI, 2026-11-17): requiring fixed input depth can be unrealistic in real-world deployment and inference. | yes |
| average baseline input | off | U-Net imputation 0.466 (0.011) against 0.432 (0.023), +0.034 beyond; conv models move no composite term. | OPEN (b). | no |
| count transform and depth offset | `offset`, pseudocount 1 | No transform beats the offset on 2+ models. Pearson, asinh and soft clip raise the U-Net composite beyond (+0.031, +0.015, +0.016). Local ratio and quantile are worse beyond on all three. Anscombe is void (code defect). Earlier: hiding depth on both sides cut the U-Net composite by 0.184 (`ladder_audit/COVERAGE.md` row 27, older composite). | Keep `offset`. U-Net transforms join mix-and-match (ABLATIONS_READ section 4). | partly (Pearson U-Net correction +0.078) |
| covariates in C and C' | 11 metadata columns | Scrambled covariates in training: correction-only falls beyond on all three. Foreign C' at scoring (depth kept): crop-pair correction falls beyond on all three. `eic_v1_2` against `eic_v1` base retrains: within on all but conv autoencoder correction (0.681 against 0.718). | Keep all fillable fields (PI, 2026-11-17). OPEN (c). | yes |
| optimizer recipe | section 6 | Recipe tuning and 160,000 windows did not raise the composite (f88). | Keep. | no |

## 8. Evaluation link

| goal | headline metric (D13, composite version 3) | weight in the composite |
|---|---|---|
| imputation | variance recovered: share of the specific variance above the noise floor | 0.5 |
| covariate artifact correction | gap closed on never-trained crop pairs, oracle reference (PI ruling 47) | 0.25 |
| calibrated uncertainty | 1 - PIT deviation of the arm / PIT deviation of the marginal-NB baseline | 0.25 |
| (reported beside) | correction-only term: covariate cost closed on same-view pairs | not in the composite |

The terms are clipped before the weighted sum. From 2026-11-17 the correction terms are scored on all 256 regions.

## 9. OPEN choices

(a) waits for the 256-region rescoring; (b) waits for the mix-and-match round; (c) waits for the minimum covariate set (item A12).

1. **(a)** Every conclusion in section 7 marked "rests on a correction term": the NB CRPS correction gain, the cost of imputation weight 4 on the conv autoencoder, all views and `same_view_prob` 0.5, two-step off and its fixed-depth form, fixed input depth, Pearson on the U-Net, and the covariate reads.
2. **(b)** Fixed input depth on the U-Net. Caveat (PI): requiring fixed input depth can be unrealistic in real-world deployment and inference.
3. **(b)** Average baseline input (U-Net).
4. **(b)** Imputation weight 4 (`loss_weights.impute` = 4; conv + attention).
5. **(b)** Pearson, asinh or soft clip count transform on the U-Net (they change the output link).
6. **(c)** Which of the 11 columns stay in C and C'.
7. **M1 decides** (no further test planned): `mask_mode` drop against the base configs' token.

## 10. Default, not tested

Weights `correct` and `same` (1); `mask_prob_range` (0.2, 0.8) and the uniform mask scheme; `bin_mask_frac` 0.25 and the random bin shape; `same_view_prob` values other than 0.5 and 1.0; the `channel` control head; the shape, multiscale and per-assay loss terms; the NB size start n0 = 2; the warm-up and schedule shape. Each is the value the code shipped with; no ablation scored an alternative in the close-out.
