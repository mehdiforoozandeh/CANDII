---
title: "Settled miniCANDI design (peeling the onion close-out, item M1)"
parent: "[[closeout M1 settled miniCANDI design]]"
created: 2026-11-17
status: from the 16-ablation evidence and the 256-region rescoring read of 20261119T0110Z (5 of 6 rules decided, scores as of 2026-11-19 01:10 UTC)
---

# Settled miniCANDI design

This document states the miniCANDI design that a new session scales from. It gives one row per design choice, the evidence for each row, the run that settles each provisional row, the tests that cover the code, the config files, and what the design does not know yet. It serves success criterion 4 of the close-out ("have the minicandi design settled and tested and ready for scaling in the next session").

**Authority.** Code and configs: branch `implementation/t152-peeling-the-onion` at e8c1955. Where this document and the code disagree, the code is right. All recommendations marked "my choice" are the agent's intent, not a PI ruling.

## 0. How to read the numbers

**Sources.** Unless a row names another file, every ablation number is from `cruxvault/results/ablations_closeout/ABLATIONS_READ.md` and `arm_summary_p16_sv_2026-11-17.tsv` beside it. That folder and the other `results/` folders below are git-ignored; they exist in the foreman worktree `/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/.claude/worktrees/onion/cruxvault/results/`. Every run: 3 seeds (0, 1, 2), 80,000 training windows, tables `eic_v1`, unless the row says otherwise.

**Format.** A number is "mean (seed range)". The seed range is max minus min over the seeds. "Beyond" means that the difference of means is larger than the larger of the two seed ranges. "2x" means that it is larger than twice that range (the bar of the selection rule D13, i.e. adopt an arm only when its composite rises by more than 2x the seed spread and no goal falls by more than 2x its own spread; otherwise the simpler arm stays).

**Which metric uses which regions.** The scorer sub-directory `p16_sv` scores imputation and calibration on all 256 evaluation regions, and only the two correction terms on the 16 pair regions (`results/region_sampling_checks/REGION_CHECKS.md`, section "Which terms use which regions", from `score.py`:1683, 1697, 2906, 2972). So:

| metric | regions in `p16_sv` | tier of an effect on it |
|---|---|---|
| imputation (variance recovered, `imputation_raw`) | 256 | measured on the full set |
| calibration (`uncertainty_raw`) | 256 | measured on the full set |
| correction (gap closed on the 18 crop pairs with the oracle reference, `correction_v3_raw`) | 16 | screening only |
| correction-only term (covariate cost closed on same-view pairs, `correction_only`) | 16 | screening only |
| composite version 3 (`composite_v3`, 0.5 / 0.25 / 0.25 of the clipped terms) | mixed | screening, because it holds the correction term |

The pre-written 16-versus-256 check failed its rule at 8, 16 and 32 pair regions (`results/region_sampling_checks/PAIR_REGIONS_V3.md`, first line "pair regions: 256 (none of 8, 16, 32 passes)"; at p = 16, condition 4 passed 7 of 8 and condition 6 passed 9 of 10, `check_v3.tsv`). So no effect on a correction term or on the composite is settled in this document. Only the 12 base runs (3 base recipes and the U-Net with DNA-only skips) have 256-region correction scores today.

**Confidence scale (mine).** Strong: 2x on 2 or more models, on a term measured on 256 regions. Moderate: beyond on 2 or more models, or within the range on all four metrics on all three models (a consistent null), or 2x on one model on a 256-region term. Weak: beyond on one model only, or the case rests on a 16-region correction term, or the value is a default that no run tested.

**Status.** "Settled": no planned run can change the value under the pre-written rules, or the PI fixed it. "Provisional": a named run can change it (section 2).

**The three models.** Conv autoencoder (`rung 2`, trunk width 60, 1,042,514 parameters), conv + attention (`rung 3`, width 104, 1,038,018), U-Net (`rung 4`, width 25, 993,599), all from `run.json` of `runs_fir/rung{2,3,4}_base_heldout/seed0` (ABLATIONS_READ, "References").

**References** (`p16_sv`, imputation / correction / calibration / composite / correction-only): conv autoencoder 0.382 (0.028) / 0.718 (0.012) / 0.631 (0.178) / 0.528 (0.045) / 0.690 (0.019); conv + attention 0.387 (0.012) / 0.673 (0.031) / 0.652 (0.157) / 0.525 (0.048) / 0.659 (0.043); U-Net 0.432 (0.023) / 0.507 (0.071) / 0.610 (0.108) / 0.495 (0.010) / 0.318 (0.127). The calibration ranges (0.108 to 0.178) are wide, so a calibration effect below about 0.11 to 0.18 cannot show as beyond.

## 1. The design table

| # | design choice | settled value | evidence (path; number with seed range) | confidence | status |
|---|---|---|---|---|---|
| 1 | Architecture family | Three models carried forward: conv autoencoder, conv + attention, U-Net. None wins on the composite. | 256 regions, `region_sampling_checks/check_v3.tsv` condition 3: D13 verdict TIE on all 4 comparisons; composite difference conv autoencoder minus U-Net -0.033 (2x bar 0.043), conv + attention minus U-Net -0.033 (2x bar 0.048). Trade-off on 256 regions (my means from `check_v3.tsv`, column `value_all_regions`): correction U-Net 0.504 (0.066) against conv autoencoder 0.714 (0.015) and conv + attention 0.685 (0.026), lower by 0.210 and 0.181, beyond, sign kept on 16 regions (condition 6 pass); correction-only U-Net 0.282 (0.129) against 0.651 (0.002) and 0.616 (0.046). Imputation (256-region term) U-Net 0.432 (0.023) against 0.382 (0.028) and 0.387 (0.012): +0.050 and +0.045, beyond, not 2x. Axial 0.498 (0.031) and Perceiver IO 0.335 (0.088) left out by ruling 57 (`results/final_round/FINAL_ROUND_EQUAL.md`). | moderate | settled (tie, with a trade-off: U-Net best on imputation, conv models best on correction); which model goes to scaling is a PI decision |
| 2 | Local context at all (pointwise MLP against the context models) | Context models (1,024 bins). | The pointwise MLP has no score under composite version 3 (`closeout A10`); no comparison exists. | weak | provisional (run 2.9) |
| 3 | Architecture internals | Base internals of each model, U-Net skips carry counts and DNA (`skip_source: both`). | Tuned recipes within the range of the base on the composite: conv autoencoder 3 levels 0.496 (0.028) against 0.528 (0.045); conv + attention attention at 128 positions 0.510 (0.044) against 0.525 (0.048); U-Net lr 3e-3 0.511 (0.028) against 0.495 (0.010) (`FINAL_ROUND_EQUAL.md`; finding f88). U-Net skips with DNA only, 256 regions: correction 0.715 (0.056) against 0.504 (0.066), +0.211, 2x; imputation 0.380 (0.022) against 0.432 (0.023), -0.052, 2x worse; composite +0.034, beyond, not 2x (`check_v3.tsv` conditions 3 and 6; `FINAL_ROUND_EQUAL.md`). D13 cannot adopt DNA-only skips because imputation falls by more than 2x. | moderate | settled |
| 4 | Parameter budget | 1.0e6 (`param_budget`); the registry refits the trunk width to the budget. | Built counts above; every ablation arm within 1% of the base except those listed in ABLATIONS_READ 5.3. | not an evidence question | settled for the prototype; scaling changes it (section 4) |
| 5 | Input encoding of counts | `count_transform: offset`: input x = log2((y + 1) / s_in), s_in = 2^(log2 depth - 24); pseudocount 1. | Local ratio and per-window quantile lower imputation 2x on all three: -0.113 / -0.112 / -0.089 and -0.095 / -0.082 / -0.054 (256-region term). Pseudocount 0.1 and 0.01 lower conv + attention imputation 2x (-0.025 and -0.032, bar 0.012). log1p, asinh, raw counts and soft clip: no imputation effect beyond on 2 or more models. No transform raises the composite beyond on 2 or more models. The Anscombe arm is void (zero gradient at the zero-initialized head, ABLATIONS_READ 5.1) and was dropped by the PI. | strong (against local ratio and quantile); moderate (offset against the rest) | settled |
| 6 | Pearson residual transform on the U-Net | Off. | U-Net composite 0.526 (0.023), +0.031 beyond; correction 0.585, +0.078 beyond (16 regions); imputation +0.018, within. Conv + attention imputation -0.020 beyond. Qualifies for the U-Net mix-and-match pool (`results/closeout_MIX/qualify_single_choices.tsv`). | weak | provisional (run 2.7) |
| 7 | Hidden-position representation (mask mode) | `mask_mode: token` (learned hidden and missing tokens). My choice. | `drop` within on all four metrics and the correction-only term on all three models (largest: U-Net correction +0.040, range 0.073). D13 would let the simpler `drop` stay; I keep `token` because every reference, every ablation and the mix-and-match reference were trained with it, and conv + attention loses its visibility flag under `drop`. Either value is supported. | moderate (null) | settled |
| 8 | Covariates on input (C) and output (C') | The 11 metadata columns in C and C' (log2 depth, run type, read length, dedup, MAPQ threshold, pipeline, pipeline version, lab, platform, antibody, donor; `contract.py`:88-95), tables `eic_v1_2` (lab, antibody, donor filled; platform unified). No QC columns. | PI ruling of 2026-11-17: every fillable field is included. Base recipes on `eic_v1_2` against `eic_v1`: within the range on every term except conv autoencoder correction 0.681 (0.032) against 0.718 (0.012) (ABLATIONS_READ 3.2; different table hashes, so no formal comparison). Use of covariates for correction: scrambled covariates in training lower the correction-only term on all three, -0.164 (2x), -0.198 (2x), -0.212 (beyond); foreign C' at scoring with true depth lowers crop-pair correction on all three (-0.165, -0.133, -0.242) (16 regions, ABLATIONS_READ 2.11 and 3.1). Imputation is within under every keep-depth foreign mode (256-region term), so the covariates carry no imputation information at test time. <!--p256:eyJjb25maWRlbmNlIjogIndlYWsgKGFsbCBjb3JyZWN0aW9uIGV2aWRlbmNlIGlzIDE2LXJlZ2lvbikiLCAic3RhdHVzIjogInZhbHVlIHNldHRsZWQgYnkgUEkgcnVsaW5nOyB0aGUgZmllbGQgc2V0IGlzIHByb3Zpc2lvbmFsIChydW4gMi44KTsgXCJ0aGUgbW9kZWxzIHVzZSBjb3ZhcmlhdGVzIHRvIGNvcnJlY3RcIiBpcyBwcm92aXNpb25hbCAocnVsZSAyLjEpIn0=-->256 regions (rescoring read 20261119T0110Z, scores as of 2026-11-19 01:10 UTC), rule 2.1: The models use covariates for same-view correction (settled, strong). Scrambled covariates lower the correction-only term beyond the seed range on the conv autoencoder, the conv + attention and the U-Net, and beyond twice the range on the conv autoencoder and the conv + attention. Recorded separately: the correction on crop pairs is lower beyond the seed range on the conv autoencoder and the conv + attention (on the 16 regions: the conv autoencoder). scrambled covariates, correction-only term: conv autoencoder -0.195 (seed range 0.046, beyond twice the seed range); conv + attention -0.230 (seed range 0.081, beyond twice the seed range); U-Net -0.242 (seed range 0.129, beyond the seed range). scrambled covariates, correction on crop pairs: conv autoencoder -0.055 (seed range 0.047, beyond the seed range); conv + attention -0.0261 (seed range 0.0258, beyond the seed range); U-Net -0.070 (seed range 0.084, within the seed range).<!--/p256--> | strong (use of covariates for correction, 256 regions); weak (field set) | value settled by PI ruling; the field set is provisional (run 2.8); "the models use covariates to correct" is settled, strong (rule 2.1, 256 regions) |
| 9 | Covariate embedding form | Base form: categorical columns embedded, numeric columns as scalars, no covariate dropout. | Covariate dropout 0.1 and per-value codes for numeric columns: within on all four metrics on all three. Numeric-only lowers U-Net correction by 0.133 (bar 0.071, 16 regions); not checked for a wiring fault. | moderate (null) | settled |
| 10 | Covariate injection | FiLM (`covariate_injection: film`). My choice. | adaLN-zero: within on all four metrics on all three. Tokens: U-Net imputation 0.383 (0.014), -0.049, 2x (256-region term); rejected. Concatenation: within on the conv autoencoder and U-Net; conv + attention correction 0.723 (0.024), +0.050 beyond (16 regions). ABLATIONS_READ recommends concatenation as the D13 simpler arm; I keep FiLM because concatenation costs 13% to 31% more FLOPs per window, the gain is a 16-region correction effect on one model, and every reference uses FiLM. | moderate (FiLM against adaLN-zero and tokens); weak (against concatenation) | settled for the conv autoencoder and U-Net; provisional for conv + attention (run 2.7) |
| 11 | QC metrics on the input side | Off (`qc_on_input: false`). | The arm has not run. The QC table holds 953 of 2,215 records (`closeout A4`, "State today"). | weak (default, not tested) | provisional (run 2.10) |
| 12 | Learned cell-type embedding | None. | The arm has not run (`closeout A6`; configs merged, not submitted). EIC holds out tracks, not cells, so the arm cannot say anything about unseen cell types. | weak (default, not tested) | provisional (run 2.11) |
| 13 | Average baseline input (training-cell average per assay and position) | Off. | U-Net imputation 0.466 (0.011) against 0.432 (0.023), +0.034 beyond, not 2x (256-region term); conv models move no composite term; conv autoencoder correction-only +0.071, 2x (16 regions). <!--p256:eyJjb25maWRlbmNlIjogIndlYWsiLCAic3RhdHVzIjogInByb3Zpc2lvbmFsIChydWxlIDIuNSwgcnVuIDIuNykifQ==-->256 regions (rescoring read 20261119T0110Z, scores as of 2026-11-19 01:10 UTC), rule 2.5: The average baseline input stays in the U-Net pool (qualify line). The conv autoencoder correction-only term is up beyond the seed range again (+0.098); it is recorded, and the value stays off on the conv models. average baseline input, correction-only term: conv autoencoder +0.098 (seed range 0.034, beyond twice the seed range); conv + attention +0.083 (seed range 0.048, beyond the seed range); U-Net +0.105 (seed range 0.129, within the seed range). average baseline input, correction on crop pairs: conv autoencoder +0.031 (seed range 0.054, within the seed range); conv + attention +0.018 (seed range 0.044, within the seed range); U-Net -0.007 (seed range 0.066, within the seed range).<!--/p256--> | weak | provisional (run 2.7; in the U-Net pool by rule 2.5) |
| 14 | ChIP control as an input | On (`control_input: true`). | Off lowers imputation on all three (-0.024, -0.026, -0.019; 256-region term), beyond on conv + attention only (0.360 (0.021) against 0.387 (0.012)). | weak | settled |
| 15 | ChIP control as an output | On (`predict_control: true`), as one channel of the NB head (`control_head: channel`). | PI ruling 14 keeps the control output. Off: conv + attention imputation -0.021 beyond; U-Net composite +0.014 beyond (screening). The `lambda_smooth` head was not scored. | settled by ruling | settled |
| 16 | DNA input | On: own DNA tower from one-hot sequence (`dna: own`), joined by early concatenation. | DNA off lowers conv + attention imputation by 0.018 (beyond); the 5 Mb shifted-sequence control lowers it by 0.020 (beyond); two independent arms agree (256-region term). Conv autoencoder and U-Net: within (the U-Net arm is width-confounded, 25 to 30). | weak | settled |
| 17 | DNA source | Own tower. | Frozen Nucleotide Transformer v2 embeddings: within on all four metrics on all three. Flashzoi did not run (ruling 26 made it an ablation arm only). | moderate (null) | settled |
| 18 | DNA fusion | Early concatenation (the base join). | Late join lowers conv + attention imputation 0.360 (0.004), -0.027, 2x; cross-attention -0.012 beyond (256-region term). FiLM by DNA: within on all three. U-Net early-join arm is 24.5% smaller (confounded). | weak | settled |
| 19 | Output head and likelihood | Negative Binomial per bin per track: mean mu = s_out * 2^eta with s_out from the target's C' depth, size n = exp(log 2 + eta_n); loss NB negative log-likelihood (`loss: nb_nll`). | Poisson mean with detached size: within on all four metrics on all three. Bin-equal weighting: opposite signs (conv + attention imputation -0.030 beyond; U-Net composite +0.015 beyond); kept off. NB CRPS: conv + attention correction 0.726 (0.015), +0.053 beyond (16 regions); within elsewhere. | moderate (NB NLL against Poisson); weak (against NB CRPS) | NB NLL settled for the conv autoencoder and U-Net; provisional for conv + attention (run 2.7) |
| 20 | Loss terms and weights | L = mean NB NLL over the loss mask, kind weights impute 1, correct 1, same 1; shape and multiscale terms off; no base leg (two-step off). | Imputation weight 4: conv + attention imputation 0.426 (0.017) against 0.387 (0.012), +0.039, 2x (256-region term), composite within; conv autoencoder correction 0.680 (0.031), -0.038 beyond (16 regions); U-Net within. Weights `correct` and `same` and the shape and multiscale terms: not tested. <!--p256:eyJjb25maWRlbmNlIjogIndlYWsiLCAic3RhdHVzIjogInByb3Zpc2lvbmFsIChydWxlIDIuNiwgcnVuIDIuNykifQ==-->256 regions (rescoring read 20261119T0110Z, scores as of 2026-11-19 01:10 UTC), rule 2.6: Imputation weight 4 stays in the conv + attention pool (qualify line). The weights stay 1 / 1 / 1 unless run 2.7 picks weight 4. The cost on the conv autoencoder is confirmed: correction on crop pairs -0.050 (seed range 0.021); weight 4 is not in its pool. imputation weight 4, correction on crop pairs: conv autoencoder -0.050 (seed range 0.021, beyond twice the seed range); conv + attention -0.013 (seed range 0.042, within the seed range); U-Net +0.021 (seed range 0.068, within the seed range). imputation weight 4, imputation: conv autoencoder +0.0276 (seed range 0.0284, within the seed range); conv + attention +0.039 (seed range 0.017, beyond twice the seed range); U-Net +0.007 (seed range 0.025, within the seed range).<!--/p256--> | weak | provisional (run 2.7; weight 4 stays in the conv + attention pool, rule 2.6) |
| 21 | Masking (imputation and denoising) | Whole-track mask: p drawn from U(0.2, 0.8), each training track masked with probability p; bin mask on same-view targets: each bin hidden with probability 0.25, random shape; same-view targets count only on hidden bins (copy-leak rule). | Defaults that no close-out run varied (`onion training objective.md` sections 4 and 10). The hidden-bins-only rule is a hard gate, not an option. | weak (default, not tested) | settled (default) |
| 22 | Pair types (`same_view_prob`) | 0.5: half of the visible tracks get a same-view target, half a different-view target. | Same-view only (1.0): conv autoencoder imputation 0.413 (0.010), +0.031 beyond; conv + attention 0.419 (0.021), +0.032 beyond (256-region term); U-Net composite 0.362, -0.133, 2x, and raw correction far below 0 (seeds -30.1, -0.90, -10.6). | moderate (U-Net); weak (conv models) | settled 0.5 for the U-Net; provisional for the conv models (run 2.7) |
| 23 | Counterfactual views in training | All view kinds (base, depth-thinned 0.5 / 0.25 / 0.125, no dedup, MAPQ 0, single-end, crop 36, crop 50); `train_views` absent. | Base view only: composite within on all three, but correction-only falls 2x on both conv models (0.402 against 0.690; 0.390 against 0.659) and beyond on the U-Net (0.069 against 0.318) (16 regions). Base + alignment-level views lower conv imputation (conv autoencoder 0.340, -0.043 beyond; conv + attention 0.323, -0.064, 2x; 256-region term). U-Net read-level and alignment-level arms have 2 seeds. <!--p256:eyJjb25maWRlbmNlIjogIndlYWsgKHRoZSBjYXNlIGZvciBhbGwgdmlld3MgcmVzdHMgb24gdGhlIDE2LXJlZ2lvbiBjb3JyZWN0aW9uLW9ubHkgdGVybSkiLCAic3RhdHVzIjogInByb3Zpc2lvbmFsIChydWxlIDIuMikifQ==-->256 regions (rescoring read 20261119T0110Z, scores as of 2026-11-19 01:10 UTC), rule 2.2: All counterfactual views stay in training (settled, strong). Base view only lowers the correction-only term beyond the seed range on the conv autoencoder, the conv + attention and the U-Net and beyond twice the range on the conv autoencoder, the conv + attention and the U-Net. base view only, correction-only term: conv autoencoder -0.341 (seed range 0.067, beyond twice the seed range); conv + attention -0.315 (seed range 0.104, beyond twice the seed range); U-Net -0.262 (seed range 0.129, beyond twice the seed range). base view only, composite: conv autoencoder +0.007 (seed range 0.043, within the seed range); conv + attention +0.008 (seed range 0.048, within the seed range); U-Net +0.020 (seed range 0.039, within the seed range).<!--/p256--> | strong (correction-only term on 256 regions) | settled (rule 2.2, 256 regions) |
| 24 | Two-step routing through the base view | Off (`two_step: false`). If ever used, fixed base depth per assay class. | Native: correction lower beyond on both conv models (0.602 (0.085), -0.115; 0.614 (0.033), -0.059; 16 regions); conv + attention imputation -0.021 beyond (256-region term). Fixed base depth on the U-Net: correction 0.636 (0.069), +0.129 beyond (16 regions), calibration 0.445 (0.119), -0.165 beyond (256-region term). FLOPs per window 2.0x. No arm raises the composite. <!--p256:eyJjb25maWRlbmNlIjogIm1vZGVyYXRlIiwgInN0YXR1cyI6ICJwcm92aXNpb25hbCB1bnRpbCBydWxlIDIuMyBpcyBhcHBsaWVkIn0=-->256 regions (rescoring read 20261119T0110Z, scores as of 2026-11-19 01:10 UTC): rule 2.3 not decided; it waits for two-step routing, native base depth on the U-Net (0 of 3 seeds); two-step routing, fixed base depth on all three models (0 of 3 seeds each). Cells so far: two-step routing, native base depth, composite: conv autoencoder -0.079 (seed range 0.043, beyond the seed range); conv + attention -0.051 (seed range 0.056, within the seed range); U-Net not finished (0 of 3 seeds). two-step routing, fixed base depth, composite: conv autoencoder not finished (0 of 3 seeds); conv + attention not finished (0 of 3 seeds); U-Net not finished (0 of 3 seeds). U-Net fixed base depth: correction on crop pairs not finished (0 of 3 seeds); calibration not finished (0 of 3 seeds).<!--/p256--> | moderate | provisional until rule 2.3 is applied |
| 25 | Depth handling | Native input depth (`input_depth: native`). Depth enters three times: input scale s_in, output offset s_out from C' depth, and column 0 of C and C'. | Hiding depth on both sides cut the U-Net composite by 0.184 (`results/ladder_audit/COVERAGE.md` row 27, older composite). Fixed input depth (every input thinned to 2^23 reads): U-Net correction 0.675 (0.013), +0.168, 2x (16 regions), imputation 0.404 (0.010), -0.028 beyond (256-region term), composite +0.023, 2x (screening); conv + attention correction +0.049 beyond (16 regions); conv autoencoder 2 seeds (seed 1 did not train; PI accepted 2 seeds), within. PI caveat, 2026-11-17: "requiring fixed input depth can be unrealistic in real world deployment and inference". <!--p256:eyJjb25maWRlbmNlIjogIndlYWsiLCAic3RhdHVzIjogInByb3Zpc2lvbmFsIChydWxlIDIuNCwgcnVuIDIuNykifQ==-->256 regions (rescoring read 20261119T0110Z, scores as of 2026-11-19 01:10 UTC), rule 2.4: U-Net: the trade-off holds (correction on crop pairs up, imputation down, both beyond the seed range); the switch stays in the U-Net pool by PI ruling, and the native-depth score of run 2.7 decides whether it can be deployed. Conv + attention: the qualify line keeps the switch in the conv + attention pool. The conv autoencoder (2 seeds) is reported only. No fixed-depth value enters the recipe from these scores alone. fixed input depth, correction on crop pairs: conv autoencoder -0.003 (seed range 0.047, within the seed range); conv + attention +0.039 (seed range 0.026, beyond the seed range); U-Net +0.166 (seed range 0.066, beyond twice the seed range). fixed input depth, imputation: conv autoencoder -0.019 (seed range 0.028, within the seed range); conv + attention -0.011 (seed range 0.012, within the seed range); U-Net -0.028 (seed range 0.023, beyond the seed range).<!--/p256--> | weak | provisional (rule 2.4 applied; run 2.7) |
| 26 | Resolution | 25 bp bins. | PI ruling 28: the final model stays at 25 bp in this run; the 1 to 125 bp ablation is reported only. The resolution arms have not run (`closeout A7`). | settled by ruling | settled for this run; provisional for scaling (run 2.12) |
| 27 | Context | 1,024 bins (25.6 kb). | Not varied in the close-out. Context is a scaling axis (PI, `peeling the onion.md`:162-165). | not tested | settled for the prototype |
| 28 | Training length | 80,000 windows (10,000 steps of batch 8). | No architecture converged at 80,000 windows (`FINAL_ROUND_EQUAL.md`, notes). 160,000 windows lowered the held-out loss but not the composite (finding f88). Matched-compute runs: 12 trainings queued on Fir, 0 done (`closeout A9`). | weak | provisional (run 2.13) |
| 29 | Optimizer | AdamW, lr 1e-3, weight decay 0.01, gradient clip 1.0, linear warm-up 500 steps, cosine decay to 10%. | U-Net lr 3e-3: composite 0.511 (0.028) against 0.495 (0.010), within. Other settings not varied. | weak | settled |
| 30 | Data version | `eic_v1_2`. | PI ruling of 2026-11-17 (corrected covariate tables are the default). The candidate with DNase-seq in the correction term (2,812 rows; 12 correction-scorable assays against 11; `closeout D3` 9.6) is not a version yet. | settled by ruling | settled; the version of the final model is a PI decision (section 4) |
| 31 | Signal p-value and peaks | Derived after training from the predicted NB (`box_d` pileup, `comonotone` dependence by default; PI ruling of 2026-11-17). Not part of training. | One run-seed scored (conv autoencoder seed 0), with the superseded in-bin pileup: signal-space variance recovered 0.178 against 0.331 in counts (`results/closeout_E2/E2_RESULTS.md` section 2). No seed range exists. | weak | provisional (run 2.14) |

Every ablated switch has a row: control input (14), control output (15), average input (13), two-step (24), DNA source (17), DNA on or off (16), covariate embedding (9), fixed input depth (25), losses (19, 20), adaLN-zero and injection form (10), scrambled covariates (8), mask mode (7), training views and pair types (22, 23), count transforms (5, 6), DNA fusion (18). The scrambled-covariates arm is a control, not a design choice.

<!-- p256-section start (apply_rescoring_rules.py) -->
## 256-region rescoring: what it settled (2026-11-19)

Generated by `tools/onion_eval/apply_rescoring_rules.py` from the arm summary of scores as of 2026-11-19 01:10 UTC (read folder `results/rescoring_p256/read_20261119T0110Z/`, git-ignored; the table is `results/rescoring_p256/arm_summary_p256_sv.tsv`). It applies rules 2.1 to 2.6 of section 2.0 as written. A rule is decided only when all its arms have their planned seeds. Rerun: `/Users/mforooz/miniforge3/envs/candii/bin/python tools/onion_eval/apply_rescoring_rules.py <arm-summary tsv> --as-of <YYYY-MM-DDTHH:MMZ of the table>`.

| rule | arms | verdict | design rows | new status |
|---|---|---|---|---|
| rule 2.1 (covariates used for correction) | scrambled covariates | The models use covariates for same-view correction (settled, strong). Scrambled covariates lower the correction-only term beyond the seed range on the conv autoencoder, the conv + attention and the U-Net, and beyond twice the range on the conv autoencoder and the conv + attention. Recorded separately: the correction on crop pairs is lower beyond the seed range on the conv autoencoder and the conv + attention (on the 16 regions: the conv autoencoder). | 8 | row 8: value settled by PI ruling; the field set is provisional (run 2.8); "the models use covariates to correct" is settled, strong (rule 2.1, 256 regions) |
| rule 2.2 (training views) | base view only | All counterfactual views stay in training (settled, strong). Base view only lowers the correction-only term beyond the seed range on the conv autoencoder, the conv + attention and the U-Net and beyond twice the range on the conv autoencoder, the conv + attention and the U-Net. | 23 | row 23: settled (rule 2.2, 256 regions) |
| rule 2.3 (two-step routing) | two-step routing, native base depth; two-step routing, fixed base depth | not decided | 24 | unchanged (waits for: two-step routing, native base depth on the U-Net (0 of 3 seeds); two-step routing, fixed base depth on all three models (0 of 3 seeds each)) |
| rule 2.4 (fixed input depth) | fixed input depth | U-Net: the trade-off holds (correction on crop pairs up, imputation down, both beyond the seed range); the switch stays in the U-Net pool by PI ruling, and the native-depth score of run 2.7 decides whether it can be deployed. Conv + attention: the qualify line keeps the switch in the conv + attention pool. The conv autoencoder (2 seeds) is reported only. No fixed-depth value enters the recipe from these scores alone. | 25 | row 25: provisional (rule 2.4 applied; run 2.7) |
| rule 2.5 (average baseline input) | average baseline input | The average baseline input stays in the U-Net pool (qualify line). The conv autoencoder correction-only term is up beyond the seed range again (+0.098); it is recorded, and the value stays off on the conv models. | 13 | row 13: provisional (run 2.7; in the U-Net pool by rule 2.5) |
| rule 2.6 (imputation loss weight 4) | imputation weight 4 | Imputation weight 4 stays in the conv + attention pool (qualify line). The weights stay 1 / 1 / 1 unless run 2.7 picks weight 4. The cost on the conv autoencoder is confirmed: correction on crop pairs -0.050 (seed range 0.021); weight 4 is not in its pool. | 20 | row 20: provisional (run 2.7; weight 4 stays in the conv + attention pool, rule 2.6) |

<!-- p256-section end -->

## 2. Provisional rows: the run that settles each, its cost, and the rule

The rules below are written before any result. "Beyond", "2x" and "within" are as in section 0. For a model, the reference is `rung2_base_heldout` (conv autoencoder), `rung3_base_heldout` (conv + attention) or `rung4_base_heldout` (U-Net); for an arm row A and reference row R of an arm-summary table and a key k: d = A[`k.mean`] - R[`k.mean`], bar = max(A[`k.range`], R[`k.range`]). The keys are `imputation_raw`, `correction_v3_raw`, `uncertainty_raw`, `composite_v3` and `correction_only`.

### 2.0 Applying the 256-region rescoring (rules 2.1 to 2.6)

**The run.** 62 run-seeds of 7 decisive arms, scored into `p256_sv` on Nibi (PI approved, about 2,225 allocated CPU-hours: 20 conv autoencoder run-seeds x 30.0 + 21 conv + attention x 25.4 + 21 U-Net x 52 = 600 + 533.4 + 1,092; notepad, "Numbers told to the PI and later corrected", item 2). Run folders: `/project/def-maxwl/mforooz/ONION/runs_fir/<arm>/seed*/p256_sv` on Nibi. I checked on 2026-11-17 that all 21 arm folders are there (62 seeds: 20 x 3 + 2) and that none has `p256_sv/SCORED`; the 4 references have 3 of 3.

| arm folder (`runs_fir/<arm>`) | rule | cost (CPU-h) |
|---|---|---|
| `abl_scrambled_covariates_scrambled_covariates_{convae,convattn,unet}` | 2.1 | 3 x (30.0 + 25.4 + 52) = 322.2 |
| `abl_views_base_only_{convae,convattn}`, `rung4_views_base_only` | 2.2 | 322.2 |
| `abl_two_step_two_step_native_{convae,convattn,unet}` and `abl_two_step_two_step_fixed_{...}` | 2.3 | 2 x 322.2 = 644.4 |
| `abl_depth_input_input_depth_fixed_{convae (2 seeds),convattn,unet}` | 2.4 | 2 x 30.0 + 3 x 25.4 + 3 x 52 = 292.2 |
| `abl_average_input_average_input_{convae,convattn,unet}` | 2.5 | 322.2 |
| `abl_losses_loss_impute_x4_{convae,convattn,unet}` | 2.6 | 322.2 |

**Procedure for the foreman,** after every `seed*/p256_sv/SCORED` of these folders exists (on a Nibi login node; `$K` is the kit that ran the rescoring, CO7 by the Log of 2026-11-17T10:40Z; I did not find `/project/def-maxwl/mforooz/onion_kit/CO7` on Nibi on 2026-11-17, so use the kit path of the submit log; Python is the environment of the scoring jobs):

```bash
module load python/3.10.13; unset EBPYTHONPREFIXES
O=/project/def-maxwl/mforooz/ONION
PYTHONPATH=$K/src python $K/tools/onion_eval/select_rule.py arm-summary \
    --runs $O/runs_fir --sub p256_sv > $O/runs_fir/ARM_SUMMARY_p256_sv.tsv
python $K/tools/minicandi/mix_arms.py qualify --summary $O/runs_fir/ARM_SUMMARY_p256_sv.tsv
```

`mix_arms.py qualify` re-applies the mix-and-match pool rule (better beyond on one of `imputation_raw`, `correction_v3_raw`, `uncertainty_raw`, and worse beyond on none) to every switch of the pool. It prints one line per switch and exits 0 (PASS), 1 (FAIL: a switch no longer qualifies) or 3 (WAIT: a switch has no 3-seed 256-region score). On this rescoring it MUST exit 3, because the single runs of same-view only (conv autoencoder, conv + attention), concatenation, NB CRPS (conv + attention) and Pearson (U-Net) are not among the 62 run-seeds. Read the lines that the rescoring does cover: `rung4 average_input`, `rung3 fixed_depth`, `rung3 impute_x4`, and `rung4 fixed_depth` (always `in_by_pi_ruling`). A line `fails` removes that switch from `POOL` in `tools/minicandi/mix_arms.py` before the mix-and-match round (then `mix_arms.py write`, and recount the cost). A line `qualifies` keeps it. The other switches wait for `STAGE=gate` of `submit_mix.sh` (run 2.7).

**Rule 2.1, covariates used for correction (row 8).** Scrambled-covariates arms against their references.
- If `correction_only` is down beyond on 2 or more models: "the models use covariates for same-view correction" becomes moderate (strong if 2x on 2 or more), settled.
- If `correction_only` is down beyond on 1 model or none: the claim becomes weak; the covariates stay in the recipe by the PI ruling, and the minimum covariate set (run 2.8) gets priority.
- Record separately on which models `correction_v3_raw` is down beyond (16 regions: conv autoencoder only).

**Rule 2.2, training views (row 23).** Base-view-only arms against their references.
- If `correction_only` is down beyond on 2 or more models: "all views" is settled (moderate; strong if 2x on 2 or more).
- If `correction_only` is within on 2 or more models, and `composite_v3` of base only is not lower beyond on any model: the views show no benefit on 256 regions; base view only is the simpler arm; the PI decides (the PI asked whether and where the counterfactual views help, ruling 52).
- Any other outcome: "all views" stays, provisional, weak.

**Rule 2.3, two-step routing (row 24).** Native and fixed arms against their references.
- Off is settled unless an arm has `composite_v3` up by 2x and none of `imputation_raw`, `correction_v3_raw`, `uncertainty_raw` down by 2x (D13). If an arm meets this, two-step enters the mix-and-match pool for that model (a new arm, about 120 to 250 CPU-h per model and 3 seeds); the PI decides.
- If `correction_v3_raw` of the U-Net fixed arm is up beyond again: record the U-Net trade-off (correction up, calibration down); off stays.

**Rule 2.4, fixed input depth (row 25).** These scores thin the inputs at scoring as in training, so they are not the deployment case.
- U-Net: if `correction_v3_raw` is up beyond and `imputation_raw` is down beyond, the trade-off holds; the switch stays in the U-Net pool (PI ruling) and its deployability is decided by the native-depth score of run 2.7. If `correction_v3_raw` is within, the 16-region gain did not hold; tell the PI, who put the switch in the pool.
- Conv + attention: the `rung3 fixed_depth` line of `qualify` decides (keeps or removes the switch).
- Conv autoencoder (2 seeds): report only.
- No fixed-depth value enters the recipe from these scores alone.

**Rule 2.5, average input (row 13).** The imputation effect is already a 256-region term; the rescoring checks the correction terms. The `rung4 average_input` line of `qualify` decides whether the switch stays in the U-Net pool. On the conv models, if `correction_only` of the conv autoencoder is up beyond again (16 regions: +0.071, 2x), record it; the value stays off there (no qualifying term).

**Rule 2.6, imputation loss weight 4 (row 20).** The `rung3 impute_x4` line of `qualify` decides whether the switch stays in the conv + attention pool. If conv autoencoder `correction_v3_raw` is down beyond on 256 regions, the cost on that model is confirmed (weight 4 is not in its pool). The weights stay 1 / 1 / 1 unless run 2.7 picks weight 4.

### 2.7 Mix-and-match round (rows 6, 10, 13, 19, 20, 22, 25)

- **Run.** `tools/minicandi/configs/ablations/mix/submit_mix.sh` (dry run by default): `STAGE=gate` (256-region scores of the 5 single runs that the rescoring lacks; 474.1 CPU-h), then `STAGE=baselines` (256-region references of `eic_v1_2`; 95.1 CPU-h), then `QUALIFY_SUMMARY=<tsv> STAGE=runs` (24 arms plus 3 references, 3 seeds, `eic_v1_2`, 256 regions; fixed-depth arms also scored at native depth into `p256_sv_native_depth`). Cost about 5,390 CPU-h on Nibi or 5,308 on Fir, plus the gate (`closeout A13` 4.8). Not submitted; the budget needs PI approval.
- **Pool today** (provisional until `qualify` exits 0): conv autoencoder: same-view only; conv + attention: concatenation, fixed input depth, imputation weight 4, NB CRPS, same-view only (half fraction, 16 arms); U-Net: average input, Pearson, fixed input depth (full factorial, 7 arms).
- **Rule** (`closeout A13` 4.6, written before any run). Per model: an arm is eligible when none of imputation, gap closed and calibration is lower than the reference beyond the range; the fixed-depth arms use the native-depth score. Among eligible arms the leader has the highest composite; the winner is the in-range arm (composite within the larger range of the leader) with the fewest switches, and the reference wins whenever it is in range. **Keep** the row's current value when the reference wins or no eligible arm is higher than the reference beyond the range on the composite or on one of the three terms. **Change** the row's value to the switch setting of the winning arm otherwise. A fixed-depth win under thinned scoring that disappears under native-depth scoring is not deployable; the PI rules. A correction-only term lower than the reference by more than 2x is flagged for the PI.

### 2.8 Minimum useful covariate set (row 8, field set)

- **Run.** `tools/minicandi/configs/ablations/covariate_fields/submit_a12.sh` (`STEP=ref|screen|lofo|groups|curve`, dry run by default), on the final recipe of 2.7. Cost 1,927 to 17,900 CPU-h by the scope that the PI chooses (`closeout A12` 7.7; the full plan is 17,901). Not submitted.
- **Rule** (`closeout A12` 7.4): per field, z = -(arm mean minus full-set mean) / larger seed range on the 256-region terms. Keep a field when z > 1 on 2 of 3 models or z > 2 on one; remove it only when the field and every scored group that holds it have z <= 1 everywhere; the exclusion curve gives the minimum set as the 11 fields minus the first k removed before the first step beyond the range.
- **Blocker found by this item.** `tests/test_minicandi_recipe_keys.py::test_every_config_on_this_branch_passes_the_guard` fails at e8c1955: the A12 configs carry the key `covariate_drop_fields`, which `train.check_config_keys` does not find in the code (the code reads it through the constant `covariates.DROP_FIELDS_KEY`). I checked that `check_config_keys` refuses `configs/ablations/covariate_fields/rung2/group_all_but_depth.json`, so every A12 training would stop at start. This must be fixed before run 2.8. I did not fix it.

### 2.9 Pointwise MLP (row 2)

- **Run.** Close-out item A10: same-view pass and version-3 composite of the pointwise MLP and its arms, 3 seeds. Cost: metrics about 11.5 CPU-h per seed plus a GPU or CPU forward pass (`closeout A10` section 7).
- **Rule** (`closeout A10` section 6): if the pointwise MLP is below the conv models beyond the D13 bar, local context helps and row 2 is settled; if it is within the bar, context buys little at prototype scale, and the scaling of the context axis needs this result.

### 2.10 QC on input (row 11)

- **Run.** Close-out item A4: finish the QC fill (1,262 rows without a record), then one arm, 3 models x 3 seeds, on the same table version as its reference. Cost: QC fill about 0.26 CPU-h per row; arm about 82 CPU-h training plus 9 x 36.3 = 326.7 CPU-h of 256-region scoring (unit costs of `closeout A12` 7.7).
- **Rule** (`closeout A4` section 6): QC on input enters the design if it is beyond the default on the composite or the correction term on 2 or more models; if within on all three, the simpler arm (metadata only) stays; worse beyond means noise or a leak.

### 2.11 Learned cell-type embedding (row 12)

- **Run.** `tools/minicandi/configs/ablations/cell_embedding/submit_cell_embedding.sh` (`MODE=train`, then `audit` and `derive`, then `zeroshot`). Cost about 82 CPU-h training plus 2 x 326.7 CPU-h of scoring plus 2 = about 738 CPU-h at the measured 256-region unit (the item file's 498 used the old unit of 23 CPU-h per scoring). It also needs the 256-region references of `eic_v1_2`.
- **Rule** (`closeout A6` 9.7): a gain in variance recovered on 2 or more models counts as "cell identity carries information" only if the zero-shot control (cell id forced to unknown) falls back to the reference; within on all three keeps "none".

### 2.12 Resolution (row 26, for scaling)

- **Run.** `slurm/onion_data/closeout_a7_submit.sh` (`plan`, `probe`, `recount`, `tables`, `variance`, `arms`). Cost: recount about 130 CPU-h; reduced arm design about 405 CPU-h, full design about 2,340 (`closeout A7` 9.7). The recount needs the tagAligns of Nibi `onion/work` before that folder leaves.
- **Rule** (`closeout A7` 9.5): 25 bp stays for histone marks if the reproducible share below 25 bp is under 5%; a finer arm that beats 25 bp beyond the range on a common table means resolution matters for the model; the context-control arms separate the effect of the bin from the effect of the context.

### 2.13 Training length (row 28)

- **Run.** Close-out item A9: 12 matched-compute trainings queued on Fir (jobs 4345024 to 4345035), then their scoring. Cost 1,507 to 2,393 CPU-h, mostly allocated already (`closeout A9` section 7).
- **Rule** (`closeout A9` section 6): same top group at matched compute keeps the order; an architecture that moves up beyond the range means the equal-window order favored fast learners, and scaling uses the matched order; matched worse than equal is a scaling risk (section 4).

### 2.14 Signal p-value and peaks (row 31)

- **Run.** `slurm/onion_eval/e2_submit.sh`: `box_d` rescoring of the 9 base runs under both dependence forms, plus the board references (`closeout E2`; cost in that item file). It needs a fresh kit and a cluster.
- **Rule.** The signal route is a reported output, not a design switch. It is settled when 3 seeds per model have signal-space scores under `box_d`; the signal-space variance recovered is then reported with its seed range beside the count-space value.

### 2.15 Other runs that the rows above wait for

- RNA-seq input (close-out item A5; ruling 27: ablation only, never in the EIC model): 2 arms, about 524 CPU-h at the old unit; rule `closeout A5` section 6. It does not change any row of the EIC design.
- 256-region references of `eic_v1_2` (`eval/eic_v1_2/baselines`, 4 baselines and the oracle reference; 95.1 CPU-h) and the 256-region scores of the 9 `eic_v1_2` base runs (9 x 36.3 = 326.7 CPU-h). Every `eic_v1_2` comparison on 256 regions needs both.

## 3. Tested

I ran on the laptop (2026-11-17, env `candii`, `PYTHONPATH=$PWD/src`):

```bash
python -m pytest -q tests/test_minicandi_contract.py tests/test_minicandi_losses.py \
  tests/test_minicandi_rung2.py tests/test_minicandi_rung3.py tests/test_minicandi_rung4.py \
  tests/test_minicandi_recipe_keys.py tests/test_minicandi_mix_arms.py tests/test_minicandi_dataset.py \
  tests/test_minicandi_train.py tests/test_onion_eval_select_rule.py tests/test_minicandi_predict.py \
  tests/test_minicandi_train_views.py tests/test_onion_eval_pair_regions_check.py tests/test_onion_eval_score.py
```

Result: 555 passed, 3 skipped, 1 failed (the A12 key guard of 2.8).

**Covered by tests (synthetic data):**

| code path of the settled design | test file |
|---|---|
| track order, target kinds, loss mask, NB NLL against scipy and the closed form, wiring check | `tests/test_minicandi_contract.py` |
| loss defaults bit-identical to the plain NB NLL; NB CRPS and Poisson arms | `tests/test_minicandi_losses.py`, `tests/test_minicandi_loss_arms.py` |
| the three models: base config keys, output shapes, parameter count within the budget, width refit, U-Net skip guardrail | `tests/test_minicandi_rung2.py`, `tests/test_minicandi_rung3.py`, `tests/test_minicandi_rung4.py`, `tests/test_minicandi_components_arms.py` |
| sampler: roles, withheld view pairs never drawn in training, masks | `tests/test_minicandi_dataset.py`, `tests/test_minicandi_dataset_arms.py` |
| training views default bit-identical; base-only reads no other view | `tests/test_minicandi_train_views.py` |
| train loop, save and load, exact resume, unknown config keys refused | `tests/test_minicandi_train.py`, `tests/test_minicandi_recipe_keys.py` |
| prediction tiling and centre crop; targets never read | `tests/test_minicandi_predict.py` |
| mix-and-match configs equal to the single-ablation key sets; the half fraction is orthogonal; the qualify gate | `tests/test_minicandi_mix_arms.py` |
| selection rule, arm summary, comparability refusals | `tests/test_onion_eval_select_rule.py` |
| scorer, pair-region subset, 16-versus-256 check | `tests/test_onion_eval_score.py`, `tests/test_onion_eval_pair_regions_check.py` |

**Not covered:**
- Training on real EIC data. All tests use synthetic tables; the real-data checks are cluster runs (for example Nibi job 27601641, "ALL OK", for the mix-and-match configs; Log 2026-11-17T11:10Z).
- A test that a transform's output link trains from the zero-initialized head. The Anscombe defect passed its 3-step synthetic check (ABLATIONS_READ 5.1).
- The submit scripts (`submit_all.sh`, `submit_mix.sh`, `base_on_version.sh`) beyond their dry runs.
- `run.json` does not record the dataset and loss defaults (`mask_prob_range`, `same_view_prob`, `bin_mask_frac`, `mask_scheme`, `loss`, `loss_weights`), and the Fir base runs record `git_sha` "unknown" (`onion training objective.md` section 6). A run cannot prove its full objective from its own record.
- The A12 key-guard failure (2.8).
- No `tools/golden.py` check ran for the ablation kits against the reference kits (ABLATIONS_READ 5.4).

## 4. Ready for scaling

**Config files of the settled recipe** (identical, key for key, to `tools/minicandi/configs/final/rung{2,3,4}/reference_equal.json`, the mix-and-match references; I checked with `diff`):

| model | config | sha256 |
|---|---|---|
| conv autoencoder | `tools/minicandi/configs/rung2/base_heldout.json` | 5b8e3db6437bd450adced20dd478fc930329286be3a1ee8f10d6eead274cb139 |
| conv + attention | `tools/minicandi/configs/rung3/base_heldout.json` | 0a1724780d9e7a88bccb1f21ee9502570ce94cc909a3cd9a740c84e87d735de6 |
| U-Net | `tools/minicandi/configs/rung4/base_heldout.json` | baecd1be6f74ebb0f2a5a6941d2f7e839001313d1b4ba76239d510af4ab40ee3 |

Every value of the design table that a config key sets agrees with these files: `covariate_injection` film, `mask_mode` token, `dna` own, `count_transform` offset, `qc_on_input` false, `two_step` false, `base_depth` native, `average_input` false, `control_input` true, `predict_control` true, `param_budget` 1.0e6, `windows_seen` 80,000, `lr` 0.001, `batch_size` 8, `context_bins` 1,024, U-Net `skip_source` both. The other values (loss, loss weights, masking, pair types, views, input depth, optimizer schedule) are code defaults, not config keys. The table version is not a config key: `tools/minicandi/configs/base_on_version.sh` trains these configs on `V=eic_v1_2` as `eic_v1_2_rung{2,3,4}_base_heldout` (9 runs exist on Fir, scored on 16 regions only). No config was missing, so I added none and changed no code.

**What changes when scaling.**
- Parameters: `param_budget`; the registry refits the trunk width (`test_param_budget_moves_the_width`). Every other key stays.
- Windows: `windows_seen`; the warm-up is min(1,000, 5% of the steps) (`train.py`:126, 164-165).
- Context: `context_bins`; the scorer cuts regions into windows of the same size, so the evaluation changes too.
- Data: `eic_v1_2` (the default) or the next version with DNase-seq in the correction term (`eic_v1_3`, from the 2,812-row candidate; `closeout D3`). A new version changes the table hash; the scorer refuses to compare runs across versions, so the references, the 256-region baselines and every compared arm must be on one version. If the final model trains on `eic_v1_3`, the 9 base runs are retrained a second time.
- Pending runs of section 2 can change rows 6, 8, 10, 11, 12, 13, 19, 20, 22, 23, 24, 25, 28 before or during scaling.

**Known risks.**
1. No architecture converged at 80,000 windows, and for the conv autoencoder more training lowered the composite in the tuning round (finding f88; 160,000 windows). The scaling curve can bend down on the composite while the held-out loss falls.
2. The correction term rests on 18 crop pairs in 6 cells (`results/correction_metric_review/REVIEW.md` section 4). Its seed ranges reach 0.071 (U-Net).
3. Correction terms must be scored on all 256 regions: 25.4 to 53.4 allocated CPU-hours per run-seed (`closeout A12` 7.7), against 0.83 for the old pilot scoring. A prediction store holds about 55 GB and 1,700 files per scoring on `/scratch`.
4. Fixed input depth is deployment-unrealistic (PI caveat). Any gain must hold under native-depth scoring.
5. Blind measurement shift: the blind tracks may differ from training in platform and lab. The category lists of `eic_v1_2` come from the training rows only (`tools/minicandi/configs/base_on_version.sh`, header), so a lab or antibody seen only in blind tracks has no learned embedding. I did not check how the code encodes such a value.
6. Training windows may overlap the evaluation regions (ruling 39). The seen-versus-unseen gap is measured by one arm, not removed.
7. The evaluation-region count stays at 256 (left for follow-up work); the region-file rule (raise regions when 2x the composite seed range exceeds 0.05) is tripped by most models.
8. The average-input file is built for `eic_v1` (`/scratch/mforooz/onion/aux/eic_v1_average.h5`); a new version needs its own if row 13 changes.
9. `run.json` does not record the full objective (section 3); a scaled run should record every dataset and loss default.

## 5. What this design does not know yet

1. Whether any correction-term effect of the 16 ablations holds on 256 regions (rescoring not run).
2. Which architecture to scale: the three tie on the composite at 256 regions, with an imputation-against-correction trade-off.
3. Whether local context is needed at all (the pointwise MLP has no version-3 score).
4. Whether the single gains combine (mix-and-match round not run).
5. Which covariate fields are needed (minimum set not run; its configs are refused by the key guard today).
6. Whether QC metrics on the input, a cell-type embedding or RNA-seq input add information.
7. Whether resolution below 25 bp matters for the model.
8. Whether more compute changes the architecture order (matched-compute runs queued on Fir).
9. Whether the masking defaults (mask fraction, bin-mask fraction, kind weights) are good: no run varied them.
10. How the design does on test data: D14 condition 2 is met on validation for imputation and calibration by every model (my arithmetic from `FINAL_ROUND_EQUAL.md`: margins over the position-average baseline 0.263 to 0.313 on imputation against 2x ranges of 0.024 to 0.056; 0.442 to 0.484 on calibration against 0.216 to 0.356), and on the 16-region correction term by both conv models (0.335 and 0.290 against 0.024 and 0.062) but not by the U-Net (0.507 - 0.383 = 0.124 against 2 x 0.071 = 0.142). I did not run `select_rule.py check-done`, and no 256-region baseline comparison is in this document. The test-split result comes from item D4.

## Decisions that the PI makes in the next session

PI ruling of 2026-11-17: "scaling is next session's task. the task for this session is to provide the necessary information and prototype minicandi for a more informed and confident scaling in the next session". On the data version: "you recommend + provide info. we'll work on it next session". So this document recommends and informs; it decides neither point.

| Decision | Recommendation (the agent's intent) | Information for it | What would change the recommendation |
|---|---|---|---|
| Architecture that goes to scaling | Carry the conv autoencoder and the U-Net into the first scaling step, and choose after the mix-and-match round. | Row 1 of the design table: the three models tie on the composite on 256 regions; the U-Net is better on imputation, both conv models are better on correction. Rows 3, 22, 24 and 25 give the U-Net-specific trade-offs. | The 256-region rescoring (rules 2.1 to 2.6) or the mix-and-match round moves one model beyond twice the seed range on the composite with no goal lower. |
| Data version of the final model | The candidate that makes DNase-seq correction-scorable, built as a version in the next session. | Row 30; `closeout D3 dataset update with DNase-seq.md`; ready queue entry 15 (4 row rebuilds, version build, baselines; the 9 base runs train again on it). | The 4 row rebuilds fail again, or the base runs on the candidate differ from the current tables beyond the seed range. |
