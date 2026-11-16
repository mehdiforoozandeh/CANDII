# Ablation 16: DNA fusion (`dna_fusion`)

Specification, 2026-11-16. Close-out item A11 (`closeout/closeout A11 DNA fusion.md`). Branch `chunk/onion-CO-A11`, base `309d0f6`.

## 1. Question

With the DNA source fixed (own tower), does the way the DNA features join the count features change imputation, correction or calibration? Two factors: the operator (concatenation with a linear projection, FiLM, cross-attention) and the stage (at the input, or at the bottleneck). The PI's line, `peeling the onion.md`:155: "how we embed and fuse different modalities ... these are targets for ablation i.e. covariate embedding and covariate injection, DNA embedding and injection, etc." Options listed in `miniCANDI architecture ablations.md`:173-177.

Not in this ablation: the DNA source (ablation 5), no DNA (ablation 6), the U-Net skip source (`skip_source`; run in the final round). Options 2 and 4 of the ablations file (a DNA token row in axial attention, DNA tokens into a Perceiver latent) belong to rungs 5 and 6, which the PI ruling does not include.

## 2. Arms

The base joins (read from the code at `309d0f6`):
- Conv autoencoder: early concatenation of the 32 DNA channels with the 36 x 16 track channels; the 1x1 stem conv projects them (`conv_ae.py` L183-185, L245-248).
- Conv + attention: early concatenation into the track-fusion Linear (`conv_attn_ae.py` L155-156, L191-195).
- U-Net: a DNA tower of its own; its bottleneck is concatenated with the count bottleneck and projected by a 1x1 conv (`fuse`); the tower also feeds the DNA skips (`unet.py` L210-213, L289-294).

Each arm sets one key, `dna_fusion`, on `configs/final/rung{2,3,4}/reference_equal.json` (and `simpler_than_base`). Code: `tools/minicandi/dna_fusion.py` and the three rung modules.

| arm | what changes | conv autoencoder | conv + attention | U-Net |
|---|---|---|---|---|
| `fusion_film` | (gamma, beta) = Conv1d(d_dna, 2C, 1)(DNA); z x (1 + gamma) + beta, zero-initialized (identity at start). Same stage as the base. | after the stem; the stem reads counts only | after the track fusion, which reads counts only | the DNA bottleneck modulates the count bottleneck in place of cat + `fuse`; a 1x1 conv on the count bottleneck (`count_proj`, the count half of `fuse`) and the DNA skips stay |
| `fusion_late` | DNA joins at the bottleneck: the mean of each block of f = 16 bins, then a 1x1 Conv1d(32, C), added to the bottleneck features; the encoder reads counts only | yes | yes (before the self-attention) | not built: the base already joins at the bottleneck stage; a late-only arm would also have to remove the DNA skips, a second change |
| `fusion_xattn` | one residual pre-norm cross-attention layer: the count bottleneck tokens are queries, the DNA tokens of the same resolution are keys and values; the same sinusoidal positions on both | not built: the model has no attention, and one layer would add the global context that defines conv + attention | DNA tokens from the pooling of `fusion_late`; before the self-attention | DNA tokens = the DNA tower bottleneck; in place of cat + `fuse`; `count_proj` and the DNA skips stay |
| `fusion_early` | DNA features (width C0) join the count stem; no DNA tower, no `fuse`, no DNA skip | not built: the base join | not built: the base join | yes; this contrast changes three things: the stage, the DNA tower and the DNA skips |

Addition is not an arm on any architecture: a linear projection of a concatenation equals the sum of two linear projections, so "project and add" is the base's function class with another initialization.

Readings and choices.
- Stage on the conv models: `fusion_late` against the base (early concatenation). Operator at the bottleneck: `fusion_xattn` against `fusion_late` (conv + attention), both reading the same pooled DNA tokens. Operator at the input stage: `fusion_film` against the base.
- On the U-Net, film and cross-attention replace only the bottleneck join operator. The DNA skips and the count half of `fuse` stay, so each arm is one change. The skip source was run in the final round (lead 1).
- Trunk width rule: an arm differs from the base in the join only. Each rung's width search reads the config without `dna_fusion`, so every arm keeps the base width (60, 104, 25), and its parameter count differs from the base by the join's own parameters.
- `simpler_than_base`: `true` for `fusion_early` (one tower instead of two), `false` for the other three (each adds a layer to the base join).

Parameters (`registry.build_model`, seed 0, trainable). Each cell: count (width); change against the base of the same architecture; change against 1.0e6.

| architecture | base (width) | `fusion_film` | `fusion_late` | `fusion_xattn` | `fusion_early` |
|---|---|---|---|---|---|
| conv autoencoder | 1,042,514 (60); vs 1.0e6 +4.3% | 1,044,554 (60); vs base +2,040 (+0.2%); vs 1.0e6 +4.5% | 1,044,554 (60); vs base +2,040 (+0.2%); vs 1.0e6 +4.5% | not built | not built |
| conv + attention | 1,038,018 (104); vs 1.0e6 +3.8% | 1,041,554 (104); vs base +3,536 (+0.3%); vs 1.0e6 +4.2% | 1,038,122 (104); vs base +104 (+0.0%); vs 1.0e6 +3.8% | 1,082,218 (104); vs base +44,200 (+4.3%); vs 1.0e6 +8.2% | not built |
| U-Net | 993,599 (25); vs 1.0e6 -0.6% | 1,003,799 (25); vs base +10,200 (+1.0%); vs 1.0e6 +0.4% | not built | 1,024,399 (25); vs base +30,800 (+3.1%); vs 1.0e6 +2.4% | 750,149 (25); vs base -243,450 (-24.5%); vs 1.0e6 -25.0% |

U-Net `fusion_early` is outside the 10% band by design: it keeps the base width and drops the DNA tower, `fuse` and the DNA skip channels. The conv + attention `fusion_xattn` keeps the cross-attention at the trunk width (104, 4 heads); it stays inside 10% of 1.0e6, so its size was not reduced.

8 arm-architecture pairs x 3 seeds = 24 runs.

## 3. Prior leads

1. U-Net skip source (`results/final_round/FINAL_ROUND_EQUAL.md` lines 14, 16, 17, foreman's worktree; mean (range)): base variance recovered 0.432 (0.023), gap closed on crop pairs 0.507 (0.071), composite 0.495 (0.010). Skips carry DNA only: 0.380 (0.022), 0.734 (0.063), 0.533 (0.029). Skips carry counts only: 0.432 (0.006), 0.518 (0.037), 0.486 (0.033). Removing the DNA skips changed nothing beyond the range. Prediction: the U-Net's bottleneck join carries little; `fusion_early`, which also removes the DNA tower and skips, lands near the counts-only skip arm.
2. DNA off, pointwise MLP (`6_dna_off.md` lead 1, re-read on Nibi): variance recovered 0.3431 (0.0029) -> 0.3400 (0.0035); composite 0.4736 (0.0039) -> 0.4908 (0.0232); all within. Prediction: the DNA effect at this scale is small, so a fusion arm has little to redistribute.
3. Mappability (`miniCANDI design notes.md` L133, from t146 h31): "A one-hot DNA input gave about a quarter of the mappability gain", local to poorly mappable bins. Prediction: any fusion effect shows first on correction (crop pairs are mappability artifacts).
4. Max's error decomposition (`miniCANDI architecture ablations.md` L356): "The Flashzoi sequence input cut shared error by 0.126 and specific error by 0.025." A rich DNA input acts mostly on shared error. Prediction: a better join raises variance recovered (which is specific) little.
5. DNA off and DNA shifted by 5 Mb on the three models (ablations 6 and 5) were submitted on Fir (kit CM); on 2026-11-16 no seed of either had a `p16_sv/scores.json`.

## 4. Expectation (before any run)

Against the base of the same architecture, 3 seeds, `p16_sv`:

| arm | imputation | correction | calibration | composite |
|---|---|---|---|---|
| `fusion_film`, all three | within | within | within | within |
| `fusion_late`, conv models | within; at most -0.01 | within | within | within |
| `fusion_xattn`, conv + attention and U-Net | within | within | within | within |
| `fusion_early`, U-Net | within (near 0.43) | within | within | within |

The DNA tower is 0.3% of the conv models' parameters, and DNA off at the pointwise MLP moved nothing beyond the range. So the expected verdict is "the join does not matter at this scale".

## 5. How the result answers the question

"Beyond" means the difference of means is larger than the larger of the two seed ranges. Each row is read per architecture.

| outcome | conclusion |
|---|---|
| A. All arms within the range of the base on all three goals | The join does not matter at this scale. D13 keeps the simpler join: concatenation on the conv models; `fusion_early` (one tower) on the U-Net. |
| B. An arm beyond the base on composite on 2+ architectures, no goal lower beyond 2 x its range | That join enters M1. |
| C. `fusion_late` beyond the base (conv models) | DNA works better as context for the latent than as a per-bin input. |
| D. `fusion_xattn` beyond `fusion_late` (conv + attention) | The operator matters at the bottleneck: a content-based alignment of DNA helps. |
| E. `fusion_early` on the U-Net lower beyond the range | The base's DNA route (the stage, the separate DNA tower with 243,450 parameters, and the DNA skips, together) carries information that a stem join does not. This contrast cannot say which of the three parts matters; the final-round skip arms (lead 1) bound the skip part. |
| F. Any arm lowers correction beyond the range | The DNA join interferes with correction. Report it per architecture. |
| G. DNA off (ablation 6) within the range of the base on an architecture | That architecture's fusion arms are uninformative: there is no DNA effect to redistribute. |

## 6. Controls and traps

- Off-switch bit-identity. Without `dna_fusion` (absent or null), each rung builds the state_dict of `309d0f6` (names, shapes, values) and gives the same outputs to 0 difference. Checked by `tests/test_ablation_dna_fusion.py::test_without_the_key_the_model_is_the_pre_change_model`, which loads the rung modules from `git show 309d0f6` and compares them under `dna` own, off, shuffled and precomputed, and with `dna_fusion: null`. The reference counts 1,042,514 / 1,038,018 / 993,599 are unchanged.
- Width. Every arm keeps the base trunk width (unit test `test_arm_keeps_the_base_width_reads_dna_and_keeps_the_mask`). Training FLOPs per window stay within 1% of the base, except U-Net `fusion_early` (1.66e9 -> 1.40e9, no DNA tower).
- U-Net film and xattn keep `count_proj`, a learned 1x1 conv on the count bottleneck (the count half of the base's `fuse`), so the count path is the base's and only the join operator changes.
- Join stage. `test_arm_joins_dna_at_its_stage` captures the input of the module before the join and of the bottleneck, and checks which one changes when the DNA changes. `test_unet_bottleneck_join_is_live_without_dna_skips` checks the U-Net join with `skip_source: counts`. `test_join_layer_gets_gradient` checks a nonzero gradient on every join parameter.
- Initialization. The NB head's last layer starts at zero (`components.py` L510-511), so step 0 sends no gradient upstream of the head in any model. Under `fusion_film` the FiLM projection also starts at zero: it gets gradient first at step 1, and the DNA path behind it at step 2. So at initialization the conv models ignore the DNA under `fusion_film`; the U-Net still reads it through the DNA skips. This favors the base if 80,000 windows are too few to learn the modulation.
- Copy path. No arm changes the count path or the mask. The wiring check passes for every arm with perturbed weights (unit test), and the U-Net count skips are unchanged.
- Initialization pairing. Arms add or remove modules, so seed k of an arm does not share the base's initial weights. Judge on ranges.
- Scorer input. No change: the arms read the same `x_dna`. The scorer rebuilds the model from the saved config through `registry.load_run` (the smoke reloads every arm).
- Other keys. `dna_fusion` with `dna: off` is refused; rungs 1, 5, 6 refuse the key in `registry.build_model`; `fusion_early` with `skip_source: dna` is refused. `dna: precomputed` works under every arm (unit test), so the fusion and source ablations can be combined later.

## 7. Readiness

- Code: `tools/minicandi/dna_fusion.py` (new), `rungs/conv_ae.py`, `rungs/conv_attn_ae.py`, `rungs/unet.py`, `registry.py`. Tests: `tests/test_ablation_dna_fusion.py` (66 tests), `tests/test_ablation_configs.py` (8 new configs).
- Configs: `tools/minicandi/configs/ablations/dna_fusion/rung{2,3,4}/fusion_<arm>.json` and `arms.tsv` (the not-built pairs are rows with their reason). Each json differs from `reference_equal.json` in `dna_fusion` and `simpler_than_base` only.
- FLOPs per training window (`train.flops_per_window`, batch of 8, 1,024 bins), base -> arm: conv autoencoder 2.11e9 -> 2.12e9 (film), 2.10e9 (late); conv + attention 2.54e9 -> 2.56e9 (film), 2.52e9 (late), 2.53e9 (xattn); U-Net 1.66e9 -> 1.67e9 (film), 1.67e9 (xattn), 1.40e9 (early). The 03:00:00 training limit of `submit_all.sh` holds.
- Kit: COA11b on Fir (git 81cfbc3, after the review fixes); COA11 (git 49c45ee) is superseded. Smoke: section 8.
- Submission (the foreman's call), on a Fir login node: `K=/project/def-maxwl/mforooz/onion_kit/COA11b ABLATIONS=dna_fusion DRY_RUN=0 bash $K/tools/minicandi/configs/ablations/submit_all.sh`. Dry run (2026-11-16): 8 arms x architectures, 144 sbatch lines (24 runs x (1 train + 4 predict shards + 1 metrics)), 0 blocked. Run directories `runs_fir/abl_dna_fusion_fusion_<arm>_<convae|convattn|unet>/seed{0,1,2}`.
- Cost (per-seed train + predict + metrics, Fir `sacct`, `5_dna_source.md` section 7): conv autoencoder 26.89, conv + attention 23.85, U-Net 28.18 CPU-h. Total 3 x (2 x 26.89 + 3 x 23.85 + 3 x 28.18) = 3 x (53.78 + 71.55 + 84.54) = 3 x 209.87 = 629.6 CPU-h. This is an upper bound: U-Net `fusion_early` trains with 16% fewer FLOPs per window.
- Reference: the existing base runs `runs_fir/rung{2,3,4}_base_heldout` (no new run).

## 8. Smoke (kits COA11 and COA11b)

- Kit COA11b (after the review fixes), Fir job 4453946 (4 CPUs, 269 s = 0.30 CPU-h, exit 0). Each of the 8 arm configs trained 320 windows on the real `eic_v1` tables (seed 0) and reloaded through `registry.load_run` with the parameter counts of section 2. The logged losses were finite (first 1.553; last 1.325 to 1.326 for every arm). Outputs: `/project/def-maxwl/mforooz/ONION/smoke_COA11b/rung{2,3,4}_<arm>`.
- Kit COA11 (before the fixes, superseded), Fir job 4449584 (0.27 CPU-h, exit 0). Outputs: `/project/def-maxwl/mforooz/ONION/smoke_COA11/`.
- Notes: `cruxvault/results/closeout_A11/SMOKE.md` (git-ignored).
