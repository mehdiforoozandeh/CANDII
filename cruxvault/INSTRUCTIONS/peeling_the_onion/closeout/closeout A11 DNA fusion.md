---
title: "A11: DNA fusion"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# A11: DNA fusion

## 1. What the PI asked

- `peeling the onion.md`:155: "how we embed and fuse different modalities i.e. epigenomic count data, dna sequence, and covariate information is an important an non trivial part of the whole architecture design. these are targets for ablation i.e. covariate embedding and covariate injection, DNA embedding and injection, etc."
- `miniCANDI architecture ablations.md`:173-177, "design-choices-to-ablate": "DNA–signal fusion. Options: 1. Concatenate DNA features with count features before the trunk. 2. Add DNA as one more token row in axial attention (rung 5). 3. Use DNA as a FiLM conditioner on count features (Perez et al. 2018). 4. Feed DNA tokens as extra inputs to a Perceiver latent (Jaegle et al. 2022)."
- `miniCANDI architecture ablations.md`:45 (rung 4): "Skips from the DNA tower, the count tower, or both are the three arms."
- PI ruling 2026-11-16 (`closeout todo.md`:96): "DNA fusion is a separate ablation on the 3 models (item A11)." Binding rulings: conv autoencoder, conv + attention, U-Net; base recipes; 3 seeds; 80,000 windows; no matched-compute arms.

Success criterion served: 4 (design settled and tested). Audit row: D13 (A1 file line 28: "DNA injection ... has no arm").

## 2. Question

With the DNA source fixed (own tower), does the operator that joins the DNA features to the count features (concatenation, FiLM, cross-attention) or the stage at which they join (input or bottleneck) change imputation, correction or calibration?

## 3. State today (read from the code, umbrella 309d0f6)

- Conv autoencoder (`rungs/conv_ae.py` L183-185, L245-248): early concatenation. The per-bin DNA features (32 channels) join the flattened track features, and the 1x1 stem conv projects them.
- Conv + attention (`rungs/conv_attn_ae.py` L155-156, L191-195): early concatenation into the track-fusion Linear.
- U-Net (`rungs/unet.py` L210-213, L289-294, L326-327): late concatenation. A DNA tower of its own is concatenated with the count bottleneck and projected by a 1x1 conv (`fuse`); the same tower feeds the DNA skips (`skip_source: both`).
- No fusion arm existed: a search of `tools/minicandi` for fusion, fuse and inject found only the base joins and the rung-4 `skip_source` key.
- Addition is not a separate arm. A linear map of a concatenation equals the sum of two linear maps, so "project and add" is the base's own function class.

## 4. Prior leads

- U-Net skip source (`results/final_round/FINAL_ROUND_EQUAL.md` lines 14, 16, 17, foreman's worktree; mean (seed range)): base variance recovered 0.432 (0.023), gap closed on crop pairs 0.507 (0.071), composite 0.495 (0.010). Skips carry DNA only: 0.380 (0.022), 0.734 (0.063), 0.533 (0.029). Skips carry counts only: 0.432 (0.006), 0.518 (0.037), 0.486 (0.033). Removing the DNA skips changed nothing beyond the range; the DNA-only gain comes from removing the count copy path. Prediction: a change at the U-Net's DNA join moves little while the count skips stay.
- DNA off, pointwise MLP (`results/ladder_audit/COVERAGE.md` row 10; re-read in `ablation_specs/6_dna_off.md` lead 1): variance recovered 0.3431 (0.0029) -> 0.3400 (0.0035), composite 0.4736 (0.0039) -> 0.4908 (0.0232), all within. Prediction: DNA carries little at this scale.
- Mappability (`miniCANDI design notes.md` L133): "A one-hot DNA input gave about a quarter of the mappability gain", local to poorly mappable bins. Prediction: if fusion matters, it shows on correction.
- Now running (Fir): DNA off (`ablation_specs/6_dna_off.md`) and DNA shifted by 5 Mb (`5_dna_source.md`) on the three models; no seed scored yet (2026-11-16). Their result sets how much DNA effect a fusion arm can redistribute. This item adds no DNA-off or DNA-source arm.

## 5. Work

Specification and readiness: `run_state/ablation_specs/16_dna_fusion.md`. Arms (`tools/minicandi/dna_fusion.py`, key `dna_fusion`):

| arm | conv autoencoder | conv + attention | U-Net |
|---|---|---|---|
| `fusion_film` (DNA scales and shifts the count features) | at the stem output | at the track-fusion output | at the bottleneck, in place of cat + `fuse` |
| `fusion_late` (DNA joins at the bottleneck by a learned patch pooling, added) | yes | yes | not built: the base join |
| `fusion_xattn` (count tokens cross-attend to DNA tokens at the bottleneck) | not built: no attention in the model | yes | yes, in place of cat + `fuse` |
| `fusion_early` (DNA into the count stem; no DNA tower, no DNA skip) | not built: the base join | not built: the base join | yes |

8 arm-architecture pairs x 3 seeds = 24 runs, Fir CPU, `ABLATIONS=dna_fusion` through `submit_all.sh` with kit COA11 or later.

## 6. How the result answers the question

"Beyond" = difference of seed means larger than the larger seed range (max minus min, 3 seeds).

| outcome | conclusion |
|---|---|
| all arms within the range of the base, on all three goals | the join does not matter at this scale. D13 keeps the simpler join: concatenation on the conv models, `fusion_early` (one tower) on the U-Net |
| an arm beyond the base on composite on 2+ architectures, no goal lower beyond 2 x its range | that join enters M1 |
| `fusion_late` within `fusion_film` and the base, `fusion_xattn` beyond `fusion_late` (conv + attention) | the operator matters at the bottleneck: content-based alignment of DNA helps |
| `fusion_early` on the U-Net lower beyond the range | the separate DNA tower and its skips carry information that the count path does not |
| any arm lowers correction beyond the range | the DNA join interferes with correction; report per architecture |
| DNA off within the range of the base on an architecture | read that architecture's fusion arms as uninformative (no DNA effect to redistribute) |

## 7. Dependencies, cost, risks

- Depends on nothing. Reads better after DNA off and DNA shifted by 5 Mb are scored.
- Cost (per-seed train + predict + metrics on Fir CPU, `5_dna_source.md` section 7): 26.89 / 23.85 / 28.18 CPU-h. 3 x (2 x 26.89 + 3 x 23.85 + 3 x 28.18) = 3 x 209.87 = 629.6 CPU-h; 144 jobs.
- Risk: width refit. `fusion_late` takes width 60 -> 56 (conv autoencoder) and 104 -> 96 (conv + attention); the U-Net `fusion_early` takes 25 -> 30. A small effect can come from the width.
- Risk: FiLM is zero-initialized (identity), so the DNA tower gets no gradient at step 0 on the conv models; it gets one after the first update.

## 8. Done when

The 24 runs are scored into `p16_sv`, each arm has a D13 outcome per architecture against the base in the notepad, and the outcome row of section 6 is recorded as a finding.
