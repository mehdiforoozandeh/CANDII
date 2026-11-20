# Ablation 12: mask token or dropped tracks (`mask_mode`)

Phase 1 specification, 2026-11-16. Close-out item A2, branch `chunk/onion-CO-A2`, base d65a980.

## 1. Question

The PI wrote (`miniCANDI architecture ablations.md`, "Ablations at every rung"): "Mask tokens, or dropping the missing tracks." D14 condition 1 needs it at every rung with 3 seeds.

Question: on the conv autoencoder, conv + attention and U-Net, does a learned mask token at hidden bins and missing tracks beat zeroing them (dropping)?

## 2. Arms

The base `mask_mode` of all three architectures is `token`. I read it in `runs_fir/rung{2,3,4}_base_heldout/seed0/run.json` on Fir (2026-11-16) and in `final/rung{2,3,4}/reference_equal.json`. So the arm is `drop`.

| arm | change from `final/rung{2,3,4}/reference_equal.json` | `simpler_than_base` |
|---|---|---|
| reference | none (`mask_mode: token`; runs `runs_fir/rung{2,3,4}_base_heldout`) | |
| `mask_drop` | `mask_mode: drop` | `true` |

What the switch does (`components.MaskHandler`): `token` puts a learned `hidden_token` at hidden bins and a learned `missing_token` at missing tracks. `drop` puts exact zeros there. The conv autoencoder in drop mode also re-zeroes invisible positions after the input injection and adds one visibility channel per track to its stem (`rungs/conv_ae.py` lines 183, 240 to 244). The U-Net appends a visibility flag in both modes (`rungs/unet.py` line 9).

`simpler_than_base: true` is my call (the `ladder_arms.py` label is null). Drop has no learned replacement vectors; the visibility flag is a fixed input. So by D13, a drop arm within the bars stays.

## 3. Prior leads

1. Pointwise MLP (base drops; arm `mask_token`), 2 seeds, composite version 2 (`results/ladder_audit/COVERAGE.md` row 16): composite +0.002, within; specific variance -0.004, beyond (seed ranges 0.002 and 0.002); gap closed +0.022, within. Prediction: no difference that matters.
2. Final round: no mask-mode arm ran on any larger model (`results/final_round/FINAL_ROUND_EQUAL.md`).

## 4. Expectation (before any run)

All four metrics within the seed ranges on all three architectures (reference ranges in `11_scrambled_covariates.md` section 4: variance recovered 0.012 to 0.028, gap closed 0.012 to 0.071, calibration 0.108 to 0.178, composite 0.010 to 0.048). A possible exception is conv + attention, which has no visibility flag in drop mode (section 6).

## 5. How the result answers the question

"Beyond" = difference of seed means larger than the larger seed range (3 seeds). Judge each architecture separately.

| outcome | conclusion |
|---|---|
| `mask_drop` within on all four metrics, on an architecture | by D13 the simpler arm (drop) stays for that architecture |
| `mask_drop` lower, beyond, on any term | the learned token carries information (hidden or missing is not the same as zero); keep `token` |
| `mask_drop` higher, beyond | drop wins; adopt it in M1 for that architecture |
| verdicts differ across architectures | the choice is architecture specific; M1 records one per architecture |

## 6. Controls and traps

- **How a dropped position stays distinct from an observed one.** The drop replaces the per-track feature vector after the lift layer, not the count. The conv autoencoder and the U-Net also get a visibility flag. Conv + attention gets no flag and no attention key mask in drop mode (`rungs/conv_attn_ae.py` lines 185 to 209); its only cue is that the zero vector differs from any lifted value (the lift has a bias). So a conv + attention loss in this arm can come from the missing flag, not from the token itself. State this when the result is read.
- Parameter counts (`registry.build_model`, seed 0): conv autoencoder 1,042,514 -> 1,044,642 (+2,128: the visibility channels); conv + attention 1,038,018 -> 1,037,986 (-32: the two tokens); U-Net 993,599 -> 993,583 (-16). Trunk widths are unchanged (60, 104, 25).
- Off switch: no code changed.
- The scorer reads the same inputs; `mask_mode` changes the model only.

## 7. Readiness

Ready. Configs `tools/minicandi/configs/ablations/mask_mode/rung{2,3,4}/mask_drop.json` and `arms.tsv`; the json diff against `reference_equal.json` shows `mask_mode` and `simpler_than_base` only (`tests/test_ablation_configs.py`). Laptop step time within 6% of the reference (`cruxvault/results/closeout_A2/step_timing_laptop.txt`). Training limit 03:00:00.

Cost (per-run figures of `7_cov_embedding.md` section 7): 3 seeds x 87.4 = 262.2 CPU-h, 9 runs, 54 jobs.
