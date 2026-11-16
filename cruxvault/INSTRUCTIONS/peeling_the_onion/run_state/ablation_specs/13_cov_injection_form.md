# Ablation 13: covariate injection form (`cov_injection_form`)

Phase 1 specification, 2026-11-16. Close-out item A2, branch `chunk/onion-CO-A2`, base d65a980.

## 1. Question

The PI wrote (`miniCANDI architecture ablations.md`, "Ablations at every rung"): "Covariate injection: concatenation, FiLM, or tokens." The constitution (`peeling the onion.md`, "# miniCANDI", important notes) names "covariate embedding and covariate injection" as targets for ablation.

Question: on the conv autoencoder, conv + attention and U-Net, does the form in which C and C' enter the network (FiLM, concatenation, tokens) change imputation, correction or calibration? Ablation 7 (`7_cov_embedding.md`) holds the embedding half. The adaLN-zero form is built by another agent (branch `chunk/onion-AB2`) and is not part of this ablation.

## 2. Arms

FiLM is the base of all three (`covariate_injection: film` in `runs_fir/rung{2,3,4}_base_heldout/seed0/run.json`, read on Fir 2026-11-16).

| arm | change from `final/rung{2,3,4}/reference_equal.json` | architectures | `simpler_than_base` |
|---|---|---|---|
| reference | none (FiLM: gamma, beta = G(emb(c)); h' = gamma h + beta) | 3 | |
| `inject_concat` | `covariate_injection: concat` (h' = Linear(cat(h, emb(c)))) | 3 | `true` |
| `inject_tokens` | `covariate_injection: tokens` (one token per track, emb(c) + a learned track embedding, joins the attention sequence) | conv + attention, U-Net | `false` |

`inject_tokens` does not apply to the conv autoencoder: it has no attention that reads tokens, and `registry.build_model` raises `NotApplicable` (checked by `tests/test_ablation_configs.py::test_inject_tokens_is_not_applicable_on_the_conv_autoencoder`). Both `simpler_than_base` labels follow `ladder_arms.simpler`.

Where tokens enter. Conv + attention: the C tokens of available tracks and the C' tokens of requested targets join the coarse sequence of its transformer, with a key padding mask for absent tracks; FiLM and concatenation act per track before the trunk. U-Net: the tokens join its bottleneck attention only (`rungs/unet.py` lines 296 to 315). So under tokens the covariates reach the U-Net at its coarsest resolution only.

## 3. Prior leads

1. Pointwise MLP, concatenation against FiLM, 2 seeds, composite version 2 (`results/ladder_audit/COVERAGE.md` row 13): composite +0.007, within; gap closed +0.028, beyond (seed ranges 0.017 and 0.023).
2. Pointwise MLP, tokens, 1 seed (`COVERAGE.md` row 14): composite +0.004, not judged.
3. Final round, FiLM as an arm where tokens are the base: axial transformer 0.521 against 0.517 (1 seed); Perceiver IO 0.395 against 0.335, seed range 0.088 (`results/final_round/FINAL_ROUND_EQUAL.md`).
4. Ablation 7, prior lead 4: the covariate path is not sensitive to its form at this budget.

Prediction: no difference beyond the seed range at the larger models either.

## 4. Expectation (before any run)

All four metrics within the seed ranges on all architectures (reference ranges in `11_scrambled_covariates.md` section 4). One exception is possible: U-Net tokens could lose correction, because C' reaches only the bottleneck and the per-bin output layers no longer see it directly; expected gap closed 0 to -0.07 (within the U-Net range of 0.071).

## 5. How the result answers the question

"Beyond" = difference of seed means larger than the larger seed range (3 seeds). Judge each architecture separately.

| outcome | conclusion |
|---|---|
| concatenation within FiLM on all four metrics, on 2+ architectures | by D13 the simpler arm (concatenation) stays; M1 records the change |
| concatenation or tokens beyond FiLM (higher) on 2+ architectures | change the injection form in M1 |
| concatenation or tokens lower, beyond, on correction | the per-feature gain of FiLM matters for correction; keep FiLM |
| tokens lower on correction at the U-Net only | bottleneck-only injection is too coarse for C'; keep a per-bin form at the U-Net |
| all within | the form is not a lever at 80,000 windows; keep the simplest (concatenation) |

## 6. Controls and traps

- **FiLM is identity at initialization; concatenation is not.** The two start from different functions. At 80,000 windows this can show as an optimization difference, not a capacity difference.
- Parameter counts (`registry.build_model`, seed 0), reference -> concat -> tokens: conv autoencoder 1,042,514 -> 1,027,378 -> n/a; conv + attention 1,038,018 -> 1,022,882 -> 1,037,346; U-Net 993,599 -> 978,263 -> 996,119. Trunk widths are unchanged (60, 104, 25). All within 10% of 1.0e6.
- Speed: laptop step time of concatenation is 1.13 to 1.42 x the reference (two measurements, `cruxvault/results/closeout_A2/step_timing_laptop.txt`). The slowest Fir reference training (conv autoencoder, 5,691 s) x 1.42 = 8,081 s, under the 03:00:00 (10,800 s) limit. Tokens are not slower (0.76 to 0.98 x).
- Off switch: no code changed.

## 7. Readiness

Ready. Configs `tools/minicandi/configs/ablations/cov_injection_form/rung{2,3,4}/inject_concat.json`, `rung{3,4}/inject_tokens.json`, and `arms.tsv` per architecture (rung 2 lists `inject_tokens` as `not_applicable`). The json diff against `reference_equal.json` shows `covariate_injection` and `simpler_than_base` only.

Cost (per-run figures of `7_cov_embedding.md` section 7): concatenation 3 seeds x 87.4 = 262.2 CPU-h; tokens 3 x (23.56 + 34.85) = 175.2 CPU-h; total 437.4 CPU-h, 15 runs, 90 jobs. With the concatenation slowdown on training and prediction (conv autoencoder 1.42, conv + attention 1.33, U-Net 1.16 at most), add at most 3 x (20.25 x 0.42 + 15.02 x 0.33 + 25.74 x 0.16) = 3 x 17.6 = 53 CPU-h.
