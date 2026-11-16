# Ablation 10: adaLN-zero covariate injection (`cov_injection`)

Written 2026-11-16 on branch `chunk/onion-AB2`. The PI ruled on 2026-11-16 that adaLN-zero injection is wanted as an extra arm (`run_state/plan.md`, Log entry 2026-11-16T22:05Z, ruling 3).

## 1. Question

The PI wrote (`peeling the onion.md`, "# miniCANDI", important notes): "how we embed and fuse different modalities i.e. epigenomic count data, dna sequence, and covariate information is an important an non trivial part of the whole architecture design. these are targets for ablation i.e. covariate embedding and covariate injection, DNA embedding and injection, etc."

This ablation asks the injection half for one form: does the adaLN-zero form, i.e. a zero-initialized gated modulation of layer-normalized features, change imputation, correction or calibration when it replaces FiLM? `miniCANDI architecture ablations.md` lists adaLN-Zero as option 3 of "Covariate injection", beside concatenation, FiLM and tokens. The embedding half is ablation 7 (`7_cov_embedding.md`).

## 2. Arms

Today's injection (`tools/minicandi/components.py`, `CovariateInjector`, mode `film`): gamma, beta = G(emb(c)); h' = gamma * h + beta, with gamma = 1 and beta = 0 at initialization. The adaLN-zero mode (same class, mode `adaln_zero`, already in the code): shift, scale, gate = G0(emb(c)), all 0 at initialization; h' = h + gate * ((1 + scale) * LayerNorm(h) + shift), with a LayerNorm without affine parameters. Both forms are the identity at initialization. The covariate embedding is the same in both.

Each of the three architectures has two injection sites, one for C on the input side and one for C' on the output side (read from the built models):

| architecture | sites | feature width at each site |
|---|---|---|
| conv autoencoder | `inj_in`, `inj_out` | 16, 16 |
| conv + attention | `cov_in`, `cov_out` | 16, 16 |
| U-Net | `in_inj`, `out_inj` | 8, 32 |

| arm | config change from `final/rung<r>/reference_equal.json` | compared with |
|---|---|---|
| reference (base recipe, FiLM) | none | existing runs `runs_fir/rung{2,3,4}_base_heldout` |
| `inject_adaln_zero` | `covariate_injection: adaln_zero` | reference |

The arm runs on conv autoencoder, conv + attention and U-Net, with 3 seeds each.

## 3. Prior leads

1. Concatenation against FiLM (`cruxvault/results/ladder_audit/COVERAGE.md` row 13, rung 1, 2 seeds): composite +0.007, within the seed ranges; gap closed +0.028, beyond; specific variance -0.003, within; uncertainty +0.006, within. Prediction: the form of injection has a small effect at this budget.
2. Tokens against FiLM (same file, row 14, rung 1, seed 0 only): composite +0.004, not judged. Prediction: as lead 1.
3. DiT (`miniCANDI architecture ablations.md` line 293): adaLN-Zero gave a lower FID than in-context conditioning and cross-attention in image diffusion. The setting (image generation, large transformers, conditioning on a class label) does not transfer to a 1.0e6-parameter count model with two injection sites. This lead gives no number for this question.
4. No miniCANDI run has used `adaln_zero` (COVERAGE.md row 25: untested).

## 4. Expectation (before any run)

| metric | `inject_adaln_zero` vs reference |
|---|---|
| imputation (variance recovered) | within range |
| correction (gap closed) | within range; a change of the size of lead 1 (about 0.03) is possible in either direction |
| calibration | within range (the calibration seed ranges are 0.108 to 0.178, `7_cov_embedding.md` section 4) |
| composite | within range |

## 5. How the result answers the question

"Beyond" means the difference of means is larger than the larger of the two seed ranges. Judge each architecture separately; a conclusion holds for all three only when the three agree.

| outcome | conclusion |
|---|---|
| all four metrics within range on all three architectures | The injection form is not a sensitive part of the design at this budget. Keep FiLM, which is the simpler form and the incumbent. |
| correction or composite higher, beyond, and no metric lower, beyond | adaLN-zero injection is better. It is a candidate for the design. |
| imputation or correction lower, beyond | The normalized, gated form loses information that FiLM keeps. Keep FiLM. |
| mixed: one metric higher and one lower, beyond | Read the composite. Report the trade, and do not change the design on one architecture alone. |

## 6. Controls and traps

- **Parameter count.** adaLN-zero regresses 3 vectors per site in place of 2, so G0 has one more output block per site. The counts rise by 2,080 (conv autoencoder), 2,080 (conv + attention) and 2,600 (U-Net). The trunk widths are unchanged, because these differences are the size of the extra output blocks only. All three counts are within 10% of 1.0e6 (section 7).
- **Identity at initialization.** Both forms start as the identity. With the gate at 0, only the gate gets a gradient on the first step; shift and scale start to learn after the gate moves from 0. A short run can therefore look like "no covariates". The arm trains for the same 80,000 windows as the reference.
- **LayerNorm on narrow features.** At the U-Net input site the feature width is 8. A LayerNorm over 8 numbers removes the mean and the scale of h before the modulation, and only the residual path keeps them. A loss of imputation on the U-Net alone can come from this. Report it beside the result.
- **Depth path.** The count transform and the NB head read depth from c_out_t, outside the injector. The arm does not change that path.
- **No change to data or scorer.** The arm changes no dataset key and no evaluation key.

## 7. Readiness

Ready on all three architectures. No code was changed: `covariate_injection: adaln_zero` was already accepted (`components.py` `INJECTION_MODES`), and `stage2_arms.py` already had the arm `inject_adaln_zero`. The configs `tools/minicandi/configs/ablations/cov_injection/rung{2,3,4}/inject_adaln_zero.json` and `arms.tsv` were written with `stage2_arms.py write --arms inject_adaln_zero --incumbent tools/minicandi/configs/final/rung<r>/reference_equal.json`. The key diff against the reference is `covariate_injection` (film -> adaln_zero) and `simpler_than_base` (false). `tests/test_ablation_configs.py` checks the key diff and the parameter band.

Parameter counts from `registry.build_model` (seed 0):

| architecture | reference (FiLM) | `inject_adaln_zero` |
|---|---|---|
| conv autoencoder | 1,042,514 | 1,044,594 |
| conv + attention | 1,038,018 | 1,040,098 |
| U-Net | 993,599 | 996,199 |

Submission: `ABLATIONS=cov_injection bash submit_all.sh` (dry run by default). The training limit is the default 03:00:00 of the script, as for the other one-key arms; the extra cost per forward pass is two LayerNorms over narrow features.

Cost: the per-run cost is the same as for the arms of ablation 7, which also change only the covariate path (`7_cov_embedding.md` section 7: 87.39 CPU-h for one seed of the three architectures). One arm, 3 seeds, 9 runs: 3 x 87.39 = 262.2 CPU-h. The reference runs exist and cost nothing.
