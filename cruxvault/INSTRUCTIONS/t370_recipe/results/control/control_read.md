# Control read (plan chunk C21, gate G5)

Written 2026-11-22 by the t370 foreman. Sources: `control_summary.tsv` (from `tools/onion_eval/control_eval.py summary --runs raw`), `raw/<run>/seed<s>/control_eval.json` (pulled from Fir `runs_rcp/control_eval/`), and `arms_p16_sv.tsv` (arm summary, eic_v1_2, correction on 16 regions). Three seeds per arm-model. The held-out group is the 8 cells whose control was never an input or a target in training (`control_holdout_cells`).

Terms: held-out skill = `crps_skill_vs_training_average` (CRPS of the training-cell average minus CRPS of the predicted NB; positive = better than the average). Signal = Pearson correlation of arcsinh signal p-values, predicted vs observed (`pred`), and training average vs observed (`avg`).

## Held-out values, mean (seed range)

| arm | model | held-out skill | signal pred | signal avg |
|---|---|---|---|---|
| shared head, weight 1 | conv AE | 0.0216 (0.0189) | 0.9208 (0.0074) | 0.9315 (0) |
| shared head, weight 1 | conv + attn | -0.0199 (0.0228) | 0.9101 (0.0244) | 0.9315 (0) |
| shared head, weight 1 | U-Net | 0.1308 (0.0169) | 0.9476 (0.0096) | 0.9315 (0) |
| weight 4 | conv AE | 0.0870 (0.0387) | 0.9287 (0.0168) | 0.9315 (0) |
| weight 4 | conv + attn | 0.0733 (0.0354) | 0.9269 (0.0249) | 0.9315 (0) |
| weight 4 | U-Net | 0.1151 (0.0234) | 0.9470 (0.0107) | 0.9315 (0) |
| separate head | conv AE | 0.0322 (0.0371) | 0.9234 (0.0129) | 0.9315 (0) |
| separate head | conv + attn | -0.0180 (0.0165) | 0.9129 (0.0153) | 0.9315 (0) |
| separate head | U-Net | 0.1338 (0.0175) | 0.9397 (0.0038) | 0.9315 (0) |

In-sample skill of the base runs (controls were training targets): conv AE 0.0399, conv + attn 0.0535, U-Net 0.0716.

## Gate G5

Rule 1, works: skill mean minus range > 0, and signal pred > signal avg beyond the range.
- U-Net, shared head: 0.1308 - 0.0169 = 0.1139 > 0; 0.9476 - 0.9315 = 0.0161 > 0.0096. **Works.**
- conv AE: 0.0216 - 0.0189 = 0.0027 > 0, but signal 0.9208 < 0.9315. Fails.
- conv + attn: skill below 0. Fails.

Rule 2, adoption (raises skill over the shared head beyond the range on 2+ models; not worse beyond the range on any of the three Pareto terms on any model):
- Weight 4: conv AE +0.0654 > 0.0387; conv + attn +0.0932 > 0.0354; U-Net -0.0157, inside 0.0234. Three terms (16 regions) vs shared head: no drop beyond range on any model (largest: conv + attn calibration -0.029, range 0.125). **Adopted.**
- Separate head: conv AE +0.0106 < 0.0371; conv + attn +0.0019; U-Net +0.0030. Not adopted.

Caveat: the adoption is carried by the two conv models. On the U-Net, which is the recipe's model, weight 4 does not raise held-out skill (-0.0157, inside its seed range). These arms predate the read-disjoint pairs.

G5 outcome: works
