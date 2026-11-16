---
title: "A8: count transforms"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# A8: count transforms

## 1. What the PI asked

- `miniCANDI architecture ablations.md`:98: "These are candidate transforms for the count inputs. Each one lists how it removes depth on the input side and how depth is added back on the output side. Only the size-factor offset makes depth an exact additive shift in η."
- `miniCANDI architecture ablations.md`:149: "Under all seven transforms, the noise around the normalized value still depends on depth (relative variance 1/μ + 1/r). The encoder therefore still needs d as a covariate."
- `long-horizon decisions.md`:18 (D1): the selection rule settles "the outcome of each miniCANDI ablation, e.g. rung, DNA source, covariate injection, depth offset, loss or preprocessing."
- PI directive, plan Log 2026-11-15T20:05Z: "every knob and ablation he named must be tested with a lead."

Success criteria served: 2 (the input side of the objective) and 4 (design settled). Audit row: D22 (flagged with a doubt, because the transforms sit in a file marked partly background; the PI selected the item on 2026-11-16, which removes the doubt).

## 2. Question

Does any of the listed input transforms (log1p, asinh, Anscombe, NB Pearson residuals, local-background ratio, quantile, raw, soft clip), or a smaller pseudocount, beat the default size-factor offset on the three goals?

## 3. State today

- Default input transform (`tools/minicandi/components.py` L387-388): `offset`, x = log2((y + 1)/s), s = 2^(log2 depth - 24); output mu = s x 2^eta; pseudocount `offset_eps` 1.0.
- Arm keys exist (`tools/minicandi/stage2_arms.py` L69-77): `tf_log1p`, `tf_asinh`, `tf_anscombe`, `tf_pearson`, `tf_local_ratio`, `tf_quantile`, `tf_raw`, `tf_soft_clip`, `offset_eps0.1`; 9 arms.
- No arm trained (`COVERAGE.md` row 30).

## 4. Prior leads

- Depth arms at the U-Net (`COVERAGE.md` rows 27-28; `VIEWS_ABLATION.md`): hiding depth from the network cost 0.184 of composite; hiding it on the input side only cost 0.062 (beyond the range). Prediction: transforms that remove depth from the input values (Pearson residuals, local ratio, quantile) hurt unless depth stays a covariate, as line 149 says.
- Builder caveat (Log 2026-11-15T07:52Z): with the offset transform, a zero-count visible bin gives x = 24 - log2 depth, so the input value itself encodes depth. Prediction: `offset_eps0.1` changes how much depth leaks through zero bins.
- `miniCANDI architecture ablations.md`, sections "pseudocount-sweep", "borzoi-soft-clip", "pearson-residuals-details" (background): read before the run.
- Ablation 8 of A1 (`input_depth_fixed`, depth-controlled input) runs now; its result predicts the direction for the depth-removing transforms.

## 5. Work

1. PI decision on scope: all 9 arms on 3 architectures (81 runs), or a two-stage screen (stage 1: all 9 arms on one architecture; stage 2: arms within 2 x the seed range of the default, or beyond it, on the other two).
2. Write the arm configs on `final/rung{2,3,4}/reference_equal.json` with `stage2_arms.py write`; check parameter counts.
3. Submit on Fir CPU, 3 seeds, scored into `p16_sv`.
4. Read against the reference; log D13 outcomes; record findings.

## 6. How the result answers the question

"Beyond" = difference of seed means larger than the larger seed range (max minus min, 3 seeds).

| outcome | conclusion |
|---|---|
| a transform beyond the offset on composite, 2+ architectures, no goal down beyond 2 x its range | the transform replaces the offset in M1 (D13) |
| all within the range | the offset stays (it is the only exact additive form, line 98) |
| depth-removing transforms worse beyond the range | depth must reach the encoder as a value or covariate; consistent with the depth arms |

## 7. Dependencies, cost, risks

- Depends on nothing; reads better after ablation 8 of A1.
- Cost: per-seed cost 32.5 / 22.4 / 32.3 CPU-h (`ablation_specs/1_control_input.md` section 7). All 9 arms on 3 architectures: 9 x 262 = 2,358 CPU-h. Two-stage: stage 1 on the conv autoencoder 9 x 3 x 32.5 = 878 CPU-h, then 3 x (22.4 + 32.3) = 164 CPU-h per arm carried to the other two architectures.
- Risk: `tf_local_ratio` and `tf_quantile` have no output-side inverse (ablations file L138, L143); check how the arm decodes the output before training.

## 8. Done when

The chosen arms are scored on their architectures with 3 seeds, and each transform's D13 outcome against the offset default is logged in the notepad.
