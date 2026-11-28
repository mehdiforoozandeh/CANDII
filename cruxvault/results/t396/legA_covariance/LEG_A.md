# t396 leg A: covariance term in the start-to-count conversion

Hand-written report of chunk C15. The numbers come from `table.tsv` and `rho.tsv` (outputs of
`run.sh`, i.e. `tools/t396/calib_cov.py table` on Fir over `records_seed{0,1,2}.jsonl` of the
three fit-then-score chains in `jobs.tsv`, all COMPLETED). Code: kit SC7b, hashes equal to
`d03fff03`.

## Scope

- Models: the read-start model `rs_halves_mix_avg_pearson_unet` and the read-count reference
  `rcp_halves_mix_avg_pearson_unet`, seeds 0, 1, 2, regime `leave_one_assay_out`.
- Targets: 43 (22 assays). 11 assays have one target; they use the fallback rule of
  `calib_cov.py` in variants (ii), (iv) and (vi).
- Regions: the odd regions of `evalset_p16.json` (128) fit `rho` and the scalar `c`. The even
  regions (128, with the 16 pair regions) are scored. The two sets are disjoint.
- Metric: `deviance.calibration_ratio` = mean over targets of `D_model / D0` over the same mean
  of `E_nb / D0`. A ratio of 1 is calibrated. A ratio above 1 says the NB is too narrow.
- Macro: `all_assays` = mean over the 22 assays of the per-assay ratio; `scorable` = the same
  over the 9 scorable assays of `scores.json`.
- Noise: the seed range is max - min over the 3 seeds.
- Reproduction check: variant (i) recomputed with `identity_rho` matched the stored `n` on all
  129 target-seed pairs (max relative difference 8.2e-7).

## Ratios on the even regions, mean over 3 seeds

| variant | all_assays (seed range) | scorable (seed range) |
|---|---|---|
| (i) read-start, independent conversion as stored | 1.258 (0.050) | 1.303 (0.098) |
| (ii) read-start, covariance term, `rho` leave-one-target-out | 1.032 (0.025) | 1.044 (0.060) |
| (iii) read-start, covariance term, `rho` in-sample | 1.055 (0.015) | 1.052 (0.041) |
| (iv) read-start, independent conversion, `n / c` leave-one-target-out | 1.050 (0.031) | 0.975 (0.024) |
| (v) reference as stored | 1.064 (0.032) | 1.064 (0.038) |
| (vi) reference, `n / c` leave-one-target-out | 1.062 (0.048) | 0.996 (0.023) |

## Gap between (i) and (v) that (ii) closes

- all_assays: (1.258 - 1.032) / (1.258 - 1.064) = 0.226 / 0.194 = 1.16.
- scorable: (1.303 - 1.044) / (1.303 - 1.064) = 0.259 / 0.239 = 1.09.

The covariance term closed all of the gap to the stored reference in both macros. The gap is
4 to 8 times the seed range of (ii) (0.194 / 0.025; 0.239 / 0.060), so the effect is above seed
noise. The overshoot past the reference (0.032 all_assays, 0.021 scorable) is inside the seed
ranges of (ii) and (v).

## Like-for-like pair: (ii) against (vi)

Both fit on the odd regions and exclude the scored target.

- all_assays: (ii) - (vi) = 1.032 - 1.062 = -0.029. Per seed: -0.066, -0.005, -0.017. The
  difference is inside the seed range of (vi) (0.048).
- scorable: (ii) - (vi) = 1.044 - 0.996 = +0.047. Per seed: +0.010, +0.042, +0.090. The
  difference is inside the seed range of (ii) (0.060).
- Distance from 1: all_assays 0.032 for (ii) and 0.062 for (vi); scorable 0.044 for (ii) and
  0.004 for (vi).

Under the same recalibration, the converted read-start model and the reference are not
separable at 3 seeds. The sign of the difference changes with the macro. On the 9 scorable
assays, (ii) is above (vi) on every seed, and per assay (ii) is above 1.05 on H3K27ac (1.06),
H3K36me3 (1.07), H3K4me3 (1.10) and H3K9ac (1.15).

## Over-correction check (the C9 caveat)

The residuals `y - mu` of `estimate_rho` include the mean error of the model (finding f156), so
`rho` can over-correct.

- `lag0_raw` (same-strand lag-0 `num / den`, forced to 1 in `rho`) has a median of 3.79 over the
  66 assay-seed pairs (min 0.97, max 22.5). The residual variance thus exceeds the model's own
  excess variance by a factor of about 4. The off-lag `rho` can carry part of that size error.
- The same-strand lag-1 `rho` has a median of 0.67 (min 0.17, max 1.0, clipped).
- Per assay (mean over seeds), 10 of 22 assays fall below 1 under (ii), against 3 under (i). The
  scalar recalibrations (iv) and (vi) also put 10 of 22 below 1. The lowest assay under (ii) is
  at 0.77; the lowest under (vi) is at 0.84.
- The in-sample variant (iii) is not an upper bound in effect: it is farther from 1 than (ii) in
  both macros (1.055 against 1.032; 1.052 against 1.044), but both differences are inside the
  seed ranges. The 11 fallback assays use the mean `rho` of other assays in (ii) and their own
  `rho` in (iii), which can explain the order.

The macro ratios of (ii) stay above 1, so the covariance term did not over-correct on average.
It over-corrects some single assays about as often as the scalar `n / c` does.

## Earlier numbers, with their scope

- 1.31: read-start calibration ratio as scored (t392, 3 seeds, all regions). Variant (i) here
  gives 1.303 (scorable) on the even regions only.
- 0.875: read-start ratio after `n / c` (t395, finding f156, seed 0 only, in-sample `c`). Variant
  (iv) here uses a leave-one-target-out `c` fitted on disjoint regions and gives 0.975 (scorable)
  and 1.050 (all_assays).
- 1.06: reference ratio as scored. Variant (v) here gives 1.064 in both macros.

## Not done

- The native start-space ratio of the read-start model (`native_start.tsv`) was not written. The
  leg B table did not exist on Fir at collection time (2026-11-28).
- No bootstrap over targets. The only noise measure is the seed range over 3 seeds.
