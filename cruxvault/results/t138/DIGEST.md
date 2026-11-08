# t138 — where the split-half disagreement concentrates (digest)

Exploratory readings, not judged. Source: `map_classes.tsv`, `map_products.tsv`, `map_context.tsv`,
`map_deciles.tsv`. Nibi run: see `NIBI_PATH.txt` (job 25402043, kit sha 6743ab6, hg38 v2
blacklist applied: 195 800 1 kb bins dropped). `map_check.txt`: 0 FAIL (260/260 rows;
R2_level ≥ 0; delta_R2 > shuffle in 256/260).

## How to read the numbers

- One row = one product at one depth: the two t117 halves untouched ("15M") or both thinned to
  ratio 0.25 ("3.75M"). The label is nominal: the depth arms (7.5M, 3.75M products) have smaller
  halves, and DNase base has 25M. 1 kb bins, all chromosomes but chrM/chrY, blacklist removed.
- d = |log2((a+0.5)/(b+0.5))| per bin. The level null d̂ is the mean of d in 40 quantile bins
  of log2(a+b+1). r = d − d̂ is the residual. Every feature is read on r.
- delta_R2 is the extra OLS R² of d from all four features **together** (GC, mappability,
  repeats, peak/domain context) over the level null. The tool does not split it per feature.
  The per-feature readings are the Spearman of r with each feature and the decile tables.
- Shuffle control: delta_R2 with the feature rows permuted (one seed). Spread: the min–max over
  the class's products. Class value = median over products.

## Signal level explains most of what is explained

| class (n) | depth | R2_level (min–max) | delta_R2, 4 features (min–max) | shuffle delta_R2 (max) |
|---|---|---|---|---|
| DNase (10) | 15M | 0.378 (0.293–0.380) | 0.00002 (0.00002–0.00013) | 0.00001 (0.00001) |
| DNase (10) | 3.75M | 0.206 (0.092–0.219) | 0.0014 (0.0007–0.0082) | 0.00001 (0.00002) |
| narrow (60) | 15M | 0.179 (0.127–0.247) | 0.00009 (0.000002–0.0003) | 0.00001 (0.00003) |
| narrow (60) | 3.75M | 0.082 (0.037–0.119) | 0.0008 (0.0001–0.0048) | 0.00001 (0.00002) |
| broad (60) | 15M | 0.194 (0.159–0.255) | 0.00005 (0.000002–0.0002) | 0.00001 (0.00003) |
| broad (60) | 3.75M | 0.090 (0.035–0.123) | 0.0006 (0.0001–0.0059) | 0.00001 (0.00003) |

R2_level_ols agrees with R2_level to ±0.002 in every class. The level null explains 8–38 % of
the per-bin disagreement; the rest is not explained by level or by any feature here. The four
features beat the shuffle in 256/260 rows (the 4 misses are mapq arms at 15M), but they add at
most 0.008 R² (median ≤ 0.0014). At 15M they add < 0.0003.

## Per feature (Spearman of r with the feature, median (min–max))

| class | depth | GC | mappability | repeats |
|---|---|---|---|---|
| DNase | 15M | 0.048 (−0.059–0.072) | 0.010 (−0.015–0.010) | −0.012 (−0.013–0.020) |
| DNase | 3.75M | −0.031 (−0.099–0.003) | 0.002 (−0.019–0.011) | −0.000 (−0.008–0.035) |
| narrow | 15M | −0.018 (−0.049–0.012) | 0.020 (−0.011–0.041) | −0.014 (−0.026–0.010) |
| narrow | 3.75M | 0.001 (−0.012–0.028) | −0.004 (−0.030–0.003) | −0.001 (−0.008–0.016) |
| broad | 15M | −0.023 (−0.028–0.011) | 0.013 (−0.014–0.036) | −0.014 (−0.023–0.011) |
| broad | 3.75M | −0.002 (−0.013–0.020) | −0.005 (−0.026–0.003) | 0.000 (−0.007–0.022) |

No |ρ| exceeds 0.1 in any product. The deciles show one tail only: the **bottom mappability
decile** carries a positive residual in every class (mean r 0.005–0.012 at 15M; 0.018–0.044 at
3.75M; other map deciles −0.024 to +0.004). The top-1 % residual bins are enriched there 1.42×
(narrow) and 1.49× (broad) at 15M (product range 0.59–2.08), but ~1.0× at 3.75M and in DNase.
The top repeat decile has a smaller positive residual (0.004–0.011 at 15M, 0.013–0.032 at
3.75M). GC deciles are flat at 15M (|mean r| ≤ 0.006); at 3.75M low-GC bins read DNase +0.074.

## Context: peak vs domain vs background (median; mean_r min–max over products)

| class | depth | mean d: other / domain / peak | mean r: other / domain / peak | top-1 % r enrichment: other / domain / peak |
|---|---|---|---|---|
| DNase | 15M | 1.36 / 0.62 / 0.49 | +0.002 / −0.003 / −0.009 (−0.019–−0.006) | 1.20 / 0.31 / 0.18 |
| DNase | 3.75M | 1.26 / 0.74 / 0.63 | +0.006 / −0.017 / −0.022 (−0.083–−0.011) | 1.14 / 0.54 / 0.43 |
| narrow | 15M | 0.97 / 0.67 / 0.46 | +0.001 / −0.000 / −0.016 (−0.118–−0.003) | 1.09 / 0.33 / 0.03 |
| narrow | 3.75M | 1.10 / 0.88 / 0.59 | +0.003 / −0.020 / −0.042 (−0.192–−0.018) | 1.05 / 0.68 / 0.13 |
| broad | 15M | 0.94 / 0.58 / 0.53 | +0.000 / −0.004 / −0.002 (−0.030–+0.003) | 1.12 / 0.09 / 0.02 |
| broad | 3.75M | 1.09 / 0.73 / 0.66 | +0.003 / −0.026 / −0.024 (−0.167–−0.011) | 1.10 / 0.30 / 0.15 |

Raw d is lowest in peaks, because peaks have high counts. After the level null, peaks and
domains still sit slightly below background (mean r −0.002 to −0.042), and the worst-1 % bins
fall in the background. Exception: the depth arms put the top-1 % residual into peaks and
domains. The 3.75M products (halves 1.875M, 8× smaller) read peak 1.8–3.2×, domain 1.6–2.5×;
the 7.5M products read peak 0.9–2.0×, domain 1.5–2.1×. At low depth, peaks become the noisy bins.

## What this says for q10

- **Signal level is the main named driver.** It explains 0.18–0.38 of per-bin disagreement at
  15M and 0.08–0.21 at 3.75M, against a shuffle of ≤ 0.00003.
- **Mappability, repeats, GC and context add almost nothing beyond level** (joint delta_R2
  ≤ 0.0003 at 15M, ≤ 0.008 at 3.75M). They pass the shuffle control, but the size is tiny.
- The only regional tail: low-mappability bins hold 1.4–1.5× their share of the worst bins in
  narrow and broad at 15M. Peaks and domains are slightly *more* reproducible than level
  predicts. So the floor is close to uniform after level, with a small low-mappability tail.
