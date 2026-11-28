## chip: 196 training tracks
best effective depth: median 24.7M, quartiles 16.6-30.5M
tracks whose best fraction is 1 (no thinning): 51 of 196
log best depth = 9.308 + 0.436 log depth
  cap      10M: training mean nMSE 0.5323, median 0.5064
  cap      15M: training mean nMSE 0.4508, median 0.3899
  cap      20M: training mean nMSE 0.4060, median 0.3391
  cap      25M: training mean nMSE 0.4129, median 0.3115
  cap      30M: training mean nMSE 0.4742, median 0.3080
  cap      40M: training mean nMSE 0.5779, median 0.3820
  cap      60M: training mean nMSE 0.7428, median 0.4712
  cap 1000000M: training mean nMSE 1.0939, median 0.4837
best cap on training: 20M
  val C02M22 22.3M: floor nMSE 0.155 -> 0.155 (frac 1.0); its own best frac 1.0
  val C04M16 26.9M: floor nMSE 0.256 -> 0.265 (frac 0.7); its own best frac 1.0
  val C09M20 36.9M: floor nMSE 0.164 -> 0.204 (frac 0.5); its own best frac 0.7
  val C10M17 33.0M: floor nMSE 0.322 -> 0.330 (frac 0.7); its own best frac 1.0
  val C12M16 65.2M: floor nMSE 0.900 -> 0.143 (frac 0.35); its own best frac 0.5
  val C12M32 44.8M: floor nMSE 0.451 -> 0.268 (frac 0.5); its own best frac 0.7
  val C13M20 7.7M: floor nMSE 0.383 -> 0.383 (frac 1.0); its own best frac 1.0
  val C16M17 24.4M: floor nMSE 0.402 -> 0.384 (frac 0.7); its own best frac 0.7
  val C17M04 38.1M: floor nMSE 0.787 -> 0.456 (frac 0.5); its own best frac 0.5
  val C17M19 52.7M: floor nMSE 0.598 -> 0.550 (frac 0.35); its own best frac 0.7
  val C17M29 26.1M: floor nMSE 1.626 -> 0.951 (frac 0.7); its own best frac 0.35
  val C18M21 26.4M: floor nMSE 2.203 -> 0.722 (frac 0.7); its own best frac 0.35
  val C18M25 43.6M: floor nMSE 0.349 -> 0.337 (frac 0.5); its own best frac 0.7
  val C20M22 98.5M: floor nMSE 4.365 -> 0.209 (frac 0.18); its own best frac 0.35
  val C23M03 50.4M: floor nMSE 0.148 -> 0.390 (frac 0.35); its own best frac 0.7
  val C23M07 30.3M: floor nMSE 0.728 -> 0.282 (frac 0.7); its own best frac 0.5
  val C24M17 36.6M: floor nMSE 0.158 -> 0.276 (frac 0.5); its own best frac 1.0
  val C24M25 52.6M: floor nMSE 0.344 -> 0.244 (frac 0.35); its own best frac 0.7
  val C25M21 59.8M: floor nMSE 14.012 -> 1.296 (frac 0.35); its own best frac 0.18
  val C27M03 18.4M: floor nMSE 0.351 -> 0.351 (frac 1.0); its own best frac 1.0
  val C27M13 16.9M: floor nMSE 0.161 -> 0.161 (frac 1.0); its own best frac 1.0
  val C27M24 89.6M: floor nMSE 1.399 -> 0.540 (frac 0.25); its own best frac 0.35
  val C29M29 10.8M: floor nMSE 0.242 -> 0.242 (frac 1.0); its own best frac 1.0
  val C31M25 151.4M: floor nMSE 7.580 -> 0.396 (frac 0.125); its own best frac 0.18
  val C32M08 51.2M: floor nMSE 1.320 -> 0.389 (frac 0.35); its own best frac 0.35
  val C32M12 33.0M: floor nMSE 0.317 -> 0.331 (frac 0.7); its own best frac 1.0
  val C32M20 135.4M: floor nMSE 2.969 -> 0.666 (frac 0.125); its own best frac 0.25
  val C34M32 68.2M: floor nMSE 2.293 -> 0.606 (frac 0.25); its own best frac 0.25
  val C45M22 46.5M: floor nMSE 0.415 -> 0.096 (frac 0.5); its own best frac 0.7
validation floor (chr9+chr21): mean nMSE 1.565 -> 0.401; above 1 on 9 -> 1 of 29; better on 18
## dnase: 16 training tracks
best effective depth: median 59.8M, quartiles 48.1-66.2M
tracks whose best fraction is 1 (no thinning): 16 of 16
log best depth = 0.000 + 1.000 log depth
  cap      10M: training mean nMSE 0.7838, median 0.8011
  cap      15M: training mean nMSE 0.7093, median 0.7295
  cap      20M: training mean nMSE 0.6348, median 0.6481
  cap      25M: training mean nMSE 0.5714, median 0.6299
  cap      30M: training mean nMSE 0.5065, median 0.5230
  cap      40M: training mean nMSE 0.4260, median 0.3978
  cap      60M: training mean nMSE 0.3220, median 0.2726
  cap 1000000M: training mean nMSE 0.2773, median 0.2527
best cap on training: 1000000M
  val C03M02 43.6M: floor nMSE 0.365 -> 0.365 (frac 1.0); its own best frac 1.0
validation floor (chr9+chr21): mean nMSE 0.365 -> 0.365; above 1 on 0 -> 0 of 1; better on 0
