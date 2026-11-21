# Glossary

Every term and metric in use. A new entry lands before its first use in anything the PI reads.

| Term | Gloss (10 words or fewer) | Origin (PI doc, PI, agent) | Row or decision |
|---|---|---|---|
| Pareto dominance | Best-or-tied on every goal term, strictly better on one | agent, PI chose | D1 |
| imputation term | variance recovered on held-out tracks (`imputation_raw`), 256 regions | onion close-out | D1 |
| correction term | gap closed on crop pairs vs oracle reference (`correction_v3_raw`) | onion close-out | D1, D3 |
| calibration term | uncertainty score of the predicted NB (`uncertainty_raw`) | onion close-out | D1 |
| seed range | max minus min over 3 seeds of a metric | onion close-out | D1 |
| conditioned skips | U-Net skip tensors modulated by FiLM from (C, C') | agent | D4 |
| pre/post CAC layers | covariate-only g sets count map f before and after network | PI | D5 |
| monotone curve f | invertible monotone map on log counts, knots from g | CAC testbed rung B | D6 |
| conditioned CNN f | small conv count map, parameters from g | CAC testbed rung D | D6 |
| FiLM | per-channel scale and shift computed from covariates | onion close-out | D4 |
| two-step CAC | (X,C) -> base view -> (X',C'), shared weights | onion close-out | D8 |
| leave-control-out evaluation | hide observed control, score the predicted control and its p-value | agent, PI chose | D9 |
| pair regions | 16 genome regions where correction pairs are scored | onion close-out | D3 |
