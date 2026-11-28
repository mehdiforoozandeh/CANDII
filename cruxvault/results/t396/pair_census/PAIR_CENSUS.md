# Pair census: dedup and SE pairs for the widened correction headline (R39, decision D4)

Hand-written; not an output. Source files: `census.tsv`, `by_covariate.tsv`, `candidates.tsv`,
from `tools/t396/audits.py pair-census` (Fir job 6727207, kit SC5). Inputs: tables
`/project/def-maxwl/mforooz/ONION/tables/eic_v1_2` (`pairs.tsv`, 15,123 pairs) and the evaluation
set `/project/def-maxwl/mforooz/ONION/eval/eic_v1_2/evalset_p16.json` (1,480 correction pairs).

## Rules used

- Headline pair now: `never_trained` = 1 and `subset_pair` = 0 (the rule of
  `score._headline_pair`; evaluation-set records carry no `rung`).
- Widened candidate (this audit's rule, i.e. the pairs that D4 adds): the differing knobs include
  `nodedup` or `se`, `subset_pair` = 0, `split` = `val`, and both members have `counts.h5`, `pr1.h5`
  and `pr2.h5` in the read-count store. Train-split pairs are left out, because the model trained
  on them.
- `census.tsv` keeps only pairs whose knobs include `nodedup` or `se`, so that the file stays
  under 1 MB. `by_covariate.tsv` counts every pair for all five covariates.

## Result

Every `val` pair of `pairs.tsv` is also an evaluation-set correction pair (`n_in_evalset` equals
`n_pairs` on every `val` row of `by_covariate.tsv`). Every candidate has both halves present.

| covariate | candidate pairs | of which headline now | experiments |
|---|---|---|---|
| dedup (`nodedup` in knobs) | 458 (86 nodedup, 258 depth+nodedup, 86 mapq+nodedup, 26 crop+nodedup, 2 nodedup+se) | 84 (mapq+nodedup) | 43 |
| SE (`se` in knobs) | 20 (8 crop+se, 6 depth+se, 2 each of se, mapq+se, nodedup+se) | 4 (crop+se) | 2 |
| union | 476 | 88 | 44 |

The headline now uses 102 pairs (84 mapq+nodedup, 14 crop, 4 crop+se). The widened set adds 388
pairs: 374 dedup pairs and 16 SE pairs with `never_trained` = 0, minus the 2 nodedup+se pairs
counted in both.

The SE breakdown rests on 2 experiments (ENCSR806XIU, H3K4me2, 16 pairs; ENCSR521AIV, H3K9ac,
4 pairs). A per-covariate SE number from 2 experiments has no usable spread over experiments.
The train split holds 254 SE pairs from 20 experiments; they are not candidates under this rule.

In the read-start store, both members are present for all 476 candidates. For 4 crop+se pairs, a
parent of one member is missing from a store (`parents_in_both_stores` = 0).

## Not checked

- Whether a `never_trained` = 0 pair biases the widened headline. The candidates are val pairs,
  but their knob values (nodedup, se) occur in training pairs of other experiments.
