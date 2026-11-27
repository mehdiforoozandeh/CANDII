# Validation boards, all 45 experiments (2026-11-25)

The final miniCANDI (seed 0) was scored on all 45 EIC validation experiments. It was scored against the original 2019 truth, with the challenge's own measures and the round-1 field of 10 entries (MOESM6). Each board below was rebuilt after its last score landed. The board files were copied from Fir `/scratch/mforooz/eicx/val/<board>/`. The summary lines come from `python3 side_by_side.py`.

| Board | Position of 11 | Mean combined rank | Median nMSE | nMSE above 1 | Combined rank 3 or better | Combined rank 10 or worse |
|---|---|---|---|---|---|---|
| Raw counts through the p-value pipeline | 2 | 4.31 | 0.970 | 22 of 45 | 23 | 5 |
| 20 M ChIP depth cap (primary) | 1 | 3.22 | 0.827 | 11 of 45 | 29 | 2 |
| Mean of final and base model counts, then the cap (second board) | 1 | 3.20 | 0.795 | 10 of 45 | 30 | 2 |

- nMSE is the squared error divided by the variance of the truth.
- Against the raw board, the depth cap was better on 18 experiments, worse on 6, and changed the mean combined rank by -1.09.
- Against the raw board, the averaged board was better on 19, worse on 12, and changed the mean combined rank by -1.10.
- On the averaged board, three experiments are final-model-only: the ChIP-seq experiment re-predicted at its single-end row (C25M26), and two DNase-seq experiments (C34M02, C50M02). The base model has no counts at their covariate rows.
- The depth cap is the one rule applied to every scored experiment. It was fitted on training tracks only (`../depth_rule/DEPTH_RULE.md`).
- The three ChIP-seq and DNase-seq experiments that were predicted late use the pooled single-end row (fragment length 227 bp for the ChIP-seq one, the read-weighted mean of its two replicates).

Files: `<board>/SUMMARY.md` and `<board>/board_validation.tsv` for each board; `SIDE_BY_SIDE.tsv` joins the combined rank and nMSE per experiment.
