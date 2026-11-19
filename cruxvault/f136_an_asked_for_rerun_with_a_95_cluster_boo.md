---
id: f136
type: finding
schema: 3
title: An asked-for rerun with a 95% cluster bootstrap, but the DNase-seq…
parent: q62
confidence: moderate
why: An asked-for rerun with a 95% cluster bootstrap, but the DNase-seq common-cell check failed for read sampling (0.0415 on the 7 common cells against a row interval of 0.0450 to 0.0517), and experiment and lab rows stay not estimable.
evidence: results/closeout_E4/ladder_excl_ENCSR000EKG_2026-11-19/ladder.md, results/closeout_E4/ladder_excl_ENCSR000EKG_2026-11-19/layers.tsv.exclusions.json
created: "2026-11-18T20:58:27"
updated: "2026-11-18T20:58:27"
---

# f136 — An asked-for rerun with a 95% cluster bootstrap, but the DNase-seq…

Parent:: [[q62_how_large_are_the_dnase_seq_variance_lay]]

## Answer

Replicate ladder half. With ENCSR000EKG left out (1 DNase-seq pair dropped), the DNase-seq common depth rose from 15,201 to 6,648,864 reads and the DNase-seq rows became interpretable: read sampling 0.0483 [0.0450, 0.0517] (33 pairs, 17 cells), biological replicate 0.00059 [-0.00023, 0.00142] (8 pairs, 8 cells; interval holds 0), cell type 0.0127 [0.0084, 0.0170] (28 matched pairs, 7 cells). Units: 0.5 E[(Y-Y')^2] of log1p counts. Cell type is 23% of the cell-type pair value, against 6% for broad and 12% for narrow histone marks. The second-experiment layers for DNase-seq are not in this answer (job still running).
