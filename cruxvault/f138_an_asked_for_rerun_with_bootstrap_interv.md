---
id: f138
type: finding
schema: 3
title: An asked-for rerun with bootstrap intervals, but on 7 other-experiment…
parent: q62
confidence: moderate
why: An asked-for rerun with bootstrap intervals, but on 7 other-experiment pairs, with cells that differ between rungs, and in the one-sided form that the PI replaced by the two-sided form for the replicate ladder.
evidence: results/closeout_D2/layers_excl_ENCSR000EKG_2026-11-19/layers_dnase.tsv, results/closeout_D2/layers_excl_ENCSR000EKG_2026-11-19/layers.tsv.exclusions.json
created: "2026-11-18T21:14:52"
updated: "2026-11-18T21:14:52"
---

# f138 — An asked-for rerun with bootstrap intervals, but on 7 other-experiment…

Parent:: [[q62_how_large_are_the_dnase_seq_variance_lay]]

## Answer

Second-experiment half. With ENCSR000EKG left out (1 DNase-seq pair dropped), the DNase-seq common depth of the second-experiment layers rose from 15,201 to 13,300,584 reads. One-sided layers (0.5 E[(Y-Y')^2] of log1p counts; the replicate value includes read sampling): biological-replicate pair value 0.0732 [0.0692, 0.0772] (8 pairs, 8 cells); other-experiment increment above it 0.0192 [0.0110, 0.0274] (7 pairs, 7 cells), i.e. 26% of the replicate pair value; cell-type increment above that 0.0113 [-0.0005, 0.0232] (136 pairs, 16 cells; interval holds 0). So for DNase-seq a second experiment of the same cell type differs by more than a biological replicate, and the cell-type step on top of it is not resolved. The split into experiment (4 pairs), lab (1 pair) and donor (2 pairs) is not estimable.
