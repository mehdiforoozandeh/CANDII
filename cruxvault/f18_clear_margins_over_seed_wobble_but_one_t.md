---
id: f18
type: finding
schema: 3
title: Clear margins over seed wobble, but one track and one f seed, and two of…
parent: q6
confidence: strong
why: "Clear margins over seed wobble, but one track and one f seed, and two of eight designs stopped by the PI. Settled by the PI's rule: no further work is planned on it (PI: no more compute on this, 2026-10-19)."
evidence: results/t125/part2_digest.md, results/t125/h_alone_summary.tsv, results/t125/h_repair.tsv, results/t125/h_vs_oldgrid.tsv
created: "2026-10-31T12:57:10"
updated: "2026-10-31T12:57:10"
---

# f18 — Clear margins over seed wobble, but one track and one f seed, and two of…

Parent:: [[q6_how_closely_does_a_map_h_reproduce_the_l]]

## Answer

On true counts, h is almost known: the fragment-window formula reads 0.0865 log CRPS and the best learned h 0.0798, so inputs matter more than flexibility. Train h on true counts, not on f's NB (better in 36 of 36 rows). In the chain, every h trails the direct-p model (0.178–0.284 against 0.143–0.165), but no chain record explodes, so the trade is safety against accuracy, and the remaining error belongs to f.
