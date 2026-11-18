---
id: f112
type: finding
schema: 3
title: An asked-for measurement on 43 validation tracks and 256 regions with…
parent: q60
confidence: moderate
why: An asked-for measurement on 43 validation tracks and 256 regions with replicate pairs, but most assays rest on one pair, DNase-seq has one track, and the 1 bp store is not yet verified end to end.
evidence: tasks/t250_onion_variance_lost_below_25_bp_run.md
created: "2026-11-18T09:48:46"
updated: "2026-11-18T09:48:46"
---

# f112 — An asked-for measurement on 43 validation tracks and 256 regions with…

Parent:: [[q60_how_much_of_the_1_bp_count_variance_of_a]]

## Answer

Almost all of it lies below 25 bp, and almost none of that is reproducible: of the 1 bp count variance, 0.927 (broad marks, 19 tracks) and 0.901 (narrow marks, 22 tracks) lies below 25 bp, against 0.960 for read-sampling noise alone; biological replicates reproduce 0.0011 and 0.0009 of it (about 0.1%). A finer bin than 25 bp adds noise and not signal.
