---
id: f26
type: finding
schema: 3
title: Counts readings clear the shuffle and floor controls by 50–1000×; moderate…
parent: q11
confidence: moderate
why: Counts readings clear the shuffle and floor controls by 50–1000×; moderate because the p-space null is weak and the PI has not judged.
evidence: results/t141/DIGEST.md, results/t142/t141_readthin/knob_classes.tsv
created: "2026-10-31T12:58:41"
updated: "2026-10-31T12:58:41"
---

# f26 — Counts readings clear the shuffle and floor controls by 50–1000×; moderate…

Parent:: [[q11_where_along_the_genome_does_each_covaria]]

## Answer

Most knobs act uniformly, so one genome-wide map corrects them: depth, duplicate removal, paired-end in histone marks, fragment extension and control depth. Two do not: the MAPQ 0 filter acts mostly through mappability (+0.45 log2 in the lowest-mappability decile) and read-length crops act through mappability and repeats, so they need a position-aware correction. The depth control passes once thinning is whole-read. Readings in −log10 p are less clean, because the level null under-fits peaks.
