---
id: f392
type: finding
schema: 3
title: chr20 only, 2 experiments; consistent with DNase (no control) matching…
parent: q91
confidence: moderate
why: chr20 only, 2 experiments; consistent with DNase (no control) matching exactly, but Jacob's control inputs are unpublished
evidence: results/t390/chip_gap/CHIP_GAP.md
created: "2026-11-22T23:22:45"
updated: "2026-11-22T23:22:45"
---

# f392 — chr20 only, 2 experiments; consistent with DNase (no control) matching…

Parent:: [[q91_why_does_our_chip_reprocessing_of_the_bl]]

## Answer

Most likely the control reads: treatment reads match Jacob's (46% / 36% of bins exactly equal, median ratio 1.000 in every value band), and the differing bins are those where a local control window sets lambda, ours higher by median 1.23. Pipeline-setting variants do not close it; a control subsampled toward treatment depth brings the mean ratio to about 1.01. Not closable exactly without Jacob's control list.
