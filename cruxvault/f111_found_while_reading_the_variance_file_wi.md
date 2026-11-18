---
id: f111
type: finding
schema: 3
title: Found while reading the variance file with no prior prediction; the…
parent: q60
confidence: moderate
why: Found while reading the variance file with no prior prediction; the mechanism is checked on one control track and on a simulation only.
evidence: tasks/t250_onion_variance_lost_below_25_bp_run.md
exploratory: true
created: "2026-11-18T09:48:46"
updated: "2026-11-18T09:48:46"
---

# f111 — Found while reading the variance file with no prior prediction; the…

Parent:: [[q60_how_much_of_the_1_bp_count_variance_of_a]]

## Answer

Read halves of one library anti-correlate at 1 bp: all 138 counts-scale read-half rows are below zero (-0.0058 to -0.0441), because duplicate removal leaves at most one read per base and strand, so the two halves exclude each other. The control track matches the uniform-density prediction -m (b - 1) / b below 5 bp (-0.0092 measured and predicted). Read halves therefore cannot measure reproducible signal at fine scale; whether the earlier 25 bp read-half layers carry the same bias is open.
