---
id: f4
type: finding
schema: 3
title: "A 64-channel f misses the law on never-trained depth pairs (0.41 per-track,…"
parent: h27
direction: against
confidence: moderate
why: "A 64-channel f misses the law on never-trained depth pairs (0.41 per-track, 0.34 across-track; trained 0.05), and its CRPS gain through the base shrank from 0.0146 to 0.006 with six seeds (95% interval −0.002 to +0.013); moderate because the check was written after the run (2026-10-31)."
verifiables: claim 0/1 met; controls 1/1 met; rule all
evidence: results/t128p3/m5_depthlaw_table.tsv, results/t128p3/rank_table.tsv, results/t128p3b/headline_table.tsv, results/t128p3b/checks.txt
migrated_from: h27
created: "2026-10-31T00:33:38"
updated: "2026-10-31T00:33:38"
---

# f4 — A 64-channel f misses the law on never-trained depth pairs (0.41 per-track,…

Parent:: [[h27_f_at_64_channels_2x_wider_with_the_same_]]
