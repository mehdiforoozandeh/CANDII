---
id: f159
type: finding
schema: 3
title: At 1 bp, on 9 replicate units of 6 single-end train experiments, the…
parent: h58
direction: supports
confidence: weak
why: At 1 bp, on 9 replicate units of 6 single-end train experiments, the dedup-on reads are exactly the distinct (5' position, strand) set of the dedup-off reads, 0 absent either way; pooled views repeat positions on 0.2-2.6% of lines, so dedup runs per replicate (K = 25 x units). This confirms the per-position cap mechanism; the closed-form law against the count-space law is not yet tested, so weak.
vs_expect: mechanism premise confirmed exactly; the law comparison is pending
evidence: results/t396/dedup_1bp/DEDUP_1BP.md
created: "2026-11-27T18:01:19"
updated: "2026-11-27T18:01:19"
---

# f159 — At 1 bp, on 9 replicate units of 6 single-end train experiments, the…

Parent:: [[h58_deduplication_is_a_per_position_cap_whos]]
