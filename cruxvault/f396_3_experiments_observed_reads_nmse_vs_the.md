---
id: f396
type: finding
schema: 3
title: 3 experiments, observed reads, nMSE vs the exact formula; no model retrained
parent: q90
confidence: moderate
why: 3 experiments, observed reads, nMSE vs the exact formula; no model retrained
evidence: results/t390/postpile/POSTPILE.md
created: "2026-11-23T02:30:43"
updated: "2026-11-23T02:30:43"
---

# f396 — 3 experiments, observed reads, nMSE vs the exact formula; no model retrained

Parent:: [[q90_can_a_count_space_model_like_minicandi_b]]

## Answer

A spread applied after an overlap-count prediction cannot close the conversion gap: spreading 25 bp overlap counts over the fragment length recovers 3%/12%/3% of the gain of a true fragment pileup (H3K4me3/H3K27me3/DNase), with or without dividing by bins per read. Start counts close it: stranded read starts extended by d recover 95%/93% on ChIP, unstranded starts recover 92% on DNase. A count-space entrant must predict 25 bp read-start counts (stranded for ChIP), not overlap counts.
