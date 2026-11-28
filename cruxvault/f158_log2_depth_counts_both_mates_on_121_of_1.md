---
id: f158
type: finding
schema: 3
title: log2_depth counts both mates on 121 of 121 PE rows read, and the count…
parent: h61
direction: against
confidence: moderate
why: log2_depth counts both mates on 121 of 121 PE rows read, and the count stores count the same mates, so depth and counts agree; the premise that our counts see half was false. For the 13 DNase PE rows depth_fragments equals depth_reads, about 2x the true fragment count (a label defect, not a depth-offset defect). MACS2's input for PE could not be determined.
vs_expect: expected PE depth about 2x the fragments the counts see; found the counts see the same mates
evidence: results/t396/pe_depth/
created: "2026-11-27T17:55:10"
updated: "2026-11-27T17:55:10"
---

# f158 — log2_depth counts both mates on 121 of 121 PE rows read, and the count…

Parent:: [[h61_our_log2_depth_counts_both_mates_of_pair]]
