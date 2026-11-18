---
id: f109
type: finding
schema: 3
title: The cause is read from the code and the depth column of the pair values, but…
parent: q59
confidence: moderate
why: The cause is read from the code and the depth column of the pair values, but the rerun without that track has not been done, so the size of the true DNase-seq layers is unknown.
evidence: tasks/t351_onion_replicate_ladder_in_the_two_s.md
created: "2026-11-18T09:48:46"
updated: "2026-11-18T09:48:46"
---

# f109 — The cause is read from the code and the depth column of the pair values, but…

Parent:: [[q59_do_the_nested_replicate_ladder_and_the_c]]

## Answer

The DNase-seq layer values of the replicate ladder and of the second-experiment layers are not interpretable: the common depth of an assay is the shallowest side of any of its pairs, and one DNase-seq track (ENCSR000EKG, cell C23) has 15,201 reads, so every DNase-seq pair value is computed at 15,201 reads; the next shallowest side of a DNase-seq pair has 13,300,584 reads. The DNase-seq increment read earlier as 3.40 is 3.40e-05.
