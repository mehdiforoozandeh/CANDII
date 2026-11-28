---
id: f147
type: finding
schema: 3
title: 16 targets across 16 assays, 3 seeds, directions stated before the run; one…
parent: q80
confidence: moderate
why: 16 targets across 16 assays, 3 seeds, directions stated before the run; one region sample and a single cell per target
evidence: results/t370/recal_study/covariate_prompt_methyl.md
created: "2026-11-25T02:09:32"
updated: "2026-11-25T02:09:32"
---

# f147 — 16 targets across 16 assays, 3 seeds, directions stated before the run; one…

Parent:: [[q80_does_the_final_minicandi_follow_a_change]]

## Answer

Depth yes, others partly. Depth +1 or -1 log2 scales the total by 1.97-2.03 or 0.496-0.514 on 16 of 16 targets with the profile unchanged (r >= 0.986). The read-length rule (36 to 100 bp raises the total) held on 12 of 16. Turning duplicate removal off lowers the NB size (more spread) but not the total; MAPQ 0 moves counts between regions (r 0.77-0.96) but not the total; run type barely moves anything. An unseen lab, platform or antibody keeps the total within 20% but does not make the model less confident.
