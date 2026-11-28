---
id: f397
type: finding
schema: 3
title: A one-command pipeline (score_entry.py + run_entry.sh) takes predict.py's NB…
parent: q90
confidence: moderate
evidence: results/t390/entry_pipeline/ENTRY_PIPELINE.md
created: "2026-11-23T06:47:21"
updated: "2026-11-23T06:47:21"
---

# f397 — A one-command pipeline (score_entry.py + run_entry.sh) takes predict.py's NB…

Parent:: [[q90_can_a_count_space_model_like_minicandi_b]]

## Answer

A one-command pipeline (score_entry.py + run_entry.sh) takes predict.py's NB counts to a -log10 p track, scores it on MOESM2 raw and MOESM4 with self-called peaks plus QN, and places it among the 25 entrants. On observed counts it reproduced rungs L6 and L7 with 0 differing bins and 0 score difference on all 3 probe experiments. Expectation mode and a predicted control are untested.
