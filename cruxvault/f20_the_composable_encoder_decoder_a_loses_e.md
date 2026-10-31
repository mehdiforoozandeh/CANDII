---
id: f20
type: finding
schema: 3
title: The composable encoder-decoder A loses even trained pairs (1.0–1.4 against…
parent: h22
direction: against
confidence: moderate
why: The composable encoder-decoder A loses even trained pairs (1.0–1.4 against old A's 0.09–0.12), because an exact inverse forces its two slopes to multiply to 1, and encoder-decoder D keeps its trained fit but stays at 0.26–0.52 on never-trained pairs, within noise of old D's 0.29–0.34.
vs_expect: missed
evidence: results/h22/report.md, results/t126/depth_table.tsv
created: "2026-10-31T12:58:16"
updated: "2026-10-31T12:58:16"
---

# f20 — The composable encoder-decoder A loses even trained pairs (1.0–1.4 against…

Parent:: [[h22_a_composable_g_one_shared_map_from_each_]]
