---
id: f164
type: finding
schema: 3
title: On 7 controlled tracks (same library, dedup on vs off) the start-space cap…
parent: h58
direction: supports
confidence: moderate
why: "On 7 controlled tracks (same library, dedup on vs off) the start-space cap with subsampling closed 0.94-0.98 of the gap to the floor in the top 1% at every plausible subsampling ratio (fitted in-sample 0.89 [0.78, 0.97]) and beat the count law at every ratio; 6 of 7 tracks reached 0.8 (H3K4me3 failed). The count law also clears 0.8 in the plausible band, so only the start-law win is robust. Contrast: across libraries on EIC pairs the cap failed (-0.56), so the closed form holds within a library, not across libraries."
vs_expect: "start-space cap closer to the floor than the count law: met; explains >= 0.8: met on controlled pairs only"
evidence: results/t396/cf_caps/CF_CAPS.md
created: "2026-11-28T07:47:19"
updated: "2026-11-28T07:47:19"
---

# f164 — On 7 controlled tracks (same library, dedup on vs off) the start-space cap…

Parent:: [[h58_deduplication_is_a_per_position_cap_whos]]
