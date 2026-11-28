---
id: f166
type: finding
schema: 3
title: On 7 controlled tracks (same library, dedup off or PE re-run), the existing…
parent: q95
confidence: moderate
evidence: results/t396/cf_follow/CF_FOLLOW.md
created: "2026-11-28T08:04:49"
updated: "2026-11-28T08:04:49"
---

# f166 — On 7 controlled tracks (same library, dedup off or PE re-run), the existing…

Parent:: [[q95_why_does_minicandi_fail_to_follow_the_de]]

## Answer

On 7 controlled tracks (same library, dedup off or PE re-run), the existing read-count model does not follow the dedup or run-type covariate: swapping it changed top-1% deviance by under ~0.16 and total by at most 3%, mostly inside the seed range; the read-start model (80,000 windows, unequal) moved in the right direction for dedup only (top-1% deviance down 6-31%). On the same pairs the closed-form start-space cap closes 0.89 (dedup) and 0.85 (run type). The covariate effect is present and closed-form within a library, but the models do not learn it from the covariate prompt.
