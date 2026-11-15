---
id: f80
type: finding
schema: 3
title: No selection arm passes the settling rule, as expected, but no weighted arm…
parent: h34
direction: mixed
confidence: weak
null: none
why: No selection arm passes the settling rule, as expected, but no weighted arm gains on specific variance either, and the arms that ran differ from the pre-registered list; weak because it is a screening score, one arm has 2 seeds, and the narrow-mark split was not read.
vs_expect: "Expected: a weighted peak-based arm gains 1 to 2 seed spreads on headline 1 of the narrow marks; no arm passes the full rule; the unweighted peak arm is worse than uniform on pooled PIT deviation and on the broad marks by more than 1 seed spread. Ran instead (continuous tilts, rung 4, 20,000 windows, against uniform tiling, paired seeds, composite version 2): cCRE-density tilt weighted at strength 0.25 and 0.5, unweighted at 0.5, variance tilt weighted (2 seeds), k-center, and a territory draw. Composite: no arm higher by more than the seed spread (best territory +0.009, spread 0.029); k-center -0.027 (0.019) and the unweighted cCRE-density tilt -0.025 (0.015) lower by more than the seed spread, almost all through the uncertainty term (0.25 x (0.524 - 0.624) = -0.025 and 0.25 x (0.523 - 0.624) = -0.025), a drop of about one seed spread of that term. Specific variance (macro over 9 assays): no arm higher by more than the spread; k-center -0.019 (0.011) and the weighted cCRE-density tilt at 0.25 -0.011 (0.011) lower. Gap closed (oracle): no arm beyond the spread."
verifiables: "Settling rule (composite gain above 2 seed spreads, no goal worse by more than 2): no arm passes, so the uniform draw stays. Pooled PIT deviation of the unweighted arm worse: in direction, through the uncertainty term; the broad-mark split was not read."
evidence: results/metrics_screening/scores_selection_arms.md
created: "2026-11-15T00:24:12"
updated: "2026-11-15T00:24:12"
---

# f80 — No selection arm passes the settling rule, as expected, but no weighted arm…

Parent:: [[h34_at_a_limited_budget_20_000_windows_a_wei]]
