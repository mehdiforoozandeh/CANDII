---
id: f71
type: finding
schema: 3
title: The seed rule of the expectation is met at rung 4 on the composite and on…
parent: h36
direction: supports
confidence: weak
null: none
why: The seed rule of the expectation is met at rung 4 on the composite and on headline 1, but its region interval and its eval_heavy arm were not measured, so the pre-registered 'no detectable gap' cannot be stated, and the score is screening.
vs_expect: "Expected: today minus no_eval inside plus or minus 1 seed spread, with a paired region interval inside the same margin; eval_heavy gains a positive amount below 2 seed spreads. Result at rung 4 (3 paired seeds, screening): rg_no_eval minus base is +0.0031 on the composite (spread of the difference 0.0391; margin 1 seed spread = 0.0341, the larger arm's), -0.0021 on specific variance above noise (spread 0.0125; base seed spread 0.0134) and +0.0009 on correction CRPS (spread 0.0168), all inside the margin: no measurable advantage of seen positions. The paired region interval was not computed, and the eval_heavy arm did not run. This answers the seen-positions measurement that PI ruling 39 asked for, at screening level; the protocol ruling stays with the PI."
verifiables: "Deciding metric (composite; headline 1 in counts): both inside 1 seed spread. Settling rule (no detectable gap only if both the seed rule and the region interval are inside the margin): seed rule met; region interval not measured."
evidence: results/metrics_screening/scores_ladder.md
created: "2026-11-14T16:38:57"
updated: "2026-11-14T16:38:57"
---

# f71 — The seed rule of the expectation is met at rung 4 on the composite and on…

Parent:: [[h36_seeing_the_evaluation_loci_in_training_d]]
