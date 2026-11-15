---
id: f78
type: finding
schema: 3
title: Read from the refreshed ladder table under composite version 2 with 3 seeds…
parent: q39
confidence: moderate
why: Read from the refreshed ladder table under composite version 2 with 3 seeds per rung and paired seed differences; moderate because it is a screening score (pair tests on 16 regions) and three seeds give a coarse seed spread.
evidence: results/metrics_screening/scores_ladder.md
created: "2026-11-15T00:23:58"
updated: "2026-11-15T00:23:58"
---

# f78 — Read from the refreshed ladder table under composite version 2 with 3 seeds…

Parent:: [[q39_do_the_context_rungs_2_to_6_convolution_]]

## Answer

Screening scores under composite version 2 (correction term = gap closed against the oracle floor, PI ruling 47), 3 seeds per rung, 80,000 training windows, about 1.0 M parameters per rung. Composite seed means (seed spread): rung 1 0.444 (0.009), rung 2 0.502 (0.031), rung 3 0.493 (0.033), rung 4 0.444 (0.030), rung 5 0.469 (0.030), rung 6 0.261 (0.052); position average 0.160. Every rung is above the position average on the composite and on gap closed (0.235). Rung 2 beats rung 1 on every column by more than the larger seed spread (composite +0.058 against 0.031; gap closed +0.102 against 0.023; specific variance +0.042 against 0.030), but its composite gain is below the 2 seed spreads (0.062) that the selection rule asks for. Rung 3 is within the seed spread of rung 2 on every column except gap closed on the 18 pairs with no duplicate-keeping side (-0.040 against 0.029). Rung 4 recovers the most specific variance above noise (0.440 against 0.389 for rung 3; +0.051, spread 0.013) and closes the least gap of rungs 1 to 5 (0.298 against 0.572; -0.273, spread 0.052). Rung 6 is lowest on every column. This supersedes the composite-version-1 reading of the earlier finding on this question, where rung 4 led.
