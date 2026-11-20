---
id: f81
type: finding
schema: 3
title: Read from the refreshed rung-4 arm table and the select_rule terms, 3 paired…
parent: q36
confidence: moderate
why: Read from the refreshed rung-4 arm table and the select_rule terms, 3 paired seeds; moderate because it is a screening score at one rung, although the composite and calibration margins are many seed spreads wide.
evidence: results/metrics_screening/scores_rung4_arms.md
created: "2026-11-15T00:24:21"
updated: "2026-11-15T00:24:21"
---

# f81 — Read from the refreshed rung-4 arm table and the select_rule terms, 3 paired…

Parent:: [[q36_does_hiding_depth_from_the_encoders_a_ha]]

## Answer

No: hiding depth from the network makes rung 4 worse, mostly in calibration. Expectation before the run (results/eta_space/MODEL_DESIGN.md and the decision rule of this question): no general accuracy gain from a hard offset; if it matches or beats the base, eta can be made depth-free with no loss; if it loses, depth carries information beyond the offset and stays a covariate. Result, rung 4 with the depth covariate hidden against rung-4 base (3 paired seeds, screening, composite version 2): composite 0.260 against 0.444 (paired -0.184, larger seed spread 0.030; all three seeds -0.18 to -0.19); specific variance above noise 0.406 against 0.440 (-0.033, spread 0.013, exceeds); gap closed (oracle) on all headline pairs 0.218 against 0.298 (-0.081, spread 0.086, does not exceed); on the pairs with no duplicate-keeping side 0.510 against 0.519 (-0.008, spread 0.120, does not exceed); correction CRPS +0.019 (spread 0.020, does not exceed). The uncertainty term (1 - PIT deviation of the model / PIT deviation of the marginal NB) fell from 0.600 to -0.099 (seeds -0.182, -0.142, +0.029; paired -0.699 against a seed spread of 0.211), i.e. the hard-offset model is less well calibrated than one constant NB per track. The uncertainty term gives -0.148 of the -0.184 composite difference. So the expectation of no gain held, the decision rule says depth stays a covariate, and the loss is in the predicted distribution more than in the mean; the mean formula (depth x 2^eta) was the same in both arms.
