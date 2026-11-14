---
id: f68
type: finding
schema: 3
title: "Settled by PI ruling 47 on the evidence of the investigation: no further…"
parent: q32
confidence: strong
why: "Settled by PI ruling 47 on the evidence of the investigation: no further work on the choice of reference is planned; the merge of the tool is engineering, not a reopening."
evidence: results/gap_closed_floor/GAP_FLOOR.md, results/gap_closed_floor/gap_closed_headline.tsv, results/oracle_floor/ORACLE_FLOOR.md
created: "2026-11-14T16:38:24"
updated: "2026-11-14T16:38:24"
---

# f68 — Settled by PI ruling 47 on the evidence of the investigation: no further…

Parent:: [[q32_which_reference_should_define_the_1_of_t]]

## Answer

PI ruling 47 (2026-11-14): the 1 of the gap-closed scale is the oracle floor O, i.e. one half of the mean absolute difference between the two read halves of the target view (the expected CRPS of the true sampling distribution), on the same bins, regions and raw counts as the scored CRPS. Gap closed = (CRPS depth-only - CRPS method) / (CRPS depth-only - O). It enters the composite from the second shipment on (composite version 2); the old value (half-read floor as the top) is shown once beside it, labelled superseded; the metric is also reported on the headline pairs with no duplicate-keeping side. Evidence on the 102 headline pairs (full set, macro over 11 assays): position average 2.86 to 0.258 [0.197, 0.368]; half-read floor 1 to 0.167; rung-1 base (mean of 3 seeds, 16 pair regions) 5.34 to 0.474 [0.408, 0.528]; the denominator rose from 0.040 [0.020, 0.084] to 0.213 [0.169, 0.258]; the half-read floor scored 1.42 x O per pair (median), against sqrt(2) = 1.41 in theory for a forecast centered on the other half, which explains the old ratios above 1; no method went below O on any headline pair. Caveat kept: when one side of a pair keeps duplicates (84 of the 102 headline pairs, knob mapq+nodedup), source and target can share DNA fragments and O is not a strict bound (depth-only went below O on 3 non-headline pairs from a nodedup source with duplicate fraction 0.84). The implementation (tools/onion_eval/oracle_floor.py, select_rule and arm_report changes, branch chunk/onion-OF) reproduced the investigation exactly and is in review, not merged into the umbrella branch yet.
