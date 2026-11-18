---
id: f117
type: finding
schema: 3
title: Three seeds on the 16-region screening set with large seed ranges in the…
parent: q44
confidence: moderate
why: Three seeds on the 16-region screening set with large seed ranges in the arms without depth views; the depth effect rests on the separation of all seeds, not on the 1x rule.
evidence: results/closeout_A3/WHERE_VIEWS.md, results/closeout_A3/where_vs_allviews.tsv
created: "2026-11-18T16:21:59"
updated: "2026-11-18T16:21:59"
---

# f117 — Three seeds on the 16-region screening set with large seed ranges in the…

Parent:: [[q44_does_training_the_u_net_on_the_counterfa]]

## Answer

Where the views act, on the U-Net (16 regions, 3 seeds, gap closed against the oracle floor, seed range in parentheses): each view type acts on its own pair type, outside the version-3 composite. Depth-thinning views help on the 516 depth-thinning pairs: 0.614 (0.012) with all views and 0.586 (0.057) with base + depth, against 0.127 (0.768), 0.355 (0.255) and -0.012 (0.864) for the three arms without them; all 6 seeds with depth views (0.551 to 0.621) lie above all 9 seeds without (at most 0.445). Alignment-level views (no duplicate removal, no MAPQ filter) hurt on the 84 MAPQ + no-dedup pairs: 0.018 (0.071) against 0.272 (0.050) for all views and 0.377 (0.095) for base only, beyond twice the range against both. Read-level views do not move the 14 crop pairs (0.507 to 0.532 in every arm). No view arm moves the composite, imputation, crop-pair correction or calibration beyond the range.
