---
id: f93
type: finding
schema: 3
title: Beyond the range on all three models; the expectation from the CAC test bed…
parent: q51
confidence: moderate
why: Beyond the range on all three models; the expectation from the CAC test bed did not hold on the composite.
evidence: results/ablations_closeout/ABLATIONS_READ.md
created: "2026-11-16T22:22:45"
updated: "2026-11-16T22:22:45"
---

# f93 — Beyond the range on all three models; the expectation from the CAC test bed…

Parent:: [[q51_does_two_step_correction_through_a_base_]]

## Answer

No gain on the composite. Native two-step lowers crop-pair gap closed beyond the seed range on both conv models (0.718 to 0.602 and 0.673 to 0.614) and lowers the U-Net composite to 0.412. A fixed per-class base depth beats native on the U-Net (gap closed 0.636 against 0.443). The correction-only term rises on the conv models (to 0.80).
