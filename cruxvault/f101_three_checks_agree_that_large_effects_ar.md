---
id: f101
type: finding
schema: 3
title: Three checks agree that large effects are stable, but the arm-versus-base…
parent: q56
confidence: moderate
why: Three checks agree that large effects are stable, but the arm-versus-base comparison on 256 regions is still owed and the bootstrap has only 16 resampling units
evidence: results/region_sampling_checks/REGION_CHECKS.md
created: "2026-11-17T03:32:12"
updated: "2026-11-17T03:32:12"
---

# f101 — Three checks agree that large effects are stable, but the arm-versus-base…

Parent:: [[q56_do_the_ablation_conclusions_drawn_from_t]]

## Answer

Partly, so far. The 16 pair regions are not low-signal (median at the 58th percentile of 6,724 candidate regions; 0 of 16 near zero) and all 39 effects beyond the seed range keep their sign in at least 95 percent of 5,000 paired region-bootstrap draws. On 256 regions (3 base models and one U-Net variant, 3 seeds): gap closed per run differs by at most 0.024, the correction-only term is 0.03 to 0.046 higher on 16 regions, model differences above about 0.05 reproduce, one near-tie flips sign, and the pre-written check fails for 8, 16 and 32 regions. Correction scoring therefore uses all 256 regions; the rescoring of the 7 decisive ablation arms on 256 regions is not read yet.
