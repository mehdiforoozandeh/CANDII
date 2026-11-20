---
id: f126
type: finding
schema: 3
title: One seed of one model, in a form now known to overstate the Poisson part;…
parent: q63
confidence: weak
why: One seed of one model, in a form now known to overstate the Poisson part; the corrected formula is checked by simulation only.
evidence: results/closeout_E2/E2_RESULTS.md, results/closeout_E2/FORMULA_EXTENSION.md
created: "2026-11-18T16:23:32"
updated: "2026-11-18T16:23:32"
---

# f126 — One seed of one model, in a form now known to overstate the Poisson part;…

Parent:: [[q63_how_well_do_the_signal_p_value_track_and]]

## Answer

First numbers, one model and one seed, in the old in-bin form. Conv autoencoder seed 0, leave-one-assay-out, mean over 8 assays: variance recovered above noise 0.331 in counts, 0.178 in signal space (expectation of the -log10 p under the NB), 0.164 (plug-in score of the NB mean), 0.143 with the predicted control in place of the observed one; r_spec 0.263 in counts and 0.267 with the predicted control. The predicted control's lambda against the observed control's: median per-target ratio 0.919 with the control hidden (15 of 20 targets within 10%) and 0.968 with it visible (35 of 41). A review found that the in-bin expectation overstates the Poisson part of the pileup by s = d / (rl + 24) (about 3.3 at d = 200 bp, rl = 36), so the expectation values 0.178 and 0.143 (and the position-average and marginal-NB rows of the AlphaGenome 2 board, which used the same route) carry that error; the plug-in values do not. In simulation the corrected pileup-scale NB gives coverage variance 12.4 against a true 12.0 (comonotone, size 2), where the count-scale form gave 21.9. The box_d rescoring has not run.
