---
title: PI decisions of 2026-11-17 (close-out, nine forks)
kind: decision record
parent: "[[closeout todo]]"
---

# PI decisions of 2026-11-17

The PI answered nine open decisions one at a time. Each choice below was the PI's click. Costs are allocated CPU-hours.

| # | Decision | PI choice | Consequence |
|---|---|---|---|
| 1 | Where the 256-region rescoring of the 7 decisive arms runs | Nibi, in the close-out window | Starts when Nibi scratch is writable; about 6 scorings at one time; the Fir copies (jobs 4535347 to 4535923) stay queued, and cancelling them is the PI's |
| 2 | Second-experiment layers and baselines | Run now on Nibi | Submitted 2026-11-17: layers 27664409; baselines 27664415, 27664417, 27664418, 27664419, 27664422 |
| 3 | Lab effect in the cross-lab analysis | Report lab together with donor, as one effect | The pure lab effect is not reported as a headline |
| 4 | The 17 unbuilt DNase-seq second-experiment rows | Drop them | The add-on stays at 71 experiments; DNase-seq has no second-experiment pairs |
| 5 | Replicate ladder definition | Two-sided subtraction, with the check on common cell types | Earlier ladder numbers are not comparable; reports state the change |
| 6 | Resolution study design | Full design, next session | 75 runs, about 2,340 CPU-hours (ready queue entry 9, `DESIGN=full`) |
| 7 | QC-on-input arm | Accept the design as built (presence flags, leak guards) | 3 models, 3 seeds, about 404 CPU-hours, next session |
| 8 | Mix-and-match round | Gate first (about 474), then the round for the switches that pass (at most about 5,390) | Next session |
| 9 | Minimum covariate set | One model (the one that mix-and-match selects), leave-one-field-out, groups and the removal curve | 3,008 (conv autoencoder), 3,298 (conv + attention) or 5,060 (U-Net) CPU-hours, next session |

Moved to the next session by the PI on the same day, with a recommendation and information in `settled miniCANDI design.md`: the architecture that goes to scaling, and the data version of the final model.
