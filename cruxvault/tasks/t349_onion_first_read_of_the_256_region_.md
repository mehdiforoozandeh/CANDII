---
id: t349
type: task
title: "Onion: first read of the 256-region rescoring by the pre-written rules (rules 2.1, 2.2 and 2.6 decided at 30 of 62 scores) and the one final-pass command"
category: implementation
parent: t152
blocked_by: None
refs: q56, h40, q44, q54
hypothesis_refs: 
status: done
created: "2026-11-18T09:46:40"
updated: "2026-11-18T09:46:59"
---

# t349 — Onion: first read of the 256-region rescoring by the pre-written rules (rules 2.1, 2.2 and 2.6 decided at 30 of 62 scores) and the one final-pass command

Refs:: [[q56_do_the_ablation_conclusions_drawn_from_t\|q56]], [[h40_scrambled_covariates_lowers_the_correcti\|h40]], [[q44_does_training_the_u_net_on_the_counterfa\|q44]], [[q54_does_another_loss_poisson_mean_nb_crps_e\|q54]]

## Why

turns the 256-region scores into design-table verdicts in one rerun

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- presentations/onion-closeout-2026-11-18-report/report.pdf

## Evidence

The read of 2026-11-18 15:54 UTC used the arm-summary table of 30 of 62 scored run-seeds (10 of 21 arms complete). Its folder `results/rescoring_p256/read_20261118T1554Z/` (RESCORING_READ.md, arms.tsv, verdicts.tsv) is git-ignored and exists in the foreman worktree only. The tracked copy of the verdicts is the design table of `INSTRUCTIONS/peeling_the_onion/closeout/settled miniCANDI design.md` (rows 8, 20, 23 and section 2.0); the deck and the report were rebuilt from it. d is the arm mean minus the reference mean; the bar is the larger of the two seed ranges (max minus min over 3 seeds).

- Rule 2.1, scrambled covariates in training, correction-only term (correction of a track to another view of the same experiment): conv autoencoder -0.195 (bar 0.046), conv + attention -0.230 (bar 0.081), U-Net -0.242 (bar 0.129). Imputation, calibration and the composite stay within the bar on all three models. Decided: settled, strong.
- Rule 2.2, base view only in training, correction-only term: conv autoencoder -0.341 (bar 0.067), conv + attention -0.315 (bar 0.104), U-Net -0.262 (bar 0.129). Composite: +0.007, +0.008, +0.020, all within. Decided: all counterfactual views stay in training, settled, strong.
- Rule 2.6, imputation loss weight 4: conv autoencoder correction on crop pairs -0.050 (bar 0.021); conv + attention imputation +0.039 (bar 0.017), crop-pair correction -0.013 (bar 0.042, within). U-Net 0 of 3 seeds. Decided: weight 4 stays a candidate for the conv + attention only.
- Sign check against the 16-region screening (arms.tsv, column sign_agrees_with_16): 48 of the 50 cells of complete arms agree in sign. The two that disagree are crop-pair correction on the conv + attention for base view only (-0.0117 on 256 regions, +0.0029 on 16) and for imputation weight 4 (-0.0131 against +0.0001); both are within the bar on 256 regions and near zero on 16.
- Not decided: rules 2.3 (two-step routing, 0 seeds), 2.4 (fixed input depth, U-Net 1 of 3 seeds) and 2.5 (average input, 0 seeds). The mix-and-match gate (`mix_arms.py qualify`) exits 3 (WAIT).
