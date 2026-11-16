---
title: "A1: the nine named ablations on three architectures"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# A1: the nine named ablations on three architectures

## 1. What the PI asked

- PI order, plan Log 2026-11-16T21:10Z: "run all 9 untested constitution ablations (ChIP control as input, predict control, average as input, two-step correction, DNA source, DNA off, covariate embedding variants, depth-controlled input, other losses) on conv autoencoder, conv + attention and U-Net".
- The constitution lines behind them (`peeling the onion.md`): :132 "ablation: experiment with adding chip controls (upon availability for those celltypes) as one assay"; :146 "ablation: also predict the chip control"; :139-140 "average baseline: for each assay type, we can get its average activity along the genome (training set only)"; :124 the two-step correction; :156-159 the three DNA sources; :155 "covariate embedding and covariate injection, DNA embedding and injection, etc."; :160 "as an ablation, the count input data to the model can be depth controlled"; :153 "ablations: any other loss/regularization/etc."
- `long-horizon decisions.md`:113 (D15): "**Arm "fixed reference depth".** The base is thinned to one reference depth per assay class."

Success criterion served: 4 (design settled and tested). Audit rows: D4 to D12, D13.

## 2. Question

Each of the nine ablations answers its own question; the questions, arms, leads, expectations and outcome tables are in the nine specifications. This item delivers their results under D13 and closes the three arms that are held or blocked.

## 3. State today

- Specifications (do not repeat them here): `run_state/ablation_specs/` `1_control_input.md`, `2_predict_control.md`, `3_average_input.md`, `4_two_step.md`, `5_dna_source.md`, `6_dna_off.md`, `7_cov_embedding.md`, `8_depth_input.md`, `9_losses.md`.
- Submitted on Fir 2026-11-16T21:10Z: 14 arms x 3 architectures x 3 seeds = 126 runs, 756 jobs, ids 4437139 to 4438442, run directories `runs_fir/abl_<ablation>_<arm>_<convae|convattn|unet>`, scored into `p16_sv` (plan Log). Never resubmit.
- The 14 arms (Fir, read by this agent): average_input; control_output_only; cov_dropout, cov_numeric_only, cov_numeric_per_value; input_depth_fixed; dna_off; dna_shift5mb; loss_bin_equal, loss_impute_x4, loss_nb_crps, loss_poisson_mean; predict_control_off; two_step_native.
- At this agent's read: 42 run directories, 0 trainings DONE, 0 scored; Fir queue 126 running, 642 pending.
- Held: `two_step_fixed` (one reference depth 2^23 for all assays; D15 says one per assay class). Blocked: `dna_flashzoi`, `dna_gnlm` (embeddings on Nibi only, 31 GB each; Flashzoi also needs a fold-3 evaluation set, ruling 26). Open PI questions from the lead: move the embeddings by Globus; the reference depth; whether adaLN-Zero injection is wanted (plan Log 2026-11-16T21:10Z).
- Gap: DNA injection, i.e. how DNA is fused (constitution :155), has no arm (audit row D13).

## 4. Prior leads

In each specification, section 3. The common scale: final-round seed ranges of the three references, `FINAL_ROUND_EQUAL.md`: composite 0.045 / 0.048 / 0.010; correction 0.012 / 0.031 / 0.071 (conv autoencoder / conv + attention / U-Net).

## 5. Work

1. Watch the 756 jobs to completion; any failed job is reported, not resubmitted, without a foreman decision.
2. After the PI answers: rewrite `two_step_fixed` with one reference depth per assay class, then submit 9 runs; move the two embedding sets to Fir by Globus (PI action), build the fold-3 evaluation set, then submit `dna_flashzoi` and `dna_gnlm` (18 runs).
3. Read each ablation with `select_rule.py arm-summary --runs runs_fir --sub p16_sv` (composite version 3) against its reference `rung{2,3,4}_base_heldout`; apply the outcome table of its specification.
4. Log each D13 outcome in the notepad (screening tier until E1), and record a finding per ablation in the crux tree.
5. Report the comparison of `average_input` with the archived blended-CANDI number where comparable (notepad L408).

## 6. How the result answers the question

Each specification's section 5 holds the outcome-to-conclusion table, judged against the larger seed range (max minus min over 3 seeds) of the two arms. Acceptance: one row per arm and architecture in a results table (`results/<task>/ABLATIONS_READ.md`) with composite, the three terms and correction-only, each mean (range), and the D13 verdict.

## 7. Dependencies, cost, risks

- Depends on nothing for the 14 submitted arms; the 3 held or blocked arms need PI decisions.
- Cost: 262 CPU-h per one-arm ablation over 3 architectures (`1_control_input.md` section 7); 14 arms is about 3,700 CPU-h, two-step arms about twice (two forward legs). DNA arms: about 250 CPU-h at one rung (`COVERAGE.md` row 24).
- Risk 1: every verdict is screening tier until E1 settles the pair-region count.
- Risk 2: the references and the arms must share kit and scorer keys; `arm_report.py` refuses otherwise.

## 8. Done when

All 17 arms (14 submitted, 3 held or blocked) are scored on 3 architectures x 3 seeds, or a PI ruling drops an arm, and each D13 outcome is in the notepad and the crux tree.
