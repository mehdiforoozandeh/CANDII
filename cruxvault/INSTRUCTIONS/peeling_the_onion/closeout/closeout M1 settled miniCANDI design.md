---
title: "M1: settle the miniCANDI design: one document, ready for scaling"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# M1: settle the miniCANDI design: one document, ready for scaling

## 1. What the PI asked

- `peeling the onion.md`:112: "miniCANDI is the prototype and we move on to CANDI when we are sure that miniCANDI prototype is well defined and working and we are ready to scale (miniCANDI -scale-> CANDI)"
- `peeling the onion.md`:162-165: "later on at scaling, we want to scale on three axes and observe the effects: data: more celltypes and assays / context: bigger context window / model size: bigger model with more parameters"
- `long-horizon prompt.md`:22: "we are satisfied with miniCANDI (scaling minicandi to candi is not a part of this session)"
- `long-horizon decisions.md`:103-105 (D14): the done rule, conditions 1 (rungs and per-rung ablations, 3 seeds, settled) and 2 (beats the strongest training-free baseline on each goal by more than 2 x the seed spread, on validation and then once on test).
- `long-horizon notepad.md`:334 (ruling 23): "different winning ideas from different rungs can have positive combination effect. do a few combination (mix and matches). select the combinations intelligently."

Success criterion served: 4 ("have the minicandi design settled and tested and ready for scaling in the next session"). Audit rows: F1 to F4, D18 to D20, D29.

## 2. Question

Which single miniCANDI configuration is the settled design, what is the evidence for each of its choices, and what must the scaling phase watch?

## 3. State today

- No design document exists. The evidence is spread over `results/final_round/FINAL_ROUND_EQUAL.md`, `views_ablation/VIEWS_ABLATION.md`, `correction_metric_review/REVIEW.md`, `ladder_audit/COVERAGE.md`, `ladder_tuning/ROUND2_READ.md`, and findings f86 to f89.
- Screening tier, composite version 3 (`FINAL_ROUND_EQUAL.md`): four architectures tie at 0.47 to 0.53 (f86); Perceiver IO last (0.335). U-Net with skips carrying DNA only: 0.533 (0.029), the highest mean; its correction gap closed 0.734 against 0.507 for the U-Net base (f87).
- Every model score is screening tier; no D13 outcome is settled (audit row F1).
- Combinations of winners (ruling 23) have not run (audit row D29).

## 4. Prior leads

The items this document reads: A1 to A10 (ablations), E1 (settled tier), D1 and D3 (the dataset version the design trains on), E2 (signal route), O1 (objective).

## 5. Work: required contents of the document

The foreman writes `cruxvault/INSTRUCTIONS/peeling_the_onion/miniCANDI settled design.md` from the results. Required sections:
1. **Inputs.** X (assays, control in or out), S (DNA source and form), C and C' (the covariate list: metadata, lab, antibody, donor, platform; QC on input or not), extras (average input or not). Each with its evidence.
2. **Outputs.** The NB (mean, size) per bin per assay; control predicted or not; the depth form of the mean; signal and peaks derived by E2.
3. **Architecture.** The chosen architecture and its internals (levels, attention placement, skip source, covariate injection and embedding, mask mode), with parameter count.
4. **Knob table.** One row per knob: settled value, the comparison that settled it, seed means and ranges, tier (screening or settled), D13 verdict, or "default, not tested" with the reason. Knobs: architecture; internals; DNA on or off; DNA source; covariate embedding; covariate injection; mask mode; control input; control output; average input; two-step correction (native or fixed base); depth handling; count transform; loss; views in training; QC on input; lab, antibody, donor; windows and compute; learning-rate recipe; resolution (25 bp, by ruling 28).
5. **Done rule.** Condition 1: list of comparisons and their settled verdicts. Condition 2: per goal, the margin over the strongest training-free baseline against 2 x the seed range, on validation; the test result comes from D4.
6. **Combinations.** The result of the few mix-and-match runs of ruling 23, if the PI keeps them in this session.
7. **Open risks for scaling.** At least: non-convergence at 80,000 windows; the composite falling with more windows for the conv models (f88); the correction term resting on 18 pairs in 6 cells; the blind measurement shift (platform, lab); the evaluation-region count (deferred to the follow-up).
8. **Config.** The final config json path and its hash.

## 6. How the result answers the question

Acceptance checks:
- The document exists with the 8 sections; every knob of section 4 has a row.
- Every number cites a results file or a finding, with seed range and tier.
- `diff` of the document's knob table against the final config json shows no disagreement.
- `select_rule.py check-done` (or the D14 check of `settlements_v3.tsv`) reports condition 1 and condition 2 for the final design.

## 7. Dependencies, cost, risks

- Depends on A1 to A10 and E1; uses D1, D3, E2 and O1.
- Cost: about 1 to 2 agent-days for the document; combinations, if run, 262 CPU-h per combination on 3 architectures, or 97 CPU-h on one (`ablation_specs/1_control_input.md` section 7).
- Risk: settled comparisons must share one dataset version and one evaluation-set version (notepad L465). If D1 or D3 changes the tables, the design's key comparisons are rerun on the new version before the document calls them settled.

## 7b. Specification as worked (2026-11-17, foreman brief of the final 24 hours)

- The deliverable is `closeout/settled miniCANDI design.md` (not the path of section 5). Its sections follow the foreman brief: (a) one design table with value, evidence, confidence and status; (b) for each provisional row the settling run, its cost and its pre-written rule, with a mechanical rule for the 256-region rescoring; (c) tested and untested code paths; (d) configs, scaling changes and risks; (e) what the design does not know.
- No new training runs before the deadline (`final 24 hours plan.md`). Correction-term effects stay screening tier (the 16-versus-256 check failed). Imputation and calibration in `p16_sv` are 256-region terms (`results/region_sampling_checks/REGION_CHECKS.md`).
- `select_rule.py check-done` was not run (no scores on the laptop); D14 condition 2 is computed by hand from `FINAL_ROUND_EQUAL.md` in section 5 of the deliverable.

## 8. Done when

`miniCANDI settled design.md` holds the 8 sections with every knob tied to evidence, its config matches, and D14 condition 1 and condition 2 on validation are stated as met or not met.
