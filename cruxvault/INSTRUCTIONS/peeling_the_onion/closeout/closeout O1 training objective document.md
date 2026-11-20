---
title: "O1: the settled training objective, as one document"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# O1: the settled training objective, as one document

## 1. What the PI asked

- `peeling the onion.md`:149-153 (five list lines, joined here with "/"): "objective(s): core loss: NB NLL on the output predicted NB distribution / on masked assays (imputation) / on available assays (reconstruction/denoising) / ablations: any other loss/regularization/etc. that can help with our three main objectives: 1) imputation 2) covariate artifact correction 3) calibrated uncertainty estimation"
- `peeling the onion.md`:49: "given the covariates (of both data generation and processing), we want to be able to go from signal obtained from one covariate set A to the same signal as if it was obtained from another covariate set B."
- `peeling the onion.md`:93: "we can also use the same data views as augmentations of the data and use it for training."
- `long-horizon decisions.md`:56 (D7): "The designed views are training data, i.e. augmentations."

Success criterion served: 2 ("have a training objective well defined").

## 2. Question

Write one document that states, for the settled miniCANDI, exactly what it is trained to do: the loss, the masking and view sampling, what is predicted on which targets, and the regularization. Each choice cites the ablation result that settles it, or is marked "default, not tested".

## 3. State today

The objective exists only in code and in scattered results:
- Loss (`tools/minicandi/losses.py`, `total_loss`; summary in `ablation_specs/9_losses.md` section 1): NB negative log-likelihood (NLL) of the target counts, averaged over the bins of `contract.loss_mask`. Kind 1 (imputation: whole track hidden, its C hidden) and kind 2 (correction: input view and target view differ) count on every bin; kind 3 (same view) counts on the hidden bins only. Weights `impute`, `correct`, `same` all 1. No loss regularizer; weight decay 0.01 and gradient clipping 1.0 (`train.py` L124-125).
- Sampling (`tools/minicandi/dataset.py` L233-240 and docstring L23-31): one cell per window of 1,024 bins; a share of tracks in `mask_prob_range` (0.2, 0.8) masked whole; for each other track one input view and one target view; `same_view_prob` 0.5; `bin_mask_frac` 0.25 with random bins; withheld (source, target) view combinations never sampled; control is an input and a target by default.
- Mean parameterization: mean = depth x 2^eta with depth also read as a covariate (notepad L564; rulings 48, 49).
- No document joins these. The two-step leg (`two_step.py`) is off by default.

## 4. Prior leads (each settles or informs one choice)

- Views: `results/views_ablation/VIEWS_ABLATION.md` and `correction_metric_review/REVIEW.md` addendum: under composite version 3 the view arms are within the seed range (0.498 to 0.517 against base 0.495); same-view pairs only gave gap closed below 0 in 3 seeds. Informs `same_view_prob` and `train_views`.
- Depth: hard offset at the U-Net, composite 0.444 to 0.260; hiding depth on the input side 0.382; on the output side 0.455, within range (`COVERAGE.md` rows 27-29; `VIEWS_ABLATION.md`). Settles: depth stays a covariate on the input side.
- Losses (ablation 9 of A1) and two-step correction (ablation 4 of A1): running on Fir, 3 architectures x 3 seeds.
- Predict control and control as input (ablations 1, 2 of A1); mask token or drop (A2).
- Matched compute and windows: 160,000 windows lowered held-out loss but not the composite (finding f88).

## 5. Work (the foreman writes the document from results)

Required contents, each a section with a cited result:
1. **Inputs read by the loss.** Targets, kinds 1 to 3, which bins count, control as target or not (ablation 2).
2. **Loss.** Per-bin form (NB NLL, or the ablation-9 winner), term weights, reduction, the two-step base leg if ablation 4 wins.
3. **Masking.** Whole-track mask range, bin-mask fraction and shape, mask token or drop (A2), mask scheme.
4. **View sampling.** Which view kinds are training data (A3, views ablation), `same_view_prob`, withheld combinations, base view only or all views.
5. **What is predicted on which targets.** The 36-track output panel, the NB (mean, size) per bin, the depth offset form, the covariates in C and C' (D1, A4).
6. **Optimization.** Learning rate, warm-up, weight decay, clipping, windows or compute budget (A9).
7. **Not settled.** Every choice that no ablation tested, marked "default, not tested", with the reason.
8. **Evaluation link.** For each goal, the headline metric that measures it (D13 terms).

## 6. How the result answers the question

Acceptance checks:
- The document `cruxvault/INSTRUCTIONS/peeling_the_onion/onion training objective.md` exists and has the 8 sections above.
- Every choice cites a finding id or a results file with its number and seed range, or carries the label "default, not tested".
- Each config key named in the document exists in the final config of M1, and the values agree (a script diff, keys against the document's table).
- No number in the document lacks its source path.

## 7. Dependencies, cost, risks

- Depends on ablation 9 (losses) and ablation 4 (two-step) of A1, and uses A2, A3, A4, A8, A9 where they finish. The index writes these dependencies as "A9 (losses), A4 (two-step)"; items A9 and A4 of this list are matched compute and QC input, so the index means the ablation numbers inside A1.
- Cost: about 1 agent-day after the results.
- Risk: results are screening tier until E1; the document must state the tier of each cited number.

## 8. Done when

`onion training objective.md` holds the 8 sections, each choice is tied to a cited result or marked untested, and its keys match the M1 final config.

## 9. Specification as worked (2026-11-17, item agent)

- Deliverable: `cruxvault/INSTRUCTIONS/peeling_the_onion/onion training objective.md`. It holds the 8 sections of section 5 (as sections 2 to 8, with OPEN choices in section 9 and untested defaults in section 10). No compute; no code change.
- Source of every value: the code at 01248f4 (file and line cited), the base configs `tools/minicandi/configs/rung{2,3,4}/base_heldout.json`, and the Fir base `run.json` files. Evidence: `results/ablations_closeout/ABLATIONS_READ.md` and `arm_summary_p16_sv_2026-11-17.tsv` (foreman worktree), 16 regions, screening tier.
- Correction to section 3: the sampler defaults are at `dataset.py`:238-246, not 233-240.
- Key check: all 34 config keys named in the document are read by the code (`train.code_keys`), and the 14 that the base configs hold have the stated values.
- Not yet met: "keys match the M1 final config" (M1 has not started); the 256-region rescoring and the mix-and-match round are OPEN in the document.
