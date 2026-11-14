---
id: q33
type: question
schema: 3
title: Can the negative binomial be written with the depth-corrected eta (mean = depth x exp(eta)) throughout the model design, the variance decomposition and the law of total variance, and does decomposing the variance in eta space instead of on counts help, and how?
parent: q18
created: "2026-11-14T16:36:44"
updated: "2026-11-14T16:36:44"
---

# q33 — Can the negative binomial be written with the depth-corrected eta (mean = depth x exp(eta)) throughout the model design, the variance decomposition and the law of total variance, and does decomposing the variance in eta space instead of on counts help, and how?

Parent:: [[q18_how_large_is_each_layer_of_the_variance_]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

PI question of 2026-11-14 (long-horizon notepad, after ruling 47). Notes: results/eta_space/THEORY.md, LAYERS_ETA.md, MODEL_DESIGN.md, LITERATURE.md. It spans the variance layers (this parent), the evaluation (the correction floor of q19 and q32) and the model design.

## Bears on

If the layers above read sampling are depth-free in eta space, the per-assay layer sizes of the parent can be stated once for any depth and compared across assays, with sampling as a computed term instead of a thinned measurement; if not, the parent's log1p layer sizes hold only at the depth where they were measured.

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
**1 children** · open 0, leaning 1, settled 0, parked 0 · 3 finding(s) on this question

- `q34` _(Q)_ [[q34_what_does_the_eta_formulation_change_in_|What does the eta formulation change in the miniCANDI model: is the depth offset already in it, and would a hard offset, a dispersion tied across depth, or an eta-space readout of the model help?]] — *leaning · moderate*
- `f61` finding (moderate) [[f61_exact_algebra_with_a_simulation_check_th|Exact algebra with a simulation check that recovers the true layers;…]]
- `f62` finding (moderate) [[f62_measured_on_all_11_assays_with_a_code_ch|Measured on all 11 assays with a code check that reproduces the log1p layers…]]
- `f63` finding (moderate) [[f63_read_from_eta_layers_by_assay_tsv_with_9|Read from eta_layers_by_assay.tsv with 95% cluster-bootstrap intervals over…]]
<!-- crux:ledger:end -->
