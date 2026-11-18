---
id: q63
type: question
schema: 3
title: How well do the signal p-value track and the peaks of a ChIP-seq target come out of the predicted negative binomial, with the observed control and with the model's own predicted control, under the box-spread signal formula?
parent: q57
created: "2026-11-18T16:23:24"
updated: "2026-11-18T16:23:24"
---

# q63 — How well do the signal p-value track and the peaks of a ChIP-seq target come out of the predicted negative binomial, with the observed control and with the model's own predicted control, under the box-spread signal formula?

Parent:: [[q57_how_closely_can_a_formula_on_25_bp_binne]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

PI ruling 71 (2026-11-17): the signal formula spreads each count over a centered box of the fragment length (box_d), for the observed-control and the predicted-control routes; the in-bin form stays an explicit option. Open for the next session: the box_d rescoring of the 9 base runs under both dependence forms (comonotone and independent) plus the board references (slurm/onion_eval/e2_submit.sh); the rule (settled design row 31) is 3 seeds per model with signal-space scores.

## Bears on

Moves q57 from observed counts to model output: it says whether miniCANDI can deliver the pipeline's -log10 p track and peaks, and whether it can do so without an observed control (PI comment of 2026-11-17: a headline row for all 3 models).

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
**0 children** · open 0, leaning 0, settled 0, parked 0 · 1 finding(s) on this question

- `f126` finding (weak) [[f126_one_seed_of_one_model_in_a_form_now_know|One seed of one model, in a form now known to overstate the Poisson part;…]]
<!-- crux:ledger:end -->
