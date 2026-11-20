---
title: "A10: pointwise MLP under the corrected composite"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# A10: pointwise MLP under the corrected composite

## 1. What the PI asked

- `miniCANDI architecture ablations.md`:5-7 (Rung 1, pointwise MLP): "The model predicts each bin from that bin alone." and "**What it adds.** Nothing. This rung is the floor."
- `long-horizon decisions.md`:104 (D14 condition 1): "Rungs 1 to 6 of [[miniCANDI architecture ablations]] and their listed per-rung ablations have run with at least 3 seeds, and the selection rule has settled each comparison."
- PI rulings, plan Log 2026-11-16T08:50Z: "composite switches to the 18 crop pairs with no duplicate-keeping side" and "add a correction-only term (one more prediction pass per run)".

Success criterion served: 4 (design settled and tested). Audit row: D17.

## 2. Question

Where does the pointwise MLP, the floor of the ladder, stand against the five context architectures under composite version 3 and the correction-only term, and do its per-rung arms change the answers they gave under version 2?

## 3. State today

Nibi `ONION/runs` (read by this agent, 2026-11-16): `rung1_base` 3 seeds DONE, 3 scored in `p16` (version-2 era), 0 in `p16_sv` (the same-view pass). The per-rung arms `dna_off`, `inject_concat`, `mask_token`, `matched_compute`, `scramble_all`, `scrambled_covariates` have 3 scored seeds in `p16`; `inject_tokens` 2; `average_input` 2 trained, 0 scored; `two_step_native` 1 trained, 0 scored. No rung-1 run has a `p16_sv` score.
- The five context architectures are scored in `p16_sv` on Fir (`FINAL_ROUND_EQUAL.md`); their references are `rung{2..6}_base_heldout`, retuned in two rounds. Rung 1 had no tuning round: its tuning trials needed a GPU and were not moved to Fir (Log 2026-11-16T04:10Z).

## 4. Prior leads

- `correction_metric_review/REVIEW.md` L150-152, Nibi bases recomputed with the 18-pair term: pointwise MLP 0.474, below conv autoencoder 0.538, conv + attention 0.519, axial 0.502, U-Net 0.499, above Perceiver IO 0.299. Prediction: the pointwise MLP stays in the lower group but within about 0.03 of the U-Net.
- `COVERAGE.md` row 1: rung 1 to rung 2 (local context) gave +0.058 on composite version 2, beyond the range but below the D13 bar (2 x 0.031).
- `ladder_audit/FAIR_CHANCE.md` (Log 2026-11-15T21:00Z): three tiers survived seeds, {conv autoencoder, conv + attention} > {axial, U-Net, pointwise MLP} > Perceiver IO.

## 5. Work

1. Compute composite version 3 for every rung-1 run from the stored `p16` files (`select_rule.py arm-summary --composite-version 3`; no new prediction).
2. Same-view pass for the correction-only term: `rung1_base` 3 seeds, then the arms in D13 comparisons. Score on CPU on Fir, as the other five architectures were (one device type per comparison, notepad L322): copy the checkpoints from Nibi (the hide-depth precedent, Log 2026-11-16T04:10Z) and time one job first (notepad L531).
3. Score the unscored seeds: `inject_tokens` seed 1 (metrics only; its prediction store was complete, `COVERAGE.md` row 14), `average_input`, `two_step_native` where trained.
4. Add the pointwise MLP rows to `FINAL_ROUND_EQUAL.md` (or its successor) and read the rung-1 arms under version 3; log each D13 outcome.

## 6. How the result answers the question

"Beyond" = difference of seed means larger than the larger seed range (max minus min, 3 seeds); the D13 bar is 2 x that range.

| outcome | conclusion |
|---|---|
| pointwise MLP below the conv models beyond the D13 bar | local context helps under the corrected composite; the ladder's first question is settled |
| within the bar of the conv models | context buys little at prototype scale (ablations file L9: "If it comes close to CANDI, genomic context buys little") |
| a rung-1 arm verdict flips between versions 2 and 3 | the version-2 correction term drove it; report the flip |

## 7. Dependencies, cost, risks

- Depends on a device for the same-view pass (Fir CPU after checkpoint copy, or a Nibi GPU).
- Cost: step 1 under 1 CPU-h. Rung-1 prediction on CPU is not measured on Fir; a Nibi GPU forward ran 8 shards of about 25 min (`COVERAGE.md`, cost unit); metrics about 11.5 CPU-h per seed (same source).
- Risk: rung 1 lacks the tuning round of the other architectures; state that it had a smaller chance.

## 8. Done when

`rung1_base` and its arms have version-3 composites and correction-only terms on 3 seeds, the pointwise MLP row sits in the final-round table, and each rung-1 D13 outcome is logged.
