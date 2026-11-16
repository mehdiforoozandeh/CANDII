---
title: "A9: final tuning round, matched-compute runs"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# A9: final tuning round, matched-compute runs

## 1. What the PI asked

- `miniCANDI architecture ablations.md`:80, "Ablations at every rung": "Matched parameters, then matched compute."
- PI directive, plan Log 2026-11-15T20:05Z: "make the architecture-ablation findings trustworthy; every rung must have had a fair chance".
- PI ruling, plan Log 2026-11-16T08:30Z: "final round FULL, no trimming (equal windows + matched compute, about 1,660 CPU-h on Fir)".

Success criterion served: 4 (design settled, ready for scaling). Audit row: D16.

## 2. Question

When every architecture gets the same training compute (2.4e15 FLOPs, `tools/minicandi/configs/budgets.json`), does the order of the architectures on the composite and its three terms change from the equal-window order?

## 3. State today

- Equal-window final round read (`results/final_round/FINAL_ROUND_EQUAL.md`, composite version 3, 16 regions): conv autoencoder 0.528 (0.045), conv + attention 0.525 (0.048), U-Net 0.495 (0.010) to 0.511 (0.028) tuned, axial 0.517 (1 seed), Perceiver IO 0.335 (0.088). "No architecture had converged at 80,000 windows."
- Matched-compute trainings submitted on Fir 2026-11-16T12:15Z, train only, 2-day limit, never resubmit (plan Log): `final_rung2_n_levels3_matched` (933,156 windows), `final_rung3_attn_at_128_positions_matched` (855,436), `final_rung4_lr3e-3_matched` (1,443,487), `final_rung6_reference_matched` (896,179); 3 seeds each = 12 trainings. At this agent's read on Fir: 0 of 12 DONE, 0 scored.
- Axial: `rung5_matched_compute` reused (106,672 windows against an exact 106,515; Log 2026-11-16T13:10Z); 2 of 3 seeds scored in `p16_sv`: 0.508 (0.013).
- Pointwise MLP: `rung1_matched_compute` (24,565 windows), 2 seeds scored under composite version 2 only (`COVERAGE.md` row 9). Its rescore belongs to A10.
- Agent call (`long-horizon notepad.md`:289): "The per-rung ablation arms are not repeated at matched compute." No PI ruling confirms it (audit row D16).

## 4. Prior leads

- Finding f88 and `correction_metric_review/REVIEW.md` addendum: at 160,000 windows the held-out loss fell for conv autoencoder, conv + attention and U-Net, but the composite fell for the two conv models (0.528 to 0.498; 0.525 to 0.490, through the imputation term) and rose for the U-Net (0.495 to 0.520). Prediction: at 10 to 18 times the windows the conv models may fall further and the U-Net may rise, so the top of the order can change.
- `COVERAGE.md` row 9: the pointwise MLP at matched compute (30% of the windows) scored +0.027 on composite, beyond the range, all from calibration. Prediction: calibration is the term most sensitive to training length.
- Question q46: held-out NB loss ranks the architectures differently from the composite.

## 5. Work

1. Watch the 12 trainings; report any failure (2-day limit) to the foreman; do not resubmit without a decision.
2. Score each finished seed with the final-round pattern (`runs_fir/score_final.sh`, kit CM, `SCORE_SAME_VIEW=1`, `SCORE_SUBDIR=p16_sv`; module `python/3.10.13` only, `EBPYTHONPREFIXES` unset; Log 2026-11-16T16:50Z).
3. Score the third axial matched seed.
4. Write `results/final_round/FINAL_ROUND_MATCHED.md`: per architecture, equal-window and matched rows side by side, composite, three terms and correction-only, mean (range).
5. Ask the PI to confirm or reverse the narrowing of notepad L289.

## 6. How the result answers the question

"Beyond" = difference of seed means larger than the larger seed range (max minus min, 3 seeds); the D13 bar is 2 x that range.

| outcome | conclusion |
|---|---|
| same top group at matched compute as at equal windows | the order is robust to the budget; M1 picks from the top group by D13 and simplicity |
| an architecture moves up beyond the range at matched compute | the equal-window order favored fast learners; M1 uses the matched order for scaling |
| matched worse than equal for an architecture | more training hurts the composite (as at 160,000 windows for the conv models); report it as an open risk for scaling |

## 7. Dependencies, cost, risks

- Depends on nothing; M1 and O1 depend on it.
- Cost: 1,507 to 2,393 CPU-h for the matched round (`tools/minicandi/configs/final/MANIFEST.tsv`, Log 2026-11-16T08:12Z), mostly spent once the trainings finish; scoring about 5 jobs per seed.
- Risk 1: the U-Net matched run is 1,443,487 windows; at about 3,300 s per 80,000 windows on 12 CPUs (`COVERAGE.md`, cost unit) that is about 16.6 h, inside the 2-day limit only on a normal node.
- Risk 2: matched compute holds FLOPs equal, not passes over the data; at 1.4 M windows each region repeats many times.

## 8. Done when

`FINAL_ROUND_MATCHED.md` reports all five CPU architectures at matched compute with 3 scored seeds each, and the PI has ruled on the per-rung narrowing of notepad L289.
