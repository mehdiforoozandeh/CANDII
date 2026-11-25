# Lab meeting deck, 2026-11-25: miniCANDI Progress

- Artifact URL: https://claude.ai/artifact/ECKjbssBLQy9gkqBGf7Pmf (private; version 1, published 2026-11-25).
- Source: `index.html` in this folder, built by `build_deck.py` from `deck_src.html` and the hashed copies in `data/`.
- Sources of the numbers: branch `implementation/t390-eic-pvalue-entry` at 6714498a (EIC boards) and branch `exp/t370-minicandi-recipe` at 4f4663f5 (read-start, recalibration, covariate-prompt results).
- Checks: `crux deck --verify --strict` and `crux deck --lint` were clean at build time.

Slides (15): title; the EIC validation board; fairness and disclosures; first of 11 on 45 experiments; rank per measure; read-depth cap; bottleneck (count overshoot, not conversion); what did not help (peak shrink, recalibration on the board); read starts against read counts; why read starts lost; recalibration of the spread; interval coverage at large peaks; covariate prompts; status and next steps (interim); glossary.

Caveats: the board numbers come from one model seed (seed 0). The next-steps slide restates status; it is not from a result file.
