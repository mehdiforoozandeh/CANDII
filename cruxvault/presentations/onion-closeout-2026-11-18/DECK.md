# Onion Close-out: the close-out deck of the peeling-the-onion run

Artifact: not published yet (the foreman publishes it, then adds the URL here and a row to `../ARTIFACTS.md`).

Source: `index.html` beside this file, built by `build_deck.py` from `deck_src.html` (prezit paper arc, 21 slides). One part per success criterion of the PI (data and where the counterfactual views help; training objective; evaluation; miniCANDI design), then a marked slide for the pending rescoring on all regions, what did not run, the ready queue, the open PI decisions, a summary and a glossary. The deck is derived, not evidence: no finding cites it. All slide text is drafted by the agent for the PI's review; the title slide says so.

## Where the numbers come from

`build_deck.py` reads every number from the files below and embeds it as DATA; figures are matplotlib PNGs built from the same files. The deck-sources manifest in `index.html` holds the sha256 of every file read (14 files). The speaker notes of each slide (press `n`) list that slide's files.

- `results/ablations_closeout/arm_summary_p16_sv_2026-11-17.tsv` and `ABLATIONS_READ.md` (the 16 ablations, screening set).
- `results/region_sampling_checks/check_v3.tsv` and `PAIR_REGIONS_V3.md` (the 16-versus-256 check; the all-region correction scores of the base recipes).
- `results/closeout_A3/where_vs_allviews.tsv` and `WHERE_VIEWS.md` (U-Net views by pair type).
- Tracked: `INSTRUCTIONS/peeling_the_onion/onion training objective.md`, `onion paper.md`, `closeout/settled miniCANDI design.md`, `closeout/ready queue for scaling.md`, `closeout/closeout todo.md`, `closeout/final 24 hours plan.md`.
- `results/` is git-ignored. The copies in this worktree are byte copies of the foreman worktree files of the same path, so the manifest verifies in either worktree.

## Rebuild

From inside the vault (`cruxvault/`). The `candii` env has no matplotlib; the base miniforge python has matplotlib and Pillow.

```bash
/Users/mforooz/miniforge3/bin/python presentations/onion-closeout-2026-11-18/build_deck.py
C="/Users/mforooz/miniforge3/envs/candii/bin/python /Users/mforooz/.claude/skills/crux/scaffold/crux.py"
${=C} deck --verify presentations/onion-closeout-2026-11-18/index.html --strict
${=C} deck --lint presentations/onion-closeout-2026-11-18/index.html
```

Verification on 2026-11-17: verify (strict) passed, 14 source files unchanged; lint clean. Every slide was rendered at 1600 x 900 in headless Chrome and checked by eye. `index.html` is 0.58 MB.

## Slides that depend on the 256-region rescoring

5 (views, correction-only term), 8 (loss arms), 12 and 13 (ablation overviews, columns b and c), 14 (design table), 16 (the marked placeholder "256-region rescoring: pending"), 20 (summary). Each carries a `depends` line in its contract header. When the rescoring lands: put the `p256_sv` arm summary under `results/`, read it in `build_deck.py`, replace slide 16 with the verdict of each rule of `settled miniCANDI design.md` section 2.0, and re-read the prose of the other dependent slides.

## Choices

- Figures are matplotlib PNGs (the foreman's brief), not inline SVG; each is quantized to a palette and is under 70 KB.
- The verdicts on the title slide (delivered, partial) are the agent's reading of the item statuses in `closeout todo.md`.
- Prose numbers are empty spans filled from DATA, so `--verify --strict` sees no typed numeral; criterion numbers (1) to (4) are marked as literals.
