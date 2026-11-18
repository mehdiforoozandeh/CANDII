# Onion Close-out: the close-out deck of the peeling-the-onion run

Artifact: https://claude.ai/artifact/Kp1yqCgAUpCBDo19Q6oL9J (private; version 1 published 2026-11-17, with the 256-region rescoring slide pending).

Version 2 (2026-11-18, not yet published; the foreman publishes): the replicate ladder on the real data (slide 12), the second-experiment layers (13), the side deliverables (5: QC side tables, 1 bp region store, archive, freed scratch), the status of the 256-region rescoring (19), and the PI decisions of 2026-11-17 (22 decisions taken, 23 moved to the next session).

Version 3 (2026-11-18, not yet published): the replicate ladder is two-sided (slide 12). Version 2 showed the one-sided form and called it two-sided. Slide 12 now shows the two-sided values, the one-sided values beside them once (panel a and the note), the check on common cell types in place of telescoping, and the DNase-seq depth defect. Slide 13 says that its pooled rows are one-sided. The run-log facts come from `run_status.json`, no longer from the run log. Evidence: `results/closeout_E4/ladder_two_sided_2026-11-18/COMPARE.md`.

Source: `index.html` beside this file, built by `build_deck.py` from `deck_src.html` (prezit paper arc, 25 slides). One part per success criterion of the PI (data and where the counterfactual views help; training objective; evaluation; miniCANDI design), then a marked slide for the pending rescoring on all regions, what did not run, the ready queue, the decisions taken and the decisions moved to the next session, a summary and a glossary. The deck is derived, not evidence: no finding cites it. All slide text is drafted by the agent for the PI's review; the title slide says so.

## Where the numbers come from

`build_deck.py` reads every number from the files below and embeds it as DATA; figures are matplotlib PNGs built from the same files. The deck-sources manifest in `index.html` holds the sha256 of every file read (23 files). The speaker notes of each slide (press `n`) list that slide's files.

- `results/ablations_closeout/arm_summary_p16_sv_2026-11-17.tsv` and `ABLATIONS_READ.md` (the 16 ablations, screening set).
- `results/region_sampling_checks/check_v3.tsv` and `PAIR_REGIONS_V3.md` (the 16-versus-256 check; the all-region correction scores of the base recipes).
- `results/closeout_A3/where_vs_allviews.tsv` and `WHERE_VIEWS.md` (U-Net views by pair type).
- `results/closeout_E4/ladder_two_sided_2026-11-18/two_sided/ladder.tsv`, `two_sided/common_cells.tsv`, `one_sided/ladder.tsv`, `COMPARE.md` (the replicate ladder in both forms, `ladder.py report`), and `results/closeout_E4/ladder_2026-11-18/pair_values.tsv.gz` (the DNase-seq common depth); `results/closeout_D2/layers_2026-11-18/layers.tsv` (second-experiment layers); `results/closeout_A4/qc_tables_2026-11-18/coverage.txt` (QC fill rates).
- Tracked: `presentations/onion-closeout-2026-11-18/run_status.json` (the run-log facts: QC fill, recount, archive, scratch, rescoring status; see "Run-log facts" below), `closeout/PI decisions 2026-11-17.md`, `closeout/closeout E4 ...`, `closeout D2 ...`, `closeout A4 ...`, `onion training objective.md`, `onion paper.md`, `closeout/settled miniCANDI design.md`, `closeout/ready queue for scaling.md`, `closeout/closeout todo.md`, `closeout/final 24 hours plan.md`.
- `results/` is git-ignored. The copies in this worktree are byte copies of the foreman worktree files of the same path, so the manifest verifies in either worktree.

## Rebuild

From inside the vault (`cruxvault/`). The `candii` env has no matplotlib; the base miniforge python has matplotlib and Pillow.

```bash
/Users/mforooz/miniforge3/bin/python presentations/onion-closeout-2026-11-18/build_deck.py
C="/Users/mforooz/miniforge3/envs/candii/bin/python /Users/mforooz/.claude/skills/crux/scaffold/crux.py"
${=C} deck --verify presentations/onion-closeout-2026-11-18/index.html --strict
${=C} deck --lint presentations/onion-closeout-2026-11-18/index.html
```

Verification on 2026-11-18 (version 2): verify (strict) passed, 23 source files unchanged; lint clean. The changed slides (1, 4, 5, 11, 12, 13, 19, 20, 21, 22, 23) were rendered at 1600 x 900 in headless Chrome and checked by eye. `index.html` is 0.73 MB.

Verification on 2026-11-18 (version 3): verify (strict) passed, 26 source files unchanged; lint clean. The figure of slide 12 was checked by eye; the slides were not rendered in a browser.

## Run-log facts (`run_status.json`)

Both builders (this deck and `../onion-closeout-2026-11-18-report/build_story.py`) read the run-log facts from `run_status.json` beside this file, not from `INSTRUCTIONS/peeling_the_onion/run_state/plan.md`. The run log changes at every snapshot commit, so a build that read it failed verify after each snapshot. To update the status (for example, the rescoring progress):

1. Edit the values in `run_status.json` by hand from the newest Log line of the run log (for the rescoring: `rescoring.status_day`, `status_utc`, `active`, `scored`). Keep each value as the run log writes it (a string; commas as written).
2. If the 1 bp verify and variance jobs are no longer running, set `bp_store.verify_state` and reword slide 5 first (the builders stop on any other value).
3. Rebuild both (`build_deck.py`, then `build_story.py` and `latexmk -pdf report.tex`) and run both verify steps with `--strict`.

## Slides that depend on the 256-region rescoring

6 (views, correction-only term), 9 (loss arms), 15 and 16 (ablation overviews, columns b and c), 17 (design table), 19 (the marked placeholder "256-region rescoring: pending", with a status line from the run log), 24 (summary). Each carries a `depends` line in its contract header. When the rescoring lands: put the `p256_sv` arm summary under `results/`, read it in `build_deck.py`, replace slide 19 with the verdict of each rule of `settled miniCANDI design.md` section 2.0, and re-read the prose of the other dependent slides.

## Choices

- Figures are matplotlib PNGs (the foreman's brief), not inline SVG; each is quantized to a palette and is under 70 KB.
- The verdicts on the title slide (delivered, partial) are the agent's reading of the item statuses in `closeout todo.md`.
- Prose numbers are empty spans filled from DATA, so `--verify --strict` sees no typed numeral; criterion numbers (1) to (4) are marked as literals.
