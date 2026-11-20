# Onion Close-out: the close-out deck of the peeling-the-onion run

Artifact: https://claude.ai/artifact/Kp1yqCgAUpCBDo19Q6oL9J (private; version 1 published 2026-11-17, with the 256-region rescoring slide pending).

Version 2 (2026-11-18, not yet published; the foreman publishes): the replicate ladder on the real data (slide 12), the second-experiment layers (13), the side deliverables (5: QC side tables, 1 bp region store, archive, freed scratch), the status of the 256-region rescoring (19), and the PI decisions of 2026-11-17 (22 decisions taken, 23 moved to the next session).

Version 3 (2026-11-18, not yet published): the replicate ladder is two-sided (slide 12). Version 2 showed the one-sided form and called it two-sided. Slide 12 now shows the two-sided values, the one-sided values beside them once (panel a and the note), the check on common cell types in place of telescoping, and the DNase-seq depth defect. Slide 13 says that its pooled rows are one-sided. The run-log facts come from `run_status.json`, no longer from the run log. Evidence: `results/closeout_E4/ladder_two_sided_2026-11-18/COMPARE.md`.

Version 4 (2026-11-18, not yet published): a new slide 6, the variance by bin size on the 1 bp region store (`results/closeout_A7/variance_2026-11-18/scale_variance.tsv` and `VARIANCE_READ.md`), with one figure and one paragraph. Slide 5 now says that the variance job has finished and the verification job of the 1 bp tables (Nibi 27692994) is still running. Every slide after slide 5 moves down by one, so the slide numbers below (version 3 and earlier) are one lower than in version 4; the rescoring placeholder is now slide 20. `run_status.json` carries `bp_store.variance_state` (`completed`); both builders stop on any other value.

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

Verification on 2026-11-18 (version 3, after review: joint bootstrap): verify (strict) passed, 26 source files unchanged; lint clean. Slides 12 and 13 were rendered at 1600 x 900 in headless Chrome and checked by eye.

Verification on 2026-11-18 (version 4): verify (strict) passed, 28 source files unchanged; lint clean. Slides 5 and 6 were rendered at 1600 x 900 in headless Chrome and checked by eye.

## Run-log facts (`run_status.json`)

Both builders (this deck and `../onion-closeout-2026-11-18-report/build_story.py`) read the run-log facts from `run_status.json` beside this file, not from `INSTRUCTIONS/peeling_the_onion/run_state/plan.md`. The run log changes at every snapshot commit, so a build that read it failed verify after each snapshot. To update the status (for example, the rescoring progress):

1. Edit the values in `run_status.json` by hand from the newest Log line of the run log (for the rescoring: `rescoring.status_day`, `status_utc`, `active`, `scored`). Keep each value as the run log writes it (a string; commas as written).
2. If the 1 bp verify and variance jobs are no longer running, set `bp_store.verify_state` and reword slide 5 first (the builders stop on any other value).
3. Rebuild both (`build_deck.py`, then `build_story.py` and `latexmk -pdf report.tex`) and run both verify steps with `--strict`.

## The 256-region rescoring: one command (version 5)

Version 5 (2026-11-18, not yet published): slide 20 is the result slide of the 256-region rescoring (one figure: per rule and model, arm minus base on the deciding metric with the seed-range bands; an arm that is not finished is drawn as "not finished (k of n seeds)" and gives no number), with a verdict line per rule. Slides 7, 10, 12, 16, 17, 18 and 25 (their `depends` lines) take their rescoring sentences from the same read. The report follows: section 3.5 (paragraphs and table), criteria rows 3 and 4, the abstract, the design table rows of rules 2.1 to 2.6, the discussion and the conclusion. The design table and the section "256-region rescoring: what it settled" of `settled miniCANDI design.md` are updated by the same command.

The foreman reruns, from the worktree root, with each new arm-summary table (`select_rule.py arm-summary --runs <runs_fir> --sub p256_sv` on Nibi):

```bash
export PYTHONPATH=$PWD/src
/Users/mforooz/miniforge3/envs/candii/bin/python tools/onion_eval/apply_rescoring_rules.py <arm-summary tsv> --as-of 2026-11-18T18:00Z
```

It applies rules 2.1 to 2.6 of the design document section 2.0 as written (a rule is decided only when every arm it names has its planned seeds), runs `mix_arms.py qualify` on the table, writes `verdicts.tsv`, `arms.tsv`, `verdicts.json`, `qualify.txt` and `RESCORING_READ.md` to `cruxvault/results/rescoring_p256/read_<UTC stamp>/` in this worktree and in the foreman worktree, copies the table to `cruxvault/results/rescoring_p256/arm_summary_p256_sv.tsv`, updates the design document and `run_status.json`, rebuilds the deck and the report (`latexmk -g -pdf`), and runs both verify steps with `--strict`. Its last line is `build and verify: all passed` or `FAILED`. `--no-build` skips the builds; `--dry-run` prints the verdicts only.

**How the builders find the evidence.** `run_status.json` `rescoring.read_dir` names the newest read folder; `rescoring.read_table` names the copied table. `build_deck.py` and `build_story.py` read `verdicts.json` from that folder and the table, and record both in their source manifests, so verify fails if either changes after a build. The run facts of the status text (start time, hours per scoring, the Fir rate, the U-Net limit and the Fir job range) are hand values in `run_status.json` `rescoring`; the counts (`scored_p256`, `arms_complete`, `rules_decided`) are written by the command.

Verification on 2026-11-18 (version 5, partial table of 16 of 62 run-seeds, scores as of 09:45 UTC): verify (strict) passed for the deck and the report; deck lint clean. Slides 7, 10, 12, 16, 17, 18, 20 and 25 were rendered at 1600 x 900 in headless Chrome and checked by eye.

## Choices

- Figures are matplotlib PNGs (the foreman's brief), not inline SVG; each is quantized to a palette and is under 70 KB.
- The verdicts on the title slide (delivered, partial) are the agent's reading of the item statuses in `closeout todo.md`.
- Prose numbers are empty spans filled from DATA, so `--verify --strict` sees no typed numeral; criterion numbers (1) to (4) are marked as literals.
