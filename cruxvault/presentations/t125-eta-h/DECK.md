# Depth offset and map h: lab deck

Artifact: https://claude.ai/artifact/AfLpe1acvSC9i43jg28AqS (version 1 published 2026-10-17; version 2, 37 slides, is final; private)

Source: `index.html` beside this file, built by `build_deck.py` from `deck_src.html`. Every number comes from `cruxvault/results/t125/` (`part1/`, `part2/`, `GRID_SUMMARY.md`), which is gitignored and per-worktree. The capacity-grid reference is `cruxvault/results/row2/`. The design is in `plan/ETA_OFFSET_AND_H_DESIGN.md`, and the run record is in `plan/T125_STATUS.md`.

State: final after the PI stop of 2026-10-19. Part 1 is complete. Part 2 covers 5 of 8 chain designs (C, D, A2, C2, D2) on track C19M16. Commit bf5ba11 adds an "ignores depth = 3.0" line to both depth-law figures in the local `index.html`; that change is not published.

The study is exploratory, so the deck gives readings and no verdicts. The readings live in the vault questions [[q5_does_a_depth_offset_on_f_the_source_dept|on the depth offset]] and [[q6_how_closely_does_a_map_h_reproduce_the_l|on the map h]].

## Arc (37 slides, with backup slides at the end)

- **Motivation.** Can a model turn one track into another from its processing settings alone? The earlier capacity grid had two problems: no design follows the depth law, and some −log10 p pairs blow up. Each problem has a likely cause and one idea to fix it.
- **Background.** 7 tracks in 3 mark classes are rebuilt with one setting changed at a time. A small network g reads the settings and builds the transformation f. Four designs of f run in two rows and with two versions of g. Scoring uses CRPS with its oracle-scaled and scale-error parts, the MSE of the mean, and a bar of 2 × the larger seed wobble.
- **Method, part 1.** f works on counts divided by the read depth, so "do nothing" means "scale by d′/d". The whole grid runs again with the offset, plus design A with its old spread (432 runs). One control (the no-covariates twin, which keeps the offset) and four comparisons each isolate one change.
- **Method, part 2.** h turns every possible count into a p-value, weighted by f's NB, so the mean and CRPS are exact. Eight versions of h run, from a fixed formula to a network with all inputs.
- **Results.**
  - (a) The depth law still does not hold: no design is within 10% on depth-only pairs.
  - (b) The offset helps design A beyond the bar, mostly in scale error; the other designs do not clear.
  - (c) The settings still improve the predicted mean beyond the fixed d′/d scale, in 54 of 54 cells.
  - (d) Push-only h: training h on f's predictions does not repair f's errors (30 of 30 rows).
  - (e) In the chain on designs D and D2, every h is above the capacity grid's direct p result, with no chain blow-ups.
- **Discussion.** The offset helps A but not the depth law. Push-only h is stable but limited by f's count errors. Three main limits and six more are listed: 5 of 8 designs, one track and one f seed, the merged mixture, own-noise blow-ups, mixed training devices, and the step cap. The next steps are options, not decisions.
- **Backup.** Glossaries, the spread rule, the two causes in detail, every depth-law cell, every cell against the capacity grid and the twin, h on the true counts by class, the repair rows, own-noise blow-ups, training devices, and the merged-mixture check.
