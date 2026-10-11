# t125 status: the depth-offset count model and the counts → p map h

Orchestrator log for t125. Read `plan/T125_HANDOFF.md` first, then
`plan/ETA_OFFSET_AND_H_DESIGN.md`. The orchestrate plan is `.orchestrate/plan_t125.md` in this
worktree (git-excluded).

## State

| phase | state |
|---|---|
| 0 verify on Nibi | running (Nibi back up 2026-10-11 01:26 EDT) |
| 1 build the count model | planning |
| 2 build the map h | planning; waits on phase 0 for the formula |
| 3 smoke on Nibi | not started |
| 4 count runs (432) | not started |
| 5 h on the true X′ | not started |
| 6 h on f's NB, chain scores | not started |
| 7 aggregate | not started |
| 8 vault readings | not started |
| 9 lab deck | not started |

## Job ids

None submitted yet. Every id goes here at submission.

## Nibi access

- 2026-10-11: `ssh -o BatchMode=yes nibi` exited 255 (`Permission denied
  (publickey,keyboard-interactive,hostbased)`). All Nibi work stops until the PI runs
  `hpc up nibi`.
- 2026-10-11 01:26 EDT: `ssh -o BatchMode=yes nibi` returns `l2.nibi.sharcnet`. Nibi work resumes.

## Phase 0 findings

Not yet read.

## PI decisions

- 2026-10-11: learned MLP h = 2 × 64, with the same training settings as the t118 g (optimizer, learning rate, epochs, early stopping on chr22).

## Log

- 2026-10-11: run started; Nibi down at start.
- 2026-10-11 01:30: Nibi up. Phase 0 reader and the planner dispatched. Chunk ids carry the prefix `t125-`.
