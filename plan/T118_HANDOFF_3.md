# t118 handoff 3 — execute the 2 × 4 grid (2026-09-28)

Written by the agent that specified and built row 2, for the agent that orchestrates the execution
phase. Read in this order, in full: (1) this file; (2) `plan/T118_ROW2_SPEC.md` — the row-2 design
and every PI ruling on it (§7 and the 2026-09-28 changes under it); (3) `plan/T118_STATUS.md` — row-1
results, the p-space explosion diagnosis, the row-2 build and smoke sections; (4)
`plan/T118_HANDOFF.md` §2 and `plan/T118_HANDOFF_2.md` §4 — hard rules, still binding. Where files
disagree, `plan/T118_COUNTERFACTUAL_F.md` (row 1) and `plan/T118_ROW2_SPEC.md` (row 2) win.

## 1. The goal

The PI's goal: **results for the 2 × 4 grid**, read cell by cell. Rows: g(C, C′) → f (row 1, one f
per source→target pair) and g(x, C, C′) → f (row 2, f changes per bin with the source value x).
Columns: designs A (affine), B (12-knot curve), C (kernel then curve), D (FiLM CNN); row 2 is A2–D2.
Plus a **shuffled-bin-value twin** for each row-2 design (PI 2026-09-28): the same design, size and
training, but g reads x from a randomly chosen other bin of the same source track, so a row-2 gain can
be put down to the bin's own level or to the extra input alone. Both readings interest the PI.

**The phase is exploratory (PI 2026-09-28): run everything, deliver the results, draw no
conclusions.** Do not file hypotheses, write nulls or checks, tick boxes, record verdicts, accept
tasks, approve syntheses, or merge to main. Draft readings for the PI only.

## 2. Where things stand

- Branch `exp/t118-counterfactual-f` (draft PR #48), worktree
  `/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/.claude/worktrees/counterfactual-f`.
  Row 2 is built and verified: forms `tools/t118/ladder/fforms/form_{a2,b2,c2,d2}.py`, per-bin g
  (`model.GBin`), `pairs.py tasks --row 2`, `train.py --row 2`, `aggregate.py --rows 1,2 --also-runs`,
  `report.py grid`, `slurm/t118/row2_{train,train_cpu,law,agg}.sh`. 428 t118 tests pass; row 1 is
  bit-exact (local golden smoke: 12 runs equal; on Nibi a finished row-1 run re-scored with the new
  code gives `RESCORE EQUAL`, 140 of 140 records).
- Build plan and log: `.orchestrate/plan_row2.md` in the worktree (git-excluded, local only).
  Chunks R1–R7 are merged; R8–R10 (the Nibi chunks) are yours. Chunk worktrees
  `/Users/mforooz/Desktop/research/libbrechteam@sfu/.orchestrate-wt/CANDII/row2-*` are merged; do
  not delete them without the PI.
- Nibi kit **K5** = c8cf98c at `/project/def-maxwl/mforooz/t118/ladder/code/K5` (the row-2 code; the
  later commits on the branch touch only `plan/`). Out dir `/project/def-maxwl/mforooz/t118/row2/`.
- **Smoke, submitted 2026-09-28 — never resubmit:** train+score array **22851946** (indices 54,
  198, 342, 486 = A2/B2/C2/D2 × C19M16 × counts × real × seed 0). At handoff: A2 done in 7 min, D2
  in 13 min (scoring 7.4 min), B2 and C2 trained (65 s, 176 s) and still scoring. Re-score check
  **22851953** done: `RESCORE EQUAL`. The one-off script is `row2/row2_rescore_check.sh` on Nibi.
- Inputs: products `/project/def-maxwl/mforooz/t112_cf/products` (MANIFEST.tsv md5 `599e2ca6…`),
  cache `/project/def-maxwl/mforooz/t118/ladder/cache` (read-only, 109 GB, keep), row-1 runs
  `/project/def-maxwl/mforooz/t118/ladder/runs` (576, all done — read, never rerun), refs
  `/project/def-maxwl/mforooz/t118/rungs_v2/rungs_v2.tsv`.

## 3. The work, in order

1. **Finish the smoke.** When 22851946 ends: read `sacct -o JobID,State,Elapsed,MaxRSS` and each
   run's `timing.json`; submit `row2_law.sh` directly (no dependency) for 54, 198, 342, 486; check
   `LAW_DONE`. The smoke ran only real-g runs, so also run a **no-covariates p-space smoke**: indices
   66, 138, 498, 570 (A2 and D2 × C19M16 and across-track `all` × pval × nocov × seed 0). The
   averaged map evaluates g over (training pairs × distinct x values) per chromosome, and in p space
   nearly every bin is distinct: this is the known cost risk. Record every time and memory in the
   status file. If a nocov run is far slower than its real twin, two fixes exist: an exact cache of
   the averaged map per chromosome (yours to build), or a grid of levels with interpolation (an
   approximation — the PI's call).
2. **Build the shuffled-bin-value twin** (a small orchestrated build; plan it first, chunks in
   worktrees cut from this branch). Contract to settle in the plan: a new model kind for row 2 only
   (row-1 table and outputs unchanged, bit-exact); g reads x at a randomly chosen other bin of the
   same source track, redrawn each training step, one fixed seeded draw at prediction and scoring;
   its own task table and **its own out dir** (for example `/project/def-maxwl/mforooz/t118/row2_xshuf/`)
   so the row-2 `tasks.tsv` check (exit 2 on mismatch) is never tripped; 192 runs = 4 designs × 8 g's
   × 2 spaces × 3 seeds; aggregation reads it with `--also-runs`, adds a check "row 2 beats its
   shuffled twin" with the standing bar (gain > 2 × the larger seed wobble) and a third grid row;
   tests, a local smoke, and the row-1 golden check. New kit **K6** for it.
3. **Full row-2 array** on K5, which does not wait for step 2: `sbatch --test-only` first, then
   `row2_train.sh` `--array=0-575%40` (the four smoke runs skip on `SCORE_DONE`), then `row2_law.sh`
   for the runs with `SCORE_DONE`, submitted directly. If the 10 GB slice nodes (g30–g37) show
   `mixed-` in `sinfo`, use `row2_train_cpu.sh`. Record every job id in the status file at submission.
4. **Twin array** (192) and its law array on K6, same pattern.
5. **Aggregate**, one job after another (they all write `agg/results.json`; chain with
   `--dependency=afterany:<previous>`): `row2_agg.sh … <row1_runs_dir> <rung>` for A2, B2, C2, D2, then
   `grid`, reading row 1 through `--also-runs` (and the twin runs once step 2 lands).
6. **Evidence and report.** rsync the reports, figures, checks and grid with
   `-rt -e 'ssh -o BatchMode=yes'` into the main checkout's `cruxvault/results/row2/` (gitignored) with
   a `FIR_PATH.txt` naming the Nibi dir; copy them into the worktree's `cruxvault/results/row2/` too.
   Then give the PI the grid: per design, row 1 vs row 2 vs the twin, per mark class, space and metric,
   and the count of exploding p-space pairs (reported, never pass/fail). Plain English, no tracker
   ids, every number beside its seed wobble. A published artifact also gets an md copy with its URL
   under `cruxvault/results/`.

Open, not in this phase unless the PI asks: the D explosion diagnosis on CPU (spec §7 (ix)); the
hand check of the depth-law computation (`plan/T118_HANDOFF_2.md` item 2).

## 4. Measured so far

Row-2 smoke on a 10 GB slice, C19M16 counts real seed 0: A2 7:03 total; D2 13:06 (train ≈ 4 min,
score 7.4 min); B2 train 65 s, C2 train 176 s (scores pending at handoff). Row 1 for comparison: ≈ 12
min per per-track run. The spec's estimate for the full row 2 (1.5 × row 1): about 215 slice-hours
(≈ 5.5 h wall at %40) plus about 430 CPU job-hours for the law test (≈ 11 h wall). Replace it with
the smoke's numbers.

## 5. Hard rules (short form; the full lists are in the two earlier handoffs)

- Never resubmit the 576 row-1 train or law arrays, the smoke 22851946 or the re-score 22851953.
- Nibi: one ssh at a time, always `ssh -o BatchMode=yes nibi …`; rsync with
  `-e 'ssh -o BatchMode=yes'` and `-rt` (not `-a`). **Exit 69 or 255 = tunnel down: stop Nibi work,
  write it in the status file, and wait for the PI to run `hpc up nibi`. Never authenticate.**
- No tool call over 240 s; poll long work with a background loop. The Bash shell is zsh: no unquoted
  `$var` word splitting; a bare `echo =====` fails.
- GPU jobs: only `--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`, `--account=def-maxwl`, job names
  `t118L_*`, `--exclude=c128,c166,c537`, always `--time`, no `--partition`, no `--export`, no
  `/usr/bin/time`, no `afterok`/`aftercorr` on a big array. Outputs under
  `/project/def-maxwl/mforooz/t118/`; `/scratch/mforooz` is over quota. Never touch `t112_*` jobs.
- A new kit for every code change that runs on Nibi; write `GIT_SHA`; copy the x-transformers wheel
  into `$KIT/wheels` from `K5/wheels`.
- Local python `/Users/mforooz/miniforge3/envs/candii/bin/python`; `export PYTHONPATH=$PWD/src` in any
  worktree (otherwise pytest tests the main checkout). No new dependency or environment change.
- Load the `orchestrate` and `dispatch` skills before spawning builders; cut their worktrees from
  `exp/t118-counterfactual-f`, never from `origin/main`. Load `slurm-hpc` and `hpc-workspace` before
  Nibi work.
- Merge `origin/main` into the branch as you go; commit and push often (end commits with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`). Keep `plan/T118_STATUS.md` updated and
  pushed: job ids at submission, times, failures, choices.
- Anything a reviewer would ask "why did you choose that?" about is the PI's call: bring it back.
