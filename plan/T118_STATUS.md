# t118 status — architecture ladder A–D and design X (as of 2026-09-25, evening)

Design authority: `plan/T118_COUNTERFACTUAL_F.md`. First work order: `plan/T118_HANDOFF.md`. Next
agent's work order: `plan/T118_HANDOFF_2.md`. Nothing here is ticked or a verdict; every reading is
a draft for the PI.

## State in one paragraph

All 576 pre-registered runs (4 designs × 3 models × 8 g's × 2 spaces × 3 seeds) are trained, scored
on chr19 + chr21, law-tested on the 2 094 never-trained arm → arm pairs, and aggregated. Each design
has a report, 9 figures and a checks file in `cruxvault/results/` (main checkout; gitignored), linked
from its hypothesis. The main-claim readings are drafted. Two problems stand out: a few p-space
pairs make the mean CRPS explode in designs B, C and D, and the depth law and swap checks are unmet
everywhere. An exploratory test of the PI's idea (g also reads the bin value; "design X") removed
the p-space explosions on the two tracks tested but did worse than B–D in DNase counts.

## 2026-09-26 — row 2 of the grid (g reads the bin value) is being built

The PI set a new session goal: results for a 2 × 4 grid. Row 1 is g(C, C′) → f (the 576 finished
runs); row 2 is g(x, C, C′) → f, one transformation per bin, for each of A–D (new forms A2–D2).
Spec: `plan/T118_ROW2_SPEC.md` — the PI adopted every recommendation in its §7 (g reads the bin's
own value only; one rule for all forms; B2 is not forced monotone; smoke, then the full 576 row-2
runs; pre-register before compute; design X kept as a reference; D's explosion diagnosed in parallel
on CPU). Build plan: `.orchestrate/plan_row2.md` (git-excluded). Wave 0 (core harness, A2, B2, C2,
D2, figures/report) is being built in worktrees under
`/Users/mforooz/Desktop/research/libbrechteam@sfu/.orchestrate-wt/CANDII/row2-*`. No Nibi compute
yet; every compute step waits for the PI's yes. Row-1 golden smoke captured before any merge.

## Row 2 build

The row-2 code is built and tested locally. It follows the spec `plan/T118_ROW2_SPEC.md`. No row-2
run exists yet, so this section holds no results.

- The four new forms of f are in `tools/t118/ladder/fforms/`: `form_a2.py` (the affine map in log
  space, chosen per bin), `form_b2.py` (the 12-knot curve, chosen per bin; the map across bins is
  not forced monotone), `form_c2.py` (the 33-bin kernel, then the curve) and `form_d2.py` (the
  dilated CNN modulated by g).
- These files changed to carry row 2: `pairs.py` (`RUNGS_ROW2`, `tasks --row 2`), `model.py`,
  `base.py`, `train.py`, `aggregate.py` (`--rows`, `--also-runs`), `figures.py`, `report.py`
  (`report.py grid` writes the 2 × 4 grid to `<agg>/grid.md`) and `smoke.py` (it accepts A2–D2).
- The SLURM scripts are `slurm/t118/row2_train.sh`, `row2_train_cpu.sh`, `row2_law.sh` and
  `row2_agg.sh`.
- The new tests are `tests/test_t118_form_{a2,b2,c2,d2}.py` and
  `tests/test_t118_row2_{core,score,aggregate,figures,slurm,e2e}.py`. They add 190 tests to the 238
  row-1 tests, so `tests/test_t118_*.py` holds 428 tests.
- The row-1 forms, `data.py`, `encoding.py` and `tools/t118/covariates.tsv` did not change. The
  row-1 task table keeps its pinned md5, and a test pins it.
- `smoke.py <work_dir> A2` (or B2, C2, D2) runs every command of the chain on synthetic products on
  CPU and prints `SMOKE OK <rung>`. The end-to-end test runs it for A2.

## Row 2 Nibi smoke (submitted 2026-09-28, PI yes)

- Kit `K5` = c8cf98c at `/project/def-maxwl/mforooz/t118/ladder/code/K5` (with `GIT_SHA`, `wheels/`).
- Out dir `/project/def-maxwl/mforooz/t118/row2/` (logs in `logs/`).
- Smoke train+score array **22851946**, indices 54, 198, 342, 486 = A2, B2, C2, D2 × C19M16 × counts
  × real × seed 0, on 10 GB slices. Never resubmit; a rerun skips runs with `SCORE_DONE`.
- Row-1 re-score check **22851953** (`row2/row2_rescore_check.sh`, a one-off outside the repo):
  re-scores the finished row-1 run `A_C19M16_counts_real_s0` with K5 on a slice into
  `row2/rescore_check/` and prints `RESCORE EQUAL` or `DIFFER` in its log.
- Law tasks for the 4 smoke runs follow when their `SCORE_DONE` exists (submitted directly).
- State at 2026-09-28 21:15 UTC: re-score check done, `RESCORE EQUAL` (140 of 140 records; row 1 is
  bit-exact on real data). A2 done in 7:03, D2 in 13:06 (score 7.4 min); B2 and C2 trained (65 s,
  176 s) and still scoring.
- PI 2026-09-28: the phase is exploratory (run both rows, conclude later; no hypotheses filed) and a
  shuffled-bin-value twin is added for every row-2 design. The execution phase is handed off:
  `plan/T118_HANDOFF_3.md`.

## Row 2 execution (from 2026-09-28, handoff 3)

Smoke 22851946 finished, all four COMPLETED (sacct elapsed, MaxRSS; timing.json train seconds,
peak RSS, steps run / best step):

| run | elapsed | MaxRSS | train | peak RSS | steps / best |
|---|---|---|---|---|---|
| A2 C19M16 counts real s0 | 7:03 | 6.2 GiB | 53 s | 7.2 GiB | 4000 / 4000 |
| B2 … | 16:23 | 5.5 GiB | 65 s | 4.9 GiB | 1750 / 750 |
| C2 … | 13:58 | 6.5 GiB | 176 s | 6.8 GiB | 3750 / 2750 |
| D2 … | 13:06 | 6.0 GiB | 269 s | 7.0 GiB | 3750 / 2750 |

Scoring, not training, dominates the per-task time (5–15 min of each total).

Jobs (never resubmit a finished one; a rerun skips on SCORE_DONE / LAW_DONE):

| submitted (UTC) | job | what | kit |
|---|---|---|---|
| 2026-09-28 21:17 | 22853853 | law smoke, indices 54, 198, 342, 486 (`row2_law.sh`, CPU) | K5 |
| 2026-09-28 21:17 | 22853854 | no-covariates p-space smoke, indices 66, 138, 498, 570 = A2 / D2 × C19M16 / `all` × pval × nocov × s0 (`row2_train.sh`) | K5 |

| 2026-09-28 22:01 | 22855414 | full row-2 train+score, the 384 real and labels-as-ids tasks (index list `row2/idx_real_ids.txt` on Nibi), `%40`, GPU slices | K5 |

Slice nodes at 21:15: g30–34 and g37 `mixed` (usable), g35–36 `mixed-` (held). At 22:00 all of
g30–37 showed `mixed-`, but `--test-only` gave an immediate start on g30, so the array went to the
GPU (same device as row 1); fallback `row2_train_cpu.sh` if the first wave does not start.

Law smoke 22853853: all four COMPLETED — A2 5:28, B2 22:33, C2 12:41, D2 14:23 (8 CPU cores).

No-covariates p-space smoke 22853854 (22:00): per-track runs fine — A2 train 137 s (11:02
total), D2 train 467 s (25:06 total). Across-track runs slow: A2 train 2021 s (row-1 A across
nocov: 280 s); each chr22 validation of the averaged map takes about 225 s and dominates the
step time; D2 across had not finished its first validation after 36 min. Cause as predicted: the
averaged map is g evaluated over (training pairs × distinct x values), 242 pairs across tracks,
and nearly every p-space bin is distinct. So the 192 no-covariates tasks (index list
`row2/idx_nocov.txt`) are held until the averaged map is made cheap; all 192 will then run on one
kit. The four smoke nocov runs finished on K5 are counted as smoke only if the fix changes their
numbers.

22:15: cancelled 22853854_138 (A2 across, trained, scoring on CPU) and 22853854_570 (D2 across,
no first validation after 52 min): both would pass 3 h on K5 and the fix below replaces them.
Before the nocov array runs on the new kit, the four K5 nocov run dirs move to
`row2/runs_smoke_K5/` (A2_all_pval_nocov_s0 holds a K5 TRAIN_DONE that would otherwise be skipped).

Diagnosis (read-only agent, local timings; scratch `scratchpad/nocov_diag/`): validation loops
all 242 pairs and rebuilds the averaged map per pair, though pairs share about 128 sources;
scoring runs on CPU (score.py forces cpu) and rebuilds it per job (854 jobs; law 2 094 jobs over
121 sources). Three exact fixes, built as chunk N1: (F1) one (loc, disp) per (source,
chromosome), reused by every pair and job with that source — bit-identical; (F4) each source's
distinct values computed once per run for validation — bit-identical; (F3) g's last linear layer
taken out of the mean over pairs, theta_bar = W · mean(h) + b — exact up to float order (max
|Δ theta| 5e-7 to 2e-6). Estimated with all three: across-track nocov p runs 1.2–1.7 h total, law
34–40 min on 8 cores (today: scoring alone 2.9–5.8 h, law 4.3–8.8 h). No approximation needed.
Full array 22855414 at 22:15: 7 done, 13 running (13 slices free).

| submitted (UTC) | job | what | kit |
|---|---|---|---|
| 2026-09-28 22:33 | 22857243 | law, batch 1: the 29 real/ids runs with SCORE_DONE at 22:30 (indices in `row2/law_batch1.txt`) | K5 |

Shuffled-bin twin build: X1 (draw, training, prediction, task table), X2 (scorer tests), X3
(aggregation: twin rows, checks `beatsxshuf` and `lawtest_xshuf`, grid row 3), X4 (schema,
figures, reports), X5 (four `slurm/t118/xshuf_*.sh` scripts) merged at 2311208. Note for the PI:
`lawtest_xshuf` uses the bar 2 × the larger of the real and twin seed wobbles (as `beatsxshuf`);
the row-1 law checks against the other twins use 2 × the real g's wobble. Both twin checks are
written only when some twin run was read. X6 (smoke, end-to-end test) and N1 (the exact nocov
fix) are building.

Build done at c3196e0 (X1–X6, N1 merged; verifier: 590 t118 tests, 2 608 in `tests/`, goldens A
counts/nocov and A2 counts EQUAL; A2 nocov within float order: CRPS ≤ 4e-7 relative, Spearman on
the top 1 % up to 1.1e-2 absolute on the synthetic smoke's small top-1 sets — N1's F3 changes the
summation order only). Kit **K6** = c3196e0 at `/project/def-maxwl/mforooz/t118/ladder/code/K6`
(GIT_SHA, wheels from K5). Row-2 real and ids runs stay on K5: K6 is bit-exact on those paths
(goldens), so the kits do not mix within a model. All 192 row-2 nocov runs run on K6. The four K5
nocov smoke run dirs moved to `row2/runs_smoke_K5/` (23:20).

K6 smoke state at 00:29 (train seconds from train_log; totals from sacct):
nocov A2 C19M16 pval 109 s (25:08 total); D2 C19M16 pval 257 s (27:39); A2 across pval 630 s
(K5: 2021 s; same best step 1000, same val_nll to 7 digits), scoring; D2 across pval 1633 s (K5:
no first validation in 52 min), scoring. Twin A2 C19M16 counts 38 s (12:36); D2 C19M16 counts
286 s (26:47); A2 across counts 727 s, scoring; D2 across pval 1746 s, scoring.
Full array 22855414 at 00:28: 154 SCORE_DONE of 384, 17 left in the queue.

K6 smokes all COMPLETED by 01:03 (sacct elapsed; MaxRSS): nocov A2 per-track 25:08 (5.8 GiB),
A2 across 1:20:20 (5.6 GiB), D2 per-track 27:39 (7.5 GiB), D2 across 1:33:54 (7.4 GiB); twin A2
per-track 12:36 (14.4 GiB), A2 across 1:26:49 (15.6 GiB, pool 406 s), D2 per-track 26:47
(5.9 GiB), D2 across 1:20:16 (15.6 GiB, pool 534 s). All inside 3 h. The twin's peak memory sits at
its 16 000 MB request (the pool reads every training chromosome of the source pids), so the twin
array runs with `--mem=32000M` given on the sbatch line (no code change).

Law smokes all COMPLETED by 03:23 (elapsed; MaxRSS): nocov A2 per-track 7:33, A2 across 45:36,
D2 per-track 17:54, D2 across 34:55 (≤ 11.8 GiB); twin A2 per-track 40:09, A2 across 44:27, D2
per-track 21:41, D2 across 1:35:44 (16.9 GiB of 24 000 MB). Law batch 2 (22861310): all 119
COMPLETED.

From 03:24, later law arrays are submitted by a background loop on this laptop every 20 min
through the one-off `/project/def-maxwl/mforooz/t118/law_loop_remote.sh` (not in the repo), which
calls `/project/def-maxwl/mforooz/t118/law_todo.sh <out_dir> <models>`: runs with SCORE_DONE, no
LAW_DONE, not yet in `<out_dir>/law_submitted.txt`. Row-2 real/ids law runs on K5, nocov and
twin law on K6. The loop's job ids are copied into the table below.

**2026-09-29 03:27 UTC: the Nibi tunnel dropped** (ssh exit 255; `hpc status`: nibi down). The
law loop stopped on its first pass before submitting anything (the helper had not run). Nibi work
is stopped until the PI runs `hpc up nibi`. The jobs already queued keep running on Nibi. On
resume: rerun the law loop (it finds every scored run not yet in `law_submitted.txt`).
Tunnel back 05:50 (PI). Checked: `law_submitted.txt` counts (312 row 2, 17 twin) match the
jobs queued, so the dropped pass left nothing half-done. State at 05:51: main array 22855414
363 of 384 COMPLETED (21 D2 tasks pending); law batch 3 (22867594) all 156 COMPLETED; twin law
batch 1 (22867637) all 13 COMPLETED; the nocov (22862752) and twin (22862753) arrays pending on
Priority (fair-share after ~400 slice-jobs), slice nodes g30–37 `mixed`. Law loop restarted.

| submitted (UTC) | job | what | kit |
|---|---|---|---|
| 2026-09-29 05:51 | 22879392 | law loop: 55 row-2 real/ids runs | K5 |

Law loop passes 06:12–08:53: nothing new scored (all three train arrays still queued). **The
tunnel dropped again** (the 11:25 UTC pass: ssh exit 255, "Permission denied"; `hpc status` at
12:10: fir and nibi down). Nibi work stopped until the PI runs `hpc up nibi`; queued jobs keep
running on Nibi.

Tunnel back 19:27 (PI). State: main row-2 array 22855414 all 384 COMPLETED; law 22879392 all
55 COMPLETED; twin array 22862753 174 of 192 COMPLETED; nocov array 22862752 has started no task
yet: both arrays sit on Priority and the twin's larger memory request gives it a slightly higher
priority (1016665 vs 1016662, the TRES memory term), so the nocov tasks start once the twin's
last 18 run. Law loop restarted 19:28.

| submitted (UTC) | job | what | kit |
|---|---|---|---|
| 2026-09-29 19:30 | 22936307 | law loop: the last 21 row-2 real/ids runs (D2) | K5 |
| 2026-09-29 19:30 | 22936365 | law loop: twin law, the scored twin runs 13–173 not yet sent (smoke indices excluded) | K6 |

Passes 19:51–20:32: nothing new to send (twin and nocov arrays still queued). **Tunnel dropped a
third time at 20:55 UTC** (ssh 255; `hpc status`: nibi down). Nibi work stopped.

Tunnel back 21:22 (PI). Twin array 184 of 192 COMPLETED; nocov array still queued behind it. 10
law tasks failed in seconds, all on node c371, with the known CVMFS "Input/output error" on
`virtualenv` (22936307_547; 22936365_14, 55, 56, 57, 59, 60, 63, 64, 68). Resubmitted with
`--exclude=c128,c166,c537,c371`; the law loop now passes that exclude list too. Loop restarted.

| submitted (UTC) | job | what | kit |
|---|---|---|---|
| 2026-09-29 21:23 | 22941573 | law retry: row-2 index 547 | K5 |
| 2026-09-29 21:23 | 22941574 | twin law retry: twin indices 14, 55, 56, 57, 59, 60, 63, 64, 68 | K6 |
| 2026-09-29 21:23 | 22941577 | law loop: twin indices 174, 175, 177, 179, 180 | K6 |

Law loop passes 21:43 → 06:08 (30 Sep): nothing new scored for about 9 h; the twin's last
tasks and the whole nocov array stayed queued (TRAINQ 2 throughout). **The tunnel dropped a
fourth time** (the 14:39 UTC pass on 30 Sep: ssh 255, "Permission denied"). Nibi work stopped. On
resume: find why the two GPU arrays do not start; if the slice nodes are held, the CPU fallback
`row2_train_cpu.sh` / `xshuf_train_cpu.sh` is the handoff's rule.

Tunnel back 2026-10-01 19:05 (PI). The stall resolved itself while the tunnel was down: nocov
array 22862752 all 192 COMPLETED, twin array 22862753 all 192 COMPLETED; every law retry
(22941573, 22941574) and 22941577 COMPLETED; nothing queued. Files: `row2/` 576 SCORE_DONE, 388
LAW_DONE (384 real/ids + 4 nocov smoke); `row2_xshuf/` 192 SCORE_DONE, 185 LAW_DONE.

| submitted (UTC) | job | what | kit |
|---|---|---|---|
| 2026-10-01 19:06 | 23073675 | nocov law: the 188 remaining row-2 nocov runs (exclude c128, c166, c537, c371) | K6 |
| 2026-10-01 19:06 | 23073676 | twin law: the last 7 twin runs, indices 184–188, 190, 191 | K6 |

Both COMPLETED by 22:38 (longest nocov law 2:09:29). Every run is scored and law-tested:
`row2/` 576 SCORE_DONE / 576 LAW_DONE; `row2_xshuf/` 192 / 192.

Aggregation chain (`xshuf_agg.sh`, K6, reads row-2 runs + row-1 runs `ladder/runs` + twin runs
`row2_xshuf/runs`, writes `row2/agg/`; refs `rungs_v2/rungs_v2.tsv`; each after the previous with
`afterany`):

| submitted (UTC) | job | what | kit |
|---|---|---|---|
| 2026-10-01 22:40 | 23087020 | aggregate + figures + report, A2 | K6 |
| 2026-10-01 22:40 | 23087026 | B2 (after 23087020) | K6 |
| 2026-10-01 22:40 | 23087028 | C2 (after 23087026) | K6 |
| 2026-10-01 22:40 | 23087033 | D2 (after 23087028) | K6 |
| 2026-10-01 22:40 | 23087035 | grid (after 23087033) | K6 |

A2, C2 and grid COMPLETED (1 344 runs present, 0 missing). B2 and D2 FAILED in `figures.py`
fig3 (law grid): 4 of 134 016 law-grid cells have no finite CRPS in any of the 3 seeds (`d =
None`): B2 per-track p space, real g, 3 cells; D2 per-track p space, labels-as-ids twin, 1 cell.
The aggregation itself finished, so `results.json`, checks and `grid.md` are complete; only the
B2 and D2 figures and reports were missing. Fix 220f671 (figures only: such a cell is drawn
blank; regression test added; no number changes). Kit **K7** = 220f671 (differs from K6 only in
`figures.py`).

| submitted (UTC) | job | what | kit |
|---|---|---|---|
| 2026-10-01 23:09 | 23088152 | aggregate + figures + report, B2 (rerun) | K7 |
| 2026-10-01 23:09 | 23088156 | D2 (rerun, after 23088152) | K7 |

Both K7 reruns drew all 9 figures, then FAILED in `report.py` section_law on the same `None`
cells (a second subtraction I had not searched for). Fix 33e09cb (a pair with no finite CRPS is
left out of that gain's mean; pair count unchanged; `n/a` if none left; reports only). Kit **K8**
= 33e09cb (differs from K6 only in `figures.py` and `report.py`). Before submitting, all four
reports were built on the login node from the real `results.json` without writing: all build; a
fresh A2 report equals the stored one except the `results.json written` timestamp.

| submitted (UTC) | job | what | kit |
|---|---|---|---|
| 2026-10-01 23:27 | 23088586 | aggregate + figures + report, B2 (second rerun) | K8 |
| 2026-10-01 23:27 | 23088588 | D2 (second rerun, after 23088586) | K8 |

Both COMPLETED (4:27, 5:39). All four row-2 reports and 9 figures each exist; `grid.md`
unchanged (md5 3cbf7c40…).

**Delivered 2026-10-01.** Evidence copied to the main checkout's `cruxvault/results/row2/` and
to the worktree's copy (gitignored; 61 files, 23 MB): `grid.md`, `GRID_SUMMARY.md` (drafted
readings, check tallies, every cell of rows 1, 2 and the twin with its seed wobble, exploding
counts), `checks_*.json`, `results_summary.tsv`, `qm_curves.json`, `<rung>/report.md` and
figures for A–D and A2–D2, and `FIR_PATH.txt`. `results.json` (373 MB with
`per_pair_rest.jsonl.gz`) stays on Nibi. No hypothesis filed, nothing ticked, no verdict.

**2026-10-02: lab deck, vault update, follow-up tasks.**

- **Lab deck** (26 slides; motivation, background, method, results, discussion), published at
  https://claude.ai/artifact/1mFz6C5jLaG1Ctc8JN572C (private until the PI shares it; version 4).
  Source in `cruxvault/presentations/t118-grid/`: `deck_src.html` (slides and chart code),
  `build_deck.py` (embeds the numbers from `cruxvault/results/row2/` and writes `index.html`),
  `snippet.json` (a real 10 kb H3K27ac snippet, chr19:12,732,350–12,742,350, read from the t112
  products on Nibi). To edit: change `deck_src.html`, run `python
  cruxvault/presentations/t118-grid/build_deck.py`, then republish `index.html` to the same URL.
  Pointer file: `cruxvault/results/row2/DECK.md` (gitignored). PI edits applied after review:
  slide 16 has dividers and a note that each dot pools all knobs (one g per track only); slide 19
  labels each mark class under every group; slide 22 has plain row labels (which twin, seen or
  unseen pairs) and no "beats the design to its left" row.
- **Vault** (commits 75c67e2, 844f5d1, 887dd06): the capacity question (q4) and the covariates
  question (q1) carry a drafted, exploratory row-2 reading; h12, h17, h13, h14 and h15 link the
  row-2 evidence under `## Artifacts`; t118 and t120 have `## Progress` sections. No tick, verdict,
  close, accept or done mark.
- **New tasks** under t118 (PI yes, 2026-10-02): t122 hand-check the depth-law computation; t123
  diagnose design D's p-space explosion; t124 a principled fix for the p-space explosion. The id
  t121 was skipped because `visualization/t121-ceehrc-poster` already uses it.
- **Not filed** (PI: wait): a question under q4, "Does letting g read the bin's own value change how
  expressive f must be?" Wait until the p-space fix lands, then test on fresh data.
- **Cockpit:** `crux serve` on http://localhost:8787 now serves this worktree's vault
  (`--dir .../worktrees/counterfactual-f/cruxvault`), not the main checkout, so the branch's vault
  changes show. After the merge, restart it from the main checkout's `cruxvault/`.
- **Depth law, from the deck review:** the PI asked why f does not follow the depth law by
  construction. Answer: in t118 the count-space f has no log-depth offset (log μ = a + b·log(1 + X)
  in A; depth is only one z-scored input of g, `tools/t118/ladder/encoding.py`), by design, so the
  covariates must carry the depth effect. Candidate reasons it misses: few depth examples and
  arm → arm ratios never trained; a per-bin NB loss dominated by low bins; a pure scale is not a
  shift in log(1 + X); or a wrong computation (t122). Offered, not decided: a variant with a
  built-in log-depth offset, as a ceiling.

| submitted (UTC) | job | what | kit |
|---|---|---|---|
| 2026-09-29 01:04 | 22862749 | nocov law smoke, indices 66, 138, 498, 570 (`row2_law.sh`) | K6 |
| 2026-09-29 01:04 | 22862750 | twin law smoke, twin indices 18, 42, 162, 189 (`xshuf_law.sh`) | K6 |
| 2026-09-29 01:04 | 22862752 | full row-2 nocov train+score, the 192 tasks in `row2/idx_nocov.txt` (the 4 smoke runs skip), `%40` | K6 |
| 2026-09-29 03:24 | 22867594 | law, batch 3: 156 more row-2 real/ids runs | K5 |
| 2026-09-29 03:24 | 22867637 | twin law, batch 1: twin indices 0–12 | K6 |
| 2026-09-29 01:04 | 22862753 | full twin train+score, twin indices 0–191 (the 4 smoke runs skip), `%40`, `--mem=32000M`, out `row2_xshuf/` | K6 |

| submitted (UTC) | job | what | kit |
|---|---|---|---|
| 2026-09-28 23:24 | 22859772 | nocov smoke on K6, indices 66, 138, 498, 570 (`row2_train.sh`, out `row2/`) | K6 |
| 2026-09-29 00:30 | 22861310 | law, batch 2: 119 more real/ids runs with SCORE_DONE (`row2/law_batch2.txt`); batch 1 (22857243) all 29 COMPLETED, longest 1:08:43 | K5 |
| 2026-09-28 23:24 | 22859778 | shuffled-bin twin smoke, twin-table indices 18, 42, 162, 189 = A2 C19M16 counts, A2 across counts, D2 C19M16 counts, D2 across pval, seed 0 (`xshuf_train.sh`, out `row2_xshuf/`) | K6 |
PI ruling 2026-09-28 (shuffled-bin-value twin): g's x at bin i is the source track's x at a
uniformly random bin of the same chromosome; redrawn each training step; one fixed seeded
permutation per chromosome at prediction and scoring. f still reads the true x.
Further PI answers the same day: training draws come from a fixed uniform pool of 2^17 bins per
(source track, chromosome), with replacement, the bin itself not excluded; padded positions read
their own padded value; one permutation per chromosome shared by all source tracks; two drafted
checks, "row 2 beats its twin" on trained pairs and on the law-test pairs (neither gates anything);
no no-covariates version of the twin (192 runs, real covariates); per-run g summaries kept.
Build plan `.orchestrate/plan_xshuf.md` (git-excluded); chunks X1 (draw, training, prediction,
task table) and X4 (schema, figures, reports) building in
`/Users/mforooz/Desktop/research/libbrechteam@sfu/.orchestrate-wt/CANDII/xshuf-X{1,4}`.

The full 576 array waits for the no-covariates smoke: if the averaged map is too slow in p space,
the fix changes the code (new kit), and runs from two kits should not mix.

## Where everything is

| what | where |
|---|---|
| code | branch `exp/t118-counterfactual-f` (PR #48, draft): `tools/t118/ladder/`, `tools/t118/covariates.tsv`, `slurm/t118/ladder_*.sh`, `slurm/t118/explore_x*.sh`, `tests/test_t118_*.py` (238 tests) |
| code run on Nibi | kit `K3` = Python 34e5705 + SLURM 82cf8db (the 576 runs); kit `K4` = K3 + design X (5b2998d, CPU script) — under `/project/def-maxwl/mforooz/t118/ladder/code/` |
| runs | `nibi:/project/def-maxwl/mforooz/t118/ladder/runs/<rung>_<g>_<space>_<model>_s<seed>/` (ckpt, scores.json, law.json, figdata.npz) |
| aggregate | `nibi:/project/def-maxwl/mforooz/t118/ladder/agg/` (results.json, results_summary.tsv, checks_*.json, `<rung>/report.md` + figures) |
| design X runs | `nibi:/project/def-maxwl/mforooz/t118/ladder/explore_x/runs/X_*` |
| cache | `nibi:/project/def-maxwl/mforooz/t118/ladder/cache/` — 260 memory-mapped files, 109 GB. Kept; deleting it is the PI's call |
| evidence | main checkout `cruxvault/results/h12` (A), `h17` (B), `h13` (C), `h14` (D), `h15` (main claim), each with `FIR_PATH.txt`; also copied into the worktree's `cruxvault/results/` so `crux validate` resolves the links |
| team page | https://claude.ai/artifact/Gj3KNBCaK1efJpfTQ8KV7V (mirror: `cruxvault/results/t118/team_overview_2026-09-25.md`) |
| build plan and log | `.orchestrate/plan.md` in the worktree (git-excluded) |

## Results — the four designs (drafted readings met; 12 per cell unless noted)

Every gain is judged against 2 × the real g's seed wobble (3 seeds), per mark class (DNase; narrow;
broad), CRPS on all bins and on the top 1%, for both versions of g (one per track, one across tracks).

| check | A counts | B counts | C counts | D counts | A p | B p | C p | D p |
|---|---|---|---|---|---|---|---|---|
| beats the no-covariates twin (held-out chromosomes) | 3 | 11 | 11 | 11 | 12 | 7 | 11 | 3 |
| law test vs the no-covariates twin | 6 | 8 | 12 | 9 | 4 | 1 | 4 | 1 |
| law test vs the labels-as-ids twin | 10 | 12 | 12 | 11 | 10 | 11 | 11 | 5 |
| beats the design below | — | 9 | 5 | 1 | — | 8 | 2 | 0 |
| depth law (of 6; counts only) | 0 | 0 | 0 | 0 | — | — | — | — |

Checks met in total: A 45/78, B 67/102, C 68/102, D 41/102.

- **Design A** (one line in log space): p space is its strength (12/12 against the twin). In counts
  it fails, because one straight line on log(1 + counts) cannot represent "no change" at low counts
  and overshoots at peaks (on the p-only arms, a ≈ −1.6, b ≈ 1.9 instead of 0 and 1).
- **Design B** (monotone curve) fixes counts (3 → 11) but is worse than A in p space, which is where
  the exploding pairs hit.
- **Design C** (kernel, then curve) is the best in counts on the law test (12/12) and is the design
  the main claim judges for the across-track g.
- **Design D** (FiLM CNN) is strong in counts against the twin but collapses in p space (3/12),
  again because of exploding pairs.
- **Depth law**: unmet by every design in every class (predicted count scale misses the true depth
  ratio by 30–140% against a 10% bar). Not yet investigated: check the depth-law computation itself
  before reading this as a model failure.

## Results — the main claim (drafted; `cruxvault/results/h15/report.md`)

Chosen design by the pre-registered chr22 rule (lowest design within the best one's seed wobble):

| g | space | A | B | C | D | chosen |
|---|---|---|---|---|---|---|
| across tracks | counts | 0.491 | 0.395 | **0.353** | 0.360 | C |
| across tracks | −log10 p | 0.226 | 4.76 | **0.181** | 8 912 | C |
| per track | counts | 0.566 | 0.374 | 0.340 | **0.332** | D |
| per track | −log10 p | **0.224** | 0.229 | 0.338 | 386 | A |

Readings of the chosen design (met/total): shuffle 22/24 met (a wrong C′ removes the advantage);
swap 0/12 met (C′ = C does not return X to within a median |log ratio| of 0.1 — ≈0.19–0.22 for C in
counts; part of this is the log(1 + X) input scale); depth law 0/6 met. Under the rule `all`,
several checks are unmet for every chosen design.

## The p-space explosion (open; the PI wants a principled fix)

In 32 of 576 run × split combinations, one or two pairs (of 14–242) score CRPS from ~25 to ~10⁷,
almost all in −log10 p and in designs B, C and D. All are **upscaling** pairs (my label): a sparse
source (depth 3.75M, 7.5M, 15M, or DNase with dedup off) to the full-depth base. Their top-1% CRPS
is ordinary (~4); the explosion is in low bins, and one such pair dominates a run's mean and seed
wobble.

Diagnosis, checked with the PI: in the worst pair (DNase depth 3.75M → base, design B, across-track
g), the lowest knot sits at x = log(1e-3) (the floor) and its learned σ is 10.9. Only 0.1% of source
bins sit at the floor, and there the target's log-space SD is 1.8 — so 10.9 is not what the data ask
for. The lowest knot has almost no training data, its σ drifts, and linear interpolation spreads the
large σ over every bin between the floor and the median, where a log-normal's mean,
median × exp(σ²/2), makes CRPS explode. A is spared because it has one σ per pair.

Ruled out by the PI: capping σ (not principled); keeping the runs and only reporting (not a fix).
Withdrawn by me: a censored log-normal (it rested on a wrong diagnosis). Also discussed: a
log-Laplace would be worse (power-law tail; mean and CRPS infinite once its scale reaches 1). Still
unexplained: design D also explodes and has no knots.

## Design X — exploratory test of the PI's idea (not pre-registered)

The idea: g reads the bin value as well as (C, C′), so each bin value gets its own transformation.
As built: g outputs a 32-number pair vector from (C, C′); a shared MLP reads [bin value, pair vector]
and gives, per bin, loc = x + a and disp = d. So f is design A with the slope fixed at 1 and a
shift a and spread d that depend on the bin value — any function of x, not forced to be monotone.
It is a change to both g and f; it is fairly compared with B (both per-bin), not with C or D, which
see neighbouring bins. Tested: one g per track, H3K27ac and DNase, both spaces, real g and the
no-covariates twin, 3 seeds, held-out chromosomes only (no law test). Ran on CPU (16 cores) because
every 10 GB GPU-slice node was held by the scheduler for another job.

| track | space | A | B | C | D | X | exploding pairs A/B/C/D/X | X beats twin |
|---|---|---|---|---|---|---|---|---|
| H3K27ac | counts | 0.419 | 0.378 | 0.364 | 0.363 | 0.376 | 0/0/0/0/0 | met (0.085 > 0.002) |
| H3K27ac | p | 0.165 | 0.441 | 1.04 | 0.144 | 0.151 | 0/1/1/0/0 | met (0.036 > 0.004) |
| DNase | counts | 3.06 | 0.824 | 0.818 | 0.772 | 1.32 | 3/0/0/0/0 | unmet (0.30 < 0.42) |
| DNase | p | 0.752 | 0.541 | 0.526 | 18 500 | 0.659 | 0/0/0/3/0 | met (0.16 > 0.13) |

Mean CRPS over all bins, real g, mean over seeds; exploding = pairs with CRPS > 20 over 3 seeds. In p
space X had no exploding pair and beat its twin on both tracks. In counts it is close to B on
H3K27ac and clearly worse than B–D on DNase. Two tracks and sporadic explosions: suggestive, not
proof.

## Choices I made (science choices the plan did not settle; the most conservative option)

1. Fragment length is not a covariate; only the extsize factor k is encoded (fragment length is in
   the table, unencoded).
2. Products with no control (DNase, control-identity = none): control fraction, ratio k and control
   depth are 0, identity none, has-control 0.
3. Count-space pairs whose target counts equal the source are kept and flagged; headline includes
   them, a variant without them is reported beside.
4. The labels-as-ids twin's permutation is a derangement.
5. B's curve above the last knot continues with the last slope.
6. The across-track g gets the same step budget as a per-track g.
7. The depth law is computed in count space only.
8. The main claim's chr22 rung choice uses CRPS on all bins (real g, mean over seeds, macro over
   tracks).
9. The law test scores all chr19 + chr21 bins, in a CPU job per run, from the checkpoint.
10. Figures use matplotlib; tested locally with `/Users/mforooz/miniforge3/bin/python` (the
    `candii` env lacks it). No environment changed.
11. The 109 GB cache stays until the PI decides.
12. At test time the no-covariates twin predicts with one θ (its mean over the training pairs), so
    it is never asked to extrapolate onto covariate pairs it never saw.

## Disagreements with the plan

- The across-track g has 242 pairs, not the plan's "246" (both DNase MAPQ arms excluded).
- Swap vs input scale: with x = log(1 + counts), the identity predicts X + 1, so design A may miss
  the 0.1 swap bar by construction in counts. Bar unchanged; reported as measured.

## Failures and fixes (none changed a number)

- 8 train tasks failed in 35 s on an NFS "Stale file handle" (every task rewrote a shared
  `tasks.tsv`); fixed in 82cf8db (shell only) and retried.
- The law array chained with `aftercorr` never released a task while the train array ran; cancelled
  and resubmitted directly.
- 10 law/aggregation tasks failed on CVMFS "Input/output error" while building the venv (nodes
  c128, c166, c537); resubmitted with those nodes excluded.
- The Nibi tunnel dropped three times; Nibi work stopped each time until the PI ran `hpc up nibi`.
- Design X's GPU array (22683580) never started: all MIG nodes (g30–g37) were held (`mixed-`) for a
  higher-priority job. Cancelled; run on CPU (22688594), 4–51 min per run.

## Nibi jobs (all finished)

cache 22654779 · early check 22655875 · pilot 22656701 / 22656702 · train 22657297 (+ retry
22667089) · law 22657301 (cancelled part) → 22667088, 22667090, 22667091, 22669098, 22676228 ·
aggregation 22678427 (A), 22678919 (B), final 22682907 (C), 22682908 (D), 22682909 (A), 22682910
(B) · design X 22683580 (cancelled) → 22688594 (CPU).

## Measured cost

Per-track train + score ≈ 12 min on a 10 GB slice (cold cache; much faster warm); across-track law
test up to ~2 h on 8 CPU cores; aggregation 2–4 min; design X on 16 CPU cores 4–51 min per run.

## Waiting on the PI

- The principled fix for the p-space explosion (see above), and whether design X should become a
  pre-registered design.
- Ticks, verdicts (`crux close`), accepting the task, merging — none done.

## 2026-10-08: the next design (depth offset in f, and a counts → p map h)

- The deck gained a slide that compares one g per track with one g across tracks (commit 09d3dcb,
  deck version 5). The PI shared the deck with the lab.
- The PI and the agent agreed a new design fork by fork: `plan/ETA_OFFSET_AND_H_DESIGN.md`. It
  replaces the log-depth-offset variant offered on 2026-10-02.
- Vault: two new questions under q1 (q5, the depth offset; q6, the map h) and task t125 (a child
  of t118). The run is exploratory. Nothing is built or submitted yet.
- First step of t125: verify the t112 DNase p recipe, the 25 bp p binning and the ChIP caller
  commands on Nibi.

## t126 — encoder-decoder rung A (spec `plan/T126_ENCDEC_A.md`; exploratory, no verdicts)

Built (commit 774cd0f): `GEncDec` in `tools/t118/ladder/model.py` (shared phi(c) -> alpha, log beta;
psi(c') -> NB log size), forms `AED` (beta learned) and `AED1` (beta = 1) in `ladder/fforms/`, an
optional hold-out in `pairs.train_pairs` / `fit_pids` / `train.run_training` (default unchanged;
config.json gains `holdout`, `heldout_pairs` only when set), held-out pairs scored as kind `heldout`
(+ depthlaw) by `score.score_trained`. Task table `tools/t126/pairs.py`: arms A, AED, AED1 x 8 g's
(7 per track + all) x real/nocov x 3 seeds = 144, `index = arm_i*48 + g_i*6 + model_i*3 + seed`.
Held out of training: depth 7.5M on every track (fit pids exclude it too, so the encoder's z-scaling
is fitted without it). SLURM `slurm/t126/train.sh`, `law.sh`. Tests `tests/test_t126_encdec.py`.

- Kit K1: `/project/def-maxwl/mforooz/t126/code/K1`, `GIT_SHA` 774cd0f (rsync of the worktree without
  .git, cruxvault, .orchestrate; wheels from t125 K14). Out dir `/project/def-maxwl/mforooz/t126/`
  (runs/, logs/). Products `/project/def-maxwl/mforooz/t112_cf/products` (manifest md5 = fixture);
  cache `/project/def-maxwl/mforooz/t118/ladder/cache` (read only).
- 24644857 `t126_verifyK1`: tests t126, t118, t125 on K1 (script and log in `t126/verify/`).
- 24644858 `t126_train` smoke: array 18,66,114 (A, AED, AED1 x C19M16 real s0).
- 24644859 `t126_law` smoke: same indices, `aftercorr:24644858`.
- 24644857 result: t126 15 passed; t118 585 passed, 6 skipped, 1 failed =
  `test_t118_xshuf_score.py::test_twin_scoring_is_deterministic[D2]`, the known Nibi-CPU failure
  that t125 also saw on t118 kit K8 (`plan/T125_STATUS.md`); row-2 xshuf code is untouched by t126.
- Smoke (2026-10-20 06:55 UTC, g30): AED 10.4 min, AED1 9.4 min train + score; scores.json holds
  36 trained, 2 heldout and 6 depthlaw records (2 from heldout), no 7.5M pid in the training pairs
  or fit pids.
- 24646366 `t126_train`: array 0-17,19-65,67-113,115-143%12 (the 141 non-smoke runs), 07:06 UTC.
- 24646367 `t126_law`: same array, `aftercorr:24646366` (submitted while 24646366 was live).
- Never resubmit these; a rerun skips on SCORE_DONE / LAW_DONE.

### t126 rung D (spec section "Rung D" in `plan/T126_ENCDEC_A.md`)

Built (commit 01579a2): form `DED` (`ladder/fforms/form_ded.py`: encoder dil 1, 2 FiLM by g_enc(c);
decoder dil 4, 8 FiLM by g_dec(c') + D's 1x1 head; no skip; context 30), `model.GEncDecD`,
`tools/t126/train_ded.py` (four-term loss + lam * mse(Z, Z')), D table `pairs.py tasks --table D`
(D, DEDl0, DEDl0.1, DEDl1, DEDl10 x 8 g x real/nocov x 3 seeds = 240; arms 1-2 = 0-95). Init change
vs D: the zero head with no skip gives the constant map loc = 0, disp = log n0 at start (D: identity).
mse(Z, Z') is the mean over channels and the latent positions the decoder reads.

- Kit K2: `/project/def-maxwl/mforooz/t126/code/K2`, `GIT_SHA` 01579a2. Out dir
  `/project/def-maxwl/mforooz/t126/D/` (runs/, logs/). Rung-A runs stay on K1 (A code unchanged).
- 24646796 `t126_verifyK2`: tests t126 and t118 on K2.
- 24646801 `t126_trainD` smoke: array 18,66,114 (D, DEDl0, DEDl0.1 x C19M16 real s0).
- 24646802 `t126_lawD` smoke: same indices, `aftercorr:24646801`.
- 24646796 result: t126 25 passed on K2; t118 585 passed, 1 failed = the same known D2 twin-determinism flake.
- D smoke (g30): D 9.6 min train + score; DEDl0 and DEDl0.1 ran all 4000 steps in ~12 min of
  training (chr22 val NLL 0.740 / 0.734 vs D 0.731 at its best step). Configs: 36 training pairs,
  no 7.5M product, 2 held-out pairs; t118 settings (lr 1e-3, batch 32, 4000 steps).
- 24647762 `t126_trainD`: array 0-17,19-65,67-113,115-239%12 (all 237 non-smoke runs: arms D and
  DEDl0 = 0-95, lambda arms DEDl0.1 / DEDl1 / DEDl10 = 96-239), 07:34 UTC.
- 24647788 `t126_lawD`: same array, `aftercorr:24647762` (submitted while 24647762 was live).
- lambda selection (later, at aggregation): per g version, lowest mean-over-seeds best val_nll.
- Never resubmit these; a rerun skips on SCORE_DONE / LAW_DONE.
- 2026-10-20 23:58 UTC: 24647788 tasks 73, 74, 174 FAILED on c329 (CVMFS: invalid tzdata wheel at venv build); re-run as 24706070 (array 73,74,174, c329 also excluded). 24646366/24646367 (A) and 24647762 (D train) all COMPLETED.
- 2026-10-21: t126 readings (exploratory) on Nibi in `/project/def-maxwl/mforooz/t126/readings/`: extraction 24706111 (final, after 24706070), stdlib re-check 24706606; report `cruxvault/results/t126/report.pdf` (chosen lambda: 0.1 per-track g, 1 across-track g).

## t128 — symmetric encoder-decoder rung D v2

Spec `plan/T126_ENCDEC_D2.md`; plan `.orchestrate/plan_t128.md` on exp/t118-counterfactual-f.
Exploratory: readings, no bar, no verdicts.

- Kit K1: `/project/def-maxwl/mforooz/t128/code/K1`, `GIT_SHA` 58c9c22888284f58fdfb00034fe702384e7726be
  (exp/t118-counterfactual-f with chunks C1–C5 merged; clean and equal to origin at snapshot).
  rsync of the worktree without .git, cruxvault, .orchestrate, .claude, `__pycache__`; `wheels/`
  copied from t126 K2 (1 file, `x_transformers-2.11.23-py3-none-any.whl`). 13 MB.
- Out dir `/project/def-maxwl/mforooz/t128/` (code, runs, logs, verify, viabase, readings).
  Products `/project/def-maxwl/mforooz/t112_cf/products`; cache `/project/def-maxwl/mforooz/t118/ladder/cache`
  (read only); old D runs `/project/def-maxwl/mforooz/t126/D/runs` (read only).
- Submitted 2026-10-21 20:48:55 UTC, all with c128, c166, c329, c371, c537 excluded:
  - 24757111 `t128_verify`: tests t128, t126, t118 on K1; log `t128/verify/t128_verify_24757111.out`.
  - 24757112 `t128_train` smoke: array 18,57,90,129,153,177,201,225%12 (one real s0 C19M16 run per setting).
  - 24757113 `t128_law` smoke: same indices, `aftercorr:24757112` (submitted while 24757112 was live).
  - 24757114 `t128_viabase` smoke: array 9 (= D_C19M16_counts_real_s0 via the base).
- At submission every Nibi partition was `down` (sinfo; no reservation listed), so all four jobs
  were PENDING with Reason=PartitionDown (law: Reason=Dependency). They start when Nibi reopens.
- Never resubmit these; a rerun skips on SCORE_DONE / LAW_DONE / LATENT_DONE.

### t128 on Fir

Nibi GPU slices and most H100 nodes were draining for over a day, so the PI ruled to copy the data
to Fir (2026-10-24). The Nibi smoke 24797365/24797366 was left in place; the PI decides which runs count.

- Globus (PI, Nibi -> Fir, same real paths under `/project/6014832/mforooz`), checked 2026-10-24 00:17 UTC:
  `t112_cf/products` 635 files, 42402281147 B on both; `t118/ladder/cache` 520 files, 126091750990 B
  on both. md5 equal on both clusters for `MANIFEST.tsv` and cache `C40M18__mapq__0__pval.npy`,
  `C40M18__mapq__10__pval.npy`, `C07M20__abproxy__f0.5__counts.npy`. No data went through the laptop.
- Blacklist on Fir: `/project/def-maxwl/mforooz/DATA_CANDI_MERGED/hg38_blacklist_v2.bed`, sha256
  31c69342df43bbc19dd8ef2886611a8150cb53e90f341de14d30e742c9251737 (equal to Nibi's
  `EIC_REPRO/002/scripts/hg38_blacklist_v2.bed`; `EpiDenoise/data/hg38_blacklist_v2.bed` also equal).
  It is passed to train.sh as the 5th positional argument and to law.sh as the 4th.
- Kit K2 on Fir: `/project/def-maxwl/mforooz/t128/code/K2`, `GIT_SHA` f47cf117649fbe388cc6de8105b291627b01f2d9
  (= origin/exp/t118-counterfactual-f); `git archive` without `cruxvault/`; `wheels/` = the one wheel
  from Nibi K1 (`x_transformers-2.11.23-py3-none-any.whl`, sha256 2ba818a5...). 15 MB. The venv recipe
  (python/3.10.13, requirements-fir.txt, requirements-pypi.txt from wheels/) builds on the Fir login
  node and imports candi from K2/src (torch 2.6.0).
- sbatch overrides on Fir (repo scripts unchanged): `--exclude=` on every job (the scripts' Nibi node
  names c128,c166,c537,c371,c329 are invalid on Fir); `--time=1:00:00` on train (b1). The MIG gres
  `nvidia_h100_80gb_hbm3_1g.10gb` exists on Fir, so the train script's own gres is kept.
- Submitted 2026-10-24 00:17-00:18 UTC, out dir `/project/def-maxwl/mforooz/t128` on Fir:
  - 61556 `t128_verify` on K2 (RUNNING at submission); log `t128/verify/t128_verify_61556.out`.
  - 61593 `t128_train` smoke: array 18,57,90,129,153,177,201,225%12, MIG 1g.10gb, 1 h (PENDING Priority).
  - 61594 `t128_law` smoke: same indices, `aftercorr:61593` (submitted while 61593 was live).
  - No via-base job on Fir: it reads t126 checkpoints that exist only on Nibi.
- Never resubmit these; a rerun skips on SCORE_DONE / LAW_DONE / LATENT_DONE.

### t128 smoke readback and the full submission (C7, 2026-10-24)

Fir smoke 61593/61594 read back on 2026-10-24 01:56 UTC. Index 129 (round trip only, sampled latent)
and 225 (cross + round trip, sampled latent) have all four marks; 177 (λ = 1) has TRAIN_DONE only.
The other five have only `config.json` (timed out at the 1 h override); their rerun trains from scratch.

| check (PI rulings applied) | 129 | 225 | 177 |
|---|---|---|---|
| rung DS, holdout `[["depth","7.5M"]]`, 36 train pairs, 2 held-out pairs | pass | pass | pass |
| no 7.5M in train pairs or the 19 fit pids | pass | pass | pass |
| form A, `cclean_init` == 0 (max abs 0.0) | pass | pass | pass |
| `val_nll_rule` | L2 NLL | L1+L2 NLL | L1+L2 NLL |
| `train_log.tsv` header `step loss mse_z kl val_nll val_cross_nll seconds` | pass | pass | pass |
| kl | 0.35–1.52 (> 0) | 0.48–1.84 (> 0) | 0 (deterministic) |
| scores.json: trained 72, heldout 2, swap 19 (= fit pids), depthlaw 6 (2 heldout) | pass | pass | — |
| law.json: all 342 law records carry `crps_oracle_scaled_all` | pass | pass | — |
| latent.json kinds: latent 20, agree 190, ident 380 | pass | pass | — |

The 6 depthlaw copies in law.json carry no split; that is the C3 design ("depthlaw copies drop
them as today"), not a defect. No DSbox run finished, so the box-init check waits for the full run.

| run | steps | best step | val_nll | val_cross_nll | train min | score min | law min | latent min | median abs(Z − x), base | corr(Z, x), base | median over 20 pids: abs(Z − x), corr | median σ_z |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 129 | 1250 | 250 | 1.0900 | 1.0351 | 29.9 | 15.1 | 18.9 | 5.4 | 0.5443 | 0.9805 | 0.5369, 0.9807 | 0.376 |
| 225 | 1750 | 750 | 2.8552 | 0.7841 | 38.2 | 18.2 | 25.3 | 6.2 | 0.5723 | 0.9886 | 0.5116, 0.9818 | 0.749 |
| 177 | 3000 | 2000 | 2.5194 | 0.7445 | 56.4 | — | — | — | — | — | — | — |

val_nll is not comparable across rules (L2 only vs L1+L2). Z tracks x closely (corr ≈ 0.98) in both
sampled-latent arms: a first look only, one seed and one g.
Walltime: train ≤ 4000 steps × ~1.1–2 s + score ~18 min fits the script's 4 h; law + latent ~32 min fits 6 h.

Nibi via-base smoke 24757114_9 COMPLETED (12 min, c501): `law.json` (342 law records, all with the
split), `LAW_DONE`, `viabase.json` (intermediate nb_mean, K1 sha 58c9c22); the old t126 run dir
has no file newer than K1.

Submitted 2026-10-24 01:58 UTC (never resubmit; a rerun skips on SCORE_DONE / LATENT_DONE / LAW_DONE):
- Fir 70473 `t128_train`: array 0-239%24, K2, script's own 4 h, `--exclude=` (Fir override, as 61593),
  blacklist DATA_CANDI_MERGED as 5th positional. Smoke 129, 225 skip; 177 scores only.
- Fir 70474 `t128_law`: array 0-239%24, `aftercorr:70473` (submitted while 70473 was live), 6 h, `--exclude=`.
- Nibi 24896767 `t128_viabase`: array 0-8,10-23%12, K1, Nibi blacklist, `--exclude=c128,c166,c537,c371,c329`
  (same as the script header), otherwise the 24757114 submit line.
- If the first Fir wave shows an ImportError, lower the throttle: `scontrol update JobId=70473 ArrayTaskThrottle=12`.
