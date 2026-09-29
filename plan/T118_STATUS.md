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
