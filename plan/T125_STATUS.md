# t125 status: the depth-offset count model and the counts → p map h

Orchestrator log for t125. Read `plan/T125_HANDOFF.md` first, then
`plan/ETA_OFFSET_AND_H_DESIGN.md`. The orchestrate plan is `.orchestrate/plan_t125.md` in this
worktree (git-excluded).

## State

| phase | state |
|---|---|
| 0 verify on Nibi | done 2026-10-11; formula forks with the PI |
| 1 build the count model | built and reviewed (offset, form A, model, train, score, aggregate) |
| 2 build the map h | built and reviewed (data side, formula, push-forward, learned h, train, score, aggregate); end-to-end smoke building |
| 3 smoke on Nibi | not started |
| 4 count runs (432) | running (N3, 24145801) |
| 5 h on the true X′ | done 2026-10-16: 36 TRAIN_DONE, 38 ALONE_DONE (see "N4 results") |
| 6 h on f's NB, chain scores | not started |
| 7 aggregate | not started |
| 8 vault readings | not started |
| 9 lab deck | not started |

## Job ids

Every id goes here at submission.

- Kit K9: `/project/def-maxwl/mforooz/t125/code/K9`, `GIT_SHA` 643054089f34e4ef662cf52acc37c84bac7d0ca2
  (rsync of the worktree without .git, cruxvault, .orchestrate; wheels copied from t118 K6).
- 23797342 `t125_verifyK9`: acceptance criteria 1, 2, 4-8 on kit K9 (CPU, 8 cores, 32 GB, 3 h).
  Script and log: `/project/def-maxwl/mforooz/t125/verify/` (script not in the repo).
- 23797345 `t125_hctl`: control count cache, array 0-129%20 ->
  `/project/def-maxwl/mforooz/t125/ctl_cache`; logs `/project/def-maxwl/mforooz/t125/h/logs/`.
- Kit K10: `/project/def-maxwl/mforooz/t125/code/K10`, `GIT_SHA` 7e0f0f2fabfc28b00c40bdc44c2886483d2e8bbd
  (K9 + the test-tolerance commit; rsync as K9, wheels copied from K9). The run code is identical:
  `git diff --stat 643054089f 7e0f0f2 -- tools slurm src` is empty.
- 24065614 `t125_verifyK10`: criterion 1 without -x, criterion 2, and the twin-determinism test 3x on
  K10 and 3x on t118 ladder kit K8 (CPU, 8 cores, 32 GB, 3 h). No t118 kit on Nibi has `tests/`, so
  the K8 runs use a `$SLURM_TMPDIR` copy of K8 plus the test file from git at 33e09cb (K8's sha),
  kept in `/project/def-maxwl/mforooz/t125/verify/K8_tests_33e09cb`. Script and log:
  `/project/def-maxwl/mforooz/t125/verify/` (`verify_K10.sh`, `t125_verifyK10_24065614.out`).
- Nibi smoke (N2), kit K10, 2026-10-15. Logs `/project/def-maxwl/mforooz/t125/{eta,h}/logs/%x_%A_%a.out`.
  - 24065701 `t125_etatrain`: array 18,66,114,162,210,258,306,354,402,45,381%11 (the 9 rungs x
    C19M16 real s0; A all nocov s0 = 45; D2 all nocov s0 = **381**, not 357 as the plan says: 357 is
    D2_C19M16_counts_nocov_s0 by `index = rung_i*48 + g_i*6 + model_i*3 + seed`, g "all" = 7).
  - 24065702 `t125_halone`: array 0,1 (formula:bin, formula:frag).
  - 24065703 `t125_htrain` table true: array 0,21 (h_affine_r1_push_true_s0; mlp r5 noise s0 =
    **21**, not 8 as the plan says: true index = learned_i*3 + seed, mlp r5 noise is learned_i 7;
    index 8 is affine r5 push s2).
  - Not yet submitted: h_train fnb indices **42** (mlp r5 push) and **47** (formula_mlp r5 noise) for
    f A_C19M16_counts_real_s0 (fnb index = f_i*12 + learned_i; f_i 3). h_chain index 3 exits 3
    unless all 36 true h and all 12 fnb h of that f have TRAIN_DONE (the script checks all 50), so
    the smoke as planned cannot produce a CHAIN_DONE; the chain smoke needs a decision.
  - 24067469 `t125_etalaw`: array 18,66,162,306%4 (law batch 1: A, B, D, C2 x C19M16 real s0).
  - 24067474 `t125_htrain` table fnb: array 42,47%2 (h_mlp_r5_push and h_formula_mlp_r5_noise on
    f A_C19M16_counts_real_s0, fnb_seeds 1).
  - Chain smoke (foreman ruling: h_chain.sh needs all 50 h): one-off
    `/project/def-maxwl/mforooz/t125/smoke/smoke_chain.sh` (h_chain.sh's resources and venv) runs
    `score.py chain` on f A_C19M16_counts_real_s0 with 6 h (2 formulas, the 2 true h of 24065703,
    the 2 fnb h of 24067474), output only into `t125/smoke/chain/`, log `t125/smoke/logs/`.
  - 24067530 `t125_smokechain`: that script, `--dependency=afterok:24065703:24067474` (both live).
  - 24067868 `t125_etalaw`: array 114,210,354%3 (law batch 2: C, A2, D2 x C19M16 real s0).
  - 24068466 `t125_etalaw`: array 258,402%2 (law batch 3: B2, Aold x C19M16 real s0).
  - 24070630 `t125_etalaw`: array 45 (law batch 4: A all nocov s0).
  - 24073862 `t125_etalaw`: array 381 (law batch 5: D2 all nocov s0).
  - 24067474_42 and _47 both TIMEOUT at 3:00:20 (no TRAIN_DONE, no timing.json). 24067530 never
    started (its dependency failed); cancelled.
  - 24083622 `t125_smokechain4`: `smoke/smoke_chain4.sh`, the same one-off chain with 4 h (2 formulas,
    2 true h; no fnb h exists), output only into `t125/smoke/chain/`.
  - 24083625 `t125_htrain` table fnb: array 42,47%2 again, `--time=12:00:00` on the command line
    (the script's 3 h is too short), only to measure one fnb fit; logs `h/logs/`.
  - 24083626 `t125_smokechain`: `smoke/smoke_chain.sh` (6 h), `--dependency=afterok:24083625`.
- N3 (part 1 in full), kit K10, 2026-10-16. Logs `/project/def-maxwl/mforooz/t125/eta/logs/%x_%A_%a.out`.
  - 24145801 `t125_etatrain`: array 0-431%40, `--mem=24000M --time=3:00:00` on the command line,
    submitted 15:53 UTC (the 11 smoke runs skip on SCORE_DONE). At submission the slice nodes
    g30, g37 show `mixed-`, g31-33 and g36 `draining@`, g35 `allocated`, g34 `inval`. At 16:16 UTC
    only 5-6 tasks run at a time (the free slices on g30/g37). At 17:50 UTC (2 h) 16-19 run at a
    time, 65 tasks COMPLETED, 0 failed: the slices are not held, so no CPU fallback.
  - 19:55 UTC: only 9 GPU tasks run (pending reason `ReqNodeNotAvail`; slice nodes `mixed-` for
    4 h). Pending tasks **384-431 (rung Aold, all 48) cancelled** in 24145801 (none had started;
    402 is the smoke run, already done on GPU) and moved to the CPU script, because Aold training
    is cheap (smoke: 20 s on the slice) and its scoring runs on CPU anyway. 24145801 keeps 191-383.
  - 24163757 `t125_etatraincpu`: `eta_train_cpu.sh`, array 384-431%40 (script resources: 16 cores,
    64000M, 8 h). So Aold s0-s2 except 402 are trained with `--device cpu`; every other rung on the
    slice. CPU Aold: train about 290 s (slice: 20 s), whole task 7-25 min.
  - 20:39 UTC: 1 new GPU task started in 30 min (7 of mine on the 1g slices; `ReqNodeNotAvail`).
    Pending tasks **240-383 (rungs B2, C2, D2, whole) cancelled** in 24145801 (none had started)
    and moved to the CPU script. 24145801 keeps 195-239 (rest of A2; A2 192-194 already on GPU).
  - 24166139 `t125_etatraincpu`: `eta_train_cpu.sh`, array 240-383%40. Device by rung: A, B, C, D,
    A2 on the slice; B2, C2, D2, Aold on CPU, except the smoke runs 258, 306, 354, 381 and 402
    (slice).
  - Law batches (`eta_law.sh`, `--mem=16000M --time=0:30:00` on the command line, `%40`, submitted
    directly for indices with SCORE_DONE; never a dependency):
    - 24149205: 3,4,5
    - 24151319: 0,1,2,6-14
    - 24152310: 15-17,20-23
    - 24153254: 19,24,25,26,27,28,29,30,33,34
    - 24154583: 31,32,35,36,37,38,39,40,41,48,49,50,51,52,53,59,60
    - 24156039: 54,55,56,57,58,61,62,63,64,65,67,68,69,70,71,72,73,74,75,76,77,83
    - 24157585: 78,79,80,81,82,84,85,86,87,88,89,96,97,98,99,100,101,102,105,106,107,108,109,110,111
    - 24159233: 103,104,112,113,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133
    - 24160871: 42,43,44,47,134,135,136,137,140,141,142,144,145,147
    - 24162067: 46,93,143,146,148,149,150,151,152,154,155,156,157,158,159,160,161,164
    - 24163123: 90,91,92,94,95,139,153,163,165,166,167,168,169,170,171,172,173,174,175,176,177
    - 24164273: 138,178,179,180,181,182,183,184,185,387,389
    - 24165412: 192,193,384,385,386,388,390,391,392,393,394,395,396,397,398,399,400,401,403,404,405,406,407,408,409,410,411,413,414,417,418,419,425
    - 24166703: 186,187,188,194,412,415,416,420,421,422,423,424
    - 24167774: 189,190,191,195,196,197,198,199,200,201,202,240,243,244,251,252,253,254,255,256,257,261,263,269,275,427,428
    - 24168951: 203,241,245,247,248,250,259,260,262,264,265,266,267,268,270,271,272,274,276,277,278,279,280,288,289,290,291,292,293,294,295,297,299,300,301,302
- N4 (h on the true X′), kit K10, 2026-10-16. Logs `/project/def-maxwl/mforooz/t125/h/logs/%x_%A_%a.out`.
  - 24145831 `t125_htrain` table true: array 0-35%36, script resources (8 cores, 32000M, 3 h),
    submitted 15:54 UTC (indices 0 and 21 skip on TRAIN_DONE from the smoke).
  - `h_alone.sh` batches (alone index = true index + 2; blacklist
    `/project/def-maxwl/mforooz/EIC_REPRO/002/scripts/hg38_blacklist_v2.bed`, as the smoke):
    - 24145944 `t125_halone`: array 2,23%2 (the two smoke-trained h), 15:55 UTC.
    - 24149709 `t125_halone`: array 9,14,17,26,27,35%6, 16:25 UTC.
    - 24151292 `t125_halone`: array 3,5,8,10,13,16,18,19,24,28%10, 16:36 UTC.
    - 24151779 `t125_halone`: array 4,6,7,11,12,15,22,25,36%9, 16:46 UTC.
    - 24152189 `t125_halone`: array 20,21,29,30,31,32,33%7, 16:57 UTC.
    - 24152669 `t125_halone`: array 34, 17:07 UTC.
    - 24154245 `t125_halone`: array 37, 17:33 UTC (true fits: 36 of 36 TRAIN_DONE at 17:33).

### Nibi smoke (K10)

2026-10-15, kit K10. Elapsed, node and MaxRSS from `sacct` (MaxRSS in GiB); train seconds, steps and
best step from each run's `timing.json`; score seconds from `scores.json` / `law.json` `timing`.
Summary script (read-only): `/project/def-maxwl/mforooz/t125/smoke/smoke_summary.py`.

**Count runs (`eta_train.sh`, one 10 GB MIG slice, 4 cores, 16000M, 3 h) and their depth-only law
pass (`eta_law.sh`, 8 cores, 24000M, 1 h).** Every run: 0 records with CRPS > 20, 0 non-finite
(scores: 140 records per track run, 896 for g = all; law: 12 records per track run, 84 for g = all).

| run (index) | train job, node | Elapsed | MaxRSS | train s | best / steps | score s | law job | law Elapsed | law MaxRSS | law score s |
|---|---|---|---|---|---|---|---|---|---|---|
| A C19M16 real (18) | 24065701, g37 | 00:18:40 | 9.8 | 171 | 250 / 1250 | 878 | 24067469 | 00:01:39 | 6.9 | 48 |
| B C19M16 real (66) | 24065701, g37 | 00:20:22 | 8.8 | 322 | 2750 / 3750 | 830 | 24067469 | 00:01:00 | 6.2 | 7 |
| C C19M16 real (114) | 24065701, g30 | 00:19:52 | 8.5 | 612 | 3500 / 4000 | 496 | 24067868 | 00:00:59 | 6.4 | 4 |
| D C19M16 real (162) | 24065701, g30 | 00:17:16 | 5.7 | 600 | 2750 / 3750 | 353 | 24067469 | 00:01:20 | 5.7 | 26 |
| A2 C19M16 real (210) | 24065701, g30 | 00:20:05 | 9.3 | 597 | 2750 / 3750 | 524 | 24067868 | 00:01:18 | 6.6 | 24 |
| B2 C19M16 real (258) | 24065701, g30 | 00:24:52 | 5.8 | 529 | 2000 / 3000 | 878 | 24068466 | 00:01:04 | 6.1 | 8 |
| C2 C19M16 real (306) | 24065701, g30 | 00:16:20 | 6.2 | 603 | 2750 / 3750 | 293 | 24067469 | 00:00:58 | 7.0 | 5 |
| D2 C19M16 real (354) | 24065701, g30 | 00:21:47 | 5.9 | 622 | **4000 / 4000** | 600 | 24067868 | 00:01:22 | 8.6 | 25 |
| Aold C19M16 real (402) | 24065701, g37 | 00:15:39 | 6.1 | **20** | 2000 / 3000 | 877 | 24068466 | 00:01:27 | 6.3 | 21 |
| A all nocov (45) | 24065701, g37 | 01:13:35 | **15.6** | 389 | 750 / 1750 | 3954 | 24070630 | 00:03:27 | 9.1 | 145 |
| D2 all nocov (381) | 24065701, g30 | 01:24:44 | **15.6** | 1275 | 2500 / 3500 | 3742 | 24073862 | 00:01:48 | 10.3 | 46 |

Notes. Scoring (4 workers) takes more wall time than training. The two g = all runs peak at 15.6
GiB against `--mem=16000M` (15.6 GiB): at the limit. D2 C19M16 picks its last step (no early stop
inside the step budget). Aold trains 3000 steps in 20 s (the old-grid form; not checked further).

**The map h.**

| run | job, node | Elapsed | MaxRSS | train s | best / steps | validation passes | notes |
|---|---|---|---|---|---|---|---|
| h_alone formula:bin | 24065702_0, c331 | 00:01:30 | 9.0 | - | - | - | score 43.8 s, 128 products, 0 non-finite, 0 CRPS > 20 |
| h_alone formula:frag | 24065702_1, c331 | 00:01:31 | 9.5 | - | - | - | score 44.4 s, 128 products, 0 non-finite, 0 CRPS > 20 |
| true affine r1 push s0 | 24065703_0, c331 | 00:26:14 | 22.4 | 1522 | 1000 / 2000 | 9: first 142 s, median 31 s, sum 394 s | |
| true mlp r5 noise s0 | 24065703_21, c331 | 01:05:10 | 22.4 | 3855 | 2750 / 3750 | 16: first 371 s, median 61 s, sum 2034 s | |
| fnb mlp r5 push, f = A C19M16 real s0 | 24067474_42, c140 | **TIMEOUT 03:00:20** | 7.9 | - | - | - | TotalCPU 13:55 h (4.6 of 8 cores busy); no timing.json, so no f_pass_seconds, val_seconds or capped-row count |
| fnb formula_mlp r5 noise, same f | 24067474_47, c146 | **TIMEOUT 03:00:20** | 4.6 | - | - | - | TotalCPU 11:48 h (3.9 cores busy); same |
| chain smoke, 4 h | 24083622, g30 | still RUNNING at 2:11:47 (last read) | 7.7 (host) | - | - | - | no output yet; one python process at 94 % of one core (the 4 workers are not used on the GPU path); the 10 GB slice shows 10.8 GB in use (full); `score.py chain` writes every h's scores.json only after all pairs are done |

The fnb fits print nothing between "h: ..." and the end, and write `train_log.tsv` and
`timing.json` only at the end, so a timed-out fit leaves no reading. Rerun 24083625 (same indices,
`--time=12:00:00`) measures one full fit; the 6-h chain smoke 24083626 waits on it.

**Recommended sizes for the full runs** (assumes the smoke's tasks are typical; slice-hours = one
10 GB MIG slice for one hour):

- **N3, 432 count runs.** 378 track runs at about 20 min (15.7-24.9 min) and 54 g = all runs at
  about 80 min (73.6-84.7 min): 378 × 0.33 h + 54 × 1.33 h ≈ 125 + 72 ≈ **200 slice-hours**. At
  `%40` the wall time is about 200 / 40 = 5 h plus a 1.4 h tail, if 40 slices are free. `--time`
  3:00:00 stays (2× the longest). **`--mem=24000M` on the command line** (the g = all runs sit at
  the 16000M limit). Law pass: 378 × about 1.3 min + 54 × about 3 min ≈ 11 task-hours × 8 cores ≈
  **90 core-hours**; `--time=0:30:00` and `--mem=16000M` are enough (longest 3:27, largest 10.3
  GiB); `%40` per batch.
- **N4, 36 true-X′ h fits.** Observed 26 and 65 min, 22.4 GiB each. Upper estimate 36 × 1.1 h × 8
  cores ≈ **320 core-hours**; at `%36` the wall time is about 1.1-1.5 h. Keep `--time=3:00:00` and
  `--mem=32000M` (70 % used). **38 alone scorings:** the formulas take 1.5 min and 9.5 GiB; a learned
  h alone was not measured. Estimate under 15 min each, 38 × 0.25 h × 8 ≈ **76 core-hours** at
  most; `%38`, keep `--time=2:00:00`, `--mem=32000M`.
- **N5, 768 fnb fits.** Not sizable yet: both smoke fits ran over 3 h on 8 cores. Lower bound
  768 × 3 h × 8 cores ≈ **18,400 core-hours**; at `%40` the wall time is at least 768 / 40 × 3 h ≈
  58 h. The script's `--time=3:00:00` is too short. **64 chains:** not sizable yet: the 4-h chain on one f ran over 2.2 h without finishing, on one core with a full 10 GB slice; a 50-h chain will be longer, and `h_chain.sh`'s `--time=6:00:00` may be too short.

**Flags (not decided here).**
(a) One fnb fit under 20 min on 8 cores: **no** — both ran over 3 h (TIMEOUT).
(b) Any fnb fit over 2 h on 8 cores: **yes**, both (the plan's trigger for the GPU-slice line in
`h_train.sh`; a repo change goes through a chunk).
(c) Explosions or non-finite: none in the 11 count runs, the 11 law passes or the 2 formula
alone scorings. The chain smoke has no result yet.

**Verify job 24065614 (K10), CRIT lines** (`t125/verify/t125_verifyK10_24065614.out`; 00:51:05,
6.8 GiB, c316):

    CRIT 2 PASS rc=0 passed=615 failed=0 skipped=0 deselected=0 error=0 (518 s)
    CRIT 3c K10 sha=7e0f0f2fabfc28b00c40bdc44c2886483d2e8bbd runs_pass=0 runs_fail=3
    CRIT 3c K8 sha=33e09cb (tmp copy, tests from git 33e09cb) runs_pass=0 runs_fail=3
    CRIT 1 FAIL rc=1 passed=610 failed=1 skipped=6 deselected=2611 error=0 (2145 s)
    CRIT 1 ID FAILED tests/test_t118_xshuf_score.py::test_twin_scoring_is_deterministic[D2]

Each 3c run is passed=1 failed=1: the A2 case passes and the D2 case fails, on K10 and equally on
the t118 kit K8. The D2 twin scoring is not deterministic on Nibi CPU in the t118 code too, so it
is not a t125 regression. The `records` lists differ (`c_star_all` agrees on the first record);
which field differs needs `pytest -vv`.

Formula sanity, MSE on log1p, chr19 + chr21 pooled (the alone scorer pools both; no chr21-only
number exists), blacklist removed. bin-only / frag-window:
C19M16 base 0.0305 / 0.0184; crop 36 0.1119 / 0.0368; crop 50 0.0693 / 0.0321; ratio k0.5
0.0347 / 0.0201; ratio k2 0.0118 / 0.0070. All 128 products: median 0.0398 / 0.0213, max
0.1477 / 0.0435 (both C40M18 crop 36). The bin-only formula is off on the crop arms (median crop
0.106 vs 0.034-0.062 elsewhere; C19M16 crop 36 c* = -1.47, i.e. the formula is high by about
2^1.47); the frag-window formula is not (crop 36 c* = -0.09). Ratio arms look right in both.

### N4 results (h on the true X′, K10)

2026-10-16 17:40 UTC. `ls h/runs_true/*/TRAIN_DONE | wc -l` = **36**; `ls h/alone/*/ALONE_DONE | wc -l` =
**38**. No task failed, none resubmitted. Elapsed and MaxRSS (GiB) from `sacct`; train s, best /
steps from `timing.json`. Indices 0 and 21 are the smoke fits of 24065703.

| h (3 seeds) | Elapsed min | MaxRSS | train s | best / steps (s0, s1, s2) |
|---|---|---|---|---|
| affine r1 push | 26.2-42.5 | 13.3-22.4 | 1522-2477 | 1000/2000, 1000/2000, 2250/3250 |
| affine r1 noise | 33.9-47.2 | 17.7-22.6 | 1934-2658 | 1250/2250, 2750/3750, 2250/3250 |
| affine r5 push | 27.4-35.3 | 21.7-26.5 | 1568-1994 | 1500/2500, 1000/2000, 2000/3000 |
| affine r5 noise | 41.5-46.8 | 13.2-31.2 | 2390-2755 | 3500/4000, 3750/4000, 2500/3500 |
| mlp r1 push | 28.8-45.2 | 25.3-31.2 | 1648-2602 | 1000/2000, 2750/3750, 1750/2750 |
| mlp r1 noise | 22.9-37.4 | 5.1-22.4 | 1297-2162 | 250/1250, 1250/2250, 1750/2750 |
| mlp r5 push | 51.9-54.0 | 15.8-31.2 | 2955-3131 | 3250/4000, 3500/4000, 2750/3750 |
| mlp r5 noise | 36.9-65.2 | 5.0-22.4 | 2103-3855 | 2750/3750, 1250/2250, 2500/3500 |
| formula_affine r5 push | 20.7-41.1 | 17.6-28.5 | 1180-2292 | 1500/2500, 500/1500, 1500/2500 |
| formula_affine r5 noise | 58.0-58.6 | 19.1-31.2 | 3326-3383 | **4000/4000, 4000/4000**, 3750/4000 |
| formula_mlp r5 push | 58.1-63.9 | 30.2-31.2 | 3390-3748 | **4000/4000 all three** |
| formula_mlp r5 noise | 25.1-96.7 | 5.2-31.2 | 1420-5735 | 250/1250, 2000/3000, 3500/4000 |

All 36: 20.7-96.7 min, sum 26.5 task-hours (about 212 core-hours). **MaxRSS reaches 31.2 GiB in 11 fits (indices 10, 11, 13, 18, 19, 27, 28, 31, 32, 34, 35)
against `--mem=32000M` (31.25 GiB): at the limit.** No log shows an OOM line and every task is
COMPLETED 0:0; the cgroup may count page cache. Five fits stop at the step budget (no early stop).

**Alone scorings** (128 records each; 1.5-4.0 min, 5.0-17.6 GiB; score 44-193 s). Explosion =
records with `crps_mean_point_all` > 20 (own-noise h: the point CRPS of `mean_P = exp(μ + σ²/2) − 1`);
non-finite = records with a non-finite `crps_mean_point` (bins in brackets). `crps_log` is finite in
every record of every h.

| kind | explode (s0/s1/s2) | non-finite records (s0/s1/s2) |
|---|---|---|
| affine r5 noise | 77 / 83 / 76 | 23 / 25 / 23 (623,131 bins) |
| formula_affine r5 noise | 41 / 49 / 47 | 8 / 8 / 9 (205,146 bins) |
| affine r1 noise | 2 / 2 / 2 | 0 |
| mlp r1 noise | 0 / 0 / 1 | 0 |
| the 2 formulas, all push h, mlp r5 noise, formula_mlp r5 noise | 0 | 0 |

`crps_top1` > 20 occurs in every kind, formulas included (3 and 4 records); that is the top 1 % by
true P, not an explosion. The affine own-noise h on rung 5 blow up in the native mean (`σ²` large);
their log-scale CRPS stays finite. **Capped rows: none.** The alone scorer writes no `n_capped`
(only chain records carry it, `score.py` 595 / 701), and `timing.json` of a true-X′ fit has no
capped field.

## Nibi access

- 2026-10-11: `ssh -o BatchMode=yes nibi` exited 255 (`Permission denied
  (publickey,keyboard-interactive,hostbased)`). All Nibi work stops until the PI runs
  `hpc up nibi`.
- 2026-10-11 01:26 EDT: `ssh -o BatchMode=yes nibi` returns `l2.nibi.sharcnet`. Nibi work resumes.

- 2026-10-15 19:40 PDT (N2): `ssh -o BatchMode=yes nibi` exited 255 (`Permission denied
  (publickey,keyboard-interactive,hostbased)`, mux master broken pipe). All Nibi work stopped until
  the PI runs `hpc up nibi`. Last state read (about 21:24 UTC): chain smoke 24083622 RUNNING 2:11:47;
  fnb rerun 24083625_42/_47 RUNNING 2:12:59 (no TRAIN_DONE yet); 6-h chain 24083626 PENDING on
  24083625. They run on without the laptop. Still to read: 24083622 and 24083625 sacct and
  timing.json (f_pass_seconds, val_seconds, n_capped), the smoke/chain scores.json
  (CRPS > 20, non-finite, n_capped_total, timing), then 24083626. Re-run
  `python3 /project/def-maxwl/mforooz/t125/smoke/smoke_summary.py` on the login node.

## Phase 0 findings

Read-only check on Nibi, 2026-10-11. Source: `/project/def-maxwl/mforooz/t112_cf/products/<pid>/provenance.json`
(exact commands), `covariates.json`, `MANIFEST.tsv`, the pipeline SIF scripts, and MACS2 2.2.4 source.

- **DNase P:** ENCODE atac-seq-pipeline v2.2.3 in DNase mode (`encode_task_macs2_signal_track_atac.py`),
  **not** dnase-seq-pipeline v3.0.0. `macs2 callpeak -t TA -f BED -g hs -p 0.01 --shift -75 --extsize 150
  --nomodel -B --SPMR --keep-dup all`, then `bdgcmp -m ppois -S lines(TA)/1e6`. No control: λ = max(λ_BG, λ_10k
  of the treatment's own reads), λ_BG = 150·N/2.7e9. No Tn5 shift. extsize arms change smooth-win to 75 or 300.
- **ChIP P:** chip-seq-pipeline v2.2.2. `callpeak -t TA -c CTL -f BED -g hs -p 0.01 --nomodel --shift 0
  --extsize FRAGLEN --keep-dup all -B --SPMR`, then `bdgcmp -m ppois -S lines(TA)/1e6`. λ_local = max(λ_BG,
  λ_d, λ_1k, λ_10k); control scaled to the smaller set (`--scale-to small`). FRAGLEN from cross-correlation.
  ctlid none: λ = max(λ_BG, λ_10k of treatment). ratio arms: `--ratio V`, V = k·30M/N_ctl. Two ctldepth q0.25
  arms (C19M16, C19M22) have treatment > control, so MACS2 scales the treatment down there.
- **ppois (tested with the SIF's MACS2):** P = −log10 P(X > floor(obs)) at λ; strict tail; no pseudocount;
  5-decimal rounding; float32 bigwig.
- **25 bp binning:** P = base-pair-weighted **mean** of per-base −log10 p over the bin (0 where no interval).
  Counts = **read-span overlap**: +1 to every bin the aligned read touches (no shift, no extension, no strand),
  chr1–22 + chrX, from the same nodup tagAlign (30M ChIP, 50M DNase). pe arms: each mate is a line.
- **Inputs for h:** `covariates.json` has depth, fraglen, control.reads, knob_value (ratio V or control depth),
  read_length, run_type. MACS2's N is all tagAlign lines (incl. chrM, contigs), not the sum of counts25.

## PI decisions

- 2026-10-11: learned MLP h = 2 × 64, with the same training settings as the t118 g (optimizer, learning rate, epochs, early stopping on chr22).
- 2026-10-11, formula h (from the phase 0 forks):
  - DNase follows the recipe that ran (ATAC pipeline v2.2.3 in DNase mode). Its conflict with the
    "dnase-seq-pipeline v3.0.0" rule is noted, not fixed.
  - Pileup ≈ count × fraglen / (read_length + 24). Bin-only uses the bin; the fragment-window variant
    uses the mean count over the bins one fragment wide.
  - λ = the full MACS2 max of windows (fragment, 1 kb, 10 kb, background) from the binned control, scaled
    as MACS2 does (to the smaller set; ratio V in ratio arms). No control: max(background, 10 kb of the
    treatment); in the chain that window uses f's predicted mean.
  - One Poisson tail per bin, with the caller's rules: strict tail, floor of the observed value,
    no pseudocount, N = all tagAlign lines.

- 2026-10-11, plan forks:
  - Law pass on depth-only pairs only (42 across tracks); `--law-scope all` stays available.
  - Repair training: 1 h seed, 768 fits (64 f × 12 h versions); seed spread quoted from the true-X′ h.
    Upgrade to 3 seeds only if the Nibi smoke shows one fit under 20 min on 8 cores (ask the PI then).
  - Own-noise loss on f's NB: expected Gaussian NLL, Σ_k NB(k)·NLL_k.
  - Own-noise scoring: exact mixture CRPS on log(1 + P); push-only also gets log-scale CRPS; push-only keeps
    native-scale CRPS; native MSE of the mean for both.
  - Control read length = the treatment base's read length; the Nibi smoke reports the formula-vs-real-P error.
    **Superseded the same day:** the crop arms cropped the control too, so every control's read length is
    measured from the tagAlign that ran (read-only Nibi step).

- 2026-10-11: formula + affine h = formula + an affine map of all rung-5 inputs (formula included), zero at init.

- 2026-10-11: the no-covariates twin keeps the depth offset. Real vs twin shows what the covariates add beyond the fixed d′/d scaling; it is not directly comparable with the old grid's twin gap.

- 2026-10-11: rung 5 of the learned h gains the treatment and control log read lengths (21 inputs).

- 2026-10-11: the shuffle control shuffles depth too, as in t118, so the offset uses the shuffled target's d′.

- 2026-10-11: a comparison of two arms uses only the pairs finite in both; it reports how many pairs it dropped and the non-finite counts.

- 2026-10-11: the log-scale CRPS gets its own oracle-scaled / scale-error split: fit c on the log scale (shift log(1 + P) by c·ln2, i.e. scale 1 + P by 2^c), closed form for the pushed set and the mixture. The native split scales the whole pushed set by 2^c, as t118's p-space split.

- 2026-10-16: fnb training and chain scoring get an exact speed-up chunk (no number may change; tests prove it), then a re-smoke, before the 768 fnb fits and 64 chains.
- 2026-10-16: the D2 twin run-to-run difference (also in t118's own kit) is measured on the smoke run; if it is far below the seed wobble, it is accepted and noted in the grid and the deck.

- 2026-10-16: in the chain, the own-noise h are scored with a merged mixture: adjacent k components merged into ≤ 256 moment-matched Gaussians before the CRPS pair sum, on all bins. It is kept only if its error against the exact CRPS, measured on a random sample of bins, is below 1e-3 relative; the measured error goes in the grid. Push-only h stay exact.
- 2026-10-16: compute scope for the repair reading and the chain: only the 16 real f with one g across tracks (192 fnb fits + 16 chains). This supersedes "768 fits". Per-track f get chain scores with the formulas and the true-X′ h only, if time allows.

## Log

- 2026-10-11: run started; Nibi down at start.
- 2026-10-11 01:30: Nibi up. Phase 0 reader and the planner dispatched. Chunk ids carry the prefix `t125-`.
- 2026-10-11: wave 0 merged (offset and task table, form A by level, eta model, h data side, formula h, exact push-forward, learned h). Control read lengths measured for 114 products (`tools/t125/hmap/control_read_length.tsv`). Wave 2 part 1 (eta training, eta scoring, SLURM part 1) building.
- 2026-10-11 (N1): kit K9 shipped to Nibi. Criterion 3: the 89 t118 files (tools/t118, slurm/t118,
  tests/test_t118_*, tests/fixtures) in K9 have the same git blob hashes as a0e6834, and
  `git diff --stat a0e6834` on those paths is empty. Verify job 23797342 and control-cache array
  23797345 submitted.
- 2026-10-11 23:17 EDT (N1): `ssh -o BatchMode=yes nibi` exited 255 during the queue poll. All Nibi
  work stopped until the PI runs `hpc up nibi`. State at 23:11 EDT: verify job 23797342 RUNNING
  (venv built, pytest started); control-cache array 23797345 tasks 0-18 COMPLETED (about 1 min
  each, `counts=built`), tasks 19-129 PENDING; 36 files in ctl_cache. Both jobs run on without
  the laptop. Still to read: the verify log's CRIT lines, sacct, and the criterion 9 count (114).
- 2026-10-15: kit K10 shipped to Nibi; verify job 24065614 submitted (criterion 1 without -x,
  criterion 2, twin-determinism test on K10 and on t118 K8).
