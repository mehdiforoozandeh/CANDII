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
| 4 count runs (432) | done 2026-10-17: 432 SCORE_DONE, 432 LAW_DONE; aggregate 24224060 (eta/agg/grid.md) |
| 5 h on the true X′ | done 2026-10-16: 36 TRAIN_DONE, 38 ALONE_DONE (see "N4 results") |
| 6 h on f's NB, chain scores | not started |
| 7 aggregate | part 1 done (eta agg job 24224060); part 2 on 5 of 8 chains done (job 24312235, PI 2026-10-17); refresh when A, B, B2 land |
| 8 vault readings | drafted: depth-offset question (part 1), map-h question (part 2, 5 of 8); refresh pending |
| 9 lab deck | published v1 (5 of 8 chains): https://claude.ai/artifact/AfLpe1acvSC9i43jg28AqS ; republish on refresh |

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
  - 00:36 UTC (2026-10-17): no GPU task started since 22:33 UTC (2 h; `ReqNodeNotAvail`). Pending
    **211-239 (rest of A2) cancelled** in 24145801 and moved to CPU. 24145801 has nothing left.
    A2 is therefore mixed: 192-209 (and smoke 210) on the slice, 211-239 on CPU.
  - 24183485 `t125_etatraincpu`: `eta_train_cpu.sh`, array 211-239%40.
  - 23:31 UTC: 24163757_429, _430, _431 (Aold_all nocov s0-s2), all three on node **c95**, hung in
    `score.py trained` for 3 h 07 min: trained at 20:24, then 15 score processes at 0 % CPU (state
    S), no output. The other Aold_all runs scored in 41 min (426, c45). Read as a node fault
    (transient); cancelled and resubmitted once on CPU with c95 also excluded (TRAIN_DONE exists, so
    only scoring reruns):
  - 24176598 `t125_etatraincpu`: `eta_train_cpu.sh`, array 429,430,431,
    `--exclude=c128,c166,c537,c371,c95`.
  - 23:49 UTC: **the hang repeats on three other nodes** (c477, c58, c59): after "TRAIN_DONE
    present", `score.py trained` parent (19 threads) and its 4 forked workers sit in
    `futex_wait_queue` at 0.0-0.2 % CPU for 15 min. So it is **not transient: a code defect,
    stopped** (24176598 cancelled; 429-431 have TRAIN_DONE, no SCORE_DONE). Likely cause, not
    proven: `score.py` 172/196 forks a `Pool` (`mp.get_context("fork")`) from a parent that holds a
    torch/OpenMP thread pool; `eta_train_cpu.sh` 120 sets `OMP_NUM_THREADS=16`. Only Aold_all nocov
    hangs: Aold_all real (426-428) and D2_all nocov on CPU scored. A fix (e.g. a spawn context or
    1 OMP thread in scoring, or scoring 429-431 with eta_train.sh) is a PI / chunk decision.
  - Resume 2026-10-17 02:48 UTC: SCORE_DONE 429, LAW_DONE 425; nothing of mine queued; no hung
    task left. Missing: score + law of 429-431; law of 213 (24190487_213 FAILED in 22 s on c329:
    `ERROR: Wheel 'tzdata' ... is invalid` from CVMFS, transient, resubmitted once) and 234-236.
  - 24200215 `t125_aoldw1` (coordinator ruling 1): one-off
    `/project/def-maxwl/mforooz/t125/eta/oneoff/aold_nocov_w1.sh` (not in the repo), array 429-431,
    2 cores, 32000M, 16 h, c95 also excluded. Per run: `score.py trained ... --workers 1` then
    `score.py law ... --workers 1 --law-scope depth` (the calls of eta_train.sh and eta_law.sh with
    workers 1: no forked pool), `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. eta_train_cpu.sh has no
    workers override (`--workers 4` is fixed at line 154). The law pass is in the same job because
    eta_law.sh forks the same pool (`--workers 8`).
  - 24200232 `t125_etalaw`: law for 213 (retry), 234, 235, 236. All COMPLETED.
  - Per-run device table (coordinator ruling 3):
    `/project/def-maxwl/mforooz/t125/eta/agg/run_device.tsv` (index, run_name, train_device,
    train_job, score_job), from the train logs (the task that printed train.py's `best_step` line)
    by `/project/def-maxwl/mforooz/t125/eta/oneoff/run_device.sh`. 216 gpu, 216 cpu: A, B, C, D 48
    gpu each; A2 19 gpu / 29 cpu; B2, C2 47 cpu + 1 gpu (smoke); D2 46 cpu + 2 gpu (smoke 354, 381);
    Aold 47 cpu + 1 gpu (smoke 402). score_job of 429-431 is `none` until 24200215 ends.
    Re-run 2026-10-17 after 24200215: every row has train_job and score_job (429-431 score_job
    24200215_*).
  - 24200215 COMPLETED: 429-431 scored serially in 2976-3009 s, law 72-75 s (52-53 min per task).
    With the forked pool removed, the scoring did not hang.
  - **Acceptance criterion 10 met (2026-10-17): 432 SCORE_DONE, 432 LAW_DONE.**
  - 24224060 `t125_etaagg`: `eta_agg.sh` K10, OLD=`/project/def-maxwl/mforooz/t125/old_grid/results_summary.tsv`
    (checks_*.json in the same dir; old-grid JSON t118 `row2/agg/results.json`), COMPLETED 1:17.
    "432 runs present, 0 missing; 750 readings". Output `/project/def-maxwl/mforooz/t125/eta/agg/`:
    grid.md, results.json, results_summary.tsv, pairs_summary.tsv, depth_law.tsv, summary_data.json.
  - Aggregate counts (`summary_data.json`), records with CRPS > 20 / non-finite / all, by kind:
    trained 12 / 0 / 52,272 (score 6, val 6); shuffle 6 / 0 / 26,136; swap 6 / 0 / 13,824; law
    0 / 0 / 4,536. **Every CRPS > 20 record is rung A2, g = all (across), nocov**: per kind, 4 DNase and
    2 narrow. Old grid (t118 row 2): trained 3 / 0 / 11,616, all rung A per-track DNase real.
    readings_skipped: 0 (750 readings).
  - 24173033 `t125_d2twin` (D2 twin nondeterminism, PI decision 2026-10-16): one-off
    `/project/def-maxwl/mforooz/t125/smoke/d2twin_rescore/d2twin_rescore.sh` (not in the repo; 4
    cores, 24000M, 3 h, CPU) copies run D2_C19M16_counts_nocov_s0 (index 357, trained on CPU by
    24166139) into that dir, drops the copy's scores, re-runs eta_train.sh's exact `score.py trained
    ... --workers 4` on the copy, then `d2twin_compare.py` compares per record with the original
    and with the 3 seeds; log `t125/smoke/logs/`. The original run dir is only read.
    FAILED at the compare step (scoring done); result in "D2 twin nondeterminism, measured".
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
    - 24170142: 249,273,281,296,298,304,305,307,308,309,310,311,313,314,315,316,317,318,319,322,324,325,327,328,329,426
    - 24171577: 242,246,303,312,321,326,335,339,348,349,350,351
    - 24173031: 285,336,337,341,342,343,344,352,355,356,357,360
    - 24174020: 204,205,206,207,208,209,282,284,330,331,338,340,346,347,358,359,361,362,364,365,366,367,369
    - 24175471: 283,286,320,323,332,333,334,353,363,368,370,371,373,375,376,377
    - 24177947: 287
    - 24179904: 372,383
    - 24181558: 374
    - 24187994: 211,212,214,216,220,221,224,225,227,228,229,230,232,345,378,379,380,382
    - 24190487: 213,217,218,219,223,226,231,233,238
    - 24191400: 215,222,237,239
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
- Kit K11: `/project/def-maxwl/mforooz/t125/code/K11`, `GIT_SHA` 7ebb9c59c83e4fc18387975733f8ff59740fca42
  (merge of chunk/t125-P2h; rsync as K9, wheels copied from K10), 2026-10-17.
- N5 (h on f's NB and the chains, track C19M16), kit K11, 2026-10-17. Logs
  `/project/def-maxwl/mforooz/t125/h/logs/%x_%A_%a.out`.
  - 24224310 `t125_htrain` table fnb (1 h seed): array
    36-47,132-143,228-239,324-335,420-431,516-527,612-623,708-719%16 (96 fits, the 8 C19M16 real
    s0 f), script resources (1g.10gb slice, 8 cores, 40000M, 1 day), submitted 04:56 UTC. The two
    smoke dirs in runs_fnb (A_C19M16 formula_mlp r5 noise, mlp r5 push) held only config.json.
  - `h_chain.sh` (h_set all, --mix-merge 512 default, script resources: slice, 4 cores, 40000M,
    1 day; blacklist `/project/def-maxwl/mforooz/EIC_REPRO/002/scripts/hg38_blacklist_v2.bed`),
    each submitted when its f's 12 fnb h had TRAIN_DONE (36 true h TRAIN_DONE):
    - 24268972 `t125_hchain`: array 19 (C_C19M16_counts_real_s0), 07:19 UTC.
    - 24272487 `t125_hchain`: array 3,11 (A, B _C19M16_counts_real_s0; task 1 of ~3 and ~2),
      07:39 UTC. At 07:38 only 2 fnb tasks run; the slice nodes g30-g32, g35-g37 show `mix-`
      (planned for higher-priority jobs), g33 `drng@`, g34 `inval`.
    - 24273272 `t125_hchain`: array 27 (D_C19M16_counts_real_s0), 07:49 UTC.
    - 24275564 `t125_hchain`: array 35 (A2_C19M16_counts_real_s0), 08:19 UTC. At 08:19 64 of 96
      fnb TRAIN_DONE; 3 fnb tasks run, every chain PENDING (Priority); GPU LevelFS 0.18.
  - 09:58 UTC: **slice stall, 2 h 10 min.** No N5 task has started since 07:48 UTC. 67 of 96 fnb
    TRAIN_DONE (A, B, C, D, A2 12 each; B2 7), 0 failed; 29 fnb pending (B2 523-527, C2 612-623,
    D2 708-719) and all 5 chain jobs (6 f) PENDING, reason Priority. Slice nodes g30, g32, g35,
    g37 `mix-`, g31, g36 `mix`, g33 `drng@`, g34 `inval`; cluster-wide 20 1g.10gb jobs run, 34
    wait. Nothing moved to CPU (needs the foreman/PI). Every job stays queued; no CHAIN_DONE yet.
    The 67 fits so far: elapsed 7 min to 2 h 28 min, MaxRSS max 18.8 GiB (of 40000M); 14 of 67
    hit the 4000-step cap (best_step 250-4000); 24 runs have capped NB rows (488 in all).
  - 10:20-10:40 UTC: slices free again; fnb tasks and all 5 chain jobs start (10:36). Nothing moved
    to CPU (foreman: keep waiting for slices).
  - 11:21 UTC: A2 CHAIN_DONE (task 24275564_35). 89 of 96 fnb TRAIN_DONE (B2 11, C2 12, D2 6).
    - 24288878 `t125_hchain`: array 51 (C2_C19M16_counts_real_s0), 11:21 UTC.
  - 11:41 UTC: C, D CHAIN_DONE; 94 of 96 fnb TRAIN_DONE (D2 10).
    - 24290367 `t125_hchain`: array 43 (B2_C19M16_counts_real_s0; task 1 of ~2), 11:41 UTC.
  - 12:01 UTC: **fnb done: 96 of 96 TRAIN_DONE**, array 24224310 96 COMPLETED, 0 failed.
    - 24291820 `t125_hchain`: array 59 (D2_C19M16_counts_real_s0), 12:02 UTC.

- N6b (part 2 on the 5 finished chains, PI decision 2026-10-17), 2026-10-17.
  - Chunk `chunk/t125-N6b` (ec77f69, merged fast-forward into exp/t118-counterfactual-f, pushed):
    `hmap/aggregate.py` expects the 8 C19M16 real seed-0 f (`--chain-track`, `all` for 64), reads a
    chain only with CHAIN_DONE (never `*.partial`), gives each missing f `readings_skipped` entries
    "chain not finished", and reads `merge_error.json` and the h fits' step cap and capped rows;
    `summary.py` adds the "5 of 8 designs" banner, the part-1 device note (`--run-device`), CRPS > 20
    counts beside the means, the merged-mixture and step-cap tables.
  - Kit K13: `/project/def-maxwl/mforooz/t125/code/K13`, `GIT_SHA` ec77f692e043126e1485e761eabe571b6c741058
    (rsync as K11, wheels from K11).
  - 24312234 `t125_testK13` (CPU, 8 cores, 32000M; one-off `t125/n6b/test_K13.sh`): test_t125_h_aggregate.py
    23 passed, test_t125_e2e.py 13 passed (2:51).
  - 24312235 `t125_agg5of8` (CPU, 4 cores, 16000M; one-off `t125/n6b/agg_sum_5of8.sh`: h_agg.sh's python
    command with agg_dir `h/agg_5of8`, then summary.py): COMPLETED 1:20, MaxRSS 4.4 GiB. "38 alone present;
    chain f 5 of 8 finished, 3 missing (chain not finished); 1192 readings". Outputs `h/agg_5of8/`,
    `summary_5of8/GRID_SUMMARY.md` (21 "CRPS > 20" lines). Local copies and the criterion-12 names in
    `cruxvault/results/t125/` (see FIR_PATH.txt); digest `cruxvault/results/t125/part2_digest.md`.
  - Merged mixture G 512: C, A2, C2 no bin above 512 components (merged = exact); D per-bin max 8.6e-4,
    averaged 8.9e-5; D2 9.7e-4 and 1.7e-4 (bar 1e-3). Fits at the 4000-step cap: fnb 22 of 96, true 11 of 36;
    fnb capped NB rows 584 in 30 fits; chain capped rows 0.
  - To do when A, B, B2 have CHAIN_DONE: re-run `agg_sum_5of8.sh` with a new agg dir (or h_agg.sh on K13) and
    replace the "5 of 8" files.
- N5b (t125-P2i, PI 2026-10-17: split the 3 slow chains across pair shards; no number changes).
  - Code: `score.py chain --pair-shard K/N` (scores only the pairs i with i % N == K that have no
    kept pair file; writes nothing else) and `--assemble-only` (writes scores.json, merge_error.json,
    CHAIN_DONE from the kept pair files once all are kept); `h_chain.sh` 10th argument n_shards
    (array task T = f index T // n, shard T % n; the shard then assembles under
    `chain/<f_run>.assemble.lock`) or `assemble`. Commit 20f2a76 (chunk/t125-P2i), merged 61ae10a.
    Tests: test_t125_h_shard.py 9 passed and test_t125_slurm_h.py 124 passed (laptop); CPU job
    24312258 (one-off `t125/smoke/p2i/pytest_cpu.sh`): test_t125_h_score, _h_mixmerge, _h_shard,
    _h_speed, _h_aggregate, _e2e 106 passed, 2 skipped (no .git, no CUDA), 24:32.
  - Kit K12: `/project/def-maxwl/mforooz/t125/code/K12`, `GIT_SHA` 61ae10ac5fcda593ade8b63ab47be348e528fbb9
    (rsync as K11, wheels from K11).
  - 16:31 UTC: cancelled 24272487_3 (A), 24272487_11 (B), 24290367_43 (B2), each with pairs 0-10
    kept (pairs 0-7 took about 1 min each, pairs 8-10 about 85-105 min each). Submitted
    24313498 `t125_hchain` on K12: array 30-39,110-119,430-439 (A, B, B2 at n_shards 10), script
    resources, `--time=12:00:00` (b2; `--test-only` put 1 day at 08:32 UTC next day, 12 h at 23:54 UTC;
    a shard holds at most 3 of the 27 open pairs).
  - Real-data equality, chain C: `chain/C_C19M16_counts_real_s0.partial` copied to
    `/project/def-maxwl/mforooz/t125/smoke/p2i/real/chain/` (runs_true, runs_fnb symlinked to `t125/h`);
    24313465 `t125_hchain` on K12, array 19, n_shards `assemble`, `--time=1:00:00`; compared by
    `smoke/p2i/compare_chain.py` to `t125/h/chain/C_C19M16_counts_real_s0`: COMPLETED 0:47 (g32),
    CHAIN_DONE written. **Equal**: the same 52 files; CHAIN_DONE and merge_error.json byte-equal; all
    1900 records of the 50 scores.json bit-equal (serialised JSON); each scores.json equal but for
    created_utc, timing and `h.spec` (the smoke's out root reaches the h run dirs through its
    runs_true / runs_fnb symlinks; 0 of 50 differ once that path is mapped back).

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

### D2 twin nondeterminism, measured (N3)

2026-10-16, job 24173033 (c294, CPU, 4 cores). Run D2_C19M16_counts_nocov_s0 (index 357; trained
on CPU by 24166139, first scored there). Second scoring = eta_train.sh's exact `score.py trained
... --workers 4` on a copy in `/project/def-maxwl/mforooz/t125/smoke/d2twin_rescore/` (844.7 s).
The compare step in the job failed (KeyError: `crps_oracle_scaled_all` is only on shuffle
records); the fixed `d2twin_compare.py` (stdlib) was re-run on the login node on the two
scores.json: `d2twin_rescore/compare.txt`.

| metric | records | records that differ | max / median abs diff | 3-seed wobble per record, median / max (max-min) | seed means range |
|---|---|---|---|---|---|
| crps_all | 140 | **0** | 0 / 0 | 5.3e-3 / 2.6e-2 (102 records) | 8.9e-4 |
| crps_oracle_scaled_all | 38 | **0** | 0 / 0 | 2.7e-3 / 2.2e-2 | 2.2e-3 |
| scale_error_all | 38 | **0** | 0 / 0 | 6.5e-3 / 2.2e-2 | 1.8e-3 |
| mse_log1p_all | 140 | **0** | 0 / 0 | 9.4e-4 / 5.8e-2 (102 records) | 2.7e-3 |

The two scorings of this run are bit-identical in every record (relative difference 0). So on
the real data the t125 D2 nocov scoring is deterministic run to run; the failing
`test_twin_scoring_is_deterministic[D2]` (fixture data, K10 and t118 K8) does not show here. The
seed wobble is the only spread. (Seed wobble over 102, not 140, records: shuffle records are keyed
by the shuffled pid, which differs by seed.)

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

- 2026-10-17 about 01:58 UTC (N3): `ssh -o BatchMode=yes nibi` exited 255 (`Permission denied
  (publickey,keyboard-interactive,hostbased)`). All Nibi work stopped until the PI runs
  `hpc up nibi`. Last state read (01:37-01:58 UTC): 427 train tasks COMPLETED, 2 RUNNING
  (24183485 of 234, 235, 236), SCORE_DONE 426, LAW_DONE 421; law batch 24191400 (215, 222, 237,
  239) submitted. 429-431 are stopped (scoring hang, code defect). They run on without the laptop.
  To do on resume: law for the last A2 indices (234-236 and any missing), then the 429-431 fix
  decision; eta_agg.sh with OLD=/project/def-maxwl/mforooz/t125/old_grid/results_summary.tsv
  (results_summary.tsv and checks_*.json already rsynced there on 2026-10-16).
- 2026-10-17 02:48 UTC: Nibi back (coordinator); N3 resumed.
- 2026-10-17 about 04:04 UTC (N3): `ssh -o BatchMode=yes nibi` exited 255 again (mux broken pipe,
  then `Permission denied`). All Nibi work stopped until the PI runs `hpc up nibi`. Last state read
  (03:34 UTC): SCORE_DONE 429, LAW_DONE 429 (law 24200232 for 213, 234-236 all COMPLETED);
  24200215_429/_430/_431 RUNNING 45 min on c99, each score.py at about 97 % of one core (working,
  not hung). They run on without the laptop. To do on resume: when 24200215 ends, check 432
  SCORE_DONE / 432 LAW_DONE; re-run `bash /project/def-maxwl/mforooz/t125/eta/oneoff/run_device.sh`
  (fills score_job of 429-431 in `eta/agg/run_device.tsv`; the train columns are already complete);
  then eta_agg.sh with OLD=/project/def-maxwl/mforooz/t125/old_grid/results_summary.tsv.

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

- 2026-10-17: **scope corrected.** There are 8 across-track real f (2 rows × 4 designs), not 16, and they are the most costly (~242 pairs each). The repair reading and the chain now use the 8 real seed-0 f of ONE track, C19M16, all 8 designs: 96 fnb fits (8 f × 12 h versions) and 8 chains. This supersedes "16 across-track f".
- 2026-10-17: the merged-mixture bar holds **per bin**: use the smallest G in {256, 512, 1024} whose per-bin max relative error on crps_log is < 1e-3 on the measured deep bins of the C19M16 f. Both per-bin and averaged errors go in the grid.

- 2026-10-17: the count runs trained on mixed devices (GPU slice for A, B, C, D and A2 192–210; CPU for A2 211–239, B2, C2, D2, Aold) are accepted. The device of every run is recorded in the grid and the deck, with the note that cross-row design comparisons also cross devices.

- 2026-10-17: K_MAX stays 4,096. Capped bins (NB support wider than 4,096 counts; 17 of 262,144,000 training bins in the timing fit, 0 in validation) are counted per run and reported in the grid and the deck.

- 2026-10-17: lab deck = one deck, two mini-arcs (shared motivation and background; part 1; part 2; one discussion). Part-1 result slides: the depth law still fails; design A gains; covariates help the mean.

- 2026-10-17: part-2 result slides: the complexity grid of h on the true X′; repair fails (true-X′ h beats fnb-trained h in the chain); chain vs the old grid.

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
- 2026-10-17: deck v1 published (private): https://claude.ai/artifact/AfLpe1acvSC9i43jg28AqS ; mirrored in cruxvault/results/t125/DECK.md.
