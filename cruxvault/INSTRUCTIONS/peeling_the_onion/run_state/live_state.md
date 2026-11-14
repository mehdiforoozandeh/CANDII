# Live state of the onion run (foreman's ledger; update at every launch, return and submit)

Updated: 2026-11-13 (after the AlphaGenome 2 start-now wave)

## Goal chain (unchanged)
EIC build -> collect (D6b) -> freeze eic_v1 (D6f) -> evaluation set (E7a/E7b), recount, second experiments, priority 4
-> ladder rungs 1-6 -> cross-rung, mix-and-match, evolution, stage 2, stage 3, resolution arms -> final model
-> single blind pass (with the AlphaGenome 2 trained-on comparison) -> /prezit deck + 12-page /storify.
PI priorities: imputation, covariate artifact correction, calibrated uncertainty.

PI DIRECTIVE 2026-11-14: results and insights in 3-4 days; iterative deployment in paper-section order (principle `iterative-deployment`). The 18-day ladder order is superseded by Amendment 9 (being written). NOTE: the Bash tool's cwd is the MAIN checkout now: use absolute paths to the onion worktree.

PI DIRECTIVE 2026-11-14: the crux tree and taskhub are the one record of the research; run a recorder chunk (opus, own worktree) at EVERY milestone: questions, hypotheses with expectations, findings with evidence, task states.

## Agents running
| recorder CR1 (opus) | backfill of the vault: findings, questions, task states of the whole run; recording rule file | review, merge; then a recorder per milestone |
| planner (fable) | Amendment 10: execution sheet of the adopted two-iteration parallel plan (rulings 43-46) | launch SP2a, SP2b, SP2c, SP3, then FWa, MTa, ... in its order |
| TRa (opus) | trainings now: rungs 2-6 base, rung-4 scrambled_covariates and no_eval, 7 selection arms at 20k (held; one per rung checked first) | log job ids; scoring by FWa/FWb/MTa |
| A1k (opus) | configs + trainings: average_input (D10), two_step_native (D15) on rung 1 | review code, merge; log job ids |
| SF0 reviewer (opus) | review of the scorer base (branch chunk/onion-SM at 17f68d5; baselines already submitted from it) | merge if ACCEPT |
SHIP TIMES (ruling 43): ship 1 = 2026-11-15 06:00 UTC (Methods + model-free Results; deck cut 03:00 UTC); ship 2 = 2026-11-15 18:00 UTC expected, hard 2026-11-16 08:00 UTC; iteration 2 = 2026-11-17 12:00 UTC.
| I1a (layers rerun with pseudoreplicate pairs), I1e (Methods, merge), AGf (pack fix), SF0 (scorer base) | iteration-1 and scorer work in flight | review, merge |
| agent | task | on return |
|---|---|---|
| (E7a returned) | eval set on eic_v1 (sha b5f7a4cf), check OK, average table done; baseline scorings resubmitted at 128000M: 27179565-27179568; layers 27164943 running; job watch -> E7b (collect: scorer validity, scoring cost AND memory), then gate G2, L1a, RCa, BRa. Ladder scoring memory: 128000M until measured | - |
| (QC fill) | batch 1 done (400/400 ok); batch 2 = job 27235942 (nice 500); batches 3-6 one at a time, specs in results/t211/SUBMITTED.md; QCb must fix `collect` | - |
| (L1a returned) | RUNG 1 RUNNING with the staging fix (kit L1fix; ONION_STAGE=1 for every later ladder submit): training 27197516-27197676, 24 runs, ~56 min each, scoring behind each at 128000M; a name watch fires when no mc_rung1 job is left -> L1b (collect and settle), with E7b's baseline validity. Open: rung 2 CPU speed is 2.7x the Fir figure | - |
WAITING ON THE PI: nothing (PI cancelled the slow rung-1 jobs, 2026-11-13).
| SF0 (SM builder), SP1, EVd (ERx builder), PCk (opus) | FAST SCORER wave S0 (Amendment 8): merge the scorer base + cross-cell rule + pair_regions + per-region pair sums; prediction store; evalset derive (nested pair regions); pair-regions check tool with thresholds committed first | review each; then SBa (4 baselines at 256), SP2, SP3; then SRa (6 rung-1 runs at 256), PCa (choose p) |
| (SM builder returned) | commit 9cd5f04 on chunk/onion-SM (workers, staging, 2-thread pin; baseline scoring 1:21 h on 16 CPUs): NOT merged yet (review after Amendment 8 fixes land) | - |
| SM reviewer (opus) | review of 1e94d92 (memory change, bit identity) | merge if ACCEPT |
WAITING ON THE PI: nothing (PI cancelled the 28 old-scorer jobs, 2026-11-14). TODO: resubmit the 4 baseline scorings and the 24 rung-1 scorings with the fast scorer (submit_arms.sh SCORE_ONLY=1) after SM's time work is merged.
| (region experiments: tools, configs and all lists ready; TR, SAx, STt2, RBk2 merged) | DONE: RB1a (42 budget-curve trainings on rung 1, all complete 2026-11-14 02:41 UTC; scoring = RB1s, waits for the fast scorer). Waiting: RTa (36+ region runs on rung 4; needs the ladder version), PRg (--per-region; after SM's time work), ECa | - |
| (deck done) | law-of-total-variance deck: cruxvault/presentations/onion-total-variance/; artifact H8ShBq8tJ1eMzwpjNpdU87 | - |
| (OXa returned) | second experiments submitted (jobs 27184156-27204216; notes results/t264); OXc collect when they end; 13 rows wait for their tail partners (with D6u). D6c (priority 4) HELD until rung 1 is through (fair-share) | - |

eic_v1 FROZEN 2026-11-13 14:30 UTC (hash a068a867; 249/267 train, 42/45 val, 37/40 controls; 8 scorable assays). Collect: DONE. TODO: renote rows 132 and 211 when their jobs end (kit D6y). TODO: D6u update when tail rows verify (watchdog 27124200; `tail_watch.py status` for an interim view).

PI RULING 38: freeze eic_v1 at ~2026-11-13 14:30 UTC with what is verified; watchdog 27124200 for the tail; update versions eic_v1_1, ... with notes (chunk D6u, t285).
SCHEDULE (UTC): 12:30 D6b collect of done rows; 14:30 D6f freeze eic_v1; then E7a, E7b, L1a, OXa, D6c.
WAITING ON THE PI: nothing (PI cancelled 26853275 on 2026-11-13).

Evaluation regions: DONE (t283), /project/def-maxwl/mforooz/ONION/eval/regions_v1/regions.json, 256 regions, sha 8e6b7a43.

AlphaGenome 2 fetch: DONE (t284), 8,960 calls, cache 4.0 G on Nibi $AGD; label check 0 leaks. `pack` runs after the last dataset version and its evaluation set exist.

QUEUE LIMIT: Nibi allows 1000 queued jobs per user, array tasks count. Priority for slots: dataset rows > ladder > second experiments > QC fill.

## Monitors
| monitor | fires when | then |
|---|---|---|
| Nibi queue watch (30 min, re-arm) | signal pass 26853275 ended, or onion_build <= 6 | see "Triggers" |

## Cluster jobs (Nibi unless stated). NEVER RESUBMIT.
| what | job ids | state at last look |
|---|---|---|
| EIC build, original arrays | 26853261, 26853270, 26855580, 26855582, 26857540 | 20 tasks running at 00:40 UTC (5 are 3-day rows: 57, 113, 280, 389, 396) |
| signal pass | 26853275 | pending |
| hang row 83 | stage 27079125, build 27079130 | queued/running |
| smokes from kit KF2 (rows 117, 211 / 95 / 132 / 313) | 27081348 / 27081352 / 27081354 / 27081357 | running, each past its old failure point (313 not yet past Picard) |
| 26 resubmitted rows from kit KF3 | 27087747/27087748, 27087755/27087756, 27087780/27087789 | submitted 2026-11-13 |
| row 396 resubmitted from kit KF4 (Java heap 58 G, 192000M, 3 d) | stage 27105142, build 27105143 | running since 2026-11-13 05:20 UTC; mark stage in ~33 h |
| row 216 resubmitted from KF4 (slow bwa sampe, 3 d) | stage 27112837, build 27112838 | submitted 2026-11-13 06:40 UTC |
| row 232 resubmitted from KF4 (slow bwa sampe, 3 d) | stage 27117063, build 27117064 | submitted 2026-11-13 07:25 UTC |
| EXPECTED TIMEOUTS (1-day DNase rows still in bwa sampe): 337, 381, 385, 387, 391 | 26855580_<row> | on timeout: resubmit from KF4 with ONION_TIME=3-00:00:00 |
| AT RISK (old kit, 12 G Java heap): rows 389, 57 | 26855582_389, 26853270_57 | if `GC overhead limit exceeded`: resubmit that row from KF4 with ONION_MEM=192000M ONION_TIME=3-00:00:00 (57 after the signal pass) |
| signal array for rows 10, 37, 63, 84 (kit KF4) | 27123484 | running since 08:35 UTC |
| rows 46, 52 (validation histone, kit KF4) | stage 27124279, build 27124280, signal 27124281 | submitted 08:50 UTC; ~4-5 h |
| row 56 (validation DNase-seq, 7 d, 192000M, kit KF4) | stage 27124283, build 27124284, signal 27124285 | submitted 08:50 UTC |
| row 388 (train DNase-seq, 128000M, kit KF4) | stage 27143105, build 27143106 | submitted 12:05 UTC; 1-day limit, first run took 20 h |
| row 337 (train DNase-seq, 3 d, 192000M, kit KF4) | stage 27150097, build 27150100 | submitted 13:20 UTC |
| nine 1-day DNase-seq rows, all rerunning at 3 d / 192000M from KF4 | builds 27150100 (337), 27161195 (381), 27165180 (387), 27165188 (384), 27165190 (385), 27165192 (390), 27165194 (391), 27167692 (394), 27167694 (395) | submitted 13:20-16:20 UTC |
| original 3-day rows 57, 113, 280, 389 (old kit; 12 G Java heap risk for 389 and 57) | 26853270_57, 26855582_113/280/389 | limits end 2026-11-14 ~16:30 UTC; on failure: `ONION_TIME=3-00:00:00 ONION_MEM=192000M` resubmit from KF4 (add the row to resub_dnase_tail.sh: it checks array 26855580 only, so resubmit these by hand) |
| row 211 (train DNase-seq; timed out at 1 d; rerun 3 d / 192000M, kit KF4) | stage 27226297, build 27226298 | submitted 2026-11-14 01:17 UTC |
| second-experiment row 2028 (DNase-seq, Java heap; rerun 3 d / 192000M, kit KX4) | stage 27232990, build 27232991 | submitted 2026-11-14 02:40 UTC |
| HEAP RISK (12 G default heap, library-dependent): rows 388, 132, 216, 232, 57, 280, 389 | their running jobs | on 'GC overhead limit exceeded': resubmit with ONION_TIME=3-00:00:00 ONION_MEM=192000M; every new DNase-seq submit uses 192000M |
| row 113 (train DNase-seq; Java heap after 2 d; rerun 3 d / 192000M, kit KF4) | stage 27246582, build 27246583 | submitted 2026-11-14 05:20 UTC |
| tail rows COMPLETED since the freeze: 46, 52 (validation histone), 388, 390 (DNase train) | - | first update eic_v1_1 possible (chunk D6u) |
| BASELINE SCORINGS on the real evaluation set (kit SBa, 16 CPUs, 64000M, 4 h) | 27265497 marginal_nb, 27265498 position_average, 27265499 depth_only_scaling, 27265500 pseudoreplicate_floor | submitted 2026-11-14 ~11:15 UTC; ~1.5 h each once started |
| WATCHDOG (tail_watch) | 27124200 | pending on afterany of the live tail jobs; files under $O/eic/tail/ |
| Max's training (not ours) | Fir 3609189 | his |

## Triggers (what to do when something ends)
1. Signal pass 26853275 ends -> from kit KF3: `submit ... 1 50 2 46,52` and `ONION_TIME=3-00:00:00 submit ... 1 50 2 56`.
2. KF2 smokes end -> `renote` from kit D6y for rows 95,117,132,211,313 (command in the plan Log).
3. Queue empty -> D6b full collect (verify, pending, paths exercised) -> D6f freeze -> launch E7a, RCa, BRa, OXa (+priority 4), L1a.
4. E7a done (frozen evaluation regions) -> AGx (AG2 fetch, ~9,700 calls; `status` needs `--outputs chip_histone,dnase,atac`) -> AGva/AGvb.
5. FDb (final model) -> AGwa/AGwb. BPa (blind pass) -> AGz.
6. Max's sprint 12 score table appears -> add the comparison to the AGm note.

## Open with the PI
None blocking. Tile access from Max: PI said skip (2026-11-13).

## Kits on Nibi in use
KF (original build, jobs running), KF2 (smokes running), KF3 (resubmitted rows), KF4 (git 0d578dc; use for ALL later submits), D6y (renote), A5chk (test check).
