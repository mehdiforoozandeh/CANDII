# Live state of the onion run (foreman's ledger; update at every launch, return and submit)

Updated: 2026-11-13 (after the AlphaGenome 2 start-now wave)

## Goal chain (unchanged)
EIC build -> collect (D6b) -> freeze eic_v1 (D6f) -> evaluation set (E7a/E7b), recount, second experiments, priority 4
-> ladder rungs 1-6 -> cross-rung, mix-and-match, evolution, stage 2, stage 3, resolution arms -> final model
-> single blind pass (with the AlphaGenome 2 trained-on comparison) -> /prezit deck + 12-page /storify.
PI priorities: imputation, covariate artifact correction, calibrated uncertainty.

## Agents running
| agent | task | on return |
|---|---|---|
| (E7a returned) | eval set on eic_v1 (sha b5f7a4cf), check OK, average table done; baseline scorings resubmitted at 128000M: 27179565-27179568; layers 27164943 running; job watch -> E7b (collect: scorer validity, scoring cost AND memory), then gate G2, L1a, RCa, BRa. Ladder scoring memory: 128000M until measured | - |
| (QCa returned) | QC fill batch 1 = job 27164529 (400 rows); batches 2-6 by the foreman when the queue has room (commands in results/t211/SUBMITTED.md); QCb must fix `collect` | - |
| L1a builder (opus) | RUNG 1 submit on eic_v1: 8 arms x 3 seeds, scoring at 128000M, Nibi CPU timing probe | log job ids; L1b collect when runs end; E7b must confirm the baselines |
| SM builder (opus) | scorer memory: per-region streaming, bit-identical scores (PI instruction 2026-11-13) | independent review (score.py is the selection scorer); merge; new memory figure for the ladder |
| V7, ERx, STt, RBk builders (opus) | region experiments, wave R1 (Amendment 7): tasks and questions; evalset --regions-file and superset; training strata tool; region configs | review each; merge; then TR (after SM's diff), SAx (after L1a merges), then RB1a, STa, PRg |
| OXa builder (opus) | second experiments build (evaluation-only add-on), current build code | then D6c batch 1 (priority 4); OXc collect |

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
