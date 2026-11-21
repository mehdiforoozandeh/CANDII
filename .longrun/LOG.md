# Log

Append-only. Written only by scripts/log.sh. Older entries rotate to log/.
- 2026-11-21T11:04Z longrun initialized (protected=cruxvault/INSTRUCTIONS/miniCANDI_closeout, job_prefix=rcp_)
- 2026-11-21T11:06Z coverage check: added R20-R22, D17; fixed R19 citation
- 2026-11-21T11:16Z PI tiers: P0+P1 only; priority 1 recipe, 2 control; R12 -> P1 with three analyses; R17 -> P3; IHEC out
- 2026-11-21T11:18Z R19 verified: 9/9 eic_v1_2 base runs p256_sv SCORED on Fir; Fir and Nibi tunnels up; 1 job of mine on Fir queue
- 2026-11-21T11:19Z planner spawned (fable) -> .orchestrate/plan.md; commit policy granted by PI ('start it and keep working'); reading time default 10 min
- 2026-11-21T11:37Z plan returned: 26 chunks, 8 waves; 256-region scoring measured 87.6/92.9/114.1 CPU-h per run-seed (2.7x older docs); upper bound ~12,000 CPU-h, 6,000 checkpoint in wave 5; D21-D25 recorded; cold review (opus) running; open PI asks: training-only Avocado retrain (~86 GPU-h), blind-row build (t184, ~22,700 CPU-h)
- 2026-11-21T11:43Z cold review: arithmetic sound (cum ~11,991 CPU-h), 4 factual fixes, coverage gaps, waits-in-chunk; sent once to planner with 15 amendments; D26-D27
- 2026-11-21T11:48Z wave 1 launched: C1-C10, 10 opus builders, base 57f3c9e8
- 2026-11-21T12:02Z ADDITION D29 read-disjoint training pairs: displaces nothing; ~150 CPU-h screen (9 run-seeds); serves R1 R3 R4; PI informed at next report
- 2026-11-21T12:09Z Fir: rcp_tests 4863258; U-Net probe train 4863259, predict 4863260-63, metrics 4863264 (kit RCP1)
- 2026-11-21T16:14Z G0: probe terms reproduce base seed0 exactly; scoring cost pass (22.0 vs 38.1); train cost fail (17.13 vs 14.12) -> continue, screen budget x1.50 = 5,209 CPU-h; rcp_tests 4863258 TIMEOUT, resubmitted 4888524 on RCP2 (12 h); screen submitted on RCP2 before test verdict (probe.md)
- 2026-11-21T16:17Z Screen submitted on RCP2: s1a 396 jobs, s1b 270, native 30, CNN seed 6, ctleval 4889711; 704 rcp_ jobs queued; ids in run_state/jobs_C12-14.tsv
