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
