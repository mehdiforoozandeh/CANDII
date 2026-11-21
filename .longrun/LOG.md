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
- 2026-11-21T18:30Z PI: deadline within 24 h (by 2026-11-22 18:30 UTC, D31); yes to curve-NaN fix (D32), yes Avocado retrain (D33), no blind (D34)
- 2026-11-21T18:32Z cancel-own.sh cancel on fir: 4889385 4889386 4889387 4889388 4889389 4889390 4889398 4889399 4889400 4889401 4889402 4889457
- 2026-11-21T18:45Z Curve NaN fixed e8dec6a8; cancelled curve U-Net jobs, dirs to _nan_RCP2; Avocado chain to 4906099; 8 base ctl evals 4906119-26; 27 ctl-arm run-seeds + chained ctl evals; EIC val base U-Net s0 4906465, Average 4906466
- 2026-11-21T18:47Z RCP3 pushed (e8dec6a8). Curve U-Net reruns trains 4908746-4908912; s1c CNN 17 seeds submitted on RCP3 (102 jobs); CNN cost seed done at 10,000 steps in ~87 min
- 2026-11-21T21:55Z screen Pareto: winner halves U-Net (imp 0.4263, corr 0.7535, cal 0.6872 vs floors 0.4072/0.6491/0.4883); tests 69 env failures; 7 train timeouts; EIC jobs failed (avocado import, frag_len, KIT) -> C40 builder; kit RCP4 pushing
- 2026-11-21T22:07Z glossary: reply sent with unregistered terms: d20, gloss, glossary
- 2026-11-21T22:12Z RCP5 pushed; eic_val 4945457, avocado chain to 4945495, average 4943718
