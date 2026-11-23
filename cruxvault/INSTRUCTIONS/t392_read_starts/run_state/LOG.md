# Log (t392)
- 2026-11-23T20:15Z run_state created; plan sent to foreman
- 2026-11-23T20:16Z foreman: plan approved; merge exp/t370 into umbrella after PR 65 lands; report input checks as soon as done
- 2026-11-23T20:35Z planner done (plan.md, 15 chunks); input checks + questions (gate wording, store location) sent to foreman
- 2026-11-23T20:40Z PI (via foreman): gate = read-level + floor (D9); build site ASAP -> Fir (D10). Fir scratch 2521GiB/19TiB, files 829K/1000K. Sent wall-time est Fir ~3 d vs Nibi ~3.5 d
- 2026-11-23T20:43Z plan review done: 23 fixes accepted, sent to planner for amendment; asked foreman (iv) new dataset class; 600 cap = CPUTimeRAW
- 2026-11-23T20:43Z foreman: (iv) new class yes, not a deviation; 600 measure ok; transfer not started yet
- 2026-11-23T20:48Z planner amended plan (24 chunks, waves 1-14). Foreman: both transfers started Nibi->Fir (6 tars 1,185.8 GB from tape -> /scratch/mforooz/rs392/tars/; 78 testbed readset folders -> /scratch/mforooz/rs392/work/testbed/). Watch Fir side; report landing/failure.
- 2026-11-23T20:48Z testbed transfer landed (18813836293 B, 156 files, matches source); reported. Wave 1: worktrees C1, C2 cut from 4712b34e
- 2026-11-23T21:18Z C7a: probe array 28712220 ran; GATE FAILED 4/4 (12 files, 2,289,096 differing bins each = chrY size; chrY missing from grid because read_starts uses bin25.MAIN_CHROMS). chr1-22,X exact. 0.35 CPU-h. STOP reported to foreman; asked "fix chrY and rerun probe" (rec yes). Waiting; preparing C3 fix only (no compute). Merged jobs.tsv.
