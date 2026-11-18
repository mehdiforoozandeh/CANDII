---
id: t350
type: task
title: Onion: finish the 256-region rescoring on Nibi (62 run-seeds of 7 arms) and rerun the final-pass command on the full arm-summary table
category: implementation
parent: t152
blocked_by: None
refs: q56, q48
hypothesis_refs: 
status: open
created: 2026-11-18T09:46:40
updated: 2026-11-18T09:46:40
---

# t350 — Onion: finish the 256-region rescoring on Nibi (62 run-seeds of 7 arms) and rerun the final-pass command on the full arm-summary table

Refs:: [[q56_do_the_ablation_conclusions_drawn_from_t\|q56]], [[q48_which_minicandi_design_choices_move_impu\|q48]]

## Why

decides rules 2.3, 2.4 and 2.5 and completes design rows 13, 24 and 25

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(none yet: 30 of 62 run-seeds were scored at 2026-11-18 15:55 UTC.)_

## Notes

- Submission (never resubmit): Nibi, kit CO9, `slurm/onion_eval/rescoring_p256_nibi.sh`, 62 run-seeds, 558 jobs (8 predict + 1 metrics each), job names `mc_<arm>_s<seed>`; list in Nibi `/project/def-maxwl/mforooz/ONION/runs_fir/SUBMITTED.tsv`.
- Final pass, from the repo root with `PYTHONPATH=$PWD/src`: `/Users/mforooz/miniforge3/envs/candii/bin/python tools/onion_eval/apply_rescoring_rules.py <arm-summary tsv> --as-of <UTC stamp>`. The arm-summary table is made on a Nibi login node, in a shell that submits no jobs, after `module load python/3.10.13 scipy-stack`, with `select_rule.py arm-summary --runs /project/def-maxwl/mforooz/ONION/runs_fir --sub p256_sv` from the kit.
- Operations fact (not science): Nibi runs the 256-region scoring about 10 times slower than Fir. At 00:50 UTC on 2026-11-18 one run-seed wrote 421 of 1,674 prediction requests in 80 minutes. The predict jobs used about 1 of 8 cores (1.1 of 8 at 02:12 UTC, node load 162), so they wait on input and output; MaxRSS sat at the 32000M limit. The cause was not diagnosed further. U-Net predict shards reach the 6 h limit at about 90 to 96% of their requests, so every U-Net run-seed needs one rescue pass with a 12 h predict limit (`SCORE_PRED_TIME=12:00:00`).
