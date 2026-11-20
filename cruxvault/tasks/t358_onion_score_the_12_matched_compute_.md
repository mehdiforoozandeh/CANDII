---
id: t358
type: task
title: Onion: score the 12 matched-compute runs on 256 regions (conv autoencoder, conv + attention, U-Net, Perceiver IO)
category: implementation
parent: t152
blocked_by: None
refs: q39
hypothesis_refs: 
status: open
created: 2026-11-18T16:21:21
updated: "2026-11-18T23:23:12"
---

# t358 — Onion: score the 12 matched-compute runs on 256 regions (conv autoencoder, conv + attention, U-Net, Perceiver IO)

Refs:: [[q39_do_the_context_rungs_2_to_6_convolution_\|q39]]

## Why

ready queue entry 8: does the architecture order hold at equal FLOPs

## Current scope (close-out reconciliation, 2026-11-18)

Ready queue entry 8 (`closeout/ready queue for scaling.md`); item file `closeout/closeout A9 matched-compute runs.md`.
- Prerequisite: the 12 trainings queued on Fir (4345024 to 4345035) are DONE. Never resubmit them.
- Command: the option set of `slurm/onion_eval/e1_pair_check_p256.sh` score mode with `CLUSTER=fir`, eic_v1 tables, `REFS_FROM=$O/eval/eic_v1/baselines`, `SCORE_SUBDIR=p256_sv`, one `submit_arms.sh` call per run.
- Cost: 322.2 CPU-hours for the three CPU models + 3 Perceiver IO seeds (cost not measured); 108 jobs.
- Decision rule: item A9 file, section 6 (difference of seed means beyond the larger seed range). PI decision open: confirm or reverse the per-rung narrowing of notepad L289.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
