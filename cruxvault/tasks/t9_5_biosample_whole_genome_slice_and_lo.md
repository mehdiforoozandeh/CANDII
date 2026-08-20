---
id: t9
type: task
title: 5-biosample whole-genome slice and loader throughput benchmark on a GPU node
category: implementation
parent: t3
blocked_by: t6, t7, t8
refs: 
hypothesis_refs: 
status: done
created: "2026-08-19T01:47:55"
updated: "2026-08-19T04:18:15"
---

# t9 — 5-biosample whole-genome slice and loader throughput benchmark on a GPU node

Refs:: _(none)_

## Why

de-risks every scale-up: proves writer to reader end to end and measures real windows/s from /localscratch before hundreds of node-hours are spent

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [5-biosample slice and throughput benchmark](results/t9/t9_summary.json) — 5-biosample whole-genome EIC slice at /project/def-maxwl/mforooz/CANDI_STORE/eic_slice (deliberately NOT eic/, so t10 cannot collide): T_DND-41, B_DND-41, V_DND-41, T_IMR-90 (widest in EIC, 26 tracks), T_G401 (narrowest, 1). 42 counts cols, 39 peaks/pval cols, 23 chroms, 121,241,684 bins/track, 4,985,338,865 B verified on disk by the orchestrator. SLURM array 55535967 all COMPLETED 0:0, 14 min wall (30-586 s/task, MaxRSS 7.45 GB); manifest 62 s; verify OK. BENCHMARK job 55540488 on fc10718 (MIG 1g.10gb, 4 CPU), COMPLETED 0:0. D5 staging of 5,869,126,833 B to /localscratch in 7.30 s = 804 MB/s. THROUGHPUT at context_bins=768, 8 assays, counts+peaks: nw=1 403.8 win/s, nw=2 683.3, nw=4 919.3, nw=8 871.4. RATIO TO THE PLAN'S ~10k WINDOWS/S: 0.040, i.e. 25x short - but the two are different quantities. The plan's 10k is a raw h5 slice-read rate (0.07-0.10 ms per 768x9 block); the loader additionally reads 8 counts + 8 peaks columns, one-hots 19,200 bp of DNA, thins binomially per element and assembles the batch dict. 404 win/s = 50 batches/s is the honest end-to-end figure. CAVEAT ON nw=8: it ran on 4 ALLOCATED cores - a 10-CPU request against the mandated 1g.10gb MIG slice never scheduled (projected 8 h out; three jobs cancelled unscheduled), so a properly-resourced 8-worker number needs a resource shape the project's hard --gres rule forbids. Throughput saturates at nw=4 within the allocation. At L=6144: nw=1 104.2, nw=8 272.2, which is 2.06x more bins/s per window than L=768. With pval: nw=1 324.6, nw=8 792.7. SIZES bytes/value: counts 0.2948 (plan 0.273, ratio 1.08), peaks 0.0147 (0.010, ratio 1.47), pval 0.7222 (0.771, ratio 0.94). Extrapolated PER TRACK and said so explicitly - 439 counts cols including 76 controls, 363 peaks/pval cols, 89 biosample dirs; per-biosample scaling would give a wrong 88.7 GB because the slice averages 8.4 tracks/biosample against the corpus 4.93. counts 15.69 + peaks 0.65 + dna/mask 0.88 = 17.22 GB against the plan's 12 GB, RATIO 1.43, OUTSIDE the +/-20% the task asked to confirm; pval 31.78 GB against +40 GB, ratio 0.79; grand total 49.00 GB against 52 GB, ratio 0.94. THE PLAN'S TWO SECTION-2 TOTALS ARE NOT BUILT ON THE SAME COLUMN SET: 12 GB is exactly 363 signal cols x 121,241,684 x 0.273 and omits the 76 control columns and dna+mask despite the line reading 'counts + peaks + dna', while +40 GB is 439 cols x 0.771. Rebuilt consistently, the plan's own B/bin gives 15.4 and 33.9 GB and the measurements land at ratios 1.12 and 0.94. D7 FIRED BOTH WAYS AND 52,051 WAS NOT THE CEILING: T_DND-41 uint16 (max 8,008, chipseq-control), B_DND-41 uint16 (52,051, DNase-seq, as predicted), V_DND-41 uint16 (8,008), T_IMR-90 uint32 (151,858, DNase-seq), T_G401 uint32 (448,686, DNase-seq). Two of five biosamples go uint32; real EIC DNase-seq reaches 448,686 reads in one 25 bp bin, 6.8x the uint16 ceiling, and that doubling is the entire counts overshoot. The PI should expect an EIC counts layer materially larger than 12 GB. D22 PASS: 40/40 shared eval windows byte-identical between num_workers 1 and 8, deterministic auto-set on train=False. ROUND-TRIP: counts and peaks BIT-EXACT on B_DND-41/DNase-seq/chr1 (with the 9,958,257 to 9,958,256 D13 truncation) and T_DND-41/H3K4me3/chr21; pval max abs err 0.0050011, which is the 0.005 quantization bound plus at most 1.1e-6 of float32 output rounding - not a codec defect, but STORE.md's flat 'max 0.005' failed a strict assertion and has been corrected to '0.005 plus a float32 epsilon' in 1afc99f. EIC METADATA GAP MEASURED AND CONFIRMED: 439 track dirs, 363 CSV rows, 76 dirs with no CSV row, and ALL 76 ARE chipseq-control - the exact problem t5 solved for MERGED and nobody has solved for EIC. 0 CSV rows without a dir, 0 empty fields among the 363. Three of this slice's 42 tracks got null metadata, all controls (D19, correct). t5's control_metadata.csv is MERGED-only and its biosample names do not exist in EIC, so t5's scan MUST be run against DATA_CANDI_EIC before t10 or 76 of 439 EIC counts columns train as all-MISSING. EIC HAS 89 BIOSAMPLE DIRECTORIES, NOT 91 - the plan's 91 counts aliases.json and navigation.json, the same raw-ls error that made MERGED 367 instead of 361. /localscratch on fc10718 is 7.0 TB, not the 561 GB section 2 states. Python env used: /project/6014832/mforooz/EpiDenoise/candi_venv (3.10.13, torch 2.6.0, h5py 3.12.0) because ~/scratch/enctest_env has no torch and is 3.11, which pyproject.toml forbids. NO REPO CODE CHANGED; gates re-run: pytest 496 passed, golden 0 ULP params=2,353,634 sd=472362cea987. Evidence in cruxvault/results/t9/.

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
