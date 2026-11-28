# SE/PE cap at 25 bp: the PE parent thinned to one mate and capped per position (t396 C11)

Serves R44, R29, R30 and the SE views of h60. Data only. No model and no seeds. Every number below
names its experiment count. Here "n = 12" means 12 pairs from 12 experiments.

## Results

1. **PE parents (n = 12, all `train`): neither law explains the SE view (gate d), and neither law
   beats the other.** In the top 1% of bins, the start-space law `thinned_dm_distinct(c_PE, q = 0.5, a, K)`
   closed 0.365 of the gap against depth scaling. Its 95% bootstrap interval over experiments was
   [-0.073, 0.548]. The count-space law `count_saturation(0.5 c_PE, kappa, L)` closed -0.117, with the
   interval [-0.443, 0.482]. Both values are below the gate of 0.8. The interval of the difference
   included 0, so `beats_other_law` is `no` for both laws.
2. **Per pair (PE, n = 12, top 1%), the start-space law was a small, consistent gain over depth
   scaling.** Its deviance was below that of depth scaling in 11 of 12 pairs. The median ratio
   law / depth scaling was 0.957 (range 0.611 to 1.192). The attainable floor was a median 0.890 of
   the depth-scaling deviance. Thus depth scaling already sits near the floor, and the law recovered
   0.365 of a small remaining gap (pooled). The count-space law was below depth scaling in 6 of 12 pairs,
   with a median ratio of 0.993 (range 0.450 to 1.858).
3. **Genome-wide (PE, n = 12), no law did better than depth scaling.** The gaps closed were -0.003
   (start, interval [-0.054, 0.030]) and -0.051 (count, interval [-0.139, 0.015]). The median ratio
   law / depth scaling was 0.998 (start) and 1.002 (count).
4. **Against identity, both laws closed most of the gap (PE, n = 12, top 1%): 0.958 (start) and
   0.941 (count).** Identity is a weak baseline here, because the SE view has about half the reads of
   its parent. These two numbers show only that the thinning q = 0.5 is about right.
5. **Mixed parents (n = 9; 8 `train`, 1 `val`): the gate outcomes in `table.tsv` are not valid, and I
   do not interpret them.** In the top 1%, the attainable floor of the start-space law was above the
   depth-scaling deviance in 7 of 9 pairs. A negative denominator makes the gap closed meaningless.
   For example, the row `thinned_dm_distinct, mixed, train, top1pct` shows a gap of 2.230 and
   `explains = yes`, but the law's deviance (1.557e7) was above that of depth scaling (9.655e6). Per
   pair, both laws were worse than depth scaling in all 9 pairs genome-wide, and in 9 of 9 (start) and
   8 of 9 (count) pairs in the top 1%. The cause is known: q = 0.5 is wrong for the single-end
   replicates of a mixed experiment, which keep all their reads. The reads ratio SE / parent was 0.593
   to 0.934 on the mixed pairs, against 0.354 to 0.528 on the PE pairs.
6. **The pooled shape `a` was not stable under leave-one-experiment-out (n = 21).** Over the 21
   fits, `a` ranged from 2.34 to 998.8. Leaving out one of 3 experiments (ENCSR113MBR ATAC-seq,
   ENCSR700ILE DNase-seq, ENCSR788TRR ATAC-seq) moved `a` to about 976 to 999, i.e. near the
   occupancy limit (uniform positions). The fit on all 21 pairs gave `a` = 14.92 and `kappa` = 129.7.
   With `q` free on a grid of 0.30 to 0.70, the best `q` was 0.58 (with `a` = 0.633). One pooled `a`
   over ATAC-seq, DNase-seq and histone marks is a poor model; a few accessibility experiments set it.
7. **The size of the PE cap (n = 21).** The ratio (PE base) / (PE dedup-off) in the top 1% of bins by
   PE dedup-off starts was a median 0.803 on the 12 PE parents (range 0.546 to 0.938) and a median
   0.769 on the 9 mixed parents (range 0.357 to 0.906). Thus the PE dedup removes a median of about 20% (PE) to 23% (mixed) of
   the reads in the top 1% of bins. The source is `pe_cap.tsv`.

## Gate (d), as set before the run

A law explains a pair set if its gap closed against depth scaling is >= 0.8 in the top 1% of bins,
on leave-one-experiment-out fits (the SE rule). One law beats the other if the 95% bootstrap interval
over experiments of the difference of their gaps excludes 0. Outcome: PE parents, no law explains,
and no law beats the other. Mixed parents, not interpreted (result 5).

## What ran

- Pairs: the 21 `*.se` views whose parent is in the read-start store, from `rows.tsv` (C10b). There
  are 42 SE views in the tables (38 `train`, 4 `val`). The parents of the other 21 are not in the
  store; they are listed in `missing.tsv` of the tables. Rows with `in_eic` = `blind` are not read.
  Of the 21: 12 PE parents (all `train`) and 9 mixed parents (8 `train`, 1 `val`: ENCSR806XIU).
- How the SE view was made (`tools/t396/sepe_cap.py`, from `tools/onion_data`): mate 1 of every read
  pair, plus the single-end runs of a mixed experiment, was realigned as single-end by the
  experiment's own pipeline. The pipeline then deduplicated each replicate by the single-end rule
  (5' position and strand; DNase-seq: reads flagged 0x400 dropped in the tagAlign step). The dedup
  unit is the replicate (`dedup_1bp/UNIT_RULE.json`), so `K = 25 U` for U replicates.
- Laws per strand, 25 bp bins: start space, `caps.thinned_dm_distinct(c_PE starts, q = 0.5, a, K)`;
  count space, `caps.count_saturation(0.5 c_PE counts, kappa, L)` with L the SE read length. The 0.5
  in the count law is a deviation from the brief, found in C10a: without it the law predicts twice the
  SE reads at low counts. Baselines: identity and depth scaling (the genome-wide read ratio).
- Fit: one pooled `a` and one pooled `kappa`, leave-one-experiment-out (`per_assay.tsv`, one row per
  held-out experiment plus the row `all`). The optional row with the dedup test's `a` was not made:
  `dedup_cap/per_assay.tsv` did not exist on Fir when the fit ran.
- Floors: start space, `caps.simulate_floor` (20 draws) from the joint tables; count space, the
  start-space simulation pushed through the SE view's span kernel (`floor-count`, one draw).
- Jobs (kit `SC10b`, `.kit_stamp` git cce0535f; the tools are byte-identical to this branch): fit
  6742771, floor-count probe 6744223, floor-count array 6745574 (11 tasks), score 6747271. See
  `jobs_c11.tsv`. Cost: 2.24 allocated core-h (sum of ElapsedRaw x AllocCPUS from `sacct`; TotalCPU
  was not recorded).
- Outputs (`OUTPUTS.txt`): `per_assay.tsv`, `table.tsv`, `pairs.tsv` (per pair: D_law, D_identity,
  D_depth, D_floor), `pe_cap.tsv`. `bash run.sh` rebuilds them from Fir. The 21 statistics files
  and 21 floor files stay on Fir under `/scratch/mforooz/sc396/caps/{sepe,sepe_floor}`.

## What was not checked

1. The count-space floor used one draw per pair. Its draw-to-draw spread is not measured.
2. `aggregate` in `tools/t396/dedup_cap.py` does not guard against a floor above depth scaling. The
   gap closed and `explains` are then invalid (result 5). I did not change the tool (not owned).
3. A law for mixed parents (q = 0.5 on the paired replicates, q = 1 on the single-end replicates).
4. Per-assay shapes. With 21 pairs over 13 assays, a per-assay fit has 1 to 4 pairs per assay.
5. The `val` split as a held-out test. Only 1 pair is `val`, and it has a mixed parent.
6. The 21 SE views whose parents are not in the store.
