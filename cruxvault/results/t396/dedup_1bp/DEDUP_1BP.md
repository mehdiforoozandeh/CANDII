# Dedup at 1 bp: mechanism and dedup unit rule (t396 C12b)

Serves R43, R24, R28. Data only. No model and no seeds.

## What ran

- Six `train` SE experiments from `picked.tsv`, all from the uniform layer with 36 bp reads. Three have 2 replicates: ENCSR956CTX H3K4me3, ENCSR928HYM H3K27me3, ENCSR925LJZ H3K36me3. Three have 1 replicate: ENCSR902EEI H3K4ac, ENCSR972SMV H3K9me3, ENCSR960VLF control. A one-replicate experiment has the unit label `pooled` with `n_units` = 1, and it is one dedup unit.
- A read position is the 5' end, i.e. `start` on the + strand and `end - 1` on the - strand, together with the chromosome and strand.
- Tool: `tools/t396/dedup_1bp.py` (`unit`, `shape`, `rule`). Job script: `slurm/t396/dedup_1bp.sh run`. Kit: `SC12b`. Jobs: `jobs.tsv`. Per-unit outputs: `fir/<experiment>/{unit,shape}_<label>.json`, pulled from `/project/def-maxwl/mforooz/t396/dedup_1bp`.
- `bash run.sh` rebuilds `mechanism.tsv`, `shape.tsv`, `UNIT_RULE.json` and `cost.tsv` from these files.

## Is dedup-on the set of distinct dedup-off positions per replicate?

Yes, exactly. The source is `mechanism.tsv`.
- Per replicate (9 units), dedup-on has no position that dedup-off lacks, and dedup-off has no distinct position that dedup-on lacks. Both counts were 0 in all 9 units. No dedup-on line repeats a position. For example, ENCSR956CTX br1 has 77,217,569 dedup-on lines and 77,217,569 distinct dedup-off positions.
- The pooled dedup-on set of each 2-replicate experiment repeats positions. The shares of repeated lines were 0.0021 (ENCSR925LJZ), 0.0037 (ENCSR928HYM) and 0.0263 (ENCSR956CTX). Thus dedup ran on each replicate, and the pooled view is the union of the deduplicated replicates.
- `UNIT_RULE.json`: `dedup_unit` = `replicate`, `on_equals_distinct_off` = true, `max_share_absent` = 0.0, over 6 experiments, 3 of them multi-unit, at tolerance 1e-3.
- Consequence for C10: use `K = caps.K_of(units)` = 25 U, where U is the number of replicates of the pooled view. This is the rule that C10a already implements for `replicate`.

## Is the 25 bp store sufficient for the mean law?

For the mean of the top 1% of strand-bins, yes, on these 6 experiments. The source is `shape.tsv`. Here "top 1%" means 2,470,617 strand-bins, i.e. 1% of the genome's strand-bins, ranked by dedup-off count.
- `dm_distinct` with the shape `a` fitted at 25 bp (`caps.fit_shape`) predicted the top-1% dedup-on sum within 2.8% on every replicate unit. Over the 9 units, the relative error ranged from -0.004 to +0.027.
- On the 3 pooled views with K = 50, the error ranged from -0.006 to +0.016. If the pooled view is treated as one unit, the target becomes the distinct pooled dedup-off count, and the error rose to between 0.023 and 0.115.
- Occupancy (`caps.occupancy`, i.e. uniform positions) erred by 0.023 to 2.75. Thus the Dirichlet shape is necessary.
- The shape `a` fitted at 1 bp and the shape fitted at 25 bp differed by up to 2.7 times (ENCSR956CTX br1: 0.50 against 1.37). The top-1% sum is not sensitive to this difference. With the 1 bp `a`, the error was below 0.011 on 6 of 9 replicate units, but it was 0.116 on ENCSR956CTX br1.
- The errors are not set next to a noise floor. These are exact counts on 6 experiments, and I did not compute a bootstrap over experiments. The 25 bp fit of C10 must supply that spread.

## What stays untested without 1 bp positions

1. PE and mixed rows. All picks are SE, and PE dedup groups fragments, not 5' ends.
2. Read lengths other than 36 bp, the native layer, and `val` and `blind` rows.
3. The per-experiment 1 bp shape `a` for the other experiments. The 25 bp fit replaces it.
4. Whether `a` holds below the top 1%. This test scored the top 1% only.
5. Replicate sets with 3 or more units.

## Cost of a full 1 bp build

The source is `cost.tsv`, which `run.sh` builds from `sacct` over the jobs in `jobs.tsv`.
- One job per experiment ran `unit` and `shape` on each replicate and on the pooled set.
- Over the 6 COMPLETED jobs, CPU-h per experiment was 0.019 to 0.307, with a mean of 0.0802 and a median of 0.0388. The largest experiment (ENCSR956CTX, 152 M dedup-off lines) took 0.307 CPU-h and 8.4 GB MaxRSS on 4 CPUs.
- The first ENCSR956CTX attempt (6727806) ran out of memory at 8000M with the whole-set tool. The tool now streams per chromosome.
- Full build: 0.0802 × 2,811 = 225.5 CPU-h (mean), or 0.0388 × 2,811 = 109.0 CPU-h (median). Here 2,811 is the number of measurement folders of the read-start store.
- The estimate is approximate. One job here covers one experiment, i.e. several units, not one measurement folder. The estimate also excludes the extraction of the tagAligns from the tars and the writing of a 1 bp store.
