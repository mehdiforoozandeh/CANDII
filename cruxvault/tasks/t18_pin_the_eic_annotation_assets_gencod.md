---
id: t18
type: task
title: "pin the EIC annotation assets: GENCODE v29 genes bed, FANTOM5 hg38 permissive enhancers, msevar variance pools"
category: data-acquisition
parent: t17
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-08-19T23:52:14"
updated: "2026-08-21T04:19:56"
---

# t18 — pin the EIC annotation assets: GENCODE v29 genes bed, FANTOM5 hg38 permissive enhancers, msevar variance pools

Refs:: _(none)_

## Why

the E-block cannot compute mseprom/msegene/mseenh/msevar without them; mirrors how t4 pinned the blacklist

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [The EIC scoring assets, and D7's variance pools — commits 2b6e340 + 3cff603](results/t18/DELIVERABLE.md) — the beds were already pinned; this closes the pools. Built on Fir by slurm/t18_varpool.sh (job 55875262, 5.6 min): the eic store's T_ split is 51 biosamples and **267 pval tracks against the challenge's 267 training experiments — exactly**, so D7's 'as close as the store allows' turns out to be no gap at all. chr21+chr22+chrX, 780 MB, staying on the cluster; REPORT.md and membership.json rsync'd down. 30 of 35 assays have a usable pool; five are carried by one training biosample and are skipped, because a one-member pool is zero everywhere and msevar divides by var.sum(). ATAC-seq has the smallest usable pool at 4 members — worth remembering before quoting an msevar for it. THE BUILDER HAD NEVER BEEN RUN: it called pval(chrom) and every reader accessor takes start as a required argument, so its first real invocation would have died on a TypeError after queueing against a 435 GB store. It had no test. Four now, including a bit-exact check that the block size cannot move a variance (variance is per bin, so a block boundary changes which read returned a column, never the column) and a check of the population form, std**2 with ddof=0 — at n=2 the sample variance is twice that, so getting it wrong doubles every msevar and still looks plausible. Blocking cuts a whole-chr1 pool from 5.1 GB held at once to 408 MB. NOT DONE: msevar is still absent from the t22 bench run, and switching it on is not only a flag — the bake's chr21 is 1,867,776 bins against the corpus store's 1,868,399, and run_bench compares the two and refuses rather than weighting the wrong positions. Gates: 713 pass, 1 skipped, golden 0 ULP.

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
