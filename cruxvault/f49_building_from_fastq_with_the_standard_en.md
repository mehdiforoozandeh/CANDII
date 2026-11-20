---
id: f49
type: finding
schema: 3
title: Building from FASTQ with the standard ENCODE pipelines hit four…
parent: q17
confidence: moderate
evidence: results/t183/EARLY_COLLECT.md, results/t183/TAIL.md, results/t183/COLLECT.md
created: "2026-11-14T02:19:21"
updated: "2026-11-14T02:19:21"
---

# f49 — Building from FASTQ with the standard ENCODE pipelines hit four…

Parent:: [[q17_what_is_the_dataset_and_what_did_it_cost]]

## Answer

Building from FASTQ with the standard ENCODE pipelines hit four library-level problems. (1) The DNase-seq pipeline's paired-end alignment (bwa sampe) exceeded 1 day on many libraries, with no relation to file size (a 7.4 GB row still in sampe while a 31.7 GB row was past it), so every DNase-seq row now gets a 3-day limit. (2) Its duplicate-marking step runs with a 12 GB Java heap by default and failed with 'GC overhead limit exceeded' on some large libraries (at 511 M, 753 M and 1,027 M reads, per the run Log and the notepad's Key facts); a request of 192000M switches the pipeline's own input to a 58 GB heap. (3) Broad paired-end FASTQs whose read names carry /1 and /2 before a comment break mate pairing in the ChIP-seq pipeline (27 rows); the build strips the suffix in a temporary copy. (4) One ENCODE FASTQ is truncated at the source (ENCFF033XNT; a re-fetch gives the same bytes); 16,647,282 of 27,423,113 pairs (0.607) are used and the provenance records it.
