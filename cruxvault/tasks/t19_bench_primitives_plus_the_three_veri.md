---
id: t19
type: task
title: "bench primitives plus the three verification layers: vendored-reference bit-match, analytic arrays, stub model"
category: implementation
parent: t17
blocked_by: t18
refs: 
hypothesis_refs: 
status: done
created: "2026-08-19T23:52:14"
updated: "2026-08-20T23:54:13"
---

# t19 — bench primitives plus the three verification layers: vendored-reference bit-match, analytic arrays, stub model

Refs:: _(none)_

## Why

every number in the suite is unquotable until its primitive is verified against a known answer

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [The E, P, D and B blocks, verified two ways — commit 930b425](results/t19/DELIVERABLE.md) — eic.py, partitions.py, distributional.py, binary.py plus the frozen vendored score_metrics.py fixture. Layer 1 measured 0 ULP against the organizers' own code, not the 1e-12 relative the plan asked for, on random, adversarial and real arrays; layer 2 is 42 analytic tests whose expected value is derived on paper in the docstring. Seven behaviours of score_metrics.py are reproduced ON PURPOSE under D16 and each is pinned by a test: overlapping annotations double-count; a promoter running off the start of the array is silently dropped because the negative left index resolves from the end; end is one bin past the true end; msevar returns a bare 0.0 sentinel; normalize_dict is a no-op so the published numbers are untransformed. TWO WERE FOUND BY LAYER 1 ITSELF rather than by reading the source: mseprom counts the CLIPPED slice while msegene and mseenh count end-start UNCLIPPED, so a gene running past the end of the scored array pays for bins it contributes no error over; and an empty result fails two different ways, nan when the annotation set selects zero bins and ZeroDivisionError when it matches no line at all, decided by whether the loop body ran and promoted sse to numpy. Also established: the bed in the challenge's annot/hg38 is NOT the ENCODE Exclusion list (38 regions, 17,040 bp against the real v2's 636 regions and 227,162,400 bp) and never runs anyway, because bw_to_dict returns load_npy() before the blacklist branch — so the E-block applies no blacklist to signal, which is D6. 113 t19 tests; re-verified at 6470a6a with 688 passing and golden 0 ULP.

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
