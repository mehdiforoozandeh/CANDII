---
id: t264
type: task
title: "Onion: Build the second experiments: submit and collect"
category: data-acquisition
parent: t152
blocked_by: t262, t183
refs: 
hypothesis_refs: 
status: done
created: "2026-11-12T03:06:51"
updated: "2026-11-18T23:23:12"
---

# t264 — Onion: Build the second experiments: submit and collect

Refs:: _(none)_

## Why

builds and verifies the second-experiment measurements on Nibi, with one resubmit and one fallback (chunks OXa, OXc)

## Decisions and runs

- Partly done: the second experiments were submitted after 3 smoke gates (OXa, 2026-11-13); 13 rows whose EIC partner is a tail DNase-seq row wait for the dataset update; row 2028 failed on the 12 GB Java heap and was resubmitted (stage 27232990, build 27232991). The collect (OXc) is not done. Notes: results/t264/SUBMITTED.md.

## Close-out note (2026-11-18)

Submitted as chunk OXa (results/t264/SUBMITTED.md); collected by close-out item D2: 71 second experiments built. The 17 unbuilt DNase-seq rows were dropped by PI decision 4 of 2026-11-17.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [Second-experiment submission notes](results/t264/SUBMITTED.md)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
