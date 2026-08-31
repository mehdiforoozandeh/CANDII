---
id: t83
type: task
title: write the prediction track writer so a track is predicted once and scored many times
category: implementation
parent: t77
blocked_by: None
refs: 
hypothesis_refs: 
status: open
created: 2026-08-31T00:52:17
updated: 2026-08-31T00:52:17
---

# t83 — write the prediction track writer so a track is predicted once and scored many times

Refs:: _(none)_

## Why

BENCHMARK_DESIGN 12.8: stream_tracks yields TrackRecords straight into score_track, so nothing persists a prediction; without it the two scopes of 4, the three V_ numbers of 5.2, both truths of 6 and 5's touch-once ruling on B_ each re-run inference

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
