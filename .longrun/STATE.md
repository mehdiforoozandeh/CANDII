# State

Rewritten, never appended. Cap: 5 KB.

## Goal now
Find and test a fair count-to-p-value route that puts miniCANDI on the EIC board (q90, task t390). Exploratory branch implementation/t390-eic-pvalue-entry, draft PR 64, never merged.

## Horizon
Time horizon: exploratory, open (PI away from 2026-11-22). First measured result due: 2026-11-23 (conversion floor on a few tracks with known p-values).

## In flight
- Two read agents (Sonnet): (a) our store, miniCANDI outputs, our EIC scorer, our SE reprocess; (b) Max's sprint 04 scorer and truth, sprint 10 count route, fairness of tcfloor and AG2 recalibration.

## Next
1. Merge both agents' facts into a route map: formula route, learned-map route, output-change route.
2. First measured result: conversion floor (true counts -> p-values vs the actual p-value track) on a handful of tracks, with data already on the clusters.
3. Plan with orchestrate; each chunk cites a ledger row.

## Open asks to the PI
None. D2 and the ledger tiers confirmed by the PI on 2026-11-22.
