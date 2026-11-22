# State

Rewritten, never appended. Cap: 5 KB.

## Goal now
Find and test a fair count-to-p-value route that puts miniCANDI on the EIC board (q90, task t390). Branch implementation/t390-eic-pvalue-entry, draft PR 64, never merged.

## Horizon
Exploratory, open. First measured result due 2026-11-23: the observed-data ladder probe on 3 blind experiments.

## Known (see LOG.md for sources)
- Exact EIC scorer in repo: tools/onion_eval/eic_score.py (raw values, nine measures, reproduces MOESM4).
- Exact normalization rule recovered by Max (s04 apply_bp.py); Jacob's SE record has bigwigs and peaks, no reads.
- Our SE reprocessing of the blind set is on Nibi /project/def-maxwl/mforooz/EIC_REPRO/003_pipeline/results (BAMs for 41, pval bigwigs, peaks).
- Count route code exists: tools/onion_eval/signal_from_counts.py (box spread, plug-in or expectation).
- miniCANDI blind predictions do not exist: blind-row build t184 is a PI gate in the recipe run.

## In flight
- Build agent (Opus): ladder probe L1-L7 on 3 blind experiments; output cruxvault/results/t390/ladder_probe/LADDER_PROBE.md; Nibi jobs eicx_*, outputs /scratch/mforooz/eicx/.

## Next
1. Read the probe; record findings under q90; decide the full 48-experiment batch.
2. Design the fair map from the count route to the corrected scale (training and validation data only, D2).
3. Representation cost (overlap counts, fragment pileup, 1 bp starts) for ledger R6.

## Open asks to the PI
None yet. Expected at the next milestone: whether to start the blind-row build so miniCANDI can be scored (it belongs to the recipe run).
