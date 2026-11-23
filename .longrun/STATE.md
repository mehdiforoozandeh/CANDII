# State

Rewritten, never appended. Cap: 5 KB.

## Goal now
Find and test a fair count-to-p-value route that puts miniCANDI on the EIC board (q90, task t390). Branch implementation/t390-eic-pvalue-entry, draft PR 64, never merged.

## Horizon
Exploratory, open. First measured result delivered 2026-11-22: ladder probe on 3 blind experiments.

## Known (sources in LOG.md and cruxvault/results/t390/ladder_probe/LADDER_PROBE.md)
- Organizers' side is exact: QN by Max's rule equals the MOESM4 truth on every base; the 1 bp MACS2 formula matches our pipeline track (nMSE <= 1.1e-4). 3 of 3.
- The truth's QN used the pseudoreplicate-overlap peak set. With it, our DNase route equals the truth exactly.
- 25 bp overlap counts cost nMSE 0.017 (H3K4me3), 0.255 (H3K27me3), 0.021 (DNase) against the exact formula.
- Our ChIP reprocessing differs from Jacob's before the p-value step (nMSE 0.013, 0.127); cause open.
- Observed-read count route + QN ranks 1-2 of 26 on MOESM4 measures: an oracle bound on conversion cost only.
- An entry must use the normalization of the truth it is scored against (QN rungs rank 14-26 on the SE truth).
- miniCANDI blind predictions do not exist: blind-row build t184 is a PI gate in the recipe run.

## In flight
- Build agent (Opus): probe 2 on the same 3 experiments: QN with the overlap peaks (oracle), QN with peaks reproducible across two read halves (fair), fragment-pileup representation. Output cruxvault/results/t390/ladder_probe2/PROBE2.md; Nibi jobs eicx_*; budget <= 20 CPU-h.
- Read agent (Sonnet): cause of the ChIP reprocessing gap (Max's s04 se vs mehdise caches, run metadata).
- longrun-auditor: post-compaction audit.

## Next
1. Read probe 2; record findings; choose the fair peak rule for the count route.
2. Decide the 48-experiment batch (about 175 CPU-h plus bam2ta for 34; 7 have no BAM). Ask the PI: D4 limits compute while away.
3. Score the rungs against the MOESM2 (original PE) truth too, as D1 asks; probe 1 used the unnormalized SE truth instead. Max's pe cache: /scratch/maxwl/epi_imputation/s04/cache/pe.
4. Fair map for model outputs (training and validation data only, D2); needs blind predictions (t184).

## Open asks to the PI
- Compute: probe 1 used about 82 CPU-h (64 of them a one-off SE-truth setup). Is that within D4's "small probes"? Probe 2 is capped at 20 CPU-h.
At the next milestone: (a) start the blind-row build t184 so miniCANDI can be scored; (b) run the 48-experiment batch.
