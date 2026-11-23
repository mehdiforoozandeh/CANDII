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
- Our ChIP reprocessing differs from Jacob's before the p-value step (nMSE 0.013, 0.127). Likely cause: control reads (treatment matches; gap sits where control sets lambda). Not closable exactly without his control list (results/t390/chip_gap/CHIP_GAP.md).
- Observed-read count route + QN ranks 1-2 of 26 on MOESM4 measures: an oracle bound on conversion cost only.
- An entry must use the normalization of the truth it is scored against (QN rungs rank 14-26 on the SE truth).
- miniCANDI blind predictions do not exist: blind-row build t184 is a PI gate in the recipe run.

## In flight
- Probe 2 DONE (results/t390/ladder_probe2/PROBE2.md): fragment pileup nMSE 0.0021/0.062/0.00003; half-read peaks within 8.5% of truth peaks.

## Next
0. t184 batch 1 submitted on Nibi 2026-11-23 08:01 UTC by the recipe session (PI-approved there): ChIP/ATAC 28629674-28629678, DNase 28629699-28629701; outputs /scratch/mforooz/onion/{work,measurements}; limits 3 d / 7 d. Never resubmit (not ours). Blind predictions follow the final architecture at original covariates; recipe session sends paths.
1. Read probe 2; record findings; choose the fair peak rule for the count route.
2. 48-experiment batch: held by the PI (D8).
3. Score the rungs against the MOESM2 (original PE) truth too, as D1 asks; probe 1 used the unnormalized SE truth instead. Max's pe cache: /scratch/maxwl/epi_imputation/s04/cache/pe.
4. Fair map for model outputs (training and validation data only, D2); needs blind predictions (t184).

## PI request
Ping the PI (push notification) when probe 2, MOESM2 or miniCANDI scores land. Foreman session local_b5c55e32-4e2a-4edc-b4b7-6f0134e8bc43 ("Open sessions on project", 2026-11-23): send it every PI-facing item and a 2-3 line status at each milestone. Overnight 2026-11-23 (relayed by foreman): compute by workers' judgment, report each job id to the foreman; 48-experiment batch and Jacob message stay held (D8); do not stop for questions, take the safe default and report it.

## Open asks to the PI
- Copy the 51 original-truth bigWigs (~70 GB) to /project/def-maxwl/mforooz/ONION/eic/original_pe? Asked via foreman 2026-11-23; default meanwhile: no copy (/project 4264/4657 GiB).
- Change miniCANDI output to a 25 bp fragment pileup for the EIC entry? Asked via foreman 2026-11-23; default meanwhile: score counts as they are.
