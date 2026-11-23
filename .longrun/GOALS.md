# Requirements ledger

One row per requirement in the PI's documents. Tier: P0 the run fails without it, P1 expected, P2 if time allows, P3 do not build.
Status: open, partial, done, dropped (dropped needs a PI decision id in Evidence). Done needs a measured result linked in Evidence.
Provenance: PI doc, PI, agent proposal approved.

| ID | Tier | Requirement (quoted) | Source | Provenance | Status | Evidence |
|---|---|---|---|---|---|---|
| R1 | P0 | "figure out a fair and tested way to compare minicandi's outputs in count space against challenge entrants (in p-value space)" | cruxvault/INSTRUCTIONS/eic_entry/goal.md:12 | PI doc | open | |
| R2 | P0 | "exploring how to make our mini-candi enter the EIC entrants ranking. fairly and correctly." | cruxvault/INSTRUCTIONS/eic_entry/goal.md:22 | PI doc | open | |
| R3 | P0 | "if we are sure that we have the 'count to signal' function/pipeline figured out exactly, then comparing minicandi against challenge entrants will become fair" (the conversion is tested for exactness) | cruxvault/INSTRUCTIONS/eic_entry/goal.md:10 | PI doc | partial | organizers-side transforms exact on 3 of 3 (QN 0 bases differ; formula nMSE <=1.1e-4); conversion floor of the 25 bp count route measured: nMSE 0.017/0.255/0.021 (cruxvault/results/t390/ladder_probe/LADDER_PROBE.md) |
| R4 | P1 | "figure out the formula by trying to reproduce eic pvalue tracks from their bam" | cruxvault/INSTRUCTIONS/eic_entry/goal.md:10 | PI doc | partial | DNase reproduced byte-identically; ChIP differs from Jacob before the p-value step, nMSE 0.013/0.127, cause open (cruxvault/results/t390/ladder_probe/LADDER_PROBE.md) |
| R5 | P1 | "the solution for this problem can be in finding the 'pipeline' recipe that works" | cruxvault/INSTRUCTIONS/eic_entry/goal.md:16 | PI doc | partial | recipe = MACS2 formula from 1 bp starts + QN with the pseudoreplicate-overlap peak set; exact for DNase (cruxvault/results/t390/ladder_probe/LADDER_PROBE.md) |
| R6 | P1 | "or we might need to modify how miniCANDI works (e.g. the resolution/binning/strand-aware fragment pileup, or 1 bp read starts, etc)" (measure what each output representation costs) | cruxvault/INSTRUCTIONS/eic_entry/goal.md:16 | PI doc | partial | 25 bp overlap counts vs 1 bp starts measured on 3 experiments; fragment pileup not measured (cruxvault/results/t390/ladder_probe/LADDER_PROBE.md) |
| R7 | P1 | "you are allowed to explore different options" | cruxvault/INSTRUCTIONS/eic_entry/goal.md:18 | PI doc | partial | 8 ladder rungs tried (pipeline, 1 bp formula, overlap counts, QN with 3 peak sets) (cruxvault/results/t390/ladder_probe/LADDER_PROBE.md); probe 2 adds fragment pileup and split-half peaks |
| R8 | P1 | "use max's repo and results and files and findings to guide you" | cruxvault/INSTRUCTIONS/eic_entry/goal.md:14 | PI doc | partial | ported Max s04 apply_bp.py and s10 phys1.py; both reproduce (cruxvault/results/t390/ladder_probe/LADDER_PROBE.md) |
| R9 | P0 | "done in a separate branch ... don't merge this new branch" (constraint) | cruxvault/INSTRUCTIONS/eic_entry/goal.md:20,24 | PI doc | partial | branch implementation/t390-eic-pvalue-entry, draft PR 64 marked do-not-merge |
| R10 | P0 | "we have to compare on signal p value space. correct?" (score on the challenge's own metrics and truth) | cruxvault/INSTRUCTIONS/eic_entry/goal.md:29 | PI doc | partial | scorer run on qn and se truths, 9 measures, placed among 25 entrants; oracle rungs only, no model yet (cruxvault/results/t390/ladder_probe/LADDER_PROBE.md) |
