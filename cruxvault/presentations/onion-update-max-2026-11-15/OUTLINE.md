# Approved outline (PI, 2026-11-15): progress report for Max

Frame: the PI's file `INSTRUCTIONS/peeling_the_onion/peeling the onion.md` is the question. This report is the answer so far (2026-11-10 to 2026-11-15). Use the section names and vocabulary of that file.

PI additions on approval:
- Each section names the paths on Nibi (and Fir where a copy exists) where Max can look at or use the material.
- The eval section uses the law of total variance formulation, if it helps.
- The eval section explains the difference between the offset-on and the offset-off formulation (mean = depth x rate, i.e. eta space), and says whether offset-on gives better sensitivity to sources of variance other than depth-related Poisson noise (sampling noise, pseudoreplicate noise).
- Not verbose. Lists, figures and schematics.

## 1. Sources of variation: what we can now separate
1. Data-processing variation is built into the data: 8 kinds of processing view per EIC experiment (single-end, read crops, depth thinning, no duplicate removal, no mapping-quality filter).
2. Pseudoreplicates exist for every view; they give the Poisson-error component.
3. Data-generation variation: only the isogenic replicates that the EIC experiments carry. Lab, platform, antigen are not separated.
4. Biological variation: cell-type-specific, position-specific, assay-specific are measured. Population-level is not addressed.

## 2. Objectives: one metric per goal
5. Biological variation (imputation across Cell, Assay, Position): share of cell-type-specific variance recovered above the pseudoreplicate floor.
6. Artifact variation (covariate set A to covariate set B): gap closed on never-trained view pairs, between depth-only scaling and the pseudoreplicate floor.
7. Uncertainty: a calibration term for the predicted NB.
8. One composite combines the three.

## 3. Eval: the onion, peeled on real data
9. Layers of variance per assay at 25 bp (law of total variance): sampling layer largest (76% to 97%), replicate layer above it small (0.1% to 6%), cell-type layer the rest.
9b. Offset on versus offset off (eta space) and sensitivity to non-sampling sources.
10. Frozen benchmark of 256 regions, 16-region screening set. Other methods on the same set: 8 baselines and AlphaGenome 2 (0.438 on its fair set of 11 tracks, against 0.333 for the pseudoreplicate floor).
11. Open: the resolution question, the breakdown by artifact source, the full breakdown by genome subset.

## 4. Data: the recap list of the PI's file
12. "Updating our EIC dataset": done. Rebuilt from reads, hard-checked by accession, with the measurement dimension: 2,740 measurements from 249 of 267 training tracks, 42 of 45 validation tracks, 37 of 40 controls.
13. "Retiring EED": done; not used.
14. "Collecting a new extended dataset": catalog and census only. Balanced split along Cell, Assay, Measurement not started.

## 5. miniCANDI: the prototype
15. Built as specified: inputs X, S, C, C'; output NB parameters; NB loss on masked assays (imputation) and on available assays under another covariate set (correction).
16. Architecture ablations: 6 architectures at about 1.0 M parameters, 3 seeds. Three tiers hold: the two conv models; then axial transformer, U-Net, pointwise MLP; then Perceiver IO. Provisional: no architecture was tuned; a per-architecture tuning round is in work.
17. Covariates C and C': scrambling them at the U-Net lowered cell-type-specific variance recovered by 0.029; it raised gap closed on all pairs, so the correction metric is under review.
18. Depth: the model needs the input depth. Hiding it lowered the composite from 0.444 to 0.260; the loss is on the input side.
19. Training regions: no selection rule beat uniform sampling.
20. Coverage of the named ablations: 10 of 38 fully tested, 7 partial, 21 untested (two-step CAC, average baseline as input, DNA embedding options, chip control on and off, loss variants, depth-controlled input).

## 6. Not started: scaling (data, context, model size)

## Figures and schematics
(1) the onion: sources of variation and which data view exposes each; (2) the measurement dimension: views and pseudoreplicates of one experiment; (3) the three objectives and their metrics; (4) variance layers per assay; (5) baselines and AlphaGenome 2 on the benchmark; (6) miniCANDI inputs and outputs; (7) the architecture ladder as tiers with seed ranges; (8) covariate and depth ablations; (9) coverage of the named ablations.
