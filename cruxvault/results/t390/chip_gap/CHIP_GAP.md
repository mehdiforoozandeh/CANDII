# Cause of the ChIP reprocessing gap (t390), read-only investigation, 2026-11-23

Question: why our SE ChIP tracks differ from Jacob Schreiber's SE tracks (nMSE 0.013 C19M22 H3K4me3, 0.127 C40M17 H3K27me3) while DNase C12M02 is byte-identical. Computed in memory on chr20, Nibi; nothing written on the cluster.

Likely cause: the control reads differ. Not proven, because Jacob's control list is not published.

Evidence:
- Treatment reads match. 25 bp bins are exactly equal in 46% (C19M22) and 36% (C40M17) of bins, in every value range including > 50; median Jacob/ours ratio 1.000 in every value band. A treatment, depth or fragment-length difference would break the peaks.
- The differing bins are the bins where a local control window sets MACS2's lambda. There ours is higher: median ratio 1.23; mean ours/Jacob 1.066 (C19M22), 1.122 (C40M17).
- Pipeline-setting variants of the 1 bp formula do not close it: control scale x1.05-1.2 gives best nMSE 0.0119 / 0.1035 vs 0.0136 / 0.1166; dropping the d, 1 kb or 10 kb window is worse. Our configuration gives the most equal bins (0.426, 0.325).
- Subsampling our control toward treatment depth brings the mean ratio to 1.016 (C19M22, 1.2x treatment) and 1.013 (C40M17, 1.2x); extrapolation reaches 1.00 near 30-36 M reads. A shallower or noisier control in Jacob's run fits.
- Our run: chip-seq-pipeline v2.2.2, --fraglen 200, ctl_depth_limit 0, exp_ctl_depth_ratio_limit 0.0, ctl_subsample_reads 0; full controls 107.0 M (C19M22) and 150.3 M (C40M17, pooled ENCFF337JNL + ENCFF256QZO) vs 30 M treatment.
- DNase has no control and matches exactly.

Max's prior work: sprint 04 notes/01_handoff.md thread D (ours vs Jacob r 0.83-0.90 raw, ours ~2x higher mean for broad marks, unresolved); eic_correction_history.md item 10 (control pairing unknown; blind.tsv never supplied; our controls rebuilt from today's ENCODE controlled_by links).

Not closable exactly without Jacob's control list. Cheap tests left: rebuild pooled controls as single fastqs or single replicates; test a ~30-36 M control subsample. An exact hit shows nMSE ~1e-4.
