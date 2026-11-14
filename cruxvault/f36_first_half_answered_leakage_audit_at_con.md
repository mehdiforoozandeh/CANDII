---
id: f36
type: finding
schema: 3
title: First half answered (leakage audit at condition-and-channel level, Nibi job…
parent: q20
confidence: moderate
evidence: results/t273/AG2_LEAKAGE.md, results/t308/SUBMITTED.md
created: "2026-11-14T02:17:59"
updated: "2026-11-14T02:17:59"
---

# f36 — First half answered (leakage audit at condition-and-channel level, Nibi job…

Parent:: [[q20_which_eic_tracks_did_alphagenome_2_train]]

## Answer

First half answered (leakage audit at condition-and-channel level, Nibi job 27085861, byte-identical on the laptop): AlphaGenome 2 trained on 36 of 51 EIC blind tracks and 32 of 45 validation tracks. The fair set (unseen by both AG2 and miniCANDI) is 15 blind tracks in 11 cells, all H3K9me3 (9) or H3K27me3 (6), and 12 validation tracks in 10 cells; every validation and blind track is queryable. On the frozen dataset eic_v1 the validation fair set is 11 tracks in 9 cells (31 trained-on tracks), because DNase-seq tracks C50M02 and C34M02 are absent. The audit cannot see accessions or replicates (AG2 publishes no accession table) and relies on the API's observed_metadata listing of the all-folds model. The second half (AG2 against the baselines) is not scored yet.
