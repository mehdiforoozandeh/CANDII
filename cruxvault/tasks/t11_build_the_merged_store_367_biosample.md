---
id: t11
type: task
title: "build the MERGED store: 367 biosamples, counts + peaks"
category: implementation
parent: t3
blocked_by: t10
refs: 
hypothesis_refs: 
status: done
created: "2026-08-19T01:47:55"
updated: "2026-08-19T05:45:51"
---

# t11 — build the MERGED store: 367 biosamples, counts + peaks

Refs:: _(none)_

## Why

~81 GB, 3045 tracks; names kept verbatim, train/test split deliberately deferred

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [the MERGED store — 361 biosamples, 133.37 GB](results/t11/t11_report.json) — MERGED store built at /project/def-maxwl/mforooz/CANDI_STORE/merged, counts+peaks only (pval is t12, D24). TRUE COUNTS: 361 biosamples, not the plan's 367 - that was a raw ls sweeping in hg38.fa/.gz, hg38_blacklist_v2.bed/.gz, navigation.json and aliases.json; 3045 source track dirs, 3026 columns stored, 47 assays, 23 chroms, 121,241,684 bins/track. SLURM array 55545024 (--array=0-33,35-68,70-147,149-360%40, 358 tasks) plus smoke array 55541898 (3 tasks): 361/361 COMPLETED 0:0, ZERO FAILURES, verified by the orchestrator; 69 min wall end to end, 52-407 s per task (median 90, p95 168), MaxRSS median 1.17 GB max 2.50 GB, 10.13 task-hours. SIZE verified on disk by the orchestrator at 133,373,691,619 B = 133.37 GB / 124 GiB in 722 files: counts 128.94 GB at 0.3515 B/value, peaks 4.43 GB at 0.0137 B/value. PROJECTIONS MADE BEFOREHAND and their ratios to actual: the plan's 81 GB headline 1.65; the plan's own B/bin rebuilt on the real column sets 103.4 GB 1.29; t9's measured B/value 112.9 GB 1.18; a 3-task smoke's per-track mean 105.7 GB 1.26. Same pattern t9 found for EIC - the headline omits control columns and uint32 does the rest. D7: 57 of 361 biosamples on uint32 (15.8%), carrying 497 of 3026 counts columns (16.4%). build-manifest ran STRICT and did not raise: 361 biosamples, 3026 tracks, 47 assays, 6 metadata gaps; sha256 154ac939dc3cd668ae93b45292dc03154e907031509eb981c6aae3d98bc76d8b, 1,902,048 B. verify OK in 5.4 s. NULL METADATA: 6 tracks, all chipseq-control, all missing only sequencing_platform - D19, correct, not a bug. t5's other three fully-empty rows never appear because the writer never stored those tracks. NEW FINDING BEYOND t5: a pre-flight scan found 16 MORE empty-shell track dirs, ALL RNA-seq (GM23248 and GM23338 grp1 rep1+rep2, adrenal_gland_nonrep, cardiac_muscle_cell, foreskin_keratinocyte, hepatocyte, neural_progenitor_cell, smooth_muscle_cell rep1+rep2, upper_lobe_of_left_lung_nonrep) on top of t5's three control shells - 19 total, which is exactly 3045-3026; t5 only looked at controls. writer.py::SourceLayout.tracks_for_kind SKIPS all of them rather than raising, because it filters to tracks whose kind dir exists AND holds npz - no flags, no workaround, no code change needed. D20 cross-check dry-run over all 3045 track dirs BEFORE the manifest: 0 conflicts, 0 multi-replicate-key raises, 3 absent file_metadata.json; t5's warning that control and signal JSON schemas differ did not break read_file_metadata because the four _CROSS_CHECK fields are shaped alike in both. Pre-flight also confirmed ZERO ragged chromosome sets - all 361 biosamples have all 23 chroms for all 3 kinds - which is why writer.py::_resolve_chroms never fired and no task failed. SPOT ROUND-TRIP, since verify is structural only: K562_nonrep (uint32), H1_nonrep (uint16, 28 tracks), SJSA1_grp1_rep1 (no control), chr1 and chr21, counts and peaks all BIT-EXACT after the D13 truncation. B_cell_nonrep/CAGE is the only MERGED track with counts but no peaks; the manifest records 2667 counts+peaks tracks and 359 counts-only (358 controls plus CAGE). D16 honoured, names verbatim; D17 honoured, merged_train_va_test_split.json untouched. QUOTA /project 15/28 TiB before and after, inodes 1667K to 1670K; 133.4 GB is 1.0% of the 13 TiB free, never tight. DECISIONS NOT IN THE PLAN: a deliberate 3-task smoke (widest, empty-control, counts/peaks-mismatched biosamples) before the 358-task array, then resources retuned off its real numbers to 16G and 1h rather than t9's 32G/2h, because t9's 7.45 GB peak was the pval float32 block this build does not write; the 3 smoke indices excluded from the main array rather than rebuilt; biosample order frozen as sorted() in t11/biosamples.txt with the array indexing into it by line, so the build is deterministic and re-runnable; the source tree pre-flighted for ragged chroms and the D20 cross-check dry-run, both to avoid discovering a hard failure after a long build; --overwrite on every task for requeue idempotence, which with the writer's .tmp plus os.replace means a killed task leaves no half-file; throttle %40 chosen because each task reads its counts npz tree twice (the D7 dtype pre-scan then the write), so 361 concurrent would put ~6 TB of Lustre reads in flight on a shared filesystem. NO REPO CODE CHANGED - Fir's src/candi/store/*.py md5s verified identical to the local tree before submitting, and the orchestrator confirmed a clean working tree and 497 tests passing afterwards. Evidence in cruxvault/results/t11/; Fir run dir /project/def-maxwl/mforooz/CANDI_STORE/t11/.

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
