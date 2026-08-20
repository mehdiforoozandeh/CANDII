---
id: t10
type: task
title: "build the EIC store: 91 biosamples, counts + peaks + dna"
category: implementation
parent: t3
blocked_by: t9
refs: 
hypothesis_refs: 
status: done
created: "2026-08-19T01:47:55"
updated: "2026-08-19T05:07:31"
---

# t10 — build the EIC store: 91 biosamples, counts + peaks + dna

Refs:: _(none)_

## Why

~12 GB, one SLURM array task per biosample; the first store a real training run can use

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [the EIC store — 89 biosamples, 20.65 GB](results/t10/t10_report.json) — EIC store built at /project/def-maxwl/mforooz/CANDI_STORE/eic: 89 biosamples, 436 tracks (436 counts cols, 363 peaks cols, 35 assays), 20,652,577,531 B = 20.65 GB (counts 19.75, peaks 0.90) verified on disk by the orchestrator, which is 1.264x t9's 16.3 GB projection - the overshoot is counts, because t9's 0.295 B/bin came from a slice averaging 8.4 tracks/biosample while the corpus averages 4.9 and narrower 2-D blocks compress worse per value. SLURM array 55545268, throttled 0-88%30, 89/89 COMPLETED rc=0 with zero stderr, verified by the orchestrator; wall clock 20m41s, per-task 16-310 s (mean 62.5), MaxRSS 0.54-2.67 GB - far under t9's 7.45 GB because counts+peaks drops pval, so t11/t12 can safely run at 16 GB. 22 of 89 biosamples landed on uint32 (25%). build-manifest and verify both OK; manifest sha256 849d29af16b656b846bec0dc725db7d6d74acfe37093093f21fd81418c07e869. PREREQUISITE THE PLAN DID NOT ANTICIPATE, discovered by t9 and closed here: eic_control_metadata.csv at /project/def-maxwl/mforooz/CANDI_STORE/, sha256 b386f2c9b0b2c10665b33d30799299996464c4d3db3f0dfa2f51742ee80d840a, 11,167 B, 76 rows, header byte-identical to eic_metadata.csv. Without it, 76 of 439 EIC counts columns would have trained as all-MISSING. Depth equality re-checked on EIC and it transfers: 363/363 signal CSV rows match signal_DSF1_res25/metadata.json exactly, 0 mismatches (t5's MERGED check was 58/58 on a sample; this is the full corpus). Per-field gaps of 76: sequencing_platform 4, everything else 3, biosample_name and assay_name 0. THE GAP IS CLOSED: of 439 source tracks, 436 are in the store and exactly ONE field on ONE track is null - V_testis/chipseq-control.platform, whose JSON has no sequencing_platform key. LOUD FAILURES FOUND: 3 empty-shell control dirs (B_Caco-2, B_SJSA1, T_SJSA1) with no file_metadata.json, no DSF1 metadata.json and no npz, which is why the store has 436 not 439 counts columns - writer.py::tracks_for_kind already skips them because it requires an npz; 0 multi-key and 0 zero-key replicate dicts across all 73 real control JSONs (all carry exactly one key, '2', derived per file and never hardcoded); 4 biosample_name disagreements between control JSON and directory name (T_BMEC, T_HAP-1, T_MG63, T_SJCRH30) where the directory name wins per D16, reported not resolved - MERGED had zero of these. EIC control file_metadata.json schema is identical to MERGED's. GENUINE CODE BUG FOUND AND FIXED, merged 174ce3f: manifest.py::read_file_metadata treated EVERY dict-valued field as replicate-keyed and raised on any with more than one key, unconditionally and past --no-strict. DATA_CANDI_EIC/T_testis/H3K9me3 carries a free-text notes dict keyed alternative_bigbed_used / original_bigbed / alternative_reason, so one file's provenance note would have KILLED build-manifest FOR THE WHOLE CORPUS - and would kill t11 if MERGED has the same quirk. The refusal now covers only the four fields the cross-check consumes; any other dict-valued field passes through unflattened, since picking a key would be a coin flip in the manifest and raising would be fatal over a field nobody reads. D19 intact. Regression test names the real file. CORRECTIONS TO THE PLAN: EIC has 89 biosample DIRECTORIES and 439 track dirs, not the 91 in section 2, which counts aliases.json and navigation.json - the same raw-ls error that made MERGED 367 instead of 361; there is also a hidden .candi_kit_state/ that a raw listdir sees as a 90th dir, correctly excluded by discover_biosamples. MEASURED SOURCE SIZES (new, not in the plan): DATA_CANDI_EIC 237 GB, DATA_CANDI_MERGED 999 GB, per track EIC 540 MB and MERGED 328 MB across all kinds and all DSF levels. DECISIONS NOT IN THE PLAN: named the file eic_control_metadata.csv because t5's MERGED file already owns control_metadata.csv at the same level; dot-directories are not biosamples; lineterminator newline for true byte-identity with eic_metadata.csv - t5's file is CRLF because csv.DictWriter defaults to it, so t5's 'byte-identical' claim covers the column names only; on a biosample_name conflict the directory name wins per D16; array throttled %30 because a 361-task MERGED array was already queued on the same account and the same Lustre tree; walltime and memory sized off t9 rather than bake.sh, with --overwrite so a failed-task resubmit is idempotent; manifest, verify and report run as one small SLURM job (55547681) rather than on the login node. The agent also edited STORE_PLAN.md to add the measured source sizes; the orchestrator REVERTED that - the plan is the PI's approved record of a decision session, and amending it while leaving the adjacent wrong 91/367 counts would make it self-contradictory, so the corrections live here in the vault instead. Gates verified by the orchestrator: pytest 497 passed, golden 0 ULP params=2,353,634 sd=472362cea987. Evidence in cruxvault/results/t10/.

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
