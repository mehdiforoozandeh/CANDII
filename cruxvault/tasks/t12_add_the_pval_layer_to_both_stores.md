---
id: t12
type: task
title: add the pval layer to both stores
category: implementation
parent: t3
blocked_by: t10
refs: 
hypothesis_refs: 
status: done
created: "2026-08-19T01:47:55"
updated: "2026-08-19T06:44:38"
---

# t12 — add the pval layer to both stores

Refs:: _(none)_

## Why

+40 GB EIC / +277 GB MERGED, deferred because the Gaussian signal head is off today but will come back

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [the pval layer, both corpora](results/t12/t12_eic_report.json) — pval layer built for BOTH corpora, verified on disk by the orchestrator. EIC /project/def-maxwl/mforooz/CANDI_STORE/eic: pval 34,290,361,927 B = 34.29 GB over 363 columns x 89 biosamples, 0.7791 B/bin = 1.079x t9's 0.722 and 1.011x section 2's 0.771; store now 54,942,948,927 B = 54.94 GB. MERGED: pval 272,684,328,137 B = 272.68 GB over 2667 columns x 361 biosamples, 0.8433 B/bin = 1.168x t9's 0.722 and 1.094x section 2's 0.771; store now 406,058,086,308 B = 406.06 GB. t9's 0.722 B/bin UNDERSHOOTS on both (7.9% EIC, 16.8% MERGED); section 2's 0.771 is the better predictor, and the plan's own guesses (+40 GB EIC, +277 GB MERGED) were closer than a 0.722-based projection. SLURM: EIC array 55548478 (0-88%15) and MERGED array 55550896 (0-360%40), 450/450 COMPLETED confirmed by the orchestrator, all rc=0, MERGED with zero stderr; EIC wall 27m24s per-task 13-258 s mean 51.3 MaxRSS 0.62-7.06 GB; MERGED wall 47m05s per-task 42-309 s mean 91.7 MaxRSS 0.93-8.16 GB - above t9's 7.45 GB, so the 32 G ask was right and t10/t11's 16 G would have been marginal. THE D9 CEILING IS NOT A PROBLEM, WHICH IS THE HEADLINE RESULT OF THIS TASK: EIC 62 of 363 tracks carry a nonzero pval_clip_frac, max 0.003706 on B_SJCRH30/H3K4me3 (449,371 bins); MERGED 550 of 2667 nonzero, max 0.003747 on motor_neuron_grp4_rep1/H3K4me3 (454,275 bins, tied with rep2). NOTHING anywhere reaches 0.01 - both maxima sit about 2.7x under the threshold STORE.md flags as top-of-signal flattening - and the pattern is entirely H3K4me3 and H3K27ac in high-depth cancer lines, exactly where -log10 p above 655.35 is expected. No evaluation weighting extreme p-values is at risk from the codec. MISSING PVAL SOURCE WHERE COUNTS EXIST: EIC 73, all chipseq-control, no signal track affected; MERGED 359 = 358 chipseq-control plus ONE REAL TRACK, B_cell_nonrep/CAGE, which has signal_DSF1_res25 but no signal_BW_res25, so that biosample carries a counts column with no pval column - writer.py::tracks_for_kind skips it silently. control_col = -1 and scale = 100 on all 450 pval files, per D9 and section 4. MANIFESTS REBUILT for both corpora, since build-manifest reads structure off the h5 attrs and a corpus that gains a kind must have it rerun: eic kinds now [counts,peaks,pval], 89 biosamples, 436 tracks, 35 assays, sha256 f68fba1be3797e7629a57670ec91cce03bb619bd23b8309ab8115dd4c3ae8ebc; merged kinds now [counts,peaks,pval], 361 biosamples, 3026 tracks, 47 assays, sha256 7f594547778dc51a431d219ee9db28a681f5891f7a9911fe22bbd6f1c705195f; verify OK on both. Quota /project 15 to 16 TiB of 28 TiB - the whole pval layer cost 307 GB of a 13 TiB headroom, never tight. NO REPO CODE CHANGED, src/candi/store/*.py md5-identical laptop to Fir before and after, so this was flags-only per CLAUDE.md; the orchestrator re-ran the gates anyway: pytest 497 passed, golden 0 ULP params=2,353,634 sd=472362cea987. DECISIONS THE PLAN DID NOT COVER: --overwrite passed but scoped to pval alone, because writer.py::_write_kind checks the per-kind output path and build_biosample never opens a file for a kind it was not given, so counts.h5 and peaks.h5 were neither read nor rewritten while a failed-task resubmit stays idempotent; throttles EIC %15 because t11's 361-task array was still live on the same account and Lustre tree, MERGED %40 with the account to itself reusing t11's proven level; mem 32G and time 2h sized off t9 rather than t10/t11 because pval is the largest kind and the only float32 source; a pure-directory-listing source pre-scan run BEFORE submitting, to find missing-pval tracks and ragged chromosome sets rather than discover them as array failures - zero chrom mismatches in either corpus; biosample lists regenerated then diffed byte-identical against t10's and t11's frozen lists so array indices match across all three builds; manifest, verify and report run as one small SLURM job per corpus rather than on a login node. Evidence in cruxvault/results/t12/.

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
