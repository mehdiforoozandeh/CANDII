---
id: t8
type: task
title: "candi.store reader and data harness: window sampling from the mask, binomial DSF thinning, batch assembly"
category: implementation
parent: t3
blocked_by: t6
refs: 
hypothesis_refs: 
status: done
created: "2026-08-19T01:47:40"
updated: "2026-08-19T02:45:13"
---

# t8 — candi.store reader and data harness: window sampling from the mask, binomial DSF thinning, batch assembly

Refs:: _(none)_

## Why

windows and context length move out of storage into the loader; needs a decided seeding rule for the thinning RNG so eval reproduces

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- reader.py (637), regime.py (436), dataset.py (537) + tests/test_store_{reader,regime,dataset}.py (1033, 76 tests) + configs/regime.eic_smoke.json, merged 627b1b3 on implementation/t6-candi-store. Gates verified by the orchestrator: pytest 494 passed, tools/golden.py check 0 ULP, params=2,353,634 sd=472362cea987 (D21 holds). D3 OO tree CorpusStore/BiosampleStore/TrackView exactly as STORE_PLAN section 5 declares it; counts upcast to int32 per D7, pval decoded to float32 per D9. D14: the regime's declared assay list IS the column order, an absent assay raises naming it, and a permuted list is asserted to actually permute the columns. D12 eligibility delegates to genome.py::eligible_starts. D23 dsf policy discrete {1,2,4,8} with loguniform option. D6 binomial thinning measured over 1e6 draws at c=50: mean/c 0.500051/0.250040/0.125037 against theory 0.5/0.25/0.125, var 12.5182/9.3890/5.4800 against 12.5000/9.3750/5.4688; thin(c,d)<=c asserted over 200k draws. D22 eval determinism is counter-based with a STABLE hash - blake2b(name.utf8, digest_size=8), never the builtin hash() which is salted per process - entropy tuple SeedSequence([run_seed, h(biosample), h(assay), h(chrom), window_start, dsf_milli]); the DSF draw itself is also counter-based, and a subprocess test pins hash invariance under PYTHONHASHSEED. Batch contract: the expected key set is derived at test time from a live CandiKitH5Dataset batch, never a literal - THE PLAN'S SECTION 5 LIST OF 13 KEYS IS INCOMPLETE, the real class emits 18 (the 13 plus x_dsf, y_dsf, control_x_dsf, window_idx, biosample_name) and StoreDataset matches the class. MISSING/CLOZE imported from _vendored.py; the loader emits only MISSING and a test asserts CLOZE never appears. h5py fork safety: a pid-keyed handle pool cleared without closing on a new pid (closing an inherited HDF5 handle damages the parent), with __getstate__/__setstate__ dropping the pool so spawn workers pickle cleanly; tested through a real os.fork, a pickle round trip, and a num_workers=2 DataLoader. TWO KEYS DELIBERATELY CARRY A DIFFERENT MEANING FROM THE OLD BAKE: y_pval is raw -log10 p rather than arcsinh because D9 puts every transform in the model, and region_type is always REGION_TILE=255 because the store tiles and has no cCRE annotation to fabricate. OTHER DECISIONS NOT IN THE PLAN: a track with missing depth/read_length/run_type is emitted as a wholly absent column rather than a partial one, because writing -1 into one meta row makes encoder.py::_prepare_signal raise on the availability disagreement - gaps are collected in ds._gaps and meta_missing='error' refuses instead; StoreDataset requires a manifest because the four meta inputs live in the CSVs per D20; assay_id is the index in the declared list and the control's is len(assays), matching bake.py's control_assay_id=F; the depth meta row moves with the DSF as meta[0] -= log2(d); the control is never thinned; cell_cond is REFUSED rather than defaulted because it needs the prefix split D16 forbids - naming the cell type of A549_nonrep is a separate task; a missing dna.h5/mask.h5 warns once rather than raising so a mid-build store stays loadable, and without a mask every window is eligible, said out loud; window_plan.type supports 'tile' only, overlap being stride_bins < context_bins; train/eval chromosome overlap is an error; chipseq-control appearing in regime.assays is an error per D18.

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
