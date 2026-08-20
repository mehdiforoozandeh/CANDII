---
id: t23
type: task
title: teach train.py to open a CANDI_STORE, not only a baked h5
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-08-19T23:57:04"
updated: "2026-08-20T00:26:33"
---

# t23 — teach train.py to open a CANDI_STORE, not only a baked h5

Refs:: _(none)_

## Why

train.py takes --h5 and reads ds.num_cells, ds.num_assays, ds.context_bins and ds.resolution off CandiKitH5Dataset, so a store cannot be trained from at all today - the store's whole point is unreachable from the training entrypoint. StoreDataset already emits the training-path batch dict bit-identically (0 differing elements of 143.2 M against the old bake) and loads 4.80x faster, so this is a plumbing gap, not a data question. Scoped to training; scoring a store-backed imputation run still needs t14.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [train.py can open a store — commit f5cafac](results/t23/DELIVERABLE.md) — train.py now takes --store <regime.json> (alias --regime-file) as a required-mutually-exclusive alternative to --h5, merged f5cafac on main. One flag, not two, because Regime.from_dict already requires a store key - a separate --store-root could only duplicate or contradict the authority file, which is the bug class the store exists to remove. The pre-existing --regime flag means the MASKING regime (type1/type2_loci); both help texts now name the other and a test asserts the cross-reference. All three CandiKitH5Dataset construction sites route through one factory, make_dataset, rather than a third divergent branch. THE OLD PATH IS PROVEN UNCHANGED TWO WAYS: golden 0 ULP, and a test drives the pre-t23 constructor spelling against the factory and asserts the batch streams are bit-identical across all 18 keys - the half a golden cannot see. The store path is training-only BY CONSTRUCTION: cell_cond (D16), type2_loci and --reference raise rather than degrade, and evaluate() is never called, so the run json carries no M1/M2/M3/S14 keys at all rather than metrics pooled over zero targets, which in a json reads exactly like a finished evaluation - t14 made structural instead of a warning. Provenance per STORE_PLAN section 4: the store path records the regime file verbatim, its sha256 and the store manifest sha256. VERIFIED INDEPENDENTLY BY THE ORCHESTRATOR, not just by the author: argparse refuses both flags and refuses neither with the right messages; 120 real optimizer steps through python -m candi.train off a synthetic store, nll 116.442 -> 21.452; regime_sha256 and store_manifest_sha256 in the run json both match a fresh recompute; regime_json round-trips through Regime.from_dict; zero M1/M2/M3/S14 keys present. Gates: 522 passed (497 baseline + 25 new), golden 0 ULP params=2,353,634 sd=472362cea987. NOTE ON THE BRANCH: implementation/t23-train-from-store was created for this, but a concurrent session working in the same worktree committed its own t17-t19 evaluation-suite work onto it and swept tests/test_train_store.py into one of those commits. The change was therefore rebuilt as t23-clean off origin/main with only t23's eight files and fast-forwarded to main, so no part of the other session's unreviewed work landed here. STILL OPEN: scoring a store-backed imputation run needs t14 (StoreDataset does not emit y_data_imp, y_pval_imp, y_peaks_imp, y_meta_imp, imp_biosample_name, log_ref).

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
