# T83_PREDICTION_WRITER.md — the prediction track writer

**Status: PROPOSAL, awaiting PI approval.** Written 2026-08-31 against
`plan/BENCHMARK_DESIGN.md` §12.8, which names this gap and puts it on the critical path.
Nothing here is approved. §4 carries two measured findings that contradict numbers in
BENCHMARK_DESIGN §12.3 and §12.6, and one collision between two of its approved rulings;
those are the parts that need a decision before code is worth writing.

---

## 1. The gap, stated exactly

`stream_tracks` (`src/candi/bench/harness.py:800`) yields `TrackRecord`s straight into
`score_track` at `harness.py:1481`. Nothing between them writes to disk. Confirmed by
inspection: the only `--out` on the bench CLI is `cli.py:131`, the results JSON.

So **there is no code that persists a prediction.** Every claim in BENCHMARK_DESIGN that
rests on scoring one prediction set more than once currently re-runs inference instead:

| the claim | where | what it costs without a writer |
|---|---|---|
| two aggregations, `held-out` and `genome-wide`, from one pass | §4 | already satisfied — both come from one `stream_tracks` call in the same process (`harness.py:1481-1484`) |
| the three `V_` numbers | §5.2 | a second scoring pass, so a second inference pass |
| both truths | §6 | a third |
| **`B_` is predicted once** | §5 | **the discipline is unenforceable** — nothing stops a re-predict, and nothing records that one happened |

The first row matters: §4's split already works today, in-process. What does *not* work is
anything that needs the predictions **after the process exits**. That is §5.2, §6, and above
all §5's touch-once ruling on `B_`, which is a claim about history and cannot be honoured by
a design that keeps no history.

At CANDI's measured 0.2363 GPU-h per genome-wide track (§12.7), a `V_` re-score is 45 tracks
= 10.6 GPU-h, and a full `B_` re-predict is 12.0 GPU-h that §5 says must never happen.

---

## 2. What a `TrackRecord` actually holds

`harness.py:131-159`. One record per `(pair, assay, kind)`. Eight `Dict[str, np.ndarray]`
fields, keyed by chromosome, `float32`, indexed by **absolute bin** on the chromosome:

| field | arm | prediction or truth |
|---|---|---|
| `mu`, `n` | count | prediction — the NB parameters |
| `counts` | count | **truth** |
| `signal_mu`, `signal_sigma` | pval | prediction — the Gaussian |
| `pval` | pval | **truth** |
| `peak_score` | peak | prediction |
| `peaks` | peak | **truth** |

So a 3-head CANDI record is **5 prediction arrays and 3 truth arrays**. That count is the
whole of §4's storage finding.

---

## 3. The design

### 3.1 One file per prediction run

One HDF5 file per `(method, regime, panel)` — i.e. per unit-panel, 30 files across the
programme, matching §12.3's 30 prediction runs one-to-one. Not one file per track: a
prediction run is the thing §5 says happens once, so it should be the thing on disk that you
can point at and say "this is the run".

```
preds.candi.eic_19.B_.h5
  /provenance                     (attrs only — §3.3)
  /tracks/<track_key>/mu/<chrom>          float32[n_bins]  gzip-9 + shuffle, chunks=(65536,)
  /tracks/<track_key>/n/<chrom>
  /tracks/<track_key>/signal_mu/<chrom>
  /tracks/<track_key>/signal_sigma/<chrom>
  /tracks/<track_key>/peak_score/<chrom>
```

`track_key` is already `harness.track_key(pair, assay, kind)` — reuse it verbatim, do not
invent a second naming scheme. `h5py` is a core dependency (`STORE.md`), so this adds nothing
to the environment.

### 3.2 Predictions only. Truth is re-read at score time — RECOMMENDED, needs a ruling

Truth is **not** written. Three reasons, in order of weight:

1. **§6 requires truth to be swappable.** The truth toggle re-scores the same predictions
   against a different truth. Truth baked into the prediction file can therefore never be the
   only truth, so writing it buys a convenience and not a capability.
2. It is 3 of the 8 arrays — see §4 for what that costs.
3. Re-reading truth is CPU work against the store, which §12.4 already says is not the scarce
   resource.

The thing this gives up: a scored pass is no longer reproducible without the store. §3.3 is
what covers that — the file records *which* store it was predicted against, so a re-score can
prove it read the same truth rather than assume it.

### 3.3 Provenance — what §27 taught

`cruxvault/results/t78/G1_REBUILD_FROM_BAMS.md` §27 item 1: every scored JSON in the repo
pins `6c0e0c3e…`, a store that no longer exists, so each one "describes an unreproducible run
until it is re-scored." A prediction file outlives the store it was made against, so it must
say which one that was:

```
store_manifest_sha256   the store the predictions were made against
regime_id               eic.19 | eic.pilot
regime_sha256           the regime config's own hash
panel                   V_ | B_
kind                    impute | denoise
ckpt_sha256             the checkpoint, so "predicted once" is checkable
code_sha                git sha of the tree that produced it
seed
scope_chroms            every chromosome present
arms                    which of the 5 prediction fields are populated
n_tracks
written_utc
```

`ckpt_sha256` is the one that makes §5 enforceable: two `B_` prediction files with different
checkpoint hashes is a visible violation, where today it is invisible.

### 3.4 The CLI seam

`run_bench` currently hard-codes its producer at `harness.py:1481`:

```python
for rec in stream_tracks(model, source, device, kind=kind, ...):
```

The change is to accept the producer instead of constructing it, defaulting to what it does
now. Then two things become expressible:

```bash
# predict once, write, score nothing
python -m candi.bench predict --store configs/regime.eic_19.json --ckpt m.pt \
    --panel B_ --out preds.candi.eic_19.B_.h5

# score any number of times, no model, no GPU
python -m candi.bench run --predictions preds.candi.eic_19.B_.h5 \
    --held-out-chroms chr20,chr21,chr22 --out bench.json
```

`read_tracks(path) -> Iterator[TrackRecord]` is the reader; it and `stream_tracks` have the
same type, which is the entire reason this is a small change and not a rewrite.

**No number moves.** `tools/golden.py` builds a model from `manual_seed(0)` and never opens a
store or a prediction file (§24.6 of the t78 memo), so it cannot move. The scoring arithmetic
is untouched — `score_track` receives the same `TrackRecord` whether it came from a GPU or a
file. That is the property a test should assert directly: predict → write → read → score must
equal predict → score, bit for bit.

### 3.5 What it does not do

No bigwig export. No merging across methods. No re-prediction, and no flag that would make
re-predicting `B_` convenient — §5 makes that an act requiring a fresh checkpoint selection,
so it should stay inconvenient.

---

## 4. Two findings that contradict §12, and one collision

### 4.1 §12.3 and §12.6 budget one array per track. CANDI needs five.

§12.3: "485 MB per track genome-wide (121,241,684 bins × 4 bytes)". That product is
484,966,736 bytes — **exactly one `float32` array.** §12.6 repeats it, §12.3's per-unit
figures are `45 × 485 MB = 21.8 GB` and `51 × 485 MB = 24.7 GB`, and §12.8 concludes "CANDI's
whole share is 93 GB", which is `192 × 485 MB`. The budget is internally consistent and it
assumes a **point prediction**.

CANDI is not a point prediction. §7 badges it `native heteroscedastic`, and that is the
point: CRPS, PIT and coverage — the measures §5.3 ranks on — need the NB `(n̂, p̂)` pair, not
a mean. So the real count is 5 prediction arrays, and the budget is short by 5×:

| | raw | gzip-9 (measured, §4.2) |
|---|---|---|
| §12.3/§12.8 as written — 1 array/track | 93 GB | — |
| predictions only — 5 arrays/track | **466 GB** | **173 GB** |
| predictions + truth — 8 arrays/track | 745 GB | 277 GB |

Against §12.3's **434 GB for the whole programme**, CANDI alone is 466 GB uncompressed. The
rivals and naive baselines are point-only on the p-value arm, so their §12.3 figures are
right as written; this is a CANDI-and-eDICE problem, not a programme-wide error.

**Compression is what rescues it**, which is why it was worth measuring rather than assuming.
At the measured 2.69× the predictions-only footprint is 173 GB, inside the existing envelope.
That is the case for gzip being part of the design and not an optimisation.

### 4.2 The compression measurement, and its named weakness

Run on Fir 2026-08-31 against the promoted store (`c9a95e4e…`), chr21 bins 1,000,000–1,600,000
(600k bins, 15 Mb), 5 tracks × 2 layers, `gzip-9 + shuffle`, `chunks=(65536,)`:

| layer | ratio range | why |
|---|---|---|
| `counts` | 5.8× – 19.7× | sparse and integer-valued; 19–89 % non-zero |
| `pval` | 1.3× – 2.1× | dense continuous floats; 62–100 % non-zero |

Blended over all 10 arrays: **24.00 MB → 8.92 MB, 2.69×**, i.e. 485 MB → 180 MB per
genome-wide track.

**The weakness, stated plainly: this measures TRUTH arrays, not predictions.** No trained
CANDI checkpoint exists on Fir (§12.7 re-verified this), so predictions could not be measured.
Model outputs are smooth full-mantissa floats and may compress **worse** than the `pval` truth
they are compared against — 1.3× is the pessimistic anchor, and at 1.3× the predictions-only
footprint is 358 GB, not 173 GB. The first real prediction run should record its own ratio,
and the storage plan should not be closed on this estimate.

### 4.3 §12.3 says predictions are deletable. §5 says `B_` cannot be re-made.

Two approved rulings collide:

- **§12.3:** predictions live on "scratch, never `/project`; deletable after scoring", and
  "**scratch purges at 60 days**".
- **§5:** "`B_` predictions are produced one time … Re-scoring those stored predictions is
  free and allowed; re-predicting `B_` is not, and needs a new checkpoint selection on `V_` to
  be legitimate."

So the `B_` prediction set is the **only legitimate copy that will ever exist** — and the plan
puts it on a filesystem that deletes it automatically after 60 days. Once it is gone, any
measure not computed in the first scoring pass is unreachable: getting it back means
re-predicting, which §5 forbids without a fresh checkpoint selection.

This is not a storage question, it is a question about what `B_` is. Options, for the PI:

1. **`B_` prediction files are archival** — they move to `/project` after scoring and are kept.
   173 GB gz for CANDI's share is real but affordable against 11 TB free (t78 memo §25 Step 3).
2. **Accept the loss, and close the arm deliberately** — declare the `B_` scoring pass
   exhaustive, write down that no later measure can be added, and let scratch purge it.
3. **Keep `V_` on scratch, `B_` on `/project`.** `V_` is re-predictable at will, so only `B_`
   carries the one-way door.

I would take 3: it spends the storage exactly where the irreversibility is. But this is a
ruling about evidence, not about disk, so it is not mine to make.

---

## 5. What I recommend, in one list

1. Predictions only, truth re-read at score time (§3.2).
2. gzip-9 + shuffle, one file per unit-panel, `track_key` reused (§3.1).
3. Provenance carries `store_manifest_sha256` and `ckpt_sha256` (§3.3).
4. `run_bench` takes its producer; `read_tracks` mirrors `stream_tracks` (§3.4).
5. The equality test — predict → write → read → score equals predict → score, bit-exactly —
   is the gate this lands behind (§3.4).
6. §12.3 and §12.6 get corrected to 5 arrays per distributional track, with the measured
   compression ratio and its weakness recorded beside them (§4.1, §4.2).
7. `B_` prediction files go to `/project`, `V_` stays on scratch (§4.3) — **PI's call.**
