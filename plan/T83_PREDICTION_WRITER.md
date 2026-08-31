# T83_PREDICTION_WRITER.md — the prediction track writer

**Status: RE-SCOPED 2026-08-31. The writer exists.** This document was first written as a
design for a writer to be built from nothing. That premise was false. What is left is two
named changes to the writer that is already merged, and they are small.

---

## 1. What this document got wrong

The first version was written against `BENCHMARK_DESIGN.md` §12.8, which said:

> `stream_tracks` yields in-memory `TrackRecord`s straight into `score_track`. The §12.3
> prediction run that persists 485 MB per track to scratch does not exist as code.

**`src/candi/bench/dump.py` already did exactly that**, with `tests/test_bench_dump.py`
covering it. It writes `mu`, `n`, `signal_mu`, `signal_sigma` and `peak_score` to the
`RIVALS_PLAN.md` §4.1 prediction-root contract, reusing `harness.stream_tracks` and
`cli._build_model`.

The §12.8 survey read one branch — `implementation/t77-benchmark-design` — and `dump.py` was
on `implementation/t62-candi-rows`, written weeks earlier. It is now merged onto the working
branch. §12.8 carries the correction.

This is the same failure as `t84`'s "only Avocado is vendored": a claim about the repo taken
from one branch. **Before saying code does not exist, search every branch.**

What this cost: §3 of the first version designed an HDF5 container, a `read_tracks` reader and
a producer seam in `run_bench`. Most of that is now dead. §4's storage analysis is not dead,
and is kept below.

---

## 2. Verdict — (b) `dump.py` needs two named changes

`dump.py` is the right shape. It satisfies four of the five claims that rest on "predict once,
score many times". It fails two, and both are small and local:

1. **It writes uncompressed.** `_write_npz` calls `np.savez`, which stores without deflating.
2. **Its manifest carries no checkpoint identity**, so §5's touch-once rule on `B_` is not
   auditable.

Neither needs a new container, a new reader, or a change to `run_bench`.

### 2.1 Evidence, claim by claim

| BENCHMARK_DESIGN claim | satisfied by `dump.py` as it stands? | evidence |
|---|---|---|
| §4 — `held-out` and `genome-wide` from one pass | **yes** | `dump_predictions` writes every chromosome of `source.eval_chroms`. `external`'s own `--chroms` help says "P2 (genome-wide) is this flag with every chromosome the store carries." Two scoring runs over one root, no second inference. |
| §5.2 — the three `V_` numbers | **yes** | `V_` matched is the 8 assays `B_` contains — an assay subset of the same scored tracks. `score_external` emits a `per_track` block keyed by track, so the matched macro is `harness.macro_mean` over a subset of the score JSON. It needs neither a second scoring pass nor a second read of the predictions. |
| §6 — both truths against one prediction set | **yes** | Truth enters only through `--store` (`stream_truth`, `open_source`); the prediction enters only through `--pred`. Swapping the source swaps the truth and nothing else, which is exactly what §6 asks for. A challenge-truth `EvalSource` still has to be built, but that is §6's task, not the writer's. |
| §5 — `B_` is predicted once, and that is auditable | **no** | `write_manifest` emits `{method, version, generated_by, date, arms, declared_tracks, notes}`. No checkpoint hash, no store hash, no panel, no code sha. Two `B_` roots made from two different checkpoints are indistinguishable on disk. |
| §12.3 / §12.6 — the footprint is affordable | **no** | `dump.py:102` is `np.savez`, not `np.savez_compressed`. §12.6's affordability rests on compression the writer does not apply, so CANDI's ≈466 GB is written raw. |

### 2.2 Why the round-trip is a fact and not a claim

`dump.py` imports `track_dirname` and `_expected` **from** `external.py`, and `external.py`'s
`PRED_ARRAYS` is exactly the five arrays `dump.py` writes. The two cannot drift apart in
naming, because there is one definition.

Beyond that, `tests/test_bench_dump.py` scores a dumped root through `score_external` and
asserts every shared numeric key equals `harness.run_bench` on the same model to `rel=1e-6`,
per track, per macro and per panel. **Verified on this worktree: 8 passed in 29 s**
(`PYTHONPATH=$PWD/src pytest tests/test_bench_dump.py -q`).

That equality test is the gate the first version of this document proposed writing (its §3.4).
It already exists and it already passes.

---

## 3. The remaining work

Two changes in one file, plus one measurement. This is the whole of `t83`.

### 3.1 Compress on write — `src/candi/bench/dump.py:102`

`np.savez` → `np.savez_compressed`. `external.read_track_arrays` uses `np.load`, which reads
either transparently, so **no consumer changes and no test changes.**

Do not expect the 2.69× of §4.2 below. That was measured with HDF5 gzip-9 **plus the shuffle
filter** on 64 k chunks; `savez_compressed` gives zlib level 6 over the whole array with no
shuffle. A synthetic check isolating only the container, on one smooth `float32` array of the
kind a model head emits — **not a prediction, and not evidence about predictions**:

| container | ratio |
|---|---|
| `np.savez` (what the code does today) | 1.00× |
| `np.savez_compressed` | 1.23× |
| HDF5 gzip-9 + shuffle, `chunks=(65536,)` | 1.67× |

So the container is worth about a third, not a multiple. Take the one-word change now.

### 3.2 Put the checkpoint in the manifest — `dump.py:write_manifest`

Add the fields that make "predicted once" checkable:

| field | why |
|---|---|
| `ckpt_sha256` | **the one that matters.** Two `B_` roots with different hashes is a visible violation of §5, where today it is invisible. |
| `store_manifest_sha256` | `t78` §27: every scored JSON in the repo pins a store that no longer exists. `StoreSource.provenance()` records the store *path*, not a content hash, so a re-score cannot prove it read the same truth. |
| `regime_id`, `panel` (`V_` / `B_`) | a root should say which exam it sat. |
| `code_sha`, `seed`, `chroms` | ordinary reproducibility. |

`external.read_manifest` already copies the manifest **verbatim** into
`provenance.manifest`, so every added field reaches the score JSON with no change on the
consumer side.

### 3.3 Record the real ratio on the first prediction run

§12.6 says this table is rewritten from the first real run. `dump_predictions` should report
bytes written against bytes raw, so the number is captured rather than reconstructed later.

### 3.4 Explicitly dropped from the first version

| dropped | why |
|---|---|
| §3.1 — one HDF5 file per unit-panel | The npz-per-`(track, chrom)` layout **is** the §4.1 contract `external.py` reads and every rival exporter targets. Changing the container to suit CANDI alone would break that contract for a gain §3.1 above measures at about a third. Revisit only if the measured ratio of §3.3 misses the budget badly, and then as a CANDI-only sidecar, never as the rival-facing format. |
| §3.2 — predictions only, truth re-read at score time | Already true. `dump.py` writes no truth array. |
| §3.4 — a producer seam in `run_bench`, and `read_tracks` | Not needed. `score_external` already is the read-and-score path, with no model and no GPU. |
| §3.4 — the predict→write→read→score equality test | Already written, already passing (§2.2). |

---

## 4. Storage — the analysis that survives

### 4.1 Five arrays per track, not one

`TrackRecord` (`harness.py:131-159`) carries eight arrays per record. Five are prediction and
three are truth:

| field | arm | prediction or truth |
|---|---|---|
| `mu`, `n` | count | prediction — the NB pair |
| `counts` | count | truth |
| `signal_mu`, `signal_sigma` | pval | prediction — the Gaussian |
| `pval` | pval | truth |
| `peak_score` | peak | prediction |
| `peaks` | peak | truth |

`dump.py` writes the five prediction arrays and none of the truth.

The original §12.3 budget counted **one** array per track: 485 MB genome-wide is
121,241,684 × 4 = 484,966,736 bytes, exactly one `float32` vector. That is right for a point
predictor and wrong for CANDI, which §7 badges `native heteroscedastic` — CRPS, PIT and
coverage all need the NB pair, not a mean.

| | raw |
|---|---|
| §12.3 as originally written — 1 array/track | 93 GB |
| CANDI, 5 prediction arrays/track, 192 track sweeps | **≈466 GB** |

Re-derived here: 192 sweeps (45 `V_` + 51 `B_`, × 2 regimes) × 5 × 484,966,736 B = 465.6 GB.
§12.6 and §12.8 carry this correction. The rivals and the naive baselines are point-only on
the p-value arm, so their §12.3 figures stand.

*Minor, and left alone because this document must not edit `BENCHMARK_DESIGN.md`:* §12.6's
per-track figure reads 2.37 GB where 5 × 484,966,736 B is 2.42 GB.

Genome-wide, the layout is 192 × 23 = **4,416 npz files**. Worth a glance against `/project`'s
file-count quota before the `B_` run, not a redesign.

### 4.2 The compression measurement, and its named weakness

Run on Fir 2026-08-31 against the promoted store (`c9a95e4e…`), chr21 bins 1,000,000–1,600,000
(600 k bins, 15 Mb), 5 tracks × 2 layers, gzip-9 + shuffle, `chunks=(65536,)`:

| layer | ratio range | why |
|---|---|---|
| `counts` | 5.8× – 19.7× | sparse and integer-valued; 19–89 % non-zero |
| `pval` | 1.3× – 2.1× | dense continuous floats; 62–100 % non-zero |

Blended over all 10 arrays: 24.00 MB → 8.92 MB, **2.69×**.

**The weakness, stated plainly: this measured TRUTH arrays, not predictions.** No trained CANDI
checkpoint existed on Fir when it was taken (§12.7). Model outputs are smooth full-mantissa
floats and may compress worse; `pval`'s 1.3× is the pessimistic anchor, and at 1.3× CANDI's
share is 358 GB, not 173 GB.

Two reasons to treat 2.69× as unconfirmed, not one:

1. It came from truth arrays.
2. It came from a filter `savez_compressed` does not have (§3.1).

**And a third, which is the one that actually kills the number.** The 2.69× is a blend of a
sparse layer and a dense one — `counts` at 5.8×–19.7× pulling the average up, `pval` at
1.3×–2.1× holding it down. That blend is a property of **truth**, where the count layer really
is sparse integers. It is not a property of a **prediction**.

Every one of CANDI's five prediction arrays is a model output: `mu` and `n` from a softplus, and
`signal_mu`, `signal_sigma`, `peak_score` likewise. **All five are smooth full-mantissa floats.
None of them is sparse.** So the count arm's 15× never applies on the prediction side at all,
and the blend has no sparse term to average in.

Measured directly, two arrays of 2 M float32, `numpy` 1.27× / 15.39× (independent check,
2026-08-31) — the same split the agent's 1.23× / 1.67× table shows:

| array kind | `savez` | `savez_compressed` |
|---|---|---|
| smooth float32 — **what a prediction is** | 1.00× | **1.27×** |
| sparse count-like — what truth counts are | 1.00× | 15.39× |

**So the working estimate for prediction storage is ≈1.27×, not 2.69×**, and CANDI's share is
**≈367 GB**, not 173 GB. The "pessimistic anchor" of 358 GB above is therefore the *expected*
case, not the bad case.

This is still an estimate on synthetic data. It does not close either — but it is the number to
plan against, and it is close enough to the raw 466 GB that **compression should be treated as a
convenience, not as what makes the plan affordable.** What makes it affordable is that Fir has
room: 13 TiB free on `/project`, 17 TiB free on scratch, checked 2026-08-31.

The storage plan does not close on any of these. §3.3 replaces them with a measurement from the
first real prediction run.

### 4.3 Where `B_` lives — RULED, no longer open

The first version raised a collision between §12.3 ("scratch, deletable, purges at 60 days")
and §5 (`B_` is predicted once, so the first set is the only legitimate copy). **The PI ruled
on 2026-08-31, and §12.6 carries it: `B_` predictions go to `/project`; `V_` stays on scratch
and is deletable once scored.** That is option 3 of the three the first version listed. It
spends the storage where the irreversibility is.

`B_` is also never read during training at all — not merely never selected on (PI, same day).

---

## 5. What t83 is, in one list

1. `dump.py:102` — `np.savez` → `np.savez_compressed`. No consumer change.
2. `dump.py:write_manifest` — add `ckpt_sha256`, `store_manifest_sha256`, `regime_id`,
   `panel`, `code_sha`, `seed`, `chroms`. `read_manifest` already copies them through.
3. `dump_predictions` — report the achieved compression ratio, so §12.6 is rewritten from a
   measurement.

Everything else the first version proposed is either already built or dropped (§3.4).
