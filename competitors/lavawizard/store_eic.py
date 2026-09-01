"""CANDI_STORE -> the same per-chromosome cache `preprocess.py` builds, then the P1/P2 predictions.

This is the **our-EIC** side of the port. `dataset3.py` reads the challenge's own bigwigs and must
never import `candi`; this file reads our store and therefore must. Nothing else crosses: both
produce the identical cache layout, so `train.py` and `model.py` cannot tell which corpus they are
looking at, which is the point — the anchor and the deliverable are the same code on two corpora.

**No binning happens here.** `STORE.md`'s grid is already `floor(chr_len / 25)` and
`BiosampleStore.pval` returns decoded `-log10 p` on exactly that grid, so the only transform is
`arcsinh` on the binned value. Upstream applies `arcsinh` per *base* and then averages into a bin;
we cannot, and the difference is recorded in the README rather than papered over. Their grid also
*ceils* where the store *floors*, so a chromosome here is one bin shorter than the anchor's.

**§6.2 is enforced in `train_columns`, and it raises rather than filters.** A fairness rule that
silently drops a column is a fairness rule nobody can audit. The T_/V_ identification comes from
the regime's own `eval_pairs`, never from splitting a biosample name — `STORE.md` D16 says a
biosample id is opaque, so a regime declaring `[T_A, V_B]` is honoured exactly as written.

The shape of this module deliberately mirrors `competitors/avocado/index.py` and `bin_store.py`.
The two methods are separate deliverables under `RIVALS_PLAN.md` §3 and neither imports the other,
so the mirroring is by convention, not by shared code — one reading of §6.2 per method, in the same
place, checkable side by side.

**Two things live here that `train.py` cannot hold.** BENCHMARK_DESIGN.md §5's checkpoint selection
scores a written prediction root with `candi.bench.external`, and D32's training scope is a BED
behind `candi.store.regime`'s hash gate — both need `candi`, and `train.py` must keep running on
Fir without it. So `selector` builds the selection callable the trainer is handed, and
`build_cache_from_store` resolves the BED once and writes the eligible bins into the cache.

```bash
python -m lavawizard.store_eic cache   --regime configs/regime.eic_19.json --chrom chr21 \
       --cache /scratch/.../eic_cache
python -m lavawizard.store_eic train   --regime configs/regime.eic_19.json --chrom chr21 \
       --cache /scratch/.../eic_cache --out runs/eic/ckpt --select-every 50
python -m lavawizard.store_eic predict --regime configs/regime.eic_19.json --chrom chr21 \
       --cache /scratch/.../eic_cache --checkpoint runs/eic/ckpt/guacamole_chr21.best.pt \
       --pred-root runs/eic/pred --clip
```
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import emit, preprocess

__all__ = ["FairnessError", "ScopeError", "RESOLUTION", "SELECT_KEY", "load_regime", "cell_index",
           "train_columns", "contained_bins", "derive_v_only", "write_v_only_regime",
           "build_cache_from_store", "write_predictions", "predict_chrom", "selector",
           "train_chrom"]

CHUNK_BINS = 1_000_000

#: The store's bin width. `STORE.md`'s grid, and D32's containment is counted against it.
RESOLUTION = 25

#: BENCHMARK_DESIGN.md §5's selection number for this method, as `<arm>:<key>` into the `macro`
#: block `candi.bench.external.score_external` returns. Lower is better.
#:
#: **CANDI selects on `count:crps` (`candi.monitor.SELECTION_KEY`) and this method cannot.** It has
#: no count arm at all — B1b forbids inventing a read depth — and the pval arm's distributional
#: keys, CRPS included, are ABSENT rather than NaN on a point-only track until a σ-table supplies a
#: spread. §7 rules that σ is fit on training residuals and never on `V_`, and no such table exists
#: during training, so `crps` is not merely inconvenient here, it is unavailable by the same rule
#: that keeps the run honest. `pval:mse` is the arm's own point key, present on every scored track,
#: lower-better, and it is what stage 2 minimises — so selection asks the panel the question the
#: loss asks the training set. It is `--select-key` rather than a constant because *which* number
#: selects is the PI's to change, and it is stamped into the run record either way.
SELECT_KEY = "pval:mse"

#: §5's panel prefixes. Membership of `V_` is carried by the target biosample's name and by nothing
#: else in a regime file, which is why the CANDI launcher (`slurm/t81_train_candi.sh`) filters on
#: the same prefix. D16's "a biosample id is opaque" still holds for everything else here: the
#: T_->V_ pairing comes from `eval_pairs`, never from taking a name apart.
V_PREFIX = "V_"
B_PREFIX = "B_"


class FairnessError(RuntimeError):
    """A track that `RIVALS_PLAN.md` §6.2 forbids reached the training pool."""


class ScopeError(RuntimeError):
    """The regime names a training scope this method's per-chromosome fit cannot express."""


def load_regime(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cell_index(regime: dict) -> Tuple[List[str], Dict[str, int]]:
    """`(cell names, biosample id -> cell index)`; a declared `[T_X, V_X]` pair shares one index.

    Guacamole factorises `(celltype, assay, position)`, so `T_X` and `V_X` must be one cell or a
    V_ target has no embedding to impute from. The embedding is still fitted from the T_ side
    alone — that is `train_columns`'s job, not this one — and the class name is the T_ id, so a
    checkpoint written here names cells in T_ terms only.
    """
    names = sorted(regime["biosamples"]["train"])
    ix = {c: i for i, c in enumerate(names)}
    for row in regime.get("eval_pairs", []):
        src, tgt = str(row[0]), str(row[1])
        if src not in ix:
            raise FairnessError(
                f"eval_pairs names `{src}` as an input biosample but biosamples.train does not; "
                f"there would be no fitted embedding for its cell.")
        ix[tgt] = ix[src]
    return names, ix


def train_columns(regime: dict, corpus) -> List[Tuple[str, str]]:
    """`[(biosample, assay), ...]` — every track this method is allowed to see. §6.2 lives here.

    Drawn from `biosamples.train` and from nowhere else. A train biosample that is also an eval
    **target** raises: that is the leak §6.2 forbids outright, and it must not be reachable by
    reading past a warning.
    """
    train = list(regime["biosamples"]["train"])
    targets = {str(row[1]) for row in regime.get("eval_pairs", [])}
    leaked = sorted(set(train) & targets)
    if leaked:
        raise FairnessError(
            f"biosamples.train contains {leaked}, which eval_pairs also names as imputation "
            f"TARGETS. Training on a target biosample's tracks is the leak §6.2 forbids.")
    declared = set(regime["assays"])
    cols: List[Tuple[str, str]] = []
    for b in sorted(train):
        have = set(corpus[b].assays()) & declared
        cols.extend((b, a) for a in regime["assays"] if a in have)
    if not cols:
        raise FairnessError("no training tracks: no declared train biosample carries a declared "
                            "assay.")
    return cols


# ---------------------------------------------------------------------------
# D32 — the BED-restricted training scope
# ---------------------------------------------------------------------------

def contained_bins(regime: dict, chrom: str, *, base: Path | str,
                   resolution: int = RESOLUTION) -> Optional[np.ndarray]:
    """The bins of `chrom` lying WHOLLY inside a region of the regime's BED. `None` without one.

    D32's rule is containment, and the parser, the `sha256` gate and the containment arithmetic are
    `candi.store.regime.RegionSet`'s — reached rather than re-implemented, because a second copy of
    the rule is a second thing to keep in step and the BED is pinned by a hash this module has no
    business checking twice. `base` is the directory a relative `regions.bed` resolves against,
    which is the regime file's own directory.

    A **locus here is one 25 bp bin**, not one 768-bin window. CANDI's `contained_starts` filters
    window starts and §3.1's 1,294 windows / 993,792 bins is that quantity; Guacamole samples single
    bins, so its scope is the containment count itself — 1,023,489 bins over the pilot regime's
    train chromosomes. The two numbers are both correct and are not the same number.

    Bin indices stay ABSOLUTE chromosome bins, so the grid is anchored at chromosome bin 0 and is
    not re-anchored per region — D32's choice, and the reason `eic.pilot` shares a grid with every
    other regime.
    """
    from candi.store.regime import RegionSet

    rs = RegionSet.from_obj(regime.get("regions"), base=Path(base))
    if rs is None:
        return None
    spans = rs.bin_spans(chrom, resolution)
    if not spans:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in spans])


def _assert_scope_is_expressible(regime: dict, chrom: str) -> None:
    """Refuse a `regions` regime on a chromosome the regime does not train on.

    THIS IS NOT "BED SUPPORT IS MISSING" — it is built, and `contained_bins` above is it. The
    refusal is about WHICH CHROMOSOME. `train.py` fits one independent Guacamole per chromosome,
    cell and assay embeddings included, on that chromosome's own bins; `predict_chrom` then indexes
    the genome tables by that chromosome's bin numbers, so the model that predicts chr20 can only
    have been fit on chr20. Under `eic.pilot` that puts the fit on the eval chromosomes, while the
    regime's `train_chroms` deliberately CUT the four Pilot Regions that fall there
    (BENCHMARK_DESIGN.md §3.1). Restricting the fit to those four would train this method on
    precisely the loci the regime excluded — the complement of the declared scope, wearing its name.

    So the scope is refused until the method has a transferable stage the regime can reach. That is
    the same PI decision the regime-collapse finding needs, and one ruling settles both.
    """
    if not regime.get("regions"):
        return
    train = list(regime.get("train_chroms") or [])
    if chrom in train:
        return
    raise ScopeError(
        f"{chrom}: this regime restricts training to a BED (D32) and its train_chroms are {train}, "
        f"which do not include {chrom}. Lavawizard fits one model per chromosome on that "
        f"chromosome's own bins, so a model for {chrom} can only be fit on {chrom} — and under this "
        f"regime {chrom}'s regions are the ones train_chroms cut. Training there would fit the "
        f"complement of the declared scope and label it with the regime's name. The BED restriction "
        f"itself is implemented and tested (`contained_bins`); what is missing is a transferable "
        f"stage this regime can reach. Raise it — it is the same ruling the regime-collapse needs.")


def derive_v_only(regime: dict) -> dict:
    """The regime with `eval_pairs` filtered to `V_` targets — §5's selection panel, and only it.

    PI ruling 2026-08-31: *"never ever we use B_ in training — V_ is only for checkpoint selection
    and monitoring, not training."* Both live regimes declare all 38 pairs inside them, so a
    selection pass that read the regime verbatim would score `B_` at every check and spend the
    single touch §5 allows. Same filter, same reasons and the same absolute-BED rewrite as
    `slurm/t81_train_candi.sh` does for CANDI, so the two methods select on the same panel.
    """
    pairs = [(str(row[0]), str(row[1])) for row in regime.get("eval_pairs", [])]
    kept = [list(p) for p in pairs if p[1].startswith(V_PREFIX)]
    if not kept:
        raise ScopeError("this regime declares no V_ eval pair, so there is no panel to select on")
    out = dict(regime)
    out["eval_pairs"] = kept
    # `eval_pairs` makes `biosamples.eval` inert, but leaving the B_ cells in a V_-only config
    # would be a false claim about the split for anyone who read the file.
    out["biosamples"] = dict(regime["biosamples"], eval=sorted({p[1] for p in kept}))
    return out


def write_v_only_regime(src: Path | str, dst: Path | str) -> Path:
    """`derive_v_only` to disk, with `regions.bed` made absolute. Returns `dst`.

    The BED path resolves against the regime file's OWN directory, so a derived copy written
    somewhere else must carry an absolute path or the D32 hash gate fails on a file it cannot find.
    """
    src, dst = Path(src), Path(dst)
    out = derive_v_only(load_regime(src))
    if out.get("regions"):
        out["regions"] = dict(out["regions"],
                              bed=str((src.parent / out["regions"]["bed"]).resolve()))
    out["_comment"] = (
        f"DERIVED from {src.name} by lavawizard.store_eic — eval_pairs filtered to V_ targets so "
        f"the mid-training selection never reads B_ (BENCHMARK_DESIGN.md §5). Do not edit; edit "
        f"the source.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return dst


def build_cache_from_store(regime_path: Path | str, chrom: str, out_root: Path | str,
                           *, chunk_bins: int = CHUNK_BINS, verbose: bool = True) -> Path:
    """Write `<cache>/<chrom>/` in exactly `preprocess.build_cache`'s layout, from our store.

    Idempotent: an existing `index.json` returns early, so a re-run of a partly finished array
    costs nothing.

    **The D32 training scope is resolved here, once, and written into the cache** as
    `train_bins.npy`. It belongs at cache time and not in the trainer because the BED sits behind
    `candi.store.regime`'s hash gate and `train.py` never imports `candi`. The whole chromosome's
    signal is still cached either way — prediction covers every bin whatever the training scope is.
    """
    from candi.store.reader import CorpusStore

    regime = load_regime(regime_path)
    _assert_scope_is_expressible(regime, chrom)
    out = preprocess.cache_dir(out_root, chrom)
    if (out / "index.json").exists():
        if verbose:
            print(f"{chrom}: cache already present at {out}", flush=True)
        return out
    out.mkdir(parents=True, exist_ok=True)

    cells, _ = cell_index(regime)
    marks: List[str] = list(regime["assays"])
    mark_ix = {m: i for i, m in enumerate(marks)}

    with CorpusStore(regime["store"]) as corpus:
        cols = train_columns(regime, corpus)
        n_bins = int(corpus.n_bins(chrom))
        n_tracks, n_marks = len(cols), len(marks)
        if verbose:
            print(f"{chrom}: {n_tracks} tracks x {n_bins} bins, {len(cells)} cells, {n_marks} "
                  f"marks ({n_tracks * n_bins * 4 / 2**30:.1f} GiB)", flush=True)

        arr = np.lib.format.open_memmap(out / "tracks.npy", mode="w+",
                                        dtype=np.float32, shape=(n_tracks, n_bins))
        ter = np.lib.format.open_memmap(out / "tercile.npy", mode="w+",
                                        dtype=np.int8, shape=(n_tracks, n_bins))
        sums = np.zeros((n_marks, n_bins), dtype=np.float64)
        sumsq = np.zeros((n_marks, n_bins), dtype=np.float64)
        counts = np.zeros(n_marks, dtype=np.int64)
        maxima = np.zeros(n_marks, dtype=np.float64)

        by_bios: Dict[str, List[Tuple[int, str]]] = {}
        for i, (b, assay) in enumerate(cols):
            by_bios.setdefault(b, []).append((i, assay))

        t0 = time.time()
        for k, (b, rows) in enumerate(sorted(by_bios.items())):
            names = [nm for _, nm in rows]
            bs = corpus[b]
            block = np.empty((len(rows), n_bins), dtype=np.float32)
            for s in range(0, n_bins, chunk_bins):
                e = min(s + chunk_bins, n_bins)
                block[:, s:e] = np.arcsinh(
                    np.asarray(bs.pval(chrom, s, e, assays=names), dtype=np.float32)).T
            for (i, assay), row in zip(rows, block):
                arr[i] = row
                ter[i] = preprocess._terciles(row)
                j = mark_ix[assay]
                sums[j] += row
                sumsq[j] += row.astype(np.float64) ** 2
                counts[j] += 1
                maxima[j] = max(maxima[j], float(row.max()))
            if verbose:
                print(f"  [{k+1}/{len(by_bios)}] {b}: {len(rows)} track(s) "
                      f"({time.time()-t0:.0f}s)", flush=True)
        arr.flush(); ter.flush()

    np.save(out / "sums.npy", sums.astype(np.float32))
    np.save(out / "sumsq.npy", sumsq.astype(np.float32))
    scope = None
    bins = contained_bins(regime, chrom, base=Path(regime_path).parent)
    if bins is not None:
        if bins.size == 0:
            raise ScopeError(f"{chrom}: no bin lies wholly inside a region of the regime's BED, so "
                             f"this chromosome has no training scope at all under D32.")
        np.save(out / "train_bins.npy", bins)
        scope = {"policy": "contain", "resolution": RESOLUTION, "n_bins": int(bins.size),
                 **{k: regime["regions"][k] for k in ("bed", "sha256") if k in regime["regions"]}}
        if verbose:
            print(f"{chrom}: D32 training scope {bins.size} of {n_bins} bins "
                  f"({100.0 * bins.size / n_bins:.2f} %) inside the BED", flush=True)
    index = {
        "chrom": chrom, "n_bins": int(n_bins), "grid": "store_floor",
        "tracks": [list(t) for t in cols], "cells": cells, "marks": marks,
        "mark_counts": {m: int(counts[mark_ix[m]]) for m in marks},
        "mark_max": {m: float(maxima[mark_ix[m]]) for m in marks},
        # D32 — OMITTED, not null, on a whole-chromosome cache: absent means "every bin", which is
        # what every cache built before the key existed also means.
        **({} if scope is None else {"train_scope": scope}),
        "source": "CANDI_STORE", "store": str(regime["store"]),
        "regime": str(Path(regime_path).name),
        "signal": "arcsinh(pval) on the store's binned -log10 p",
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "upstream": "github.com/ccchang0111/ENCODE_imputation_2019@d638b204",
    }
    (out / "index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    if verbose:
        print(f"{chrom}: done in {(time.time()-t0)/60:.1f} min -> {out}", flush=True)
    return out


def _declared_tracks(regime: dict, corpus) -> List[Tuple[str, str, str]]:
    """`[(input, target, assay), ...]` — the §4.1 tracks a P1/P2 run must cover.

    An assay is declared for a pair when the **target** biosample carries it, because the target is
    where the truth comes from. Read from the store rather than assumed, so a pair with a partial
    panel produces exactly the tracks the scorer will look for.
    """
    declared = list(regime["assays"])
    out: List[Tuple[str, str, str]] = []
    for row in regime.get("eval_pairs", []):
        src, tgt = str(row[0]), str(row[1])
        have = set(corpus[tgt].assays())
        out.extend((src, tgt, a) for a in declared if a in have)
    return out


def write_predictions(model, cix: Dict[str, int], cache, tracks: Sequence[Tuple[str, str, str]],
                      pred_root: Path | str, *, chrom: str, n_bins: int, clip: bool,
                      device: str = "cpu", batch: int = 200_000,
                      verbose: bool = True) -> List[str]:
    """`tracks` for one chromosome from a LIVE model, written as §4.1 npz. Returns directory names.

    Split out of `predict_chrom` so the §5 selection pass can score the weights it currently holds
    without saving and re-loading them every check. Two call sites is normally under the bar for a
    helper; the §6.2 exclusion arithmetic below is the reason it is over it here — this module's own
    docstring says one reading of §6.2 per method, in one place, and a second copy of the moment
    subtraction is a second place for it to drift.
    """
    import torch

    dev = torch.device(device)
    written: List[str] = []
    cell_of_track = np.array([cix[b] for b, _ in cache.tracks], dtype=np.int64)
    t0 = time.time()
    for i, (src, tgt, assay) in enumerate(tracks):
        ci, mi = cix[src], cache.marks.index(assay)
        rows = np.flatnonzero((cache.mark_ix == mi) & (cell_of_track == ci))
        k = int(cache.mark_count[mi]) - len(rows)
        if k < 1:
            print(f"  skip {src}->{tgt} {assay}: 0 contributors after §6.2 exclusion", flush=True)
            continue
        raw = np.empty(n_bins, dtype=np.float32)
        for s in range(0, n_bins, batch):
            e = min(s + batch, n_bins)
            drop = cache.values[rows, s:e].astype(np.float64) if len(rows) else 0.0
            ssum = cache.sums[mi, s:e].astype(np.float64) - (drop.sum(0) if len(rows) else 0.0)
            ssq = cache.sumsq[mi, s:e].astype(np.float64) - ((drop ** 2).sum(0) if len(rows) else 0.0)
            avg = (ssum / k).astype(np.float32)
            var = np.maximum(ssq / k - avg.astype(np.float64) ** 2, 0.0).astype(np.float32)
            with torch.no_grad():
                raw[s:e] = model(
                    celltype=torch.full((e - s,), ci, dtype=torch.long, device=dev),
                    assay=torch.full((e - s,), mi, dtype=torch.long, device=dev),
                    pos25=torch.arange(s, e, dtype=torch.long, device=dev),
                    average=torch.from_numpy(avg).to(dev),
                    variance=torch.from_numpy(var).to(dev),
                ).float().cpu().numpy()
        cap = float(np.sinh(cache.mark_max[mi])) if clip else None
        emit.write_track(pred_root, emit.Pair(src, tgt), assay, chrom, raw,
                         n_bins=n_bins, clip_max=cap)
        written.append(emit.track_dirname(emit.Pair(src, tgt), assay))
        if verbose:
            print(f"  [{i+1}/{len(tracks)}] {src}->{tgt} {assay}  k={k}  "
                  f"cap={'off' if cap is None else f'{cap:.1f}'}  ({time.time()-t0:.0f}s)",
                  flush=True)
    return written


def predict_chrom(regime_path: Path | str, chrom: str, cache_root: Path | str,
                  checkpoint: Path | str, pred_root: Path | str, *,
                  clip: bool, device: str = "cpu", batch: int = 200_000,
                  verbose: bool = True) -> List[str]:
    """Every declared track for one chromosome, written as §4.1 npz. Returns the directory names.

    The contributor average is pooled over the mark's training tracks **minus the pair's input
    cell** — §6.2's exclusion, taken from the declared pair and not from a name suffix. On our
    store the target itself is never in the pool (a V_ track is not a training track), so this
    subtraction removes the one biosample that shares the target's cell.
    """
    from .anchor import load_checkpoint
    from candi.store.reader import CorpusStore

    regime = load_regime(regime_path)
    cache = preprocess.CachedChrom(cache_root, chrom, mmap=True)
    if clip and cache.mark_max is None:
        raise ValueError(f"{chrom}: --clip needs `mark_max`, and this cache predates it. Rebuild "
                         f"the cache; a cap guessed from a cache that never measured one is worse "
                         f"than no cap.")
    model, meta = load_checkpoint(checkpoint, device=device)
    cells, cix = cell_index(regime)
    if meta["cells"] != cells or meta["marks"] != cache.marks:
        raise ValueError(
            f"{chrom}: the checkpoint's index space is not this regime's. An off-by-one in the "
            f"cell or mark order silently predicts the wrong track, so this refuses rather than "
            f"guesses. checkpoint {len(meta['cells'])} cells / {len(meta['marks'])} marks, "
            f"regime {len(cells)} / {len(cache.marks)}.")

    with CorpusStore(regime["store"]) as corpus:
        tracks = _declared_tracks(regime, corpus)
        n_bins = int(corpus.n_bins(chrom))
    if n_bins != cache.n_bins:
        raise ValueError(f"{chrom}: store says {n_bins} bins, cache says {cache.n_bins}")

    return write_predictions(model, cix, cache, tracks, pred_root, chrom=chrom, n_bins=n_bins,
                             clip=clip, device=device, batch=batch, verbose=verbose)


# ---------------------------------------------------------------------------
# §5 — checkpoint selection on the V_ panel
# ---------------------------------------------------------------------------

def _macro_value(result: dict, key: str) -> float:
    """`<arm>:<key>` out of `score_external`'s `macro` block. Raises if the arm never scored."""
    arm, _, name = key.partition(":")
    if not name:
        raise ValueError(f"--select-key must be `<arm>:<key>`, e.g. {SELECT_KEY!r}; got {key!r}")
    macro = (result.get("macro") or {}).get(arm) or {}
    if name not in macro:
        raise ValueError(
            f"score_external produced no `{name}` on the {arm} arm — it carries "
            f"{sorted(k for k in macro if not k.endswith('_n_tracks'))}. A selection metric that "
            f"is absent selects nothing, so this refuses rather than treating it as +inf.")
    return float(macro[name])


def selector(regime_path: Path | str, chrom: str, cache, work_dir: Path | str, *,
             select_key: str = SELECT_KEY, clip: bool, device: str = "cpu",
             batch: int = 200_000, seed: int = 0,
             verbose: bool = True) -> Tuple[Callable[[object, int, int], float], dict]:
    """`(select_fn, info)` — §5's uniform selection, wired for `train.train_chromosome`.

    A check is: write this model's `V_` predictions for `chrom` into a scratch §4.1 root, hand the
    root to `candi.bench.external.score_external`, and read one number off the result. That is the
    SAME instrument that scores CANDI and the same one that produces the board row, which is what
    "every method selects by the same rule" has to mean if it is to mean anything.

    **`B_` is never read.** The scored panel comes from a derived V_-only regime written next to the
    run (`write_v_only_regime`), so both the declared track list and the truth reads are V_ only.
    `info["pairs"]` and `info["b_pairs"]` are there so a caller can assert that on the real regime.

    The scratch root is REUSED across checks. Every check writes the same track set, so the npz
    files are overwritten in place and the disk cost is one panel, not one per check.
    """
    from candi.bench.external import score_external
    from candi.bench.harness import open_source
    from candi.store.reader import CorpusStore

    work = Path(work_dir)
    v_regime = write_v_only_regime(regime_path, work / f"regime.vsel.{chrom}.json")
    regime = load_regime(v_regime)
    targets = [str(t) for _, t in regime["eval_pairs"]]
    if any(t.startswith(B_PREFIX) for t in targets):        # belt and braces on an absolute ruling
        raise ScopeError(f"the derived selection regime still names B_ targets: "
                         f"{sorted(t for t in targets if t.startswith(B_PREFIX))}")
    cells, cix = cell_index(regime)
    if cells != cache.cells or list(regime["assays"]) != cache.marks:
        raise ValueError(f"{chrom}: the selection regime's index space is not the cache's "
                         f"({len(cells)} cells / {len(regime['assays'])} assays against "
                         f"{len(cache.cells)} / {len(cache.marks)})")
    with CorpusStore(regime["store"]) as corpus:
        tracks = _declared_tracks(regime, corpus)
        n_bins = int(corpus.n_bins(chrom))
    if n_bins != cache.n_bins:
        raise ValueError(f"{chrom}: store says {n_bins} bins, cache says {cache.n_bins}")

    pred_root = work / f"select_pred_{chrom}"
    emit.write_manifest(
        pred_root, version="0.1.0", generated_by="lavawizard.store_eic:selector",
        contributor_mode="loo", weights="in-training", clip=bool(clip),
        sparse_assays=[m for j, m in enumerate(cache.marks) if int(cache.mark_count[j]) <= 2],
        notes=("MID-TRAINING SELECTION ROOT, overwritten at every check — not a board root. V_ "
               "panel only (BENCHMARK_DESIGN.md §5); B_ is not read."))
    # ONE source, opened once and re-scored at every check, for `candi.train`'s reason: selection
    # compares epoch 6 against epoch 12 and that comparison is only paired if both read the same
    # positions. Restricted to THIS chromosome because the checkpoints are per-chromosome — chr20's
    # model has no opinion about chr21, and scoring it there would need chr21's own weights.
    source = open_source(store=str(v_regime), chroms=(chrom,))

    def select_fn(model, epoch: int, step: int) -> float:
        write_predictions(model, cix, cache, tracks, pred_root, chrom=chrom, n_bins=n_bins,
                          clip=clip, device=device, batch=batch, verbose=False)
        result = score_external(source, pred_root, seed=seed, progress=False)
        return _macro_value(result, select_key)

    # The source outlives every check by design, so its handles are the caller's to release.
    select_fn.close = getattr(source, "close", lambda: None)

    info = {"metric": select_key, "regime": str(v_regime), "pred_root": str(pred_root),
            "pairs": len(regime["eval_pairs"]), "tracks": len(tracks),
            "b_pairs": sum(t.startswith(B_PREFIX) for t in targets), "chroms": [chrom]}
    if verbose:
        print(f"{chrom}: selection on {select_key} over {info['pairs']} V_ pairs / "
              f"{info['tracks']} tracks, scored by candi.bench.external on {chrom} alone",
              flush=True)
    return select_fn, info


def train_chrom(regime_path: Path | str, chrom: str, cache_root: Path | str, out_dir: Path | str,
                *, select_every: int = 0, select_key: str = SELECT_KEY, clip: bool = True,
                early_stop_epochs: int = 0, device: str = "cuda", seed: int = 0,
                select_device: str = "", **kw) -> dict:
    """`train.train_chromosome` with §5's selection loop attached. `select_every=0` turns it off.

    This is the entry point a board run uses. `train.py` cannot build the selector itself — scoring
    reads the store and the store is `candi`'s — so the wiring lives here, on the one side of the
    split that is allowed to import it.
    """
    from .train import train_chromosome

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    select_fn = None
    info: dict = {}
    if select_every:
        cache = preprocess.CachedChrom(cache_root, chrom, mmap=True)
        select_fn, info = selector(regime_path, chrom, cache, out_dir, select_key=select_key,
                                   clip=clip, device=(select_device or device), seed=seed)
        (out_dir / f"selection_{chrom}.json").write_text(json.dumps(info, indent=1) + "\n",
                                                         encoding="utf-8")
    try:
        record = train_chromosome(Path(cache_root), chrom, out_dir, device=device, seed=seed,
                                  select_fn=select_fn, select_every=select_every,
                                  select_metric=(select_key if select_fn else ""),
                                  early_stop_epochs=early_stop_epochs, **kw)
    finally:
        if select_fn is not None:
            select_fn.close()
    if info:
        record["selection"].update(panel=info)
        (out_dir / f"train_{chrom}.json").write_text(json.dumps(record, indent=1) + "\n",
                                                     encoding="utf-8")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache", help="build one chromosome's cache from the store")
    c.add_argument("--regime", required=True)
    c.add_argument("--chrom", required=True)
    c.add_argument("--cache", type=Path, required=True)
    c.add_argument("--chunk-bins", type=int, default=CHUNK_BINS)

    q = sub.add_parser("predict", help="write one chromosome's declared §4.1 tracks")
    q.add_argument("--regime", required=True)
    q.add_argument("--chrom", required=True)
    q.add_argument("--cache", type=Path, required=True)
    q.add_argument("--checkpoint", type=Path, required=True)
    q.add_argument("--pred-root", type=Path, required=True)
    q.add_argument("--device", default="cpu")
    q.add_argument("--clip", action="store_true",
                   help="cap signal_mu at the mark's training max on this chromosome (PI ruling "
                        "2026-08-26); off reproduces the faithful port")
    q.add_argument("--manifest", action="store_true",
                   help="also write manifest.json — pass on one task only, not all 23")

    t = sub.add_parser("train", help="train one chromosome, selecting the checkpoint on V_ (§5)")
    t.add_argument("--regime", required=True)
    t.add_argument("--chrom", required=True)
    t.add_argument("--cache", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--contributor-mode", default="loo", choices=("upstream", "loo"))
    t.add_argument("--device", default="cuda")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--epoch-scale", type=float, default=1.0)
    t.add_argument("--max-steps-per-stage", type=int, default=None)
    # 50 EPOCHS, and the default is sized rather than picked. Measured per check on chr20 — the
    # largest eval chromosome, 45 V_ tracks x 2,577,766 bins:
    #
    #   forward      ~3.1 min   45 x 2.58 M bins at the port's own measured 16.1 ms / 10,000-bin
    #                           forward on a 1g.10gb MIG slice (see `train.enable_tf32`)
    #   npz write     0.3 min   0.4 GiB, overwritten in place at every check
    #   metrics       2.3 min   `harness.score_track`'s pval arm, timed on real shapes
    #   truth reads   not measurable off-cluster — the CANDI_STORE reads, and the reason CANDI's
    #                 full-coverage V_ pass costs 91 min for three chromosomes
    #
    # So ~6 min of compute plus the store's reads, against ~4.5 GPU-h of stage 2 on chr20. Every 50
    # epochs is 16 checks. `train_<chrom>.json` records `stage2.eval_seconds` beside
    # `stage2.seconds`, so the first real run says for itself whether the cadence held to the ~20 %
    # of walltime `cruxvault/results/t30/TIMING.md` holds CANDI's to — and this is a flag because
    # that is the number to move if it did not.
    t.add_argument("--select-every", type=int, default=50,
                   help="epochs between selection checks on V_; 0 selects nothing at all")
    t.add_argument("--select-key", default=SELECT_KEY,
                   help="`<arm>:<key>` into score_external's macro block; lower is better")
    t.add_argument("--select-device", default="",
                   help="device for the selection forward pass (default: --device)")
    t.add_argument("--early-stop-epochs", type=int, default=0,
                   help="stop when the metric has not improved for MORE than this many epochs; "
                        "0 runs the whole schedule. The best weights are already on disk either "
                        "way — they are written the moment the metric improves.")
    t.add_argument("--no-clip", action="store_true",
                   help="predict without the per-mark cap during selection; the cap is on by "
                        "default because it is on in the run this selects a checkpoint for")

    ns = p.parse_args(argv)
    if ns.cmd == "cache":
        build_cache_from_store(ns.regime, ns.chrom, ns.cache, chunk_bins=ns.chunk_bins)
        return 0

    if ns.cmd == "train":
        train_chrom(ns.regime, ns.chrom, ns.cache, ns.out,
                    select_every=ns.select_every, select_key=ns.select_key,
                    clip=not ns.no_clip, early_stop_epochs=ns.early_stop_epochs,
                    device=ns.device, seed=ns.seed, select_device=ns.select_device,
                    contributor_mode=ns.contributor_mode, epoch_scale=ns.epoch_scale,
                    max_steps_per_stage=ns.max_steps_per_stage)
        return 0

    predict_chrom(ns.regime, ns.chrom, ns.cache, ns.checkpoint, ns.pred_root,
                  clip=ns.clip, device=ns.device)
    if ns.manifest:
        cache = preprocess.CachedChrom(ns.cache, ns.chrom, mmap=True)
        sparse = [m for j, m in enumerate(cache.marks) if int(cache.mark_count[j]) <= 2]
        emit.write_manifest(
            ns.pred_root, version="0.1.0", generated_by="lavawizard.store_eic",
            contributor_mode="loo", weights=f"ported-retrain:{Path(ns.checkpoint).name}",
            clip=bool(ns.clip), sparse_assays=sparse,
            notes=("Retrained on our EIC store, training-split biosamples only (§6.2). "
                   "Point-only pval arm: sigma comes from the §6.1 table, not from this root."))
    return 0


if __name__ == "__main__":              # run as `python -m lavawizard.store_eic`
    raise SystemExit(main())
