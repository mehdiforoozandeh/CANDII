"""The naive baseline suite (`RIVALS_PLAN.md` §5): a leave-one-out average, written as §4.1 roots.

    python -m competitors.baselines.generate --store configs/regime.eic_val.json \\
        --out /project/.../preds [--chroms chr21] [--methods avg,avg-arcsinh,knn1,knn5,marginal]

Six roots come out of one pass over the store, because they all read the same contributor blocks:

| root           | signal_mu            | signal_sigma        | mu, n                   | peak_score   |
|----------------|----------------------|---------------------|-------------------------|--------------|
| `avg`          | plain mean (§5.2)    | cross-cell std      | moment-matched NB (§5.1)| fraction     |
| `avg-arcsinh`  | sinh(mean(arcsinh))  | —                   | —                       | —            |
| `knn1`         | the top-1 cell       | —                   | Poisson at that cell    | that cell    |
| `knn5`         | mean over top 5      | std over top 5      | moment-matched NB       | fraction     |
| `marginal`     | per-assay constant   | per-assay constant  | per-assay constant NB   | —            |

`avg` is the baseline of record. Scoring is not this module's job: every root goes through
`python -m candi.bench.external` exactly as a rival's would.

HOW MANY TIMES EACH OF THEM RUNS (D1, 2026-09-01)
-------------------------------------------------
`avg` and `avg-arcsinh` are generated ONCE and their one root is printed in both regime rows;
`knn1`, `knn5` and `marginal` are generated once per regime, because their fit reads the regime's
training chromosomes. `REGIME_INDEPENDENT` / `REGIME_DEPENDENT` below carry the reasoning, and

    python -m competitors.baselines.generate --store <regime A> --out <root> \\
        --methods avg,avg-arcsinh --assert-regime-independent <regime B>

is the assertion that licenses the collapse: it re-predicts under B and compares every array.

WHAT THIS FILE IS ALLOWED TO SEE (§6.2, and it is checkable)
------------------------------------------------------------
`_contributors` below is where the fairness rule lives, and it is the ONLY place a contributor set
is built. It starts from `regime.biosamples["train"]` — never `eval` — and then drops every training
cell that shares the target's cell-type suffix, which is the same leave-one-out mask
`harness._apply_loo_mask` applies to CANDI's own input for a declared pair. `T_K562` therefore
contributes nothing to `V_K562`, for any assay. The kNN similarity table (§5.4) is built on the
regime's TRAIN chromosomes only and correlates against the INPUT cell, which is a training cell; no
eval-chromosome bin and no target-cell bin reaches it. The per-assay marginal is fitted on training
cells over training chromosomes for the same reason.

`n_contributors` is written per track into the manifest, and a track with `k <= 2` is listed under
`sparse_assays` — §5 requires that flag to travel into any table that quotes such a row. A track with
`k = 0` is skipped and listed under `skipped_tracks`; scoring it would need `--allow-missing`.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
The exact empirical-ensemble marginal (§5.4 "Follow-up") — it needs an ensemble-CRPS extension to
`candi.bench` and is a separate task. Nothing in this file anticipates it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from candi.store.reader import CorpusStore
from candi.store.regime import Regime

from competitors.baselines import heads as Hd

__all__ = [
    "METHODS", "VERSION", "REGIME_INDEPENDENT", "REGIME_DEPENDENT", "cell_type", "available",
    "log2_depth", "depth_center", "Panel", "RegimeIdentityError", "assert_regime_independent",
    "generate", "build_parser", "main",
]

VERSION = "0.1.0"

#: Every root this module can write. `avg` is the baseline of record (§5.2: "the row that must exist
#: for comparability"); the rest are variants and weaker tiers, each labelled as such in its manifest.
METHODS: Tuple[str, ...] = ("avg", "avg-arcsinh", "knn1", "knn5", "marginal")

#: WHICH METHODS COLLAPSE TO ONE RUN ACROSS REGIMES, AND WHICH DO NOT (D1, 2026-09-01).
#:
#: `BENCHMARK_DESIGN.md` §12.2 first ruled that all five naive baselines run ONCE rather than once
#: per regime, "because their fit is regime-independent". Read against the code that is true of two
#: of them and false of three:
#:
#:   avg, avg-arcsinh   every written bin is a function of the contributors' values AT THE PREDICTED
#:                      POSITION plus the contributor set, and the contributor set is
#:                      `biosamples.train` minus the target's cell type. No training LOCUS enters,
#:                      so `train_chroms` and `regions` cannot move the output. Collapse: correct.
#:   knn1, knn5         `similarity_table` correlates over `panel.train_chroms`. Different train
#:                      chromosomes -> a different ranking -> different predictions.
#:   marginal           `fit_marginal` pools over `panel.train_chroms`.
#:
#: So the collapse holds for `REGIME_INDEPENDENT` and the three in `REGIME_DEPENDENT` are generated
#: once per regime. §12.2 asks for an ASSERTION rather than an argument, and
#: `assert_regime_independent` below is it: the two collapsed roots are re-predicted under the other
#: regime and every array is compared.
REGIME_INDEPENDENT: Tuple[str, ...] = ("avg", "avg-arcsinh")
REGIME_DEPENDENT: Tuple[str, ...] = ("knn1", "knn5", "marginal")

#: `store/dataset.py::_RUN_TYPE_ID`. A track whose run type is neither of these is NOT available, and
#: guessing at single-ended is the fallback D19 deleted.
_RUN_TYPES = ("single-ended", "paired-ended")

#: The `T_`/`V_`/`B_` split prefixes. §5's exclusion rule is stated in terms of the CELL-TYPE SUFFIX,
#: so this is the one place in the repo that takes a biosample name apart, and it does so because the
#: plan says to — not to learn anything else about the cell.
_SPLIT_PREFIXES = ("T_", "V_", "B_")


def cell_type(biosample: str) -> str:
    """`T_K562` -> `K562`. The exclusion key of §5, and nothing else is derived from a name."""
    for p in _SPLIT_PREFIXES:
        if biosample.startswith(p):
            return biosample[len(p):]
    return biosample


# ---------------------------------------------------------------------------
# availability and depth — the same rule the loader applies (D19)
# ---------------------------------------------------------------------------

def available(corpus: CorpusStore, biosample: str, assay: str) -> bool:
    """Is this track present AND fully described? `store/dataset.py::_build_meta`'s rule, copied.

    A column is available only when the store holds it and the manifest gives depth, read length and
    run type. A partially-known track is MISSING to the loader, so `harness.targets` will never
    declare it a target and it must not be averaged into a baseline either — otherwise the two sides
    disagree about which panel is being scored.
    """
    bs = corpus[biosample] if biosample in corpus else None
    if bs is None or not bs.has(assay, "counts"):
        return False
    rec = corpus.track_meta(biosample, assay) or {}
    d, rl, rt = rec.get("depth"), rec.get("read_length"), rec.get("run_type")
    if d is None or rl is None or str(rt) not in _RUN_TYPES:
        return False
    try:
        return float(d) > 0 and float(rl) > 0
    except (TypeError, ValueError):
        return False


def log2_depth(corpus: CorpusStore, biosample: str, assay: str) -> float:
    """`log2(depth)` at DSF1 — the same number `meta_dsf1[0]` carries on the h5 path."""
    return math.log2(float((corpus.track_meta(biosample, assay) or {})["depth"]))


def depth_center(corpus: CorpusStore, train_biosamples: Sequence[str],
                 assays: Sequence[str]) -> float:
    """`store/dataset.py::StoreDataset.depth_center` — median `log2(depth)` over the TRAIN pool.

    Re-derived rather than obtained by constructing a `StoreDataset`, which would build the whole
    window plan to read one number. The end-to-end test asserts the two agree; and the value cancels
    out of every baseline anyway (see `heads.py`), so it is recorded for traceability rather than
    relied on.
    """
    vals = [log2_depth(corpus, b, a) for b in train_biosamples for a in assays
            if available(corpus, b, a)]
    if not vals:
        raise ValueError(f"no available column of {list(train_biosamples)[:3]}… carries a depth")
    return float(np.median(vals))


# ---------------------------------------------------------------------------
# the panel: pairs, targets, contributors
# ---------------------------------------------------------------------------

class Panel:
    """The regime resolved against the store: which tracks to write, and who may contribute.

    Deliberately NOT `bench.harness.EvalSource`. That class opens a `StoreDataset` to serve windowed
    training batches; a generator wants whole chromosomes and the declared track list, so it reads
    the regime and the corpus directly and re-derives the two rules that must match — availability
    (D19) and the target rule (`targets` below). Both are asserted against the harness in
    `tests/test_baselines.py`, which is what keeps the copy honest.
    """

    def __init__(self, regime_path: Path | str, *, chroms: Optional[Sequence[str]] = None):
        self.regime_path = Path(regime_path)
        self.regime = Regime.from_file(self.regime_path)
        self.corpus = CorpusStore(self.regime.store)
        self.assays: List[str] = list(self.regime.assays)
        self.train: List[str] = [b for b in self.regime.biosamples("train")]
        self.pairs: List[Tuple[str, str]] = [tuple(p) for p in self.regime.eval_pairs]
        want = list(chroms) if chroms else list(self.regime.eval_chroms)
        nb = self.corpus.n_bins()
        missing = [c for c in want if c not in nb]
        if missing:
            raise ValueError(f"{self.corpus.root} has no {missing}; it carries {sorted(nb)}")
        self.chroms: List[str] = want
        self.n_bins: Dict[str, int] = {c: int(nb[c]) for c in want}
        self.train_chroms: List[str] = [c for c in self.regime.train_chroms if c in nb]
        self.depth_center = depth_center(self.corpus, self.train, self.assays)

    def close(self) -> None:
        self.corpus.close()

    def targets(self, pair: Tuple[str, str]) -> List[str]:
        """Assays the TARGET cell has and the INPUT cell does not — `harness.StoreSource.targets`.

        The same rule, spelled the same way: an assay neither cell has has no truth, and an assay
        both cells have can be read straight off the prompt, so neither is imputation.
        """
        x, y = pair
        return [a for a in self.assays
                if available(self.corpus, y, a) and not available(self.corpus, x, a)]

    def contributors(self, pair: Tuple[str, str], assay: str) -> List[str]:
        """§5's exclusion rule, and the only place a contributor set is built.

        Training cells carrying `assay`, minus every biosample sharing the target's cell-type suffix.
        The input cell's suffix is dropped too: under the regime's `T_X -> V_X` declaration it is the
        same suffix, and taking the union means a regime that ever declares a cross-cell-type pair
        cannot leak the prompt cell's own copy of the answer through the average.
        """
        x, y = pair
        banned = {cell_type(x), cell_type(y)}
        return [b for b in self.train
                if cell_type(b) not in banned and available(self.corpus, b, assay)]


# ---------------------------------------------------------------------------
# §5.4 — the kNN similarity table
# ---------------------------------------------------------------------------
#
# SPEC READING, STATED OUT LOUD. §5.4 says to "rank contributors by Pearson correlation with the
# *input* cell's track of that assay". A declared pair exists precisely BECAUSE the input cell lacks
# the assay being imputed (`Panel.targets` above), so there is no input-cell track of that assay to
# correlate against and the sentence cannot be read literally. The reading implemented here is the
# one that keeps every other constraint in the paragraph — training chromosomes only, arcsinh(-log10
# p) space, no target-cell bins, k = 1 is BestSingle:
#
#     similarity(input cell, contributor) = mean over the assays BOTH cells carry of the Pearson
#     correlation of their arcsinh(-log10 p) tracks on the regime's train chromosomes.
#
# It is a CELL-level similarity — which is what BestSingle means in this literature — computed from
# training cells on training chromosomes, and it is the same table for every assay of a given pair.

def similarity_table(panel: Panel, cells: Sequence[str],
                     progress: bool = False) -> Dict[Tuple[str, str], float]:
    """`{(cell_a, cell_b): mean Pearson r}` over the assays both carry, ONE table for every pair.

    Assay-outer and computed as a Gram matrix: z-score each cell's `arcsinh(-log10 p)` over the
    train chromosomes, stack the cells that carry the assay into `[h, N]`, and `Z @ Z.T / N` is
    every pairwise correlation for that assay at once. Pair-outer with `np.corrcoef` would re-read
    the same 51 cells for each of the 26 declared pairs — the same track read 26 times.

    Only `panel.train_chroms` and only training cells are ever read here (§5.4).
    """
    tot: Dict[Tuple[str, str], float] = {}
    cnt: Dict[Tuple[str, str], int] = {}
    cells = list(cells)
    for assay in panel.assays:
        have = [b for b in cells if available(panel.corpus, b, assay)]
        if len(have) < 2:
            continue
        rows = []
        for b in have:
            t = np.arcsinh(np.concatenate(
                [panel.corpus[b][assay].pval(c, 0) for c in panel.train_chroms]).astype(np.float64))
            sd = t.std()
            rows.append((t - t.mean()) / sd if sd > 0 else np.zeros_like(t))
        Z = np.stack(rows)
        R = (Z @ Z.T) / Z.shape[1]
        for i, bi in enumerate(have):
            for j, bj in enumerate(have):
                if i >= j:
                    continue
                key = (bi, bj)
                tot[key] = tot.get(key, 0.0) + float(R[i, j])
                cnt[key] = cnt.get(key, 0) + 1
        if progress:
            print(f"[baselines] similarity: {assay} over {len(have)} cells", flush=True)
    return {k: tot[k] / cnt[k] for k in tot}


def top_k(sim: Mapping[Tuple[str, str], float], input_cell: str,
          contributors: Sequence[str], k: int) -> List[str]:
    """The `k` contributors most similar to the input cell.

    Ties and unmeasurable similarities break on the biosample name, so the selection is
    deterministic and does not depend on dictionary or directory order.
    """
    def r(b: str) -> float:
        key = (input_cell, b) if input_cell < b else (b, input_cell)
        return sim.get(key, float("-inf"))
    return sorted(contributors, key=lambda b: (-r(b), b))[:k]


# ---------------------------------------------------------------------------
# §5.4 — the per-assay marginal, fitted on training cells x training chromosomes
# ---------------------------------------------------------------------------

def fit_marginal(panel: Panel, assay: str) -> Optional[Dict[str, float]]:
    """`{count_mean, count_var, pval_mean, pval_std}` at `depth_center`, or None with no data.

    §5.4 fits ONE NB per assay by moment matching "over all bins of the training cells' tracks on the
    training chromosomes". Two readings had to be settled here and both are recorded in the manifest:

    * **Depth.** A count marginal pooled over cells at different exposures is a depth mixture, and
      §5.1 already fixed the convention, so every contributing cell is rescaled to `depth_center`
      first and the fitted moments are then scaled to the TARGET track's depth exactly as the average
      is. The written array is still one constant per track — the constant just knows the exposure it
      is predicting at, which is the difference between the weakest baseline and a broken one.
    * **A pval marginal too.** §5.4 names only the NB, but §5.5's first sanity anchor compares the
      plain-mean pval baseline against "the per-assay marginal on macro mse", which is a pval-arm
      comparison and has nothing to compare against unless this tier carries a pval arm. It is fitted
      the same way — mean and std of `-log10 p` over the same pool — and is the Gaussian analogue of
      the NB, not a new idea.

    No leave-one-out here, and §5.4 does not ask for one: the pool is training cells on training
    chromosomes, and the target cell (a `V_`/`B_`) is in neither.
    """
    cells = [b for b in panel.train if available(panel.corpus, b, assay)]
    if not cells or not panel.train_chroms:
        return None
    cs: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    for b in cells:
        scale = Hd.depth_scale(log2_depth(panel.corpus, b, assay), panel.depth_center)
        for c in panel.train_chroms:
            cs.append(panel.corpus[b][assay].counts(c, 0).astype(np.float64) * scale)
            ps.append(panel.corpus[b][assay].pval(c, 0).astype(np.float64))
    cc, pp = np.concatenate(cs), np.concatenate(ps)
    return {"count_mean": float(cc.mean()), "count_var": float(cc.var(ddof=1)),
            "pval_mean": float(pp.mean()), "pval_std": float(pp.std(ddof=1)),
            "n_cells": len(cells), "n_bins": int(cc.size)}


# ---------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------

def _write(root: Path, dirname: str, chrom: str, arrays: Mapping[str, np.ndarray]) -> None:
    d = root / dirname
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{chrom}.npz",
             **{k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()})


def _manifest(method: str, panel: Panel, notes: str, arms: Sequence[str],
              tracks: Mapping[str, Dict[str, object]], skipped: Sequence[str],
              poisson_n: float) -> Dict[str, object]:
    # `n_eligible`, not `n_contributors`. The flag §5 asks for is "this assay is thin in the
    # training split", which is a property of the PANEL. `knn1` uses one contributor on purpose out
    # of however many were eligible, and flagging every BestSingle row as sparse would make the flag
    # mean nothing exactly where it is supposed to mean something.
    sparse = sorted(k for k, v in tracks.items() if int(v["n_eligible"]) <= 2)
    return {
        "method": method,
        "version": VERSION,
        "generated_by": "competitors/baselines/generate.py",
        "date": date.today().isoformat(),
        "arms": list(arms),
        "notes": notes,
        "regime": str(panel.regime_path),
        "store": str(panel.corpus.root),
        "chroms": list(panel.chroms),
        "depth_center": panel.depth_center,
        # §5.1's Poisson floor, on the row, because `candi.metrics.nb_crps` cannot evaluate the
        # pre-registered 1e6 (see `heads.POISSON_N`). A leaderboard reader has to be able to tell
        # which value a count-arm CRPS was — or was not — computed at.
        "poisson_n": float(poisson_n),
        "poisson_n_is_preregistered": bool(poisson_n == Hd.POISSON_N),
        "n_train_biosamples": len(panel.train),
        "tracks": dict(tracks),
        # §5 — a track averaged over <= 2 cells must carry that flag into any table that quotes it.
        "sparse_assays": sparse,
        "skipped_tracks": list(skipped),
        "exclusion_rule": "training cells carrying the assay, minus every biosample sharing the "
                          "input's or the target's cell-type suffix (RIVALS_PLAN.md §5)",
    }


#: Fields that describe WHAT the root is. Two runs writing into one root must agree on all of them,
#: or the root holds two different methods' output under one name.
_MANIFEST_IDENTITY = ("method", "version", "regime", "store", "depth_center", "poisson_n",
                      "arms", "exclusion_rule")


def _merge_manifest(old: Mapping[str, object], new: Dict[str, object],
                    path: Path) -> Dict[str, object]:
    """Fold a per-chromosome run into a root another run already started (the P2 array shape).

    P2 is one SLURM task per chromosome writing into shared roots, so the second task must not
    overwrite the first task's record of what is on disk. `chroms` and `tracks` are unioned;
    everything in `_MANIFEST_IDENTITY` must match, and a mismatch raises rather than being merged —
    a root holding chr1 at one Poisson floor and chr2 at another is not one prediction track.

    THE RACE IS REAL AND IS BOUNDED. Two tasks finishing within the same instant can lose one
    chromosome name from `chroms`; the npz files themselves are per (track, chromosome) and never
    collide, and `bench.external` reads the grid off the store rather than off this field, so the
    worst case is an understated provenance line, not a mis-scored track. Check `chroms` against the
    directory listing before quoting a P2 row.
    """
    bad = {k: (old.get(k), new.get(k)) for k in _MANIFEST_IDENTITY if old.get(k) != new.get(k)}
    if bad:
        raise ValueError(
            f"{path} already describes a different run: {bad}. Write to a fresh --out, or delete "
            f"the root — merging two settings into one manifest would make every row it labels "
            f"unattributable.")
    merged = dict(new)
    merged["chroms"] = sorted(set(old.get("chroms", [])) | set(new.get("chroms", [])),
                              key=lambda c: (len(c), c))
    merged["tracks"] = {**old.get("tracks", {}), **new.get("tracks", {})}
    merged["skipped_tracks"] = sorted(set(old.get("skipped_tracks", []))
                                      | set(new.get("skipped_tracks", [])))
    merged["sparse_assays"] = sorted(k for k, v in merged["tracks"].items()
                                     if int(v["n_eligible"]) <= 2)
    return merged


# ---------------------------------------------------------------------------
# §12.2's identity assertion — the thing that licenses the collapse (D1)
# ---------------------------------------------------------------------------

class RegimeIdentityError(Exception):
    """A method that was claimed regime-independent wrote a different array under the other regime.

    Raised by `assert_regime_independent`; the CLI turns it into exit code 5. It is never a reason
    to relabel the method — it is a reason to run that method once per regime.
    """


def assert_regime_independent(roots: Mapping[str, Path], panel: Panel, regime_b: Path | str, *,
                              poisson_n: float = Hd.POISSON_N,
                              progress: bool = False) -> Dict[str, object]:
    """Re-predict `roots`' methods under `regime_b` and require every array to be identical.

    This is the assertion `BENCHMARK_DESIGN.md` §12.2 claims exists and did not: one run of the
    collapsed methods is printed in BOTH regime rows, and nothing but a comparison licenses that.

    IT COMPARES ARRAYS, NOT MANIFESTS. `_MANIFEST_IDENTITY` includes `regime`, so two manifests from
    two regime files never match and never could; what has to match is what a scorer reads, which is
    the npz payload. One chromosome is enough and all that is affordable — `panel.chroms[0]`, the
    first chromosome this pass actually wrote — because the collapsed heads are per-bin functions of
    the contributors at that bin, so a regime that moved any bin would move every chromosome.

    The two regimes must declare the SAME `eval_pairs`: identical predictions for different panels
    would be an identity between two different objects, which licenses nothing.

    Returns the stamp `generate` writes into each manifest. Raises `RegimeIdentityError` listing the
    first differences on any mismatch, and that is the correct outcome for a regime-DEPENDENT method
    — `tests/test_baselines.py` runs it on `marginal` for exactly that reason.
    """
    b_path = Path(regime_b)
    b_pairs = [tuple(p) for p in Regime.from_file(b_path).eval_pairs]
    if b_pairs != panel.pairs:
        raise ValueError(
            f"{b_path} declares {len(b_pairs)} eval pair(s) and {panel.regime_path} declares "
            f"{len(panel.pairs)}; the two must declare the SAME panel or an identity between their "
            f"outputs says nothing about the collapse.")
    chrom = panel.chroms[0]
    diffs: List[str] = []
    with tempfile.TemporaryDirectory(prefix="baselines-regime-assert-") as tmp:
        other = generate(b_path, tmp, chroms=[chrom], methods=list(roots),
                         poisson_n=poisson_n, progress=progress)
        for m, a_root in roots.items():
            a_root, b_root = Path(a_root), Path(other[m])
            a_dirs = {d.name for d in a_root.iterdir() if (d / f"{chrom}.npz").is_file()}
            b_dirs = {d.name for d in b_root.iterdir() if (d / f"{chrom}.npz").is_file()}
            if a_dirs != b_dirs:
                diffs.append(f"{m}: track sets differ on {chrom} — only under "
                             f"{panel.regime_path.name}: {sorted(a_dirs - b_dirs)[:3]}; only under "
                             f"{b_path.name}: {sorted(b_dirs - a_dirs)[:3]}")
                continue
            for dirname in sorted(a_dirs):
                with np.load(a_root / dirname / f"{chrom}.npz") as A, \
                        np.load(b_root / dirname / f"{chrom}.npz") as B:
                    if set(A.files) != set(B.files):
                        diffs.append(f"{m}/{dirname}: array names differ — "
                                     f"{sorted(A.files)} vs {sorted(B.files)}")
                        continue
                    for k in sorted(A.files):
                        if not np.array_equal(A[k], B[k]):
                            n = int(np.sum(A[k] != B[k]))
                            diffs.append(f"{m}/{dirname}/{chrom}.npz[{k}]: {n} bin(s) differ")
    if diffs:
        raise RegimeIdentityError(
            f"{len(diffs)} difference(s) between {panel.regime_path.name} and {b_path.name} on "
            f"{chrom}: {diffs[:5]}. These method(s) are NOT regime-independent and must be "
            f"generated once per regime (BENCHMARK_DESIGN.md §12.2, D1).")
    return {"asserted_against": b_path.name, "chrom": chrom, "identical": True}


def generate(regime_path: Path | str, out_root: Path | str, *,
             chroms: Optional[Sequence[str]] = None, methods: Sequence[str] = METHODS,
             poisson_n: float = Hd.POISSON_N, progress: bool = True,
             assert_against: Optional[Path | str] = None) -> Dict[str, Path]:
    """Write one §4.1 prediction root per method. Returns `{method: root}`.

    The loop is chromosome -> assay -> the pairs that need it, so each contributor's track is read
    ONCE per (assay, chromosome) and serves every target that assay has. Reading pair-first would
    re-read the same 51 cells for every pair.

    `assert_against` names a SECOND regime file. With it set, the same methods are re-predicted
    under that regime for the first chromosome of this pass and every array is compared; each
    manifest then carries `regime_independent`. See `assert_regime_independent`.
    """
    bad = [m for m in methods if m not in METHODS]
    if bad:
        raise ValueError(f"unknown method(s) {bad}; this module writes {list(METHODS)}")
    panel = Panel(regime_path, chroms=chroms)
    out_root = Path(out_root)
    roots = {m: out_root / m for m in methods}
    for r in roots.values():
        r.mkdir(parents=True, exist_ok=True)

    # {method: {dirname: {n_contributors, ...}}}
    tracks: Dict[str, Dict[str, Dict[str, object]]] = {m: {} for m in methods}
    skipped: List[str] = []
    marginals: Dict[str, Optional[Dict[str, float]]] = {}
    sim: Dict[Tuple[str, str], float] = {}
    t0 = time.time()

    # Which (pair, assay) tracks exist at all, grouped by assay.
    by_assay: Dict[str, List[Tuple[Tuple[str, str], List[str]]]] = {}
    for pair in panel.pairs:
        for a in panel.targets(pair):
            contribs = panel.contributors(pair, a)
            dirname = f"{pair[0]}__{pair[1]}__{a}"
            if not contribs:
                skipped.append(dirname)
                continue
            by_assay.setdefault(a, []).append((pair, contribs))

    if any(m in ("knn1", "knn5") for m in methods):
        sim = similarity_table(panel, panel.train, progress=progress)
        if progress:
            print(f"[baselines] similarity table: {len(sim)} cell pairs "
                  f"({time.time() - t0:.0f}s)", flush=True)

    for chrom in panel.chroms:
        for assay, rows in sorted(by_assay.items()):
            if "marginal" in methods and assay not in marginals:
                marginals[assay] = fit_marginal(panel, assay)
            cells = sorted({b for _, cs in rows for b in cs})
            idx = {b: i for i, b in enumerate(cells)}
            # float32 for the two blocks that are only ever READ per track, float64 for the count
            # block the moments are taken over. On chr1 (9.96 M bins) a 34-cell panel is 2.7 GB per
            # float64 block, so this is the difference between a 16 GB job and a 32 GB one; `counts`
            # is dropped the moment `norm` exists for the same reason.
            counts = np.stack([panel.corpus[b][assay].counts(chrom, 0).astype(np.float64)
                               for b in cells])
            depths = np.array([log2_depth(panel.corpus, b, assay) for b in cells])
            norm = Hd.normalize_to_center(counts, depths, panel.depth_center)
            del counts
            pvals = np.stack([panel.corpus[b][assay].pval(chrom, 0).astype(np.float32)
                              for b in cells])
            peaks = np.stack([panel.corpus[b][assay].peaks(chrom, 0).astype(np.float32)
                              for b in cells])

            for pair, contribs in rows:
                dirname = f"{pair[0]}__{pair[1]}__{assay}"
                d_t = log2_depth(panel.corpus, pair[1], assay)
                for m in methods:
                    pick = (contribs if m in ("avg", "marginal") else
                            top_k(sim, pair[0], contribs, 1 if m == "knn1" else 5))
                    sel = np.array([idx[b] for b in pick])
                    arrays = _arrays(m, norm[sel], pvals[sel], peaks[sel], d_t,
                                     panel.depth_center, marginals.get(assay),
                                     poisson_n=poisson_n)
                    if arrays is None:
                        continue
                    _write(roots[m], dirname, chrom, arrays)
                    row: Dict[str, object] = {"n_contributors": int(len(pick)),
                                              "n_eligible": int(len(contribs)),
                                              "target_log2_depth": float(d_t)}
                    if m.startswith("knn"):
                        # WHICH cells k=1 and k=5 actually chose. Without it a BestSingle row is
                        # unauditable: the whole method is the choice.
                        row["contributors"] = list(pick)
                    tracks[m][dirname] = row
            del pvals, peaks, norm
            if progress:
                print(f"[baselines] {chrom}/{assay}: {len(cells)} cells, {len(rows)} track(s) "
                      f"({time.time() - t0:.0f}s)", flush=True)

    notes = {
        "avg": "leave-one-out cross-cell average over training biosamples: plain-mean -log10 p "
               "(the EIC Average definition), cross-cell std, moment-matched NB counts, "
               "fraction-of-contributors peak score. RIVALS_PLAN.md §5.1-5.3.",
        "avg-arcsinh": "VARIANT, NOT THE EIC BASELINE (§5.2): the pval mean taken in arcsinh space "
                       "and sinh'd back. Point only — a mean in a transformed space carries no "
                       "cross-cell spread this file is willing to claim in -log10 p.",
        "knn1": "BestSingle (§5.4): the single most similar training cell, ranked by mean Pearson r "
                "with the INPUT cell over their shared assays on the train chromosomes in "
                "arcsinh(-log10 p). One contributor, so counts take the Poisson floor and there is "
                "no cross-cell sigma.",
        "knn5": "kNN average, k=5 (§5.4). Same similarity ranking as knn1; heads as `avg` over the "
                "top 5 contributors.",
        "marginal": "per-assay marginal (§5.4), the weakest distributional tier: one constant NB and "
                    "one constant Gaussian per assay, moment-matched over the training cells' "
                    "tracks on the TRAIN chromosomes, scaled to the target track's depth. No peak "
                    "head — a marginal has no per-bin ranking to offer.",
    }
    arms = {"avg": ["pval", "count", "peak"], "avg-arcsinh": ["pval"],
            "knn1": ["pval", "count", "peak"], "knn5": ["pval", "count", "peak"],
            "marginal": ["pval", "count"]}
    # Before any manifest is written: a root that claims `regime_independent` must have earned it,
    # and a failure here must leave no stamped manifest behind to be quoted.
    stamp: Optional[Dict[str, object]] = None
    if assert_against is not None:
        stamp = assert_regime_independent(roots, panel, assert_against,
                                          poisson_n=poisson_n, progress=progress)
        if progress:
            print(f"[baselines] regime-independent against {stamp['asserted_against']} on "
                  f"{stamp['chrom']}: every array identical", flush=True)

    out: Dict[str, Path] = {}
    for m in methods:
        path = roots[m] / "manifest.json"
        obj = _manifest(m, panel, notes[m], arms[m], tracks[m], skipped, poisson_n)
        if stamp is not None:
            obj["regime_independent"] = dict(stamp)
        if path.exists():
            obj = _merge_manifest(json.loads(path.read_text(encoding="utf-8")), obj, path)
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        out[m] = roots[m]
        if progress:
            print(f"[baselines] {m}: {len(tracks[m])} tracks -> {roots[m]}", flush=True)
    if skipped and progress:
        print(f"[baselines] {len(skipped)} track(s) had NO eligible contributor and were skipped: "
              f"{skipped[:5]} — score with --allow-missing", flush=True)
    panel.close()
    return out


def _arrays(method: str, norm: np.ndarray, pvals: np.ndarray, peaks: np.ndarray,
            target_log2_depth: float, dc: float, marginal: Optional[Mapping[str, float]], *,
            poisson_n: float = Hd.POISSON_N) -> Optional[Dict[str, np.ndarray]]:
    """The arrays one method writes for one track on one chromosome, or None when it writes nothing.

    `norm`/`pvals`/`peaks` are `[k, N]` already restricted to this method's contributors.
    """
    if method == "avg-arcsinh":
        return {"signal_mu": Hd.arcsinh_mean(pvals)}

    if method == "marginal":
        if marginal is None:
            return None
        n_bins = pvals.shape[1]
        mu, n = Hd.nb_from_moments(np.full(n_bins, marginal["count_mean"]),
                                   np.full(n_bins, marginal["count_var"]),
                                   target_log2_depth, dc, poisson_n=poisson_n)
        return {"mu": mu, "n": n,
                "signal_mu": np.full(n_bins, marginal["pval_mean"]),
                "signal_sigma": np.full(n_bins, max(marginal["pval_std"], Hd.SIGMA_FLOOR))}

    out: Dict[str, np.ndarray] = {}
    mu, n = Hd.moment_matched_nb(norm, target_log2_depth, dc, poisson_n=poisson_n)
    out["mu"], out["n"] = mu, n
    out["signal_mu"] = Hd.plain_mean(pvals)
    sigma = Hd.cross_cell_sigma(pvals)
    if sigma is not None:
        out["signal_sigma"] = sigma
    out["peak_score"] = Hd.peak_fraction(peaks)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m competitors.baselines.generate",
        description="Write the naive baseline prediction roots (RIVALS_PLAN.md §5). Score them "
                    "with `python -m candi.bench.external`.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--store", required=True, help="a CANDI_STORE regime file (json)")
    p.add_argument("--out", required=True, help="directory to hold one prediction root per method")
    p.add_argument("--chroms", default=None,
                   help="comma-separated chromosomes (default: the regime's eval_chroms). P2 is "
                        "this flag with one chromosome per array task.")
    p.add_argument("--methods", default=",".join(METHODS),
                   help=f"comma-separated subset of {list(METHODS)}")
    p.add_argument("--poisson-n", type=float, default=Hd.POISSON_N,
                   help="§5.1's NB Poisson floor. The pre-registered value is 1e6 and is the "
                        "default; `candi.metrics.nb_crps` returns NaN above about 2e4, which costs "
                        "the count arm its whole CRPS tier. See heads.POISSON_N before changing it "
                        "— an amended value belongs in RIVALS_PLAN.md §5.1 first.")
    p.add_argument("--assert-regime-independent", default=None, metavar="REGIME_B",
                   help=f"a SECOND regime file. After writing under --store, re-predict the same "
                        f"methods under this one for the first chromosome of the pass and require "
                        f"every array to be identical (exit 5 otherwise); each manifest then "
                        f"carries `regime_independent`. Accepted only for {list(REGIME_INDEPENDENT)}"
                        f" — {list(REGIME_DEPENDENT)} fit on the regime's training loci and are "
                        f"generated once per regime (exit 2).")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    methods = [m.strip() for m in a.methods.split(",")]
    if a.assert_regime_independent is not None:
        dependent = [m for m in methods if m not in REGIME_INDEPENDENT]
        if dependent:
            print(f"[baselines] REFUSING --assert-regime-independent with {dependent}: those "
                  f"methods fit on the regime's training loci (similarity_table and fit_marginal "
                  f"both read train_chroms), so they are NOT regime-independent and asserting it "
                  f"would be asserting something false. Generate them once per regime "
                  f"(BENCHMARK_DESIGN.md §12.2, D1).", file=sys.stderr)
            return 2
    try:
        generate(a.store, a.out,
                 chroms=[c.strip() for c in a.chroms.split(",")] if a.chroms else None,
                 methods=methods, poisson_n=a.poisson_n, progress=not a.quiet,
                 assert_against=a.assert_regime_independent)
    except RegimeIdentityError as exc:
        print(f"[baselines] {exc}", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
