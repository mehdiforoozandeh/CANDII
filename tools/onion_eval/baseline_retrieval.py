"""BRt -- the training-free retrieval baseline (a reported row; no part of the done rule D14).

For a target (cell c, assay a) the prediction is the assay-a track of one training cell d != c,
the neighbor, scaled to the target's depth:

    mu = max(y_{d,a} * 2**(log2_depth C'_target - log2_depth_{d,a}), PRED_FLOOR),  n = size_a

The neighbor is chosen once per (cell, assay, input assays) from the fit regions only: for every
training cell d that measured assay a, the score is the mean, over the shared input assays b (the
target's available inputs other than a that d also measured), of the Pearson correlation of
log1p counts of (c, b) and (d, b) over the fit bins; the best score wins (ties: more shared
assays, then the cell name). The target cell itself is never a candidate. A candidate with no
finite correlation is skipped; a target with no candidate is NaN (undefined), like a baseline of
`baselines.py`. One NB size per assay is fitted as `baselines._fit_size` on the training tracks
of the fit bins: each training track (d, a) is predicted by the same rule from d's other tracks.

    Retrieval(read=None).fit(batches)   training tracks over the fit bins (y_counts, y_avail,
                                        c_out depth, meta cell / chrom / start_bin)
    Retrieval.predict(batch) -> {"n", "mu"}   float32 [B, L, A]; defined on y_kind 1 (imputation)
                                        entries only, NaN elsewhere (pairs, reconstruction)

`read(cell, track_index, chrom, start_bin, L) -> counts` gives the neighbor's track where the
prediction is asked for; without it the counts kept by `fit` are used (NaN outside the fit bins).
Correlations of assay b use the bins that every training cell with b covers.

    python tools/onion_eval/baseline_retrieval.py run --tables <dir> --evalset <json> --out <dir>
    -> <out>/scores.json, suffstats.npz, SCORED (score.py's writer and keys; `predictor` is
       `retrieval`), and <out>/retrieval_neighbors.tsv (one row per chosen neighbor).

Scoring route (agent decision): score.py's external entry (`signal_override`) is signal only and
writes every count-space metric undefined, so this tool uses score.py's in-memory model route
(`Scorer` with `model=`), with `predict_control` off and the copy-leak guardrail reported as for a
training-free baseline (`copy_leak_pass` None). Validation split only; score.py is not edited.
An existing <out>/SCORED skips the job, as score.py does.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import torch

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from onion_eval.baselines import DEPTH_COL, PRED_FLOOR, _fit_size, _np, _out, _shape  # noqa: E402

KIND_IMPUTATION = 1
NAME = "retrieval"
NEIGHBORS_TSV = "retrieval_neighbors.tsv"
NEIGHBOR_COLUMNS = ("use", "cell", "assay", "neighbor", "mean_pearson", "n_shared_assays",
                    "shared_assays")


class Retrieval:
    """Copy the target assay's track from the best-correlated training cell (module docstring)."""

    def __init__(self, read: Optional[Callable] = None, track_names=None):
        self.read = read
        self.track_names = track_names
        self.tracks: Dict[tuple, Dict[str, tuple]] = {}   # (cell, a) -> chrom -> (bins, counts)
        self.log2_depth: Dict[tuple, float] = {}          # (cell, a) -> log2 depth of the track
        self.corr: Dict[int, tuple] = {}                  # b -> (cells, Pearson matrix)
        self.size: Dict[int, float] = {}
        self.choices: Dict[tuple, tuple] = {}             # (cell, a, inputs) -> choice
        self.fit_choices: Dict[tuple, tuple] = {}         # (cell, a) -> choice, NB-size fit

    # ---------------------------------------------------------------- fit
    def fit(self, batches: Iterable[dict]) -> "Retrieval":
        acc = defaultdict(lambda: defaultdict(list))
        for batch in batches:
            y, avail = _np(batch["y_counts"]), _np(batch["y_avail"]).astype(bool)
            ld = _np(batch["c_out"])[:, :, DEPTH_COL].astype(np.float64)
            L = y.shape[1]
            for b, a in zip(*np.nonzero(avail)):
                m = batch["meta"][b]
                key = (m["cell"], int(a))
                bins = int(m["start_bin"]) + np.arange(L, dtype=np.int64)
                acc[key][m["chrom"]].append((bins, y[b, :, a].astype(np.float32)))
                d = float(ld[b, a])
                if key in self.log2_depth and abs(self.log2_depth[key] - d) > 1e-6:
                    raise ValueError(f"track {key}: two depths {self.log2_depth[key]} and {d}")
                self.log2_depth[key] = d
        for key, chroms in acc.items():
            self.tracks[key] = {}
            for chrom, parts in chroms.items():
                bins = np.concatenate([p[0] for p in parts])
                vals = np.concatenate([p[1] for p in parts])
                u, first = np.unique(bins, return_index=True)
                self.tracks[key][chrom] = (u, vals[first])
        acc.clear()
        for b in sorted({a for _, a in self.tracks}):
            self.corr[b] = self._corr_matrix(b)
        self._fit_sizes()
        return self

    def _corr_matrix(self, b: int) -> tuple:
        """Pearson of log1p counts between every pair of training cells with track b, over the
        bins that all of them cover."""
        cells = sorted(c for c, a in self.tracks if a == b)
        chroms = set.intersection(*(set(self.tracks[(c, b)]) for c in cells))
        cols = []
        for chrom in sorted(chroms):
            common = self.tracks[(cells[0], b)][chrom][0]
            for c in cells[1:]:
                common = np.intersect1d(common, self.tracks[(c, b)][chrom][0], assume_unique=True)
            if common.size:
                cols.append(np.stack([self._at(c, b, chrom, common) for c in cells]))
        R = np.full((len(cells), len(cells)), np.nan)
        if cols:
            X = np.log1p(np.concatenate(cols, axis=1).astype(np.float64))
            X -= X.mean(1, keepdims=True)
            sd = np.sqrt((X ** 2).mean(1))
            ok = sd > 0
            Z = np.zeros_like(X)
            Z[ok] = X[ok] / sd[ok, None]
            R = Z @ Z.T / X.shape[1]
            R[~ok, :] = np.nan
            R[:, ~ok] = np.nan
        return cells, R

    def _at(self, cell, a, chrom, bins) -> np.ndarray:
        """Stored counts of track (cell, a) at `bins`; NaN where the fit did not cover a bin."""
        out = np.full(bins.size, np.nan)
        t = self.tracks.get((cell, a), {}).get(chrom)
        if t is None:
            return out
        u, v = t
        idx = np.minimum(np.searchsorted(u, bins), u.size - 1)
        hit = u[idx] == bins
        out[hit] = v[idx[hit]]
        return out

    def neighbor(self, cell, a: int, inputs) -> tuple:
        """(neighbor cell or None, mean Pearson, shared assays) for target (cell, a)."""
        inputs = tuple(sorted({int(b) for b in inputs} - {int(a)}))
        best = (None, float("nan"), ())
        rank = None
        for d in sorted(c for c, aa in self.tracks if aa == a and c != cell):
            rs, shared = [], []
            for b in inputs:
                cells, R = self.corr.get(b, ((), None))
                if cell in cells and d in cells:
                    r = R[cells.index(cell), cells.index(d)]
                    if np.isfinite(r):
                        rs.append(float(r))
                        shared.append(b)
            if not rs:
                continue
            k = (float(np.mean(rs)), len(rs))
            if rank is None or k > rank:      # ties keep the earlier (sorted) cell name
                rank, best = k, (d, k[0], tuple(shared))
        return best

    def _fit_sizes(self) -> None:
        ms, ys = defaultdict(list), defaultdict(list)
        for (d, a) in sorted(self.tracks, key=lambda k: (k[1], str(k[0]))):
            inputs = [b for (c, b) in self.tracks if c == d and b != a]
            nb, r, shared = self.neighbor(d, a, inputs)
            self.fit_choices[(d, a)] = (nb, r, shared)
            if nb is None:
                continue
            scale = 2.0 ** (self.log2_depth[(d, a)] - self.log2_depth[(nb, a)])
            for chrom, (bins, y) in self.tracks[(d, a)].items():
                x = self._at(nb, a, chrom, bins)
                ok = np.isfinite(x)
                ms[a].append(np.maximum(x[ok] * scale, PRED_FLOOR))
                ys[a].append(y[ok].astype(np.float64))
        for a in ms:
            m = np.concatenate(ms[a])
            if m.size:
                self.size[a] = _fit_size(m, np.concatenate(ys[a]))

    # ---------------------------------------------------------------- predict
    def _neighbor_counts(self, d, a, chrom, start, L) -> np.ndarray:
        if self.read is not None:
            return np.asarray(self.read(d, a, chrom, start, L), np.float64)
        return self._at(d, a, chrom, int(start) + np.arange(L, dtype=np.int64))

    def predict(self, batch) -> Dict[str, torch.Tensor]:
        B, L, A = _shape(batch)
        n, mu = np.full((B, L, A), np.nan), np.full((B, L, A), np.nan)
        avail = _np(batch["y_avail"]).astype(bool)
        if "y_kind" in batch:
            avail &= _np(batch["y_kind"]) == KIND_IMPUTATION
        x_avail = _np(batch["x_avail"]).astype(bool)
        ld = _np(batch["c_out"])[:, :, DEPTH_COL].astype(np.float64)
        for b, a in zip(*np.nonzero(avail)):
            a = int(a)
            m = batch["meta"][b]
            inputs = tuple(int(i) for i in np.nonzero(x_avail[b])[0] if i != a)
            key = (m["cell"], a, inputs)
            if key not in self.choices:
                self.choices[key] = self.neighbor(m["cell"], a, inputs)
            d = self.choices[key][0]
            if d is None or a not in self.size:
                continue
            x = self._neighbor_counts(d, a, m["chrom"], int(m["start_bin"]), L)
            mu[b, :, a] = np.maximum(x * 2.0 ** (ld[b, a] - self.log2_depth[(d, a)]), PRED_FLOOR)
            n[b, :, a] = np.where(np.isfinite(x), self.size[a], np.nan)
        return _out(n, mu)

    def neighbor_rows(self) -> list:
        name = (lambda a: self.track_names[a]) if self.track_names else str
        rows = []
        for use, table in (("predict", self.choices), ("nb_size_fit", self.fit_choices)):
            for (c, a, *_), (d, r, shared) in sorted(table.items(), key=lambda kv: (
                    str(kv[0][0]), kv[0][1], kv[0][2:])):
                rows.append({"use": use, "cell": c, "assay": name(a),
                             "neighbor": "" if d is None else d,
                             "mean_pearson": "" if d is None else f"{r:.6g}",
                             "n_shared_assays": len(shared),
                             "shared_assays": ",".join(name(b) for b in shared)})
        return rows


# -------------------------------------------------------------------- scoring through score.py

class _AsModel:
    """What score.ModelPredictor calls: .to(device), .eval(), and batch -> {"n", "mu"}."""

    def __init__(self, retrieval: Retrieval):
        self.retrieval = retrieval

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, batch):
        return self.retrieval.predict(batch)


def _scorer_class():
    from onion_eval import score as S

    class RetrievalScorer(S.Scorer):
        """score.Scorer with the retrieval baseline as the main predictor: fitted on the
        training tracks' base views (`train_base`) over the evalset's fit regions."""

        def __init__(self, evalset, **kw):
            r = Retrieval()
            super().__init__(evalset, main=NAME, model=_AsModel(r), **kw)
            self.predict_control = False          # a baseline never predicts the control
            C, ds = self.C, self.ds
            r.track_names = C.TRACKS
            r.read = lambda d, a, chrom, s, L: ds.read(ds.train_base[(d, C.TRACKS[a])], chrom, s, L)
            self.retrieval = r
            by_assay = defaultdict(list)
            for (c, a), mid in sorted(ds.train_base.items()):
                by_assay[a].append(mid)

            def fit_batches():
                for c, s in self.fit_regions:
                    self._set_region(c, s)
                    for a in sorted(by_assay):
                        yield self._region_batch(by_assay[a], c, s)
            r.fit(fit_batches())
            self.log(f"retrieval: fitted on {len(r.tracks)} training tracks, "
                     f"{len(self.fit_regions)} fit regions; NB sizes for {len(r.size)} assays")

        def _copy_leak(self):
            model, self.model = self.model, None  # report as a training-free baseline
            try:
                return super()._copy_leak()
            finally:
                self.model = model

    return RetrievalScorer


def run(tables, evalset, out, *, config=None, **kw) -> dict:
    """Fit and score the retrieval baseline on the validation split; returns scores.json."""
    from minicandi.dataset import tables_sha256
    from onion_eval import score as S
    from onion_eval.evalset import tables_dir_of
    import json
    ev = json.loads(Path(evalset).read_text(encoding="utf-8"))
    tdir = tables_dir_of(ev, evalset)
    if Path(tables).resolve() != Path(tdir).resolve():
        raise S.ScoreRefused(f"--tables {tables} is not the eval set's tables {tdir}")
    if kw.get("split", "val") != "val":
        raise S.ScoreRefused("the retrieval baseline is scored on the validation split only")
    sc = _scorer_class()(evalset, config=config, run_tables_sha256=tables_sha256(tdir), **kw)
    scores = sc.run()
    S.write_outputs(scores, sc.suff, out)
    with open(Path(out) / NEIGHBORS_TSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=NEIGHBOR_COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(sc.retrieval.neighbor_rows())
    return scores


def main(argv=None) -> int:
    from onion_eval import score as S
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--tables", required=True)
    p.add_argument("--evalset", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if (Path(a.out) / S.SCORED).is_file():
        print(f"skip: {a.out}/{S.SCORED} exists")
        return 0
    S.set_threads()
    try:
        s = run(a.tables, a.evalset, a.out)
    except (PermissionError, S.ScoreRefused) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    sel = s["selection"]
    print(f"scores -> {a.out}/scores.json; composite {sel['composite']} (imputation "
          f"{sel['imputation_gap_closed']}, correction {sel['correction_gap_closed']}, uncertainty "
          f"{sel['uncertainty_gap_closed']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
