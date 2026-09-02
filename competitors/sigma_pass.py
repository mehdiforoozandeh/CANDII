#!/usr/bin/env python3
"""The TRAINING-RESIDUAL σ pass — one fitter for every point-only method (D2).

    python -m competitors.sigma_pass --regime <derived.json> --pred <training_pred_root> \
        --out sigma.json --method Avocado [--chroms chr19] [--eval-regions <abs bed>]

    sigma[assay] = sqrt( mean over bins and over tracks of (signal_mu - truth pval)^2 ),
                   in -log10 p, over the TRAINING cells scored against THEMSELVES.

WHAT IT REPLACES, AND WHY
-------------------------
Four near-identical files — `competitors/{avocado,edice,lavawizard,chromimpute}/fit_sigma.py` —
fitted σ on the residuals of the DECLARED EVAL PAIRS and wrote `fitted_on = "<regime> eval_pairs"`.
Rule 1 (`BENCHMARK_DESIGN.md` §12.2) voids every table they produced: a σ fitted on `V_` truth and
reused unchanged on `B_` still spends the eval panel to set the width of the forecast the panel then
scores. All four are retired; this is the one instrument that replaces them, so a σ difference
between two methods is a method difference and not a fitter difference.

THE ONE STRING THAT MATTERS
---------------------------
`fitted_on` begins with `training-residuals:` and a consumer accepts a table only on that prefix
(`SIGMA_FITTED_ON_PREFIX` below, which the scoring scripts import rather than retype). That is the
whole leak-free/leaky test, months later, from the table alone: an eval-pair table cannot start with
those characters and a training-residual table cannot fail to.

WHAT IT REFUSES
---------------
Any regime that names a `V_` or `B_` cell — as an `eval_pairs` target, or in `biosamples.eval` —
exits **3** before the store is opened. Rule 1 is not a warning here: a fitter that can be pointed at
the eval panel by a typo in a launcher is a fitter that eventually is.

HOW THE TRUTH IS READ
---------------------
`candi.bench.external.stream_truth` over `harness.open_source(store=…, eval_regions=…)` — the same
read the scorer performs, through the same window plan and the same `--eval-regions` scope, so the
residual σ is fitted on is the residual the scorer will measure. Two mechanical notes:

* **Self-pairs need the `y_*_imp` keys aliased.** `StoreDataset._make_batch` emits the imputation
  keys only for a cross-cell target (`imp_target=None` on a self-pair), and `stream_truth` reads
  them by name. On a self-pair the input cell IS the target cell, so the batch's own `y_data` /
  `y_pval` / `y_peaks` are exactly that cell's truth — verified bit-identical against
  `CorpusStore[cell][assay].pval(chrom, 0)` in `tests/test_sigma_pass.py`. `_SelfPairTruth` aliases
  them and changes nothing else.
* **One chromosome at a time.** `stream_truth` allocates three full-length float32 buffers per
  target assay per chromosome and returns them all at once. Over `eic_pilot`'s 18 training
  chromosomes and a 35-assay cell that is ~50 GB; per chromosome it is ~4 GB. The arithmetic is
  per-chromosome anyway, so nothing changes but the peak.

The residual is taken on `EvalSource.scored_bins` — the same index `external.build_record` cuts a
scored track with. Without it the bins no window covered would enter the mean as zeros.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: §4.2/§6.1 — a consumer accepts a σ table ONLY when `fitted_on` starts with this. Imported, never
#: retyped: the prefix and the check have to be one object or they drift apart.
SIGMA_FITTED_ON_PREFIX = "training-residuals:"

#: The panels a σ fit may never touch (§6.2, Rule 1).
EVAL_PANEL_PREFIXES = ("V_", "B_")

#: Exit code for "this regime names an eval-panel cell". Shared with `tools/sigma_training_regime.py`.
EXIT_EVAL_PANEL = 3

#: Used only when `competitors.baselines.heads` cannot be imported — a rival's job may run with
#: `competitors/` on the path but not the baselines tree. `read_sigma_table` refuses a non-positive
#: σ, so an assay whose residual is exactly zero needs SOME floor or the whole table is rejected.
FALLBACK_SIGMA_FLOOR = 1e-3


def sigma_floor() -> Tuple[float, str]:
    """`(floor, where it came from)`. The baselines' own floor when it is importable.

    One floor across the suite or the naive baselines and the rivals clamp at different widths, and
    the CRPS gap between them would carry that difference.
    """
    try:
        from competitors.baselines.heads import SIGMA_FLOOR
    except Exception:                                              # pragma: no cover - path-dependent
        return FALLBACK_SIGMA_FLOOR, "fallback (competitors.baselines.heads not importable)"
    return float(SIGMA_FLOOR), "competitors.baselines.heads.SIGMA_FLOOR"


class _SelfPairTruth:
    """An `EvalSource` view that carries the `y_*_imp` keys a SELF-PAIR does not get, on one chrom.

    Two overrides and nothing else; every other attribute is the real source's.

    `batch` — `StoreDataset._make_batch` adds `y_data_imp` / `y_pval_imp` / `y_peaks_imp` only when
    it is given a cross-cell `imp_target`, and `StoreSource._raw_batch` passes `None` when the pair's
    two halves are the same cell. `stream_truth` reads those keys by name. On a self-pair the batch
    is built FOR the target cell, so its own `y_*` keys are that cell's truth; aliasing is exact, not
    an approximation. It is done here rather than in `candi.bench` because the benchmark path never
    self-pairs and should not grow a branch for a fitter.

    `eval_chroms` — one chromosome per pass, to bound `stream_truth`'s buffers (see the module
    docstring).
    """

    def __init__(self, inner: Any, chrom: str):
        self._inner = inner
        self.eval_chroms = (chrom,)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def batch(self, pair, chrom, starts, kind, **kw):
        b = self._inner.batch(pair, chrom, starts, kind, **kw)
        if "y_pval_imp" in b:
            return b
        out = dict(b)
        out["y_data_imp"] = b["y_data"]
        out["y_pval_imp"] = b["y_pval"]
        out["y_peaks_imp"] = b["y_peaks"]
        return out


def eval_panel_names(obj: Dict[str, Any]) -> List[str]:
    """Every `V_`/`B_` cell this regime object names on a side that supplies TRUTH.

    Read off the JSON rather than off an opened source, so the refusal happens before the store is
    touched: the point is that no eval-panel track is ever read, not that it is read and discarded.
    """
    names: List[str] = [str(b) for b in (obj.get("biosamples") or {}).get("eval", ())]
    for item in obj.get("eval_pairs") or ():
        if isinstance(item, dict):
            names.append(str(item.get("target")))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            names.append(str(item[1]))
    return sorted({n for n in names if n.startswith(EVAL_PANEL_PREFIXES)})


def fit(source: Any, root: Path, *, batch_windows: int = 4, progress: bool = True
        ) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, int], List[str]]:
    """`(sse, n_points, n_tracks, cells)` — the pooled squared residual per assay.

    Pair-outer, chromosome-inner, track-innermost: one pass over the store per (cell, chromosome),
    and each prediction npz read exactly once.
    """
    from candi.bench.external import ExternalError, _expected, read_track_arrays, stream_truth

    expected = _expected(source)
    chroms = list(source.eval_chroms)
    n_bins = {c: source.n_bins(c) for c in chroms}
    scoped = {c: source.scored_bins(c) for c in chroms}

    by_pair: Dict[Any, List[Tuple[str, str]]] = {}
    for d, (pair, assay) in sorted(expected.items()):
        if (root / d).is_dir():
            by_pair.setdefault(pair, []).append((assay, d))
    if not by_pair:
        raise SystemExit(
            f"{root} covers none of the {len(expected)} tracks {source.regime_path} declares. The "
            f"σ pass fits on the TRAINING tracks of the derived regime — point --pred at the "
            f"training-track prediction root, not at the V_/B_ one.")

    sse: Dict[str, float] = defaultdict(float)
    n_points: Dict[str, int] = defaultdict(int)
    n_tracks: Dict[str, int] = defaultdict(int)
    for k, (pair, rows) in enumerate(by_pair.items()):
        cols = [source.assays.index(assay) for assay, _ in rows]
        for chrom in chroms:
            truth = stream_truth(_SelfPairTruth(source, chrom), pair, cols,
                                 batch_windows=batch_windows)
            idx = scoped[chrom]
            for (assay, d), col in zip(rows, cols):
                pred = read_track_arrays(root / d, [chrom], {chrom: n_bins[chrom]})
                if "signal_mu" not in pred[chrom]:
                    raise ExternalError(
                        f"{d}/{chrom}.npz has no `signal_mu`. σ is the spread of the PVAL arm's "
                        f"residual; a track with no point prediction in -log10 p has no residual "
                        f"to fit.")
                mu = pred[chrom]["signal_mu"].astype(np.float64)
                tv = np.asarray(truth[col][chrom]["pval"], dtype=np.float64)
                if idx is not None:
                    mu, tv = mu[idx], tv[idx]
                r = mu - tv
                sse[assay] += float(np.dot(r, r))
                n_points[assay] += int(r.size)
            del truth
        for assay, _ in rows:
            n_tracks[assay] += 1
        if progress:
            print(f"[sigma] {k + 1}/{len(by_pair)} {pair} — {len(rows)} track(s)", flush=True)

    cells = sorted({p.target_biosample for p in by_pair})
    return dict(sse), dict(n_points), dict(n_tracks), cells


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m competitors.sigma_pass",
                                description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regime", required=True,
                   help="the DERIVED training-residual regime (tools/sigma_training_regime.py)")
    p.add_argument("--pred", required=True, help="the §4.1 root of TRAINING-track predictions")
    p.add_argument("--out", required=True)
    p.add_argument("--method", required=True, help="the method slug written into the table")
    p.add_argument("--chroms", default=None,
                   help="comma-separated; default every chromosome the derived regime scores")
    p.add_argument("--eval-regions", default=None,
                   help="absolute BED the residual is cut to (eic_pilot: the Pilot BED)")
    p.add_argument("--batch-windows", type=int, default=4)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    regime_path = Path(a.regime)
    obj = json.loads(regime_path.read_text(encoding="utf-8"))

    banned = eval_panel_names(obj)
    if banned:
        print(f"REFUSING {regime_path}: it names eval-panel cell(s) {banned[:5]} as a source of "
              f"truth. Rule 1 (BENCHMARK_DESIGN.md 12.2) voids a sigma fitted on any "
              f"{'/'.join(EVAL_PANEL_PREFIXES)} residual; fit on the training-residual regime "
              f"tools/sigma_training_regime.py writes.", file=sys.stderr)
        return EXIT_EVAL_PANEL

    from candi.bench.external import read_manifest
    from candi.bench.harness import open_source

    root = Path(a.pred)
    if not root.is_dir():
        raise SystemExit(f"--pred {root} is not a directory")
    manifest_path = root / "manifest.json"
    read_manifest(root)                       # §4.1 — refuses a root with no `method`

    chroms = tuple(c.strip() for c in a.chroms.split(",")) if a.chroms else None
    source = open_source(store=regime_path, chroms=chroms, eval_regions=a.eval_regions)
    try:
        # The same refusal again, on what the source RESOLVED rather than on what the file says.
        # The two can differ (`--biosamples`, a regime edited after it was derived), and this one is
        # the last gate before a batch is read.
        bad = sorted({p.target_biosample for p in source.pairs("impute")
                      if p.target_biosample.startswith(EVAL_PANEL_PREFIXES)})
        if bad:
            print(f"REFUSING {regime_path}: it resolves to eval-panel target(s) {bad[:5]}.",
                  file=sys.stderr)
            return EXIT_EVAL_PANEL
        used = list(source.eval_chroms)
        sse, n_points, n_tracks, cells = fit(source, root, batch_windows=a.batch_windows)
        scope = source.scope()
        regions_sha = None if source.eval_regions is None else source.eval_regions.sha256
    finally:
        source.close()

    floor, floor_source = sigma_floor()
    sigma = {k: max(float(np.sqrt(sse[k] / n_points[k])), floor)
             for k in sorted(sse) if n_points.get(k)}
    floored = sorted(k for k in sigma if sigma[k] <= floor)

    fitted_on = (f"{SIGMA_FITTED_ON_PREFIX} {regime_path.name} T_ self-pairs, {len(cells)} cells, "
                 f"chroms {used}")
    if regions_sha is not None:
        fitted_on += f", regions {regions_sha[:12]}"

    table = {
        "method": a.method,
        "fitted_on": fitted_on,
        "sigma": sigma,
        "n_tracks": {k: int(n_tracks[k]) for k in sorted(sigma)},
        "n_points": {k: int(n_points[k]) for k in sorted(sigma)},
        "regime_sha256": hashlib.sha256(regime_path.read_bytes()).hexdigest(),
        "pred_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "date": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sigma_floor": floor,
        "sigma_floor_source": floor_source,
        "floored_assays": floored,
        "cells": cells,
        "chroms": used,
        "eval_scope": scope,
        "note": ("D2. sigma = sqrt(pooled mean squared residual) per assay, over the TRAINING "
                 "cells scored against themselves on the training chromosomes, in -log10 p. No "
                 "V_/B_ track is opened, so no leaderboard number is in-sample for this sigma."),
    }
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")

    for k in sorted(sigma):
        print(f"  sigma[{k}] = {sigma[k]:.4f}  (n_tracks={n_tracks[k]}, n={n_points[k]:,})")
    if floored:
        print(f"[sigma] {len(floored)} assay(s) at the floor {floor:g}: {floored}")
    print(f"[sigma] {fitted_on}")
    print(f"[sigma] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":                                            # pragma: no cover
    raise SystemExit(main())
