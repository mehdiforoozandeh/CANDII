"""Write a `RIVALS_PLAN.md` §4.1 prediction root. Scoring is `candi.bench.external`'s job.

Lavawizard is a **point-only, pval-arm** rival, so a track carries `signal_mu` and nothing else:

- no `mu`/`n` — decision B1b forbids inventing a read depth to manufacture a count prediction from
  a `-log10 p` point;
- no `signal_sigma` — the σ-table of §6.1 supplies it at scoring time, fitted on V-pair residuals,
  and a producer that guessed its own spread would put a made-up number inside the CRPS;
- no `peak_score` — the entry point falls back to ranking by `signal_mu` and records
  `has_peak_head=False`, which is the coverage-ranking caveat of B3.

**Inversion happens here.** The model predicts in `arcsinh(-log10 p)`; §4.1 says external
`signal_mu` is always already in `-log10 p`. `write_track` applies upstream's own inversion —
clip the arcsinh-space prediction at 0, then `sinh` (`04_guacamole6_generate.py:189-193`). The
order matters: clipping after `sinh` would be a different function on negatives, and theirs is the
one that produced their submission.

`track_dirname` is imported from `candi.bench.external` rather than re-derived. The §4.1 contract
gets one definition, and it lives with the reader that enforces it.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from candi.bench.external import track_dirname
from candi.bench.harness import Pair

__all__ = ["ARCSINH_INVERSION", "invert_arcsinh", "write_track", "write_manifest"]

#: Recorded in the manifest so a reader knows which way round the clip and the sinh went.
ARCSINH_INVERSION = "clip_at_zero_then_sinh"


def invert_arcsinh(pred: np.ndarray) -> np.ndarray:
    """`arcsinh(-log10 p)` back to `-log10 p`, upstream's way: clip at 0 first, then `sinh`.

    `sinh` is monotone and `sinh(0) == 0`, so clipping first and clipping after agree on the
    result — but only because the clip floor is zero. Written in their order anyway: this function
    exists to reproduce a submission, not to be improved.
    """
    out = np.array(pred, dtype=np.float32, copy=True).reshape(-1)
    out[out < 0] = 0.0
    return np.sinh(out).astype(np.float32, copy=False)


def write_track(root: Path | str, pair: Pair, assay: str, chrom: str,
                signal_arcsinh: np.ndarray, *, n_bins: int,
                already_inverted: bool = False) -> Path:
    """One `<pred_root>/<track>/<chrom>.npz` holding `signal_mu` in `-log10 p`.

    `signal_arcsinh` is the model's raw output, in `arcsinh(-log10 p)`; pass
    `already_inverted=True` to write a vector that is already in `-log10 p`. `n_bins` is the
    store's `floor(chr_len / 25)` for this chromosome and is checked here, not only by the reader —
    a length error found at write time names one track, the same error found at score time has
    already cost the whole run.
    """
    vec = np.asarray(signal_arcsinh).reshape(-1)
    if vec.shape[0] != int(n_bins):
        raise ValueError(
            f"{pair}/{assay} {chrom}: prediction has {vec.shape[0]} bins, grid wants {int(n_bins)}. "
            f"Index i is the bin at i*25 bp — a length mismatch shifts every downstream bin.")
    mu = np.asarray(vec, dtype=np.float32) if already_inverted else invert_arcsinh(vec)
    if not np.all(np.isfinite(mu)):
        raise ValueError(f"{pair}/{assay} {chrom}: {int((~np.isfinite(mu)).sum())} non-finite bins; "
                         f"sinh overflows above ~arcsinh(3e38) and the scorer will not accept NaN.")
    out_dir = Path(root) / track_dirname(pair, assay)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{chrom}.npz"
    np.savez_compressed(path, signal_mu=mu)
    return path


def write_manifest(root: Path | str, *, version: str, generated_by: str,
                   contributor_mode: str, weights: str,
                   notes: str = "", extra: Optional[Mapping[str, Any]] = None,
                   sparse_assays: Sequence[str] = ()) -> Path:
    """`<pred_root>/manifest.json` — §4.1's `{method, version, generated_by, date, arms, notes}`.

    Copied verbatim into the score file's provenance, so the three fields that decide whether a row
    is comparable go in as data rather than prose: `contributor_mode` (`loo` for anything we
    report, `upstream` only for a parity run — see `features.py`), `weights` (`ported-retrain` or
    the Synapse id their released weights came from), and `signal_inversion`.

    `sparse_assays` carries §5's `<= 2 contributors` flag into every table that reads this root.
    """
    obj: Dict[str, Any] = {
        "method": "Lavawizard",
        "version": version,
        "generated_by": generated_by,
        "date": date.today().isoformat(),
        "arms": ["pval"],
        "notes": notes,
        "contributor_mode": contributor_mode,
        "weights": weights,
        "signal_inversion": ARCSINH_INVERSION,
        "signal_space": "-log10 p",
        "upstream": "github.com/ccchang0111/ENCODE_imputation_2019@d638b204",
        "sparse_assays": list(sparse_assays),
    }
    if extra:
        obj.update(extra)
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    manifest = path / "manifest.json"
    manifest.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    return manifest
