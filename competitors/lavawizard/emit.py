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

**No runtime dependency on `candi`.** An earlier version imported `track_dirname` from
`candi.bench.external` so the §4.1 contract had a single definition. That broke the moment this
package was rsynced to Fir on its own — `ModuleNotFoundError: No module named 'candi'` — and a
rival's generator has to run wherever the rival's data is. So the naming rule is implemented here,
and `tests/test_lavawizard.py::test_track_dirname_agrees_with_the_bench_reader` asserts character
for character that it matches `candi.bench.external.track_dirname`. One definition of *truth*,
enforced by a test instead of by an import that dictates deployment.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence

import numpy as np

__all__ = ["ARCSINH_INVERSION", "CLIP_RULE", "Pair", "track_dirname", "invert_arcsinh",
           "write_track", "write_manifest"]


class Pair(NamedTuple):
    """`(input biosample, target biosample)` — structurally what `candi.bench.harness.Pair` is.

    A `NamedTuple` rather than an import, for the reason in the module docstring. `write_track`
    also accepts the real `harness.Pair`, since it only reads the two attributes.
    """
    input_biosample: str
    target_biosample: str


def track_dirname(pair, assay: str) -> str:
    """§4.1: the bench `track_key` with `|` swapped for the filesystem-safe `__`.

    `kind` is `impute` and is implied — `denoise` has no external producer. Kept in step with
    `candi.bench.external.track_dirname` by test, not by import.
    """
    return f"{pair.input_biosample}__{pair.target_biosample}__{assay}"

#: Recorded in the manifest so a reader knows which way round the clip and the sinh went.
ARCSINH_INVERSION = "clip_at_zero_then_sinh"

#: The cap rule, stamped into every manifest whether or not the cap is on. PI ruling, 2026-08-26.
CLIP_RULE = "training_max_per_mark_per_chrom"


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
                already_inverted: bool = False,
                clip_max: Optional[float] = None) -> Path:
    """One `<pred_root>/<track>/<chrom>.npz` holding `signal_mu` in `-log10 p`.

    `signal_arcsinh` is the model's raw output, in `arcsinh(-log10 p)`; pass
    `already_inverted=True` to write a vector that is already in `-log10 p`. `n_bins` is the
    store's `floor(chr_len / 25)` for this chromosome and is checked here, not only by the reader —
    a length error found at write time names one track, the same error found at score time has
    already cost the whole run.

    **`clip_max` is a deviation from the faithful port** (PI ruling, 2026-08-26) and is `None` —
    off — for anything reproducing upstream. The head is `Dense(1)(x) + average` with nothing
    bounding it, and `sinh` turns a `+6.2` overshoot in `arcsinh` space into a 500x multiplicative
    one: on the anchor, five bins of `chr17` where the model emitted 15.50 against a truth of 0.57
    carried the whole of that chromosome's MSE. `clip_max` caps `signal_mu` at the largest value
    that mark reaches in this chromosome's **training** data — data-derived, never tuned, and never
    read off the target. Whether it was applied belongs in the manifest, not in a commit message,
    so a score file says for itself which rule made it.
    """
    vec = np.asarray(signal_arcsinh).reshape(-1)
    if vec.shape[0] != int(n_bins):
        raise ValueError(
            f"{pair}/{assay} {chrom}: prediction has {vec.shape[0]} bins, grid wants {int(n_bins)}. "
            f"Index i is the bin at i*25 bp — a length mismatch shifts every downstream bin.")
    mu = np.asarray(vec, dtype=np.float32) if already_inverted else invert_arcsinh(vec)
    if clip_max is not None:
        cap = float(clip_max)
        if not np.isfinite(cap) or cap <= 0.0:
            raise ValueError(f"{pair}/{assay} {chrom}: clip_max must be finite and positive, "
                             f"got {clip_max!r}; a cap of zero would erase the track.")
        mu = np.minimum(mu, np.float32(cap))
    if not np.all(np.isfinite(mu)):
        raise ValueError(f"{pair}/{assay} {chrom}: {int((~np.isfinite(mu)).sum())} non-finite bins; "
                         f"sinh overflows above ~arcsinh(3e38) and the scorer will not accept NaN.")
    out_dir = Path(root) / track_dirname(pair, assay)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{chrom}.npz"
    np.savez_compressed(path, signal_mu=mu)
    return path


def write_manifest(root: Path | str, *, version: str, generated_by: str,
                   contributor_mode: str, weights: str, clip: bool,
                   notes: str = "", extra: Optional[Mapping[str, Any]] = None,
                   sparse_assays: Sequence[str] = ()) -> Path:
    """`<pred_root>/manifest.json` — §4.1's `{method, version, generated_by, date, arms, notes}`.

    Copied verbatim into the score file's provenance, so the three fields that decide whether a row
    is comparable go in as data rather than prose: `contributor_mode` (`loo` for anything we
    report, `upstream` only for a parity run — see `features.py`), `weights` (`ported-retrain` or
    the Synapse id their released weights came from), and `signal_inversion`.

    `clip` is required rather than defaulted, because "was the output capped" is exactly the fact a
    reader must not have to guess: `false` is a faithful-port root, `true` is a capped one, and a
    manifest that simply omitted the key would be indistinguishable from one written before the cap
    existed.

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
        "clip": bool(clip),
        "clip_rule": CLIP_RULE,
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
