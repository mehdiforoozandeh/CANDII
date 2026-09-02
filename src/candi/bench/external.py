"""external — score prediction tracks somebody else wrote to disk (`RIVALS_PLAN.md` §4).

    python -m candi.bench.external --store regime.json --pred <pred_root> --out scores.json

`candi.bench` scores a checkpoint: it opens a corpus, prompts a model, and compares what came back
to the truth. A rival method has no checkpoint we can load — Avocado is a TensorFlow port,
ChromImpute is Java, the naive baselines are a numpy script — so the only thing every one of them
can hand over is **the prediction itself, on our grid**. This module is the entry that takes it.

Three properties make it worth a module rather than a converter script.

**It is the same instrument.** Nothing here computes a metric. Truth comes from
`harness.open_source` and the same window walk `harness.stream_tracks` performs; scoring is
`harness.score_track`; the roll-up is `harness.macro_mean`; the panel measure is
`harness.panel_specificity`. The acceptance gate in §4.3 is a round-trip — CANDI's own predictions
streamed out to this format and scored back through here must reproduce the model path's numbers to
float re-order tolerance — and it is a gate precisely because "the same instrument" is a claim that
can be checked instead of asserted.

**It refuses to guess.** A prediction array whose length is not the chromosome's bin count is an
error naming the track, never a truncation or a pad; a track directory that names no declared pair
is an error naming the directory; a declared pair the root does not cover is listed in
`provenance.missing_tracks` and FAILS the run unless `--allow-missing` says otherwise. That last one
is the D2 lesson — a partial panel scored as if it were whole — applied to a producer we do not
control.

**It scores only the arms the method actually predicted.** A rival that emits `-log10 p` has no
count prediction and B1b forbids inventing a read depth to manufacture one; a rival that emits a
point has no forecast distribution until a σ-table (§6.1) supplies one. Both cases come out as
ABSENT KEYS. `harness.TrackRecord.has_count` / `has_sigma` are where that is decided, and every
record the model path builds satisfies both, so the model path is untouched.

The C block is absent by construction and not by omission: every covariate instrument perturbs a
prompt and re-decodes, which needs the model.

**And the truth itself can come from somewhere else.** `--truth-root` reads the pval truth from a
second §4.1-layout root — the 2019 challenge's own blind-truth bigwigs, converted by
`tools/challenge_bigwigs.py` — instead of from the store, so a CANDI row and a 2019 entrant's row
are measured against the same bytes the entrants were originally ranked on. It swaps ONE input:
the store still owns the grid, the declared track list and the provenance, and the scoring is the
same `score_track`. What the challenge never distributed — read counts, peak calls — is absent from
the row rather than filled in (`WITHHELD_WITHOUT_PEAK_TRUTH`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from candi.bench import annotations as ann
from candi.bench import distributional as D
from candi.bench.harness import (
    ARMS, SCOPE_HELD_OUT, EvalSource, Pair, TrackRecord, _binarise, _varpool, cross_cell,
    macro_mean, open_source, panel_macros, panel_specificity, score_track, track_key,
)

__all__ = [
    "PRED_ARRAYS", "WITHHELD_WITHOUT_PEAK_TRUTH", "track_dirname", "read_manifest",
    "read_truth_manifest", "read_sigma_table", "read_track_arrays", "read_truth_root_arrays",
    "stream_truth", "build_record", "score_external", "build_parser", "main",
]

#: The arrays §4.1 recognises inside a `chr*.npz`. Anything else in the file is ignored — a producer
#: may carry its own diagnostics alongside — but a file with NONE of these is an error, because a
#: track that predicts nothing is not a track.
PRED_ARRAYS: Tuple[str, ...] = ("signal_mu", "signal_sigma", "mu", "n", "peak_score")

#: §4.1 — the directory name is the bench `track_key` with `|` swapped for the filesystem-safe `__`.
#: `kind` is `impute` and is implied: `denoise` has no external producer, since a rival that denoises
#: a cell's own tracks is a different task from the one this leaderboard ranks.
SEP = "__"

#: §4.1 — external `signal_mu` is ALWAYS already in `-log10 p`. A rival carries no training-space
#: transform of ours, so there is nothing to invert and the identity is the honest answer. Recorded
#: in provenance as `pred_inversion: "external"` so a reader can tell "already in the eval space"
#: from "a CANDI head trained in it".
SIGNAL_TARGET_TRANSFORM = "none"

#: The pval-arm keys that read a PEAK CALL, withheld under `--truth-root`.
#:
#: A challenge truth root is one number per bin — the organizers' `-log10 p` signal — and the 2019
#: challenge distributed no peak calls at all. So the count arm has no truth (it is absent by
#: `has_count`) and every measure below has no truth either. They are REMOVED from the row rather
#: than computed against a zero-filled stand-in, for the same reason §4.2 gives for an unpredicted
#: arm: `peak_base_rate` would be a finite 0.0 that `macro_mean` averages, and `auprc` a nan a
#: reader cannot tell from a real undefined one.
#:
#: `peak_overlap_<p>` and `n_points` are NOT here, and that is not an oversight: `binary_suite`
#: computes both from the truth SIGNAL and the ranking score, never from `y_peaks`.
#: `prom_corr_h3k4me3` stays for the same reason — its windows come from the gene annotation.
WITHHELD_WITHOUT_PEAK_TRUTH: Tuple[str, ...] = (
    "acc_by_imp_strength", "acc_by_obs_strength", "auprc", "bernoulli_nll", "peak_base_rate",
    "peak_shape_corr_dnase",
)


class ExternalError(ValueError):
    """A prediction root that does not meet the §4.1 contract. Always names the offending track."""


# ---------------------------------------------------------------------------
# the on-disk contract (§4.1)
# ---------------------------------------------------------------------------

def track_dirname(pair: Pair, assay: str) -> str:
    return track_key(pair, assay, "impute").replace("|", SEP)


def read_manifest(root: Path) -> Dict[str, Any]:
    """`<pred_root>/manifest.json`, verbatim. `method` is the one field the result cannot do without.

    Copied into provenance rather than summarised: a score file whose method string is the only
    trace of what produced it is a score file nobody can re-run.
    """
    path = root / "manifest.json"
    if not path.exists():
        raise ExternalError(
            f"{root} has no manifest.json. RIVALS_PLAN.md §4.1 requires "
            f"{{method, version, generated_by, date, arms, notes}} — a scored track with no record "
            f"of the code that wrote it cannot be placed in a leaderboard.")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or not obj.get("method"):
        raise ExternalError(f"{path} carries no `method` — provenance.method has nothing to name.")
    return obj


def read_truth_manifest(root: Path) -> Tuple[Dict[str, Any], str]:
    """`(manifest, sha256 of manifest.json)` for a TRUTH root — the challenge's own signal tracks.

    A truth root is the same on-disk layout as a prediction root and is deliberately NOT the same
    kind of thing, so it carries `kind: "truth"` instead of `method` and is refused here when it
    does not. Passing a prediction root to `--truth-root` would otherwise score a method against
    itself and produce a perfect, meaningless row.

    The hash is of the manifest bytes, not of the arrays: it is what a score file quotes to say
    WHICH build of the truth it was measured against, and `tools/challenge_bigwigs.py` writes the
    source directory, the bridge hash and the bin rule into that manifest for exactly that purpose.
    """
    path = root / "manifest.json"
    if not path.exists():
        raise ExternalError(
            f"--truth-root {root} has no manifest.json. A truth root must say which bigwigs it was "
            f"built from and under which bin rule, or a score file cannot name its own truth.")
    raw = path.read_bytes()
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict) or obj.get("kind") != "truth":
        raise ExternalError(
            f"{path} is not a truth manifest (`kind` is {obj.get('kind')!r} and must be 'truth'). "
            f"A PREDICTION root passed as --truth-root would score a method against its own output.")
    return obj, hashlib.sha256(raw).hexdigest()


def read_truth_root_arrays(track_dir: Path, chroms: Sequence[str],
                           n_bins: Mapping[str, int]) -> Dict[str, Dict[str, np.ndarray]]:
    """One track's TRUTH from a §4.1-layout root — `signal_mu`, on the same length assertion.

    The same loader as the prediction side, with one extra requirement: `signal_mu` must be there.
    A truth root built from a bigwig has exactly one array per bin, and a root that carried `mu`/`n`
    instead would be a prediction root that slipped through `read_truth_manifest`.
    """
    got = read_track_arrays(track_dir, chroms, n_bins)
    for c in chroms:
        if "signal_mu" not in got[c]:
            raise ExternalError(
                f"{track_dir.name}/{c}.npz carries {sorted(got[c])} and no `signal_mu`. A truth "
                f"root holds the experimental signal, one value per bin.")
    return got


def read_sigma_table(path: Optional[Path | str]) -> Optional[Dict[str, Any]]:
    """§4.2/§6.1 — `{method, fitted_on, sigma: {assay: value}}`, the σ a point-only rival is given.

    `fitted_on` is not decoration. B1a fits σ on V-pair residuals and the B-pair run reuses that
    table unchanged, so the string that says which panel it was fitted on is what makes a quoted
    CRPS leak-free or not.
    """
    if path is None:
        return None
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    sig = obj.get("sigma")
    if not isinstance(sig, dict) or not sig:
        raise ExternalError(f"{path} has no `sigma` map; §4.2 wants {{assay: value}}.")
    bad = {a: v for a, v in sig.items() if not (float(v) > 0.0)}
    if bad:
        raise ExternalError(f"{path}: σ must be positive, got {bad}")
    return obj


def read_track_arrays(track_dir: Path, chroms: Sequence[str],
                      n_bins: Mapping[str, int]) -> Dict[str, Dict[str, np.ndarray]]:
    """`{chrom: {array name: float32 vector}}` for one track, length-checked against the grid.

    **The length assertion is the whole point of this function.** Index `i` of every array is the bin
    starting at `i * 25` bp, so an array one element short is not "nearly right" — it is every bin
    after the first gap compared against the wrong position. `DATA.md`'s grid is
    `floor(chr_len / 25)`, the store computes it, and a producer that disagrees is refused here
    rather than silently trimmed.
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for chrom in chroms:
        npz = track_dir / f"{chrom}.npz"
        if not npz.exists():
            raise ExternalError(
                f"{track_dir.name}: no {chrom}.npz. The scored chromosomes are {list(chroms)}; a "
                f"track that covers only some of them cannot be scored over the concatenation "
                f"(the top-1% thresholds of mse1obs/mse1imp are taken over all of them at once). "
                f"Emit the missing chromosome or narrow the run with --chroms.")
        want = int(n_bins[chrom])
        got: Dict[str, np.ndarray] = {}
        with np.load(npz) as z:
            present = [k for k in PRED_ARRAYS if k in z]
            if not present:
                raise ExternalError(
                    f"{track_dir.name}/{chrom}.npz holds {sorted(z.files)} and none of "
                    f"{list(PRED_ARRAYS)}. §4.1 names the arrays a track may supply.")
            for k in present:
                arr = np.asarray(z[k], dtype=np.float32)
                if arr.ndim != 1 or arr.shape[0] != want:
                    raise ExternalError(
                        f"{track_dir.name}/{chrom}.npz: `{k}` is {arr.shape} and {chrom} is {want} "
                        f"bins on the absolute 25 bp grid. The loader does not truncate or pad — "
                        f"bin i must be the bin starting at i*25 bp, or every metric is measured "
                        f"against the wrong positions.")
                got[k] = arr
        if ("mu" in got) != ("n" in got):
            raise ExternalError(
                f"{track_dir.name}/{chrom}.npz carries {'mu' if 'mu' in got else 'n'} without the "
                f"other. The count arm is a NEGATIVE BINOMIAL — both parameters or neither; the "
                f"entry point derives nothing (§5.1).")
        out[chrom] = got
    return out


# ---------------------------------------------------------------------------
# truth — the same walk `stream_tracks` performs, with the model taken out
# ---------------------------------------------------------------------------

def stream_truth(source: EvalSource, pair: Pair, cols: Sequence[int], *,
                 batch_windows: int = 4) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
    """`{assay col: {chrom: {counts, pval, peaks}}}` over every bin of every eval chromosome.

    Pair-outer, window-inner, written by ABSOLUTE bin — `harness.stream_tracks` with the forward
    pass and the prediction buffers removed. It reads the same three keys off the same batch
    (`y_*_imp`, because a declared pair reads its truth from the target cell), so the truth an
    external track is scored against is the same array a checkpoint would have been scored against.
    That is what makes the §4.3 round-trip an equality rather than a correlation.

    It is a second copy of a loop and that is deliberate: the alternative is a null model handed to
    `stream_tracks`, which would run an encoder over the whole panel to throw its output away.
    """
    out: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {a: {} for a in cols}
    for chrom in source.eval_chroms:
        n = source.n_bins(chrom)
        buf = {a: {k: np.zeros(n, dtype=np.float32) for k in ("counts", "pval", "peaks")}
               for a in cols}
        starts = source.windows(chrom)
        for lo in range(0, len(starts), batch_windows):
            chunk = starts[lo:lo + batch_windows]
            batch = source.batch(pair, chrom, chunk, "impute")
            truth = {"counts": batch["y_data_imp"], "pval": batch["y_pval_imp"],
                     "peaks": batch["y_peaks_imp"]}
            for j, s in enumerate(chunk):
                sl = slice(int(s), int(s) + source.context_bins)
                for a in cols:
                    for k, t in truth.items():
                        buf[a][k][sl] = t[j, :, a].float().cpu().numpy()
        for a in cols:
            out[a][chrom] = buf[a]
    return out


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def build_record(pair: Pair, assay: str, chroms: Sequence[str],
                 truth: Mapping[str, Mapping[str, np.ndarray]],
                 pred: Mapping[str, Mapping[str, np.ndarray]],
                 sigma_table: Optional[Mapping[str, Any]] = None,
                 bins: Optional[Mapping[str, np.ndarray]] = None,
                 bin_scope: Optional[str] = None,
                 truth_signal_only: bool = False) -> Tuple[TrackRecord, str]:
    """One `(record, σ source)` — the truth from the store, the prediction from the npz.

    The peak tier follows the harness exactly: `peak_score` when the producer supplies one,
    otherwise the predicted LEVEL as a coverage ranking, with `has_peak_head=False` recorded so
    `loss_block` withholds `bernoulli_nll` and every table can label the row (B3). No rival has a
    peak head; the naive fraction-of-contributors baseline (§5.3) is the one producer that will
    supply a real `peak_score`.

    `bins` is t89's eval scope, `{chrom: bin indices}` from `EvalSource.scored_bins`. The rival
    still hands over a FULL-LENGTH array per chromosome — §4.1's length assertion is what makes bin
    `i` the bin at `i * 25` bp, and a producer allowed to emit a short array could not be checked at
    all — so the cut happens here, on both the truth and the prediction, with the same index. That
    is what makes a rival's selection number and CANDI's the same measurement: not the same flag,
    the same positions.

    `truth_signal_only` is the `--truth-root` case: `truth` then carries `pval` alone, because the
    2019 challenge distributed one signal track per experiment and no counts and no peak calls. The
    count arm is forced absent — a count PREDICTION with no count truth is not scoreable — and
    `counts`/`peaks` are filled with zeros of the scored length purely as carriers, since
    `TrackRecord.n_bins` measures the track off `counts` and `score_track` reads `peaks`
    unconditionally. Nothing in `WITHHELD_WITHOUT_PEAK_TRUTH` survives into the row, so no number a
    reader sees was computed against those zeros. The truth is cut with the SAME index as the
    prediction, exactly as the store truth is.
    """
    rec = TrackRecord(pair=pair, assay=assay, kind="impute", chroms=tuple(chroms),
                      bin_scope=(None if bins is None else (bin_scope or "regions")))
    sigma_const = None
    if sigma_table is not None:
        got = sigma_table.get("sigma", {}).get(assay)
        sigma_const = None if got is None else float(got)

    has_signal = all("signal_mu" in pred[c] for c in chroms)
    has_own_sigma = all("signal_sigma" in pred[c] for c in chroms)
    has_count_pred = all("mu" in pred[c] for c in chroms)
    has_peak = all("peak_score" in pred[c] for c in chroms)
    for name, seen in (("signal_mu", has_signal), ("signal_sigma", has_own_sigma),
                       ("mu", has_count_pred), ("peak_score", has_peak)):
        if not seen and any(name in pred[c] for c in chroms):
            raise ExternalError(
                f"{track_dirname(pair, assay)}: `{name}` is present on some chromosomes and absent "
                f"on others. A track is scored over the CONCATENATION of every eval chromosome, so "
                f"an arm exists for all of them or for none.")
    if not (has_signal or has_count_pred):
        raise ExternalError(
            f"{track_dirname(pair, assay)}: neither the pval arm (`signal_mu`) nor the count arm "
            f"(`mu`+`n`) is present. There is nothing to score.")
    if truth_signal_only and not has_signal:
        raise ExternalError(
            f"{track_dirname(pair, assay)}: the truth root carries a SIGNAL track only, and this "
            f"prediction has no `signal_mu`. A count prediction has no truth under challenge "
            f"truth — score it against the store instead (drop --truth-root).")
    # The count arm needs count TRUTH, and a challenge truth root has none. Absent rather than
    # scored, by the same rule §4.2 states for an arm the producer never predicted.
    has_count = has_count_pred and not truth_signal_only

    for c in chroms:
        # `cut` is the identity when there is no scope, and it hands back the caller's own object,
        # so an unscoped run builds the record it always built — the same arrays, not copies.
        def cut(v, _c=c):
            a = np.asarray(v, dtype=np.float32)
            return a if bins is None else a[bins[_c]]

        if truth_signal_only:
            # CARRIERS, NEVER SCORED. One array serves both slots — every consumer of `peaks`
            # copies it (`.astype(bool)`) and nothing writes through either dict.
            zero = np.zeros_like(cut(truth[c]["pval"]))
            rec.counts[c] = zero
            rec.peaks[c] = zero
        else:
            rec.counts[c] = cut(truth[c]["counts"])
            rec.peaks[c] = cut(truth[c]["peaks"])
        if has_count:
            rec.mu[c] = cut(pred[c]["mu"])
            rec.n[c] = cut(pred[c]["n"])
        if has_signal:
            rec.pval[c] = cut(truth[c]["pval"])
            rec.signal_mu[c] = cut(pred[c]["signal_mu"])
            if has_own_sigma:
                rec.signal_sigma[c] = cut(pred[c]["signal_sigma"])
            elif sigma_const is not None:
                rec.signal_sigma[c] = np.full_like(rec.signal_mu[c], sigma_const)
        if has_peak:
            rec.peak_score[c] = cut(pred[c]["peak_score"])
        else:
            rec.peak_score[c] = rec.signal_mu[c] if has_signal else rec.mu[c]
    rec.has_peak_head = bool(has_peak)
    sig_src = ("track" if has_own_sigma else
               "sigma_table" if (has_signal and sigma_const is not None) else
               "none" if has_signal else "n/a")
    return rec, sig_src


def _expected(source: EvalSource) -> Dict[str, Tuple[Pair, str]]:
    """`{directory name: (pair, assay)}` — every track the regime DECLARES, and nothing else.

    Built from `source.pairs`/`source.targets`, so the same rule that decides what a checkpoint is
    scored on decides what an external root must contain: an assay the target cell has and the input
    cell does not. Matching by this map rather than by parsing a directory name keeps D16 — a
    biosample name is an opaque id and nothing here splits one apart to learn something about it.
    """
    out: Dict[str, Tuple[Pair, str]] = {}
    for pair in source.pairs("impute"):
        for a in source.targets(pair, "impute"):
            out[track_dirname(pair, source.assays[a])] = (pair, source.assays[a])
    return out


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def score_external(source: EvalSource, pred_root: Path | str, *, seed: int = 0,
                   c_index_pairs: int = 200_000, varpool_root: Optional[Path | str] = None,
                   varpool_corpus: str = "eic", sigma_table: Optional[Mapping[str, Any]] = None,
                   sigma_table_path: Optional[str] = None, allow_missing: bool = False,
                   batch_windows: int = 4, with_curve: bool = False,
                   crps_approx: Optional[int] = None, crps_seed: int = 0,
                   truth_root: Optional[Path | str] = None,
                   held_out_chroms: Optional[Sequence[str]] = None,
                   progress: bool = False) -> Dict[str, Any]:
    """Score a whole prediction root and return `run_bench`'s result shape (§4.2).

    Same keys, same nesting, `provenance.method` from the manifest — so a leaderboard row from a
    rival and a leaderboard row from CANDI are read by the same code and diffed against each other
    without a translation step. `ranking` is `None` here as it is in `run_bench`: the rank
    aggregation is the CLI's `--competitors` path and is inert without a competitor table (D4).

    **`truth_root` swaps the truth, not the instrument** (`plan/BENCHMARK_DESIGN.md` §4.1). Given a
    §4.1-layout root of the challenge's own blind-truth tracks, the pval truth is read from there
    instead of from the store, cut with the same index as the prediction, and scored by the same
    `score_track`. That is what puts CANDI and a 2019 entrant on one row: the entrants were ranked
    against those bigwigs, and our store's `-log10 p` is our own MACS2 recomputation of the same
    experiments. The store remains the authority for the GRID, the declared track list and the
    provenance — only the numbers being compared against change. `provenance.truth` says which.

    **`held_out_chroms` turns one pass into two aggregations**, exactly as `run_bench` does it:
    `per_track`/`macro`/`panels` carry the held-out scope and a `genome_wide` block carries the
    same over every scored chromosome. Left `None`, there is one scope and no block — §4's blanking
    rule, with the reason written into `provenance.scope`.
    """
    root = Path(pred_root)
    if not root.is_dir():
        raise ExternalError(f"--pred {root} is not a directory")
    # t77 moved `cross_cell` off the source and onto the pair: a regime may declare some pairs that
    # cross cells and some that do not, so "is this source cross-cell" is only answerable as "does
    # any declared pair cross". `EvalSource.cross_cell(kind)` no longer exists.
    if not any(cross_cell(p, "impute") for p in source.pairs("impute")):
        raise ExternalError(
            f"{getattr(source, 'regime_path', source)} declares no `eval_pairs`. The external "
            f"format is `<input>{SEP}<target>{SEP}<assay>` — without declared pairs there is no "
            f"input cell to name, and imputation would be leave-one-assay-out inside one biosample, "
            f"a task no rival was asked to perform. Score that regime with a checkpoint instead.")

    manifest = read_manifest(root)
    expected = _expected(source)
    found = sorted(p.name for p in root.iterdir() if p.is_dir())
    unknown = [d for d in found if d not in expected]
    if unknown:
        raise ExternalError(
            f"{root}: {len(unknown)} track directory(ies) name no declared pair x target assay — "
            f"{unknown[:5]}. The regime declares {len(expected)} tracks. A directory whose "
            f"biosamples are not a declared eval pair, or whose assay the input cell already "
            f"carries, is not a track this regime can score.")
    have = set(found)
    truth_sha: Optional[str] = None
    troot: Optional[Path] = None
    if truth_root is not None:
        troot = Path(truth_root)
        if not troot.is_dir():
            raise ExternalError(f"--truth-root {troot} is not a directory")
        # The manifest is read to be CHECKED (`kind: "truth"`) and hashed; `provenance.truth` names
        # it by hash rather than copying it, since the root itself is the thing to go back to.
        _truth_manifest, truth_sha = read_truth_manifest(troot)
        # A track the TRUTH root does not cover is a hole in exactly the sense D2 means, so it
        # joins the same `missing` list rather than getting a second, quieter one.
        have &= {p.name for p in troot.iterdir() if p.is_dir()}
    missing = [d for d in sorted(expected) if d not in have]
    if missing and not allow_missing:
        where = f"{root}" if truth_root is None else f"{root} / --truth-root {troot}"
        raise ExternalError(
            f"{where} is missing {len(missing)} of the {len(expected)} declared tracks — "
            f"{missing[:5]}. A panel scored with holes in it reads as a whole panel in every table "
            f"downstream (D2). Emit them, or pass --allow-missing to record the gap in "
            f"provenance.missing_tracks and score what is there.")

    gene = ann.gene_annotations()
    enh = ann.enhancer_annotations()
    chroms = list(source.eval_chroms)
    n_bins = {c: source.n_bins(c) for c in chroms}
    # t89 — the source's eval scope, or `None` for every bin. Taken off the SOURCE rather than
    # added as a parameter here: `stream_truth` already walks `source.windows()`, so the truth is
    # cut by opening the source and the prediction has to be cut by the same index or the two would
    # not line up. A caller selects the cheap scope by opening the source with `eval_regions=`.
    scoped = {c: source.scored_bins(c) for c in chroms}
    bins = None if any(v is None for v in scoped.values()) else scoped
    var_cache: Dict[Tuple[str, str], object] = {}

    # §4's two aggregations, shaped exactly as `run_bench` shapes them. The split happens at the
    # TRACK — no measure is linear in position, so a genome-wide number cannot be narrowed to a
    # held-out one afterwards.
    scored_chroms = tuple(chroms)
    if held_out_chroms is None:
        held = scored_chroms
    else:
        held = tuple(c for c in scored_chroms if c in set(held_out_chroms))
        absent = [c for c in held_out_chroms if c not in scored_chroms]
        if absent:
            raise ExternalError(
                f"--held-out-chroms names {absent}, which this run does not score "
                f"({list(scored_chroms)}). The held-out scope is a subset of what was predicted; "
                f"naming a chromosome outside it would rank on positions that were never scored.")
        if not held:
            raise ExternalError("--held-out-chroms selected nothing from the scored chromosomes")
    split = len(held) < len(scored_chroms)
    if split and getattr(source, "eval_regions", None) is not None:
        raise ExternalError(
            f"--held-out-chroms splits the scope in two, but this source is restricted to "
            f"{source.eval_regions.resolved}. `genome_wide` means every bin of every scored "
            f"chromosome and would be a region cut under that name. Score the leaderboard at full "
            f"coverage (open the source without eval_regions=), or drop --held-out-chroms.")

    per_track: Dict[str, Dict[str, Dict[str, object]]] = {}
    per_track_gw: Dict[str, Dict[str, Dict[str, object]]] = {}
    binarised: Dict[str, List[Tuple[str, np.ndarray, np.ndarray, int]]] = {}
    sigma_sources: Dict[str, str] = {}
    todo = [(pair, assay, d) for d, (pair, assay) in sorted(expected.items()) if d in have]
    by_pair: Dict[Pair, List[Tuple[str, str]]] = {}
    for pair, assay, d in todo:
        by_pair.setdefault(pair, []).append((assay, d))

    for pi, (pair, rows) in enumerate(by_pair.items()):
        cols = [source.assays.index(assay) for assay, _ in rows]
        # Under a truth root the store is never read for truth at all — the walk it would do is
        # the expensive half of this entry point, and none of what it returns would be used.
        truth = (None if troot is not None else
                 stream_truth(source, pair, cols, batch_windows=batch_windows))
        for (assay, dirname), col in zip(rows, cols):
            pred = read_track_arrays(root / dirname, chroms, n_bins)
            if troot is None:
                truth_here: Mapping[str, Mapping[str, np.ndarray]] = truth[col]
            else:
                truth_here = {c: {"pval": v["signal_mu"]} for c, v in
                              read_truth_root_arrays(troot / dirname, chroms, n_bins).items()}
            rec, sig_src = build_record(pair, assay, chroms, truth_here, pred, sigma_table,
                                        bins=bins, truth_signal_only=troot is not None)
            # The D7 pool is aligned bin-for-bin with the WHOLE chromosome, so under a scope it
            # would be the one vector still on the genomic grid while the track is not. Dropped
            # rather than gathered: `msevar` is a leaderboard key and the scope is for selection.
            var = (None if bins is not None else
                   _varpool(varpool_root, varpool_corpus, assay, rec.chroms, n_bins, var_cache))
            common = dict(gene_annotations=gene, enh_annotations=enh, var=var, seed=seed,
                          c_index_pairs=c_index_pairs, with_curve=with_curve,
                          signal_target_transform=SIGNAL_TARGET_TRANSFORM,
                          crps_approx=crps_approx, crps_seed=crps_seed)
            held_here = tuple(c for c in rec.chroms if c in set(held))
            per_track[rec.key] = _withhold(score_track(rec, chroms=held_here or None, **common),
                                           troot is not None)
            if split:
                per_track_gw[rec.key] = _withhold(score_track(rec, chroms=rec.chroms, **common),
                                                  troot is not None)
            sigma_sources[rec.key] = sig_src
            # `panel_specificity` compares a `>= 2` CALL against MACS2 peak membership, and a
            # challenge truth root has no peak calls — so the panel block is absent there rather
            # than computed against the zero carriers.
            bits = None if troot is not None else _binarise(rec, SIGNAL_TARGET_TRANSFORM)
            if bits is not None:
                binarised.setdefault(assay, []).append((rec.key, *bits))
        del truth
        if progress:
            print(f"[bench.external] {pi + 1}/{len(by_pair)} {pair} — {len(rows)} track(s)",
                  flush=True)

    result: Dict[str, Any] = {
        "provenance": {
            **source.provenance(),
            "suite": "candi.bench.external",
            "method": manifest["method"],
            "manifest": manifest,
            "pred_root": str(root),
            "seed": int(seed),
            "kinds": ["impute"],
            # No C: every covariate instrument re-decodes a perturbed prompt, which needs the model.
            "blocks": ["E", "P", "D", "B"],
            "depth_center": source.depth_center(),
            "annotation_assets": ann.verify_assets(),
            "signal_target_transform": SIGNAL_TARGET_TRANSFORM,
            "pval_pred_space": D.SIGNAL_EVAL_SPACE,
            # §4.1 — an external `signal_mu` arrives in `-log10 p` already, so no inversion was
            # applied. `"external"` rather than `"none"` so a reader can tell a rival's row (never
            # transformed by us) from a CANDI row whose head was trained in the eval space.
            "pred_inversion": "external",
            # ABSENT WHEN THE CLOSED FORM WAS USED — see the same block in `harness.run_bench`. The
            # presence of these three keys is what says a count-arm `crps` in this file is an
            # ESTIMATE, and carries the two numbers (k, seed) a re-run would need.
            **({} if crps_approx is None else
               {"crps_estimator": "fair_sampled", "crps_k": int(crps_approx),
                "crps_seed": int(crps_seed)}),
            "declared_tracks": len(expected),
            "missing_tracks": missing,
            "allow_missing": bool(allow_missing),
            # WHICH TRUTH these numbers were measured against. Written on both paths, because a
            # score file that carries the key only when the answer is unusual leaves every older
            # file ambiguous.
            "truth": ({"source": "store"} if troot is None else
                      {"source": "challenge", "root": str(troot), "manifest_sha256": truth_sha}),
            "sigma_table": (None if sigma_table is None else
                            {"path": sigma_table_path, "method": sigma_table.get("method"),
                             "fitted_on": sigma_table.get("fitted_on"),
                             "assays": sorted(sigma_table.get("sigma", {}))}),
            "sigma_source": sigma_sources,
            "point_only_tracks": sorted(k for k, v in sigma_sources.items() if v == "none"),
            "msevar": ({"pool": varpool_corpus,
                        **{k[1]: v for k, v in var_cache.items() if k[0] == "__members__"}}
                       if varpool_root else
                       {"pool": None,
                        "note": "no --varpool given, so msevar is ABSENT rather than 0.0"}),
        },
        "tracks": sorted(per_track),
        "per_track": per_track,
        "macro": {arm: macro_mean(per_track, arm) for arm in ARMS},
        "panels": {arm: panel_macros(per_track, arm) for arm in ARMS},
        "panel": panel_specificity(binarised) if binarised else {},
        "ranking": None,
    }
    result["provenance"]["scope"] = {
        "ranked": SCOPE_HELD_OUT,
        "held_out_chroms": list(held),
        "scored_chroms": list(scored_chroms),
        "genome_wide_computed": bool(split),
        "note": (
            "`per_track`, `macro` and `panels` are the HELD-OUT scope, which is the ranked number "
            "(plan/BENCHMARK_DESIGN.md 4). `genome_wide` carries the same three over every scored "
            "chromosome."
            if split else
            "One scope only: the run scored exactly the held-out chromosomes, so there is no "
            "genome-wide aggregation to make. Under 4's blanking rule a method fit at every "
            "position is run this way on purpose -- its genome-wide number would be a memorisation "
            "score, so it is NOT COMPUTED rather than computed and withheld."),
    }
    if split:
        result["genome_wide"] = {
            "chroms": list(scored_chroms),
            "per_track": per_track_gw,
            "macro": {arm: macro_mean(per_track_gw, arm) for arm in ARMS},
            "panels": {arm: panel_macros(per_track_gw, arm) for arm in ARMS},
            "note": "Comparability with a literature that scores at the positions it fits. Not "
                    "ranked, and carries the per-cell in-sample fraction on the board (4).",
        }
    return result


def _withhold(arms: Dict[str, Dict[str, object]], truth_signal_only: bool
              ) -> Dict[str, Dict[str, object]]:
    """Remove every key of `WITHHELD_WITHOUT_PEAK_TRUTH` when the truth carried no peak calls.

    The identity on the store path — it hands back the caller's own dict untouched — so a run with
    no `--truth-root` produces the row it always produced, key for key.
    """
    if not truth_signal_only:
        return arms
    for row in arms.values():
        for k in WITHHELD_WITHOUT_PEAK_TRUTH:
            row.pop(k, None)
    return arms


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m candi.bench.external",
        description="Score externally-produced prediction tracks with candi.bench's own "
                    "instruments (RIVALS_PLAN.md §4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--store", required=True,
                   help="a CANDI_STORE REGIME FILE (json) declaring `eval_pairs`. Truth, grid and "
                        "the declared track list all come from it.")
    p.add_argument("--pred", required=True, help="prediction root: manifest.json + one dir per track")
    p.add_argument("--out", required=True, help="results JSON to write")
    p.add_argument("--truth-root", default=None,
                   help="score against SOMEBODY ELSE'S TRUTH: a §4.1-layout root of `signal_mu` "
                        "tracks (`tools/challenge_bigwigs.py truth-root`) whose manifest carries "
                        "`kind: \"truth\"`. The store still owns the grid, the declared track list "
                        "and the provenance; only what the prediction is compared against changes. "
                        "The challenge distributed signal and no peak calls, so the count arm and "
                        "every peak-derived key are ABSENT rather than nan. Without this flag "
                        "provenance.truth.source is `store`.")
    p.add_argument("--chroms", default=None,
                   help="comma-separated subset of the eval chromosomes (default: all of them). "
                        "P2 (genome-wide) is this flag with every chromosome the store carries.")
    p.add_argument("--held-out-chroms", default=None,
                   help="comma-separated: the RANKED scope (plan/BENCHMARK_DESIGN.md §4), e.g. "
                        "chr20,chr21,chr22. Must be a subset of what is scored. Given a proper "
                        "subset, one pass yields two aggregations -- `macro`/`panels` are the "
                        "held-out numbers and a `genome_wide` block carries the same over every "
                        "scored chromosome. Omitted, or naming everything scored, there is one "
                        "scope and no genome-wide block.")
    p.add_argument("--biosamples", default=None,
                   help="comma-separated eval biosamples (default: the regime's)")
    p.add_argument("--sigma-table", default=None,
                   help="B1a — {method, fitted_on, sigma: {assay: value}}. Fills a constant sigma "
                        "for a track that has signal_mu and no signal_sigma. Without it a "
                        "point-only track carries the E and P blocks and NO gauss_suite keys.")
    p.add_argument("--varpool", default=None,
                   help="root of the D7 msevar variance pools. Without it msevar is ABSENT rather "
                        "than the organizers' bare 0.0.")
    p.add_argument("--varpool-corpus", default="eic")
    p.add_argument("--allow-missing", action="store_true",
                   help="score a root that does not cover every declared pair. The gap is recorded "
                        "in provenance.missing_tracks either way; without this flag it is fatal.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--c-index-pairs", type=int, default=200_000)
    p.add_argument("--crps-approx", type=int, default=None, metavar="K",
                   help="score the COUNT arm's CRPS from K sampled draws per bin instead of the "
                        "closed form (candi.metrics.nb_crps_sampled). Off by default. It buys the "
                        "genome-wide panel: the closed form is ~2.6 h/track on P2 and the sampler "
                        "is minutes, and it stays finite at the n = 1e6 Poisson floor where the "
                        "closed form is NaN. Only `crps`, `crps_oracle_scaled`, "
                        "`crps_oracle_scaled_and_n`, `scale_error` and `beats_marginal` change; "
                        "`marg_crps`, `nb_nll`, ece, coverage and the C-index do not. Recorded as "
                        "provenance.crps_estimator/crps_k/crps_seed.")
    p.add_argument("--crps-seed", type=int, default=0,
                   help="RNG seed for --crps-approx. Inert without it.")
    p.add_argument("--batch-windows", type=int, default=4,
                   help="windows per truth read. Throughput only — every bin is read either way.")
    p.add_argument("--with-curve", action="store_true",
                   help="also emit the full correspondence curve per track (large)")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    from candi.bench.cli import jsonable

    a = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    chroms = tuple(c.strip() for c in a.chroms.split(",")) if a.chroms else None
    bios = tuple(b.strip() for b in a.biosamples.split(",")) if a.biosamples else None
    held = (tuple(c.strip() for c in a.held_out_chroms.split(","))
            if a.held_out_chroms else None)
    table = read_sigma_table(a.sigma_table)
    source = open_source(store=a.store, chroms=chroms, biosamples=bios)
    try:
        result = score_external(
            source, a.pred, seed=a.seed, c_index_pairs=a.c_index_pairs,
            varpool_root=a.varpool, varpool_corpus=a.varpool_corpus, sigma_table=table,
            sigma_table_path=a.sigma_table, allow_missing=a.allow_missing,
            batch_windows=a.batch_windows, with_curve=a.with_curve,
            crps_approx=a.crps_approx, crps_seed=a.crps_seed,
            truth_root=a.truth_root, held_out_chroms=held, progress=not a.quiet)
    finally:
        source.close()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsonable(result), indent=2))
    if not a.quiet:
        prov = result["provenance"]
        print(f"[bench.external] method={prov['method']} — {len(result['tracks'])} of "
              f"{prov['declared_tracks']} declared tracks", flush=True)
        for arm in ARMS:
            m = result["macro"].get(arm) or {}
            if m:
                head = ", ".join(f"{k}={m[k]:.4f}" for k in
                                 ("mse", "gwcorr", "crps", "nb_nll", "gaussian_nll",
                                  "bernoulli_nll")
                                 if k in m and np.isfinite(m[k]))
                print(f"[bench.external] macro {arm}: {m['n_tracks']} tracks — {head}", flush=True)
        print(f"[bench.external] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
