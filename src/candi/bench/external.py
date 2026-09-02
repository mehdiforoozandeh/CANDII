"""external — score prediction tracks somebody else wrote to disk (`RIVALS_PLAN.md` §4).

    python -m candi.bench.external --store regime.json --pred <pred_root> --out scores.json
    python -m candi.bench.external fill-panels --v store.V_.json --b store.B_.json

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

**And one thing it does with no scoring at all.** `fill-panels` measures a `V_` pass's `V_matched`
from the sibling `B_` pass's already-scored rows, because §5.2's matched panel is defined by the
assays `B_` poses and the two panels are scored in separate passes here. See the block above
`panel_union` for why that is an aggregation and not a re-score.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from candi.bench import annotations as ann
from candi.bench import distributional as D
from candi.bench.harness import (
    ARMS, SCOPE_HELD_OUT, EvalSource, Pair, TrackRecord, _binarise, _varpool, cross_cell,
    macro_mean, open_source, panel_macros, panel_of, panel_specificity, score_track, track_key,
)

__all__ = [
    "PRED_ARRAYS", "WITHHELD_WITHOUT_PEAK_TRUTH", "FILL_PANELS", "track_dirname", "read_manifest",
    "read_truth_manifest", "read_sigma_table", "read_track_arrays", "read_truth_root_arrays",
    "stream_truth", "build_record", "score_external", "panel_union", "fill_panels",
    "build_parser", "build_fill_parser", "main_fill_panels", "main",
]

#: The arrays §4.1 recognises inside a `chr*.npz`. Anything else in the file is ignored — a producer
#: may carry its own diagnostics alongside — but a file with NONE of these is an error, because a
#: track that predicts nothing is not a track.
PRED_ARRAYS: Tuple[str, ...] = ("signal_mu", "signal_sigma", "mu", "n", "peak_score")

#: §4.1 — the directory name is the bench `track_key` with `|` swapped for the filesystem-safe `__`.
#: `kind` is `impute` and is implied: `denoise` has no external producer, since a rival that denoises
#: a cell's own tracks is a different task from the one this leaderboard ranks.
SEP = "__"

#: The one named sub-command. `python -m candi.bench.external --store … --pred … --out …` is the
#: default and is NOT spelled `score`: four launchers already invoke it that way, and a required
#: positional would break every one of them. So the dispatch below is "first token names the
#: sub-command, or there is no sub-command", not an `add_subparsers`.
FILL_PANELS = "fill-panels"

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
# fill-panels — `V_matched` measured from the sibling `B_` pass (§5.2)
# ---------------------------------------------------------------------------
# `harness.panel_macros` computes the three §5.2 numbers from ONE scored pass, and the matched
# panel's assay set is MEASURED from the `B_` rows of that pass rather than listed — a hard-coded
# list would go stale the first time the panel moves, and silently.
#
# The rivals programme scores the two panels in SEPARATE passes: the regimes are panel-derived
# (`tools/declare_eval_pairs.py split`), and the prediction roots are split too. A `V_` pass
# therefore contains no `B_` row, so `panel_macros` measures an EMPTY matched set and every
# `V_matched` block on the board is blank — the one number §5.3's reading rule needs.
#
# Nothing here re-scores. Both passes already carry their `per_track` blocks, and a per-track score
# is independent of which other tracks shared the pass, so `V_matched` can be measured from the
# sibling pass's rows by handing `panel_macros` the UNION of the two tables. That is the whole
# mechanism: the same function, the same definition, more rows.

def _renan(per_track: Mapping[str, Mapping[str, Mapping[str, Any]]]
           ) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Undo `cli.jsonable`'s `NaN -> null` on the NUMERIC keys of a scored per-track table.

    JSON has no NaN, so every non-finite score is written `null` — and `macro_mean` reads a row by
    asking `isinstance(v, (int, float, bool))`. A `null` is neither, so a key that is nan in ONE
    row and finite in another would be dropped from that row's contribution here and averaged in a
    joint pass: the same rows, two different numbers. Worse, `macro_mean` then calls
    `float(row[k])` on it and raises.

    So a `null` is restored to a nan exactly when the key is a real (non-bool) number in some row
    of the table — which is what it must have been. A `null` under a key that is numeric NOWHERE is
    left alone: `bin_scope` is `None` on an unscoped run and is not a measure, and both spellings
    are equally invisible to `macro_mean`.

    Call it on the MERGED table, never on one side: a key finite only in the `B_` pass still has to
    restore the `V_` pass's nulls, or the union would disagree with a joint pass.
    """
    numeric: Dict[str, Set[str]] = {}
    for arms in per_track.values():
        for arm, row in arms.items():
            for k, v in row.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    numeric.setdefault(arm, set()).add(k)
    return {
        key: {arm: {k: (float("nan") if v is None and k in numeric.get(arm, ()) else v)
                    for k, v in row.items()}
              for arm, row in arms.items()}
        for key, arms in per_track.items()
    }


def _row_arms(per_track: Mapping[str, Mapping[str, Any]]) -> Set[str]:
    return {str(arm) for arms in per_track.values() for arm in arms}


def _panel_counts(per_track: Mapping[str, Mapping[str, Any]]) -> Dict[Optional[str], int]:
    """How many rows each §5.2 panel holds, by the TARGET cell's prefix (`harness.panel_of`)."""
    out: Dict[Optional[str], int] = {}
    for key in per_track:
        fields = str(key).split("|")
        p = panel_of(fields[1]) if len(fields) >= 3 else None
        out[p] = out.get(p, 0) + 1
    return out


def _regime_family(prov: Mapping[str, Any]) -> Optional[str]:
    """`regime.eic_r1.V_.json` -> `regime.eic_r1` — the two panel regimes' common ancestor.

    `tools/declare_eval_pairs.py split` writes `<workspace>/regime.<name>.<panel>.json`, so the
    derived siblings differ from each other in exactly one dot-separated segment: the panel, which
    is the only segment that ends in `_`. Dropping those segments is what lets a `V_` json and a
    `B_` json be recognised as two halves of ONE exam without either one carrying a shared id.

    `None` when the name carries no panel segment — the caller refuses rather than guessing.
    """
    regime = prov.get("regime")
    if not regime:
        return None
    stem = Path(str(regime)).name
    if stem.endswith(".json"):
        stem = stem[: -len(".json")]
    parts = [s for s in stem.split(".") if not s.endswith("_")]
    return ".".join(parts) if len(parts) < len(stem.split(".")) else None


def panel_union(v_per_track: Mapping[str, Mapping[str, Mapping[str, Any]]],
                b_per_track: Mapping[str, Mapping[str, Mapping[str, Any]]],
                *, what: str = "per_track") -> Dict[str, Dict[str, Dict[str, Any]]]:
    """`{**B, **V}`, with a key COLLISION refused and `jsonable`'s nulls restored.

    A shared `track_key` means the same experiment was scored in both passes, and a merge would
    keep one of the two silently. Whichever one that is, the union is no longer "the rows a joint
    pass would have had", so it is an error rather than a precedence rule.
    """
    clash = sorted(set(v_per_track) & set(b_per_track))
    if clash:
        raise ExternalError(
            f"the two passes share {len(clash)} {what} key(s) — {clash[:5]}. The `V_` and `B_` "
            f"passes must score DISJOINT track sets (that is what makes their union the rows one "
            f"joint pass would have held); a shared key means one of the two files is not the "
            f"sibling it claims to be, or a regime declares a pair on both panels.")
    return _renan({**b_per_track, **v_per_track})


def fill_panels(v: Dict[str, Any], b: Mapping[str, Any], *, b_json: Path | str,
                filled: Optional[str] = None) -> Dict[str, Any]:
    """Fill a `V_` pass's `V_matched` from the sibling `B_` pass's SCORED ROWS. Mutates `v`.

    What is written is exactly `harness.panel_macros(union, arm)["V_matched"]` — the block with its
    `matched_to`, its `note` and `ranked: False` — and nothing else. `V_breadth` is the `V_` pass's
    own and is not recomputed (it is the same rows either way). `B` is NOT written: the `B_` json is
    the file that describes the `B_` panel, and a `V_` json that also carried `B`'s numbers would be
    two rows in one file, which is the thing the split passes exist to keep apart.

    The refusals are all one question — are these two files halves of ONE exam? Same corpus, same
    assay order, same chromosomes, same scored POSITIONS, same method, same truth, same arms, same
    regime family, disjoint tracks, and one of each panel. Anything else and the filled `V_matched`
    would be a number measured against a different exam's assay set.

    `genome_wide.panels` gets the same treatment when both files carry the block; carrying it on
    one side only means the two passes were given different `--held-out-chroms`, and the held-out
    aggregation they do agree on would still be two different scopes.
    """
    for name, obj in (("--v", v), ("--b", b)):
        for k in ("provenance", "per_track", "panels"):
            if k not in obj:
                raise ExternalError(f"{name} carries no `{k}` — that is not a bench score file.")
    vp, bp = v["provenance"], b["provenance"]

    for key in ("data_source", "store", "h5", "assays", "eval_chroms", "eval_scope", "method"):
        mine, theirs = vp.get(key), bp.get(key)
        if mine != theirs:
            raise ExternalError(
                f"the two passes disagree on `provenance.{key}` ({mine!r} vs {theirs!r}). "
                f"`V_matched` is the `V_` rows aggregated over the assays `B_` poses, so the two "
                f"files must be one exam scored in two passes — same corpus, same assay order, "
                f"same chromosomes, same scored positions, same method.")
    vt = (vp.get("truth") or {}).get("source")
    bt = (bp.get("truth") or {}).get("source")
    if vt != bt:
        raise ExternalError(
            f"`provenance.truth.source` is {vt!r} on --v and {bt!r} on --b. A challenge-truth row "
            f"and a store-truth row are two different exams and must never be quoted in one "
            f"column (EVAL.md), so one cannot supply the other's matched assay set.")
    if set(v["panels"]) != set(b["panels"]) or _row_arms(v["per_track"]) != _row_arms(
            b["per_track"]):
        raise ExternalError(
            f"the two passes carry different arms — panels {sorted(v['panels'])} vs "
            f"{sorted(b['panels'])}, scored rows {sorted(_row_arms(v['per_track']))} vs "
            f"{sorted(_row_arms(b['per_track']))}. An arm present on one side only (a σ-table "
            f"passed to one pass and not the other, say) would put an empty `V_matched` on the "
            f"board under a heading that has numbers everywhere else.")

    vfam, bfam = _regime_family(vp), _regime_family(bp)
    if vfam is None or bfam is None:
        raise ExternalError(
            f"cannot tell which regime family these two passes belong to — "
            f"{vp.get('regime')!r} / {bp.get('regime')!r}. A panel regime is named "
            f"`regime.<name>.<panel>.json` (`tools/declare_eval_pairs.py split`), and the panel "
            f"segment is what says the two files are siblings of one exam rather than two "
            f"revisions of it. Re-derive the regimes with `split` and re-score, or rename them.")
    if vfam != bfam:
        raise ExternalError(
            f"--v was scored on regime family {vfam!r} and --b on {bfam!r}. Two revisions of a "
            f"regime declare different panels; measuring one's matched set from the other would "
            f"quote an assay set no `V_` row was ever scored on.")

    vc, bc = _panel_counts(v["per_track"]), _panel_counts(b["per_track"])
    if not vc.get("V"):
        raise ExternalError(
            f"--v holds no row whose TARGET cell starts `V_` (panels present: "
            f"{sorted(str(k) for k in vc)}). There is no breadth panel to narrow. Did --v and --b "
            f"get swapped?")
    if not bc.get("B"):
        raise ExternalError(
            f"--b holds no row whose TARGET cell starts `B_` (panels present: "
            f"{sorted(str(k) for k in bc)}). The matched assay set is MEASURED from scored `B_` "
            f"rows, so this would fill `V_matched` with the same empty block it already has.")

    union = panel_union(v["per_track"], b["per_track"])
    for arm in sorted(v["panels"]):
        v["panels"][arm]["V_matched"] = panel_macros(union, arm)["V_matched"]

    gw = ("genome_wide" in v, "genome_wide" in b)
    if gw[0] != gw[1]:
        raise ExternalError(
            f"`genome_wide` is present on {'--v' if gw[0] else '--b'} and absent on "
            f"{'--b' if gw[0] else '--v'}. The block exists only when --held-out-chroms named a "
            f"proper subset of what was scored, so the two passes were run over different scopes "
            f"and neither aggregation is comparable with the other's.")
    if gw[0]:
        gwu = panel_union(v["genome_wide"]["per_track"], b["genome_wide"]["per_track"],
                          what="genome_wide.per_track")
        for arm in sorted(v["genome_wide"]["panels"]):
            v["genome_wide"]["panels"][arm]["V_matched"] = panel_macros(gwu, arm)["V_matched"]

    v["provenance"]["panels_from"] = {
        "b_json": str(b_json),
        # The prediction manifest is copied VERBATIM into a score file's provenance and is not
        # hashed there, so this is `None` on a pass that recorded no hash of its own rather than a
        # hash of a re-serialised dict — which would name bytes that never existed on disk.
        "b_pred_manifest_sha256": bp.get("pred_manifest_sha256", bp.get("manifest_sha256")),
        "filled": filled or datetime.date.today().isoformat(),
    }
    return v


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m candi.bench.external",
        description="Score externally-produced prediction tracks with candi.bench's own "
                    "instruments (RIVALS_PLAN.md §4).",
        epilog=f"There is one named sub-command, `{FILL_PANELS}` (see `{FILL_PANELS} --help`): it "
               f"fills a scored `V_` json's `V_matched` from the sibling `B_` pass. Scoring is the "
               f"default and has no sub-command name.",
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


def build_fill_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"python -m candi.bench.external {FILL_PANELS}",
        description="Recompute a scored `V_` json's `V_matched` from the sibling `B_` pass's "
                    "scored rows (plan/BENCHMARK_DESIGN.md §5.2). Scores nothing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--v", required=True,
                   help="the `V_` pass's score json. Its `panels[arm].V_matched` (and "
                        "`genome_wide.panels[arm].V_matched`, when the block is there) is the only "
                        "thing rewritten; `V_breadth` and `B` are left as they are.")
    p.add_argument("--b", required=True,
                   help="the SIBLING `B_` pass's score json — same corpus, same positions, same "
                        "method, same truth, same regime family, disjoint tracks. The matched "
                        "assay set is MEASURED from its scored `B_` rows, never listed.")
    p.add_argument("--out", default=None,
                   help="where to write. Default: rewrite --v in place, after copying the "
                        "original once to `<--v>.bak` (an existing .bak is never overwritten, so "
                        "the pre-fill file survives a second run).")
    p.add_argument("--quiet", action="store_true")
    return p


def main_fill_panels(argv: Sequence[str]) -> int:
    from candi.bench.cli import jsonable

    a = build_fill_parser().parse_args(list(argv))
    vpath, bpath = Path(a.v), Path(a.b)
    v = json.loads(vpath.read_text(encoding="utf-8"))
    b = json.loads(bpath.read_text(encoding="utf-8"))
    got = fill_panels(v, b, b_json=bpath)

    out = Path(a.out) if a.out else vpath
    if out.resolve() == vpath.resolve():
        bak = vpath.with_name(vpath.name + ".bak")
        if not bak.exists():
            # The bytes AS THEY ARE ON DISK, before anything is written over them. Once only: a
            # second fill must not turn the backup into a copy of an already-filled file.
            bak.write_bytes(vpath.read_bytes())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsonable(got), indent=2))
    if not a.quiet:
        for arm in sorted(got["panels"]):
            m = got["panels"][arm]["V_matched"]
            print(f"[bench.external] {FILL_PANELS} {arm}: V_matched over "
                  f"{m.get('matched_to')} — {m.get('n_experiments')} experiment(s)", flush=True)
        print(f"[bench.external] wrote {out}", flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    from candi.bench.cli import jsonable

    args = list(sys.argv[1:] if argv is None else argv)
    # The default command is the one four launchers already spell `--store … --pred … --out …`,
    # so the sub-command is recognised by its own name and nothing else changes shape.
    if args and args[0] == FILL_PANELS:
        return main_fill_panels(args[1:])

    a = build_parser().parse_args(args)
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
