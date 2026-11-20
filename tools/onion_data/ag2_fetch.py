"""AGf -- fetch AlphaGenome 2 (AG2) predictions through the API into a restartable cache, and pack them as
method track files (onion plan Amendment 5, pinned interface F). AG2 outputs are evaluation data only.

    ag2_fetch.py fetch  --conditions <tsv> (--evalset <json> | --chroms chr20,chr21,chr22)
                        --outputs chip_histone[,dnase,atac] --cache <dir> [--model all_folds]
                        [--max-calls 2000] [--workers 5] [--chunk 25] [--strict] [--dry-run]
    ag2_fetch.py status --conditions <tsv> (--evalset <json> | --chroms ...) --cache <dir>
                        --outputs chip_histone[,dnase,atac] [--model all_folds] [--strict]

`--dry-run` prints the planned calls per condition and reads no key and creates no client. `--max-calls`
counts every attempt: a retry after a quota error resends its whole chunk and counts again. An unreadable
part file (truncated, corrupt) is named on stderr and its windows count as missing (they are refetched);
with `--strict` it is an error.
    ag2_fetch.py pack   --cache <dir> --evalset <json> --audit <ag2_eic_audit.tsv> --tables <dir>
                        --split val|blind --out <dir> [--model all_folds]

Ported from Max's sprint 11 (github mlibbrecht/01_2026-10-19_research at bde905a, folder
projects/03_2026-10-20_epi_imputation/sprints/11_2026-11-11_alphagenome2/):
  scripts/api/fetch_ag2.py l.27-28  (1 Mb window, 128 bp, chunks of 25 intervals, 5 workers, stride W/2)
                           l.42-50  (`windows`: Max's tiling; each 128 bp bin from the nearest window centre)
                                    -> `tile_windows`
                           l.53-66  (`predict`: 8 tries, wait min(600, 30 * 2^k)) -> `_predict_with_backoff`
                           l.88-95  (float16 clipped at 6e4; temporary name then rename) -> `_write_part`
  scripts/api/pack_ag2.py  l.5-7, l.29-36 (matched channel = the target's mark and read type; the other
                                    read type kept too) -> `pack`; here the channels come from the leakage
                                    audit (`matched_channel`, `otherend_channel`), which applies that rule.
                                    When the audit names no channel of the target's read type, the other
                                    read type is written as `matched.signal.npy` (plan section A) and
                                    `channels.tsv` says `matched_source = otherend_fallback`.
Changes against Max's code: the cache holds one file per chunk of windows (`part_<first window index>.npz`)
instead of one file per chromosome, so a kill loses at most the chunk in flight; windows can also be centred
on the regions of an onion eval set; only RESOURCE_EXHAUSTED, UNAVAILABLE and DEADLINE_EXCEEDED are retried;
an error is logged as its status code name only (Max's log quoted the message, which holds the account's
project number).

Windows. With `--evalset`: window i is 1,048,576 bp centred on region i (a region is `region_bins` 25 bp bins
from `start_bin * 25`), floored to the 128 bp grid and shifted inward at a chromosome end. With `--chroms`:
Max's tiling; the "region" of a window is the range of 128 bp bins it supplies.

Cache. `<cache>/<model>/<condition slug>/part_<first window index>.npz`, written as `tmp_part_*.npz` and
renamed. Keys: `condition`, `model`, `outputs`, `window_index` int64[n], `chroms` str[n], `coords` int64[n, 6]
= (window_start, window_end, region_start, region_end, chip_start, acc_start) in bp; per output
`<output>_names`, `<output>_modes` (the API's `imputation_mode` per channel); per window k of the part
`chip_histone_<k>` float16 [bins, channels] at 128 bp from `chip_start` (the region plus 32 bins each side;
values clipped at 6e4; NaN outside the window) and `dnase_<k>` / `atac_<k>` float32 [bins, channels], the 1 bp
output summed to the 25 bp grid from `acc_start` over the region. A window counts as cached when a part holds
it with every requested output. `<cache>/calls.tsv` appends `time, condition, window, status, seconds`.
Channel names are cached as the API returns them, `<condition>_<channel>`; `pack` strips the prefix in
`bare_channel` and matches the audit's bare channel names (a prefixed audit copy also works).

Exit codes of `fetch`: 0 when no window is missing; 75 when it stopped with windows missing (8 retryable errors
in a row on one chunk, or `--max-calls` reached): rerun the same command; 1 on a non-retryable API error;
2 on a refused key file or bad arguments. The cache is consistent after every exit. An exception from the
client (its creation or a call) is reported by its status code name or class name only, never by its text.

The key is read only from the file named by `AG2_KEY_FILE` (default `~/.secrets/alphagenome_api_key`), which
must have mode 600 and hold exactly one non-empty line. No command takes, prints or writes the key.

Module-level imports are the standard library and numpy only; `alphagenome` is imported inside `_client`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import stat
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

W = 1_048_576                   # dna_client.SEQUENCE_LENGTH_1MB
RES = 128                       # ChIP output resolution (bp)
STRIDE = W // 2                 # Max's tiling stride
MARGIN_BINS = 32                # 128 bp bins kept on each side of a region
BIN = 25                        # the onion grid (bp)
CLIP = 6e4                      # float16 range guard (Max)
MAX_TRIES = 8
WAIT0, WAIT_MAX = 30, 600
EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_RETRY = 0, 1, 2, 75
OUTPUTS = ("chip_histone", "dnase", "atac")
RETRYABLE = {"RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED"}
GRPC_CODES = ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "INVALID_ARGUMENT", "NOT_FOUND",
              "PERMISSION_DENIED", "UNAUTHENTICATED", "FAILED_PRECONDITION", "INTERNAL", "UNKNOWN",
              "CANCELLED", "ABORTED", "OUT_OF_RANGE", "UNIMPLEMENTED", "DATA_LOSS", "ALREADY_EXISTS")
DEFAULT_KEY_FILE = "~/.secrets/alphagenome_api_key"
CALLS_COLUMNS = ("time", "condition", "window", "status", "seconds")
CHANNEL_COLUMNS = ("measurement_id", "condition", "matched_channel", "otherend_channel",
                   "api_label_matched", "api_label_otherend",
                   "matched_source", "output_type", "chip_bin_value", "native_divisor")   # last four: additive
# The API value of a ChIP-seq 128 bp bin is the SUM of the per-bp fold change (API smoke, t267 section 2).
CHIP_NATIVE_DIVISOR = {"sum": RES, "mean": 1}
ASSAY_OUTPUT = {"DNase-seq": "dnase", "ATAC-seq": "atac"}     # every other assay: chip_histone
# hg38 chromosome lengths (UCSC); Max's table (fetch_ag2.py l.26) holds chr6, chr9, chr13, chr20-22 and agrees.
HG38 = {"chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555, "chr5": 181538259,
        "chr6": 170805979, "chr7": 159345973, "chr8": 145138636, "chr9": 138394717, "chr10": 133797422,
        "chr11": 135086622, "chr12": 133275309, "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
        "chr16": 90338345, "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
        "chr21": 46709983, "chr22": 50818468, "chrX": 156040895, "chrY": 57227415}


class KeyFileError(Exception):
    pass


class FetchError(Exception):
    pass


# ------------------------------------------------------------------------------------------- key and client

def read_key(path=None) -> str:
    """The API key from `path` (default: $AG2_KEY_FILE, else ~/.secrets/alphagenome_api_key). Refuses a file
    whose mode is not 600 or that does not hold exactly one non-empty line. The error text never holds the key."""
    p = Path(path or os.environ.get("AG2_KEY_FILE") or DEFAULT_KEY_FILE).expanduser()
    if not p.is_file():
        raise KeyFileError(f"key file {p} does not exist")
    mode = stat.S_IMODE(p.stat().st_mode)
    if mode != 0o600:
        raise KeyFileError(f"key file {p} has mode {mode:o}; it must be 600")
    lines = p.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1 or not lines[0].strip():
        raise KeyFileError(f"key file {p} must hold exactly one non-empty line")
    return lines[0].strip()


def _track(output, values, metadata):
    """(values, channel names, imputation_mode labels) of one output; refuses metadata without the label."""
    for col in ("name", "imputation_mode"):
        if col not in metadata:
            raise FetchError(f"{output}: the API metadata has no `{col}` column")
    return (np.asarray(values), np.asarray(metadata["name"]).astype(str),
            np.asarray(metadata["imputation_mode"]).astype(str))


class _Api:
    """Adapter over `alphagenome` 2.0.0 `DnaClient`: the one method `predict` that the fetch needs."""

    def __init__(self, model):
        from alphagenome.data import genome
        from alphagenome.models.v2 import dna_client
        self._model, self._genome, self._ot = model, genome, dna_client.OutputType

    def predict(self, intervals, condition, outputs, workers):
        res = self._model.predict_intervals(
            [self._genome.Interval(c, s, e) for c, s, e in intervals], conditions=[condition] * len(intervals),
            requested_outputs=[self._ot[o.upper()] for o in outputs], progress_bar=False, max_workers=workers)
        out = []
        for o in res:
            d = {}
            for name in outputs:
                t = getattr(o, name)
                d[name] = _track(name, t.values, t.metadata)
            out.append(d)
        return out


def _client(key: str, model: str):
    """The real client (`alphagenome` is imported here only)."""
    from alphagenome.models.v2 import dna_client
    return _Api(dna_client.create(key, model_version=dna_client.ModelVersion[model.upper()]))


def status_name(e: BaseException) -> str:
    """The gRPC status code name of an API error, else the exception class name. Never the message."""
    code = getattr(e, "code", None)
    if callable(code):
        try:
            name = getattr(code(), "name", None)
            if isinstance(name, str) and name in GRPC_CODES:
                return name
        except Exception:                                   # noqa: BLE001 -- a code() that fails is no code
            pass
    text = str(e)
    for name in GRPC_CODES:
        if name in text:
            return name
    return type(e).__name__


# ------------------------------------------------------------------------------------------------- windows

def slug(condition: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", condition).strip("-") or "condition"


def _window(index, chrom, ws, rs, re_):
    cs = rs // RES * RES - MARGIN_BINS * RES
    ce = -(-re_ // RES) * RES + MARGIN_BINS * RES
    acc = rs // BIN * BIN
    return {"index": index, "chrom": chrom, "ws": ws, "we": ws + W, "rs": rs, "re": re_, "cs": cs,
            "n_chip": (ce - cs) // RES, "as": acc, "n_acc": -(-re_ // BIN) - rs // BIN}


def evalset_windows(ev: dict, sizes=HG38) -> list[dict]:
    """One window per eval-set region, centred on it, floored to 128 bp, shifted inward at a chromosome end."""
    rb = int(ev["region_bins"])
    out = []
    for i, (chrom, sb) in enumerate(ev["regions"]):
        if chrom not in sizes:
            raise ValueError(f"no length for {chrom}")
        L = sizes[chrom]
        rs, re_ = int(sb) * BIN, (int(sb) + rb) * BIN
        if re_ - rs > W or L < W or re_ > L:
            raise ValueError(f"region {chrom}:{sb} does not fit a {W} bp window on {chrom}")
        ws = ((rs + re_) // 2 - W // 2) // RES * RES
        ws = max(0, min(ws, (L - W) // RES * RES))
        if not (ws <= rs and re_ <= ws + W):                 # a region the shifted window does not cover
            raise ValueError(f"region {chrom}:{sb} ([{rs}, {re_})) is not inside its window [{ws}, {ws + W})")
        out.append(_window(i, chrom, ws, rs, re_))
    return out


def tile_windows(chroms, sizes=HG38) -> list[dict]:
    """Max's tiling (fetch_ag2.py l.42-50), indexed across the chromosomes in the order given."""
    out = []
    for c in chroms:
        L = sizes[c]
        starts = list(range(0, L - W + 1, STRIDE))
        last = (L - W) // RES * RES
        if last > starts[-1]:
            starts.append(last)
        centres = [s + W // 2 for s in starts]
        cuts = [0] + [((a + b) // 2) // RES for a, b in zip(centres[:-1], centres[1:])] + [L // RES]
        for k, s in enumerate(starts):
            out.append(_window(len(out), c, s, cuts[k] * RES, cuts[k + 1] * RES))
    return out


def chip_cut(values, w) -> np.ndarray:
    """The 128 bp rows [chip_start, chip_start + n_chip * 128) of a window's ChIP output; NaN outside it."""
    out = np.full((w["n_chip"], values.shape[1]), np.nan, dtype=np.float64)
    off = (w["cs"] - w["ws"]) // RES
    lo, hi = max(0, off), min(values.shape[0], off + w["n_chip"])
    if hi > lo:
        out[lo - off:hi - off] = values[lo:hi]
    return out


def acc_cut(values, w) -> np.ndarray:
    """A window's 1 bp output summed to the 25 bp bins from acc_start over the region; NaN where not covered."""
    n = w["n_acc"] * BIN
    buf = np.full((n, values.shape[1]), np.nan, dtype=np.float64)
    off = w["as"] - w["ws"]
    lo, hi = max(0, off), min(values.shape[0], off + n)
    if hi > lo:
        buf[lo - off:hi - off] = values[lo:hi]
    return buf.reshape(w["n_acc"], BIN, values.shape[1]).sum(axis=1).astype(np.float32)


# --------------------------------------------------------------------------------------------------- cache

def cond_dir(cache, model, condition) -> Path:
    return Path(cache) / model / slug(condition)


def _part_files(d: Path):
    return sorted(d.glob("part_*.npz"), key=lambda p: int(p.stem.split("_")[1]))


def _coord_row(w):
    return [w["ws"], w["we"], w["rs"], w["re"], w["cs"], w["as"]]


def _read_part_header(f: Path):
    """(condition, outputs, window_index, chroms, coords) of a part file; zip or format errors propagate."""
    with np.load(f) as z:
        return (str(z["condition"]), set(z["outputs"].tolist()), z["window_index"].tolist(),
                [str(c) for c in z["chroms"]], z["coords"].tolist())


def cache_index(cache, model, condition, windows, strict=False) -> dict:
    """window index -> list of (part file, outputs held). Refuses a cache of another condition or other windows.
    A part file that cannot be read (truncated, corrupt) is named on stderr and its windows count as missing,
    so a fetch refetches them; with `strict` it is an error."""
    d = cond_dir(cache, model, condition)
    idx: dict[int, list] = {}
    for f in _part_files(d):
        try:
            cond_f, outs, wins, chroms, coords = _read_part_header(f)
        except (zipfile.BadZipFile, OSError, EOFError, ValueError, KeyError) as e:
            msg = f"unreadable part file {f} ({type(e).__name__})"
            if strict:
                raise FetchError(msg) from None
            print(f"warning: {msg}; its windows count as missing", file=sys.stderr, flush=True)
            continue
        if cond_f != condition:
            raise FetchError(f"{f} holds condition {cond_f!r}, not {condition!r}")
        for k, i in enumerate(wins):
            if i >= len(windows) or chroms[k] != windows[i]["chrom"] or coords[k] != _coord_row(windows[i]):
                raise FetchError(f"{f}: window {i} has other coordinates than this window set "
                                 f"(another eval set or tiling uses this cache; use another --cache)")
            idx.setdefault(i, []).append((f, outs))
    return idx


def missing_windows(cache, model, condition, windows, outputs, strict=False) -> list[int]:
    idx = cache_index(cache, model, condition, windows, strict=strict)
    need = set(outputs)
    return [w["index"] for w in windows if not any(need <= o for _, o in idx.get(w["index"], []))]


def _write_part(d: Path, condition, model, outputs, group, results) -> int:
    """Write one chunk of windows atomically; returns the count of clipped ChIP values."""
    d.mkdir(parents=True, exist_ok=True)
    arrs = {"condition": np.array(condition), "model": np.array(model), "outputs": np.array(list(outputs)),
            "window_index": np.array([w["index"] for w in group], dtype=np.int64),
            "chroms": np.array([w["chrom"] for w in group]),
            "coords": np.array([_coord_row(w) for w in group], dtype=np.int64).reshape(len(group), 6)}
    nclip = 0
    for o in outputs:
        # Channels are matched by name on every response, never by a stored index (the channel order is not
        # known to be stable): each window's columns are put in the order of the first window's names.
        names0, modes0 = [str(x) for x in results[0][o][1]], [str(x) for x in results[0][o][2]]
        if len(set(names0)) != len(names0):
            raise FetchError(f"{o}: duplicate channel names in a response")
        label0 = dict(zip(names0, modes0))
        arrs[f"{o}_names"], arrs[f"{o}_modes"] = np.asarray(names0, dtype=str), np.asarray(modes0, dtype=str)
        for k, (w, r) in enumerate(zip(group, results)):
            vals, names, modes = r[o]
            names, modes = [str(x) for x in names], [str(x) for x in modes]
            if sorted(names) != sorted(names0) or dict(zip(names, modes)) != label0:
                raise FetchError(f"{o}: the channels or their labels differ between windows of one call")
            vals = np.asarray(vals, dtype=np.float64)
            if vals.ndim != 2 or vals.shape[1] != len(names):
                raise FetchError(f"{o}: shape {vals.shape} does not match {len(names)} channel names")
            pos = {n: j for j, n in enumerate(names)}
            vals = vals[:, [pos[n] for n in names0]]
            if o == "chip_histone":
                if vals.shape != (W // RES, len(names0)):
                    raise FetchError(f"chip_histone: shape {vals.shape}, expected {(W // RES, len(names0))}")
                v = chip_cut(vals, w)
                nclip += int(np.nansum(v > CLIP))
                arrs[f"{o}_{k}"] = np.minimum(v, CLIP).astype(np.float16)
            else:
                if vals.shape != (W, len(names0)):
                    raise FetchError(f"{o}: shape {vals.shape}, expected {(W, len(names0))}")
                arrs[f"{o}_{k}"] = acc_cut(vals, w)
    first = group[0]["index"]
    tmp, final = d / f"tmp_part_{first}.npz", d / f"part_{first}.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, final)
    return nclip


def _log_calls(cache, condition, group, status, seconds):
    f = Path(cache) / "calls.tsv"
    new = not f.exists()
    with open(f, "a", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh, delimiter="\t", lineterminator="\n")
        if new:
            wr.writerow(CALLS_COLUMNS)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        for w in group:
            wr.writerow([now, condition, w["index"], status, f"{seconds:.2f}"])


# --------------------------------------------------------------------------------------------------- fetch

def _predict_with_backoff(client, cache, condition, group, outputs, workers, sleep, log, budget):
    """Max's retry loop (fetch_ag2.py l.53-66). Every attempt sends every interval of the chunk, so every
    attempt is counted in `budget` ({"calls", "max"}) before it is sent. Returns (results, "ok"),
    (None, "retries") after MAX_TRIES retryable errors, or (None, "max_calls") when the next attempt would pass
    the cap. A client exception is reported by its status code name only, never by its message."""
    intervals = [(w["chrom"], w["ws"], w["we"]) for w in group]
    for attempt in range(MAX_TRIES):
        if budget["calls"] + len(group) > budget["max"]:
            return None, "max_calls"
        budget["calls"] += len(group)
        t0 = time.time()
        try:
            res = client.predict(intervals, condition, list(outputs), workers)
        except FetchError:                                  # our own message (no client text in it)
            raise
        except Exception as e:                              # noqa: BLE001 -- classified by its status code
            name = status_name(e)
            del e
            _log_calls(cache, condition, group, name, time.time() - t0)
            if name not in RETRYABLE:
                raise FetchError(f"{condition}: windows {group[0]['index']}..{group[-1]['index']}: {name}") from None
            wait = min(WAIT_MAX, WAIT0 * 2 ** attempt)
            log(f"{condition}: {name}; try {attempt + 1} of {MAX_TRIES}, waiting {wait} s")
            sleep(wait)
            continue
        if len(res) != len(group):
            raise FetchError(f"{condition}: {len(res)} results for {len(group)} windows")
        _log_calls(cache, condition, group, "OK", time.time() - t0)
        return res, "ok"
    return None, "retries"


def plan(conditions, windows, outputs, cache, model="all_folds", strict=False) -> list[str]:
    """Dry run: the calls a fetch would send per condition (without retries); no key, no client."""
    lines, total = [], 0
    for c in conditions:
        m = len(missing_windows(cache, model, c, windows, outputs, strict=strict))
        total += m
        lines.append(f"{c}\tmissing windows: {m}, planned calls: {m}")
    lines.append(f"total planned calls: {total} (each retry after a quota error resends its chunk)")
    return lines


def fetch(conditions, windows, outputs, cache, *, model="all_folds", max_calls=2000, workers=5, chunk=25,
          client_factory=_client, sleep=time.sleep, key_file=None, log=None, strict=False) -> int:
    log = log or (lambda m: print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}", flush=True))
    Path(cache).mkdir(parents=True, exist_ok=True)
    todo = {c: missing_windows(cache, model, c, windows, outputs, strict=strict) for c in conditions}
    total = sum(len(v) for v in todo.values())
    log(f"{len(conditions)} conditions x {len(windows)} windows; {total} missing")
    if not total:
        return EXIT_OK
    key = read_key(key_file)
    try:
        client = client_factory(key, model)
    except Exception as e:                                  # noqa: BLE001 -- the message may echo the key
        name = status_name(e)
        del e
        raise FetchError(f"creating the client failed: {name}") from None
    finally:
        del key
    budget, t0 = {"calls": 0, "max": int(max_calls)}, time.time()
    for cond in conditions:
        miss = todo[cond]
        d = cond_dir(cache, model, cond)
        for i in range(0, len(miss), chunk):
            group = [windows[j] for j in miss[i:i + chunk]]
            res, why = _predict_with_backoff(client, cache, cond, group, outputs, workers, sleep, log, budget)
            if why == "max_calls":
                log(f"stopped: --max-calls {max_calls} reached after {budget['calls']} calls; rerun to continue")
                return EXIT_RETRY
            if why == "retries":
                log(f"stopped: {MAX_TRIES} retryable errors in a row on {cond}; the cache is consistent; "
                    f"rerun later")
                return EXIT_RETRY
            nclip = _write_part(d, cond, model, outputs, group, res)
            log(f"{cond}: windows {group[0]['index']}..{group[-1]['index']} cached ({nclip} values clipped); "
                f"{budget['calls']} calls in {time.time() - t0:.0f} s")
    left = sum(len(missing_windows(cache, model, c, windows, outputs, strict=strict)) for c in conditions)
    log(f"done: {left} windows missing")
    return EXIT_OK if not left else EXIT_RETRY


def status(conditions, windows, outputs, cache, model="all_folds", strict=False) -> list[str]:
    lines = []
    for c in conditions:
        m = len(missing_windows(cache, model, c, windows, outputs, strict=strict))
        n = len(windows)
        lines.append(f"{c}\twindows: {n}, cached: {n - m}, missing: {m}")
    return lines


# ---------------------------------------------------------------------------------------------------- pack

def resample_mean(x, src_start_bp, src_bin_bp, dst_start_bp, dst_bin_bp, n_dst) -> np.ndarray:
    """Overlap-weighted mean (ag2_space.resample_mean, restated in numpy; a test holds the two equal). A bin
    partly outside the source is the mean over its covered part; a bin wholly outside is NaN."""
    x = np.asarray(x, dtype=np.float64)
    F = np.concatenate([[0.0], np.cumsum(x * float(src_bin_bp))])
    edges = float(dst_start_bp) + np.arange(int(n_dst) + 1, dtype=np.float64) * float(dst_bin_bp)
    pos = np.clip((edges - float(src_start_bp)) / float(src_bin_bp), 0.0, float(x.size))
    integ = np.diff(np.interp(pos, np.arange(x.size + 1, dtype=np.float64), F))
    covered = np.diff(pos) * float(src_bin_bp)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(covered > 0, integ / np.where(covered > 0, covered, 1.0), np.nan)


def read_tsv(path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_tsv(path, cols, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)


def regions_sha256(ev: dict) -> str:
    """sha256 of the JSON of the region list and region_bins (pinned interface F)."""
    text = json.dumps({"region_bins": int(ev["region_bins"]), "regions": [[str(c), int(s)] for c, s in ev["regions"]]},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bare_channel(name: str, condition: str, *, required: bool = True) -> str:
    """The channel name without the `<condition>_` prefix. The API names every returned track
    `<condition>_<channel>` (e.g. `CL:0000047_Histone ChIP-seq-encode-H2BK15ac-single-ended`); the cache keeps
    that name as returned, and the audit and `prediction_channels.tsv` hold the bare channel (as
    `ag2_meta.channel_of`). This is the one place where the prefix is handled. `required=False` accepts an
    already bare name (audit tables; a prefixed audit copy is also accepted)."""
    prefix = f"{condition}_"
    name = str(name)
    if name.startswith(prefix):
        return name[len(prefix):]
    if required:
        raise FetchError(f"cached track name {name!r} does not start with {prefix!r}")
    return name


def _condition_tracks(cache, model, condition, windows, channels):
    """{channel: (float32 [n_regions * region_bins], api label)} for the channels asked, from the cache."""
    idx = cache_index(cache, model, condition, windows)
    out = {ch: [] for ch in channels}
    labels = {ch: set() for ch in channels}
    kinds = {ch: set() for ch in channels}
    opened: dict[Path, object] = {}
    try:
        for w in windows:
            parts = idx.get(w["index"])
            if not parts:
                raise FetchError(f"{condition}: window {w['index']} is not cached (run fetch until status "
                                 f"prints missing: 0)")
            for ch in channels:
                hit = None
                for f, outs in parts:
                    if f not in opened:
                        opened[f] = np.load(f)
                    z = opened[f]
                    for o in outs:
                        names = [bare_channel(n, condition) for n in z[f"{o}_names"].tolist()]
                        if ch in names:
                            hit = (z, o, names.index(ch), int(np.flatnonzero(z["window_index"] == w["index"])[0]))
                            break
                    if hit:
                        break
                if hit is None:
                    raise FetchError(f"{condition}: channel {ch!r} is in no cached output of window {w['index']}")
                z, o, j, k = hit
                labels[ch].add(str(z[f"{o}_modes"][j]))
                kinds[ch].add(o)
                n = w["re"] - w["rs"]
                if o == "chip_histone":
                    lo, hi = (w["rs"] - w["cs"]) // RES, -(-(w["re"] - w["cs"]) // RES)   # rows under the
                    v = resample_mean(z[f"{o}_{k}"][lo:hi, j].astype(np.float64),       # region only (the
                                      w["cs"] + lo * RES, RES, w["rs"], BIN, n // BIN)  # margins may be NaN)
                else:
                    a = (w["rs"] - w["as"]) // BIN
                    v = z[f"{o}_{k}"][a:a + n // BIN, j].astype(np.float64)
                out[ch].append(v)
            if len(opened) > 4:                              # parts are read in window order
                for zz in opened.values():
                    zz.close()
                opened.clear()
    finally:
        for zz in opened.values():
            zz.close()
    for ch in channels:
        if len(kinds[ch]) != 1:
            raise FetchError(f"{condition}: channel {ch!r} is found in several outputs {sorted(kinds[ch])}")
    return {ch: (np.concatenate(out[ch]).astype(np.float32), ";".join(sorted(labels[ch])), kinds[ch].pop())
            for ch in channels}


def pack(cache, evalset, audit, tables, split, out, model="all_folds", chip_bin_value="sum", log=print) -> dict:
    """Write `<out>/<regime>/` (pinned interface F, kind `ag2`) for the split's queryable targets.

    `matched.signal` and `otherend.signal` are the API values as returned, 128 bp to 25 bp by the overlap mean
    (plan B.2: raw, no transform; the recalibrated metrics are scale-free). `native` is AG2's own unit: for
    ChIP-seq the fold change per bp, i.e. the API value divided by CHIP_NATIVE_DIVISOR[chip_bin_value]
    (`sum`: the API value of a 128 bp bin is the sum of the per-bp fold change, measured by the API smoke
    `cruxvault/results/t267/API_SMOKE.md` section 2, so the divisor is 128; `mean`: 1); for DNase-seq and
    ATAC-seq the insertions per 25 bp bin, the same array as `matched.signal`. The convention is recorded in
    `tracks.json` and per row in `channels.tsv`."""
    if split not in ("val", "blind"):
        raise ValueError("--split is val or blind")
    if chip_bin_value not in CHIP_NATIVE_DIVISOR:
        raise ValueError(f"--chip-bin-value is one of {sorted(CHIP_NATIVE_DIVISOR)}")
    chip_div = CHIP_NATIVE_DIVISOR[chip_bin_value]
    ev_bytes = Path(evalset).read_bytes()
    ev = json.loads(ev_bytes)
    rb = int(ev["region_bins"])
    windows = evalset_windows(ev)
    meas = {r["measurement_id"]: r for r in read_tsv(Path(tables) / "measurements.tsv")}
    aud = {r["eic_track"]: r for r in read_tsv(audit)}
    by_regime: dict[str, list] = {}
    skipped: dict[str, int] = {}
    for t in ev["imputation_targets"]:
        if t.get("split") != split:
            continue
        m = meas.get(t["target"])
        if m is None:
            raise FetchError(f"target {t['target']} is not in measurements.tsv")
        a = aud.get(m.get("eic_track", ""))
        if a is None:
            reason = "not_in_audit"
        elif str(a.get("queryable", "0")) != "1":
            reason = f"status_{a.get('status', '')}"
        elif not (a.get("matched_channel") or a.get("otherend_channel")):
            reason = "no_channel_named"
        else:
            reason = ""
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        by_regime.setdefault(t["regime"], []).append((t["target"], m, a))
    need: dict[str, set] = {}
    for items in by_regime.values():
        for _, m, a in items:
            need.setdefault(a["condition"], set()).update(
                bare_channel(ch, a["condition"], required=False)
                for ch in (a.get("matched_channel"), a.get("otherend_channel")) if ch)
    tracks = {c: _condition_tracks(cache, model, c, windows, sorted(chs)) for c, chs in sorted(need.items())}
    base = {"method": "ag2", "kind": "ag2", "evalset_sha256": hashlib.sha256(ev_bytes).hexdigest(),
            "regions_sha256": regions_sha256(ev), "split": split, "region_bins": rb,
            "n_regions": len(ev["regions"]), "lambda_source": "none", "model": model,
            "evaluation_only": True, "skipped": dict(sorted(skipped.items())),
            "chip_bin_value": chip_bin_value, "chip_native_divisor": chip_div,
            "chip_bin_value_source": "cruxvault/results/t267/API_SMOKE.md section 2",
            "signal_unit": "API value as returned, overlap mean to 25 bp",
            "native_unit": "ChIP-seq: fold change per bp (API value / chip_native_divisor); "
                           "accessibility: normalized insertions per 25 bp bin"}
    summary = {}
    for regime, items in sorted(by_regime.items()):
        d = Path(out) / regime
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for mid, m, a in items:
            cond = a["condition"]
            mc, oc = (bare_channel(a.get(k, ""), cond, required=False) for k in ("matched_channel", "otherend_channel"))
            # plan section A (Max): with no channel of the target's read type, the other one is scored as
            # "matched"; channels.tsv says so (matched_source = otherend_fallback).
            scored = mc or oc
            row = {"measurement_id": mid, "condition": cond, "matched_channel": scored, "otherend_channel": oc,
                   "api_label_matched": "", "api_label_otherend": "",
                   "matched_source": "matched" if mc else "otherend_fallback"}
            arr, row["api_label_matched"], kind = tracks[cond][scored]
            div = chip_div if kind == "chip_histone" else 1
            row.update(output_type=kind, chip_bin_value=chip_bin_value if kind == "chip_histone" else "",
                       native_divisor=div)
            np.save(d / f"{mid}.matched.signal.npy", arr)
            np.save(d / f"{mid}.native.npy", (arr / np.float32(div)).astype(np.float32))
            if oc:
                arr2, row["api_label_otherend"], _ = tracks[cond][oc]
                np.save(d / f"{mid}.otherend.signal.npy", arr2)
            rows.append(row)
        write_tsv(d / "channels.tsv", CHANNEL_COLUMNS, rows)
        (d / "EVALUATION_ONLY").write_bytes(b"")
        meta = dict(base, regime=regime, targets=[r["measurement_id"] for r in rows])
        (d / "tracks.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        summary[regime] = len(rows)
        log(f"{regime}: {len(rows)} targets packed into {d}")
    log(f"skipped: {dict(sorted(skipped.items()))}")
    return {"packed": summary, "skipped": skipped}


# ----------------------------------------------------------------------------------------------------- CLI

def read_conditions(path) -> list[str]:
    rows = read_tsv(path)
    if rows and "condition" not in rows[0]:
        raise ValueError(f"{path} has no `condition` column")
    out = []
    for r in rows:
        c = (r.get("condition") or "").strip()
        if c and c not in out:
            out.append(c)
    return out


def _windows_of(a):
    if bool(a.evalset) == bool(a.chroms):
        raise ValueError("give exactly one of --evalset and --chroms")
    if a.evalset:
        return evalset_windows(json.loads(Path(a.evalset).read_text(encoding="utf-8")))
    chroms = [c for c in a.chroms.split(",") if c]
    bad = [c for c in chroms if c not in HG38]
    if bad:
        raise ValueError(f"unknown chromosomes {bad}")
    return tile_windows(chroms)


def _outputs_of(text) -> list[str]:
    outs = [o.strip().lower() for o in text.split(",") if o.strip()]
    bad = [o for o in outs if o not in OUTPUTS]
    if not outs or bad or "chip_histone" not in outs:
        raise ValueError(f"--outputs: chip_histone plus optionally dnase, atac (got {text!r})")
    return [o for o in OUTPUTS if o in outs]


def main(argv=None, *, client_factory=_client, sleep=time.sleep) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fetch", "status"):
        p = sub.add_parser(name)
        p.add_argument("--conditions", required=True)
        p.add_argument("--evalset")
        p.add_argument("--chroms")
        p.add_argument("--outputs", required=True)
        p.add_argument("--cache", required=True)
        p.add_argument("--model", default="all_folds", choices=["all_folds"])
        p.add_argument("--strict", action="store_true", help="an unreadable part file is an error, not missing")
        if name == "fetch":
            p.add_argument("--dry-run", action="store_true",
                           help="print the planned calls per condition; read no key, create no client")
            p.add_argument("--max-calls", type=int, default=2000)
            p.add_argument("--workers", type=int, default=5)
            p.add_argument("--chunk", type=int, default=25)
    p = sub.add_parser("pack")
    for k in ("--cache", "--evalset", "--audit", "--tables", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--split", required=True, choices=["val", "blind"])
    p.add_argument("--model", default="all_folds", choices=["all_folds"])
    p.add_argument("--chip-bin-value", default="sum", choices=sorted(CHIP_NATIVE_DIVISOR),
                   help="what one API ChIP-seq 128 bp value is (API smoke t267 section 2 measured: sum)")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "pack":
            pack(a.cache, a.evalset, a.audit, a.tables, a.split, a.out, model=a.model,
                 chip_bin_value=a.chip_bin_value)
            return EXIT_OK
        windows, outputs, conds = _windows_of(a), _outputs_of(a.outputs), read_conditions(a.conditions)
        if a.cmd == "status" or getattr(a, "dry_run", False):
            lines = (status(conds, windows, outputs, a.cache, a.model, strict=a.strict) if a.cmd == "status"
                     else plan(conds, windows, outputs, a.cache, a.model, strict=a.strict))
            for line in lines:
                print(line)
            return EXIT_OK
        if a.chunk < 1 or a.max_calls < 0 or a.workers < 1:
            raise ValueError("--chunk and --workers must be >= 1, --max-calls >= 0")
        return fetch(conds, windows, outputs, a.cache, model=a.model, max_calls=a.max_calls, workers=a.workers,
                     chunk=a.chunk, client_factory=client_factory, sleep=sleep, strict=a.strict)
    except KeyFileError as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_USAGE
    except (ValueError, FetchError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR if isinstance(e, FetchError) else EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
