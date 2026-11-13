"""AGk -- the peak side of the AlphaGenome 2 comparison (plan Amendment 5, section B.3).

    python tools/onion_eval/ag2_peaks.py fetch --ledger <accession_ledger.tsv> --roles train,val[,blind]
        --out <dir> [--files <catalog files.tsv>]
        [--allow-blind --evalset <evalset_blind.json> [--repeat-blind-reason "<text>"]]
    -> <out>/<eic_track>.narrowPeak.gz (md5-checked) and <out>/peak_files.tsv
       (eic_track, role, experiment, file, output_type, md5, n_peaks, rule)

Truth peaks, primary: one ENCODE peak file per track, from the track's own experiment. The rule
(section B.3): a released GRCh38 `bed narrowPeak` file; files of the experiment's default analysis
(`default_analysis` in the portal JSON) first, else files of its newest released analysis; then a
file that ENCODE marks `preferred_default`; then replicated peaks before pseudoreplicated peaks
before IDR peaks before plain peaks; then the newest file; then the accession. The `rule` column of
`peak_files.tsv` names the route and the keys that decided. A track with no peak file is listed with
the reason in `rule` and an empty `file`. The catalog route (`--files`) is used for an experiment
when the catalog lists peak files for it; the catalog has no `file_type`, no analysis and no
`preferred_default`, so that route ranks by output type, then accession only, and says so in `rule`.
Controls are skipped. A rerun skips a file already on disk with the right md5, and merges its rows
into an existing `peak_files.tsv` by `eic_track`.

Blind discipline. A peak file and its `n_peaks` are statistics of the blind truth. `--roles` with
`blind` needs `--allow-blind --evalset <json>`; the tool appends a `started` row to `BLIND_PASSES.tsv`
beside that eval set before the first blind portal read, and a `fetched` row after the last blind
download (`score.ledger_append`, key `peaks:<resolved out dir>`). A second blind fetch for the same
key and eval set is refused without `--repeat-blind-reason`.

Tiles and AUPRC (Max's design; ported, not rewritten; sources in GitHub
mlibbrecht/01_2026-10-19_research at bde905a, projects/03_2026-10-20_epi_imputation/sprints/:
S11 = 11_2026-11-11_alphagenome2, S12 = 12_2026-11-12_ag2_fair_test). A tile is 20 bins of 25 bp (500 bp), laid
from the start of each eval region; a region of 16,384 bins holds 819 whole tiles and its last 4 bins
belong to no tile (Max also drops partial tiles). A tile is called for a track when any of its bins
is a peak bin (`tile_any`; S12 `scripts/api/score_f1.py` `tile_calls` l.109-116, which reads the
per-tile peak-bin counts of S11 `scripts/bridge/peak_tiles.py` l.2-13 and l.40-44 and tests > 0).
Consensus tiles: tiles called in any track of the tile set (score_f1.py l.119-136); a track with no
called tile does not count (l.116, `nopeak`). Positive: the target's tile is called; negative: the
other consensus tiles; a tile outside the consensus (background) never enters. Frequency of a
consensus tile: the fraction of counted tracks that call it; bin k of `n_bins` holds frequencies in
((k)/n, (k+1)/n] (score_f1.py l.137). The ranking score of a row is the mean of its prediction over
the tile (`tile_mean`; score_f1.py l.273). AUPRC is `candi.bench.binary.average_precision`. A target
with no positive or with every consensus tile positive has an undefined AUPRC (NaN), not 0 or 1.
Max merged replicate biosamples of one term before counting (l.127-131); here the caller chooses the
keys of `consensus` (one per track, or one per cell after a union).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from candi.bench.binary import average_precision  # noqa: E402

BIN_BP = 25
TILE_BINS = 20
PORTAL = "https://www.encodeproject.org"
USER_AGENT = "candi-onion-peaks/1.0"
PEAK_COLUMNS = ("eic_track", "role", "experiment", "file", "output_type", "md5", "n_peaks", "rule")
ROLES = ("train", "val", "blind")
#: rank of an ENCODE peak output type (lower first); replicated before pseudoreplicated (B.3)
OUTPUT_RANK = {"replicated peaks": 0, "pseudoreplicated peaks": 1,
               "conservative IDR thresholded peaks": 2, "optimal IDR thresholded peaks": 3,
               "IDR thresholded peaks": 4, "peaks": 5}
DERIVED_THRESHOLD = 2.0          # -log10 p >= 2, the MACS2 p threshold of the ENCODE histone pipeline


# ------------------------------------------------------------------------------------ rasterize

def _open_text(path):
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return io.StringIO(raw.decode("utf-8"))


def read_peaks(path) -> dict:
    """chrom -> (starts, ends) int64 arrays of a BED-like file (narrowPeak: columns 1-3)."""
    by = {}
    for line in _open_text(path):
        if not line.strip() or line.startswith(("#", "track", "browser")):
            continue
        f = line.split("\t")
        by.setdefault(f[0], []).append((int(f[1]), int(f[2])))
    return {c: (np.array([s for s, _ in v], np.int64), np.array([e for _, e in v], np.int64))
            for c, v in by.items()}


def rasterize(path, regions, rb, bin_bp: int = BIN_BP) -> np.ndarray:
    """Peak mask bool[n_regions * rb] in the eval set's region order. `regions` = [(chrom, start_bin)];
    region i covers bins [start_bin, start_bin + rb) of `bin_bp` bp. A bin is a peak bin when its
    interval overlaps a peak [start, end) (BED half-open) by at least 1 bp."""
    peaks = read_peaks(path)
    out = np.zeros(len(regions) * rb, dtype=bool)
    for i, (chrom, sb) in enumerate(regions):
        if chrom not in peaks:
            continue
        s, e = peaks[chrom]
        keep = e > s
        lo = s[keep] // bin_bp - int(sb)               # first and last bin of each peak, region frame
        hi = (e[keep] - 1) // bin_bp - int(sb)
        hit = (hi >= 0) & (lo < rb)
        if not hit.any():
            continue
        lo, hi = np.clip(lo[hit], 0, rb - 1), np.clip(hi[hit], 0, rb - 1)
        diff = np.zeros(rb + 1, np.int64)              # interval union by a difference array
        np.add.at(diff, lo, 1)
        np.add.at(diff, hi + 1, -1)
        out[i * rb:(i + 1) * rb] = np.cumsum(diff[:rb]) > 0
    return out


def derived_peaks(neglog10p, threshold: float = DERIVED_THRESHOLD) -> np.ndarray:
    """Pipeline-derived peak bins: -log10 p >= threshold (p <= 0.01 at the default)."""
    return np.asarray(neglog10p, dtype=np.float64) >= threshold


# ---------------------------------------------------------------------------------------- tiles

def _tiled(x, tile_bins: int, region_bins):
    x = np.asarray(x)
    rb = x.size if region_bins is None else int(region_bins)
    if rb <= 0 or x.size % rb:
        raise ValueError(f"{x.size} bins is not a whole number of regions of {rb} bins")
    nt = rb // tile_bins
    return x.reshape(-1, rb)[:, :nt * tile_bins].reshape(-1, nt, tile_bins)


def tile_any(bins, tile_bins: int = TILE_BINS, region_bins=None) -> np.ndarray:
    """bool per tile: any bin of the tile set. Tiles are laid from each region's start
    (`region_bins`; None = one region); the tail of a region shorter than a tile is dropped."""
    return _tiled(np.asarray(bins, dtype=bool), tile_bins, region_bins).any(-1).ravel()


def tile_mean(values, tile_bins: int = TILE_BINS, region_bins=None) -> np.ndarray:
    return _tiled(np.asarray(values, dtype=np.float64), tile_bins, region_bins).mean(-1).ravel()


def tile_max(values, tile_bins: int = TILE_BINS, region_bins=None) -> np.ndarray:
    return _tiled(np.asarray(values, dtype=np.float64), tile_bins, region_bins).max(-1).ravel()


# ------------------------------------------------------------------------------ consensus, AUPRC

def consensus(tile_calls_by_track: dict):
    """(consensus bool[n_tiles], frequency float64[n_tiles]). A track that calls no tile is not
    counted (Max's `nopeak`); frequency = called tracks / counted tracks, 0 off the consensus."""
    calls = [np.asarray(v, dtype=bool) for v in tile_calls_by_track.values()]
    calls = [c for c in calls if c.any()]
    if not calls:
        n = len(next(iter(tile_calls_by_track.values()), []))
        return np.zeros(n, bool), np.zeros(n, np.float64)
    cnt = np.sum(calls, axis=0)
    return cnt > 0, cnt / float(len(calls))


def frequency_bin(frequency, n_bins: int = 8) -> np.ndarray:
    """Bin k (0-based) holds frequencies in (k/n, (k+1)/n] (score_f1.py l.137)."""
    f = np.asarray(frequency, dtype=np.float64)
    return np.clip(np.ceil(f * n_bins - 1e-9).astype(int) - 1, 0, n_bins - 1)


def auprc(target_tiles, consensus_tiles, score):
    """(AUPRC, positive fraction) on the consensus tiles. Undefined (NaN) when the target has no
    positive or every consensus tile is positive (a ranking with no negative measures nothing)."""
    c = np.asarray(consensus_tiles, dtype=bool)
    y = np.asarray(target_tiles, dtype=bool)[c]
    s = np.asarray(score, dtype=np.float64)[c]
    if y.size == 0:
        return float("nan"), float("nan")
    npos = int(y.sum())
    pf = npos / y.size
    if npos == 0 or npos == y.size:
        return float("nan"), pf
    return float(average_precision(y, s)), pf


def auprc_by_frequency(target_tiles, consensus_tiles, score, frequency, n_bins: int = 8) -> list:
    """One dict per frequency bin (`bin` 1..n_bins, `n_tiles`, `n_pos`, `auprc`); the bins
    partition the consensus tiles. An empty bin has n_tiles 0 and AUPRC NaN."""
    c = np.asarray(consensus_tiles, dtype=bool)
    fb = frequency_bin(frequency, n_bins)
    y = np.asarray(target_tiles, dtype=bool)
    out = []
    for k in range(n_bins):
        m = c & (fb == k)
        a, _ = auprc(y, m, score) if m.any() else (float("nan"), float("nan"))
        out.append({"bin": k + 1, "n_tiles": int(m.sum()), "n_pos": int((y & m).sum()), "auprc": a})
    return out


def binary_prf(target_tiles, consensus_tiles, called_tiles):
    """(precision, recall, F1) of a binary call on the consensus tiles; NaN where undefined."""
    c = np.asarray(consensus_tiles, dtype=bool)
    y = np.asarray(target_tiles, dtype=bool)[c]
    z = np.asarray(called_tiles, dtype=bool)[c]
    tp = int((y & z).sum())
    p = tp / int(z.sum()) if z.any() else float("nan")
    r = tp / int(y.sum()) if y.any() else float("nan")
    f = (2 * p * r / (p + r) if (p + r) > 0 else 0.0) if not (math.isnan(p) or math.isnan(r)) \
        else float("nan")
    return p, r, f


# ---------------------------------------------------------------------------------- file choice

def _urlopen(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def _get(opener, url: str, tries: int = 4) -> bytes:
    for k in range(tries):
        try:
            return opener(url)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2.0 * 2 ** k)
    raise RuntimeError("unreachable")


def choose_from_portal(exp: dict):
    """(file dict, rule text) or (None, reason) from an experiment's portal JSON."""
    files = [f for f in exp.get("files") or [] if isinstance(f, dict)
             and f.get("file_type") == "bed narrowPeak" and f.get("assembly") == "GRCh38"
             and f.get("status") == "released"]
    if not files:
        return None, "no released GRCh38 bed narrowPeak file"

    def ans(f):
        return {a.get("@id") if isinstance(a, dict) else a for a in f.get("analyses") or []}

    default = exp.get("default_analysis")
    default = default.get("@id") if isinstance(default, dict) else default
    scope = "default_analysis"
    pool = [f for f in files if default and default in ans(f)]
    if not pool:
        rel = [a for a in exp.get("analyses") or [] if isinstance(a, dict)
               and a.get("status") == "released" and a.get("assembly") in ("GRCh38", None)]
        rel.sort(key=lambda a: a.get("date_created") or "", reverse=True)
        for a in rel:
            pool = [f for f in files if a.get("@id") in ans(f)]
            if pool:
                scope = "newest_released_analysis"
                break
    if not pool:
        pool, scope = files, "no_analysis_match"
    pool.sort(key=lambda f: f.get("accession") or "")                    # last key
    pool.sort(key=lambda f: f.get("date_created") or "", reverse=True)    # newest file
    pool.sort(key=lambda f: (not f.get("preferred_default"),             # stable: first keys
                             OUTPUT_RANK.get(f.get("output_type"), 9)))
    f = pool[0]
    rule = (f"portal:{scope};preferred_default={'yes' if f.get('preferred_default') else 'no'};"
            f"output_type_rank={OUTPUT_RANK.get(f.get('output_type'), 9)}")
    return {"file": f.get("accession"), "output_type": f.get("output_type") or "",
            "md5": f.get("md5sum") or "", "url": PORTAL + f["href"] if f.get("href") else ""}, rule


def catalog_peaks(files_tsv) -> dict:
    """experiment -> peak rows of a catalog files.tsv (released GRCh38 bed with a peak output type)."""
    by = {}
    with open(files_tsv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if (r.get("file_format") == "bed" and r.get("assembly") == "GRCh38"
                    and r.get("status") == "released" and r.get("output_type") in OUTPUT_RANK):
                by.setdefault(r["experiment"], []).append(r)
    return by


def choose_from_catalog(rows: list):
    rows = sorted(rows, key=lambda r: (OUTPUT_RANK.get(r["output_type"], 9), r["file"]))
    r = rows[0]
    url = r.get("url") or f"{PORTAL}/files/{r['file']}/@@download/{r['file']}.bed.gz"
    return ({"file": r["file"], "output_type": r["output_type"], "md5": r.get("md5", ""),
             "url": url},
            f"catalog:no_default_flag;output_type_rank={OUTPUT_RANK.get(r['output_type'], 9)}")


def count_peaks(path) -> int:
    return sum(1 for line in _open_text(path)
               if line.strip() and not line.startswith(("#", "track", "browser")))


def download(opener, url: str, md5: str, dest: Path) -> None:
    """Download to a temporary name, check the md5, rename. A file already on disk with the right
    md5 is kept."""
    if dest.is_file() and md5 and hashlib.md5(dest.read_bytes()).hexdigest() == md5:
        return
    data = _get(opener, url)
    got = hashlib.md5(data).hexdigest()
    if md5 and got != md5:
        raise ValueError(f"md5 mismatch: portal {md5}, downloaded {got}")
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)


# ---------------------------------------------------------------------------------------- fetch

def _read_tsv(path) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh, delimiter="\t")]


def _write_rows(path: Path, rows: list) -> None:
    order = {r: i for i, r in enumerate(ROLES)}
    rows = sorted(rows, key=lambda r: (order.get(r["role"], 9), r["eic_track"]))
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(PEAK_COLUMNS), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in PEAK_COLUMNS})
    tmp.replace(path)


def _fetch_track(t: dict, out: Path, cat: dict, opener) -> dict:
    row = {"eic_track": t["eic_track"], "role": t["role"], "experiment": t["experiment"]}
    if cat.get(t["experiment"]):
        pick, rule = choose_from_catalog(cat[t["experiment"]])
    else:
        try:
            exp = json.loads(_get(opener, f"{PORTAL}/experiments/{t['experiment']}/?format=json"))
        except Exception as e:
            return dict(row, rule=f"failed: portal JSON ({type(e).__name__})")
        pick, rule = choose_from_portal(exp)
    if pick is None:
        return dict(row, rule=f"no peak file: {rule}")
    if not pick["url"]:
        return dict(row, rule=f"no peak file: {pick['file']} has no download URL")
    dest = out / f"{t['eic_track']}.narrowPeak.gz"
    try:
        download(opener, pick["url"], pick["md5"], dest)
    except Exception as e:
        return dict(row, rule=f"failed: download {pick['file']} ({type(e).__name__}: {e})"[:300])
    return dict(row, file=pick["file"], output_type=pick["output_type"], md5=pick["md5"],
                n_peaks=count_peaks(dest), rule=rule)


def fetch(ledger, roles, out, *, files=None, allow_blind=False, evalset=None,
          repeat_blind_reason=None, opener=None) -> list:
    """Section B.3 primary truth peaks for the non-control tracks of `roles`. Returns the rows."""
    roles = tuple(roles)
    bad = [r for r in roles if r not in ROLES]
    if bad or not roles:
        raise ValueError(f"--roles {roles}: each role is one of {ROLES}")
    if "blind" in roles and not (allow_blind and evalset):
        raise PermissionError("--roles blind needs --allow-blind and --evalset <evalset_blind.json> "
                              "(a peak file is a statistic of the blind truth)")
    opener = opener or _urlopen
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cat = catalog_peaks(files) if files else {}
    tracks = [r for r in _read_tsv(ledger) if r["role"] in roles
              and r.get("assay_class") != "control" and "control" not in r.get("assay", "")]
    open_rows = [t for t in tracks if t["role"] != "blind"]
    blind_rows = [t for t in tracks if t["role"] == "blind"]

    ledger_path, entry = None, None
    if "blind" in roles:                       # refuse before any download of this call
        from onion_eval import score as S      # lazy: score imports torch
        ledger_path = Path(evalset).resolve().parent / S.BLIND_LEDGER
        key = f"peaks:{out.resolve()}"
        esha = S.sha256_file(evalset)
        prior = [r for r in S.ledger_rows(ledger_path)
                 if r["run"] == key and r["evalset_sha256"] == esha]
        reason = (repeat_blind_reason or "").strip()
        if prior and not reason:
            raise PermissionError(f"{key} already had {len(prior)} blind ledger rows for this eval "
                                  f"set ({ledger_path}); a repeat needs --repeat-blind-reason")
        ev = json.loads(Path(evalset).read_text(encoding="utf-8"))
        entry = {"run": key, "evalset_sha256": esha, "tables_sha256": ev.get("tables_sha256", ""),
                 "git_sha": S.git_sha(), "out": str(out.resolve()),
                 "repeat_reason": reason if prior else ""}

    rows = [_fetch_track(t, out, cat, opener) for t in open_rows]
    if entry is not None:
        S.ledger_append(ledger_path, dict(entry, time=S._utc(), status="started"))
        rows += [_fetch_track(t, out, cat, opener) for t in blind_rows]
        S.ledger_append(ledger_path, dict(entry, time=S._utc(), status="fetched"))

    table = out / "peak_files.tsv"
    old = {r["eic_track"]: r for r in _read_tsv(table)} if table.is_file() else {}
    old.update({r["eic_track"]: r for r in rows})
    _write_rows(table, list(old.values()))
    return rows


def main(argv=None, opener=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="pick, download and md5-check one ENCODE peak file per track")
    f.add_argument("--ledger", required=True)
    f.add_argument("--roles", required=True, help="comma list of train, val, blind (no default)")
    f.add_argument("--out", required=True)
    f.add_argument("--files", default=None, help="catalog files.tsv (used when it lists peak files)")
    f.add_argument("--allow-blind", action="store_true")
    f.add_argument("--evalset", default=None, help="the blind eval set; BLIND_PASSES.tsv is beside it")
    f.add_argument("--repeat-blind-reason", default=None)
    a = ap.parse_args(argv)
    try:
        rows = fetch(a.ledger, [r for r in a.roles.split(",") if r], a.out, files=a.files,
                     allow_blind=a.allow_blind, evalset=a.evalset,
                     repeat_blind_reason=a.repeat_blind_reason, opener=opener)
    except (PermissionError, ValueError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    n_ok = sum(1 for r in rows if r.get("file"))
    n_fail = sum(1 for r in rows if str(r.get("rule", "")).startswith("failed"))
    print(f"peak files: {n_ok} fetched, {len(rows) - n_ok - n_fail} without a peak file, "
          f"{n_fail} failed; {Path(a.out) / 'peak_files.tsv'}")
    return 3 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
