#!/usr/bin/env python3
"""miniCANDI beside the EIC methods on the challenge's validation tracks (chunk C16, D10, R10).

    python tools/onion_eval/eic_val_compare.py score --pred <predict out>/signal --name <entry>
        --truth /project/def-maxwl/mforooz/DATA_EIC_SYNAPSE/validation_data --cache <dir> --out <tsv>
    python tools/onion_eval/eic_val_compare.py average
        --train /project/def-maxwl/mforooz/DATA_EIC_SYNAPSE/training_data --out <dir> [--truth <dir>]
    python tools/onion_eval/eic_val_compare.py table --scores <tsv> [<tsv> ...] --out <tsv>
    python tools/onion_eval/eic_val_compare.py val-cells --tables <dir> --evalset <json>

`score` runs `eic_score score` (the nine measures of the challenge, per track, against the truth
folder) on one entry's folder of challenge .npy files (`predict.py` writes `signal/` = the
expectation signal, the headline, and `signal_plug_in/` = the plug-in signal) or bigWigs (the
Avocado route). The rows are eic_score's own `ROW_KEYS`, with `entry` = `--name`.

`average` builds the challenge's Average baseline from the TRAINING tracks only: per assay, the
mean over every training track of that assay of its 25 bp signal (each track binned as the scorer
bins a bigWig). It writes `<out>/assay/<Mxx>.npy` (the challenge .npy: pickled {chrom: float32} on
the raw 25 bp grid) for each assay that has a truth track in `--truth` (default: the folder
`validation_data` beside `--train`), and `<out>/<exp>.npy`, a relative symlink to its assay file,
for each truth track. `<out>/AVERAGE.json` records the members of each assay. A truth assay with
no training track gets no file and is listed under `no_training_track`.

`table` reads score files (all23 chromosome set only) and writes one row per entry and measure:
`n_tracks` (tracks every entry scored; the mean and the counts use only these), the mean over
them, and `beats:<other>` = the number of those tracks on which the entry is strictly better than
the other entry (higher for gwcorr and gwspear, lower for the MSE measures; a NaN never wins).

`val-cells` prints the number of cells of the eval set's validation leave-one-assay-out targets
(stdlib only; `slurm/onion_eval/eic_val.sh` sizes its array with it).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

REGIME = "leave_one_assay_out"
SET_ALL = "all23"
DEFAULT_DATA = "/project/def-maxwl/mforooz/DATA_EIC_SYNAPSE"
# eic_score.MEASURES9 and HIGHER_IS_BETTER, restated so `table` and `val-cells` need no numpy
# (the test holds them equal).
MEASURES9 = ["mse", "gwcorr", "gwspear", "mseprom", "msegene", "mseenh",
             "msevar", "mse1obs", "mse1imp"]
HIGHER_IS_BETTER = {"gwcorr", "gwspear"}


def _eic():
    from onion_eval import eic_score as EIC          # numpy; imported only where it is needed
    return EIC


# --------------------------------------------------------------------------------------- tracks

def track_files(folder) -> dict[str, Path]:
    """{C##M##: path} of the .bigwig/.bw/.npy tracks in a folder (first extension wins)."""
    out = {}
    for ext in (".bigwig", ".bw", ".npy"):
        for p in sorted(Path(folder).glob(f"C*M*{ext}")):
            stem = p.name[: -len(ext)]
            if len(stem) == 6 and stem[1:3].isdigit() and stem[4:].isdigit():
                out.setdefault(stem, p)
    return out


def val_cells(tables, evalset) -> list[str]:
    ev = json.loads(Path(evalset).read_text(encoding="utf-8"))
    with open(Path(tables) / "measurements.tsv", newline="", encoding="utf-8") as fh:
        cell = {r["measurement_id"]: r["cell"] for r in csv.DictReader(fh, delimiter="\t")}
    return sorted({cell[t["target"]] for t in ev["imputation_targets"]
                   if t["split"] == "val" and t.get("regime", REGIME) == REGIME})


# --------------------------------------------------------------------------------------- average

def _bin_track(path, chroms):
    EIC = _eic()
    if str(path).endswith(".npy"):
        import numpy as np
        d = np.load(path, allow_pickle=True)[()]
        return {c: np.asarray(d[c], np.float32) for c in chroms}
    return EIC.bin_bigwig(path, chroms)


def average_assay(paths, chroms):
    """{chrom: float32} mean of the binned tracks; a shorter track is zero-padded (missing = 0)."""
    import numpy as np
    acc = {}
    for p in paths:
        y = _bin_track(p, chroms)
        for c in chroms:
            v = y[c].astype(np.float64)
            a = acc.get(c)
            if a is None:
                acc[c] = v.copy()
                continue
            if v.shape[0] > a.shape[0]:
                a = np.concatenate([a, np.zeros(v.shape[0] - a.shape[0])])
                acc[c] = a
            a[: v.shape[0]] += v
        del y
    return {c: (acc[c] / len(paths)).astype(np.float32) for c in chroms}


def _average_one(args):
    assay, paths, chroms, out = args
    import numpy as np
    dst = Path(out) / "assay" / f"{assay}.npy"
    if dst.is_file():
        return assay, False
    avg = average_assay(paths, chroms)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".tmp{os.getpid()}.npy")
    np.save(tmp, avg, allow_pickle=True)
    os.replace(tmp, dst)
    return assay, True


def build_average(train, out, truth=None, chroms=None, workers=1, log=print) -> dict:
    train, out = Path(train), Path(out)
    truth = Path(truth) if truth else train.parent / "validation_data"
    chroms = list(chroms or _eic().CHROMS)
    tr, tv = track_files(train), track_files(truth)
    if not tr:
        raise SystemExit(f"no C##M## training track under {train}")
    if not tv:
        raise SystemExit(f"no C##M## truth track under {truth}")
    overlap = sorted(set(tr) & set(tv))
    if overlap:
        raise SystemExit(f"{len(overlap)} truth tracks are also in the training folder "
                         f"({overlap[:3]}...): not a training-only Average")
    members = {}
    for e in sorted(tr):
        members.setdefault(e[3:], []).append(e)
    want = sorted({e[3:] for e in tv})
    missing = [a for a in want if a not in members]
    todo = [(a, [str(tr[e]) for e in members[a]], chroms, str(out)) for a in want if a in members]
    out.mkdir(parents=True, exist_ok=True)
    log(f"[average] {len(todo)} assays from {sum(len(t[1]) for t in todo)} training tracks; "
        f"{len(tv)} truth tracks; no training track: {missing or 'none'}")
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for a, new in ex.map(_average_one, todo):
                log(f"[average] {a}: {'built' if new else 'exists'}")
    else:
        for t in todo:
            a, new = _average_one(t)
            log(f"[average] {a}: {'built' if new else 'exists'}")
    linked = []
    for e in sorted(tv):
        if e[3:] not in members:
            continue
        link = out / f"{e}.npy"
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(os.path.join("assay", f"{e[3:]}.npy"), link)
        linked.append(e)
    rec = {"train": str(train), "truth": str(truth), "chroms": chroms,
           "members": {a: members[a] for a in want if a in members},
           "tracks": linked, "no_training_track": missing}
    (out / "AVERAGE.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    log(f"[average] {len(linked)} truth tracks linked under {out}")
    return rec


# --------------------------------------------------------------------------------------- score

def score(pred, name, truth, cache, out, workers=0, log=print) -> int:
    pred = Path(pred)
    if not pred.is_dir():
        raise SystemExit(f"no prediction folder {pred}")
    if (pred.parent / "signal_plug_in").is_dir() and not (pred.parent / "DONE").exists():
        log(f"warning: {pred.parent}/DONE is missing; the predict.py run may be partial")
    have, truth_t = track_files(pred), track_files(truth)
    log(f"[score] {name}: {len(set(have) & set(truth_t))} of {len(truth_t)} truth tracks "
        f"predicted; not predicted: {sorted(set(truth_t) - set(have)) or 'none'}")
    argv = ["score", "--pred", str(pred), "--entry", name, "--truth", str(truth),
            "--cache", str(cache), "--out", str(out)]
    if workers:
        argv += ["--workers", str(workers)]
    return _eic().main(argv)


# --------------------------------------------------------------------------------------- table

def read_scores(paths) -> dict[str, dict[str, dict[str, float]]]:
    """{entry: {experiment: {measure: value}}} of the all23 rows of eic_score score files."""
    out: dict = {}
    for p in paths:
        seen = set()
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if r["chrom_set"] != SET_ALL:
                    continue
                e = r["entry"]
                if e in out and e not in seen:
                    raise SystemExit(f"entry {e!r} appears in more than one score file ({p})")
                seen.add(e)
                if r["experiment"] in out.setdefault(e, {}):
                    raise SystemExit(f"{p}: two {SET_ALL} rows for {e} {r['experiment']}")
                out[e][r["experiment"]] = {m: float(r[m]) for m in MEASURES9}
    return out




def _better(a: float, b: float, measure: str) -> bool:
    if math.isnan(a) or math.isnan(b):
        return False
    return a > b if measure in HIGHER_IS_BETTER else a < b


def table_rows(scores) -> list[dict]:
    entries = list(scores)
    common = sorted(set.intersection(*(set(v) for v in scores.values()))) if scores else []
    rows = []
    for e in entries:
        for m in MEASURES9:
            vals = [scores[e][x][m] for x in common]
            ok = [v for v in vals if not math.isnan(v)]
            row = {"entry": e, "measure": m,
                   "higher_is_better": int(m in HIGHER_IS_BETTER),
                   "n_tracks": len(common), "n_nan": len(vals) - len(ok),
                   "mean": (sum(ok) / len(ok)) if ok else float("nan"),
                   "n_tracks_entry": len(scores[e])}
            for o in entries:
                row[f"beats:{o}"] = ("" if o == e else
                                     sum(_better(scores[e][x][m], scores[o][x][m], m)
                                         for x in common))
            rows.append(row)
    return rows


def write_table(paths, out, log=print) -> list[dict]:
    scores = read_scores(paths)
    if not scores:
        raise SystemExit("no all23 rows in the score files")
    rows = table_rows(scores)
    cols = ["entry", "measure", "higher_is_better", "n_tracks", "n_nan", "mean",
            "n_tracks_entry"] + [f"beats:{e}" for e in scores]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([repr(r[c]) if isinstance(r[c], float) else r[c] for c in cols])
    n = rows[0]["n_tracks"]
    log(f"[table] {len(scores)} entries on {n} common tracks -> {out}")
    for e, v in scores.items():
        if len(v) != n:
            log(f"[table] {e}: scored {len(v)} tracks, {len(v) - n} outside the common set")
    return rows


# --------------------------------------------------------------------------------------- cli

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("score", help="nine EIC measures of one entry on the validation truth")
    p.add_argument("--pred", required=True, help="folder of C##M##.npy or .bigwig")
    p.add_argument("--name", required=True, help="entry name in the score file")
    p.add_argument("--truth", default=f"{DEFAULT_DATA}/validation_data")
    p.add_argument("--cache", required=True, help="eic_score cache (binned truth, per-entry scores)")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=0, help="default SLURM_CPUS_PER_TASK // 4")

    p = sp.add_parser("average", help="the Average baseline of the training tracks only")
    p.add_argument("--train", default=f"{DEFAULT_DATA}/training_data")
    p.add_argument("--truth", default=None, help="default: validation_data beside --train")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=1, help="assays averaged in parallel")

    p = sp.add_parser("table", help="one row per entry and measure, with pairwise track wins")
    p.add_argument("--scores", nargs="+", required=True)
    p.add_argument("--out", required=True)

    p = sp.add_parser("val-cells", help="number of validation cells of the eval set")
    p.add_argument("--tables", required=True)
    p.add_argument("--evalset", required=True)

    a = ap.parse_args(argv)
    if a.cmd == "score":
        return score(a.pred, a.name, a.truth, a.cache, a.out, workers=a.workers) or 0
    if a.cmd == "average":
        build_average(a.train, a.out, truth=a.truth, workers=a.workers)
        return 0
    if a.cmd == "table":
        write_table(a.scores, a.out)
        return 0
    print(len(val_cells(a.tables, a.evalset)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
