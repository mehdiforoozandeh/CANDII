"""Signal-space baselines of Max's AG2 benchmark, and the replicate-half reference row (chunk AGb).

    ag2_baselines.py run --tables <dir> --evalset <json> --truth-from <method track dir>
        --split val|blind [--allow-blind] [--repeat-blind-reason "<text>"] --out <dir>

Writes three directories of pinned interface F (plan, Amendment 5) under `<out>`:

  pseudoreplicate_half   kind `replicate_floor`. `signal` = asinh of the target's `pr1_signal.h5`;
                         its own `truth` = asinh of the target's `pr2_signal.h5` (two halves with no
                         shared read). A target without both halves is left out and counted.
  training_mean_signal   kind `oracle_baseline`. Per target: the mean, over the training tracks of
                         the target's assay in OTHER cells, of their signal (Max's training average;
                         `score_f1.py` l.13-19 of S12).
  closest5_oracle        kind `oracle_baseline`. Per block: the mean of the 5 such tracks whose signal
                         is most correlated (Pearson) with the truth of the block. A block is 8,000
                         bins (200 kb); the tail of a region joins its last block, so a region of
                         16,384 bins is two blocks of 8,000 and 8,384 bins.

A training track has no `signal.h5` (`build_list.py:54`), so its signal is
`signal_from_counts.signal_track(observed counts, mode="plug_in")` with its own observed control and
covariates, per eval region (as `signal_from_nb`), then asinh. A ChIP-seq training track with no
control is left out and counted. Candidates are the base views (`evalset.base_views`) of training
rows (`in_eic = train`) after every second experiment (`evalset.second_experiment_ids`) is removed;
a second experiment never enters the average or the oracle.

The truth comes from `--truth-from` (a `run` or `baseline` directory of chunk AGd: `<dir>/<regime>/`
with `tracks.json` and `<mid>.truth.npy`, or one such regime directory), never from a target file
read by this tool. The two halves of `pseudoreplicate_half` are the only target files it reads. The
targets of each regime are the ones that `--truth-from` lists; each must be an imputation target of
the eval set on `--split` and that regime.

Blind discipline: `--split blind` needs `--allow-blind`. The tool then appends a `started` row before
its first data read and a `scored` row at the end (`score.ledger_append`) to `BLIND_PASSES.tsv` beside
the eval set, under the key `oracle_baselines:<resolved out dir>`, and refuses a second blind run of
the same key and eval set without `--repeat-blind-reason`.

`closest5_oracle/<regime>/picks.tsv` (target, region, block, start, end, picked measurement ids) is
an audit file beside the interface-F arrays.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from onion_eval import evalset as E  # noqa: E402
from onion_eval import signal_from_counts as SFC  # noqa: E402

BLOCK_BINS = 8000          # 200 kb at 25 bp (Max: per 200 kb block)
K_CLOSEST = 5
#: an observed count of 0 enters `signal_track` as this (see `training_signal`)
ZERO_LIFT = 1e-6
HALF, MEAN, ORACLE = "pseudoreplicate_half", "training_mean_signal", "closest5_oracle"
KIND = {HALF: "replicate_floor", MEAN: "oracle_baseline", ORACLE: "oracle_baseline"}
LAMBDA_SOURCE = {HALF: "pipeline_signal",
                 MEAN: "training_tracks_own_observed_control",
                 ORACLE: "training_tracks_own_observed_control"}
TRUTH_KINDS = ("run", "baseline")
LEDGER_PREFIX = "oracle_baselines:"


class Refused(RuntimeError):
    pass


# ------------------------------------------------------------------------------------ helpers

def regions_sha256(regions, region_bins: int) -> str:
    text = json.dumps({"regions": [[str(c), int(s)] for c, s in regions],
                       "region_bins": int(region_bins)}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def blocks(rb: int, block_bins: int = BLOCK_BINS) -> list[tuple[int, int]]:
    """[start, end) bins of the blocks of one region; the tail joins the last block."""
    n = max(1, int(rb) // int(block_bins))
    edges = [i * block_bins for i in range(n)] + [int(rb)]
    return list(zip(edges[:-1], edges[1:]))


def asinh(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("a signal track holds non-finite or negative values")
    return np.arcsinh(x)


def training_mean(tracks: list[np.ndarray]) -> np.ndarray:
    if not tracks:
        raise ValueError("no training track to average")
    return np.mean(np.stack([np.asarray(t, np.float64) for t in tracks]), axis=0)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else float("nan")


def closest_k(truth: np.ndarray, cands: list[np.ndarray], k: int = K_CLOSEST):
    """(mean of the k candidates most correlated with `truth`, their indices). A candidate whose
    correlation is undefined (constant block) ranks last; ties keep candidate order."""
    r = [_pearson(np.asarray(c, np.float64), np.asarray(truth, np.float64)) for c in cands]
    order = sorted(range(len(cands)), key=lambda i: (np.isnan(r[i]), -np.nan_to_num(r[i])))
    pick = order[:k]
    return training_mean([cands[i] for i in pick]), pick


def _read_h5(path: Path, regions, rb: int, dtype) -> np.ndarray:
    """Regions end to end; bins past a chromosome end are 0 (as `OnionDataset.read`)."""
    out = np.zeros(len(regions) * rb, dtype=dtype)
    with h5py.File(path, "r") as f:
        for i, (c, s) in enumerate(regions):
            ds = f[c]
            end = min(int(s) + rb, ds.shape[0])
            if end > int(s):
                out[i * rb: i * rb + end - int(s)] = ds[int(s):end]
    return out


def _h5_path(row: dict) -> Path | None:
    p = row.get("counts_path", "")
    return Path(p[3:]) if p.startswith("h5:") else None


def training_signal(row: dict, rows_by_id: dict, regions, rb: int):
    """(asinh signal of a training track over the regions, None) or (None, reason)."""
    p = _h5_path(row)
    if p is None:
        return None, "counts_path is not h5: (no count file to read)"
    route = SFC.route_of(row["assay_class"] or row["assay"])
    cov = {"depth_reads": row["depth_reads"], "read_length": row["read_length"],
           "frag_len": row.get("frag_len")}
    ctl = None
    if route == "chip":
        cid = (row.get("control_measurement_id") or "").strip()
        crow = rows_by_id.get(cid) if cid else None
        if crow is None or _h5_path(crow) is None:
            return None, "ChIP-seq training track with no observed control"
        cov.update(control_depth_reads=crow["depth_reads"],
                   control_read_length=crow["read_length"], control_is_predicted=False)
        ctl = _read_h5(_h5_path(crow), regions, rb, np.float64)
    # plug_in_neglog10p takes mu > 0 only; a zero count is lifted to ZERO_LIFT, whose
    # floor(ZERO_LIFT * scale) is still 0, so the score of that bin is the score of count 0
    mu = np.maximum(_read_h5(p, regions, rb, np.float64), ZERO_LIFT)
    out = []
    for i in range(len(regions)):
        sl = slice(i * rb, (i + 1) * rb)
        sig, _ = SFC.signal_track(mu[sl], None, row["assay_class"] or row["assay"],
                                  None if ctl is None else ctl[sl], cov, mode="plug_in")
        out.append(sig)
    return asinh(np.concatenate(out)), None


def candidate_rows(rows: list[dict], target: dict, excluded: set) -> list[dict]:
    """Training base views of the target's assay in other cells, second experiments removed."""
    keep = [r for r in rows if r["measurement_id"] not in excluded and r["in_eic"] == "train"]
    views = E.base_views(keep, ("train",))
    return sorted((r for (cell, assay), r in views.items()
                   if assay == target["assay"] and cell != target["cell"]),
                  key=lambda r: r["measurement_id"])


def _truth_regimes(truth_from: Path) -> dict:
    """regime -> (regime dir, tracks.json)."""
    if (truth_from / "tracks.json").is_file():
        meta = json.loads((truth_from / "tracks.json").read_text(encoding="utf-8"))
        return {meta["regime"]: (truth_from, meta)}
    out = {}
    for d in sorted(p for p in truth_from.iterdir() if (p / "tracks.json").is_file()):
        meta = json.loads((d / "tracks.json").read_text(encoding="utf-8"))
        out[meta.get("regime", d.name)] = (d, meta)
    if not out:
        raise Refused(f"{truth_from}: no tracks.json (pinned interface F)")
    return out


def _write_dir(d: Path, meta: dict, arrays: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for name, a in arrays.items():
        np.save(d / name, np.asarray(a, dtype=np.float32))
    (d / "tracks.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n",
                                   encoding="utf-8")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------- run

def run(tables, evalset, truth_from, out, *, split="val", allow_blind=False,
        repeat_blind_reason=None, log=print) -> dict:
    if split not in ("val", "blind"):
        raise ValueError(f"split {split!r} not in (val, blind)")
    if split == "blind" and not allow_blind:
        raise PermissionError("--split blind needs --allow-blind (gate G4)")
    tables, evalset = Path(tables), Path(evalset).resolve()
    truth_from, out = Path(truth_from), Path(out)
    ev = json.loads(evalset.read_text(encoding="utf-8"))
    esha = E.sha256_file(evalset)
    rb = int(ev.get("region_bins", 2 ** 14))
    regions = [(str(c), int(s)) for c, s in ev["regions"]]
    rsha = regions_sha256(regions, rb)
    targets_of = {}
    for t in ev.get("imputation_targets", []):
        if t.get("split", "val") == split:
            targets_of.setdefault(t["regime"], set()).add(t["target"])

    regimes = _truth_regimes(truth_from)
    for regime, (d, meta) in regimes.items():
        if meta.get("kind") not in TRUTH_KINDS:
            raise Refused(f"{d}: kind {meta.get('kind')!r}; the truth comes from a run or "
                          f"baseline directory only")
        if meta.get("split") != split:
            raise Refused(f"{d}: split {meta.get('split')!r}, not {split!r}")
        if meta.get("evalset_sha256") != esha:
            raise Refused(f"{d}: evalset_sha256 differs from {evalset}")
        stray = sorted(set(meta.get("targets", [])) - targets_of.get(regime, set()))
        if stray:
            raise Refused(f"{d}: {len(stray)} target(s) are not {split} imputation targets of "
                          f"regime {regime} in the eval set, e.g. {stray[0]}")
        if meta.get("regions_sha256") and meta["regions_sha256"] != rsha:
            raise Refused(f"{d}: regions_sha256 {meta['regions_sha256'][:12]} differs from this "
                          f"eval set's {rsha[:12]}")

    ledger, entry = None, None
    if split == "blind":
        from minicandi.dataset import tables_sha256          # lazy: imports torch
        from onion_eval import score as S                     # lazy: imports torch
        ledger = evalset.parent / S.BLIND_LEDGER
        key = f"{LEDGER_PREFIX}{out.resolve()}"
        prior = [r for r in S.ledger_rows(ledger)
                 if r["run"] == key and r["evalset_sha256"] == esha]
        reason = (repeat_blind_reason or "").strip()
        if prior and not reason:
            raise Refused(f"{key} already had {len(prior)} blind-pass ledger rows for this eval set "
                          f"({ledger}); a repeat needs --repeat-blind-reason \"<text>\"")
        entry = {"run": key, "evalset_sha256": esha, "tables_sha256": tables_sha256(tables),
                 "git_sha": S.git_sha(), "out": str(out.resolve()),
                 "repeat_reason": reason if prior else ""}
        S.ledger_append(ledger, dict(entry, time=_utc(), status="started"))

    rows = E.read_tsv(tables / "measurements.tsv")
    by_id = {r["measurement_id"]: r for r in rows}
    excluded = E.second_experiment_ids(tables)
    sig_cache: dict = {}
    left_out = {HALF: {}, MEAN: {}, ORACLE: {}}
    summary = {}

    def cand_signal(r):
        mid = r["measurement_id"]
        if mid not in sig_cache:
            sig_cache[mid] = training_signal(r, by_id, regions, rb)
        return sig_cache[mid]

    for regime, (d, meta) in regimes.items():
        done = {HALF: [], MEAN: [], ORACLE: []}
        arrays = {HALF: {}, MEAN: {}, ORACLE: {}}
        picks = []
        cand_left = {}
        n_cands = {}
        for mid in meta.get("targets", []):
            row = by_id.get(mid)
            if row is None:
                raise Refused(f"{mid}: not in {tables / 'measurements.tsv'}")
            # pseudoreplicate_half: the only target files read here
            p = _h5_path(row)
            h1 = p.parent / "pr1_signal.h5" if p else None
            h2 = p.parent / "pr2_signal.h5" if p else None
            if h1 is not None and h1.is_file() and h2.is_file():
                arrays[HALF][f"{mid}.signal.npy"] = asinh(_read_h5(h1, regions, rb, np.float64))
                arrays[HALF][f"{mid}.truth.npy"] = asinh(_read_h5(h2, regions, rb, np.float64))
                done[HALF].append(mid)
            else:
                left_out[HALF][mid] = "no pr1_signal.h5 and pr2_signal.h5"
            # training tracks of the assay, other cells
            sigs, ids = [], []
            for r in candidate_rows(rows, row, excluded):
                s, why = cand_signal(r)
                if s is None:
                    cand_left[r["measurement_id"]] = why
                else:
                    sigs.append(s)
                    ids.append(r["measurement_id"])
            n_cands[mid] = len(ids)
            if not sigs:
                left_out[MEAN][mid] = left_out[ORACLE][mid] = "no usable training track"
                continue
            arrays[MEAN][f"{mid}.signal.npy"] = training_mean(sigs)
            done[MEAN].append(mid)
            tp = d / f"{mid}.truth.npy"
            if not tp.is_file():
                left_out[ORACLE][mid] = f"no truth in {d}"
                continue
            truth = np.load(tp).astype(np.float64)
            if truth.shape != (len(regions) * rb,):
                raise Refused(f"{tp}: shape {truth.shape}, expected {(len(regions) * rb,)}")
            orc = np.empty(len(regions) * rb, np.float64)
            for i in range(len(regions)):
                for b, (s0, s1) in enumerate(blocks(rb)):
                    sl = slice(i * rb + s0, i * rb + s1)
                    m, pk = closest_k(truth[sl], [s[sl] for s in sigs])
                    orc[sl] = m
                    picks.append({"target": mid, "region": i, "block": b, "start": s0,
                                  "end": s1, "picked": ";".join(ids[j] for j in pk)})
            arrays[ORACLE][f"{mid}.signal.npy"] = orc
            done[ORACLE].append(mid)
        for method in (HALF, MEAN, ORACLE):
            m = {"method": method, "kind": KIND[method], "evalset_sha256": esha,
                 "regions_sha256": rsha, "split": split, "regime": regime, "region_bins": rb,
                 "n_regions": len(regions), "lambda_source": LAMBDA_SOURCE[method],
                 "targets": done[method], "truth_from": str(d.resolve()),
                 "left_out": {k: v for k, v in left_out[method].items()
                              if k in meta.get("targets", [])}}
            if method != HALF:
                m["training_tracks_left_out"] = dict(sorted(cand_left.items()))
                m["n_training_tracks"] = n_cands
            _write_dir(out / method / regime, m, arrays[method])
        with open(out / ORACLE / regime / "picks.tsv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["target", "region", "block", "start", "end",
                                               "picked"], delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(picks)
        summary[regime] = {k: len(v) for k, v in done.items()}
        log(f"{regime}: " + ", ".join(f"{k} {len(v)} targets, {len(left_out[k])} left out"
                                      for k, v in done.items())
            + f"; training tracks left out {len(cand_left)}")

    if ledger is not None:
        from onion_eval import score as S
        S.ledger_append(ledger, dict(entry, time=_utc(), status="scored"))
    return {"regimes": summary, "left_out": left_out, "ledger": str(ledger) if ledger else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tables", required=True)
    r.add_argument("--evalset", required=True)
    r.add_argument("--truth-from", required=True)
    r.add_argument("--split", choices=("val", "blind"), required=True)
    r.add_argument("--allow-blind", action="store_true")
    r.add_argument("--repeat-blind-reason", default=None)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        run(a.tables, a.evalset, a.truth_from, a.out, split=a.split, allow_blind=a.allow_blind,
            repeat_blind_reason=a.repeat_blind_reason)
    except (Refused, PermissionError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
