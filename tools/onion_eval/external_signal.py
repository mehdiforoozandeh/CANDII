"""XE -- score a published EIC entry in the onion signal space, beside miniCANDI (ruling 13, D16).

    python tools/onion_eval/external_signal.py score --entry-dir <dir of C##M##.bigwig> \
        --entry <team id> --evalset <evalset_blind.json> --allow-blind --out <dir>/blind \
        [--repeat-blind-reason "<text>"] [--dnase-route ...] [--n-boot 200]
    -> <out>/scores.json (signal-space keys only), <out>/scores_full.json (score.py's own output),
       <out>/suffstats.npz, <out>/SCORED. An existing SCORED skips the job.
    python tools/onion_eval/external_signal.py table --scores <out dir> [...] \
        --model <run dir> [...] --out <tsv>
    -> one TSV with the miniCANDI rows and the entry rows (headline 1 and r_spec in signal_arcsinh).

What is scored. Each submitted track is read on the eval set's regions with `eic_score.BigWig` (the
pure-Python reader) and binned exactly as `eic_score.bin_bigwig` bins it for the challenge (the
mean of each 25 bp bin, missing = 0, the challenge's fix for the bin of the last item); only the
eval set's regions are binned (`read_regions`; equal to slices of `bin_bigwig`, test). It goes to
`score.score(..., signal_override=...)` (chunk SC): truth asinh(signal.h5), noise floor from the
target's `pr1_signal.h5` and `pr2_signal.h5`, training average = score.py's position average, so the
numbers sit on the same scale as the `signal_arcsinh` keys of a miniCANDI scores.json. No model is
retrained or run (D16).

The two places where the entry is not passed on unchanged (agent decisions, recorded per target in
scores.json under `entry_values`):
  - a negative value is set to 0 (`metrics.to_space` refuses negative signal; on Nibi 2026-11-11,
    301 of 1,284 submitted bigWigs have a negative minimum in their header summary);
  - a non-finite bin value is set to 0 (the challenge's missing = 0).
A target whose track the entry did not submit is left out: the entry is scored on an eval set
written beside the original, `<stem>_xe<team id>.json`, with only those targets removed (so the
blind ledger stays the original's). The removed targets are listed in `targets_without_entry_track`.
On 2026-11-11 one entry lacks one track (3391272, C38M18).

The signal-space error budget (`error_budget.signal_arcsinh`) is not written by score.py; it is
computed here with `error_budget.error_budget` on the same truth, entry signal and halves (noise
share per Haar scale from the halves, depth ratio 2, as the signal-space noise floor), per assay over
the targets that have halves, and averaged over the scorable assays like score.py's count-space one.
A miniCANDI scores.json has no signal-space error budget, so the table leaves those cells empty for
model rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from onion_eval import eic_score as ES  # noqa: E402
from onion_eval import score as S  # noqa: E402

SPACE = "signal_arcsinh"
TRAINING_AVERAGE = ("position_average (score.py: leave-one-cell-out position average of the "
                    "training cells, put through the same NB-to-signal route as for miniCANDI)")
EXTS = (".bigwig", ".bw")


# ------------------------------------------------------------------------------------ reading
def _bin_window(st, en, va, clen, s, rb, window, last_end):
    """Bins [s, s + rb) of one chromosome, as `eic_score.bin_bigwig` computes them."""
    lo, hi = s * window, min((s + rb) * window, clen)
    keep = (en > lo) & (st < hi)
    raw = ES._per_base(st[keep] - lo, en[keep] - lo, va[keep], hi - lo)
    buf = np.zeros(rb * window, np.float64)
    buf[:raw.shape[0]] = np.nan_to_num(raw)
    y = buf.reshape(rb, window).mean(axis=1)
    if last_end is not None:
        last = last_end // window
        if s <= last < s + rb:
            a, b = last * window, min((last + 1) * window, clen)
            near = np.flatnonzero((en > a) & (st < a + window))
            tot, cov = 0.0, 0
            for s0, e0, v0 in zip(st[near].tolist(), en[near].tolist(), va[near].tolist()):
                o = min(e0, b) - max(s0, a)
                if o > 0:
                    tot += o * float(v0)
                    cov += o
            y[last - s] = tot / cov if cov else 0.0
    return y.astype(np.float32)


def read_regions(path, regions, rb, window=ES.WINDOW) -> list:
    """float32 [rb] per (chrom, start bin) region of one bigWig, on the challenge's 25 bp grid."""
    by = defaultdict(list)
    for i, (c, s) in enumerate(regions):
        by[c].append((i, int(s)))
    out = [None] * len(regions)
    with ES.BigWig(path) as bw:
        sizes = bw.chroms()
        for c, lst in by.items():
            if c not in sizes:
                raise ValueError(f"{path}: no chromosome {c}")
            clen = sizes[c]
            n = (clen - 1) // window + 1
            st, en, va = bw.items(c)
            st, en = st.astype(np.int64), en.astype(np.int64)
            last_end = int(en[-1]) if en.size else None
            for i, s in lst:
                if s < 0 or s + rb > n:
                    raise ValueError(f"{path}: region {c}:{s}+{rb} runs past the {n} bins of {c}")
                out[i] = _bin_window(st, en, va, clen, s, rb, window, last_end)
    return out


class EntryTracks:
    """The signal_override of one entry: target row -> its submitted track on the regions.

    Keeps each target's regions in memory (a target is asked for once per regime), and records per
    target how many bins were negative or non-finite (both set to 0; module docstring)."""

    def __init__(self, entry_dir, rb):
        self.dir, self.rb = Path(entry_dir), int(rb)
        self.values: dict = {}
        self._cache: dict = {}

    def path(self, track):
        if not track:
            return None
        for ext in EXTS:
            p = self.dir / f"{track}{ext}"
            if p.is_file():
                return p
        return None

    def __call__(self, row, regions):
        mid = row["measurement_id"]
        if mid not in self._cache:
            p = self.path(row.get("eic_track", ""))
            if p is None:
                raise FileNotFoundError(f"{self.dir}: no track for {mid} "
                                        f"(eic_track {row.get('eic_track')!r})")
            y = np.concatenate(read_regions(p, regions, self.rb))
            bad = ~np.isfinite(y)
            y[bad] = 0.0
            neg = y < 0
            self.values[mid] = {"eic_track": row["eic_track"], "file": str(p),
                                "bins": int(y.size), "nonfinite_set_to_0": int(bad.sum()),
                                "negative_set_to_0": int(neg.sum()),
                                "min_before": float(y.min()) if y.size else None}
            y[neg] = 0.0
            self._cache[mid] = y
        return self._cache[mid]


# ------------------------------------------------------------------------------------ helpers
def _rows(ev, evp) -> dict:
    from onion_eval.evalset import read_tsv, tables_dir_of
    return {r["measurement_id"]: r for r in read_tsv(tables_dir_of(ev, evp) / "measurements.tsv")}


def _read_h5_regions(path, regions, rb):
    import h5py
    with h5py.File(path, "r") as f:
        return np.concatenate([np.asarray(f[c][s:s + rb], np.float64) for c, s in regions])


def signal_error_budget(scores, rows, tracks, regions, rb, scorable) -> dict:
    """Error budget in signal_arcsinh of the scored leave-one-assay-out targets with halves."""
    from onion_eval import error_budget as EB
    from onion_eval import metrics as Mx
    per = scores["specific_variance_recovered_above_noise"][SPACE].get("per_target", {})
    by = defaultdict(list)
    for mid, v in sorted(per.items()):
        if v.get("N") is not None:
            by[v["assay"]].append((mid, v["cell"]))
    per_assay = {}
    for a, lst in sorted(by.items()):
        T, P, H1, H2, cells = [], [], [], [], []
        for mid, cell in lst:
            d = Path(rows[mid]["counts_path"][3:]).parent
            T.append(Mx.to_space(_read_h5_regions(d / "signal.h5", regions, rb), SPACE))
            H1.append(Mx.to_space(_read_h5_regions(d / "pr1_signal.h5", regions, rb), SPACE))
            H2.append(Mx.to_space(_read_h5_regions(d / "pr2_signal.h5", regions, rb), SPACE))
            P.append(Mx.to_space(np.asarray(tracks._cache[mid], np.float64), SPACE))
            cells.append(cell)
        nbl = EB.noise_share_from_halves(np.stack(H1), np.stack(H2), 2.0)
        bud = EB.error_budget(np.stack(T), np.stack(P), np.array(cells), noise_by_level=nbl)
        per_assay[a] = {k: S._j(bud[k]) for k in S.BUDGET_KEYS}
        per_assay[a]["n_targets"] = len(lst)
    present = [a for a in scorable if a in per_assay]
    want = [a for a in scorable if a in {v["assay"] for v in per.values()}]
    out = {}
    for k in S.BUDGET_KEYS:
        vals = [per_assay[a][k] for a in present]
        out[k] = (float(np.mean(vals)) if vals and all(v is not None for v in vals)
                  and len(vals) == len(want) else None)
    out["per_assay"] = per_assay
    out["definition"] = ("error_budget.error_budget on asinh(signal.h5) and asinh(entry), noise share "
                         "per Haar scale from asinh(pr1_signal.h5), asinh(pr2_signal.h5) at depth "
                         "ratio 2; mean over the scorable assays (None if one is undefined)")
    return out


def signal_only(full, *, team, entry_dir, base_evalset, base_sha, used_evalset, missing, tracks,
                budget) -> dict:
    """The entry's scores.json: the signal-space keys and what is needed to compare them."""
    h1 = full["specific_variance_recovered_above_noise"][SPACE]
    keep = ("split", "evalset", "evalset_sha256", "tables", "tables_sha256", "scorable",
            "dnase_route", "n_regions", "region_bins", "noise_floor", "count_space_reason",
            "signal_source", "blind_ledger")
    out = {k: full[k] for k in keep if k in full}
    out.update({
        "predictor": S.EXTERNAL, "space": SPACE,
        "entry": {"team_id": int(team), "name": ES.TEAM_NAME[int(team)], "dir": str(entry_dir)},
        "evalset_base": str(base_evalset), "evalset_base_sha256": base_sha,
        "evalset_used": str(used_evalset),
        "targets_without_entry_track": missing,
        "n_targets": len(h1.get("per_target", {})),
        "specific_variance_recovered_above_noise": {SPACE: h1},
        "r_spec": {SPACE: full["r_spec"][SPACE]},
        "error_budget": {SPACE: budget},
        "noise_floor_files": "pr1_signal.h5, pr2_signal.h5 (score.py, Spearman-Brown per Haar level)",
        "training_average": TRAINING_AVERAGE,
        "entry_values": tracks.values,
        "scores_full": "scores_full.json",
    })
    return out


# ------------------------------------------------------------------------------------ commands
def cmd_score(a) -> int:
    out = Path(a.out)
    if (out / S.SCORED).is_file():
        print(f"skip: {out}/{S.SCORED} exists")
        return 0
    team = int(a.entry)
    if team not in ES.TEAM_NAME:
        print(f"refused: {team} is not one of the {len(ES.TEAMS)} published entries", file=sys.stderr)
        return 2
    split = "blind" if a.allow_blind else "val"
    evp = Path(a.evalset).resolve()
    ev = json.loads(evp.read_text(encoding="utf-8"))
    rows = _rows(ev, evp)
    rb = int(ev.get("region_bins", 2 ** 14))
    tracks = EntryTracks(a.entry_dir, rb)
    ids = sorted({t["target"] for t in ev["imputation_targets"] if t["split"] == split})
    missing = [m for m in ids if tracks.path(rows[m].get("eic_track", "")) is None]
    if len(missing) == len(ids):
        print(f"refused: {a.entry_dir} holds no track of the {len(ids)} {split} targets",
              file=sys.stderr)
        return 2
    used = evp
    if missing:
        used = evp.with_name(f"{evp.stem}_xe{team}.json")
        ev2 = dict(ev, imputation_targets=[t for t in ev["imputation_targets"]
                                           if t["target"] not in set(missing)],
                   xe_derived_from=str(evp), xe_derived_from_sha256=S.sha256_file(evp),
                   xe_dropped_targets=missing)
        text = json.dumps(ev2, indent=1, sort_keys=True) + "\n"
        if not used.is_file() or used.read_text(encoding="utf-8") != text:
            used.write_text(text, encoding="utf-8")
        print(f"entry {team}: no track for {len(missing)} target(s) {missing}; scoring {used}")
    S.set_threads()
    try:
        full = S.score(used, out, signal_override=tracks, split=split, allow_blind=a.allow_blind,
                       n_boot=a.n_boot, dnase_route=a.dnase_route,
                       repeat_blind_reason=a.repeat_blind_reason)
    except (PermissionError, S.ScoreRefused) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    budget = signal_error_budget(full, rows, tracks, [(c, int(s)) for c, s in json.loads(
        used.read_text(encoding="utf-8"))["regions"]], rb, full["scorable"].get("imputation", []))
    sig = signal_only(full, team=team, entry_dir=a.entry_dir, base_evalset=evp,
                      base_sha=S.sha256_file(evp), used_evalset=used, missing=missing,
                      tracks=tracks, budget=budget)
    (out / "scores.json").replace(out / "scores_full.json")
    tmp = out / "scores.json.tmp"
    tmp.write_text(json.dumps(sig, indent=1, sort_keys=True, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(out / "scores.json")
    h1 = sig["specific_variance_recovered_above_noise"][SPACE]
    print(f"entry {team} ({ES.TEAM_NAME[team]}): headline 1 {SPACE} {h1['macro']} "
          f"(ci {h1['ci']}); r_spec {sig['r_spec'][SPACE]['macro']} -> {out}/scores.json")
    return 0


TABLE_COLUMNS = ("kind", "name", "team_id", "split", "n_targets", "h1_signal_arcsinh", "h1_ci_lo",
                 "h1_ci_hi", "r_spec_signal_arcsinh", "r_spec_ci_lo", "r_spec_ci_hi") + tuple(
    f"budget_{k}" for k in S.BUDGET_KEYS) + ("n_targets_without_entry_track", "reason", "source")


def _model_scores(d: Path) -> Path:
    for p in (d / "blind" / "scores.json", d / "scores.json"):
        if p.is_file():
            return p
    raise FileNotFoundError(f"{d}: no blind/scores.json or scores.json")


def _table_row(kind, s, src) -> dict:
    h1 = s["specific_variance_recovered_above_noise"][SPACE]
    rs = s["r_spec"][SPACE]
    bud = (s.get("error_budget") or {}).get(SPACE) if kind == "external" else None
    row = {"kind": kind, "split": s.get("split"), "source": str(src),
           "n_targets": s.get("n_targets"),
           "h1_signal_arcsinh": h1.get("macro"), "h1_ci_lo": (h1.get("ci") or [None, None])[0],
           "h1_ci_hi": (h1.get("ci") or [None, None])[1], "r_spec_signal_arcsinh": rs.get("macro"),
           "r_spec_ci_lo": (rs.get("ci") or [None, None])[0],
           "r_spec_ci_hi": (rs.get("ci") or [None, None])[1],
           "reason": "; ".join(x for x in (h1.get("reason"), rs.get("reason")) if x)}
    for k in S.BUDGET_KEYS:
        row[f"budget_{k}"] = bud.get(k) if bud else None
    if kind == "external":
        row.update(name=s["entry"]["name"], team_id=s["entry"]["team_id"],
                   n_targets_without_entry_track=len(s.get("targets_without_entry_track", [])))
    else:
        row.update(name=str(s.get("run") or src.parent), team_id="",
                   n_targets_without_entry_track="")
    for a, v in sorted(h1.get("per_assay", {}).items()):
        row[f"h1_{a}"] = v
    for a, v in sorted(rs.get("per_assay", {}).items()):
        row[f"r_spec_{a}"] = v
    row["_evalset"] = s.get("evalset_base_sha256") or s.get("evalset_sha256")
    row["_tables"] = s.get("tables_sha256")
    return row


def build_table(scores_dirs, model_dirs, out) -> list[dict]:
    rows = []
    for d in model_dirs:
        p = _model_scores(Path(d))
        rows.append(_table_row("model", json.loads(p.read_text(encoding="utf-8")), p))
    for d in scores_dirs:
        p = Path(d) / "scores.json"
        s = json.loads(p.read_text(encoding="utf-8"))
        if s.get("predictor") != S.EXTERNAL or "entry" not in s:
            raise ValueError(f"{p}: not an external_signal.py scores.json")
        rows.append(_table_row("external", s, p))
    if not rows:
        raise ValueError("no scores given")
    for key, what in (("split", "split"), ("_evalset", "eval set sha256"),
                      ("_tables", "tables sha256")):
        vals = {r[key] for r in rows}
        if len(vals) != 1:
            raise ValueError(f"rows differ in {what}: " + "; ".join(
                f"{r['source']}={r[key]}" for r in rows))
    extra = sorted({k for r in rows for k in r if k.startswith(("h1_", "r_spec_"))}
                   - set(TABLE_COLUMNS))
    cols = list(TABLE_COLUMNS) + extra
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    return rows


def cmd_table(a) -> int:
    rows = build_table(a.scores or [], a.model or [], a.out)
    print(f"{len(rows)} rows -> {a.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("score", help="score one entry's submitted bigWigs")
    p.add_argument("--entry-dir", required=True)
    p.add_argument("--entry", required=True, help="team id (eic_score.TEAMS)")
    p.add_argument("--evalset", required=True)
    p.add_argument("--out", required=True, help="a blind pass writes into a dir named 'blind'")
    p.add_argument("--allow-blind", action="store_true", help="score the blind targets (gate G4)")
    p.add_argument("--repeat-blind-reason", default=None)
    p.add_argument("--dnase-route", default=None, choices=("hotspot2_density", "macs2_no_control"),
                   help="route of the training average's DNase-seq signal (score.py's option)")
    p.add_argument("--n-boot", type=int, default=200)
    p.set_defaults(fn=cmd_score)
    t = sub.add_parser("table", help="entries and miniCANDI rows in one TSV")
    t.add_argument("--scores", nargs="*", help="external_signal.py score output dirs")
    t.add_argument("--model", nargs="*", help="miniCANDI run dirs (reads blind/scores.json, else "
                                              "scores.json)")
    t.add_argument("--out", required=True)
    t.set_defaults(fn=cmd_table)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
