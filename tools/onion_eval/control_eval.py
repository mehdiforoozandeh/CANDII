"""C6 (plan t370, decision D9, first step): leave-control-out evaluation of the predicted ChIP control.

    python tools/onion_eval/control_eval.py run --run <run dir> --evalset <json> --out <dir>
        [--regions 64] [--cells c1,c2] [--device cpu] [--infer-batch 8] [--stage-dir <dir>]
    -> <out>/control_eval.json, <out>/CONTROL_EVAL_DONE
    python tools/onion_eval/control_eval.py summary --runs <dir> [--prefix p] --out <tsv>

Question: in a cell without a ChIP-seq control, can the model's predicted control stand in for the
observed one?

Cells. Every cell of the eval set's validation ChIP-seq targets (regime `leave_one_assay_out`) that
has an observed base control (the cell's training base view of `chipseq-control`, or its
validation control target). ChIP-seq targets of that cell count when their row names an observed
control (`control_measurement_id`).

Inputs. The cell's leave-one-assay-out inputs (the union of the inputs of its validation targets)
with every control row removed. The control never reaches the inputs: each batch is checked
(`x_avail` of the control track is false), and a test changes the control counts and finds the
prediction unchanged.

Predictions (`registry.load_run`; the scorer's `Scorer` builds the windows and its batches):
  - the control at the control's own C' (target = the observed base control), scored in counts;
  - the control at each ChIP-seq target's own C' (`Scorer._predicted_control`, the scorer's
    `--lambda-source predicted_control` route), used for the signal.

Count space (`control_count`), per cell over the scored bins, then the mean over the cells of the
group: `crps_skill_vs_training_average` = CRPS(training average) - CRPS(predicted NB) (positive =
the model is better), where the training average is `Scorer._average_control_rate` (leave-one-cell-
out mean control rate) times the control's depth, with the NB size of the scorer's position average
of the control; `log1p_pearson` and `log1p_r2` of log1p(predicted mean) against log1p(observed).

Signal space (`signal`), per ChIP-seq target, then the mean over the targets of the group. The
signal p-value (`signal_from_counts.signal_from_nb`, `plug_in` mode on the target's OBSERVED counts,
`box_d` pileup) and the fold change log2((pileup + 1) / (lambda + 1)) are computed three times:
"obs" with the observed control (the row's own control and its depth and read length), "pred"
with the predicted control at the target's C' and "avg" with the training-average control at the
target's depth (both through `predicted_control=`, i.e. the control is taken at the treatment's
depth and read length, as the scorer does). Keys: Pearson of arcsinh(p-value) and of the log2 fold
change, pred vs obs and avg vs obs. The `box_reach` bins at both ends of each region are masked
(`signal_from_counts.edge_keep`), as in the scorer.

Groups: `all`, `held_out` (cells in the run's `control_holdout_cells`), `in_sample` (the others).
A group with no cell gives null values. Nothing is written into a scorer's directory: an --out that
holds scores.json or SCORED is refused.

`summary`: per arm directory under --runs (name starting with --prefix), every control_eval.json
below it is one seed; the TSV gives `n_seeds` and the mean and range (max - min) of every key.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

DONE = "CONTROL_EVAL_DONE"
OUT_JSON = "control_eval.json"
GROUPS = ("all", "held_out", "in_sample")
COUNT_KEYS = ("crps_skill_vs_training_average", "log1p_pearson", "log1p_r2")
SIGNAL_KEYS = ("pval_arcsinh_pearson_pred_vs_obs", "pval_arcsinh_pearson_avg_vs_obs",
               "fc_log2_pearson_pred_vs_obs", "fc_log2_pearson_avg_vs_obs")
TOP_KEYS = ("n_cells", "n_control_targets", "n_chip_targets", "regions_n")
PILEUP = "box_d"
SIGNAL_MODE = "plug_in"
#: observed counts can be 0; the plug-in score needs a positive mean (floor(eps * s) = 0 = floor(0))
OBS_EPS = 1e-6


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def holdout_cells(config: dict) -> set:
    v = (config or {}).get("control_holdout_cells") or []
    if isinstance(v, str):
        v = [c for c in v.split(",") if c.strip()]
    return {str(c).strip() for c in v}


def region_subset(n_all: int, n: int | None) -> list[int]:
    """Evenly spaced positions of `n` regions among `n_all` (all when n is None or >= n_all)."""
    if n is None or n >= n_all:
        return list(range(n_all))
    if n < 1:
        raise ValueError(f"--regions {n}: want >= 1")
    return sorted({int(i) for i in np.round(np.linspace(0, n_all - 1, n))})


def _corr(x, y, mask=None) -> float:
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        ok &= mask
    if ok.sum() < 2:
        return float("nan")
    x, y = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    d = math.sqrt(float(x @ x) * float(y @ y))
    return float(x @ y / d) if d > 0 else float("nan")


def _r2(pred, obs) -> float:
    pred, obs = np.asarray(pred, np.float64), np.asarray(obs, np.float64)
    ok = np.isfinite(pred) & np.isfinite(obs)
    if ok.sum() < 2:
        return float("nan")
    tot = float(((obs[ok] - obs[ok].mean()) ** 2).sum())
    return 1.0 - float(((obs[ok] - pred[ok]) ** 2).sum()) / tot if tot > 0 else float("nan")


def _num(x):
    """JSON value: a finite float, else None."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _mean(vals) -> float | None:
    v = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    return float(np.mean(v)) if v else None


def log2_fold_change(y, ctl, row_t: dict, ctl_cov: dict, regions, rb: int, sig) -> np.ndarray:
    """log2((pileup + 1) / (lambda + 1)) per bin over the concatenated regions, with the box_d
    pileup and MACS2's local lambda of `signal_from_counts` (each region spread alone).
    `ctl_cov`: control depth_reads and read_length (the observed control's, or the treatment's own
    for a control taken at the treatment's C')."""
    d = float(row_t["frag_len"])
    depth, rl = float(row_t["depth_reads"]), float(row_t["read_length"])
    cdepth, crl = float(ctl_cov["depth_reads"]), float(ctl_cov["read_length"])
    out = []
    for i in range(len(regions)):
        sl = slice(i * rb, (i + 1) * rb)
        pile = sig.pileup_from_counts(sig.box_spread(np.asarray(y, np.float64)[sl], d), d, rl)
        lam = sig.macs2_lambda(lambda_bg=d * depth / sig.MACS2_GSIZE_HS,
                               control_pileup=sig.pileup_from_counts(
                                   sig.box_spread(np.asarray(ctl, np.float64)[sl], d), d, crl),
                               ratio_treat2control=depth / cdepth)
        out.append(np.log2((pile + 1.0) / (lam + 1.0)))
    return np.concatenate(out)


# --------------------------------------------------------------------------------------- the plan

def cell_plan(sc, cells=None) -> dict:
    """cell -> {control, inputs, targets}: the cells that have an observed base control and at least
    one validation ChIP-seq target whose row names an observed control."""
    from onion_eval.score import CONTROL
    rows = sc.ds.rows
    val_ctl = {t["cell"] if "cell" in t else rows[t["target"]]["cell"]: t["target"]
               for t in sc.ev.get("control_targets", []) if t["split"] == "val"}
    by_cell: dict = {}
    for t in sc.ev["imputation_targets"]:
        if t["split"] != "val" or t.get("regime") != "leave_one_assay_out":
            continue
        row = rows[t["target"]]
        try:
            route = sc.sig.route_of(row["assay_class"] or row["assay"])
        except ValueError:
            continue
        if route != "chip":
            continue
        cell = row["cell"]
        e = by_cell.setdefault(cell, {"inputs": set(), "targets": []})
        e["inputs"] |= set(t["inputs"])
        if (row.get("control_measurement_id") or "").strip() in rows:
            e["targets"].append(t["target"])
    want = None if cells is None else set(cells)
    plan = {}
    for cell, e in sorted(by_cell.items()):
        if want is not None and cell not in want:
            continue
        ctl = sc.ds.train_base.get((cell, CONTROL)) or val_ctl.get(cell)
        if not ctl or not e["targets"]:
            continue
        inputs = sorted(m for m in e["inputs"] if rows[m]["assay"] != CONTROL)
        plan[cell] = {"control": ctl, "inputs": inputs, "targets": sorted(set(e["targets"]))}
    if want is not None and want - set(plan):
        sc.log(f"control_eval: cells without a scorable control or target: {sorted(want - set(plan))}")
    return plan


def _check_no_control_input(sc, batch) -> None:
    from onion_eval.score import CONTROL
    ci = sc.C.TRACK_INDEX[CONTROL]
    if bool(batch["x_avail"][:, ci].any()) or bool(batch["x_counts"][:, :, ci].any()):
        raise AssertionError("the control reached the inputs")


def predict_cell(sc, entry: dict) -> dict:
    """The model's arrays of one cell over sc.regions (region then tile order): `ctl.n`, `ctl.mu`
    (the control at its own C') and `pc.<target>` (the control mean at the target's C')."""
    from onion_eval.score import CONTROL
    ci = sc.C.TRACK_INDEX[CONTROL]
    out: dict = {"ctl.n": [], "ctl.mu": []}
    for t in entry["targets"]:
        out[f"pc.{t}"] = []
    for ri, tiles in sc._tiles(sc.regions):
        sc._set_region(*sc.regions[ri])
        for b0 in range(0, len(tiles), sc.tile_bs):
            tl = tiles[b0:b0 + sc.tile_bs]
            batch = sc.C.collate([sc.ds.window(c, s, entry["inputs"], [entry["control"]])
                                  for c, s in tl])
            _check_no_control_input(sc, batch)
            p = sc.model_pred(batch)
            out["ctl.n"].append(np.asarray(p["n"][:, :, ci], np.float64).reshape(-1))
            out["ctl.mu"].append(np.asarray(p["mu"][:, :, ci], np.float64).reshape(-1))
            for tgt in entry["targets"]:
                ti = sc.C.TRACK_INDEX[sc.ds.rows[tgt]["assay"]]
                b = sc.C.collate([sc.ds.window(c, s, entry["inputs"], [tgt]) for c, s in tl])
                _check_no_control_input(sc, b)
                out[f"pc.{tgt}"].append(np.asarray(sc._predicted_control(b, ti),
                                                   np.float64).reshape(-1))
    return {k: np.concatenate(v) for k, v in out.items()}


def _read_regions(sc, mid) -> np.ndarray:
    parts = []
    for c, s in sc.regions:
        sc._set_region(c, s)
        parts.append(np.asarray(sc.ds.read(mid, c, s, sc.rb), np.float64))
    return np.concatenate(parts)


def _avg_control(sc, cell, depth: float) -> np.ndarray:
    """The training-average control mean at `depth` over sc.regions (NaN where undefined)."""
    from onion_eval.baselines import PRED_FLOOR
    rates = [sc._average_control_rate(ri, cell) for ri in range(len(sc.regions))]
    return np.maximum(np.concatenate(rates) * depth, PRED_FLOOR)       # NaN stays NaN


def _avg_control_size(sc) -> np.ndarray:
    """Per bin, the NB size of the scorer's position average of the control, fitted per region
    (NaN where no training control exists); the fits are dropped after each region."""
    from onion_eval.score import CONTROL
    ci = sc.C.TRACK_INDEX[CONTROL]
    if getattr(sc, "_ctl_size", None) is None:
        parts = []
        for ri in range(len(sc.regions)):
            size = sc._pa(CONTROL, ri).size.get(ci)
            sc._pa_cache.clear()
            parts.append(np.full(sc.rb, np.nan if size is None else float(size)))
        sc._ctl_size = np.concatenate(parts)
    return sc._ctl_size


def score_cell(sc, cell: str, entry: dict, arrays: dict) -> dict:
    from onion_eval import metrics as Mx
    rows = sc.ds.rows
    crow = rows[entry["control"]]
    y = _read_regions(sc, entry["control"])
    depth_c = 2.0 ** float(sc._encoder_depth(crow)[0])
    avg_mu, avg_n = _avg_control(sc, cell, depth_c), _avg_control_size(sc)
    n, mu = arrays["ctl.n"], arrays["ctl.mu"]
    m_c = Mx.crps_split(n, mu, y)["crps"]
    if np.isfinite(avg_n).all() and np.isfinite(avg_mu).all():
        a_c = Mx.crps_split(avg_n, avg_mu, y)["crps"]
        skill = float(a_c - m_c)
        why = ""
    else:
        skill, why = float("nan"), "no training-average control on every bin"
    cc = {"crps_skill_vs_training_average": _num(skill),
          "log1p_pearson": _num(_corr(np.log1p(mu), np.log1p(y))),
          "log1p_r2": _num(_r2(np.log1p(mu), np.log1p(y)))}
    if why:
        cc["reason"] = why
    kw = dict(ds=sc.ds, regions=sc.regions, rb=sc.rb, mode=SIGNAL_MODE, pileup=PILEUP)
    sig = sc.sig
    targets = {}
    for tgt in entry["targets"]:
        row = rows[tgt]
        rec = {"assay": row["assay"], "observed_control": row["control_measurement_id"]}
        try:
            yt = _read_regions(sc, tgt)
            ones, yt_pos = np.ones_like(yt), np.maximum(yt, OBS_EPS)
            d = sig.box_d_of(row)
            if d is None:
                raise ValueError(f"{tgt}: no frag_len")
            keep = sig.edge_keep(len(sc.regions), sc.rb, sig.box_reach(d))
            no_ctl = dict(row, control_measurement_id="")
            p_obs, _ = sig.signal_from_nb(ones, yt_pos, row, **kw)
            p_pred, _ = sig.signal_from_nb(ones, yt_pos, no_ctl, predicted_control=arrays[f"pc.{tgt}"],
                                           **kw)
            depth_t = 2.0 ** float(sc._encoder_depth(row)[0])
            avg_t = _avg_control(sc, cell, depth_t)
            ok_avg = bool(np.isfinite(avg_t).all())
            p_avg = (sig.signal_from_nb(ones, yt_pos, no_ctl, predicted_control=avg_t, **kw)[0]
                     if ok_avg else None)
            ocrow = rows[row["control_measurement_id"]]
            own = {"depth_reads": row["depth_reads"], "read_length": row["read_length"]}
            fc_obs = log2_fold_change(yt, _read_regions(sc, ocrow["measurement_id"]), row,
                                      ocrow, sc.regions, sc.rb, sig)
            fc_pred = log2_fold_change(yt, arrays[f"pc.{tgt}"], row, own, sc.regions, sc.rb, sig)
            fc_avg = (log2_fold_change(yt, avg_t, row, own, sc.regions, sc.rb, sig)
                      if ok_avg else None)
            a = lambda v: np.arcsinh(np.asarray(v, np.float64))  # noqa: E731
            rec.update({
                "pval_arcsinh_pearson_pred_vs_obs": _num(_corr(a(p_pred), a(p_obs), keep)),
                "pval_arcsinh_pearson_avg_vs_obs": (_num(_corr(a(p_avg), a(p_obs), keep))
                                                    if ok_avg else None),
                "fc_log2_pearson_pred_vs_obs": _num(_corr(fc_pred, fc_obs, keep)),
                "fc_log2_pearson_avg_vs_obs": (_num(_corr(fc_avg, fc_obs, keep))
                                               if ok_avg else None)})
            if not ok_avg:
                rec["reason"] = "no training-average control on every bin"
        except (ValueError, KeyError) as e:
            rec.update({k: None for k in SIGNAL_KEYS})
            rec["reason"] = f"no signal: {e}"
        targets[tgt] = rec
    return {"control": entry["control"], "inputs": entry["inputs"], "control_count": cc,
            "targets": targets}


def group_block(per_cell: dict, cells) -> dict:
    cells = [c for c in cells if c in per_cell]
    recs = [r for c in cells for r in per_cell[c]["targets"].values()]
    return {"n_cells": len(cells), "n_chip_targets": len(recs),
            "control_count": {k: _mean(per_cell[c]["control_count"][k] for c in cells)
                              for k in COUNT_KEYS},
            "signal": {k: _mean(r[k] for r in recs) for k in SIGNAL_KEYS}}


def build_scorer(evalset, *, model, config, run_tables_sha256, regions=64, device="cpu",
                 infer_batch=None, log=print):
    """The scorer that builds the windows and batches, restricted to `regions` eval regions."""
    from onion_eval import score as S
    sc = S.Scorer(evalset, main="model", model=model, config=config,
                  run_tables_sha256=run_tables_sha256, device=device, infer_batch=infer_batch,
                  log=log, signal_pileup=PILEUP)
    if sc.sig is None:
        raise S.ScoreRefused(f"no signal module: {sc.sig_reason}")
    if not sc.predict_control:
        raise S.ScoreRefused("the run does not predict the control (predict_control off)")
    keep = region_subset(len(sc.regions), regions)
    sc.regions = [sc.regions[i] for i in keep]
    return sc


def evaluate(evalset, out, *, model, config, run_tables_sha256, regions=64, cells=None,
             device="cpu", infer_batch=None, stage_dir=None, run=None, log=print) -> dict:
    """Score the predicted control of `model`; write <out>/control_eval.json and DONE."""
    from onion_eval import score as S
    out = Path(out)
    if (out / "scores.json").exists() or (out / S.SCORED).exists():
        raise S.ScoreRefused(f"--out {out} is a scorer's directory; control_eval writes elsewhere")
    t0 = time.time()
    sc = build_scorer(evalset, model=model, config=config, run_tables_sha256=run_tables_sha256,
                      regions=regions, device=device, infer_batch=infer_batch, log=log)
    plan = cell_plan(sc, cells)
    if stage_dir:
        sc.stage_dir = Path(stage_dir)
        imp = [{"target": t, "inputs": e["inputs"], "assay": sc.ds.rows[t]["assay"]}
               for e in plan.values() for t in e["targets"]]
        ctl = [{"target": e["control"], "inputs": e["inputs"], "assay": S.CONTROL}
               for e in plan.values()]
        sc.stage_info = sc._stage(imp, [], [], ctl)
    held = holdout_cells(config)
    per_cell = {}
    for cell, entry in plan.items():
        log(f"control_eval: {cell}: control {entry['control']}, {len(entry['targets'])} ChIP-seq "
            f"targets, {len(entry['inputs'])} inputs")
        res = score_cell(sc, cell, entry, predict_cell(sc, entry))
        res["in_sample"] = cell not in held
        per_cell[cell] = res
    groups = {"all": list(per_cell), "held_out": [c for c in per_cell if c in held],
              "in_sample": [c for c in per_cell if c not in held]}
    result = {
        "n_cells": len(per_cell), "n_control_targets": len(per_cell),
        "n_chip_targets": sum(len(r["targets"]) for r in per_cell.values()),
        "regions_n": len(sc.regions), "regions": [list(r) for r in sc.regions],
        "tables_sha256": sc.tables_sha256, "evalset_sha256": sha256_file(sc.evalset_path),
        "control_holdout_cells": sorted(held), "signal_pileup": PILEUP, "signal_mode": SIGNAL_MODE,
        "run": str(Path(run).resolve()) if run else None, "per_cell": per_cell,
        **{g: group_block(per_cell, cs) for g, cs in groups.items()},
        "wall_seconds": round(time.time() - t0, 1)}
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / (OUT_JSON + ".tmp")
    tmp.write_text(json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(out / OUT_JSON)
    (out / DONE).write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n",
                            encoding="utf-8")
    return result


# --------------------------------------------------------------------------------------- summary

def flat_keys(res: dict) -> dict:
    """The summary keys of one control_eval.json: TOP_KEYS and <group>.<block>.<key>."""
    flat = {k: res.get(k) for k in TOP_KEYS}
    for g in GROUPS:
        b = res.get(g) or {}
        for blk, keys in (("control_count", COUNT_KEYS), ("signal", SIGNAL_KEYS)):
            for k in keys:
                flat[f"{g}.{blk}.{k}"] = (b.get(blk) or {}).get(k)
    return flat


def summary(runs, prefix: str = "") -> list[dict]:
    rows = []
    for arm in sorted(p for p in Path(runs).iterdir() if p.is_dir() and p.name.startswith(prefix)):
        files = sorted(arm.rglob(OUT_JSON))
        if not files:
            continue
        seeds = [flat_keys(json.loads(f.read_text(encoding="utf-8"))) for f in files]
        row = {"arm": arm.name, "n_seeds": len(seeds)}
        for k in seeds[0]:
            v = [float(s[k]) for s in seeds if s.get(k) is not None]
            row[f"{k}_mean"] = float(np.mean(v)) if v else None
            row[f"{k}_range"] = float(max(v) - min(v)) if v else None
        rows.append(row)
    return rows


def write_summary(rows: list[dict], out) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0]) if rows else ["arm", "n_seeds"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else
                            f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k in cols})


# --------------------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run", required=True)
    r.add_argument("--evalset", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--regions", type=int, default=64)
    r.add_argument("--cells", default=None, help="comma-separated cell ids (default: all)")
    r.add_argument("--device", default="cpu")
    r.add_argument("--infer-batch", type=int, default=8)
    r.add_argument("--stage-dir", default=None)
    s = sub.add_parser("summary")
    s.add_argument("--runs", required=True)
    s.add_argument("--prefix", default="")
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "summary":
        rows = summary(a.runs, a.prefix)
        write_summary(rows, a.out)
        print(f"control_eval summary: {len(rows)} arms -> {a.out}")
        return 0
    from minicandi.registry import load_run
    from onion_eval import score as S
    S.tf32_off()
    S.set_threads()
    sha = S.run_tables_sha256(a.run)
    model, config = load_run(a.run, map_location=a.device)
    cells = [c for c in a.cells.split(",") if c] if a.cells else None
    res = evaluate(a.evalset, a.out, model=model, config=config, run_tables_sha256=sha,
                   regions=a.regions, cells=cells, device=a.device, infer_batch=a.infer_batch,
                   stage_dir=a.stage_dir, run=a.run)
    print(f"control_eval: {res['n_cells']} cells, {res['n_chip_targets']} ChIP-seq targets, "
          f"{res['regions_n']} regions -> {Path(a.out) / OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
