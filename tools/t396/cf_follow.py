"""t396 B2: does a trained model follow the dedup and the run-type covariate on the controlled pairs?

    python tools/t396/cf_follow.py pairs --tables <dir> --out pairs.tsv
    python tools/t396/cf_follow.py predict --run <run dir> --pairs pairs.tsv --evalset <json>
        --out <dir> [--shard i/N] [--model-name <name>] [--infer-batch B] [--device cpu|cuda]
    python tools/t396/cf_follow.py table --pred <dirs...> [--caps-table <B1 table.tsv ...>]
        --out <dir>

Controlled pair: two `layer = testbed` rows of one track that differ in one processing setting:
(`.base`, `.nodedup`) (the variant has `arm_knob = nodedup`; covariate `dedup`) or (`.base`, `.pe`)
(`arm_knob = se`, `arm_value = pe`; covariate `run_type`). The base row is the variant's
`parent_measurement_id`. Rows are chosen by `layer` and `arm_knob`, never by `in_eic`.

pairs -> one row per controlled pair and direction (`base_to_variant`: source = base, target =
  variant; `variant_to_base`: the reverse). Columns: track (the `experiment`), assay, covariate,
  direction, source, target, fields_changed. `fields_changed` (comma list, `covariate_prompt.COL`
  names) holds the output-covariate fields of the covariate that differ between the two rows'
  encoded C' (`CovariateEncoder.c_out`): `dedup` -> {dedup_on}; `run_type` -> {run_type_pe,
  log2_depth, read_length}, each only if it differs.

predict -> per pair `<out>/<track>__<covariate>__<direction>.npz` and `<out>/meta.json`.
  Window: `OnionDataset.window(chrom, start, [source] + the cell's other training tracks,
  [target], halves_tracks=[assay])`, the training path of a read-disjoint pair: the source track
  reads its `pr1` half, the target its `pr2` half, both at depth scale 0.5; every other input is
  the cell's training base view (`evalset.base_views`, roles `train`) of another assay, read in
  full, as an imputation job builds its inputs. The dataset runs in mode `blind` (the testbed rows
  are `in_eic = blind`). Two forwards per batch: (1) the batch as built, whose C' row of the target
  track is the target's (`target_cov`); (2) a copy whose C' row of the target track has the fields
  of `fields_changed` set to the source's values and nothing else changed (`source_cov`).
  Regions: every region of the evaluation set, tiled as `Scorer._tiles`.
  Spaces: `overlap_count` for every run (the adapter's converted output for a read-start run);
  `start_sum` in addition for a read-start run (`ReadStartAdapter.inner`'s `mu_sub`, `n_sub`
  pooled over strands by `score_spaces.pool_strands`).
  npz keys: track, assay, covariate, direction, source, target, fields_changed (str), spaces (str
  list), region_indices int64 [R]; per space S: `mu_tgt_S`, `n_tgt_S`, `mu_src_S`, `n_src_S`
  float32 [R * rb] (prediction for the target's pr2 half at depth scale 0.5), `y_S` (target pr2),
  `y1_S` (target pr1), `x_S` (source pr1, the input of the pair track) int32 [R * rb], and
  `shares_S` (f1, f2) of the target's halves (`score._half_shares` for counts,
  `dump_native._shares_starts` for starts). An existing npz is skipped.

table -> `table.tsv` and `per_track.tsv`. Per track, space and scope, with mu scaled to the
  target's pr2 share (mu * f2 / 0.5) and `deviance.deviance` (Poisson):
    D_source_cov = deviance(y, mu_src), D_target_cov = deviance(y, mu_tgt),
    D_halves = deviance(y, max(y1 * f2 / f1, MU_MIN)) (the half-vs-half floor of the target);
    follow_score = (D_source_cov - D_target_cov) / (D_source_cov - D_halves).
  A track whose denominator is <= 0 is excluded from its row and counted in `tracks_excluded`.
  Scopes: `genome` (all bins of the 256 regions) and `top1pct` (the ceil(1%) bins with the largest
  source pr1 `x`, stable order). Pooled over tracks (one seed): sum of numerators / sum of
  denominators over the included tracks. `follow_score` = mean of the pooled score over seeds;
  `seed_range` = max - min; `per_seed` = `seed:score` list; `halves_floor` = mean over seeds of
  sum D_halves / sum D_source_cov over the included tracks; `tracks_following` = tracks whose
  seed-mean score is >= 0.8; `tracks_excluded` = the largest count over seeds; `total_ratio` =
  mean over seeds of sum mu_tgt / sum mu_src over every track; `follows` (gate h, top1pct rows
  only) = `yes` when follow_score >= 0.8. With `--caps-table` (B1 testbed `table.tsv` files of
  dedup_cap.py and sepe_cap.py), `caps_gap_closed` is B1's `gap_closed_vs_depth_scaling` of the row
  with the same covariate and scope, space count (overlap_count) or start (start_sum), shape
  `loto`, assay_group `all`, law count_saturation (count) or subsampled_dm_distinct (start). It is
  set on `variant_to_base` rows only (B1 scores variant -> base); nan on the others. A needed key
  with no row, or a key with more than one row, is an error.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

#: C' column index of each META column (as `onion_eval.covariate_prompt.COL`)
COL = {"log2_depth": 0, "run_type_pe": 1, "read_length": 2, "dedup_on": 3, "mapq_threshold": 4,
       "pipeline_id": 5, "pipeline_version_id": 6, "lab_id": 7, "platform_id": 8,
       "antibody_id": 9, "donor_id": 10}
#: covariate -> ((arm_knob, arm_value or None for any), candidate fields in COL order)
COVARIATES = {"dedup": (("nodedup", None), ("dedup_on",)),
              "run_type": (("se", "pe"), ("log2_depth", "run_type_pe", "read_length"))}
DIRECTIONS = ("base_to_variant", "variant_to_base")
PAIR_COLUMNS = ("track", "assay", "covariate", "direction", "source", "target", "fields_changed")
SCOPES = ("genome", "top1pct")
TOP_FRAC = 0.01
GATE_H = 0.8
HALF = 0.5                      # the depth scale of a pr1 / pr2 half in C'
LIMIT = ("# stated limit: source and target share reads (one library), so the follow score is an "
         "upper bound on the correction of an unseen library.")
TABLE_COLUMNS = ("model", "space", "covariate", "direction", "scope", "follow_score",
                 "seed_range", "per_seed", "halves_floor", "tracks_following", "tracks_excluded",
                 "total_ratio", "follows")
TRACK_COLUMNS = ("model", "seed", "space", "covariate", "direction", "scope", "track", "assay",
                 "source", "target", "D_source_cov", "D_target_cov", "D_halves", "follow_score",
                 "excluded", "total_ratio")


# ------------------------------------------------------------------------------------ helpers
def read_tsv(path) -> list:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader((ln for ln in fh if not ln.startswith("#")), delimiter="\t"))


def write_tsv(path, cols, rows, header: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        if header:
            fh.write(header + "\n")
        w = csv.DictWriter(fh, fieldnames=list(cols), delimiter="\t", lineterminator="\n",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in cols})


def _fmt(v) -> str:
    if isinstance(v, float):
        return "nan" if not math.isfinite(v) else f"{v:.6g}"
    return str(v)


def parse_shard(spec: str) -> tuple:
    try:
        i, n = (int(x) for x in str(spec).split("/"))
    except ValueError:
        raise SystemExit(f"--shard {spec!r}: want i/N")
    if not (n >= 1 and 0 <= i < n):
        raise SystemExit(f"--shard {spec!r}: need 0 <= i < N")
    return i, n


def pair_key(p: dict) -> str:
    return f"{p['track']}__{p['covariate']}__{p['direction']}"


def save_npz(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _encoder(tables):
    from minicandi.covariates import CovariateEncoder, load_covariates_json
    p = Path(tables) / "covariates.json"
    return CovariateEncoder(load_covariates_json(p) if p.is_file() else {})


# -------------------------------------------------------------------------------------- pairs
def fields_changed(enc, a: dict, b: dict, covariate: str) -> list:
    """The candidate fields of `covariate` whose encoded C' values differ between rows a and b."""
    va, vb = enc.c_out(a, HALF), enc.c_out(b, HALF)
    return [f for f in COVARIATES[covariate][1] if float(va[COL[f]]) != float(vb[COL[f]])]


def covariate_of(row: dict):
    knob, value = row.get("arm_knob", ""), row.get("arm_value", "")
    for cov, ((k, v), _) in COVARIATES.items():
        if knob == k and (v is None or value == v):
            return cov
    return None


def build_pairs(rows: list, enc) -> list:
    by_id = {r["measurement_id"]: r for r in rows}
    out = []
    for r in rows:
        if r.get("layer") != "testbed":
            continue
        cov = covariate_of(r)
        if cov is None:
            continue
        base = by_id.get(r.get("parent_measurement_id", ""))
        if base is None or base.get("layer") != "testbed" or base.get("arm_knob", ""):
            raise ValueError(f"{r['measurement_id']}: parent {r.get('parent_measurement_id')!r} "
                             f"is not a testbed base row")
        if base["assay"] != r["assay"] or base["cell"] != r["cell"]:
            raise ValueError(f"{r['measurement_id']}: assay or cell differs from its base")
        fc = ",".join(fields_changed(enc, base, r, cov))
        for d in DIRECTIONS:
            src, tgt = (base, r) if d == "base_to_variant" else (r, base)
            out.append({"track": r.get("experiment") or base["measurement_id"].split(".")[0],
                        "assay": r["assay"], "covariate": cov, "direction": d,
                        "source": src["measurement_id"], "target": tgt["measurement_id"],
                        "fields_changed": fc})
    out.sort(key=lambda p: (p["track"], p["covariate"], p["direction"]))
    return out


def cmd_pairs(a) -> int:
    from minicandi.dataset import read_tsv as mread
    tables = Path(a.tables)
    rows = mread(tables / "measurements.tsv")
    pairs = build_pairs(rows, _encoder(tables))
    write_tsv(a.out, PAIR_COLUMNS, pairs)
    none = [pair_key(p) for p in pairs if not p["fields_changed"]]
    print(f"{len(pairs)} directed pairs -> {a.out}")
    if none:
        print(f"warning: no field differs for {none}", file=sys.stderr)
    return 0


# ------------------------------------------------------------------------------------ predict
def source_c_out(enc, src_row: dict, tgt_row: dict, fields) -> np.ndarray:
    """The target's C' row at half scale with `fields` set to the source's values."""
    v = np.asarray(enc.c_out(tgt_row, HALF), np.float32).copy()
    s = enc.c_out(src_row, HALF)
    for f in fields:
        v[COL[f]] = s[COL[f]]
    return v


def other_inputs(rows: list, cell: str, assay: str) -> list:
    """The cell's training base views of every other assay (sorted ids), as an imputation job."""
    from onion_eval.evalset import base_views
    pool = base_views(rows, ("train",))
    return sorted(r["measurement_id"] for (c, a), r in pool.items() if c == cell and a != assay)


class _Capture:
    """Wraps a ReadStartAdapter's inner model; keeps the last `mu_sub`, `n_sub` [B, L, 2] of
    track `t`."""

    @staticmethod
    def install(adapter):
        import torch

        class Cap(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner, self.t, self.last = inner, None, None

            def forward(self, b):
                o = self.inner(b)
                if self.t is not None:
                    self.last = (o["mu_sub"][:, :, self.t].detach().cpu().double().numpy(),
                                 o["n_sub"][:, :, self.t].detach().cpu().double().numpy())
                return o

        if not isinstance(adapter.inner, Cap):
            adapter.inner = Cap(adapter.inner)
        return adapter.inner


def is_read_start(model) -> bool:
    from minicandi.rs_adapter import ReadStartAdapter
    return isinstance(model, ReadStartAdapter)


def predict_pair(model, ds, rsds, rows_by_id, enc, pair, regions, rb, L, infer_batch,
                 inputs, device="cpu") -> dict:
    """The arrays of one directed pair (see the module docstring)."""
    import torch
    from minicandi import contract as C
    from onion_eval.score import _half_shares
    src_row, tgt_row = rows_by_id[pair["source"]], rows_by_id[pair["target"]]
    fields = [f for f in pair["fields_changed"].split(",") if f]
    got = fields_changed(enc, src_row, tgt_row, pair["covariate"])
    if sorted(got) != sorted(fields):
        raise ValueError(f"{pair_key(pair)}: fields_changed {fields} but the rows differ in {got}")
    assay = tgt_row["assay"]
    t = C.TRACK_INDEX[assay]
    v_tgt = np.asarray(enc.c_out(tgt_row, HALF), np.float32)
    v_src = source_c_out(enc, src_row, tgt_row, fields)
    rs = rsds is not None
    cap = _Capture.install(model) if rs else None
    if cap is not None:
        cap.t = t
    spaces = ["overlap_count"] + (["start_sum"] if rs else [])
    acc = {k: [] for s in spaces for k in (f"mu_tgt_{s}", f"n_tgt_{s}", f"mu_src_{s}",
                                            f"n_src_{s}", f"y_{s}", f"y1_{s}", f"x_{s}")}
    stranded = rs and not bool(model._un[t])
    win_inputs = [pair["source"]] + list(inputs)
    for (c, s0) in regions:
        tiles = [(c, s0 + k * L) for k in range(rb // L)]
        for b0 in range(0, len(tiles), infer_batch):
            tl = tiles[b0:b0 + infer_batch]
            batch = C.collate([ds.window(cc, ss, win_inputs, [pair["target"]],
                                         halves_tracks=[assay]) for cc, ss in tl])
            if not np.array_equal(batch["c_out"][:, t].numpy(),
                                  np.broadcast_to(v_tgt, batch["c_out"][:, t].shape)):
                raise ValueError(f"{pair_key(pair)}: the window's C' is not the target's")
            b_src = dict(batch)
            co = batch["c_out"].clone()
            co[:, t, :] = torch.as_tensor(v_src, dtype=co.dtype)
            b_src["c_out"] = co
            for tag, b in (("tgt", batch), ("src", b_src)):
                b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
                with torch.no_grad():
                    o = model(b)
                acc[f"mu_{tag}_overlap_count"].append(o["mu"][:, :, t].detach().cpu().double()
                                                      .numpy().reshape(-1))
                acc[f"n_{tag}_overlap_count"].append(o["n"][:, :, t].detach().cpu().double()
                                                     .numpy().reshape(-1))
                if rs:
                    from t396.score_spaces import pool_strands
                    m, n = pool_strands(cap.last[0], cap.last[1], stranded)
                    acc[f"mu_{tag}_start_sum"].append(m.reshape(-1))
                    acc[f"n_{tag}_start_sum"].append(n.reshape(-1))
            acc["y_overlap_count"].append(batch["y_counts"][:, :, t].numpy().reshape(-1))
            acc["x_overlap_count"].append(batch["x_counts"][:, :, t].numpy().reshape(-1))
            acc["y1_overlap_count"].append(np.concatenate(
                [ds.read(pair["target"], cc, ss, L, "pr1") for cc, ss in tl]))
            if rs:
                for key, mid, which in (("y", pair["target"], "pr2"), ("y1", pair["target"], "pr1"),
                                        ("x", pair["source"], "pr1")):
                    acc[f"{key}_start_sum"].append(np.concatenate(
                        [rsds.read_strands(mid, cc, ss, L, which).astype(np.int64).sum(-1)
                         for cc, ss in tl]))
    out = {k: np.concatenate(v).astype(np.float32 if k.startswith(("mu_", "n_")) else np.int32)
           for k, v in acc.items()}
    out["shares_overlap_count"] = np.asarray(_half_shares(tgt_row), np.float64)
    if rs:
        from t396.dump_native import _shares_starts
        out["shares_start_sum"] = np.asarray(_shares_starts(rsds, pair["target"]), np.float64)
    out.update({k: np.asarray(pair[k]) for k in PAIR_COLUMNS})
    out["spaces"] = np.asarray(spaces)
    return out


def predict_all(model, config: dict, tables, evalset: dict, evalset_path, pairs: list, out,
                shard=(0, 1), infer_batch=None, meta: dict | None = None, log=print,
                device="cpu") -> int:
    """Predict every pair of the shard into `out`; returns the number of npz written."""
    from minicandi.dataset import OnionDataset
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rb = int(evalset.get("region_bins", 2 ** 14))
    cfg = dict(config)
    L = int(cfg.get("context_bins") or min(1024, rb))
    if rb % L:
        raise ValueError(f"region_bins {rb} is not a multiple of context_bins {L}")
    cfg.update({"context_bins": L, "evalset": str(evalset_path), "region_bins": rb,
                "allow_blind": True})
    cfg.setdefault("batch_size", 8)
    ds = OnionDataset(tables, "blind", cfg, seed=0)
    rsds = None
    if is_read_start(model):
        from minicandi.rs_adapter import _STRIP
        from minicandi.rs_dataset import ReadStartDataset
        rcfg = {k: v for k, v in model.config.items() if k not in _STRIP}
        rcfg.update({"context_bins": L, "allow_blind": True})
        rsds = ReadStartDataset(tables, "blind", rcfg)
        model._ds = rsds            # the adapter reads blind (testbed) rows in mode blind
    regions = [(c, int(s)) for c, s in evalset["regions"]]
    rows = list(ds.rows.values())
    B = int(infer_batch or cfg["batch_size"])
    i, n = shard
    mine = [p for k, p in enumerate(pairs) if k % n == i]
    m = dict(meta or {})
    m.update({"pairs": [pair_key(p) for p in pairs], "spaces": ["overlap_count"]
              + (["start_sum"] if rsds is not None else []), "evalset": str(evalset_path),
              "n_regions": len(regions), "region_bins": rb, "infer_batch": B})
    tmp = out / f".meta.json.{os.getpid()}"
    tmp.write_text(json.dumps(m, indent=1), encoding="utf-8")
    os.replace(tmp, out / "meta.json")
    wrote = 0
    for p in mine:
        path = out / f"{pair_key(p)}.npz"
        if path.is_file():
            log(f"skip {path.name} (exists)")
            continue
        inputs = other_inputs(rows, ds.rows[p["target"]]["cell"], ds.rows[p["target"]]["assay"])
        arr = predict_pair(model, ds, rsds, ds.rows, ds.encoder, p, regions, rb, L, B, inputs,
                           device)
        arr["region_indices"] = np.arange(len(regions), dtype=np.int64)
        arr["inputs"] = np.asarray(inputs)
        save_npz(path, arr)
        wrote += 1
        log(f"wrote {path.name}")
    return wrote


def cmd_predict(a) -> int:
    from minicandi.dataset import tables_sha256
    from minicandi.registry import load_run
    from onion_eval import score as S
    from onion_eval.evalset import tables_dir_of
    S.set_threads()
    S.tf32_off()
    ev_path = Path(a.evalset)
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    tables = tables_dir_of(ev, ev_path)
    run = Path(a.run)
    want, have = tables_sha256(tables), S.run_tables_sha256(run)
    if want != have:
        raise SystemExit(f"refused: the run was trained on tables {have[:12]}, the eval set's "
                         f"tables are {want[:12]}")
    model, config = load_run(run, map_location=a.device)
    seed = run.name[4:] if run.name.startswith("seed") else run.name
    meta = {"run": str(run), "model": a.model_name or run.parent.name, "seed": seed,
            "kind": "read_start" if config.get("read_start") else "count"}
    pairs = read_tsv(a.pairs)
    predict_all(model, config, tables, ev, ev_path, pairs, a.out, parse_shard(a.shard),
                a.infer_batch, meta, device=a.device)
    return 0


# -------------------------------------------------------------------------------------- table
def track_terms(z, space: str, scope: str) -> dict:
    """D_source_cov, D_target_cov, D_halves and the mean sums of one npz in one space and scope."""
    from onion_eval import deviance as DV
    y = z[f"y_{space}"].astype(np.float64)
    y1 = z[f"y1_{space}"].astype(np.float64)
    f1, f2 = (float(v) for v in z[f"shares_{space}"])
    k = f2 / HALF
    mu_t = z[f"mu_tgt_{space}"].astype(np.float64) * k
    mu_s = z[f"mu_src_{space}"].astype(np.float64) * k
    if scope == "top1pct":
        x = z[f"x_{space}"].astype(np.float64)
        m = max(1, int(math.ceil(TOP_FRAC * x.size)))
        idx = np.argsort(-x, kind="stable")[:m]
        y, y1, mu_t, mu_s = y[idx], y1[idx], mu_t[idx], mu_s[idx]
    return {"D_source_cov": DV.deviance(y, mu_s), "D_target_cov": DV.deviance(y, mu_t),
            "D_halves": DV.deviance(y, np.maximum(y1 * (f2 / f1), DV.MU_MIN)),
            "sum_tgt": float(mu_t.sum()), "sum_src": float(mu_s.sum())}


def follow_of(t: dict):
    """(score, excluded): the per-track follow score, or (nan, True) when the denominator <= 0."""
    den = t["D_source_cov"] - t["D_halves"]
    if not den > 0:
        return float("nan"), True
    return (t["D_source_cov"] - t["D_target_cov"]) / den, False


def pooled(recs: list) -> dict:
    inc = [r for r in recs if not r["excluded"]]
    num = sum(r["D_source_cov"] - r["D_target_cov"] for r in inc)
    den = sum(r["D_source_cov"] - r["D_halves"] for r in inc)
    ds = sum(r["D_source_cov"] for r in inc)
    st, ss = sum(r["sum_tgt"] for r in recs), sum(r["sum_src"] for r in recs)
    return {"score": num / den if den > 0 else float("nan"),
            "floor": sum(r["D_halves"] for r in inc) / ds if ds > 0 else float("nan"),
            "excluded": sum(1 for r in recs if r["excluded"]),
            "total_ratio": st / ss if ss > 0 else float("nan")}


def load_pred_dir(d) -> tuple:
    """(meta, {pair key: npz path}); SystemExit when a listed pair has no npz."""
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    files = {k: d / f"{k}.npz" for k in meta["pairs"]}
    missing = [k for k, p in files.items() if not p.is_file()]
    if missing:
        raise SystemExit(f"{d}: {len(missing)} of {len(files)} pairs have no npz: {missing[:4]}")
    return meta, files


#: cf_follow space -> (B1 table `space`, the cap law B1 reports in that space)
CAPS_SPACE = {"overlap_count": ("count", "count_saturation"),
              "start_sum": ("start", "subsampled_dm_distinct")}
CAPS_SHAPE, CAPS_GROUP, CAPS_COLUMN = "loto", "all", "gap_closed_vs_depth_scaling"


def _caps_lookup(paths) -> dict:
    """{(covariate, scope, cf_follow space): gap closed} from B1 testbed `table.tsv` files
    (`dedup_cap.TB_TABLE_COLUMNS`): the rows with shape `loto`, assay_group `all` and the cap law
    of the space. ValueError when a key has more than one such row."""
    out: dict = {}
    inv = {v[0]: (k, v[1]) for k, v in CAPS_SPACE.items()}
    for path in ([paths] if isinstance(paths, (str, Path)) else paths):
        rows = read_tsv(path)
        need = {"covariate", "shape", "assay_group", "law", "space", "scope", CAPS_COLUMN}
        if rows and not need <= set(rows[0]):
            raise ValueError(f"{path}: not a B1 testbed table; missing {sorted(need - set(rows[0]))}")
        for r in rows:
            if r["space"] not in inv or r["shape"] != CAPS_SHAPE or r["assay_group"] != CAPS_GROUP:
                continue
            space, law = inv[r["space"]]
            if r["law"] != law:
                continue
            key = (r["covariate"], r["scope"], space)
            if key in out:
                raise ValueError(f"caps table: more than one row for {key}")
            out[key] = float(r[CAPS_COLUMN])
    return out


def _mean_finite(v) -> float:
    v = [x for x in v if math.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def build_table(pred_dirs, caps_table=None) -> tuple:
    per_track = []
    for d in pred_dirs:
        meta, files = load_pred_dir(d)
        for key, path in sorted(files.items()):
            with np.load(path, allow_pickle=False) as z:
                info = {k: str(z[k]) for k in PAIR_COLUMNS}
                for space in [str(s) for s in z["spaces"]]:
                    for scope in SCOPES:
                        t = track_terms(z, space, scope)
                        f, ex = follow_of(t)
                        per_track.append(dict(info, **t, model=meta["model"],
                                              seed=str(meta["seed"]), space=space, scope=scope,
                                              follow_score=f, excluded=int(ex),
                                              total_ratio=(t["sum_tgt"] / t["sum_src"]
                                                           if t["sum_src"] > 0 else float("nan"))))
    caps = _caps_lookup(caps_table) if caps_table else None
    groups: dict = {}
    for r in per_track:
        groups.setdefault((r["model"], r["space"], r["covariate"], r["direction"], r["scope"]),
                          {}).setdefault(r["seed"], []).append(r)
    table = []
    for (model, space, cov, d, scope), by_seed in sorted(groups.items()):
        seeds = sorted(by_seed)
        pl = {s: pooled(by_seed[s]) for s in seeds}
        sc = [pl[s]["score"] for s in seeds]
        fin = [v for v in sc if math.isfinite(v)]
        per_tr: dict = {}
        for s in seeds:
            for r in by_seed[s]:
                if not r["excluded"]:
                    per_tr.setdefault(r["track"], []).append(r["follow_score"])
        score = float(np.mean(fin)) if fin else float("nan")
        row = {"model": model, "space": space, "covariate": cov, "direction": d, "scope": scope,
               "follow_score": score,
               "seed_range": float(max(fin) - min(fin)) if fin else float("nan"),
               "per_seed": ";".join(f"{s}:{_fmt(v)}" for s, v in zip(seeds, sc)),
               "halves_floor": _mean_finite([pl[s]["floor"] for s in seeds]),
               "tracks_following": sum(1 for v in per_tr.values() if np.mean(v) >= GATE_H),
               "tracks_excluded": max(pl[s]["excluded"] for s in seeds),
               "total_ratio": float(np.mean([pl[s]["total_ratio"] for s in seeds])),
               "follows": ("" if scope != "top1pct" else
                           "yes" if math.isfinite(score) and score >= GATE_H else "no")}
        if caps is not None:          # B1 scores variant -> base only
            v = float("nan")
            if d == "variant_to_base":
                if (cov, scope, space) not in caps:
                    raise ValueError(f"caps table: no row for {(cov, scope, space)}")
                v = caps[(cov, scope, space)]
            row["caps_gap_closed"] = v
        table.append(row)
    return table, per_track


def cmd_table(a) -> int:
    table, per_track = build_table(a.pred, a.caps_table)
    out = Path(a.out)
    cols = TABLE_COLUMNS + (("caps_gap_closed",) if a.caps_table else ())
    write_tsv(out / "table.tsv", cols, table, header=LIMIT)
    write_tsv(out / "per_track.tsv", TRACK_COLUMNS, per_track, header=LIMIT)
    print(f"{len(table)} rows -> {out / 'table.tsv'}; {len(per_track)} -> per_track.tsv")
    return 0


# --------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--tables", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--run", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--evalset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--model-name", default=None,
                   help="the model label in meta.json (default: the run dir's parent name)")
    p.add_argument("--infer-batch", type=int, default=None)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p = sub.add_parser("table")
    p.add_argument("--pred", nargs="+", required=True)
    p.add_argument("--caps-table", nargs="+", default=None,
                   help="B1 testbed table.tsv files (dedup_cap, sepe_cap)")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"pairs": cmd_pairs, "predict": cmd_predict, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
