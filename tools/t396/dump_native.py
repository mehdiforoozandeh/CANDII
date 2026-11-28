"""t396 C6, iteration 2 A4: dump the arrays of the three-space scoring (truth, and the native
start output of any start-output arm).

    python tools/t396/dump_native.py truth --evalset <json> --tables <dir> --rs-config <rs run dir>
        --out <dir> [--shard i/N]
    python tools/t396/dump_native.py model --run <rs run dir> --evalset <json> --pred-store <dir>
        --out <dir> [--regions pair|all] [--shard i/N]

Targets: the `leave_one_assay_out` imputation targets of the validation split, in evaluation-set
order (`Scorer.jobs`); target j belongs to shard i when j mod N == i. Regions: the evaluation
set's regions in its order (R = 256, rb = region_bins). An existing output file is skipped, so a
rerun writes only what is missing. Every file is written to a temporary name and renamed.

truth -> `<out>/<target id>.npz`:
  mid, assay, cell (str); stranded (bool); read_length (int, -1 when not a number);
  chrom [R] (str), start_bin [R] (int64); depth_reads (float);
  shares (f1, f2) = `score._half_shares` of the row (overlap halves);
  shares_starts (f1, f2) from `n_reads_strand` of `starts_pr1.h5` and `starts_pr2.h5`;
  kernel float64 [2, 17], n_reads_strand [2] (attributes of `starts.h5`);
  y, y_pr1, y_pr2 int32 [R, rb]: overlap counts of `counts.h5`, `pr1.h5`, `pr2.h5` (what the
    scorer reads, `OnionDataset.read`);
  s, s_pr1, s_pr2 uint16 [R, rb, 2]: starts of `starts.h5`, `starts_pr1.h5`, `starts_pr2.h5`
    (`ReadStartDataset.read_strands`; column 0 plus strand, column 1 minus strand);
  pa_counts float32 [R, rb]: the scorer's position-average mean of the target (`Scorer._pa`,
    `PositionAverage.predict`, depth from the target window's `c_out`);
  pa_starts float32 [R, rb]: `ReadStartDataset.average_strands` of the target track, strand sum,
    times depth_reads / 1e6, floored at `baselines.PRED_FLOOR`;
  pa_counts_ids, pa_starts_ids (str): the training measurements each average used.
  Exit 1, before the target's file is written, when the two id lists differ.

model -> `<out>/<target id>.npz` (R = the 256 regions, or the pair regions under `--regions pair`):
  mu_sub, n_sub float32 [R, rb, 2]: the inner model's output before conversion, captured inside
    `ReadStartAdapter.forward` while `predict_store.request_arrays` rebuilds the item's windows
    and batches exactly as the prediction store was made (its `infer_batch`). An unstranded track
    holds its total in channel 0; its channel 1 (never trained) is set to mu 0, n N_MAX;
  mu, n float32 [R, rb]: the prediction store's arrays, key `item/<target>/leave_one_assay_out`;
  max_abs_rel_diff (float): gate (a), max |fresh - stored| / |stored| of `mu`, where the fresh mu
    is `read_start.convert_nb` of the dumped mu_sub over the runs the adapter converts
    (`read_start.stitch` within each forward batch; the adapter's span kernel);
  max_abs_rel_diff_region (float): the same against one conversion per whole region (open edges
    at the region ends only); the batch edges inside a region make it larger than gate (a);
  region_indices int64 [R]: positions in the evaluation set's region list.
  Gate (a) is enforced: when max_abs_rel_diff > GATE_A (1e-4), that target's file is NOT written,
  the value is printed to stderr, and the command exits 3 without dumping further targets.

`model` accepts any run whose config sets `read_start` (`registry.load_run` wraps it in
`rs_adapter.ReadStartAdapter`): `start_in_start_out`, `start_in_count_out` (the kernel head; the
adapter's span-kernel conversion is its training conversion) and `count_in_start_out` (count input;
the adapter feeds it counts, the output conversion is unchanged). Every arm is converted per
forward batch, as the adapter does. Any other run exits 2.

A count-output arm (`count_in_count_out`, the read-count reference) needs no dump: its `mu`, `n`
are read from its prediction store.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

REGIME = "leave_one_assay_out"
GATE_A = 1e-4                # gate (a): max_abs_rel_diff of the fresh conversion


def _quiet(*_a, **_k):
    return None


def parse_shard(spec: str) -> tuple:
    try:
        i, n = (int(x) for x in str(spec).split("/"))
    except ValueError:
        raise SystemExit(f"--shard {spec!r}: want i/N")
    if not (n >= 1 and 0 <= i < n):
        raise SystemExit(f"--shard {spec!r}: need 0 <= i < N")
    return i, n


def target_jobs(sc) -> list:
    """The leave-one-assay-out imputation jobs of the scorer's split, in evaluation-set order."""
    imp = sc.jobs()[0]
    return [j for j in imp if j["regime"] == REGIME]


def mine(jobs: list, shard: tuple) -> list:
    i, n = shard
    return [j for k, j in enumerate(jobs) if k % n == i]


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


def rel_diff(a, b) -> float:
    """max |a - b| / max(|b|, 1e-30) (as `predict_store._rel`)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30))) if a.size else 0.0


def run_config(run_dir) -> tuple:
    """(config, tables) of a run directory, from its `run.json`."""
    d = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))
    return dict(d["config"]), d.get("tables")


# --------------------------------------------------------------------------------------- truth

def pa_ids(sc, rsds, row) -> tuple:
    """(ids of the scorer's position average, ids of `average_strands`) for the target `row`."""
    import h5py
    from minicandi.rs_dataset import _ids
    assay, cell = row["assay"], row["cell"]
    counts = sorted(m for (c, a), m in sc.ds.train_base.items() if a == assay and c != cell)

    def has_depth(m):
        return float(rsds.rows[m]["depth_reads"] or 0) > 0

    if rsds.rs_average_h5 is not None:
        with h5py.File(rsds.rs_average_h5, "r") as f:
            summed = _ids(f[assay].attrs["measurement_ids"]) if assay in f else []
        own = [m for m in summed if rsds.rows[m]["cell"] == cell]
        drop = own[0] if own and has_depth(own[0]) else None      # as average_strands subtracts
        starts = sorted(m for m in summed if m != drop)
    else:
        starts = sorted(m for (c, a), m in rsds.train_base.items()
                        if a == assay and c != cell and has_depth(m))
    return counts, starts


def _shares_starts(rsds, mid) -> tuple:
    import h5py
    tot = []
    for which in ("pr1", "pr2"):
        with h5py.File(rsds.start_path(mid, which), "r") as f:
            tot.append(float(np.asarray(f.attrs["n_reads_strand"], np.float64).sum()))
    return (tot[0] / (tot[0] + tot[1]), tot[1] / (tot[0] + tot[1]))


def truth_arrays(sc, rsds, job) -> dict:
    """The truth file of one target (module docstring); raises IdMismatch when the averages'
    id lists differ."""
    import h5py
    from minicandi import contract as C
    from minicandi.read_start import UNSTRANDED_TRACKS
    from onion_eval import score as S
    from onion_eval.baselines import PRED_FLOOR
    mid = job["target"]
    row = sc._row(mid)
    assay, cell = row["assay"], row["cell"]
    ids_c, ids_s = pa_ids(sc, rsds, row)
    if ids_c != ids_s:
        raise IdMismatch(f"{mid}: position-average ids differ: counts only "
                         f"{sorted(set(ids_c) - set(ids_s))}, starts only "
                         f"{sorted(set(ids_s) - set(ids_c))}")
    t = C.TRACK_INDEX[assay]
    rb = sc.rb
    regions = sc.regions
    depth = float(row["depth_reads"])
    try:
        rl = int(float(row.get("read_length") or "nan"))
    except ValueError:
        rl = -1
    with h5py.File(rsds.start_path(mid, "counts"), "r") as f:
        kernel = np.asarray(f.attrs["kernel"], np.float64)
        nrs = np.asarray(f.attrs["n_reads_strand"], np.float64)
    out = {"mid": np.array(mid), "assay": np.array(assay), "cell": np.array(cell),
           "stranded": np.array(assay not in UNSTRANDED_TRACKS),
           "read_length": np.array(rl, np.int64),
           "chrom": np.array([c for c, _ in regions]),
           "start_bin": np.array([s for _, s in regions], np.int64),
           "depth_reads": np.array(depth, np.float64),
           "shares": np.array(S._half_shares(row), np.float64),
           "shares_starts": np.array(_shares_starts(rsds, mid), np.float64),
           "kernel": kernel, "n_reads_strand": nrs,
           "pa_counts_ids": np.array(ids_c, dtype=str), "pa_starts_ids": np.array(ids_s, dtype=str)}
    for key, which in (("y", "counts"), ("y_pr1", "pr1"), ("y_pr2", "pr2")):
        out[key] = np.stack([sc.ds.read(mid, c, s, rb, which) for c, s in regions]).astype(np.int32)
    for key, which in (("s", "counts"), ("s_pr1", "pr1"), ("s_pr2", "pr2")):
        out[key] = np.stack([rsds.read_strands(mid, c, s, rb, which)
                             for c, s in regions]).astype(np.uint16)
    pa_c = np.zeros((len(regions), rb), np.float32)
    pa_s = np.zeros((len(regions), rb), np.float32)
    c_out = None
    for ri, (c, s) in enumerate(regions):
        if c_out is None:   # the target's C' as the scorer's windows carry it (constant per target)
            win = sc.ds.window(c, s, job["inputs"], [mid])
            c_out = np.asarray(win["c_out"], np.float32)
        pa = sc._pa(assay, ri)
        batch = {"y_counts": np.zeros((1, rb, C.A), np.int32), "c_out": c_out[None],
                 "meta": [{"cell": cell, "chrom": c, "start_bin": s}]}
        pa_c[ri] = np.asarray(pa.predict(batch)["mu"][0, :, t], np.float32)
        a = rsds.average_strands(cell, c, s, rb)[:, t, :].astype(np.float64).sum(-1)
        pa_s[ri] = np.maximum(a * (depth / 1e6), PRED_FLOOR).astype(np.float32)
    out["pa_counts"], out["pa_starts"] = pa_c, pa_s
    return out


class IdMismatch(Exception):
    pass


def cmd_truth(a) -> int:
    from minicandi.dataset import tables_sha256
    from minicandi.rs_dataset import ReadStartDataset
    from onion_eval import score as S
    cfg, _ = run_config(a.rs_config)
    sc = S.Scorer(a.evalset, main="position_average", config=cfg, n_boot=0, log=_quiet)
    if tables_sha256(a.tables) != sc.tables_sha256:
        print(f"--tables {a.tables} are not the evaluation set's tables {sc.tables}",
              file=sys.stderr)
        return 2
    rcfg = {k: v for k, v in cfg.items() if k not in ("two_step", "control_head", "evalset")}
    rsds = ReadStartDataset(a.tables, "val", rcfg)
    out = Path(a.out)
    jobs = mine(target_jobs(sc), parse_shard(a.shard))
    for j in jobs:
        p = out / f"{j['target']}.npz"
        if p.is_file():
            print(f"truth: {p.name} exists, skipped", flush=True)
            continue
        try:
            arr = truth_arrays(sc, rsds, j)
        except IdMismatch as e:
            print(f"truth: {e}", file=sys.stderr)
            return 1
        save_npz(p, arr)
        print(f"truth: wrote {p}", flush=True)
    return 0


# --------------------------------------------------------------------------------------- model

def _tap_class():
    import torch

    class Tap(torch.nn.Module):
        """Wraps the adapter's inner model; keeps, per forward, the batch meta and the target
        channel of `mu_sub`, `n_sub` [B, L, 2]."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.t = None
            self.seen = []

        def forward(self, b):
            o = self.inner(b)
            if self.t is not None:
                self.seen.append((b["meta"], o["mu_sub"][:, :, self.t].detach().cpu(),
                                  o["n_sub"][:, :, self.t].detach().cpu()))
            return o

    return Tap


def model_arrays(sc, adapter, tap, store, job, region_idx, B: int) -> dict:
    """The model file of one target (module docstring)."""
    import torch
    from minicandi import contract as C
    from minicandi import read_start as RS
    from onion_eval import predict_store as PS
    mid = job["target"]
    row = sc._row(mid)
    t = C.TRACK_INDEX[row["assay"]]
    stranded = row["assay"] not in RS.UNSTRANDED_TRACKS
    key = f"item/{mid}/{REGIME}"
    req = {"key": key, "kind": job["kind"], "region_indices": list(region_idx),
           "needs_ctl": False, "job": job}
    tap.t, tap.seen = t, []
    try:
        PS.request_arrays(sc, req, B)
        seen = tap.seen
    finally:
        tap.t, tap.seen = None, []
    L, rb, R = sc.L, sc.rb, len(region_idx)
    mu_sub = torch.cat([m for _, m, _ in seen]).reshape(R, rb, RS.STRANDS)
    n_sub = torch.cat([n for _, _, n in seen]).reshape(R, rb, RS.STRANDS)
    # fresh conversion over the runs the adapter converts (stitch within each forward batch)
    fresh = []
    kern = None
    for meta, m, n in seen:
        mu_b = torch.empty(m.shape[:2], dtype=m.dtype)
        for run in RS.stitch(meta, L):
            k = adapter._window_kernels(meta[run[0]])[t]
            kern = k if kern is None else kern
            idx = torch.as_tensor(run)
            mt, _ = RS.convert_nb(m[idx].reshape(-1, RS.STRANDS), n[idx].reshape(-1, RS.STRANDS),
                                  k, stranded=stranded)
            mu_b[idx] = mt.reshape(len(run), L)
        fresh.append(mu_b.reshape(-1))
    fresh = torch.cat(fresh).reshape(R, rb).numpy()
    region = np.stack([RS.convert_nb(mu_sub[r], n_sub[r], kern, stranded=stranded)[0].numpy()
                       for r in range(R)])
    st = store.read(key)
    mu_st = np.asarray(st["mu"], np.float32)[list(region_idx)]
    n_st = np.asarray(st["n"], np.float32)[list(region_idx)]
    ms, ns = mu_sub.numpy().astype(np.float32), n_sub.numpy().astype(np.float32)
    if not stranded:
        ms[..., 1] = 0.0
        ns[..., 1] = float(C.N_MAX)
    return {"mu_sub": ms, "n_sub": ns, "mu": mu_st, "n": n_st,
            "max_abs_rel_diff": np.array(rel_diff(fresh, mu_st), np.float64),
            "max_abs_rel_diff_region": np.array(rel_diff(region, mu_st), np.float64),
            "region_indices": np.array(region_idx, np.int64)}


def cmd_model(a) -> int:
    import torch
    from minicandi.registry import load_run
    from minicandi.rs_adapter import ReadStartAdapter
    from onion_eval import predict_store as PS
    from onion_eval import score as S
    S.set_threads()
    S.tf32_off()
    model, config = load_run(a.run)
    if not config.get("read_start") or not isinstance(model, ReadStartAdapter):
        print(f"{a.run}: not a start-output run (config sets no read_start)", file=sys.stderr)
        return 2
    tap = _tap_class()(model.inner)
    model.inner = tap
    sc = S.Scorer(a.evalset, main="model", model=model, config=config, device="cpu",
                  run_tables_sha256=S.run_tables_sha256(a.run), n_boot=0, log=_quiet)
    store = PS.PredictionStore(a.pred_store)
    info = store.info
    if not info:
        print(f"{a.pred_store}: no {PS.STORE_FILE}", file=sys.stderr)
        return 2
    if info.get("evalset_sha256") != S.sha256_file(a.evalset):
        print(f"{a.pred_store}: made for another evaluation set", file=sys.stderr)
        return 2
    B = int(info["infer_batch"])
    region_idx = (PS.pair_region_indices(sc) if a.regions == "pair"
                  else list(range(len(sc.regions))))
    out = Path(a.out)
    with torch.no_grad():
        for j in mine(target_jobs(sc), parse_shard(a.shard)):
            p = out / f"{j['target']}.npz"
            if p.is_file():
                print(f"model: {p.name} exists, skipped", flush=True)
                continue
            arr = model_arrays(sc, model, tap, store, j, region_idx, B)
            if not float(arr["max_abs_rel_diff"]) <= GATE_A:
                print(f"model: gate (a) failed for {j['target']}: max_abs_rel_diff "
                      f"{float(arr['max_abs_rel_diff']):.3g} > {GATE_A}; nothing written",
                      file=sys.stderr)
                return 3
            save_npz(p, arr)
            print(f"model: wrote {p} max_abs_rel_diff {float(arr['max_abs_rel_diff']):.3g} "
                  f"(region {float(arr['max_abs_rel_diff_region']):.3g})", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("truth")
    t.add_argument("--evalset", required=True)
    t.add_argument("--tables", required=True)
    t.add_argument("--rs-config", required=True, help="run dir of a read-start run (run.json)")
    t.add_argument("--out", required=True)
    t.add_argument("--shard", default="0/1")
    m = sub.add_parser("model")
    m.add_argument("--run", required=True)
    m.add_argument("--evalset", required=True)
    m.add_argument("--pred-store", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--regions", choices=("pair", "all"), default="all")
    m.add_argument("--shard", default="0/1")
    a = ap.parse_args(argv)
    return cmd_truth(a) if a.cmd == "truth" else cmd_model(a)


if __name__ == "__main__":
    sys.exit(main())
