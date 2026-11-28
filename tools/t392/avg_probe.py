"""t392 chunk C13fix: time the two leave-one-cell-out averages of the read-start scoring path.

    python tools/t392/avg_probe.py --run <run dir> --evalset <json> --local <dir> --out <json>
        [--shard 2/4] [--windows 64]

Per average (`OnionDataset.average` from `average_h5`, used by the scorer's window building, and
`ReadStartDataset.average_strands` from `read_start_average_h5`, used by the adapter), on the
windows of the shard's first item request:
  1. set A (first `--windows` windows of the request's first region), file in place, cold;
  2. set A again (warm);
  3. a cProfile of set A (warm) — where the time goes inside the call;
  4. the file copied to `--local`; set B (the same number of windows of the request's sixth
     region, never read before) cold from the copy, then set A from the copy;
the arrays of set A from the copy are compared with `np.array_equal` to the in-place arrays.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
for p in (_TOOLS, _TOOLS / "onion_eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import predict_store as P  # noqa: E402
import score as S  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    for k in ("--run", "--evalset", "--local", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--shard", default="2/4")
    ap.add_argument("--windows", type=int, default=64)
    a = ap.parse_args(argv)
    a.local = os.path.expandvars(a.local)
    i, N = (int(x) for x in a.shard.split("/"))
    S.set_threads()
    from minicandi.registry import load_run
    model, config = load_run(a.run)
    sc = S.Scorer(a.evalset, main="model", model=model, config=config, device="cpu",
                  run_tables_sha256=S.run_tables_sha256(a.run), log=print,
                  **({"same_view": True} if S.same_view_from_env() else {}))
    reqs = [r for j, r in enumerate(P.requests(sc)) if j % N == i]
    item = next(r for r in reqs if r["key"].startswith("item/"))
    cell = sc.ds.rows[item["job"]["target"]]["cell"]
    tiles = list(sc._tiles([sc.regions[k] for k in item["region_indices"]]))
    W = a.windows
    set_a = [(c, s) for c, s in tiles[0][1][:W]]
    set_b = [(c, s) for c, s in tiles[5][1][:W]]
    L = sc.L
    ov, rs = sc.ds, sc.model.ds
    calls = {"overlap_average": lambda c, s: ov.average(cell, c, s, L),
             "read_start_average_strands": lambda c, s: rs.average_strands(cell, c, s, L)}
    out = {"run": a.run, "item": item["key"], "cell": cell, "windows": W, "L": L, "results": {}}
    Path(a.local).mkdir(parents=True, exist_ok=True)

    def timed(fn, ws):
        t = time.perf_counter()
        arrs = [fn(c, s) for c, s in ws]
        return time.perf_counter() - t, arrs

    for name, fn in calls.items():
        res = {}
        res["A_cold_seconds"], ref = timed(fn, set_a)
        res["A_warm_seconds"], _ = timed(fn, set_a)
        pr = cProfile.Profile()
        pr.enable()
        fn_t, _ = timed(fn, set_a)
        pr.disable()
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(14)
        res["A_profiled_seconds"] = fn_t
        res["profile_tottime_top"] = buf.getvalue()
        src = Path(ov.config["average_h5"] if name == "overlap_average" else rs.rs_average_h5)
        dst = Path(a.local) / src.name
        t = time.perf_counter()
        if not dst.is_file():
            shutil.copyfile(src, dst)
        res["copy_seconds"], res["bytes"] = time.perf_counter() - t, dst.stat().st_size
        if name == "overlap_average":
            ov._close_all()
            ov.config["average_h5"] = str(dst)
        else:
            rs._close_all()
            rs.rs_average_h5 = dst
        res["local_B_cold_seconds"], _ = timed(fn, set_b)
        res["local_A_seconds"], loc = timed(fn, set_a)
        res["local_A_identical"] = all(np.array_equal(x, y) for x, y in zip(ref, loc))
        res = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}
        out["results"][name] = res
        print(name, {k: v for k, v in res.items() if k != "profile_tottime_top"}, flush=True)
        print(res["profile_tottime_top"], flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
