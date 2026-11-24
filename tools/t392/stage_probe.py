"""t392 chunk C13fix: time the read-start scoring path unstaged and staged, and check bit identity.

    python tools/t392/stage_probe.py --run <run dir> --evalset <json> --pred-store <store dir>
        --shard i/N --stage-dir <local dir> --out <json> [--k-items 2] [--k-items-staged 4]
        [--k-pairs 4] [--infer-batch 8]

Builds the scorer as `predict_store.py run` does (same-view on when ONION_SCORE_SAME_VIEW=1),
takes the requests of shard i of N, and stages the overlap-count files of the shard as
`predict_store._stage` does, with the read-start hook off. Then:
  A. the first `--k-items` item requests of the shard, read-start files read in place;
  B. `ReadStartAdapter.stage_read_starts` on the shard's measurements and `train.stage_average`
     of the scorer's `average_h5` (what `_stage` does for a read-start run);
  C. the next `--k-items-staged` item requests and the first `--k-pairs` pair requests, staged.
Per phase it records the wall time and the seconds inside the adapter's input reads
(`_inputs`), the read-start average (`average_strands`), the span kernels, the inner model, and
the overlap-count window building. Each array of each request is compared with `np.array_equal`
against the same request in `--pred-store` (written by the unstaged production jobs) when the
store holds it. Writes nothing into the store.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
for p in (_TOOLS, _TOOLS / "onion_eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import predict_store as P  # noqa: E402
import score as S  # noqa: E402


def _timed(acc, name, fn):
    def wrap(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            acc[name] += time.perf_counter() - t0
    return wrap


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    for k in ("--run", "--evalset", "--pred-store", "--stage-dir", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--shard", default="2/4")
    ap.add_argument("--k-items", type=int, default=2)
    ap.add_argument("--k-items-staged", type=int, default=4)
    ap.add_argument("--k-pairs", type=int, default=4)
    ap.add_argument("--infer-batch", type=int, default=8)
    a = ap.parse_args(argv)
    a.stage_dir = os.path.expandvars(a.stage_dir)
    i, N = (int(x) for x in a.shard.split("/"))
    S.set_threads()
    from minicandi.registry import load_run
    t0 = time.perf_counter()
    model, config = load_run(a.run)
    sc = S.Scorer(a.evalset, main="model", model=model, config=config, device="cpu",
                  run_tables_sha256=S.run_tables_sha256(a.run), log=print,
                  **({"same_view": True} if S.same_view_from_env() else {}))
    reqs = P.requests(sc)
    mine = [r for j, r in enumerate(reqs) if j % N == i]
    items = [r for r in mine if r["key"].startswith("item/")]
    pairs = [r for r in mine if r["key"].startswith("pair/")]
    out = {"run": a.run, "shard": a.shard, "build_seconds": round(time.perf_counter() - t0, 1),
           "n_requests_shard": len(mine), "n_items_shard": len(items),
           "n_pairs_shard": len(pairs), "n_items_needing_control": sum(r["needs_ctl"] for r in items),
           "phases": {}}
    ad = sc.model
    hook = ad.stage_read_starts
    ad.stage_read_starts = None                              # overlap-count staging only
    out["stage_overlap"] = P._stage(sc, mine, Path(a.stage_dir), print)
    del ad.stage_read_starts
    acc = defaultdict(float)
    ds = ad.ds
    ds.average_strands = _timed(acc, "average_strands", ds.average_strands)
    ad._inputs = _timed(acc, "inputs_read_strands", ad._inputs)
    ad._window_kernels = _timed(acc, "window_kernels", ad._window_kernels)
    ad.inner.forward = _timed(acc, "inner_forward", ad.inner.forward)
    sc.ds.window = _timed(acc, "overlap_window", sc.ds.window)

    def phase(name, rs):
        acc.clear()
        tm = defaultdict(float)
        res = {"requests": [], "identical_to_store": {}}
        t = time.perf_counter()
        for r in rs:
            t1 = time.perf_counter()
            arr = P.request_arrays(sc, r, a.infer_batch, tm)
            res["requests"].append({"key": r["key"], "needs_ctl": r["needs_ctl"],
                                    "seconds": round(time.perf_counter() - t1, 2)})
            f = Path(a.pred_store) / P.key_file(r["key"])
            if f.is_file():
                with np.load(f) as z:
                    res["identical_to_store"][r["key"]] = {
                        k: bool(np.array_equal(z[k], v)) for k, v in arr.items()}
            print(f"probe {name}: {r['key']} {res['requests'][-1]['seconds']} s", flush=True)
            out["phases"][name] = res
            Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
        res["wall_seconds"] = round(time.perf_counter() - t, 1)
        res["parts_seconds"] = {k: round(v, 1) for k, v in acc.items()}
        res["timing"] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in tm.items()}
        out["phases"][name] = res
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")

    k, ks = a.k_items, a.k_items_staged
    phase("A_unstaged_items", items[:k])
    mids = set()
    for r in mine:                                           # the set `predict_store._stage` uses
        j = r["job"]
        mids |= {j["target"], *j.get("inputs", []), *([j["source"]] if "source" in j else [])}
    mids |= {(sc.ds.rows[m].get("control_measurement_id") or "").strip() for m in list(mids)
             if m in sc.ds.rows}
    out["stage_read_starts"] = hook(sorted(m for m in mids if m in sc.ds.rows),
                                    Path(a.stage_dir) / "rs392")
    from minicandi.train import stage_average                # as predict_store._stage does
    out["stage_average_h5"] = stage_average(sc.ds, Path(a.stage_dir) / "rs392")
    sc._region, sc._rc = None, {}
    print(f"probe: staged read starts {out['stage_read_starts']}, average_h5 "
          f"{out['stage_average_h5']}", flush=True)
    phase("B_staged_items", items[k:k + ks])
    phase("C_staged_pairs", pairs[:a.k_pairs])
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"probe: wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
