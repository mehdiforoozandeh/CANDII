"""Prediction store of the two-phase scorer (plan t152, Amendment 8, pinned interface L, first item).

The model's forwards of one scoring are made once, on the scoring device, and written to disk; the
metrics phase (chunk SP2) then reads them in CPU workers in place of calling the model. This module
calls the scorer's own window, collate and model-wrapper code (`Scorer.ds.window`,
`Scorer._pair_window`, `contract.collate`, `Scorer.model_pred`, `Scorer._predicted_control`) and
owns no line of `score.py`.

    requests(scorer) -> list[dict]
        one element per job of `Scorer.jobs()`, in the order in which `Scorer.run()` calls
        `_job_arrays`: the imputation groups of each regime (REGIMES, then "control"; assays sorted),
        then the correction pairs, then the replicate pairs. Each element is
        {"key": "item/<target>/<regime>" | "pair/<pair id>", "kind", "region_indices", "needs_ctl",
         "job"}; a pair lists only the evaluation set's `pair_regions` when it has them.
    predict(scorer, requests, out_dir, infer_batch, shard=(i, N), *, run=None, guardrail=True)
        writes `<out_dir>/<sha1 of key>.npz` per request of shard i (request j belongs to shard
        j mod N): `n`, `mu` (float32 [len(region_indices) x region_bins], region order then window
        order, the target channel only), `ctl_mu` when `needs_ctl`, and `key`; `guardrail.json`
        (shard 0 only); and `PREDICTED_<i>of<N>.json` once every request of the shard is stored.
        A rerun skips stored requests. TF32 is off for matmul and cuDNN inside; no mixed precision.
    PredictionStore(out_dir, **expected).read(key) / .complete(requests)
        `complete` is False while a request or a shard is missing, and raises `ScoreRefused` when
        the store was made for another run, evaluation set, tables, request list or settings.

Batches. At an inference batch B no larger than the windows of one region (region_bins /
context_bins, 16 on eic_v1), the windows of each region are cut into batches of B exactly as the
legacy `_job_arrays` cuts them, so at the legacy batch the stored floats are the legacy floats. A
larger B cuts the request's windows end to end (a batch may span regions; never two requests), so
the largest batch a request can fill is its own window count.

`<out_dir>/STORE.json` (written before the first array) holds the identity and settings of the
store; a predict call with another identity or settings refuses to write into it.

Measurement (chunk SP1, results/<task>/FORWARD_SPEED.md):

    python tools/onion_eval/predict_store.py bench --run <run dir> --evalset <json> --out <json>
        --store-root <dir> [--devices cuda,cpu] [--batches 8,16,32,64] [--probe 128,256,512,1024]
        [--regions 32] [--n-items 2] [--n-pairs 4] [--stage-dir $SLURM_TMPDIR/stage]
        [--allow-other-tables]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from onion_eval import score as S  # noqa: E402

STORE_FILE = "STORE.json"
GUARDRAIL_FILE = "guardrail.json"
STORE_FORMAT = 1
IDENTITY_KEYS = ("run", "run_tables_sha256", "evalset_sha256", "tables_sha256")
SETTING_KEYS = ("device", "infer_batch", "tf32", "amp")


# ------------------------------------------------------------------------------------ requests

def key_file(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest() + ".npz"


def pair_region_indices(scorer) -> list:
    """Indices into `scorer.regions` of the evaluation set's `pair_regions` (all regions without
    the key). The key may list region indices or [chrom, start] regions."""
    pr = scorer.ev.get("pair_regions")
    n = len(scorer.regions)
    if pr is None:
        return list(range(n))
    pos = {(c, int(s)): i for i, (c, s) in enumerate(scorer.regions)}
    out = []
    for x in pr:
        if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
            i = int(x)
        else:
            i = pos.get((x[0], int(x[1])), -1)
        if not 0 <= i < n:
            raise ValueError(f"pair_regions entry {x!r} is not a region of the evaluation set")
        out.append(i)
    if len(set(out)) != len(out) or out != sorted(out):
        raise ValueError("pair_regions must be distinct and in region order")
    return out


def requests(scorer) -> list:
    """The forward requests of one scoring, in the scorer's job order (module docstring)."""
    imp, cor, rep, ctl = scorer.jobs()
    allr = list(range(len(scorer.regions)))
    pr = pair_region_indices(scorer)
    out = []
    for regime in S.REGIMES + ("control",):
        groups = defaultdict(list)
        for j in (ctl if regime == "control" else imp):
            if j["regime"] == regime:
                groups[j["assay"]].append(j)
        for a in sorted(groups):
            for j in groups[a]:
                out.append({"key": f"item/{j['target']}/{regime}", "kind": j["kind"],
                            "region_indices": allr,
                            "needs_ctl": bool(scorer.model_pred is not None
                                              and scorer._needs_predicted_control(j)),
                            "job": j})
    for j in cor + rep:
        pid = j.get("pair_id") or j["source"] + ">" + j["target"]
        out.append({"key": f"pair/{pid}", "kind": j["kind"], "region_indices": list(pr),
                    "needs_ctl": False, "job": j})
    keys = [r["key"] for r in out]
    if len(set(keys)) != len(keys):
        dup = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(f"two jobs share a store key: {dup[:3]}")
    return out


def requests_sha256(reqs) -> str:
    """Hash of what a store holds per request: key, regions and the control flag."""
    body = json.dumps([[r["key"], [int(i) for i in r["region_indices"]], bool(r["needs_ctl"])]
                       for r in reqs], separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def identity(scorer, run=None) -> dict:
    return {"run": str(Path(run).resolve()) if run is not None else None,
            "run_tables_sha256": scorer.run_tables_sha256,
            "evalset_sha256": S.sha256_file(scorer.evalset_path),
            "tables_sha256": scorer.tables_sha256}


# ------------------------------------------------------------------------------------ predict

def _batches(scorer, region_indices, B: int) -> list:
    """Lists of (region index, chrom, start) per forward pass (module docstring: Batches)."""
    per_region = [[(ri, c, s) for c, s in tiles]
                  for ri, (_, tiles) in zip(region_indices,
                                            scorer._tiles([scorer.regions[i] for i in region_indices]))]
    if B <= scorer.rb // scorer.L:
        return [tl[b0:b0 + B] for tl in per_region for b0 in range(0, len(tl), B)]
    flat = [w for tl in per_region for w in tl]
    return [flat[b0:b0 + B] for b0 in range(0, len(flat), B)]


def _f32(a: np.ndarray, what: str) -> np.ndarray:
    """float32 copy of a model output; refuses a value that float32 would round (the store must
    give back the legacy float64 arrays exactly)."""
    f = np.ascontiguousarray(a, dtype=np.float32)
    if not np.array_equal(f.astype(np.float64), a, equal_nan=True):
        raise ValueError(f"{what}: the model output is not float32-exact; the store keeps float32")
    return f


def request_arrays(scorer, req, B: int, timing=None) -> dict:
    """{n, mu[, ctl_mu]} of one request: the target channel, region order then window order."""
    job = req["job"]
    tgt = scorer._row(job["target"])
    t = scorer.C.TRACK_INDEX[tgt["assay"]]
    pair = job["kind"] in ("correction", "replicate")
    tm = timing if timing is not None else defaultdict(float)
    n_parts, mu_parts, ctl_parts = [], [], []
    for tl in _batches(scorer, req["region_indices"], int(B)):
        t0 = time.perf_counter()
        wins = []
        for ri, c, s in tl:
            scorer._set_region(*scorer.regions[ri])
            wins.append(scorer._pair_window(job, c, s) if pair
                        else scorer.ds.window(c, s, job["inputs"], [job["target"]]))
        batch = scorer.C.collate(wins)
        t1 = time.perf_counter()
        p = scorer.model_pred(batch)
        t2 = time.perf_counter()
        n_parts.append(p["n"][:, :, t].reshape(-1))
        mu_parts.append(p["mu"][:, :, t].reshape(-1))
        if req["needs_ctl"]:
            ctl_parts.append(scorer._predicted_control(batch, t).reshape(-1))
        t3 = time.perf_counter()
        tm["windows_seconds"] += t1 - t0
        tm["forward_seconds"] += t2 - t1
        tm["control_forward_seconds"] += t3 - t2
        tm["windows"] += len(tl)
        tm["control_windows"] += len(tl) if req["needs_ctl"] else 0
        tm["batches"] += 1
    shape = (len(req["region_indices"]), scorer.rb)
    out = {"n": _f32(np.concatenate(n_parts), "n").reshape(shape),
           "mu": _f32(np.concatenate(mu_parts), "mu").reshape(shape)}
    if req["needs_ctl"]:
        out["ctl_mu"] = _f32(np.concatenate(ctl_parts), "ctl_mu").reshape(shape)
    return out


def _atomic_write(path: Path, write) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        with open(tmp, "wb") as fh:
            write(fh)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _write_json(path: Path, obj) -> None:
    _atomic_write(path, lambda fh: fh.write((json.dumps(obj, indent=1, default=_jsonable)
                                             + "\n").encode("utf-8")))


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return S._j(x) if S._undef(x) else str(x)


def _claim_store(out_dir: Path, head: dict) -> None:
    """Write STORE.json, or refuse when it exists with another identity or settings."""
    p = out_dir / STORE_FILE
    body = (json.dumps(head, indent=1, sort_keys=True) + "\n").encode("utf-8")
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        old = json.loads(p.read_text(encoding="utf-8"))
        diff = sorted(k for k in set(old) | set(head) if old.get(k) != head.get(k))
        if diff:
            raise S.ScoreRefused(f"{out_dir}: a prediction store of another scoring ({', '.join(diff)}"
                                 f" differ); use another --pred-store")
        return
    with os.fdopen(fd, "wb") as fh:
        fh.write(body)


def tf32_flags() -> tuple:
    return (bool(torch.backends.cuda.matmul.allow_tf32), bool(torch.backends.cudnn.allow_tf32))


def predict(scorer, reqs, out_dir, infer_batch, shard=(0, 1), *, run=None, guardrail=True,
            timing=None, log=None) -> dict:
    """Write the predictions of shard `shard` = (i, N) of `reqs` (module docstring). Returns
    {"stored", "skipped", "n_requests", "timing"}."""
    if scorer.model_pred is None:
        raise ValueError("predict needs a scorer with a model")
    i, N = (int(x) for x in shard)
    if not (N >= 1 and 0 <= i < N):
        raise ValueError(f"shard {shard!r}: need 0 <= i < N")
    B = int(infer_batch)
    if B < 1:
        raise ValueError("infer_batch must be at least 1")
    log = log or scorer.log
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ident = identity(scorer, run)
    settings = {"device": str(scorer.device), "infer_batch": B, "tf32": False, "amp": False}
    _claim_store(out_dir, {"format": STORE_FORMAT, **ident, **settings, "n_shards": N,
                           "region_bins": scorer.rb, "n_requests_total": len(reqs),
                           "requests_sha256": requests_sha256(reqs)})
    mine = [r for j, r in enumerate(reqs) if j % N == i]
    tm = timing if timing is not None else defaultdict(float)
    old = tf32_flags()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    stored = skipped = 0
    t_all = time.perf_counter()
    try:
        for k, r in enumerate(mine):
            p = out_dir / key_file(r["key"])
            if p.is_file():
                with np.load(p) as z:
                    if str(z["key"]) != r["key"]:
                        raise S.ScoreRefused(f"{p}: holds {z['key']!s}, not {r['key']}")
                skipped += 1
                continue
            arr = request_arrays(scorer, r, B, tm)
            _atomic_write(p, lambda fh: np.savez(fh, key=np.array(r["key"]), **arr))
            stored += 1
            if (k + 1) % 50 == 0 or k + 1 == len(mine):
                log(f"predict: shard {i}/{N}: {k + 1}/{len(mine)} requests "
                    f"({time.perf_counter() - t_all:.0f} s)")
        if guardrail and i == 0 and not (out_dir / GUARDRAIL_FILE).is_file():
            tg = time.perf_counter()
            _write_json(out_dir / GUARDRAIL_FILE, scorer._copy_leak())
            tm["guardrail_seconds"] += time.perf_counter() - tg
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old
    tm["wall_seconds"] += time.perf_counter() - t_all
    _write_json(out_dir / f"PREDICTED_{i}of{N}.json",
                {**ident, "git_sha": S.git_sha(), **settings, "n_requests": len(mine)})
    return {"stored": stored, "skipped": skipped, "n_requests": len(mine), "timing": dict(tm)}


# ------------------------------------------------------------------------------------ read

class PredictionStore:
    """Read side. Expected values given here (identity keys, `requests_sha256` is derived from the
    requests, and any of `settings`) are checked by `complete`; None means not checked."""

    def __init__(self, out_dir, *, run=None, run_tables_sha256=None, evalset_sha256=None,
                 tables_sha256=None, settings=None):
        self.dir = Path(out_dir)
        self.expected = {"run": str(Path(run).resolve()) if run is not None else None,
                         "run_tables_sha256": run_tables_sha256,
                         "evalset_sha256": evalset_sha256, "tables_sha256": tables_sha256}
        self.settings = dict(settings or {})

    @classmethod
    def for_scorer(cls, out_dir, scorer, run=None, settings=None):
        return cls(out_dir, run=run, settings=settings,
                   **{k: v for k, v in identity(scorer, run).items() if k != "run"})

    @property
    def info(self) -> dict:
        p = self.dir / STORE_FILE
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    def read(self, key: str) -> dict:
        p = self.dir / key_file(key)
        with np.load(p) as z:
            out = {k: z[k] for k in z.files if k != "key"}
            if str(z["key"]) != key:
                raise S.ScoreRefused(f"{p}: holds {z['key']!s}, not {key}")
        return out

    def guardrail(self) -> dict:
        return json.loads((self.dir / GUARDRAIL_FILE).read_text(encoding="utf-8"))

    def complete(self, reqs) -> bool:
        head = self.info
        if not head:
            return False
        bad = [k for k, v in self.expected.items() if v is not None and head.get(k) != v]
        bad += [k for k, v in self.settings.items() if head.get(k) != v]
        if head.get("requests_sha256") != requests_sha256(reqs):
            bad.append("requests_sha256")
        if bad:
            raise S.ScoreRefused(f"{self.dir}: the prediction store was made for another scoring "
                                 f"({', '.join(sorted(bad))} differ)")
        N = int(head["n_shards"])
        for i in range(N):
            p = self.dir / f"PREDICTED_{i}of{N}.json"
            if not p.is_file():
                return False
            d = json.loads(p.read_text(encoding="utf-8"))
            diff = [k for k in IDENTITY_KEYS + SETTING_KEYS if d.get(k) != head.get(k)]
            if diff:
                raise S.ScoreRefused(f"{p}: shard of another scoring ({', '.join(diff)} differ)")
        return all((self.dir / key_file(r["key"])).is_file() for r in reqs)


# ------------------------------------------------------------------------------------ bench

def _stage(scorer, reqs, root: Path, log) -> dict:
    """Copy counts.h5, pr1.h5 and pr2.h5 of every measurement the sampled requests read (and their
    controls) to local disk and point the rows at the copies; other files are links. Byte copies,
    so every value read is the same (as score.py --stage-dir of 9cd5f04, reduced to the bench)."""
    import shutil
    rows = scorer.ds.rows
    mids = set()
    for r in reqs:
        j = r["job"]
        mids |= {j["target"], *j.get("inputs", []), *([j["source"]] if "source" in j else [])}
    mids |= {(rows[m].get("control_measurement_id") or "").strip() for m in list(mids)}
    dirs, nbytes, t0 = {}, 0, time.time()
    for m in sorted(x for x in mids if x in rows):
        path = rows[m]["counts_path"]
        if not path.startswith("h5:"):
            continue
        src = Path(path[3:]).parent
        dst = root / src.name
        dirs[src] = dst
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            d = dst / f.name
            if d.exists() or d.is_symlink():
                continue
            if f.name in ("counts.h5", "pr1.h5", "pr2.h5"):
                shutil.copyfile(f, d)
                nbytes += d.stat().st_size
            else:
                d.symlink_to(f)
    for row in rows.values():
        path = row["counts_path"]
        if path.startswith("h5:") and Path(path[3:]).parent in dirs:
            row["counts_path"] = "h5:" + str(dirs[Path(path[3:]).parent] / Path(path[3:]).name)
    scorer.ds._close_all() if hasattr(scorer.ds, "_close_all") else None
    scorer._region, scorer._rc = None, {}
    info = {"directories": len(dirs), "bytes_copied": nbytes, "seconds": round(time.time() - t0, 1)}
    log(f"bench: staged {info}")
    return info


def _rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30))) if a.size else 0.0


def _store_diff(dir_a, dir_b, reqs) -> dict:
    A, Bs = PredictionStore(dir_a), PredictionStore(dir_b)
    out = {}
    for r in reqs:
        a, b = A.read(r["key"]), Bs.read(r["key"])
        for k in a:
            out[k] = max(out.get(k, 0.0), _rel(a[k], b[k]))
    return out


def _sample(scorer, all_reqs, n_items, n_pairs, n_regions) -> list:
    R = len(scorer.regions)
    step = max(1, R // n_regions)
    sub = list(range(0, R, step))[:n_regions]
    items = [r for r in all_reqs if r["key"].startswith("item/") and r["job"]["regime"] != "control"]
    items = ([r for r in items if r["needs_ctl"]][:1] + [r for r in items if not r["needs_ctl"]])
    pairs = [r for r in all_reqs if r["kind"] == "correction"]
    pick = items[:n_items] + pairs[:n_pairs]
    return [dict(r, region_indices=sub) for r in pick]


def _bench_device(a, device, log) -> dict:
    from minicandi.registry import load_run
    model, config = load_run(a.run)
    run_sha = S.run_tables_sha256(a.run)
    ev_tables = json.loads(Path(a.evalset).read_text(encoding="utf-8"))["tables_sha256"]
    other = run_sha != ev_tables
    if other and not a.allow_other_tables:
        raise S.ScoreRefused("the run was trained on other tables (bench: --allow-other-tables)")
    t0 = time.time()
    sc = S.Scorer(a.evalset, main="model", model=model, config=config, device=device,
                  run_tables_sha256=ev_tables if other else run_sha, log=log,
                  guardrail_windows=8, n_boot=0)
    t_init = time.time() - t0
    all_reqs = requests(sc)
    reqs = _sample(sc, all_reqs, a.n_items, a.n_pairs, a.regions)
    res = {"device": device, "scorer_init_seconds": round(t_init, 1), "n_requests_all": len(all_reqs),
           "requests": [r["key"] for r in reqs], "needs_ctl": [r["needs_ctl"] for r in reqs],
           "n_regions_per_request": a.regions, "windows_per_region": sc.rb // sc.L,
           "legacy_batch": sc.tile_bs, "torch_threads": torch.get_num_threads(),
           "trained_on_other_tables": other, "params": sum(p.numel() for p in model.parameters())}
    if a.stage_dir:
        res["stage"] = _stage(sc, reqs, Path(a.stage_dir), log)
    cuda = str(device).startswith("cuda")
    if cuda:
        props = torch.cuda.get_device_properties(0)
        res["gpu"] = {"name": props.name, "total_memory_bytes": int(props.total_memory)}
    # warm-up (h5 opens, CUDA context, kernels): one request at the legacy batch, not timed
    request_arrays(sc, dict(reqs[0], region_indices=reqs[0]["region_indices"][:1]), sc.tile_bs)
    # probe: the largest batch whose single forward fits (GPU only)
    fits, probe = [], {}
    if cuda:
        tiles = [(ri, c, s) for ri, (_, tl) in zip(reqs[-1]["region_indices"], sc._tiles(
            [sc.regions[i] for i in reqs[-1]["region_indices"]])) for c, s in tl]
        for B in a.probe:
            wins = []
            for ri, c, s in (tiles * (B // len(tiles) + 1))[:B]:
                sc._set_region(*sc.regions[ri])
                wins.append(sc._pair_window(reqs[-1]["job"], c, s))
            batch = sc.C.collate(wins)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                sc.model_pred(batch)
                probe[B] = {"fits": True, "peak_allocated": int(torch.cuda.max_memory_allocated()),
                            "peak_reserved": int(torch.cuda.max_memory_reserved())}
                fits.append(B)
            except torch.cuda.OutOfMemoryError:
                probe[B] = {"fits": False}
                del batch, wins
                torch.cuda.empty_cache()
                break
            del batch, wins
        res["probe"] = probe
    win_per_req = a.regions * (sc.rb // sc.L)
    batches = list(a.batches) + [B for B in fits if B > max(a.batches) and B <= win_per_req][-1:]
    res["batches"] = {}
    for B in batches:
        d = Path(a.store_root) / device.replace(":", "") / f"b{B}"
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        tm = defaultdict(float)
        t0 = time.perf_counter()
        predict(sc, reqs, d, B, run=a.run, guardrail=False, timing=tm, log=log)
        wall = time.perf_counter() - t0
        row = {"wall_seconds": round(wall, 3), **{k: (round(v, 3) if isinstance(v, float) else v)
                                                  for k, v in tm.items()},
               "windows_per_second_end_to_end": tm["windows"] / wall,
               "windows_per_second_forward_only": tm["windows"] / tm["forward_seconds"],
               "forwards_per_second_with_control": (tm["windows"] + tm["control_windows"]) / wall}
        if cuda:
            row["peak_allocated"] = int(torch.cuda.max_memory_allocated())
            row["peak_reserved"] = int(torch.cuda.max_memory_reserved())
        res["batches"][B] = row
        log(f"bench: {device} batch {B}: {row['windows_per_second_end_to_end']:.1f} windows/s end "
            f"to end, {row['windows_per_second_forward_only']:.1f} forward only")
    base = Path(a.store_root) / device.replace(":", "") / f"b{batches[0]}"
    for B in batches[1:]:
        res["batches"][B]["max_rel_diff_vs_first"] = _store_diff(
            Path(a.store_root) / device.replace(":", "") / f"b{B}", base, reqs)
    # the stored floats against the legacy _job_arrays at the legacy batch (one item, one pair)
    if sc.tile_bs in batches:
        st = PredictionStore(Path(a.store_root) / device.replace(":", "") / f"b{sc.tile_bs}")
        eq = {}
        for r in [x for x in reqs if x["kind"] != "correction"][:1] + [
                x for x in reqs if x["kind"] == "correction"][:1]:
            old_regions = sc.regions
            sc.regions = [old_regions[i] for i in r["region_indices"]]
            try:
                L = sc._job_arrays(r["job"], ())
            finally:
                sc.regions = old_regions
            s = st.read(r["key"])
            eq[r["key"]] = {k: bool(np.array_equal(s[k].reshape(-1).astype(np.float64),
                                                   L[f"main.{k}" if k != "ctl_mu" else "main.ctl_mu"]))
                            for k in s}
        res["legacy_bit_identical"] = eq
    return res


def bench(argv=None) -> int:
    ap = argparse.ArgumentParser(description="forward speed and memory of the prediction store")
    ap.add_argument("--run", required=True)
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--store-root", required=True)
    ap.add_argument("--devices", default="cuda")
    ap.add_argument("--batches", default="8,16,32,64")
    ap.add_argument("--probe", default="128,256,512,1024")
    ap.add_argument("--regions", type=int, default=32)
    ap.add_argument("--n-items", type=int, default=2)
    ap.add_argument("--n-pairs", type=int, default=4)
    ap.add_argument("--stage-dir", default=None)
    ap.add_argument("--allow-other-tables", action="store_true")
    a = ap.parse_args(argv)
    for k in ("run", "evalset", "out", "store_root", "stage_dir"):     # "$SLURM_TMPDIR/..." works
        if getattr(a, k):
            setattr(a, k, os.path.expandvars(getattr(a, k)))
    a.batches = [int(x) for x in a.batches.split(",")]
    a.probe = [int(x) for x in a.probe.split(",") if x]
    S.set_threads()

    def log(*m):
        print(*m, flush=True)
    out = {"run": str(Path(a.run).resolve()), "evalset": a.evalset, "git_sha": S.git_sha(),
           "host": os.uname().nodename, "devices": {}}
    for dev in a.devices.split(","):
        out["devices"][dev] = _bench_device(a, dev, log)
        Path(a.out).write_text(json.dumps(out, indent=1, default=_jsonable) + "\n")
    devs = list(out["devices"])
    if len(devs) == 2:          # CPU against GPU, per common batch
        d0, d1 = devs
        out["cross_device_max_rel_diff"] = {}
        for B in out["devices"][d0]["batches"]:
            if B in out["devices"][d1]["batches"]:
                sc_reqs = [{"key": k} for k in out["devices"][d0]["requests"]]
                out["cross_device_max_rel_diff"][B] = _store_diff(
                    Path(a.store_root) / d0.replace(":", "") / f"b{B}",
                    Path(a.store_root) / d1.replace(":", "") / f"b{B}", sc_reqs)
    Path(a.out).write_text(json.dumps(out, indent=1, default=_jsonable) + "\n")
    log(f"bench: wrote {a.out}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        sys.exit(bench(sys.argv[2:]))
    print(__doc__)
    sys.exit(2)
