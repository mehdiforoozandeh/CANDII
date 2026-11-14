"""miniCANDI train loop (plan pinned interface C).

    python tools/minicandi/train.py --config <json> --tables <dir> --seed <k> --out <run dir>
        [--device cpu|cuda] [--max-windows N] [--val-every W] [--val-batches 8]
        [--log-every 50] [--ckpt-minutes 15] [--workers 0] [--stage-dir <local dir>]

One run directory holds:
  ckpt.pt         the pinned dict (`contract.make_ckpt`): state_dict, the full config, param_count,
                  windows_seen, tables_sha256, git_sha. `registry.load_run(run_dir)` reads it.
  train_state.pt  what an exact resume needs beyond ckpt.pt: optimizer, step, the RNG states
                  (torch, CUDA, the numpy sampler, the scramble generator), wall seconds, FLOPs
  run.json        config, seed, git sha, tables sha256, param count, FLOPs per window (forward +
                  backward, measured by `torch.utils.flop_counter.FlopCounterMode` on one batch,
                  divided by its windows), training FLOPs (= windows seen x FLOPs per window: the
                  matched-compute currency), windows seen, steps, wall seconds, device
  log.jsonl       one line per `--log-every` steps (and per validation pass)
  DONE            written once windows_seen reaches `config["windows_seen"]`, or (with CAPPED)
                  once it reaches `--max-windows`
  CAPPED          written (before DONE) when the run stopped at `--max-windows` short of the
                  budget: json {"windows_seen", "max_windows", "budget"}. A run with CAPPED is
                  NOT a full-budget run; run.json then has "done": false, "capped": true.

Staging (`--stage-dir`, default off). Before the first batch, the `counts.h5` of every measurement
the training dataset may read is copied (8 threads, size-checked) to `<stage dir>/<measurement dir
name>/counts.h5`; its `pr1.h5` / `pr2.h5` become symlinks to the originals; the dataset rows then
point at the copies. The bytes are the same, so batches, losses and checkpoints are bit-identical
to an unstaged run, and a resume may switch staging on or off. Reason (chunk L1a, Nibi
2026-11-13): an h5 open on Nibi's NFS /scratch took 0.21 s, the LRU of 256 handles reopens most
files, and a rung-1 batch took 11.4 s; staged on the node's local disk it took 0.17 s.

Budget. `config["windows_seen"]` is the training budget in windows; steps = ceil(budget /
batch_size). `--max-windows N` stops this invocation at N windows (checkpointed) and ends it in
the terminal state CAPPED + DONE, so a launcher that skips on DONE (`submit_arms.sh`) does not
resubmit a finished capped smoke. Rerunning with a larger `--max-windows`, or none, removes CAPPED
and DONE and resumes from ckpt.pt, continuing the step count, so a long run can still be split
across jobs; with DONE present otherwise it exits 0 at once. A resume refuses a changed config
or changed tables (sha256). A resume is exact: on CPU, k steps then a resume to n equals n steps.

Determinism (AGENTS.md invariant 10): cudnn deterministic, benchmark off, then
`torch.manual_seed(seed)` immediately before `registry.build_model`. The window sampler draws from
`numpy.random.default_rng([seed, 1])`; the FLOPs batch and the pearson statistics come from their
own generators and the forked torch RNG, so they move no training draw.

Scrambled covariates (`scramble_covariates`, `scramble_all`): the model applies them, once per
forward (`covariates.model_covariates`); neither this loop nor the dataset scrambles.

Schedule: AdamW (lr = `config["lr"]`, weight decay 0.01), linear warm-up over the first 5% of steps
(at most 1000), cosine decay to 10% of lr at the last step; gradient norm clipped at 1.0. The loss
is `losses.total_loss(out, batch, config)` (config keys `loss`, `loss_weights`, and the base leg of
`two_step`); without those keys it is `contract.nb_nll` over `contract.loss_mask`, bit for bit.
Each log line carries the loss `parts`. A non-finite loss stops the run with an error and leaves
the last checkpoint in place.

Data workers. `--workers 0` (default) draws every training batch in this process from the sampler
generator above. `--workers N` (N > 0) draws them in N DataLoader worker processes: batch k comes
from its own generator `numpy.random.default_rng([seed, 4, k])`, so the batch sequence depends on
the seed only (not on N, nor on which worker draws a batch), a resume is exact, and the worker
base seed comes from a generator seeded with `seed`, so no draw of the training torch RNG moves.
Each worker opens its own h5 handles (`OnionDataset` drops inherited handles after a fork and
pickles without them under spawn). The two streams differ, so a resume refuses a switch between
0 and N > 0 workers. Measured on Fir: 0.93 s per batch in-process against 0.081 s with 8 workers.

Training regions (pinned interface I): with `train_regions` or `exclude_regions`, run.json also
holds the path and sha256 of the region file read (`OnionDataset.region_record`); with
`train_regions`, each log line holds `windows_per_stratum`, the cumulative window count per stratum
(kept in train_state.pt across a resume), and the loss weights ride in the batch (`w_window`). With
`exclude_regions`, each log line holds `exclude_rejections` (cumulative starts rejected for overlap)
and `exclude_rejection_rate` = rejections / (rejections + windows), i.e. the share of drawn starts
that overlapped. A resume refuses a region file whose sha256 differs from the one in run.json.
Without the keys run.json, log.jsonl and train_state.pt are unchanged.

`--val-every W` (windows, 0 = off) logs the NB loss (of the main output, whatever `loss` is) over the first `--val-batches` batches of the
dataset in val mode (needs the tables' evalset.json); the batches are fixed for the whole run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from minicandi import contract as C  # noqa: E402
from minicandi.components import CountTransform  # noqa: E402
from minicandi.dataset import OnionDataset, tables_sha256  # noqa: E402
from minicandi.losses import check_config as check_loss_config  # noqa: E402
from minicandi.losses import total_loss  # noqa: E402
from minicandi.registry import CKPT_NAME, build_model  # noqa: E402

DEFAULTS = {"context_bins": 1024, "batch_size": 8, "lr": 1e-3, "param_budget": 1.0e6,
            "scramble_covariates": False}
STATE_NAME = "train_state.pt"
WEIGHT_DECAY = 0.01
CLIP = 1.0
WARMUP_FRAC, WARMUP_MAX, LR_FLOOR = 0.05, 1000, 0.1
PEARSON_BATCHES = 4
CAPPED_NAME = "CAPPED"
WORKER_STREAM = 4          # batch k with workers: default_rng([seed, WORKER_STREAM, k])


class StepBatches(torch.utils.data.Dataset):
    """Batch k of a run with workers, from its own generator: picklable for DataLoader workers."""

    def __init__(self, ds: OnionDataset, batch_size: int, seed: int):
        self.ds, self.bs, self.seed = ds, int(batch_size), int(seed)

    def __getitem__(self, k: int) -> dict:
        return self.ds.batch(self.bs, np.random.default_rng([self.seed, WORKER_STREAM, int(k)]))


def worker_batches(ds: OnionDataset, batch_size: int, seed: int, first_step: int, total_steps: int,
                   workers: int):
    """Iterator over the batches of steps first_step .. total_steps - 1 drawn by `workers`
    DataLoader worker processes (in step order)."""
    loader = torch.utils.data.DataLoader(
        StepBatches(ds, batch_size, seed), batch_size=None,
        sampler=range(first_step, total_steps), num_workers=workers,
        generator=torch.Generator().manual_seed(seed))
    return iter(loader)


def stage_counts(ds: OnionDataset, stage_dir, threads: int = 8) -> dict:
    """`--stage-dir`: copy the counts.h5 of every row `ds` may read to local disk and point the rows
    at the copies (pr1.h5 / pr2.h5 symlinked to the originals). Returns files, bytes, seconds."""
    from concurrent.futures import ThreadPoolExecutor
    import shutil
    root = Path(stage_dir)
    t0 = time.time()
    jobs = {}
    for mid in sorted(ds.allowed):
        path = ds.rows[mid]["counts_path"]
        if not path.startswith("h5:"):
            continue
        src = Path(path[3:])
        dst = root / src.parent.name / src.name
        if jobs.setdefault(dst, src) != src:
            raise ValueError(f"--stage-dir: two sources map to {dst}")

    def copy(item):
        dst, src = item
        size = src.stat().st_size
        if not (dst.is_file() and dst.stat().st_size == size):
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            shutil.copyfile(src, tmp)
            if tmp.stat().st_size != size:
                raise OSError(f"--stage-dir: short copy of {src}")
            os.replace(tmp, dst)
        for sib in ("pr1.h5", "pr2.h5"):
            s_, d_ = src.parent / sib, dst.parent / sib
            if s_.exists() and not d_.exists():
                d_.symlink_to(s_)
        return size

    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as pool:
        nbytes = sum(pool.map(copy, sorted(jobs.items())))
    staged = {str(src): str(dst) for dst, src in jobs.items()}
    for row in ds.rows.values():
        path = row["counts_path"]
        if path.startswith("h5:") and path[3:] in staged:
            row["counts_path"] = "h5:" + staged[path[3:]]
    ds._close_all()
    return {"files": len(jobs), "bytes": int(nbytes), "seconds": round(time.time() - t0, 1)}


def full_config(config: dict) -> dict:
    """The config with the train loop's defaults filled in; this is what ckpt.pt stores."""
    cfg = dict(DEFAULTS)
    cfg.update(config)
    if "rung" not in cfg:
        raise KeyError("config has no 'rung'")
    if not cfg.get("windows_seen"):
        raise KeyError("config has no 'windows_seen' (the training budget in windows)")
    return cfg


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "-C", str(_TOOLS.parent), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return os.environ.get("MINICANDI_GIT_SHA", "unknown")


def lr_at(step: int, total: int, lr: float) -> float:
    warm = max(1, min(WARMUP_MAX, int(WARMUP_FRAC * total)))
    if step < warm:
        return lr * (step + 1) / warm
    frac = (step - warm) / max(1, total - warm)
    return lr * (LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0))))


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def step_loss(model, batch: dict, config: dict | None = None,
              parts: dict | None = None) -> torch.Tensor:
    """The training loss of one batch. With `config`, `losses.total_loss` (its parts go into
    `parts` when given); without, the NB NLL of the main output over `contract.loss_mask`."""
    out = model(batch)
    if config is None:
        return C.nb_nll(out["n"], out["mu"], batch["y_counts"], C.loss_mask(batch))
    loss, p = total_loss(out, batch, config)
    if parts is not None:
        parts.update(p)
    return loss


def _devices(device) -> list:
    return [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []


def flops_per_window(model, batch: dict, device, config: dict | None = None) -> tuple[float, float]:
    """(training, forward) FLOPs per window on one batch (the loss of `config`); moves no RNG and
    leaves no gradient."""
    B = batch["x_counts"].shape[0]
    with torch.random.fork_rng(devices=_devices(device)):
        with FlopCounterMode(display=False) as fc:
            out = model(batch)
        fwd = fc.get_total_flops()
        with FlopCounterMode(display=False) as fc:
            loss = step_loss(model, batch, config)
            loss.backward()
        total = fc.get_total_flops()
    model.zero_grad(set_to_none=True)
    return total / B, fwd / B


def set_pearson_stats(model, batches: list) -> None:
    """Per-track mean rate l_a and NB size r_a (method of moments) of the visible input counts at
    the 2**24-read reference depth, into every CountTransform of the model."""
    s1 = torch.zeros(C.A, dtype=torch.float64)
    s2 = torch.zeros(C.A, dtype=torch.float64)
    n = torch.zeros(C.A, dtype=torch.float64)
    for b in batches:
        vis = (b["x_avail"][:, None, :] & ~b["x_bin_mask"]).double()
        s = CountTransform.size_factor(b["c_in"][..., C.LOG2_DEPTH_COL]).double()
        r = b["x_counts"].double() / s
        s1 += (r * vis).sum((0, 1))
        s2 += (r * r * vis).sum((0, 1))
        n += vis.sum((0, 1))
    lam = torch.ones(C.A, dtype=torch.float64)
    size = torch.full((C.A,), 10.0, dtype=torch.float64)
    ok = n > 0
    lam[ok] = (s1[ok] / n[ok]).clamp_min(1e-3)
    var = torch.zeros_like(lam)
    var[ok] = s2[ok] / n[ok] - lam[ok] ** 2
    over = ok & (var > lam * 1.01)
    size[over] = (lam[over] ** 2 / (var[over] - lam[over])).clamp(0.1, 1e4)
    for mod in model.modules():
        if isinstance(mod, CountTransform):
            mod.set_pearson_stats(lam.float(), size.float())


def _save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_json(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def train(config: dict, tables, seed: int, out, device: str = "cpu",
          max_windows: int | None = None, val_every: int = 0, val_batches: int = 8,
          log_every: int = 50, ckpt_minutes: float = 15.0, workers: int = 0,
          stage_dir=None) -> dict:
    """Train (or resume) one run; returns the run.json dict."""
    cfg = full_config(config)
    check_loss_config(cfg)
    workers = int(workers)
    if workers < 0:
        raise ValueError(f"--workers {workers}: must be >= 0")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").is_file():
        cap = (json.loads((out / CAPPED_NAME).read_text(encoding="utf-8"))
               if (out / CAPPED_NAME).is_file() else None)
        if cap is None or (max_windows is not None and max_windows <= cap["windows_seen"]):
            print(f"skip: {out / 'DONE'} exists" + (" (capped)" if cap else ""))
            return json.loads((out / "run.json").read_text(encoding="utf-8"))
        print(f"extend: capped at {cap['windows_seen']} windows; resuming")
        (out / "DONE").unlink()
        (out / CAPPED_NAME).unlink()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but CUDA is not available")
    dev = torch.device(device)
    tsha = tables_sha256(tables)
    bs = int(cfg["batch_size"])
    budget = int(cfg["windows_seen"])
    total_steps = math.ceil(budget / bs)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ds = OnionDataset(tables, "train", cfg, seed)
    if stage_dir:
        st_ = stage_counts(ds, stage_dir)
        print(f"staged {st_['files']} counts files, {st_['bytes'] / 1e9:.2f} GB, in "
              f"{st_['seconds']} s to {stage_dir}", flush=True)
    torch.manual_seed(seed)
    model = build_model(cfg)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=WEIGHT_DECAY)
    data_rng = np.random.default_rng([seed, 1])
    scr_gen = torch.Generator().manual_seed(seed)
    region = ds.region_record()                      # interface I; {} without its keys
    strata: dict = {}
    excl: dict = {}                                  # exclude_regions: rejections and windows

    ckpt_path, state_path = out / CKPT_NAME, out / STATE_NAME
    resumed = ckpt_path.is_file()
    if resumed:
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        C.check_ckpt(ck)
        if ck["config"] != cfg:
            raise ValueError(f"{ckpt_path}: saved config differs from --config; use a new --out")
        if ck["tables_sha256"] != tsha:
            raise ValueError(f"{ckpt_path}: tables sha256 changed since the run started")
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(st["optimizer"])
        step, windows = int(st["step"]), int(ck["windows_seen"])
        torch.set_rng_state(st["torch_rng"])
        if dev.type == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        data_rng.bit_generator.state = st["data_rng"]
        scr_gen.set_state(st["scramble_rng"])
        wall0, fpw, fpw_fwd = float(st["wall_seconds"]), st["flops_per_window"], \
            st["flops_per_window_forward"]
        n_resumes = int(st.get("n_resumes", 0)) + 1
        strata = dict(st.get("windows_per_stratum", {}))
        excl = dict(st.get("exclude_counts", {}))
        prev = json.loads((out / "run.json").read_text(encoding="utf-8")) \
            if (out / "run.json").is_file() else {}
        for k in ("train_regions_sha256", "exclude_regions_sha256"):
            if prev.get(k) != region.get(k):
                raise ValueError(f"{out / 'run.json'}: {k} {prev.get(k)} differs from the region "
                                 f"file read now ({region.get(k)}); a changed list changes the "
                                 f"stream: use a new --out")
        if (int(st.get("workers", 0)) > 0) != (workers > 0):
            raise ValueError(f"{state_path}: the run started with --workers {st.get('workers', 0)}; "
                             f"0 and N > 0 workers draw different batch streams, resume with "
                             f"{'--workers 0' if not st.get('workers', 0) else '--workers N > 0'}")
        print(f"resume: step {step}, windows {windows} of {budget}")
    else:
        step, windows, wall0, n_resumes = 0, 0, 0.0, 0
        probe = ds.batch(bs, np.random.default_rng([seed, 2]))
        model.train()
        fpw, fpw_fwd = flops_per_window(model, to_device(probe, dev), dev, cfg)
        if cfg.get("count_transform") == "pearson":
            prng = np.random.default_rng([seed, 3])
            set_pearson_stats(model, [ds.batch(bs, prng) for _ in range(PEARSON_BATCHES)])
    sha = git_sha()

    val = None
    if val_every > 0:
        try:
            vds = OnionDataset(tables, "val", cfg, seed)
            val = [b for _, b in zip(range(val_batches), vds)]
        except FileNotFoundError as e:
            print(f"validation off: {e}")

    def val_loss() -> float:
        model.eval()
        tot, k = 0.0, 0
        with torch.no_grad():
            for b in val:
                b = to_device(b, dev)
                m = C.loss_mask(b)
                if m.any():
                    tot += step_loss(model, b).item() * int(m.sum())
                    k += int(m.sum())
        model.train()
        return tot / max(k, 1)

    def summary(done: bool, last: dict) -> dict:
        return {"config": cfg, "seed": int(seed), "git_sha": sha, "tables_sha256": tsha,
                "tables": str(Path(tables).resolve()), "param_count": model.param_count(),
                "d_model": getattr(model, "d_model", None),
                "flops_per_window": fpw, "flops_per_window_forward": fpw_fwd,
                "training_flops": windows * fpw, "windows_seen": windows, "steps": step,
                "total_steps": total_steps, "wall_seconds": wall0 + (time.time() - t0),
                "device": str(dev), "host": socket.gethostname(), "torch": torch.__version__,
                "n_resumes": n_resumes, "done": done, **({"workers": workers} if workers else {}),
                **region, **last}

    def checkpoint(done: bool, last: dict) -> dict:
        _save(C.make_ckpt(model, cfg, windows, tsha, sha), ckpt_path)
        _save({"optimizer": opt.state_dict(), "step": step, "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if dev.type == "cuda" else None,
               "data_rng": data_rng.bit_generator.state, "scramble_rng": scr_gen.get_state(),
               "wall_seconds": wall0 + (time.time() - t0), "flops_per_window": fpw,
               "flops_per_window_forward": fpw_fwd, "n_resumes": n_resumes,
               **({"workers": workers} if workers else {}),
               **({"windows_per_stratum": dict(strata)} if strata else {}),
               **({"exclude_counts": dict(excl)} if excl else {})}, state_path)
        info = summary(done, last)
        _write_json(info, out / "run.json")
        return info

    log = open(out / "log.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    t_ckpt = t0
    last: dict = {}
    next_val = (windows // val_every + 1) * val_every if val else None
    model.train()
    loader = worker_batches(ds, bs, seed, step, total_steps, workers) if workers > 0 else None
    while windows < budget:
        if max_windows is not None and windows >= max_windows:
            break
        batch = ds.batch(bs, data_rng) if loader is None else next(loader)
        batch = to_device(batch, dev)
        if "w_window" in batch:                      # interface I: windows per stratum
            for m in batch["meta"]:
                strata[m["stratum"]] = strata.get(m["stratum"], 0) + 1
        if batch["meta"] and "exclude_rejections" in batch["meta"][0]:
            for m in batch["meta"]:
                excl["rejections"] = excl.get("rejections", 0) + int(m["exclude_rejections"])
                excl["windows"] = excl.get("windows", 0) + 1
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total_steps, float(cfg["lr"]))
        parts: dict = {}
        loss = step_loss(model, batch, cfg, parts)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss {loss.item()} at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP))
        opt.step()
        step += 1
        windows += batch["x_counts"].shape[0]
        last = {"last_train_loss": float(loss.item())}
        if step % log_every == 0 or step == 1 or windows >= budget:
            rec = {"step": step, "windows_seen": windows, "training_flops": windows * fpw,
                   "loss": float(loss.item()), "grad_norm": gnorm,
                   "lr": opt.param_groups[0]["lr"],
                   "wall_seconds": round(wall0 + time.time() - t0, 2), "parts": parts,
                   **({"windows_per_stratum": dict(strata)} if strata else {}),
                   **({"exclude_rejections": excl["rejections"],
                       "exclude_rejection_rate": excl["rejections"]
                       / (excl["rejections"] + excl["windows"])} if excl else {})}
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if next_val is not None and windows >= next_val:
            vl = val_loss()
            last["last_val_loss"] = vl
            log.write(json.dumps({"step": step, "windows_seen": windows, "val_loss": vl}) + "\n")
            log.flush()
            next_val += val_every
        if time.time() - t_ckpt > 60.0 * ckpt_minutes:
            checkpoint(False, last)
            t_ckpt = time.time()
    del loader                                   # stops the worker processes
    done = windows >= budget
    capped = not done and max_windows is not None and windows >= max_windows
    if capped:
        last = {**last, "capped": True, "max_windows": int(max_windows)}
    info = checkpoint(done, last)
    log.close()
    if done:
        (out / "DONE").write_text(f"{windows}\n", encoding="utf-8")
    elif capped:                     # CAPPED first: a DONE is never seen without it on a capped run
        _write_json({"windows_seen": windows, "max_windows": int(max_windows), "budget": budget},
                    out / CAPPED_NAME)
        (out / "DONE").write_text(f"{windows} capped at --max-windows {max_windows} of {budget}\n",
                                  encoding="utf-8")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--val-every", type=int, default=0)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-minutes", type=float, default=15.0)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader worker processes for training batches (0 = in-process)")
    ap.add_argument("--stage-dir", default=None,
                    help="copy the training counts.h5 files to this local dir first (identical "
                         "batches; see 'Staging')")
    a = ap.parse_args(argv)
    config = json.loads(Path(a.config).read_text(encoding="utf-8"))
    info = train(config, a.tables, a.seed, a.out, a.device, a.max_windows, a.val_every,
                 a.val_batches, a.log_every, a.ckpt_minutes, a.workers, a.stage_dir)
    print(json.dumps({k: info[k] for k in ("windows_seen", "steps", "param_count",
                                           "flops_per_window", "training_flops", "done")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
