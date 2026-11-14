"""EVt -- the frozen proxy scorer that a `crux auto` run calls on one candidate recipe (PI ruling 23).

    python3 tools/onion_eval/auto_proxy.py --recipe <dir with config.json and model.py>
        [--windows 20000] [--seeds 0,1] [--tables <eic_v1>] [--evalset <proxy evalset>]
        [--cluster nibi] [--kit main] [--runs-root <R>] [--local]
    python3 tools/onion_eval/auto_proxy.py job ...   (inside the Slurm job, slurm/onion_eval/auto_proxy.sh)

Stdout is exactly one JSON object (the crux-autopilot scorer contract; every log line goes to
stderr):
    {"passed": true|false, "reason": str, "composite": {"value": x}, "imputation": {"value": x},
     "correction": {"value": x}, "uncertainty": {"value": x}, "seed_values": [...],
     "param_count": n, "flops_per_window": f, "cpu_hours": c, "attempt": <attempt dir>}
and the exit code is 0 only when `passed` is true. A refused candidate gets `passed` false, the
reason, and nulls for every score: no score is ever invented.

Steps, in order (the first failure ends the attempt):
  1. Static checks on the recipe. `config.json` holds only keys of ALLOWED_KEYS (pinned interfaces
     C and E, the rung hyperparameters, the training sampler's two keys); none of FORBIDDEN_KEYS
     (`allow_blind`, `train_on_validation`, `final_model`); no path key (`evalset`, `genome_dir`,
     `average_h5`, `dna_embedding_h5`, `qc_table`, `region_bins`, ... are outside the set); no
     string value with a `/` except `model_file`. `model_file` points at the recipe's own
     `model.py` (required except under --local). `model.py`, read as a syntax tree, imports only
     torch, an allowlist of pure-computation standard-library modules (STDLIB_OK) and
     minicandi.{contract,components,covariates}; it names no file, exec or introspection builtin
     or attribute (BANNED_NAMES, BANNED_ATTRS), takes no dynamic getattr, and holds no string
     `y_counts` / `y_base`.
  2. Dynamic checks on the built model, on a batch of synthetic tables (`minicandi.synth`): the
     model is built by `registry.build_model`, and a class from the recipe's `model.py` must be in
     it when `model_file` is set (so a registry without `model_file` support never scores the
     wrong module); the parameter count is within 10% of 1.0e6 (a constant: the config's own
     `param_budget` is not trusted); then, with a perturbed head (noise on every parameter, as the
     rung tests), `contract.wiring_check` must pass (the hard gate, PI ruling 17) and the outputs
     must not move when `y_counts` / `y_base` change (target independence, four calls, so a model
     that keeps state across calls also fails).
  3. The budget: the CPU-h in `<runs root>/evolve/TALLY.tsv` must be under ONION_EVOLVE_CAP_CPUH
     (default 3000), before the attempt and after it (the attempt that crosses the cap is refused).
  4. Per seed, one training run of `--windows` windows and one score with `mode = val` on the
     proxy eval set: in-process with --local, else one Slurm job per seed through
     `hpc run <cluster>` (slurm/onion_eval/auto_proxy.sh), which this scorer waits for. Before
     training, the job (and --local) refuses an eval set that lists any blind row, a blind pair or
     a non-validation target, or whose tables hash differs; in cluster mode the tables must be
     the `eic_v1` tables. After training the trained model must again be target-independent on
     real training batches, and each seed's scores.json must pass the wiring part of the copy-leak
     guardrail (`copy_leak.part1_wiring_pass`).
  5. The composite: per seed, `select_rule`'s D13 composite on the clipped goal terms (PI ruling
     18); refused when the scores are not comparable or a goal term or denominator fails
     `select_rule.screen_terms`. Reported: the mean over seeds, and each goal's clipped mean.

Cluster mode also refuses when the kit's copy of a frozen file differs from this tree's (so the
static checks and the job run the same code), and maps `hpc` exit 69 to `passed` false with the
reason `tunnel down`. On SIGTERM / SIGINT and after --max-wait-minutes the submitted jobs are
cancelled. The job appends its own CPU-h row to TALLY.tsv when it exits (any state); a job that
left no row is filled in from sacct.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

PARAM_TARGET = 1.0e6
PARAM_TOL = 0.10
DEFAULT_TABLES = "/project/def-maxwl/mforooz/ONION/tables/eic_v1"
DEFAULT_EVALSET = "/project/def-maxwl/mforooz/ONION/eval/eic_v1/proxy/evalset.json"
DEFAULT_RUNS = "/project/def-maxwl/mforooz/ONION/runs"
CLUSTER_TABLES_NAME = "eic_v1"
KITS = "/project/def-maxwl/mforooz/onion_kit"
CAP_ENV = "ONION_EVOLVE_CAP_CPUH"
DEFAULT_CAP_CPUH = 3000.0
TUNNEL_DOWN = 69
JOB_CPUS, JOB_MEM_MB, JOB_WORKERS = 12, 48000, 8
TALLY_COLUMNS = ("time", "attempt", "seed", "job_id", "cpus", "wall_seconds", "cpu_hours", "exit")
PROXY_FILE = "PROXY.json"
UMBRELLA_ORCH = Path("/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/.claude/worktrees/"
                     "onion/.orchestrate")

FORBIDDEN_KEYS = ("allow_blind", "train_on_validation", "final_model")
ALLOWED_KEYS = frozenset((
    # pinned interface C
    "rung", "context_bins", "covariate_injection", "mask_mode", "dna", "scramble_covariates",
    "scramble_all", "count_transform", "qc_on_input", "two_step", "base_depth", "average_input",
    "control_input", "predict_control", "param_budget", "windows_seen", "lr", "batch_size",
    "simpler_than_base",
    # pinned interface E (path keys excluded: dna_embedding_h5, qc_table)
    "model_file", "control_head", "lambda_window_bins", "loss", "loss_weights", "input_depth",
    "mask_scheme", "bin_mask_shape", "covariate_dropout", "covariate_embedding", "offset_eps",
    "head", "dna_shift_bp",
    # rung hyperparameters read by tools/minicandi/rungs/*.py
    "width", "d_model", "n_heads", "n_self_layers", "n_latents", "mlp_ratio", "patch_bins",
    "d_dna", "d_bin", "d_bin_out", "axis", "skip_source",
    # the training sampler (dataset.DEFAULTS; train mode only)
    "mask_prob_range", "same_view_prob",
))

STDLIB_OK = frozenset(("__future__", "math", "functools", "itertools", "collections",
                       "collections.abc", "dataclasses", "typing", "enum", "numbers", "abc", "copy",
                       "operator", "warnings", "sys", "pathlib", "re", "string", "bisect", "heapq",
                       "statistics", "fractions", "contextlib"))
MINICANDI_OK = frozenset(("contract", "components", "covariates"))
TORCH_BANNED = ("torch.hub", "torch.package", "torch.utils.cpp_extension", "torch.multiprocessing",
                "torch.distributed", "torch.utils.data", "torch.serialization", "torch.onnx",
                "torch.utils.tensorboard", "torch._C")
SYS_OK = frozenset(("path", "version", "version_info", "maxsize", "float_info"))
BANNED_NAMES = frozenset(("open", "exec", "eval", "compile", "__import__", "globals", "locals",
                          "vars", "breakpoint", "input", "__builtins__", "__loader__", "__spec__",
                          "exit", "quit"))
BANNED_ATTRS = frozenset((
    "open", "read_text", "read_bytes", "write_text", "write_bytes", "iterdir", "glob", "rglob",
    "touch", "unlink", "rmdir", "mkdir", "symlink_to", "hardlink_to", "load", "loads", "save",
    "from_file", "load_library", "load_inline", "cpp_extension", "hub", "system", "popen",
    "modules", "_getframe", "settrace", "setprofile", "f_globals", "f_locals", "f_back",
    "f_builtins", "gi_frame", "cr_frame", "tb_frame", "__globals__", "__builtins__",
    "__subclasses__", "__bases__", "__base__", "__mro__", "__code__", "__dict__",
    "__getattribute__", "__import__", "__loader__", "__spec__", "__reduce__", "__reduce_ex__",
    "__closure__", "attrgetter", "methodcaller", "exec_module", "_C", "environ", "getenv"))
TARGET_KEYS = ("y_counts", "y_base")
BANNED_STRINGS = BANNED_NAMES | BANNED_ATTRS | frozenset(TARGET_KEYS)
DYN_ATTR = frozenset(("getattr", "setattr", "delattr", "hasattr"))

#: frozen files whose kit copy must equal this tree's (cluster mode)
FROZEN_GLOBS = ("tools/onion_eval/*.py", "tools/minicandi/*.py", "tools/minicandi/rungs/*.py",
                "tools/ladder/model.py", "tools/t146/model.py", "slurm/onion_eval/auto_proxy.sh",
                "slurm/_kit_pin.sh")


class Reject(Exception):
    """The attempt is refused; the message is the reason."""


class TunnelDown(Reject):
    def __init__(self, what: str = ""):
        super().__init__("tunnel down" + (f" ({what})" if what else ""))


def log(*a) -> None:
    print("[auto_proxy]", *a, file=sys.stderr, flush=True)


def result(passed: bool, reason: str, **kw) -> dict:
    out = {"passed": bool(passed), "reason": str(reason),
           "composite": {"value": None}, "imputation": {"value": None},
           "correction": {"value": None}, "uncertainty": {"value": None},
           "seed_values": [], "param_count": None, "flops_per_window": None, "cpu_hours": None,
           "attempt": None}
    for k, v in kw.items():
        out[k] = {"value": v} if k in ("composite", "imputation", "correction", "uncertainty") else v
    return out


# --------------------------------------------------------------------------------- static checks

def load_recipe(recipe: Path) -> tuple[dict, Path]:
    recipe = Path(recipe)
    cp, mp = recipe / "config.json", recipe / "model.py"
    if not recipe.is_dir():
        raise Reject(f"recipe {recipe} is not a directory")
    if not cp.is_file() or not mp.is_file():
        raise Reject(f"recipe {recipe} must hold config.json and model.py")
    try:
        cfg = json.loads(cp.read_text(encoding="utf-8"))
    except ValueError as e:
        raise Reject(f"config.json is not JSON: {e}") from None
    if not isinstance(cfg, dict):
        raise Reject("config.json is not a JSON object")
    return cfg, mp


def _scalar(v) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


def check_config(cfg: dict, recipe: Path, *, local: bool) -> str | None:
    """Raise Reject on a bad config; return the recipe's path relative to the repository root
    (None when the recipe lies outside it, allowed only under --local)."""
    bad = [k for k in FORBIDDEN_KEYS if k in cfg]
    if bad:
        raise Reject(f"forbidden config key(s): {', '.join(bad)} (validation-only scorer)")
    extra = sorted(k for k in cfg if k not in ALLOWED_KEYS)
    if extra:
        raise Reject(f"config keys outside the allowed set: {', '.join(extra)}")
    if "rung" not in cfg:
        raise Reject("config has no 'rung'")
    for k, v in cfg.items():
        if k == "loss_weights":
            if not (isinstance(v, dict) and all(isinstance(x, (int, float)) and not
                                                isinstance(x, bool) for x in v.values())):
                raise Reject("loss_weights must map names to numbers")
        elif k == "mask_prob_range":
            if not (isinstance(v, list) and len(v) == 2 and
                    all(isinstance(x, (int, float)) for x in v)):
                raise Reject("mask_prob_range must be a list of two numbers")
        elif not _scalar(v):
            raise Reject(f"config key {k!r}: value must be a scalar")
        elif isinstance(v, str) and k != "model_file" and ("/" in v or "\\" in v):
            raise Reject(f"config key {k!r}: a string value may not be a path")
    if "windows_seen" in cfg and not (isinstance(cfg["windows_seen"], int)
                                      and not isinstance(cfg["windows_seen"], bool)):
        raise Reject("windows_seen must be an integer (the scorer replaces it with --windows)")
    recipe = Path(recipe).resolve()
    try:
        rel = recipe.relative_to(ROOT).as_posix()
    except ValueError:
        rel = None
    mf = cfg.get("model_file")
    if mf is None:
        if not local:
            raise Reject("config has no model_file (it must point at the recipe's own model.py)")
        return rel
    if not isinstance(mf, str) or not mf:
        raise Reject("model_file must be a path string")
    p = Path(mf)
    if not local and (p.is_absolute() or rel is None):
        raise Reject("model_file must be relative to the repository root, inside it")
    target = (p if p.is_absolute() else ROOT / p).resolve()
    if target != (recipe / "model.py").resolve():
        raise Reject(f"model_file {mf!r} is not the recipe's own model.py")
    if not local and not rel.startswith("tools/minicandi/"):
        raise Reject(f"recipe {rel} must live under tools/minicandi/")
    return rel


def check_module(src: str, name: str = "model.py") -> list[str]:
    """Problems of the candidate module, from its syntax tree (empty = passes)."""
    try:
        tree = ast.parse(src, filename=name)
    except SyntaxError as e:
        return [f"{name}: syntax error: {e}"]
    bad, sys_alias, dyn_ok = [], set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DYN_ATTR:
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) \
                    and isinstance(node.args[1].value, str):
                dyn_ok.add(id(node.func))

    def mod_ok(mod: str) -> bool:
        if mod in STDLIB_OK:
            return True
        if mod == "torch" or mod.startswith("torch."):
            return not any(mod == b or mod.startswith(b + ".") for b in TORCH_BANNED)
        if mod.startswith("minicandi."):
            return mod.split(".", 1)[1] in MINICANDI_OK
        return False

    for node in ast.walk(tree):
        ln = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            for a in node.names:
                if not mod_ok(a.name):
                    bad.append(f"line {ln}: import {a.name} is outside the allowed set")
                if a.name == "sys":
                    sys_alias.add(a.asname or "sys")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:
                bad.append(f"line {ln}: relative import")
            elif mod == "minicandi":
                for a in node.names:
                    if a.name not in MINICANDI_OK:
                        bad.append(f"line {ln}: from minicandi import {a.name} is outside the "
                                   f"allowed set")
            elif not mod_ok(mod):
                bad.append(f"line {ln}: from {mod} import is outside the allowed set")
            elif mod == "sys":
                bad += [f"line {ln}: from sys import {a.name}" for a in node.names
                        if a.name not in SYS_OK]
            for a in node.names:
                if a.name == "*":
                    bad.append(f"line {ln}: star import")
                elif a.name in BANNED_ATTRS or a.name in BANNED_NAMES:
                    bad.append(f"line {ln}: import of {a.name}")
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES:
                bad.append(f"line {ln}: name {node.id}")
            elif node.id in DYN_ATTR and id(node) not in dyn_ok:
                bad.append(f"line {ln}: {node.id} with a non-constant attribute name")
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_ATTRS:
                bad.append(f"line {ln}: attribute .{node.attr}")
            if isinstance(node.value, ast.Name) and node.value.id in sys_alias \
                    and node.attr not in SYS_OK:
                bad.append(f"line {ln}: sys.{node.attr}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in BANNED_STRINGS:
                bad.append(f"line {ln}: string {node.value!r}")
    return bad


# -------------------------------------------------------------------------------- dynamic checks

def _class_file(cls) -> str | None:
    import inspect
    try:
        return str(Path(inspect.getfile(cls)).resolve())
    except (TypeError, OSError):
        pass
    for v in vars(cls).values():
        code = getattr(v, "__code__", None)
        if code is not None:
            return str(Path(code.co_filename).resolve())
    return None


def model_from_file(model, model_py: Path) -> bool:
    want = str(Path(model_py).resolve())
    return any(_class_file(type(m)) == want for m in model.modules())


def perturbed(model, seed: int = 0):
    """A copy with noise on every parameter: an untrained NB head starts at zero, so an untrained
    model ignores its inputs and a wiring check on it would pass vacuously."""
    import torch
    m = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return m.eval()


def _max_rel(a: dict, b: dict) -> float:
    import torch
    worst = 0.0
    if set(a) != set(b):
        return math.inf
    for k in a:
        if isinstance(a[k], torch.Tensor):
            if a[k].shape != b[k].shape:
                return math.inf
            d = ((a[k].double() - b[k].double()).abs() / a[k].double().abs().clamp_min(1e-12))
            if d.numel():
                v = float(d.max())
                worst = max(worst, v if math.isfinite(v) else math.inf)
    return worst


def target_independence(model, batch: dict, seed: int = 0) -> dict:
    """Outputs must not move when the targets change: model(batch), model(y perturbed once),
    model(y perturbed again), model(batch) must all agree to 1e-6 relative. Eval mode, no grad."""
    import torch
    keys = [k for k in TARGET_KEYS if k in batch]
    g = torch.Generator().manual_seed(seed)

    def pert():
        p = dict(batch)
        for k in keys:
            p[k] = torch.randint(0, 500, batch[k].shape, generator=g, dtype=batch[k].dtype)
        return p

    p1, p2 = pert(), pert()
    with torch.no_grad():
        outs = [model(batch), model(p1), model(p2), model(batch)]
    worst = max(_max_rel(outs[0], o) for o in outs[1:])
    return {"max_rel_change": worst, "keys": keys, "pass": bool(worst <= 1e-6)}


def synthetic_batch(cfg: dict, root: Path, L: int) -> dict:
    import numpy as np
    from minicandi import synth
    from minicandi.dataset import OnionDataset
    td = synth.make_synthetic_tables(root, chrom_bins={"chr1": max(2048, 2 * L),
                                                        "chr2": max(1024, L)})
    ds = OnionDataset(td, "train", cfg, 0)
    for s in range(8):
        b = ds.batch(2, np.random.default_rng(s))
        if b["x_bin_mask"].any() and (~b["x_avail"]).any():
            return b
    raise Reject("could not draw a synthetic batch with hidden bins and a missing track")


def check_model(cfg: dict, model_py: Path) -> dict:
    """Build the candidate and run the dynamic checks; returns param_count and flops_per_window."""
    import torch
    from minicandi import contract as C
    from minicandi import registry
    from minicandi import train as T
    full = T.full_config(dict(cfg, windows_seen=int(cfg.get("windows_seen") or 1)))
    torch.manual_seed(0)
    try:
        model = registry.build_model(full)
    except Exception as e:                                   # noqa: BLE001
        raise Reject(f"build_model failed: {type(e).__name__}: {e}") from None
    if cfg.get("model_file") and not model_from_file(model, model_py):
        raise Reject("model_file not honored: no module of the built model comes from the "
                     "recipe's model.py (registry without model_file support?)")
    n = int(model.param_count())
    if abs(n - PARAM_TARGET) > PARAM_TOL * PARAM_TARGET:
        raise Reject(f"param_count {n} is outside 10% of {PARAM_TARGET:.1e}")
    L = int(full["context_bins"])
    with tempfile.TemporaryDirectory(prefix="auto_proxy_synth_") as tmp:
        try:
            batch = synthetic_batch(full, Path(tmp), L)
            pm = perturbed(model)
            w = C.wiring_check(pm, batch)
            ti = target_independence(pm, batch)
            fpw = float(T.flops_per_window(model, batch, torch.device("cpu"))[0])
        except Reject:
            raise
        except Exception as e:                               # noqa: BLE001
            raise Reject(f"dynamic check crashed: {type(e).__name__}: {e}") from None
    if not w["pass"]:
        raise Reject(f"wiring check failed (perturbed head): max_rel_change "
                     f"{w['max_rel_change']:.3g} > 1e-6")
    if not ti["pass"]:
        raise Reject(f"target independence failed: outputs move by {ti['max_rel_change']:.3g} "
                     f"when {'/'.join(ti['keys'])} change")
    return {"param_count": n, "flops_per_window": fpw}


# ----------------------------------------------------------------------------- inputs (job side)

def _tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def check_inputs(tables, evalset, *, cluster: bool = False) -> None:
    """Refuse an eval set that could read a blind or non-validation row."""
    from minicandi.dataset import tables_sha256
    from onion_eval import evalset as E
    tables, evalset = Path(tables), Path(evalset)
    if cluster and tables.name != CLUSTER_TABLES_NAME:
        raise Reject(f"tables {tables} are not the {CLUSTER_TABLES_NAME} tables")
    if "blind" in evalset.name:
        raise Reject(f"eval set {evalset.name} is a blind eval set")
    try:
        ev = json.loads(evalset.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Reject(f"eval set {evalset} unreadable: {e}") from None
    meas = {r["measurement_id"]: r for r in _tsv(tables / "measurements.tsv")}
    bad = []
    if ev.get("include_blind"):
        bad.append("eval set was built with include_blind")
    if ev.get("tables_sha256") != tables_sha256(tables):
        bad.append("eval set tables_sha256 differs from the tables")

    def role(m):
        return meas.get(m, {}).get("in_eic", "missing")

    for t in ev.get("imputation_targets", []) + ev.get("control_targets", []):
        if t.get("split") != "val" or role(t.get("target")) != "val":
            bad.append(f"target {t.get('target')} is not a validation track")
        bad += [f"input {m} is not a training track" for m in t.get("inputs", [])
                if role(m) != "train"]
    for p in ev.get("correction_pairs", []):
        if p.get("split") != "val" or role(p.get("target")) != "val" \
                or role(p.get("source")) not in ("train", "val"):
            bad.append(f"correction pair {p.get('source')} -> {p.get('target')} is not "
                       f"validation-only")
    want = set(ev.get("replicate_pairs", []))
    if want:
        pairs = {p["pair_id"]: p for p in _tsv(tables / "pairs.tsv")}
        for pid in sorted(want):
            p = pairs.get(pid)
            if p is None or p.get("split") == "blind" or "blind" in (role(p["a"]), role(p["b"])):
                bad.append(f"replicate pair {pid} is blind or unknown")
    with contextlib.redirect_stdout(sys.stderr):
        bad += E.check(evalset, tables, allow_blind=False)
    if bad:
        raise Reject(f"eval set refused ({len(bad)} problems): " + "; ".join(bad[:4]))


def run_seed(cfg: dict, tables, evalset, seed: int, windows: int, out, *, model_py=None,
             cluster: bool = False, device: str = "cpu", workers: int = 0, n_boot: int = 200,
             guardrail_windows: int = 256) -> dict:
    """Train one seed and score it on validation; writes <out>/PROXY.json and returns it."""
    import numpy as np
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        check_inputs(tables, evalset, cluster=cluster)
        from minicandi import registry
        from minicandi import train as T
        from minicandi.dataset import OnionDataset
        from onion_eval import score as S
        eff = dict(cfg, windows_seen=int(windows))
        (out / "proxy_config.json").write_text(json.dumps(eff, indent=1), encoding="utf-8")
        info = T.train(eff, tables, int(seed), out, device=device, workers=workers)
        if int(info.get("windows_seen", -1)) < int(windows):
            raise Reject(f"trained {info.get('windows_seen')} windows, budget {windows}")
        model, conf = registry.load_run(out)
        if model_py is not None and cfg.get("model_file") and not model_from_file(model, model_py):
            raise Reject("model_file not honored by load_run")
        n = int(model.param_count())
        if abs(n - PARAM_TARGET) > PARAM_TOL * PARAM_TARGET:
            raise Reject(f"trained param_count {n} is outside 10% of {PARAM_TARGET:.1e}")
        ds = OnionDataset(tables, "train", conf, int(seed) + 7919)
        for k in range(2):
            ti = target_independence(model, ds.batch(2, np.random.default_rng([int(seed), k])), k)
            if not ti["pass"]:
                raise Reject(f"trained model is not target-independent: outputs move by "
                             f"{ti['max_rel_change']:.3g}")
        S.score(evalset, out, run=out, split="val", device=device, n_boot=n_boot,
                guardrail_windows=guardrail_windows, log=log)
        res = {"status": "ok", "reason": ""}
    except Reject as e:
        res = {"status": "refused", "reason": str(e)}
    except Exception as e:                                   # noqa: BLE001
        res = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
    res.update(seed=int(seed), wall_seconds=time.time() - t0)
    (out / PROXY_FILE).write_text(json.dumps(res), encoding="utf-8")
    return res


# ------------------------------------------------------------------------------------- composite

def collect(seed_dirs: list[Path], windows: int) -> dict:
    """Composite and goal terms from each seed's PROXY.json, run.json and scores.json."""
    from onion_eval import select_rule as SR
    seeds, fpw = [], None
    for d in seed_dirs:
        d = Path(d)
        pf = d / PROXY_FILE
        if not pf.is_file():
            raise Reject(f"{d.name}: no {PROXY_FILE} (the job died before it finished)")
        pr = json.loads(pf.read_text(encoding="utf-8"))
        if pr.get("status") != "ok":
            raise Reject(f"{d.name}: {pr.get('status')}: {pr.get('reason')}")
        for f in ("run.json", "scores.json"):
            if not (d / f).is_file():
                raise Reject(f"{d.name}: no {f}")
        run = json.loads((d / "run.json").read_text(encoding="utf-8"))
        if int(run.get("windows_seen", -1)) < windows:
            raise Reject(f"{d.name}: trained {run.get('windows_seen')} windows, budget {windows}")
        n = int(run.get("param_count", -1))
        if abs(n - PARAM_TARGET) > PARAM_TOL * PARAM_TARGET:
            raise Reject(f"{d.name}: param_count {n} is outside 10% of {PARAM_TARGET:.1e}")
        fpw = run.get("flops_per_window") if fpw is None else fpw
        seeds.append(SR.load_seed(d / "scores.json"))
    try:
        SR.check_comparable(seeds, "val")
    except SR.Refused as e:
        raise Reject(str(e)) from None
    for d, s in zip(seed_dirs, seeds):
        if s.get("copy_leak_part1") is not True:
            raise Reject(f"{Path(d).name}: wiring check failed on the trained model "
                         f"(copy_leak.part1_wiring_pass = {s.get('copy_leak_part1')})")
    problems, sc = SR.screen_terms(seeds, clip=True)
    if problems:
        raise Reject("goal terms refused: " + "; ".join(m for _, m in problems[:3]))
    for d, s in zip(seed_dirs, sc):
        if s["composite"] is None:
            why = ", ".join(f"{k}: {s['goal_reasons'].get(k) or 'undefined'}"
                            for k in SR.GOALS if s["goals"][k] is None)
            raise Reject(f"{Path(d).name}: composite undefined ({why})")
    comps = [float(s["composite"]) for s in sc]
    goals = {k: sum(float(s["goals"][k]) for s in sc) / len(sc) for k in SR.GOALS}
    return {"composite": sum(comps) / len(comps), "seed_values": comps, "goals": goals,
            "flops_per_window": fpw}


# ----------------------------------------------------------------------------------------- tally

def parse_tally(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        f = line.rstrip("\n").split("\t")
        if not line.strip() or f[0] == "time" or len(f) < len(TALLY_COLUMNS):
            continue
        r = dict(zip(TALLY_COLUMNS, f))
        try:
            r["cpu_hours"] = float(r["cpu_hours"])
        except ValueError:
            continue
        rows.append(r)
    return rows


def tally_line(attempt: str, seed, job_id, cpus, wall: float, rc) -> str:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return "\t".join(map(str, (now, attempt, seed, job_id, cpus, f"{wall:.0f}",
                               f"{wall * float(cpus) / 3600:.4f}", rc))) + "\n"


def cap_cpuh() -> float:
    v = os.environ.get(CAP_ENV, "")
    try:
        return float(v) if v.strip() else DEFAULT_CAP_CPUH
    except ValueError:
        raise Reject(f"{CAP_ENV}={v!r} is not a number") from None


def budget_check(rows: list[dict], cap: float, when: str) -> float:
    total = sum(r["cpu_hours"] for r in rows)
    if total >= cap:
        raise Reject(f"budget exhausted: the tally holds {total:.1f} CPU-h {when}, cap {cap:g} "
                     f"({CAP_ENV})")
    return total


def append_local_tally(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8") as fh:
        if new:
            fh.write("\t".join(TALLY_COLUMNS) + "\n")
        fh.write(line)


# --------------------------------------------------------------------------------------- cluster

def hpc(cluster: str, cmd: str, stdin: bytes | None = None, timeout: float = 300) -> str:
    exe = os.environ.get("ONION_HPC") or "hpc"
    try:
        p = subprocess.run([exe, "run", cluster, cmd], capture_output=True, timeout=timeout,
                           **({"input": stdin} if stdin is not None
                              else {"stdin": subprocess.DEVNULL}))
    except FileNotFoundError:
        raise Reject(f"no {exe} command on this machine") from None
    except subprocess.TimeoutExpired:
        raise Reject(f"hpc run {cluster} timed out after {timeout:.0f} s") from None
    if p.returncode == TUNNEL_DOWN:
        raise TunnelDown(f"hpc exit 69 on {cluster}: run `hpc up {cluster}`")
    if p.returncode == 255:
        raise TunnelDown(f"ssh exit 255 on {cluster}")
    if p.returncode != 0:
        err = p.stderr.decode(errors="replace").strip().splitlines()[-3:]
        raise Reject(f"remote command failed (exit {p.returncode}): {' | '.join(err)}")
    return p.stdout.decode(errors="replace")


def frozen_files() -> list[str]:
    out = []
    for g in FROZEN_GLOBS:
        out += sorted(p.relative_to(ROOT).as_posix() for p in ROOT.glob(g) if p.is_file())
    return out


def check_kit(cluster: str, kit: str) -> None:
    files = frozen_files()
    want = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}
    cmd = f"cd {shlex.quote(kit)} && sha256sum -- " + " ".join(shlex.quote(f) for f in files) \
        + " 2>/dev/null; true"
    got = {}
    for line in hpc(cluster, cmd).splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            got[parts[1].strip().lstrip("*")] = parts[0]
    diff = [f for f in files if got.get(f) != want[f]]
    if diff:
        raise Reject(f"kit {kit} differs from the scorer's tree in {len(diff)} frozen file(s): "
                     + ", ".join(diff[:4]))


def exclude_nodes(cluster: str) -> str:
    for p in (ROOT / ".orchestrate" / f"exclude_{cluster}", UMBRELLA_ORCH / f"exclude_{cluster}"):
        if p.is_file():
            txt = re.sub(r"#.*", "", p.read_text(encoding="utf-8"))
            nodes = [n for n in re.split(r"[,\s]+", txt) if n]
            if all(re.fullmatch(r"[A-Za-z0-9._-]+", n) for n in nodes):
                return ",".join(dict.fromkeys(nodes))
    return ""


def sbatch_command(*, kit: str, att: str, seed: int, windows: int, tables: str, evalset: str,
                   recipe_rel: str, tally: str, job_time: str, key: str, exclude: str = "") -> str:
    q = shlex.quote
    opts = [f"--account=def-maxwl", f"--job-name=evo_{key[-12:]}_s{seed}", f"--time={job_time}",
            f"--cpus-per-task={JOB_CPUS}", f"--mem={JOB_MEM_MB}M", "--gres=none",
            f"--output={att}/seed{seed}.log"]
    if exclude:
        opts.append(f"--exclude={exclude}")
    args = [f"{kit}/slurm/onion_eval/auto_proxy.sh", kit, att, str(seed), str(windows), tables,
            evalset, recipe_rel, tally]
    return "sbatch --parsable " + " ".join(q(o) for o in opts) + " " + " ".join(q(a) for a in args)


TERMINAL = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE")


def parse_sacct(text: str) -> dict:
    """{job id: state} from `sacct -n -X -P -o JobID,State` (the first word of the state)."""
    st = {}
    for line in text.splitlines():
        f = line.strip().split("|")
        if len(f) >= 2 and f[0].isdigit():
            st[f[0]] = f[1].split()[0] if f[1].split() else ""
    return st


class Jobs:
    """Submitted job ids; cancelled on a signal or a timeout."""

    def __init__(self, cluster: str):
        self.cluster, self.ids = cluster, []

    def cancel(self) -> None:
        if self.ids:
            try:
                hpc(self.cluster, "scancel " + " ".join(self.ids), timeout=60)
                log(f"cancelled {' '.join(self.ids)}")
            except Reject as e:
                log(f"scancel failed: {e}")


def run_cluster(a, cfg: dict, recipe: Path, recipe_rel: str, key: str, seeds: list[int],
                workspace: Path) -> tuple[list[Path], float]:
    cl, q = a.cluster, shlex.quote
    if Path(a.tables).name != CLUSTER_TABLES_NAME:
        raise Reject(f"tables {a.tables} are not the {CLUSTER_TABLES_NAME} tables")
    if "blind" in Path(a.evalset).name:
        raise Reject(f"eval set {a.evalset} is a blind eval set")
    kit = f"{KITS}/{a.kit}"
    tally = f"{a.runs_root}/evolve/TALLY.tsv"
    att = f"{a.runs_root}/evolve/attempts/{key}"
    cap = cap_cpuh()
    budget_check(parse_tally(hpc(cl, f"cat {q(tally)} 2>/dev/null; true")), cap,
                 "before this attempt")
    check_kit(cl, kit)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for f in ("config.json", "model.py"):
            tf.add(str(recipe / f), arcname=f)
    hpc(cl, f"mkdir -p {q(att + '/recipe')} && tar -xf - -C {q(att + '/recipe')}",
        stdin=buf.getvalue())
    jobs = Jobs(cl)

    def on_signal(signum, frame):
        jobs.cancel()
        raise Reject(f"scorer stopped by signal {signum}; jobs cancelled")

    old = {s: signal.signal(s, on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        excl = exclude_nodes(cl)
        for s in seeds:
            out = hpc(cl, sbatch_command(kit=kit, att=att, seed=s, windows=a.windows,
                                         tables=a.tables, evalset=a.evalset,
                                         recipe_rel=recipe_rel, tally=tally,
                                         job_time=a.job_time, key=key, exclude=excl))
            jid = out.strip().splitlines()[-1].split(";")[0] if out.strip() else ""
            if not jid.isdigit():
                raise Reject(f"sbatch returned no job id: {out.strip()[-200:]!r}")
            jobs.ids.append(jid)
            log(f"seed {s}: job {jid} on {cl}, {att}/seed{s}")
        deadline = time.monotonic() + 60 * a.max_wait_minutes
        states = {}
        while True:
            states = parse_sacct(hpc(cl, "sacct -n -X -P -o JobID,State -j "
                                     + ",".join(jobs.ids)))
            if all(states.get(j, "") in TERMINAL for j in jobs.ids):
                break
            if time.monotonic() > deadline:
                jobs.cancel()
                raise Reject(f"timed out after {a.max_wait_minutes} min waiting for jobs "
                             f"{' '.join(jobs.ids)} ({states})")
            time.sleep(a.poll_seconds)
        log(f"jobs ended: {states}")
        # the attempt's CPU-h: the jobs' own rows, else sacct
        rows = [r for r in parse_tally(hpc(cl, f"cat {q(tally)} 2>/dev/null; true"))
                if r["attempt"] == key]
        have = {r["job_id"] for r in rows}
        for s, j in zip(seeds, jobs.ids):
            if j in have:
                continue
            f = (hpc(cl, f"sacct -n -X -P -o ElapsedRaw,AllocCPUS -j {j}").strip().splitlines()
                 or ["0|0"])[0].split("|") + ["0"]
            try:
                wall, cpus = float(f[0] or 0), int(f[1] or 0)
            except ValueError:
                wall, cpus = 0.0, JOB_CPUS
                log(f"sacct gave no cost for job {j}: {f}; tallied as 0 CPU-h")
            line = tally_line(key, s, j, cpus, wall, "sacct")
            hpc(cl, f"mkdir -p {q(os.path.dirname(tally))} && "
                    f"( flock 9; cat >> {q(tally)} ) 9>>{q(tally + '.lock')}",
                stdin=line.encode())
            rows += parse_tally(line)
        cpu_hours = sum(r["cpu_hours"] for r in rows)
        dirs = []
        for s, j in zip(seeds, jobs.ids):
            d = workspace / f"seed{s}"
            d.mkdir(parents=True, exist_ok=True)
            for f in (PROXY_FILE, "run.json", "scores.json"):
                txt = hpc(cl, f"cat {q(f'{att}/seed{s}/{f}')} 2>/dev/null; true")
                if txt.strip():
                    (d / f).write_text(txt, encoding="utf-8")
            if not (d / PROXY_FILE).is_file():
                (d / PROXY_FILE).write_text(json.dumps(
                    {"status": "error", "reason": f"job {j} ended {states.get(j)} with no "
                     f"{PROXY_FILE}; log {att}/seed{s}.log"}), encoding="utf-8")
            dirs.append(d)
        return dirs, cpu_hours
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def run_local(a, cfg: dict, model_py: Path, key: str, seeds: list[int]) -> tuple[list[Path], float]:
    import torch
    runs = Path(a.runs_root)
    tally = runs / "evolve" / "TALLY.tsv"
    budget_check(parse_tally(tally.read_text(encoding="utf-8")) if tally.is_file() else [],
                 cap_cpuh(), "before this attempt")
    att = runs / "evolve" / "attempts" / key
    dirs, cpu_hours = [], 0.0
    for s in seeds:
        d = att / f"seed{s}"
        res = run_seed(cfg, a.tables, a.evalset, s, a.windows, d, model_py=model_py,
                       n_boot=a.n_boot, guardrail_windows=a.guardrail_windows)
        line = tally_line(key, s, "local", torch.get_num_threads(), res["wall_seconds"],
                          0 if res["status"] == "ok" else 1)
        append_local_tally(tally, line)
        cpu_hours += parse_tally(line)[0]["cpu_hours"]
        dirs.append(d)
        if res["status"] != "ok":
            break
    return dirs, cpu_hours


# ------------------------------------------------------------------------------------------ main

def attempt_key(recipe: Path) -> str:
    h = hashlib.sha256()
    for f in ("config.json", "model.py"):
        h.update((recipe / f).read_bytes())
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    n = re.sub(r"[^0-9A-Za-z]", "", os.environ.get("CRUX_ATTEMPT", ""))[:12]
    return f"{stamp}_{h.hexdigest()[:12]}" + (f"_a{n}" if n else "")


def parse_seeds(text: str) -> list[int]:
    try:
        seeds = [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise Reject(f"--seeds {text!r}: integers separated by commas") from None
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise Reject(f"--seeds {text!r}: at least one distinct non-negative seed")
    return seeds


def score_candidate(a) -> dict:
    seeds = parse_seeds(a.seeds)
    if a.windows <= 0:
        raise Reject("--windows must be positive")
    recipe = Path(a.recipe)
    cfg, model_py = load_recipe(recipe)
    recipe_rel = check_config(cfg, recipe, local=a.local)
    problems = check_module(model_py.read_text(encoding="utf-8"))
    if problems:
        raise Reject(f"model.py refused ({len(problems)} problems): " + "; ".join(problems[:4]))
    stat = check_model(cfg, model_py)
    log(f"static and dynamic checks passed: param_count {stat['param_count']}, "
        f"flops_per_window {stat['flops_per_window']:.3g}")
    key = attempt_key(recipe)
    ws = Path(os.environ.get("CRUX_WORKSPACE") or tempfile.mkdtemp(prefix="auto_proxy_"))
    if a.local:
        if not a.runs_root:
            raise Reject("--local needs --runs-root (or CRUX_WORKSPACE)")
        dirs, cpu_hours = run_local(a, cfg, model_py, key, seeds)
        att = str(Path(a.runs_root) / "evolve" / "attempts" / key)
        tally_rows = parse_tally((Path(a.runs_root) / "evolve" / "TALLY.tsv").read_text())
    else:
        dirs, cpu_hours = run_cluster(a, cfg, recipe, recipe_rel, key, seeds, ws / key)
        att = f"{a.runs_root}/evolve/attempts/{key}"
        tally_rows = parse_tally(hpc(a.cluster, f"cat {shlex.quote(a.runs_root + '/evolve/TALLY.tsv')}"
                                     f" 2>/dev/null; true"))
    base = dict(param_count=stat["param_count"], flops_per_window=stat["flops_per_window"],
                cpu_hours=round(cpu_hours, 4), attempt=att)
    try:
        c = collect(dirs, a.windows)          # a failed seed raises here, with its reason
        if len(dirs) != len(seeds):
            raise Reject("not every seed ran")
        budget_check(tally_rows, cap_cpuh(), "after this attempt")
    except Reject as e:
        return result(False, str(e), **base)
    base["flops_per_window"] = c["flops_per_window"] or base["flops_per_window"]
    g = c["goals"]
    return result(True, f"{len(seeds)} seeds scored on validation", composite=c["composite"],
                  imputation=g["imputation"], correction=g["correction"],
                  uncertainty=g["uncertainty"], seed_values=c["seed_values"], **base)


def scorer_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="auto_proxy.py", description=__doc__.splitlines()[0])
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--windows", type=int, default=20000)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--tables", default=DEFAULT_TABLES)
    ap.add_argument("--evalset", default=DEFAULT_EVALSET)
    ap.add_argument("--cluster", default="nibi", choices=("nibi", "fir"))
    ap.add_argument("--kit", default="main", help="kit id under /project/def-maxwl/mforooz/onion_kit")
    ap.add_argument("--runs-root", default=None,
                    help=f"default: {DEFAULT_RUNS} (cluster) or $CRUX_WORKSPACE (--local)")
    ap.add_argument("--local", action="store_true",
                    help="train and score in-process on the given tables (laptop tests)")
    ap.add_argument("--job-time", default="02:00:00")
    ap.add_argument("--max-wait-minutes", type=float, default=150.0)
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    ap.add_argument("--n-boot", type=int, default=200, help="--local only (cluster: score.py's)")
    ap.add_argument("--guardrail-windows", type=int, default=256,
                    help="--local only (cluster: score.py's)")
    return ap


def job_main(argv) -> int:
    ap = argparse.ArgumentParser(prog="auto_proxy.py job")
    ap.add_argument("--recipe", required=True, help="the recipe dir inside the job's tree")
    ap.add_argument("--tables", required=True)
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--windows", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args(argv)
    try:
        cfg, model_py = load_recipe(Path(a.recipe))
        # the scorer already required model_file; here only its target is re-checked
        check_config(cfg, Path(a.recipe), local=True)
        problems = check_module(model_py.read_text(encoding="utf-8"))
        if problems:
            raise Reject("model.py refused: " + "; ".join(problems[:4]))
    except Reject as e:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / PROXY_FILE).write_text(json.dumps({"status": "refused", "reason": str(e),
                                                          "seed": a.seed}), encoding="utf-8")
        log(f"refused: {e}")
        return 1
    from onion_eval.score import set_threads
    set_threads()
    with contextlib.redirect_stdout(sys.stderr):
        res = run_seed(cfg, a.tables, a.evalset, a.seed, a.windows, a.out, model_py=model_py,
                       cluster=True, workers=a.workers)
    log(f"seed {a.seed}: {res['status']} {res['reason']}")
    return 0 if res["status"] == "ok" else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["job"]:
        return job_main(argv[1:])
    real = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        try:
            a = scorer_parser().parse_args(argv)
            if a.runs_root is None:
                a.runs_root = os.environ.get("CRUX_WORKSPACE") if a.local else DEFAULT_RUNS
            res = score_candidate(a)
        except SystemExit as e:                              # argparse
            res = result(False, f"bad arguments (exit {e.code})")
        except Reject as e:
            res = result(False, str(e))
        except Exception as e:                               # noqa: BLE001
            res = result(False, f"scorer error: {type(e).__name__}: {e}")
    real.write(json.dumps(res) + "\n")
    real.flush()
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
