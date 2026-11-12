"""miniCANDI model registry (plan pinned interface C): `build_model(config)` and `load_run(run_dir)`.

Registration. A rung is a module `tools/minicandi/rungs/<name>.py` that defines

    class Model(contract.MiniCandi):
        def __init__(self, config: dict): ...

and nothing else is required: no decorator, no call into this file. The registry imports the
module lazily, by name, the first time a config asks for it. `config["rung"]` is either the rung
number (1 to 6, mapped through `RUNGS`) or the module name itself, so a new module dropped into
`rungs/` is buildable at once as `{"rung": "<name>", ...}`. `available()` lists every module in
`rungs/` that defines `Model`.

Wrappers (plan pinned interface E). `build_model(config)` builds the model class with `two_step`
and `control_head` removed from the config, then applies `minicandi.control_head.wrap(model,
config)` when `config["control_head"]` is set and not `"channel"`, then
`minicandi.two_step.wrap(model, config)` when `config["two_step"]` is true (lazy imports, this
order). A rung's own `Model(config)` still raises NotApplicable for `two_step`.

`config["model_file"]` (a path relative to the repository root, or absolute; default absent) makes
`build_model` load `class Model` from that file with importlib in place of the rung module named
by `config["rung"]` (the mix-and-match hybrids and the evolved candidates use it). The file is
executed once per resolved path and process, as module `minicandi_model_file.<stem>_<hash>`.

Extra inputs (Amendment 3, chunk XI). `loss_assay_weight = learned` registers the parameter
`assay_log_w` float32 [A] (zeros; no RNG drawn) on the outermost model and a forward hook that
adds it to the output as `out["assay_log_w"]`; `losses.total_loss` reads it. `aug_revcomp`
together with `dna = precomputed` is refused (`check_extra_config`). The rung keys
`rna_track`, `cell_vector`, `cell_embedding` act inside the rung (`contract.MiniCandi`).

Resolution (chunk RM): `res_bp` != 25 and `res_in_sub` / `res_out_sub` > 1 build in rungs 2 and 4
only (else NotApplicable); `load_run` wraps a run of `res_bp` < 25 or `res_in_sub` > 1 in
`res_adapter.ResAdapter`, which presents it to the scorer at 25 bp.

`load_run(run_dir)` is the only loader the scorer (E6) and the guardrail (E4) use. It reads
`<run_dir>/ckpt.pt` (the pinned dict, `contract.check_ckpt`), rebuilds the model from the saved
config with the RNG forked (so loading never moves the caller's RNG), loads the state dict and
returns `(model, config)` with the model in eval mode on `map_location`. The saved config holds
the wrapper keys and `model_file`, so `load_run` rebuilds the same wrapped model.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path

import torch

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from minicandi import contract as C  # noqa: E402

RUNGS = {1: "pointwise", 2: "conv_ae", 3: "conv_attn_ae", 4: "unet", 5: "axial", 6: "perceiver"}
RUNGS_DIR = Path(__file__).resolve().parent / "rungs"
PACKAGE = "minicandi.rungs"
CKPT_NAME = "ckpt.pt"
REPO = _TOOLS.parent
#: config keys that select a wrapper; the inner model is built without them
WRAPPER_KEYS = ("two_step", "control_head")
#: `loss_assay_weight` values (pinned interface E); `learned` adds the parameter `assay_log_w`
LOSS_ASSAY_WEIGHTS = (None, "none", "equal", "learned")
_MODEL_FILES: dict = {}


def module_name(rung) -> str:
    """The module name of `config["rung"]`: an int 1..6 (or its string form) or a module name."""
    if isinstance(rung, bool):
        raise ValueError(f"rung {rung!r} is not a rung number or a module name")
    if isinstance(rung, int) or (isinstance(rung, str) and rung.strip().isdigit()):
        r = int(rung)
        if r not in RUNGS:
            raise ValueError(f"rung {r} not in {sorted(RUNGS)}")
        return RUNGS[r]
    if isinstance(rung, str) and rung.isidentifier():
        return rung
    raise ValueError(f"rung {rung!r} is not a rung number or a module name")


def rung_module(rung):
    """Import `minicandi.rungs.<name>` lazily and return it; it must define `Model`."""
    name = module_name(rung)
    if not (RUNGS_DIR / f"{name}.py").is_file():
        raise ModuleNotFoundError(f"no rung module {RUNGS_DIR / (name + '.py')}")
    mod = importlib.import_module(f"{PACKAGE}.{name}")
    if not isinstance(getattr(mod, "Model", None), type):
        raise TypeError(f"{PACKAGE}.{name} defines no class Model")
    return mod


def available() -> list[str]:
    """Every module in `rungs/` whose source defines `class Model` (not imported)."""
    out = []
    for p in sorted(RUNGS_DIR.glob("*.py")):
        if p.stem == "__init__":
            continue
        if "class Model" in p.read_text(encoding="utf-8"):
            out.append(p.stem)
    return out


def model_file_module(model_file):
    """Import the file `model_file` (relative to the repository root, or absolute) once and return
    it; it must define `class Model`."""
    path = Path(model_file)
    if not path.is_absolute():
        path = REPO / path
    path = path.resolve()
    if not path.is_file():
        raise ModuleNotFoundError(f"model_file {model_file!r}: no file {path}")
    key = str(path)
    if key not in _MODEL_FILES:
        tag = hashlib.sha256(key.encode()).hexdigest()[:12]
        name = f"minicandi_model_file.{path.stem}_{tag}"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        _MODEL_FILES[key] = mod
    mod = _MODEL_FILES[key]
    if not isinstance(getattr(mod, "Model", None), type):
        raise TypeError(f"model_file {path} defines no class Model")
    return mod


def model_class(config: dict) -> type:
    """`class Model` of `config["model_file"]` when set, else of the rung module."""
    if config.get("model_file"):
        return model_file_module(config["model_file"]).Model
    if "rung" not in config:
        raise KeyError("config has no 'rung'")
    return rung_module(config["rung"]).Model


def build_model(config: dict) -> C.MiniCandi:
    """Build the model that `config` names, then its wrappers (module docstring). Seed the RNG
    before the call (AGENTS.md invariant 10); this function draws nothing itself."""
    if "rung" not in config and not config.get("model_file"):
        raise KeyError("config has no 'rung'")
    check_extra_config(config)
    from minicandi.components import NotApplicable, res_bp_of, res_subs_of
    if (res_bp_of(config) != C.RES or max(res_subs_of(config)) > 1) and (
            config.get("model_file") or module_name(config["rung"]) not in ("conv_ae", "unet")):
        raise NotApplicable("res_bp, res_in_sub and res_out_sub are built in rungs 2 and 4 only")
    inner_cfg = {k: v for k, v in config.items() if k not in WRAPPER_KEYS}
    model = model_class(config)(inner_cfg)
    if not isinstance(model, C.MiniCandi):
        raise TypeError(f"{type(model).__name__} is not a contract.MiniCandi")
    if config.get("control_head") not in (None, "channel"):
        from minicandi import control_head as CH
        model = CH.wrap(model, dict(config))
    if config.get("two_step"):
        from minicandi import two_step as TS
        model = TS.wrap(model, dict(config))
    if config.get("loss_assay_weight", "none") == "learned":
        model.register_parameter("assay_log_w", torch.nn.Parameter(torch.zeros(C.A)))
        model.register_forward_hook(_emit_assay_log_w)
    return model


def _emit_assay_log_w(module, inputs, out):
    """Forward hook under `loss_assay_weight = learned`: the output also carries the learned log
    weight per track, `out["assay_log_w"]` float32 [A] (pinned interface E)."""
    return {**out, "assay_log_w": module.assay_log_w}


def check_extra_config(config: dict) -> None:
    """Refuse the extra-input key combinations no rung can build (chunk XI)."""
    w = config.get("loss_assay_weight", "none")
    if w not in LOSS_ASSAY_WEIGHTS:
        raise ValueError(f"loss_assay_weight {w!r} not in {LOSS_ASSAY_WEIGHTS}")
    if config.get("aug_revcomp") and config.get("dna") == "precomputed":
        raise ValueError("aug_revcomp with dna = precomputed: the reversal flips the frozen "
                         "embedding rows, and a flipped embedding is not the embedding of the "
                         "reverse strand; use dna = own")


def load_run(run_dir, map_location="cpu") -> tuple[C.MiniCandi, dict]:
    """(model, config) from `<run_dir>/ckpt.pt`; the model is in eval mode on `map_location`."""
    path = Path(run_dir) / CKPT_NAME
    ck = torch.load(path, map_location=map_location, weights_only=False)
    C.check_ckpt(ck)
    config = dict(ck["config"])
    with torch.random.fork_rng(devices=[]):
        model = build_model(config)
    model.load_state_dict(ck["state_dict"])
    if model.param_count() != int(ck["param_count"]):
        raise ValueError(f"{path}: param_count {model.param_count()} != saved "
                         f"{ck['param_count']}")
    if config.get("res_bp", C.RES) < C.RES or config.get("res_in_sub", 1) > 1:
        from minicandi.res_adapter import ResAdapter     # chunk RM: scored at 25 bp
        model = ResAdapter(model, config, run_dir)
    model.to(map_location)
    model.eval()
    return model, config


__all__ = ["CKPT_NAME", "LOSS_ASSAY_WEIGHTS", "REPO", "RUNGS", "RUNGS_DIR", "WRAPPER_KEYS",
           "available", "build_model", "check_extra_config",
           "load_run", "model_class", "model_file_module", "module_name", "rung_module"]
