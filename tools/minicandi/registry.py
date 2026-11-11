"""miniCANDI model registry (plan pinned interface C): `build_model(config)` and `load_run(run_dir)`.

Registration. A rung is a module `tools/minicandi/rungs/<name>.py` that defines

    class Model(contract.MiniCandi):
        def __init__(self, config: dict): ...

and nothing else is required: no decorator, no call into this file. The registry imports the
module lazily, by name, the first time a config asks for it. `config["rung"]` is either the rung
number (1 to 6, mapped through `RUNGS`) or the module name itself, so a new module dropped into
`rungs/` is buildable at once as `{"rung": "<name>", ...}`. `available()` lists every module in
`rungs/` that defines `Model`.

`load_run(run_dir)` is the only loader the scorer (E6) and the guardrail (E4) use. It reads
`<run_dir>/ckpt.pt` (the pinned dict, `contract.check_ckpt`), rebuilds the model from the saved
config with the RNG forked (so loading never moves the caller's RNG), loads the state dict and
returns `(model, config)` with the model in eval mode on `map_location`.
"""
from __future__ import annotations

import importlib
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


def build_model(config: dict) -> C.MiniCandi:
    """Build the rung that `config["rung"]` names, from `config`. Seed the RNG before the call
    (AGENTS.md invariant 10); this function draws nothing itself."""
    if "rung" not in config:
        raise KeyError("config has no 'rung'")
    model = rung_module(config["rung"]).Model(dict(config))
    if not isinstance(model, C.MiniCandi):
        raise TypeError(f"{type(model).__name__} is not a contract.MiniCandi")
    return model


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
    model.to(map_location)
    model.eval()
    return model, config


__all__ = ["CKPT_NAME", "RUNGS", "RUNGS_DIR", "available", "build_model", "load_run",
           "module_name", "rung_module"]
