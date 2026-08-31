"""Their Keras `.h5` into our torch module. `h5py` only — **no TensorFlow anywhere**.

This is the primary parity anchor of `RIVALS_PLAN.md` §7.4 (PI amendment, 2026-08-25). Their 23
pretrained models (Synapse `syn21519009`, one per chromosome) are plain Keras 2.2.4 archives; a
Keras `.h5` is an ordinary HDF5 file with the weights under `model_weights/`, so reading them
needs no part of their runtime. Predicting with their loaded weights and comparing against their
submitted tracks is the gate. The TF 1.15 environment on Fir exists only to localise a failure of
that gate, and is used for nothing else.

Layout, read off a real checkpoint (`cruxvault/results/t53/SPIKE_MEMO.md` §3):

```
/                         attrs: keras_version=b'2.2.4', backend=b'tensorflow', model_config
  model_weights/          attrs: layer_names = [b'celltype_input', …, b'add_1']
    dense_1/              attrs: weight_names = [b'dense_1/kernel:0', b'dense_1/bias:0']
      dense_1/kernel:0    (225, 2048) float32
      dense_1/bias:0      (2048,)     float32
  optimizer_weights/      Adam state — ignored; it is why an h5 is ~3x the weight count
```

The weight name repeats the layer name, so a tensor is `f["model_weights"][layer][name]`.

Four conversions, none of them the identity:

| Keras | torch | conversion |
|---|---|---|
| `<dense>/kernel:0` `(in, out)` | `Linear.weight` `(out, in)` | **transpose** |
| `<dense>/bias:0` | `Linear.bias` | as-is |
| `<prelu>/alpha:0` `(units,)` | `PReLU.weight` | as-is |
| `<bn>/gamma:0`, `beta:0`, `moving_mean:0`, `moving_variance:0` | `weight`, `bias`, `running_mean`, `running_var` | as-is |
| `<embedding>/embeddings:0` | `Embedding.weight` | as-is |

The transpose is the one that fails silently when both dimensions are 2048 (`dense_2`), which is
why `load_keras_h5` checks every shape rather than trusting the name.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
from torch import nn

from .model import Guacamole, Precamole

__all__ = ["KerasWeightError", "keras_layer_names", "read_layer_weights", "load_keras_h5"]


class KerasWeightError(ValueError):
    """A checkpoint that does not map onto the module. Always names the offending tensor."""


#: torch attribute path -> the Keras layer that supplies it. The stage-2 (`Guacamole`) map; the
#: stage-1 (`Precamole`) map is this one minus `block3`/`y_pred`, plus `y_pred_dense`.
_GUACAMOLE_LAYERS: Dict[str, str] = {
    "factors.celltype_embedding": "celltype_embedding",
    "factors.assay_embedding": "assay_embedding",
    "factors.genome_25bp_embedding": "genome_25bp_embedding",
    "factors.genome_250bp_embedding": "genome_250bp_embedding",
    "factors.genome_5kbp_embedding": "genome_5kbp_embedding",
    "block1.dense": "dense_1", "block1.ac": "dense_1_ac", "block1.bn": "dense_1_bn",
    "block2.dense": "dense_2", "block2.ac": "dense_2_ac", "block2.bn": "dense_2_bn",
    "block3.dense": "dense_3", "block3.ac": "dense_3_ac", "block3.bn": "dense_3_bn",
    "y_pred": "y_pred",
}

_PRECAMOLE_LAYERS: Dict[str, str] = {
    k: v for k, v in _GUACAMOLE_LAYERS.items() if not k.startswith(("block3", "y_pred"))
}
_PRECAMOLE_LAYERS["y_pred_dense"] = "y_pred_dense"


def keras_layer_names(h5_path: Path | str) -> List[str]:
    """`model_weights.attrs['layer_names']`, decoded. Cheap enough to call before committing to a
    file — it is how you tell a stage-1 checkpoint from a stage-2 one without loading 700 MB."""
    import h5py

    with h5py.File(str(h5_path), "r") as f:
        if "model_weights" not in f:
            raise KerasWeightError(f"{h5_path}: no 'model_weights' group — not a Keras model file")
        return [n.decode() if isinstance(n, bytes) else str(n)
                for n in f["model_weights"].attrs["layer_names"]]


def read_layer_weights(h5_path: Path | str) -> Dict[str, Dict[str, np.ndarray]]:
    """Every weight-bearing layer as `{layer: {short_name: array}}`.

    `short_name` drops the repeated layer prefix and the `:0` suffix, so
    `dense_1/kernel:0` arrives as `kernel`. Layers with no weights (inputs, flattens, concatenates,
    dropouts, the adds) are absent, not empty.
    """
    import h5py

    out: Dict[str, Dict[str, np.ndarray]] = {}
    with h5py.File(str(h5_path), "r") as f:
        if "model_weights" not in f:
            raise KerasWeightError(f"{h5_path}: no 'model_weights' group — not a Keras model file")
        mw = f["model_weights"]
        for raw in mw.attrs["layer_names"]:
            layer = raw.decode() if isinstance(raw, bytes) else str(raw)
            grp = mw[layer]
            names = [n.decode() if isinstance(n, bytes) else str(n)
                     for n in grp.attrs.get("weight_names", [])]
            if not names:
                continue
            tensors: Dict[str, np.ndarray] = {}
            for full in names:
                short = full.split("/")[-1].split(":")[0]
                tensors[short] = np.asarray(grp[full], dtype=np.float32)
            out[layer] = tensors
    return out


def _assign(dst: torch.Tensor, src: np.ndarray, where: str) -> None:
    if tuple(dst.shape) != src.shape:
        raise KerasWeightError(
            f"{where}: keras gives {src.shape}, torch wants {tuple(dst.shape)}")
    dst.copy_(torch.from_numpy(np.ascontiguousarray(src)))


def _load_one(module: nn.Module, tensors: Mapping[str, np.ndarray], layer: str) -> int:
    """Copy one Keras layer into one torch submodule. Returns how many tensors moved."""
    if isinstance(module, nn.Embedding):
        _assign(module.weight.data, tensors["embeddings"], f"{layer}/embeddings")
        return 1
    if isinstance(module, nn.Linear):
        # (in, out) -> (out, in). The only conversion that is wrong-but-shaped-right on a square
        # kernel, so it is checked by value in the round-trip test, not just by shape.
        _assign(module.weight.data, tensors["kernel"].T, f"{layer}/kernel")
        _assign(module.bias.data, tensors["bias"], f"{layer}/bias")
        return 2
    if isinstance(module, nn.PReLU):
        _assign(module.weight.data, tensors["alpha"], f"{layer}/alpha")
        return 1
    if isinstance(module, nn.BatchNorm1d):
        _assign(module.weight.data, tensors["gamma"], f"{layer}/gamma")
        _assign(module.bias.data, tensors["beta"], f"{layer}/beta")
        _assign(module.running_mean.data, tensors["moving_mean"], f"{layer}/moving_mean")
        _assign(module.running_var.data, tensors["moving_variance"], f"{layer}/moving_variance")
        return 4
    raise KerasWeightError(f"{layer}: no rule for torch module {type(module).__name__}")


def load_keras_h5(module: Precamole | Guacamole, h5_path: Path | str) -> Dict[str, int]:
    """Load one of their checkpoints into `module`, in place. Returns a small report.

    Refuses rather than guesses. A layer the module expects and the file lacks, a shape that does
    not match, an unmapped torch submodule — each raises `KerasWeightError` naming the tensor. The
    report is `{"layers": n, "tensors": n, "parameters": n}`; a caller that wants to prove the
    whole file moved should compare `parameters` against `sum(p.numel() for p in module.parameters())`
    plus the batch-norm buffers.

    `module` is left in **eval mode**: their predictions come from a trained graph, so the running
    batch-norm statistics we just loaded are the ones that must be used.
    """
    mapping = _PRECAMOLE_LAYERS if isinstance(module, Precamole) else _GUACAMOLE_LAYERS
    available = read_layer_weights(h5_path)

    missing = [k for k in mapping.values() if k not in available]
    if missing:
        raise KerasWeightError(
            f"{h5_path}: missing layer(s) {missing}; file has {sorted(available)}. "
            f"A stage-1 file loaded as Guacamole (or the reverse) looks exactly like this.")

    named = dict(module.named_modules())
    layers = tensors = 0
    with torch.no_grad():
        for attr, layer in mapping.items():
            if attr not in named:
                raise KerasWeightError(f"module has no submodule {attr!r} for keras layer {layer!r}")
            tensors += _load_one(named[attr], available[layer], layer)
            layers += 1

    module.eval()
    n_params = sum(int(np.prod(a.shape)) for layer in mapping.values()
                   for a in available[layer].values())
    return {"layers": layers, "tensors": tensors, "parameters": n_params}
