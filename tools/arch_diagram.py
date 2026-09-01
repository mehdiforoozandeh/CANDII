"""Generate `src/candi/README.md` and its diagrams FROM THE MODEL, never from prose.

WHY THIS EXISTS
---------------
Every hand-written description of an architecture is a copy, and every copy drifts. The top-level
README carried one for months; the only reason it was still right is that nobody had changed a
default. This tool removes the copy: it builds `build_model()` at the constructor defaults, runs one
real forward pass, and writes the README out of what that model actually is.

Nothing here is typed by hand. Every number on the page — channel widths, sequence lengths, lane
shapes, parameter counts, the head arithmetic's constants — is read off the built module tree or off
a recorded forward hook. If you change a default in `CandiModel.__init__`, the page changes.

THE ANCHOR CHECK IS THE POINT
-----------------------------
A generator that silently draws a stale picture is worse than a stale comment, because it looks
freshly made. `ANCHORS` names every module path the diagram depends on. If a rename or a refactor
removes one, `build` RAISES with the missing path rather than emitting a diagram with a hole in it.
Adding a module is caught the other way, by the byte-compare gate in `tests/test_arch_readme.py`.

WHAT IS GATED, AND WHAT IS NOT
------------------------------
`check` compares `arch.json`, `candi_graph.dot`, `torchinfo.txt` and `README.md` byte for byte. The
rendered `candi_graph.svg` is NOT compared: graphviz's layout engine is only stable within a version,
so gating the SVG would turn a `brew upgrade graphviz` into a failed test about a model nobody
touched. The DOT is the graph; the SVG is a picture of it, and the DOT is what carries the meaning.

    python tools/arch_diagram.py build     # rewrite the four artifacts + the README
    python tools/arch_diagram.py check     # exit 1 if any committed artifact is out of date

Needs the `test` extra (`torchview`, `torchinfo`, `graphviz`) and the `dot` binary on PATH.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from candi._vendored import CLOZE, MISSING  # noqa: E402
from candi.model import (  # noqa: E402
    ALL_FILM_TAPS,
    CandiModel,
    arch_keys,
    build_model,
    film_mode_from_taps,
    parse_film_taps,
)

OUT_DIR = SRC / "candi" / "arch"
README = SRC / "candi" / "README.md"
ARCH_JSON = OUT_DIR / "arch.json"
DOT_FILE = OUT_DIR / "candi_graph.dot"
SVG_FILE = OUT_DIR / "candi_graph.svg"
INFO_FILE = OUT_DIR / "torchinfo.txt"

# The trace runs at batch 1. The diagram writes `B` everywhere, so the batch axis is symbolic and a
# larger trace batch would only slow the forward down.
TRACE_BATCH = 1
# Depth 1 of the torchview graph: the candi modules themselves, not x-transformers' internals. Depth
# 2 is 162 nodes and unreadable as a picture; the module census in arch.json covers that detail.
GRAPH_DEPTH = 1

# Every module path the Mermaid diagram reads. Missing one is a hard error — see the module
# docstring. Paths are exact `named_modules()` keys on a default-built `CandiModel`.
ANCHORS: Tuple[str, ...] = (
    "encoder.metadata_embedding",
    "encoder.signal_tower",
    "encoder.signal_tower.blocks.0",
    "encoder.mask_injector",
    "encoder.dna_tower",
    "encoder.dna_tower.blocks.0",
    "encoder.fusion",
    "encoder.transformer_blocks.0",
    "decoder.meta_embedding",
    "decoder.input_proj",
    "decoder.blocks.0",
    "decoder.film_layers.0",
    "decoder.head_eta",
    "decoder.head_n",
)


# ---------------------------------------------------------------------------
# The trace
# ---------------------------------------------------------------------------

def trace_inputs(model: CandiModel) -> Tuple[torch.Tensor, ...]:
    """A representative batch, sized off the built encoder rather than off the flag values.

    Track 0 is CLOZE and track 1 is MISSING in BOTH the signal and the metadata, because
    `_prepare_signal` refuses a batch where the two disagree. The control channel (index A) is never
    masked. Everything else is observed, so the trace exercises the mask-token path and the observed
    path at once.
    """
    a = model.decoder.A
    length = model.encoder.l1
    dna_len = model.encoder.required_dna_len
    b = TRACE_BATCH

    x_data = torch.zeros(b, length, a + 1)
    x_meta = torch.empty(b, 4, a + 1)
    x_meta[:, 0, :] = 25.0                                    # log2 depth
    x_meta[:, 1, :] = torch.arange(a + 1, dtype=torch.float)  # assay id
    x_meta[:, 2, :] = 100.0                                   # read length
    x_meta[:, 3, :] = 1.0                                     # run type
    for track, sentinel in ((0, CLOZE), (1, MISSING)):
        x_data[:, :, track] = sentinel
        x_meta[:, :, track] = sentinel

    y_meta = torch.empty(b, 4, a)
    y_meta[:, 0, :] = 25.0
    y_meta[:, 1, :] = torch.arange(a, dtype=torch.float)
    y_meta[:, 2, :] = 100.0
    y_meta[:, 3, :] = 1.0

    x_dna = torch.zeros(b, 4, dna_len)
    x_dna[:, 0, :] = 1.0
    return x_data, x_dna, x_meta, y_meta


def _shape(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return list(obj.shape)
    if isinstance(obj, (tuple, list)):
        return [_shape(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _shape(v) for k, v in obj.items()}
    return None


def run_trace(model: CandiModel) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (recorded, outputs): output shape per module path, in firing order."""
    recorded: "OrderedDict[str, Any]" = OrderedDict()
    handles = []

    def make_hook(path: str):
        def hook(_mod, _inp, out):
            # A module called twice keeps its FIRST shape; nothing in the default model is, and a
            # module that becomes reused should show up as a diff rather than as a silent overwrite.
            recorded.setdefault(path, _shape(out))
        return hook

    for path, mod in model.named_modules():
        if path:
            handles.append(mod.register_forward_hook(make_hook(path)))
    try:
        with torch.no_grad():
            out = model(*trace_inputs(model))
    finally:
        for h in handles:
            h.remove()
    return recorded, {k: list(v.shape) for k, v in out.items()}


# ---------------------------------------------------------------------------
# The spec — arch.json
# ---------------------------------------------------------------------------

def _param_count(mod: torch.nn.Module) -> int:
    return sum(p.numel() for p in mod.parameters())


def introspect() -> Dict[str, Any]:
    torch.manual_seed(0)
    model = build_model().eval()
    modules = dict(model.named_modules())

    missing = [a for a in ANCHORS if a not in modules]
    if missing:
        raise RuntimeError(
            f"the diagram anchors {missing} are no longer module paths on a default CandiModel. "
            "The model was renamed or restructured; update ANCHORS in tools/arch_diagram.py so the "
            "diagram describes the model that exists, then regenerate.")

    recorded, outputs = run_trace(model)
    missing_shapes = [a for a in ANCHORS if a not in recorded]
    if missing_shapes:
        raise RuntimeError(
            f"the anchors {missing_shapes} exist but never fired during the forward pass, so the "
            "diagram would show a shape for a module the model does not run. Update ANCHORS.")

    import inspect
    sig = inspect.signature(CandiModel.__init__)
    defaults = {k: sig.parameters[k].default for k in arch_keys()}
    defaults = {k: (list(v) if isinstance(v, tuple) else v) for k, v in defaults.items()}

    taps = parse_film_taps(defaults["film_taps"])
    enc, dec = model.encoder, model.decoder
    tower = enc.signal_tower

    spec: Dict[str, Any] = {
        "_generated_by": "tools/arch_diagram.py — do not edit; run `python tools/arch_diagram.py build`",
        "arch_defaults": defaults,
        "derived": {
            "num_tracks": enc.num_tracks,
            "context_bins": enc.l1,
            "latent_bins": enc.l2,
            "resolution_bp": enc.resolution,
            "dna_length_bp": enc.required_dna_len,
            "span_bp": enc.l1 * enc.resolution,
            "signal_tower_out_channels": tower.out_channels,
            "signal_tower_out_per_assay": tower.out_per_assay,
            "signal_tower_lane_shapes": tower.lane_shapes(enc.l1),
            "dna_tower_out_channels": enc.dna_tower.out_channels,
            "dna_tower_n_blocks": len(enc.dna_tower.blocks),
            "d_model": enc.d_model,
            "d_model_is_auto": enc.d_model_is_auto,
            "encoder_film_mode": film_mode_from_taps(taps),
            "film_taps_on": list(taps),
            "film_taps_off": [t for t in ALL_FILM_TAPS if t not in taps],
            "decoder_lane": dec.lane,
            "decoder_n_blocks": len(dec.blocks),
            "heads": list(dec.heads),
            "head_arithmetic": {
                "use_offset": dec.use_offset,
                "depth_center": dec.depth_center,
                "log2_mu_clamp": [dec.clamp_lo, dec.clamp_hi],
                "mu_eps": dec.eps,
                "depth_row": dec.depth_row,
            },
        },
        "params": {
            "total": _param_count(model),
            "encoder": _param_count(enc),
            "decoder": _param_count(dec),
            "by_part": {
                "encoder.metadata_embedding": _param_count(enc.metadata_embedding),
                "encoder.signal_tower": _param_count(tower),
                "encoder.mask_injector": _param_count(enc.mask_injector),
                "encoder.dna_tower": _param_count(enc.dna_tower),
                "encoder.fusion": _param_count(enc.fusion),
                "encoder.transformer_blocks": _param_count(enc.transformer_blocks),
                "decoder.input_proj": _param_count(dec.input_proj),
                "decoder.blocks": _param_count(dec.blocks),
                "decoder.film_layers": _param_count(dec.film_layers),
                "decoder.head_eta": _param_count(dec.head_eta),
                "decoder.head_n": _param_count(dec.head_n),
                "decoder.meta_embedding": _param_count(dec.meta_embedding),
            },
        },
        "io": {
            "inputs": {
                "x_data": [ "B", enc.l1, enc.num_tracks ],
                "x_dna": [ "B", 4, enc.required_dna_len ],
                "x_meta": [ "B", 4, enc.num_tracks ],
                "y_meta": [ "B", 4, dec.A ],
            },
            "outputs": {k: ["B"] + v[1:] for k, v in outputs.items()},
        },
        # The full census: every module the default model owns, its class and its parameter count.
        # This is what makes a refactor deep inside the towers show up as a diff.
        "modules": [
            {"path": p, "class": type(m).__name__, "params": _param_count(m),
             "out_shape": recorded.get(p)}
            for p, m in model.named_modules() if p
        ],
    }
    return spec


# ---------------------------------------------------------------------------
# torchview + torchinfo
# ---------------------------------------------------------------------------

def render_graph() -> str:
    """The traced computation graph as graphviz DOT. Returns the source; also writes the SVG."""
    from torchview import draw_graph

    torch.manual_seed(0)
    model = build_model().eval()
    graph = draw_graph(
        model, input_data=trace_inputs(model), depth=GRAPH_DEPTH, device="cpu",
        graph_name="CANDI", save_graph=False, expand_nested=False, hide_inner_tensors=True,
    )
    dot = graph.visual_graph
    # Restyled for legibility and for a stable render. torchview's default font is "Linux libertine",
    # which is absent on most machines, so the SVG's metrics depend on what fonts the renderer
    # happens to have. Helvetica is present everywhere that matters.
    dot.graph_attr.update(rankdir="TB", bgcolor="transparent", splines="ortho",
                          nodesep="0.25", ranksep="0.45", fontname="Helvetica")
    dot.node_attr.update(fontname="Helvetica", fontsize="11")
    dot.edge_attr.update(fontname="Helvetica", fontsize="9", color="#5a6570", arrowsize="0.7")
    # `size` is a DPI-dependent bounding box torchview computes from the graph; dropping it lets the
    # SVG render at its natural size instead of being squeezed into a fixed square.
    dot.graph_attr.pop("size", None)
    return dot.source


def render_svg(dot_source: str) -> bytes:
    if shutil.which("dot") is None:
        raise RuntimeError("graphviz's `dot` binary is not on PATH; install graphviz "
                           "(`brew install graphviz`) and re-run")
    return subprocess.run(["dot", "-Tsvg"], input=dot_source.encode(), check=True,
                          stdout=subprocess.PIPE).stdout


def render_torchinfo() -> Tuple[str, Dict[str, Any]]:
    """The layer table, and the cost numbers torchinfo derives while building it.

    The cost block is the one thing on the page that is not visible in the module tree: how much
    arithmetic and how much activation memory one window of this model actually costs. It rides in
    `arch.json` so it is gated like everything else.
    """
    from torchinfo import summary

    torch.manual_seed(0)
    model = build_model().eval()
    stats = summary(
        model, input_data=trace_inputs(model), depth=3, device="cpu", verbose=0,
        col_names=("input_size", "output_size", "num_params"), row_settings=("var_names",),
    )
    cost = {
        "trace_batch": TRACE_BATCH,
        "mult_adds": int(stats.total_mult_adds),
        "param_bytes": int(stats.total_param_bytes),
        "activation_bytes": int(stats.total_output_bytes),
        "input_bytes": int(stats.total_input),
    }
    return str(stats) + "\n", cost


# ---------------------------------------------------------------------------
# The Mermaid hero diagram — every label is a lookup into the spec
# ---------------------------------------------------------------------------

def _lane(shape_str: str) -> str:
    """`'[B, 384, 36, 2]'` -> `'B x 384 x 36 x 2'`, because Mermaid labels cannot hold brackets."""
    return shape_str.strip("[]").replace(", ", " x ")


def _dims(dims: List[Any]) -> str:
    return " x ".join(str(d) for d in dims)


def _bdims(dims: List[Any]) -> str:
    """A traced shape with its batch axis made symbolic again.

    The trace runs at batch 1, and a label reading `1 x 36 x 32` would state a batch size the model
    does not have. Only axis 0 is rewritten, and only here — `arch.json` keeps the literal shape the
    forward pass produced, because that is the record.
    """
    return " x ".join(["B"] + [str(d) for d in dims[1:]])


def _params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.2f} M"
    if n >= 1_000:
        return f"{n / 1e3:.1f} k"
    return str(n)


def mermaid(spec: Dict[str, Any]) -> str:
    d = spec["derived"]
    a = spec["arch_defaults"]
    p = spec["params"]["by_part"]
    mods = {m["path"]: m for m in spec["modules"]}
    lanes = d["signal_tower_lane_shapes"]
    n_conv = a["n_cnn_layers"]
    tracks, assays = d["num_tracks"], a["num_assays"]
    per_conv = "per_conv" in d["film_taps_on"]

    def dna_block(i: int) -> str:
        out = mods[f"encoder.dna_tower.blocks.{i}"]["out_shape"]
        return f"B x {out[1]} x {out[2]}"

    dna_rungs = "<br/>".join(
        f"block {i}: {dna_block(i)}" for i in range(d["dna_tower_n_blocks"]))
    sig_rungs = "<br/>".join(
        f"block {i}: {_lane(lanes[i + 1])}" + (" + FiLM" if per_conv else "")
        for i in range(n_conv))
    dec_rungs = "<br/>".join(
        f"block {i}: x{a['deconv_upsample']} -> "
        f"B x {d['latent_bins'] * a['deconv_upsample'] ** (i + 1)} x {assays} x {d['decoder_lane']}"
        + (" + FiLM" if "per_deconv" in d["film_taps_on"] else "")
        for i in range(d["decoder_n_blocks"]))

    def kp(part: str) -> str:
        return _params(p[part])

    heads = ", ".join(d["heads"])
    out_keys = ", ".join(spec["io"]["outputs"])
    hc = d["head_arithmetic"]

    return f"""%%{{init: {{"flowchart": {{"htmlLabels": true, "wrappingWidth": 460, "nodeSpacing": 34, "rankSpacing": 40, "curve": "basis"}}}}}}%%
flowchart TB

  %% ---------------- inputs ----------------
  XD["<b>x_data</b><br/>{_dims(spec['io']['inputs']['x_data'])}<br/><i>raw counts · {assays} assays + 1 control</i>"]
  DNA["<b>x_dna</b><br/>{_dims(spec['io']['inputs']['x_dna'])}<br/><i>one-hot DNA · {d['span_bp'] / 1000:g} kb</i>"]
  XM["<b>x_meta</b><br/>{_dims(spec['io']['inputs']['x_meta'])}<br/><i>log2 depth · assay · read len · run type</i>"]
  YM["<b>y_meta</b><br/>{_dims(spec['io']['inputs']['y_meta'])}<br/><i>the same four rows, for the TARGET</i>"]

  %% ---------------- conditioning ----------------
  XME["MetadataEmbedding<br/>{_bdims(mods['encoder.metadata_embedding']['out_shape'])} · {kp('encoder.metadata_embedding')}"]
  YME["MetadataEmbedding<br/>{_bdims(mods['decoder.meta_embedding']['out_shape'])} · {kp('decoder.meta_embedding')}"]
  XM --> XME
  YM --> YME

  %% ---------------- encoder ----------------
  subgraph ENC["<b>ENCODER</b> · {_params(spec['params']['encoder'])} params"]
    direction TB
    SIG["<b>SignalConvTower</b> · grouped Conv1d, groups={tracks} · {kp('encoder.signal_tower')}<br/>in: {_lane(lanes[0])}<br/>{sig_rungs}"]
    MASK["<b>MaskTokenInjector</b> · {kp('encoder.mask_injector')}<br/>learned token replaces every CLOZE / MISSING lane"]
    DNAT["<b>DNAConvTower</b> · dense Conv1d, {d['dna_tower_n_blocks']} blocks · {kp('encoder.dna_tower')}<br/>{dna_rungs}"]
    FUSE["<b>LinearFusion</b> · concat -> Linear -> GELU · {kp('encoder.fusion')}<br/>B x {d['latent_bins']} x {d['signal_tower_out_channels'] + d['dna_tower_out_channels']} -> B x {d['latent_bins']} x {d['d_model']}"]
    TR["<b>{a['n_transformer_layers']} x RoPE Transformer</b> · d_model={d['d_model']}, heads={a['nhead']} · {kp('encoder.transformer_blocks')}<br/>pre-norm, ff_mult=4, dropout={a['dropout']}"]
    SIG --> MASK --> FUSE
    DNAT --> FUSE --> TR
  end

  XD --> SIG
  DNA --> DNAT
  XME -. FiLM after every conv .-> SIG

  TR --> Z(["<b>z</b> · latent<br/>B x {d['latent_bins']} x {d['d_model']}<br/><i>1 token per {d['resolution_bp'] * a['pool_size'] ** n_conv} bp</i>"])

  %% ---------------- decoder ----------------
  subgraph DEC["<b>DECODER</b> · {_params(spec['params']['decoder'])} params"]
    direction TB
    IP["<b>input_proj</b> · Linear({d['d_model']} -> {assays} x {d['decoder_lane']}) · {kp('decoder.input_proj')}<br/>the one cross-assay mixer; every layer after it is grouped"]
    DFILM["<b>FiLM</b> (pre_deconv) — re-establishes assay identity"]
    DB["<b>{d['decoder_n_blocks']} x LaneDeconvBlock</b> · lane={d['decoder_lane']}, norm={a['deconv_norm']} · {kp('decoder.blocks')}<br/>{dec_rungs}"]
    HEAD["<b>NB head</b> · weight-shared across assays · {_params(p['decoder.head_eta'] + p['decoder.head_n'])}<br/>Linear({d['decoder_lane']} -> {d['decoder_lane']}) -> GELU -> Linear({d['decoder_lane']} -> 1), twice"]
    IP --> DFILM --> DB --> HEAD
  end

  Z --> IP
  YME -. FiLM at every tap .-> DFILM

  LINK["<b>depth-offset log link</b> — fp32-fenced<br/>log2_mu = (d - {hc['depth_center']}) + eta<br/>mu = 2^clamp(log2_mu, {hc['log2_mu_clamp'][0]:.0f}, {hc['log2_mu_clamp'][1]:.0f})<br/>n = softplus(raw_n) + {hc['mu_eps']:g}<br/>p = n / (n + mu)"]
  HEAD --> LINK
  YM -. "log2 depth (row {hc['depth_row']})" .-> LINK

  OUT(["<b>Negative Binomial per assay per {d['resolution_bp']} bp bin</b><br/>{out_keys}<br/>each B x {a['context_length']} x {assays} · heads = {heads}"])
  LINK --> OUT

  style ENC fill:#f4faf5,stroke:#4caf72,stroke-width:1.5px;
  style DEC fill:#faf7fd,stroke:#9061d6,stroke-width:1.5px;

  classDef inp fill:#e8f0fe,stroke:#4285f4,stroke-width:1.5px,color:#12304f;
  classDef meta fill:#fff4e5,stroke:#f0a04b,stroke-width:1.5px,color:#4a3213;
  classDef enc fill:#eaf6ec,stroke:#4caf72,stroke-width:1.5px,color:#12351f;
  classDef dec fill:#f3ecfb,stroke:#9061d6,stroke-width:1.5px,color:#2c1447;
  classDef lat fill:#fdecef,stroke:#e05a72,stroke-width:2px,color:#4a1220;
  class XD,DNA,XM,YM inp;
  class XME,YME meta;
  class SIG,MASK,DNAT,FUSE,TR enc;
  class IP,DFILM,DB,HEAD,LINK dec;
  class Z,OUT lat;
"""


# ---------------------------------------------------------------------------
# The README
# ---------------------------------------------------------------------------

def _tap_table(spec: Dict[str, Any]) -> str:
    d = spec["derived"]
    rows = []
    where = {
        "pre_conv": "encoder — once, on the signal before the conv tower",
        "per_conv": "encoder — after every conv block",
        "post_conv": "encoder — once, after the whole conv tower",
        "per_transformer": "encoder — before every transformer layer",
        "pre_deconv": "decoder — after `input_proj`, before the first deconv",
        "per_deconv": "decoder — after every deconv block",
        "post_head": "decoder — on `eta` and `raw_n` after the heads",
    }
    for tap in ALL_FILM_TAPS:
        on = tap in d["film_taps_on"]
        rows.append(f"| `{tap}` | {'**on**' if on else 'off'} | {where[tap]} |")
    return "\n".join(rows)


def _biggest(spec: Dict[str, Any]) -> Tuple[str, float]:
    """The heaviest single part, and its share. Derived, because prose like "four fifths of the
    model is the transformer" is exactly the kind of sentence that survives the change that falsifies
    it."""
    part, n = max(spec["params"]["by_part"].items(), key=lambda kv: kv[1])
    return part, 100.0 * n / spec["params"]["total"]


def _param_table(spec: Dict[str, Any]) -> str:
    total = spec["params"]["total"]
    rows = []
    for part, n in spec["params"]["by_part"].items():
        rows.append(f"| `{part}` | {n:,} | {100.0 * n / total:.1f}% |")
    rows.append(f"| **total** | **{total:,}** | **100.0%** |")
    return "\n".join(rows)


def _flag_table(spec: Dict[str, Any]) -> str:
    rows = []
    for k, v in spec["arch_defaults"].items():
        shown = ",".join(str(x) for x in v) if isinstance(v, list) else v
        rows.append(f"| `{k}` | `{shown}` |")
    return "\n".join(rows)


def readme(spec: Dict[str, Any]) -> str:
    d = spec["derived"]
    a = spec["arch_defaults"]
    io = spec["io"]
    hc = d["head_arithmetic"]
    lanes = d["signal_tower_lane_shapes"]

    inputs = "\n".join(f"  {k:<8}{_dims(v)}" for k, v in io["inputs"].items())
    outputs = ", ".join(f"`{k}`" for k in io["outputs"])
    first_out = next(iter(io["outputs"].values()))

    return f"""<!-- GENERATED FILE — DO NOT EDIT.
     Written by tools/arch_diagram.py from a real forward pass through build_model().
     Change a default in src/candi/model.py, then run:  python tools/arch_diagram.py build
     tests/test_arch_readme.py fails until you do. -->

# CANDI — the default architecture

Everything on this page was read off `build_model()` at its constructor defaults and off one real
forward pass through it. No number here was typed by a person, and none of it can drift: the byte
gate in [`tests/test_arch_readme.py`](../../tests/test_arch_readme.py) fails the suite the moment
the model and this page disagree.

**{spec['params']['total']:,} parameters.** Per assay, per {d['resolution_bp']} bp bin, CANDI emits a Negative
Binomial `(n̂, p̂)` over raw counts, conditioned on four experimental covariates on the input side
*and* on the output side. That two-sided conditioning is what makes zero-shot imputation and
denoising on an unseen cell type possible.

```
{inputs}
   -> {outputs}
      each {_dims(['B'] + first_out[1:])}
```

## The graph

```mermaid
{mermaid(spec).rstrip()}
```

## Why the shapes are what they are

The whole model is one downsample and one upsample of the same factor, and the two are checked
against each other in the constructor before a single module is built.

- **`{a['pool_size']}^{a['n_cnn_layers']}` down, `{a['deconv_upsample']}^{a['n_deconv_layers']}` up.** {a['context_length']} bins in, {d['latent_bins']} latent tokens, {a['context_length']} bins out.
  A mismatch is a `ValueError` naming both flags, not a shape error inside a deconv.
- **The signal tower is grouped by track** (`groups = {d['num_tracks']}`), so an assay's channels never mix with
  another's. The lane view at each rung, asserted against a live forward pass by
  `tests/test_lane_view.py`:
  `{'` → `'.join(lanes)}` — that is `[B, bins, tracks, channels-per-track]`.
- **The DNA tower's pooling is derived, never chosen.** Two of its {d['dna_tower_n_blocks']} blocks pool by
  `isqrt({d['resolution_bp']}) = {int(d['resolution_bp'] ** 0.5)}` so that exactly {d['resolution_bp']} bp collapse into one bin. A resolution
  that is not a perfect square is refused rather than rounded.
- **`d_model` is {'derived' if d['d_model_is_auto'] else 'set'}**: {d['d_model']}, {'the signal tower’s output width' if d['d_model_is_auto'] else 'given explicitly'}.
- **The decoder is a mirror, not a trunk.** `input_proj` is the one cross-assay mixer; every layer
  after it is grouped at a constant lane width of {d['decoder_lane']}, so the whole decoder costs
  {spec['params']['decoder']:,} parameters — {100.0 * spec['params']['decoder'] / spec['params']['total']:.1f}% of the model. The ungrouped decoder this replaced ran a
  dense conv trunk instead and held the large majority of the parameters on its own.

## Where the conditioning enters

`film_taps` is a single set naming every place FiLM may enter, on both towers. The encoder's
`film_mode` enum is *derived* from it, so "where is the conditioning?" has one answer.

| tap | default | where it sits |
|---|---|---|
{_tap_table(spec)}

Encoder FiLM is initialised `{a['film_init_encoder']}`, decoder FiLM `{a['film_init_decoder']}` — a zero-initialised tap is an exact
identity at step 0, which is what lets a new tap be switched on without re-sampling the model.

## The head

`heads = {','.join(d['heads'])}`. The optional `signal` and `peak` heads are not constructed by default, own no
parameters, and add no keys to the output dict. The count head's arithmetic runs as one fp32-fenced
block, because `log2_mu` is an exponent and every bit lost there is a multiplicative error on `mu`:

```
log2_mu = (d - {hc['depth_center']}) + eta        # d = log2 depth; falls back to eta on a sentinel
log2_mu = log2_mu + log_ref        # only when a reference track is supplied
mu      = 2 ** clamp(log2_mu, {hc['log2_mu_clamp'][0]:.0f}, {hc['log2_mu_clamp'][1]:.0f})
n       = softplus(raw_n) + {hc['mu_eps']:g}
p       = n / (n + mu)
```

Telling the model a different sequencing depth therefore *scales* the prediction, rather than making
it relearn scale.

## Where the parameters are

| module | params | share |
|---|---:|---:|
{_param_table(spec)}

The single largest part is `{_biggest(spec)[0]}`, at {_biggest(spec)[1]:.1f}% of the weights. One window of {a['context_length']} bins costs
**{spec['cost']['mult_adds'] / 1e6:.0f} M multiply-adds** at batch {spec['cost']['trace_batch']}, {spec['cost']['param_bytes'] / 1e6:.1f} MB of fp32 weights and
{spec['cost']['activation_bytes'] / 1e6:.0f} MB of activations — the last of those scales with batch size and is what
`--precision bf16` halves.

## Every architecture flag, at its default

These are the keyword defaults of `CandiModel.__init__`, which `build_model_from_arch()` reads back
out of a run's own JSON so a checkpoint stays scorable forever. `num_assays` and `context_length`
come from the panel at train time; the rest are the model's own.

| flag | default |
|---|---|
{_flag_table(spec)}

## The traced graph, and the layer table

<details>
<summary><b>torchview</b> — the computation graph, traced from the real forward pass (depth {GRAPH_DEPTH})</summary>

<img src="arch/candi_graph.svg" alt="CANDI computation graph traced by torchview" width="100%">

Source: [`arch/candi_graph.dot`](arch/candi_graph.dot). The DOT is what the gate compares; the SVG is
a picture of it, and graphviz's layout is only stable within one version.

The tall run of `_eq` / `any` / `where` / `full_like` nodes above the coloured modules is real, not
noise: `CandiModel.forward` calls `encoder.encode(...)` rather than `encoder(...)`, so `V2Encoder`
never appears as a module box and the sentinel-availability bookkeeping inside `_prepare_signal`
surfaces as top-level ops. The Mermaid diagram at the top of this page is the readable view; this one
is the literal one.

</details>

<details>
<summary><b>torchinfo</b> — the layer / parameter table</summary>

See [`arch/torchinfo.txt`](arch/torchinfo.txt).

</details>

<details>
<summary><b>arch.json</b> — the machine-readable spec every artifact above is rendered from</summary>

[`arch/arch.json`](arch/arch.json) carries the flag defaults, the derived geometry, the parameter
census, and every one of the {len(spec['modules'])} modules the default model owns with its class, its parameter
count and its traced output shape. A refactor deep inside a tower shows up as a diff here.

</details>

## Regenerating

```bash
python tools/arch_diagram.py build     # rewrite this page and everything under arch/
python tools/arch_diagram.py check     # what the test runs: exit 1 if anything is stale
```

Needs the `test` extra (`pip install -e '.[test]'`) and graphviz's `dot` on PATH.
"""


# ---------------------------------------------------------------------------
# build / check
# ---------------------------------------------------------------------------

def generate() -> Dict[str, Any]:
    """Every artifact, as in-memory bytes. `build` writes them; `check` compares them."""
    spec = introspect()
    info_text, spec["cost"] = render_torchinfo()
    dot = render_graph()
    return {
        str(ARCH_JSON): (json.dumps(spec, indent=1, sort_keys=False) + "\n").encode(),
        str(DOT_FILE): dot.encode(),
        str(SVG_FILE): render_svg(dot),
        str(INFO_FILE): info_text.encode(),
        str(README): readme(spec).encode(),
    }


def cmd_build() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, blob in generate().items():
        Path(path).write_bytes(blob)
        print(f"wrote {Path(path).relative_to(REPO)}  ({len(blob):,} bytes)")
    return 0


def cmd_check() -> int:
    fresh = generate()
    stale = []
    for path, blob in fresh.items():
        p = Path(path)
        if p == SVG_FILE:
            # Not compared — graphviz layout is version-stable, not version-independent. Its content
            # is fully determined by the DOT, which IS compared.
            if not p.exists():
                stale.append(f"{p.relative_to(REPO)} is missing")
            continue
        if not p.exists():
            stale.append(f"{p.relative_to(REPO)} is missing")
        elif p.read_bytes() != blob:
            stale.append(f"{p.relative_to(REPO)} is out of date")
    if stale:
        print("the architecture page no longer matches the model:", file=sys.stderr)
        for s in stale:
            print(f"  - {s}", file=sys.stderr)
        print("\nrun: python tools/arch_diagram.py build", file=sys.stderr)
        return 1
    print("architecture page matches the model")
    return 0


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["build", "check"])
    args = ap.parse_args(argv)
    return cmd_build() if args.command == "build" else cmd_check()


if __name__ == "__main__":
    raise SystemExit(main())
