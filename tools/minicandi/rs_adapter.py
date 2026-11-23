"""Present a read-start miniCANDI run to the overlap-count scorer (t392 chunk C9).

`ReadStartAdapter(inner, config, run_dir=None, tables=None)` is a `contract.MiniCandi` whose
forward takes the ordinary overlap-count batch that `score.py` and `predict_store.py` build (L bins
of 25 bp, `meta` with `cell`, `chrom`, `start_bin`, `targets`, `measurements_in`, `input_file`,
`measurements_out`, `target_file`) and returns `{"n", "mu"}` [B, L, A] in overlap-count space.
`registry.load_run` applies it to a run whose config sets `read_start`, so neither scorer file is
edited.

Steps of `forward`:
  1. For every input track with `x_avail` true, `x_counts_sub` [B, L, A, 2] is read from the
     read-start store (`ReadStartDataset.read_strands`, the file `input_file` of `meta`; `pr1`
     for a track in `meta["halves_tracks"]`), set to 0 on every bin where `x_bin_mask` is true and
     on every track where `x_avail` is false, and folded as in training (an unstranded track holds
     its total in channel 0, channel 1 is 0). `x_counts` becomes its channel sum. The batch's own
     `x_counts` is never read.
  2. Under `average_input`, `avg_sub` comes from `ReadStartDataset.average_strands` and `avg` is
     its channel sum.
  3. The inner model runs and returns `mu_sub`, `n_sub` [B, L, A, 2].
  4. Span kernel per target track: the `kernel` [2, 17] and `n_reads_strand` [2] attributes of
     the target's store file (`measurements_out`, `target_file`; `pr2` for a halved track). A
     track with no target measurement in the window takes the read-weighted mean of the kernels
     of the window's target measurements (per strand, weighted by that strand's reads). For an
     unstranded track (`read_start.UNSTRANDED_TRACKS`) the kernel is
     `read_start.unstranded_kernel(kernel, n_reads_strand)`, [17], with `stranded` False.
  5. `read_start.stitch` groups the windows into runs; each run is converted as one array by
     `read_start.convert_nb`. Every run has two open edges, filled with the edge bin's values.

Counters: `n_windows` (windows converted) and `n_open_edges` (2 per run). When the environment
variable `RS392_EDGE_LOG` names a file, both are written there as JSON after every forward (the
caller points it at `<run_dir>/<score dir>/open_edges.json`).

Refused (ValueError): a config without `read_start`, and `input_depth` other than native (the
read-start store holds native counts only). The copy-leak wiring check of the scorer perturbs the
batch's `x_counts`, which this adapter does not read, so that check cannot reach the inner model;
the masking is held by step 1 and its tests.

Store location: `config["read_start_store"]`; the tables as in `res_adapter.tables_of`, read with
`rs_dataset.ReadStartDataset` in mode `val`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from minicandi import contract as C  # noqa: E402
from minicandi import read_start as RS  # noqa: E402
from minicandi.res_adapter import tables_of  # noqa: E402

#: the environment variable that names the open-edge log file
EDGE_LOG_ENV = "RS392_EDGE_LOG"
_STRIP = ("two_step", "control_head", "evalset")


class ReadStartAdapter(C.MiniCandi):
    """See the module docstring."""

    def __init__(self, inner: torch.nn.Module, config: dict, run_dir=None, tables=None):
        super().__init__()
        self.inner = inner
        self.config = dict(config)
        if not self.config.get("read_start"):
            raise ValueError("ReadStartAdapter is for a run whose config sets read_start")
        if self.config.get("input_depth", "native") != "native":
            raise ValueError(f"ReadStartAdapter: input_depth {self.config['input_depth']!r} at "
                             f"scoring; only native is served")
        self.tables = tables_of(self.config, run_dir, tables)
        self.store = Path(self.config["read_start_store"])
        self.n_windows = 0
        self.n_open_edges = 0
        self._ds = None
        self._kernels: dict = {}
        self._un = RS._unstranded_tracks()

    @property
    def ds(self):
        if self._ds is None:
            from minicandi.rs_dataset import ReadStartDataset
            cfg = {k: v for k, v in self.config.items() if k not in _STRIP}
            self._ds = ReadStartDataset(self.tables, "val", cfg)
        return self._ds

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_ds"] = None
        return state

    # ---------------------------------------------------------------- inputs

    def _inputs(self, batch: dict) -> torch.Tensor:
        """int32 [B, L, A, 2]: the store's starts, 0 at hidden bins and missing tracks, folded."""
        x_avail = batch["x_avail"].cpu()
        B, L, A = batch["x_bin_mask"].shape
        out = torch.zeros(B, L, A, RS.STRANDS, dtype=torch.int32)
        for b, meta in enumerate(batch["meta"]):
            halved = frozenset(meta.get("halves_tracks") or ())
            start = int(meta["start_bin"])
            for t, m in enumerate(meta["measurements_in"]):
                if not m or not bool(x_avail[b, t]):
                    continue
                which = "pr1" if C.TRACKS[t] in halved else meta.get("input_file", "counts")
                v = self.ds.read_strands(m, meta["chrom"], start, L, which)
                out[b, :, t] = torch.from_numpy(v.astype(np.int32))
        hidden = batch["x_bin_mask"].cpu().bool() | ~x_avail.bool()[:, None, :]
        out[hidden] = 0
        un = self._un
        out[:, :, un, 0] = out[:, :, un, 0] + out[:, :, un, 1]
        out[:, :, un, 1] = 0
        return out

    # ---------------------------------------------------------------- kernels

    def _kernel_of(self, mid: str, which: str) -> tuple[np.ndarray, np.ndarray]:
        key = (mid, which)
        if key not in self._kernels:
            from minicandi.rs_dataset import STORE_FILES
            with h5py.File(self.store / mid / STORE_FILES[which], "r") as f:
                k = np.asarray(f.attrs["kernel"], np.float64)
                nr = np.asarray(f.attrs["n_reads_strand"], np.float64)
            if k.shape != (RS.STRANDS, RS.KERNEL_WIDTH) or nr.shape != (RS.STRANDS,):
                raise ValueError(f"{mid} {which}: kernel {k.shape}, n_reads_strand {nr.shape}")
            self._kernels[key] = (k, nr)
        return self._kernels[key]

    def _window_kernels(self, meta: dict) -> list:
        """Per track, the kernel to pass `convert_nb`: [2, 17] for a stranded track, the
        read-weighted [17] mix for an unstranded one."""
        halved = frozenset(meta.get("halves_tracks") or ())
        own = {}
        for t, m in enumerate(meta["measurements_out"]):
            if not m:
                continue
            which = ("pr2" if meta["measurements_in"][t] and C.TRACKS[t] in halved
                     else meta.get("target_file", "counts"))
            own[t] = self._kernel_of(m, which)
        if not own:
            raise ValueError(f"{meta['chrom']}:{meta['start_bin']}: no target measurement in "
                             f"the window, so no span kernel")
        nr_mix = sum(nr for _, nr in own.values())
        k_mix = sum(nr[:, None] * k for k, nr in own.values()) / np.maximum(nr_mix, 1e-300)[:, None]
        out = []
        for t in range(C.A):
            k, nr = own.get(t, (k_mix, nr_mix))
            out.append(RS.unstranded_kernel(k, nr) if bool(self._un[t]) else torch.from_numpy(k))
        return out

    # ---------------------------------------------------------------- forward

    def forward(self, batch: dict) -> dict:
        dev = batch["x_bin_mask"].device
        x_sub = self._inputs(batch).to(dev)
        b = dict(batch, x_counts_sub=x_sub, x_counts=x_sub.sum(-1, dtype=torch.int32))
        B, L, A = batch["x_bin_mask"].shape
        if self.config.get("average_input"):
            a = np.stack([self.ds.average_strands(m["cell"], m["chrom"], int(m["start_bin"]), L)
                          for m in batch["meta"]])
            avg_sub = torch.from_numpy(a).to(dev)
            b["avg_sub"], b["avg"] = avg_sub, avg_sub.sum(-1)
        o = self.inner(b)
        mu_sub, n_sub = o["mu_sub"], o["n_sub"]
        mu = torch.empty(B, L, A, dtype=mu_sub.dtype, device=mu_sub.device)
        n = torch.empty_like(mu)
        runs = RS.stitch(batch["meta"], L)
        for run in runs:
            ks = self._window_kernels(batch["meta"][run[0]])
            for i in run[1:]:
                other = self._window_kernels(batch["meta"][i])
                if not all(torch.equal(x, y) for x, y in zip(ks, other)):
                    raise ValueError("ReadStartAdapter: windows of one stitched run have "
                                     "different span kernels")
            idx = torch.as_tensor(run, device=mu_sub.device)
            ms = mu_sub[idx].reshape(len(run) * L, A, RS.STRANDS)
            ns = n_sub[idx].reshape(len(run) * L, A, RS.STRANDS)
            for t in range(A):
                mt, nt = RS.convert_nb(ms[:, t], ns[:, t], ks[t], stranded=not bool(self._un[t]))
                mu[idx, :, t] = mt.reshape(len(run), L)
                n[idx, :, t] = nt.reshape(len(run), L)
        self.n_windows += B
        self.n_open_edges += 2 * len(runs)
        self._log_edges()
        return {"n": n, "mu": mu}

    def _log_edges(self) -> None:
        path = os.environ.get(EDGE_LOG_ENV)
        if not path:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps({"n_open_edges": self.n_open_edges,
                                   "n_windows": self.n_windows}, sort_keys=True) + "\n",
                       encoding="utf-8")
        tmp.replace(p)


__all__ = ["EDGE_LOG_ENV", "ReadStartAdapter"]
