"""Present a miniCANDI run of `res_bp` < 25, or of `res_in_sub` = 5, to the 25 bp scorer (chunk RM).

`ResAdapter(inner, config, run_dir=None, tables=None)` is a `contract.MiniCandi` whose forward
takes the ordinary 25 bp batch that `score.py` and `guardrail.py` build (L bins of 25 bp, `meta`
with `chrom`, `start_bin`, `measurements_in`, `input_file`) and returns `{"n", "mu"}` [B, L, A] at
25 bp. `registry.load_run` applies it to such runs, so `score.py --run` scores them unchanged.

  res_bp < 25   f = 25 / res_bp fine tokens per bin. For every available input track it reads the
                fine counts of the same span from the region store (`ResDataset.read`, the file
                `input_file` of `meta`), sets them to 0 on every fine token of a hidden 25 bp bin
                and on every missing track, and checks that the visible fine counts sum to the
                batch's `x_counts` (else ValueError: the store is not the scored tables). The
                batch's `x_dna` (L * 25 bases) is already the fine model's sequence. The span of
                L * f fine tokens is cut into consecutive windows of the run's `context_bins`
                (L * f must be a multiple of it), the inner model runs on them (at most
                `max_windows` per forward), and per 25 bp bin mu is summed and n pooled with
                `losses.pool_nb` (the moment-matched size under conditional independence).
  res_in_sub    the batch gains `x_counts_sub` [B, L, A, 5] from the store (0 where hidden or
                missing; checked against `x_counts`); the inner model's 25 bp output is returned.

`y_counts` of the fine windows is 0 (no model reads it); `avg` (`average_input`) is refused. The
copy-leak wiring check therefore cannot reach the inner model through `x_counts`: the adapter
reads the store, and only at visible positions; the inner model's own wiring check runs at its
own resolution (tests/test_minicandi_resolution.py).

Store location: the tables directory `tables` (else the config's `res_tables`, else the `tables`
of `<run_dir>/run.json`), read with `res_dataset.ResDataset` in mode `val` (blind rows are not in
the region tables).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from minicandi import contract as C  # noqa: E402
from minicandi.components import res_bp_of, res_subs_of  # noqa: E402
from minicandi.losses import pool_nb  # noqa: E402
from onion_data.recount import pool  # noqa: E402

#: fine windows per forward pass of the inner model
MAX_WINDOWS = 32
#: batch keys with one row per window and no position axis: repeated for each fine window
PER_WINDOW = ("x_avail", "c_in", "c_out", "y_avail", "y_kind", "x_rna_avail", "x_cell_vec",
              "x_cell_vec_avail", "cell_id")


def tables_of(config: dict, run_dir=None, tables=None) -> Path:
    if tables is not None:
        return Path(tables)
    if config.get("res_tables"):
        return Path(config["res_tables"])
    if run_dir is not None and (Path(run_dir) / "run.json").is_file():
        return Path(json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))["tables"])
    raise FileNotFoundError("ResAdapter needs the region tables: pass tables, set config "
                            "res_tables, or give a run dir with run.json")


class ResAdapter(C.MiniCandi):
    """See the module docstring."""

    def __init__(self, inner: torch.nn.Module, config: dict, run_dir=None, tables=None,
                 max_windows: int = MAX_WINDOWS):
        super().__init__()
        self.inner = inner
        self.config = dict(config)
        self.res = res_bp_of(self.config)
        self.in_sub, self.out_sub = res_subs_of(self.config)
        if self.res > C.RES or (self.res == C.RES and self.in_sub == 1):
            raise ValueError(f"ResAdapter is for res_bp < 25 or res_in_sub > 1 (res_bp "
                             f"{self.res}, res_in_sub {self.in_sub}); score the run directly")
        if self.config.get("average_input"):
            raise ValueError("ResAdapter: average_input is not served at another bin size")
        self.L_fine = int(self.config.get("context_bins", 1024))
        self.max_windows = int(max_windows)
        self.tables = tables_of(self.config, run_dir, tables)
        self._ds = None

    @property
    def ds(self):
        if self._ds is None:
            from minicandi.res_dataset import ResDataset
            cfg = {k: v for k, v in self.config.items()
                   if k not in ("two_step", "control_head", "evalset")}
            self._ds = ResDataset(self.tables, "val", cfg)
        return self._ds

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_ds"] = None
        return state

    def _fine_inputs(self, batch: dict, f: int) -> torch.Tensor:
        """int32 [B, L*f, A] fine input counts (or [B, L, A, f] sub-bins when f is a sub count
        under res_in_sub), 0 at hidden and missing positions, checked against x_counts."""
        x = batch["x_counts"]
        B, L, A = x.shape
        hidden = (batch["x_bin_mask"] | ~batch["x_avail"][:, None, :]).cpu()
        out = torch.zeros(B, L, A, f, dtype=torch.int32)
        for b, meta in enumerate(batch["meta"]):
            start = int(meta.get("start_bp", int(meta["start_bin"]) * C.RES))
            which = meta.get("input_file", "counts")
            for t, m in enumerate(meta["measurements_in"]):
                if not m or not bool(batch["x_avail"][b, t]):
                    continue
                v = self.ds.read_bp(m, meta["chrom"], start, L * C.RES, which)
                out[b, :, t] = torch.from_numpy(pool(v, C.RES // f).reshape(L, f).astype("int32"))
        out[hidden] = 0
        if not torch.equal(out.sum(-1), x.cpu().masked_fill(hidden, 0)):
            raise ValueError("ResAdapter: the region store's counts do not sum to the batch's "
                             "x_counts at visible bins (another store, or a thinned input)")
        return out.to(x.device)

    def forward(self, batch: dict) -> dict:
        if self.res == C.RES:                                  # res_in_sub at 25 bp tokens
            return self.inner(dict(batch, x_counts_sub=self._fine_inputs(batch, self.in_sub)))
        f = C.RES // self.res
        x = batch["x_counts"]
        B, L, A = x.shape
        Lf = self.L_fine
        if (L * f) % Lf:
            raise ValueError(f"ResAdapter: {L} bins x {f} fine tokens is not a multiple of the "
                             f"run's context_bins {Lf}")
        nw = L * f // Lf
        sub = self._fine_inputs(batch, f)                      # [B, L, A, f]
        fine = sub.permute(0, 1, 3, 2).reshape(B, L * f, A)
        mask = batch["x_bin_mask"].repeat_interleave(f, dim=1)
        fb = {"x_counts": fine.reshape(B * nw, Lf, A),
              "x_bin_mask": mask.reshape(B * nw, Lf, A),
              "x_dna": batch["x_dna"].reshape(B * nw, Lf * self.res),
              "y_counts": torch.zeros(B * nw, Lf, A, dtype=torch.int32, device=x.device)}
        for k in PER_WINDOW:
            if k in batch:
                fb[k] = batch[k].repeat_interleave(nw, dim=0)
        fb["meta"] = []
        for meta in batch["meta"]:
            s0 = int(meta.get("start_bp", int(meta["start_bin"]) * C.RES))
            for w in range(nw):
                fb["meta"].append(dict(meta, start_bp=s0 + w * Lf * self.res, res_bp=self.res))
        outs = {"n": [], "mu": []}
        n_all = B * nw
        for i in range(0, n_all, self.max_windows):
            part = {k: (v[i:i + self.max_windows]) for k, v in fb.items()}
            o = self.inner(part)
            outs["n"].append(o["n"])
            outs["mu"].append(o["mu"])
        n = torch.cat(outs["n"]).reshape(B, L * f, A)
        mu = torch.cat(outs["mu"]).reshape(B, L * f, A)
        mu25, n25 = pool_nb(mu, n, f, dim=1)
        return {"n": n25, "mu": mu25}


__all__ = ["MAX_WINDOWS", "ResAdapter", "tables_of"]
