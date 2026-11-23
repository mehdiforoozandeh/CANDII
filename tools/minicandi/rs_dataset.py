"""The read-start dataset (t392 chunk C6): `OnionDataset` on stranded 25 bp read-start counts.

    res_dataset.make_dataset(tables, mode, {..., "read_start": true,
                                            "read_start_store": <store root>,
                                            "read_start_average_h5": <file>}, seed)

`ReadStartDataset(tables_dir, mode, config, seed)` is `OnionDataset` with every read of counts
sent to the read-start store of chunk C3: `<read_start_store>/<measurement id>/starts.h5` (`pr1`,
`pr2`: `starts_pr1.h5`, `starts_pr2.h5`), one dataset per chromosome, `[n_bins, 2]` (column 0
plus-strand starts, column 1 minus-strand starts). The tables stay the read-count tables
(`eic_v1_2`); only the counts change. A new class, not a patch of the 1 bp `ResDataset`
(decision D14).

  read(mid, chrom, start, L, which) -> int32 [L]          the strand sum of the starts
  read_strands(mid, chrom, start, L, which) -> int32 [L, 2]   the store slice
  window(...)  the parent's window, plus `x_counts_sub`, `y_counts_sub` int32 [L, A, 2]
               (0 where hidden or missing) and, under `average_input`, `avg_sub` float32
               [L, A, 2] with `avg` its channel sum
  average_strands(cell, chrom, start, L) -> float32 [L, A, 2]   leave-one-cell-out average

Unstranded tracks (`read_start.UNSTRANDED_TRACKS`, D11): in `x_counts_sub`, `y_counts_sub` and
`avg_sub` channel 0 holds the total and channel 1 is 0. `read_strands` returns the store as is.

The sampler is the parent's: `plan` and `materialize` are not overridden and `window` draws no
random number, so the window sequence per seed is that of `OnionDataset` on the same tables.

The average. `average_h5` keeps its read-count path (only the scorer's own `OnionDataset` reads
it). `read_start_average_h5` (built by `build_average_h5` below) holds per track the sum over the
same measurements of starts x 1e6 / depth_reads, `[n_bins, 2]`; the window's own cell is
subtracted as in `OnionDataset.average`. Without it, the average is computed from the store
directly (small tables only).

Refused (ValueError): `input_depth = fixed`, `aug_revcomp`, `two_step`, `res_bp` other than 25,
`res_in_sub` or `res_out_sub` above 1, a `store:` row, and a config without `read_start_store`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from minicandi import contract as C  # noqa: E402
from minicandi.components import res_bp_of, res_subs_of  # noqa: E402
from minicandi.dataset import FILES, OnionDataset, RoleError, read_tsv  # noqa: E402
from minicandi.read_start import STRANDS, UNSTRANDED_TRACKS  # noqa: E402

STORE_FILES = {"counts": "starts.h5", "pr1": "starts_pr1.h5", "pr2": "starts_pr2.h5"}
_UNSTRANDED_T = frozenset(C.TRACK_INDEX[a] for a in UNSTRANDED_TRACKS)


def _ids(attr) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in np.atleast_1d(attr)]


def _fold(a: np.ndarray) -> np.ndarray:
    """[..., A, 2] with the unstranded tracks' total in channel 0 and channel 1 set to 0."""
    for t in _UNSTRANDED_T:
        a[..., t, 0] = a[..., t, 0] + a[..., t, 1]
        a[..., t, 1] = 0
    return a


class ReadStartDataset(OnionDataset):
    """See the module docstring."""

    def __init__(self, tables_dir, mode: str, config: dict | None = None, seed: int = 0):
        cfg = dict(config or {})
        bad = []
        if cfg.get("input_depth", "native") == "fixed":
            bad.append("input_depth = fixed")
        if cfg.get("aug_revcomp"):
            bad.append("aug_revcomp")
        if cfg.get("two_step"):
            bad.append("two_step")
        if res_bp_of(cfg) != C.RES:
            bad.append(f"res_bp = {cfg.get('res_bp')}")
        if max(res_subs_of(cfg)) > 1:
            bad.append("res_in_sub or res_out_sub above 1")
        if not cfg.get("read_start_store"):
            bad.append("no read_start_store")
        if bad:
            raise ValueError(f"ReadStartDataset does not serve {bad}")
        super().__init__(tables_dir, mode, cfg, seed)
        rows = [m for m, r in self.rows.items() if not r["counts_path"].startswith("h5:")]
        if rows:
            raise ValueError(f"ReadStartDataset does not serve a store: row (or any row that is "
                             f"not h5:): {sorted(rows)[:5]}")
        self.rs_store = Path(self.config["read_start_store"])
        avg = self.config.get("read_start_average_h5")
        self.rs_average_h5 = Path(avg) if avg else None
        self._strand_cache: dict | None = None

    # ---------------------------------------------------------------- reading

    def start_path(self, mid: str, which: str = "counts") -> Path:
        self.check_row(mid)
        if which not in FILES:
            raise ValueError(f"which {which!r} not in {FILES}")
        return self.rs_store / mid / STORE_FILES[which]

    def read_strands(self, mid: str, chrom: str, start: int, L: int | None = None,
                     which: str = "counts") -> np.ndarray:
        """int32 [L, 2]: plus- and minus-strand starts of one measurement; bins past the
        chromosome end are 0."""
        L = self.L if L is None else int(L)
        key = (mid, chrom, int(start), L, which)
        if self._strand_cache is not None and key in self._strand_cache:
            return self._strand_cache[key].copy()
        self._fresh()
        out = np.zeros((L, STRANDS), np.int32)
        ds = self._h5file(self.start_path(mid, which))[chrom]
        end = min(int(start) + L, ds.shape[0])
        if end > int(start):
            out[: end - int(start)] = ds[int(start):end]
        if self._strand_cache is not None:
            self._strand_cache[key] = out.copy()
        return out

    def read(self, mid: str, chrom: str, start: int, L: int | None = None,
             which: str = "counts") -> np.ndarray:
        """int32 [L]: the strand sum of the starts."""
        return self.read_strands(mid, chrom, start, L, which).sum(1, dtype=np.int32)

    # ---------------------------------------------------------------- windows

    def window(self, chrom: str, start_bin: int, inputs: list, targets: list, **kw) -> dict:
        """`OnionDataset.window` on the starts, plus the strand keys (module docstring)."""
        avg = kw.pop("avg", None)
        want_avg = avg if avg is not None else self.config.get("average_input")
        halved = frozenset(kw.get("halves_tracks") or ())
        self._strand_cache = {}
        try:
            win = super().window(chrom, start_bin, inputs, targets, avg=False, **kw)
            meta = win["meta"]
            L = self.L
            x_sub = np.zeros((L, C.A, STRANDS), np.int32)
            y_sub = np.zeros((L, C.A, STRANDS), np.int32)
            m_in, m_out = meta["measurements_in"], meta["measurements_out"]
            f_in, f_out = meta["input_file"], meta["target_file"]
            for t, m in enumerate(m_in):
                if m:
                    x_sub[:, t] = self.read_strands(
                        m, chrom, start_bin, L, "pr1" if C.TRACKS[t] in halved else f_in)
            for t, m in enumerate(m_out):
                if not m:
                    continue
                if m_in[t] and C.TRACKS[t] in halved:
                    y_sub[:, t] = self.read_strands(m, chrom, start_bin, L, "pr2")
                elif m_in[t] == m and f_out == f_in:
                    y_sub[:, t] = x_sub[:, t]            # same view: the unmasked input
                else:
                    y_sub[:, t] = self.read_strands(m, chrom, start_bin, L, f_out)
        finally:
            self._strand_cache = None
        x_sub[win["x_bin_mask"].numpy()] = 0
        _fold(x_sub)
        _fold(y_sub)
        if not np.array_equal(x_sub.sum(-1), win["x_counts"].numpy()):
            raise ValueError(f"{chrom}:{start_bin}: the input strands do not sum to x_counts")
        if not np.array_equal(y_sub.sum(-1), win["y_counts"].numpy()):
            raise ValueError(f"{chrom}:{start_bin}: the target strands do not sum to y_counts")
        win["x_counts_sub"] = torch.from_numpy(x_sub)
        win["y_counts_sub"] = torch.from_numpy(y_sub)
        if want_avg:
            a = self.average_strands(meta["cell"], chrom, start_bin, L)
            win["avg_sub"] = torch.from_numpy(a)
            win["avg"] = torch.from_numpy(a.sum(-1, dtype=np.float32))
        return win

    # ---------------------------------------------------------------- average

    def _rate_strands(self, mid: str, chrom: str, start: int, L: int) -> np.ndarray | None:
        r = self.rows[mid]
        depth = float(r["depth_reads"]) if r["depth_reads"] else 0.0
        if depth <= 0:
            return None
        return self.read_strands(mid, chrom, start, L).astype(np.float64) * (1e6 / depth)

    def average(self, cell: str, chrom: str, start: int, L: int | None = None) -> np.ndarray:
        """float32 [L, A]: the channel sum of `average_strands`."""
        return self.average_strands(cell, chrom, start, L).sum(-1, dtype=np.float32)

    def average_strands(self, cell: str, chrom: str, start: int,
                        L: int | None = None) -> np.ndarray:
        """Leave-one-cell-out per-position average rate per strand, float32 [L, A, 2], by the
        rule of `OnionDataset.average`, read from `read_start_average_h5` when the config sets
        it. Unstranded tracks hold their total in channel 0."""
        L = self.L if L is None else int(L)
        out = np.zeros((L, C.A, STRANDS), np.float32)
        if self.rs_average_h5 is not None:
            f = self._h5file(self.rs_average_h5)
            sources = set(self.train_base.values())
            for track, g in f.items():
                t = C.TRACK_INDEX[track]
                n = int(g.attrs["n_cells"])
                summed = _ids(g.attrs["measurement_ids"])
                bad = sorted(set(summed) - sources)
                if bad:
                    raise RoleError(f"{self.rs_average_h5}: track {track} sums {bad}, which this "
                                    f"dataset may not average over (roles {self.train_roles})")
                mine = [m for m in summed if self.rows[m]["cell"] == cell]
                own = mine[0] if mine else None
                if chrom not in g:
                    continue
                ds = g[chrom]
                s = np.zeros((L, STRANDS), np.float64)
                end = min(int(start) + L, ds.shape[0])
                if end > int(start):
                    s[: end - int(start)] = ds[int(start):end]
                if own is not None:
                    rate = self._rate_strands(own, chrom, start, L)
                    if rate is not None:
                        s -= rate
                        n -= 1
                if n > 0:
                    out[:, t] = np.maximum(s, 0.0) / n
            return _fold(out)
        n = np.zeros(C.A, np.int64)
        acc = np.zeros((L, C.A, STRANDS), np.float64)
        for (c, a), m in self.train_base.items():
            if c == cell:
                continue
            rate = self._rate_strands(m, chrom, start, L)
            if rate is None:
                continue
            t = C.TRACK_INDEX[a]
            acc[:, t] += rate
            n[t] += 1
        nz = n > 0
        out[:, nz] = (acc[:, nz] / n[nz][None, :, None]).astype(np.float32)
        return _fold(out)


def build_average_h5(reference_average_h5, store, tables, out) -> Path:
    """Write the read-start average file. Per track, the `measurement_ids` and `n_cells` are
    copied from the attributes of the read-count average file `reference_average_h5` (that set
    holds single-end views such as `ENCSR191FOV.pooled.uniform.se`; it is not recomputed), and
    one dataset per chromosome of that file, float32 [n_bins, 2], holds the sum over those
    measurements of starts x 1e6 / depth_reads (`depth_reads` of `<tables>/measurements.tsv`,
    starts of `<store>/<id>/starts.h5`). A listed id with no row, no depth or no store file
    raises."""
    store, out = Path(store), Path(out)
    rows = {r["measurement_id"]: r for r in read_tsv(Path(tables) / "measurements.tsv")}
    out.parent.mkdir(parents=True, exist_ok=True)
    step = 1 << 20
    with h5py.File(reference_average_h5, "r") as fr, h5py.File(out, "w") as fo:
        fo.attrs["counting_rule"] = "read_start"
        fo.attrs["strand_columns"] = "plus,minus"
        fo.attrs["reference_average_h5"] = str(reference_average_h5)
        for track, gr in fr.items():
            mids = _ids(gr.attrs["measurement_ids"])
            missing = [m for m in mids if m not in rows]
            if missing:
                raise KeyError(f"track {track}: {missing} not in {tables}/measurements.tsv")
            depth = {m: float(rows[m]["depth_reads"] or 0) for m in mids}
            nodepth = [m for m, d in depth.items() if d <= 0]
            if nodepth:
                raise ValueError(f"track {track}: {nodepth} have no depth_reads")
            go = fo.create_group(track)
            go.attrs["n_cells"] = gr.attrs["n_cells"]
            go.attrs["measurement_ids"] = np.asarray(mids, dtype=h5py.string_dtype())
            files = [h5py.File(store / m / STORE_FILES["counts"], "r") for m in mids]
            try:
                for chrom in gr.keys():
                    nb = int(gr[chrom].shape[0])
                    s = np.zeros((nb, STRANDS), np.float64)
                    for m, f in zip(mids, files):
                        ds = f[chrom]
                        n = min(nb, int(ds.shape[0]))
                        for k in range(0, n, step):
                            e = min(k + step, n)
                            s[k:e] += ds[k:e].astype(np.float64) * (1e6 / depth[m])
                    go.create_dataset(chrom, data=s.astype(np.float32),
                                      chunks=(min(4096, nb), STRANDS), compression="gzip")
            finally:
                for f in files:
                    f.close()
    return out


__all__ = ["STORE_FILES", "ReadStartDataset", "build_average_h5"]
