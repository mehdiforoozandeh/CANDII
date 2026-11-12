"""Dataset-side readers of the extra inputs (plan Amendment 3, pinned interface E) and the window
reversal of the `aug_revcomp` arm.

  RnaTracks(dir).window(cell, chrom, start_bin, L) -> (float32 [L, 2], bool)
      `<dir>/<cell>.h5` holds `/plus/<chrom>` and `/minus/<chrom>` (float16, the 25 bp mean of the
      unique-read signal). The window is log1p of that mean, plus strand in column 0, minus strand
      in column 1. Bins past the chromosome end are 0. A cell with no file, or a file with no
      dataset for the chromosome, gives zeros and False.
  Expression(tsv, genes_tsv, train_cells=None).vector(cell) -> (float32 [n_genes], bool)
      `expression.tsv` holds log1p TPM, one row per cell (first column `cell`, one column per gene)
      or one row per gene (first column a gene id, one column per cell). `genes.tsv` gives the
      genes and their order (column `gene_id`, else the first column). Gene ids match without
      their version suffix. Each gene is z-scored with the mean and SD over the training cells
      (`train_cells`; all cells of the table when None); a gene with SD 0 is centred only. A cell
      without a row gives zeros and False.
  cell_index(tables_dir) -> {cell: id}
      The cells of the `in_eic = train` rows of `measurements.tsv`, sorted, numbered from 1;
      0 is reserved for an unknown cell.
  revcomp_window(win) -> win'
      Reverses axis L of every per-position key, reverse-complements `x_dna` (codes 0..3 to 3..0,
      N = 4 kept) and swaps the plus and minus strand columns of `x_rna`. It is an involution.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - the minicandi environment has torch
    torch = None

#: per-position keys of pinned interfaces C and E, with their L axis in one window. `x_dna` is
#: per position at 1 bp (L * 25) and is handled apart.
PER_POSITION_KEYS = ("x_counts", "x_bin_mask", "y_counts", "avg", "y_base", "x_dna_emb", "x_rna")
DNA_KEY = "x_dna"
RNA_KEY = "x_rna"
N_CODE = 4


def _strip_version(gene_id: str) -> str:
    return gene_id.strip().split(".", 1)[0]


# ---------------------------------------------------------------- RNA-seq tracks


class RnaTracks:
    """Reader of the per-cell RNA-seq signal files of pinned interface E."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self._handles: dict = {}

    def path(self, cell: str) -> Path:
        return self.dir / f"{cell}.h5"

    def has(self, cell: str) -> bool:
        return self.path(cell).is_file()

    def _open(self, cell: str):
        import h5py
        h = self._handles.get(cell)
        if h is None:
            h = h5py.File(self.path(cell), "r")
            self._handles[cell] = h
        return h

    def window(self, cell: str, chrom: str, start_bin: int, L: int) -> tuple[np.ndarray, bool]:
        out = np.zeros((int(L), 2), np.float32)
        if not cell or not self.has(cell):
            return out, False
        h = self._open(cell)
        key_p, key_m = f"plus/{chrom}", f"minus/{chrom}"
        if key_p not in h or key_m not in h:
            return out, False
        s, e = int(start_bin), int(start_bin) + int(L)
        for col, key in ((0, key_p), (1, key_m)):
            ds = h[key]
            n = ds.shape[0]
            lo, hi = max(s, 0), min(e, n)
            if hi > lo:
                v = np.asarray(ds[lo:hi], np.float32)
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                out[lo - s:hi - s, col] = np.log1p(np.maximum(v, 0.0))
        return out, True

    def close(self) -> None:
        for h in self._handles.values():
            h.close()
        self._handles.clear()

    def __getstate__(self):
        return {"dir": self.dir, "_handles": {}}


# ---------------------------------------------------------------- expression vector


def _read_genes(genes_tsv) -> list[str]:
    with open(genes_tsv, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        col = "gene_id" if "gene_id" in (rd.fieldnames or ()) else rd.fieldnames[0]
        genes = [_strip_version(r[col]) for r in rd if r[col].strip()]
    if len(set(genes)) != len(genes):
        raise ValueError(f"{genes_tsv}: duplicate gene ids after the version suffix is removed")
    return genes


def _read_expression(tsv) -> dict[str, dict[str, float]]:
    """cell -> {gene id without version: log1p TPM}, in either table orientation."""
    with open(tsv, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    if not rows:
        raise ValueError(f"{tsv}: empty")
    head, body = rows[0], rows[1:]
    table: dict[str, dict[str, float]] = {}
    if head[0] == "cell":
        genes = [_strip_version(g) for g in head[1:]]
        for r in body:
            table[r[0]] = {g: float(v) for g, v in zip(genes, r[1:])}
    else:
        cells = head[1:]
        for c in cells:
            table[c] = {}
        for r in body:
            g = _strip_version(r[0])
            for c, v in zip(cells, r[1:]):
                table[c][g] = float(v)
    return table


class Expression:
    """Per-cell expression vector of pinned interface E (`x_cell_vec`)."""

    def __init__(self, tsv, genes_tsv, train_cells=None):
        self.genes = _read_genes(genes_tsv)
        self.n_genes = len(self.genes)
        table = _read_expression(tsv)
        self.cells = sorted(table)
        mat = np.full((len(self.cells), self.n_genes), np.nan, np.float64)
        for i, c in enumerate(self.cells):
            row = table[c]
            for j, g in enumerate(self.genes):
                if g in row:
                    mat[i, j] = row[g]
        missing = np.isnan(mat).any(axis=0)
        if missing.any():
            bad = [self.genes[j] for j in np.flatnonzero(missing)[:5]]
            raise ValueError(f"{tsv}: {int(missing.sum())} genes of {genes_tsv} lack a value in "
                             f"some cell, e.g. {bad}")
        ref = self.cells if train_cells is None else [c for c in self.cells
                                                      if c in set(train_cells)]
        if not ref:
            raise ValueError("no training cell has an expression row; the z-score is undefined")
        idx = [self.cells.index(c) for c in ref]
        self.ref_cells = ref
        self.mean = mat[idx].mean(axis=0)
        sd = mat[idx].std(axis=0)
        self.sd = np.where(sd > 0, sd, 1.0)
        self._z = {c: ((mat[i] - self.mean) / self.sd).astype(np.float32)
                   for i, c in enumerate(self.cells)}

    def has(self, cell: str) -> bool:
        return cell in self._z

    def vector(self, cell: str) -> tuple[np.ndarray, bool]:
        v = self._z.get(cell)
        if v is None:
            return np.zeros(self.n_genes, np.float32), False
        return v.copy(), True


# ---------------------------------------------------------------- cell index


def cell_index(tables_dir) -> dict[str, int]:
    """Sorted training cells of `measurements.tsv` (rows with `in_eic = train`), from 1."""
    with open(Path(tables_dir) / "measurements.tsv", newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for col in ("cell", "in_eic"):
            if col not in (rd.fieldnames or ()):
                raise ValueError(f"{tables_dir}/measurements.tsv lacks column {col!r}")
        cells = sorted({r["cell"] for r in rd if r["in_eic"] == "train" and r["cell"]})
    return {c: i + 1 for i, c in enumerate(cells)}


# ---------------------------------------------------------------- reverse complement


def _flip(x, axis: int):
    if torch is not None and isinstance(x, torch.Tensor):
        return torch.flip(x, dims=(axis,))
    return np.flip(np.asarray(x), axis=axis).copy()


def _complement(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return torch.where(x < N_CODE, 3 - x, x).to(x.dtype)
    a = np.asarray(x)
    return np.where(a < N_CODE, 3 - a, a).astype(a.dtype)


def revcomp_window(win: dict) -> dict:
    """The window read on the other strand. Works on one window (x_dna of rank 1, L on axis 0)
    or on a collated batch (x_dna of rank 2, L on axis 1). Other keys (covariates, availability,
    kinds, meta, x_cell_vec, cell_id) are passed through unchanged. Applied twice it returns the
    input."""
    if DNA_KEY in win:
        batched = win[DNA_KEY].ndim == 2
    elif "x_counts" in win:
        batched = win["x_counts"].ndim == 3
    else:
        raise ValueError("revcomp_window needs x_dna or x_counts to find the L axis")
    ax = 1 if batched else 0
    out = dict(win)
    for key in PER_POSITION_KEYS:
        if key in win and win[key] is not None:
            out[key] = _flip(win[key], ax)
    if RNA_KEY in out and out[RNA_KEY] is not None:
        # the plus strand read backwards is the minus strand of the reversed window
        out[RNA_KEY] = _flip(out[RNA_KEY], ax + 1)
    if DNA_KEY in win:
        out[DNA_KEY] = _complement(_flip(win[DNA_KEY], ax))
    return out
