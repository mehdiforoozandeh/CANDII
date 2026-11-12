"""Extra-input readers and the reverse-complement transform (chunk XD of plan Amendment 3,
pinned interface E): RNA-seq track windows, the z-scored expression vector, the training-cell
index, and `revcomp_window` on synthetic arrays and on a real `OnionDataset` window."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import contract as C  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi.dataset import OnionDataset, read_tsv  # noqa: E402
from minicandi.extra_data import (Expression, RnaTracks, cell_index,  # noqa: E402
                                  revcomp_window)


# ---------------------------------------------------------------- RNA-seq tracks


@pytest.fixture()
def rna_dir(tmp_path):
    d = tmp_path / "tracks"
    d.mkdir()
    plus = np.arange(100, dtype=np.float16)
    minus = (np.arange(100, dtype=np.float16) * 2)
    with h5py.File(d / "cellA.h5", "w") as f:
        f["plus/chr1"] = plus
        f["minus/chr1"] = minus
    return d


def test_rna_window_values(rna_dir):
    rt = RnaTracks(rna_dir)
    x, ok = rt.window("cellA", "chr1", 10, 5)
    assert ok and x.dtype == np.float32 and x.shape == (5, 2)
    np.testing.assert_allclose(x[:, 0], np.log1p(np.arange(10, 15)), rtol=1e-6)
    np.testing.assert_allclose(x[:, 1], np.log1p(2 * np.arange(10, 15)), rtol=1e-6)


def test_rna_window_past_end_is_zero(rna_dir):
    x, ok = RnaTracks(rna_dir).window("cellA", "chr1", 97, 6)
    assert ok
    np.testing.assert_allclose(x[:3, 0], np.log1p([97, 98, 99]), rtol=1e-6)
    assert (x[3:] == 0).all()


def test_rna_window_missing_cell_or_chrom(rna_dir):
    rt = RnaTracks(rna_dir)
    for cell, chrom in (("cellB", "chr1"), ("cellA", "chr2")):
        x, ok = rt.window(cell, chrom, 0, 8)
        assert not ok and x.shape == (8, 2) and x.dtype == np.float32 and (x == 0).all()


# ---------------------------------------------------------------- expression


def _write(path, rows):
    with open(path, "w", newline="") as fh:
        csv.writer(fh, delimiter="\t").writerows(rows)


@pytest.fixture()
def expr_files(tmp_path):
    rng = np.random.default_rng(0)
    cells = ["c1", "c2", "c3", "c4"]
    genes = [f"ENSG{i:011d}" for i in range(6)]
    vals = rng.random((4, 6)) * 5
    vals[:, 5] = 1.5                                  # a constant gene: SD 0
    _write(tmp_path / "expression.tsv",
           [["cell"] + [g + ".7" for g in genes]] +
           [[c] + [f"{v:.6f}" for v in row] for c, row in zip(cells, vals)])
    # genes.tsv carries another version suffix and a shuffled order
    order = [3, 0, 5, 1]
    _write(tmp_path / "genes.tsv", [["gene_id"]] + [[genes[i] + ".12"] for i in order])
    vals = np.round(vals, 6)
    return tmp_path, cells, order, vals


def test_expression_zscore_on_training_cells(expr_files):
    d, cells, order, vals = expr_files
    train = ["c1", "c2", "c3"]
    ex = Expression(d / "expression.tsv", d / "genes.tsv", train_cells=train)
    ref = vals[:3][:, order]
    mean, sd = ref.mean(0), ref.std(0)
    sd[sd == 0] = 1.0
    for i, c in enumerate(cells):
        v, ok = ex.vector(c)
        assert ok and v.dtype == np.float32 and v.shape == (len(order),)
        np.testing.assert_allclose(v, (vals[i, order] - mean) / sd, rtol=1e-5, atol=1e-6)
    z = np.stack([ex.vector(c)[0] for c in train])
    np.testing.assert_allclose(z.mean(0), 0, atol=1e-6)
    np.testing.assert_allclose(z[:, [0, 1, 3]].std(0), 1, atol=1e-5)
    assert (z[:, 2] == 0).all()                       # the constant gene is centred only


def test_expression_missing_cell(expr_files):
    d, *_ = expr_files
    v, ok = Expression(d / "expression.tsv", d / "genes.tsv").vector("nope")
    assert not ok and v.shape == (4,) and (v == 0).all()


def test_expression_gene_by_cell_orientation(expr_files):
    d, cells, order, vals = expr_files
    genes = [f"ENSG{i:011d}" for i in range(6)]
    _write(d / "expr_t.tsv", [["gene_id"] + cells] +
           [[g] + [f"{vals[i, j]:.6f}" for i in range(4)] for j, g in enumerate(genes)])
    a = Expression(d / "expression.tsv", d / "genes.tsv")
    b = Expression(d / "expr_t.tsv", d / "genes.tsv")
    for c in cells:
        np.testing.assert_array_equal(a.vector(c)[0], b.vector(c)[0])


def test_expression_missing_gene_raises(expr_files):
    d, *_ = expr_files
    _write(d / "genes_bad.tsv", [["gene_id"], ["ENSG99999999999"]])
    with pytest.raises(ValueError):
        Expression(d / "expression.tsv", d / "genes_bad.tsv")


# ---------------------------------------------------------------- cell index


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    return synth.make_synthetic_tables(tmp_path_factory.mktemp("xd"))


def test_cell_index(tables):
    idx = cell_index(tables)
    rows = read_tsv(Path(tables) / "measurements.tsv")
    want = sorted({r["cell"] for r in rows if r["in_eic"] == "train"})
    assert list(idx) == want
    assert list(idx.values()) == list(range(1, len(want) + 1))
    assert 0 not in idx.values()


# ---------------------------------------------------------------- reverse complement


def _window(L=6, d_emb=3, rng=None):
    rng = rng or np.random.default_rng(1)
    return {
        "x_counts": torch.from_numpy(rng.integers(0, 9, (L, C.A)).astype(np.int32)),
        "x_avail": torch.ones(C.A, dtype=torch.bool),
        "x_bin_mask": torch.from_numpy(rng.random((L, C.A)) < 0.3),
        "x_dna": torch.from_numpy(rng.integers(0, 5, L * C.RES).astype(np.uint8)),
        "c_in": torch.from_numpy(rng.random((C.A, C.K_META)).astype(np.float32)),
        "y_counts": torch.from_numpy(rng.integers(0, 9, (L, C.A)).astype(np.int32)),
        "avg": torch.from_numpy(rng.random((L, C.A)).astype(np.float32)),
        "x_dna_emb": torch.from_numpy(rng.random((L, d_emb)).astype(np.float32)),
        "x_rna": torch.from_numpy(rng.random((L, 2)).astype(np.float32)),
        "x_rna_avail": torch.tensor(True),
        "x_cell_vec": torch.from_numpy(rng.random(7).astype(np.float32)),
        "cell_id": torch.tensor(3),
        "meta": {"cell": "c1", "chrom": "chr1", "start_bin": 4},
    }


def _assert_equal(a: dict, b: dict):
    assert set(a) == set(b)
    for k in a:
        if isinstance(a[k], torch.Tensor):
            assert a[k].dtype == b[k].dtype, k
            assert torch.equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


def test_revcomp_contents():
    w = _window()
    r = revcomp_window(w)
    L = w["x_counts"].shape[0]
    for k in ("x_counts", "x_bin_mask", "y_counts", "avg", "x_dna_emb"):
        assert torch.equal(r[k], torch.flip(w[k], (0,))), k
    assert torch.equal(r["x_rna"][:, 0], torch.flip(w["x_rna"][:, 1], (0,)))
    assert torch.equal(r["x_rna"][:, 1], torch.flip(w["x_rna"][:, 0], (0,)))
    comp = {0: 3, 1: 2, 2: 1, 3: 0, 4: 4}
    d = w["x_dna"].numpy()
    want = np.array([comp[int(c)] for c in d[::-1]], np.uint8)
    np.testing.assert_array_equal(r["x_dna"].numpy(), want)
    assert r["x_dna"].dtype == torch.uint8 and r["x_dna"].shape == (L * C.RES,)
    # bin i of the reversed window holds the reverse complement of bin L-1-i's bases
    np.testing.assert_array_equal(
        r["x_dna"].numpy()[:C.RES],
        np.array([comp[int(c)] for c in d[(L - 1) * C.RES:][::-1]], np.uint8))
    for k in ("x_avail", "c_in", "x_rna_avail", "x_cell_vec", "cell_id"):
        assert torch.equal(r[k], w[k]), k
    assert r["meta"] == w["meta"]
    # the input is not modified
    assert not torch.equal(r["x_counts"], w["x_counts"])


def test_revcomp_involution_torch_numpy_batched():
    w = _window()
    _assert_equal(revcomp_window(revcomp_window(w)), w)
    wn = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in w.items()}
    rn = revcomp_window(wn)
    for k, v in revcomp_window(w).items():
        if isinstance(v, torch.Tensor):
            np.testing.assert_array_equal(rn[k], v.numpy())
    # a collated batch: L on axis 1
    ws = [_window(rng=np.random.default_rng(s)) for s in range(3)]
    keys = ("x_counts", "x_bin_mask", "x_dna", "y_counts", "avg", "x_dna_emb", "x_rna")
    batch = {k: torch.stack([x[k] for x in ws]) for k in keys}
    rb = revcomp_window(batch)
    for i, x in enumerate(ws):
        rx = revcomp_window(x)
        for k in keys:
            assert torch.equal(rb[k][i], rx[k]), k
    _assert_equal(revcomp_window(rb), batch)


def test_revcomp_on_dataset_window(tables):
    L = 64
    ds = OnionDataset(tables, "train", {"context_bins": L}, seed=0)
    plan = ds.plan(np.random.default_rng(0))
    w = ds.materialize(plan, np.random.default_rng(0))
    r = revcomp_window(w)
    for k in ("x_counts", "x_bin_mask", "y_counts"):
        assert torch.equal(r[k], torch.flip(w[k], (0,)))
    dna = w["x_dna"].numpy()
    rd = r["x_dna"].numpy()
    n = dna == 4
    assert (rd[::-1][n] == 4).all() and (rd[::-1][~n] == 3 - dna[~n]).all()
    C.validate_batch({k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v])
                      for k, v in r.items()})
    _assert_equal(revcomp_window(r), w)
