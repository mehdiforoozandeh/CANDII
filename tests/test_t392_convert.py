"""t392 chunk C1, case (4): strand channels and the conversion from read starts to overlap counts.

Written against `cruxvault/INSTRUCTIONS/t392_read_starts/INSTRUCTIONS.md` section 2 (map (1),
moment matching), decision D11 (ATAC-seq and DNase-seq have one channel) and the Interfaces lines
of plan chunk C4, before `tools/minicandi/read_start.py` existed.

Conventions taken from the plan: a span kernel is `[2, 17]`, entry `[s, 8 + k]` the weight that a
start at bin j gives to bin j + k; `mu[i] = sum_s sum_k w[s, 8 + k] * mu_sub[i - k, s]`; outside
the array the edge bin's values are used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

REACH = 8


def _rsm():
    from minicandi import read_start
    return read_start


def _contract():
    from minicandi import contract
    return contract


def _edge_conv(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """x [L, 2], kernel [2, 17] -> [L, 2] per-strand contributions summed over k, edge padded."""
    L = x.shape[0]
    pad = np.pad(x, ((REACH, REACH), (0, 0)), mode="edge")
    out = np.zeros_like(x, dtype=np.float64)
    for s in range(2):
        for k in range(-REACH, REACH + 1):
            out[:, s] += kernel[s, REACH + k] * pad[REACH - k:REACH - k + L, s]
    return out


def _box(lo: int, hi: int) -> np.ndarray:
    row = np.zeros(17)
    row[REACH + lo:REACH + hi + 1] = 1.0
    return row


def _expected_nb(mu_sub, n_sub, kernel):
    """The moment match written out: per contribution c = w * mu_sub[i - k, s], mean c and
    variance c + c^2 / n_sub[i - k, s]; sum, then size = mu^2 / (var - mu)."""
    L = mu_sub.shape[0]
    pm = np.pad(mu_sub, ((REACH, REACH), (0, 0)), mode="edge")
    pn = np.pad(n_sub, ((REACH, REACH), (0, 0)), mode="edge")
    mu = np.zeros(L)
    var = np.zeros(L)
    for s in range(2):
        for k in range(-REACH, REACH + 1):
            w = kernel[s, REACH + k]
            c = w * pm[REACH - k:REACH - k + L, s]
            mu += c
            var += c + c ** 2 / pn[REACH - k:REACH - k + L, s]
    return mu, mu ** 2 / (var - mu)


# --------------------------------------------------------------------------- strand channels

def test_two_strands_and_the_unstranded_tracks():
    rsm = _rsm()
    assert rsm.STRANDS == 2
    assert set(rsm.UNSTRANDED_TRACKS) == {"ATAC-seq", "DNase-seq"}


def test_channel_mask_is_false_only_at_channel_one_of_atac_and_dnase():
    rsm, C = _rsm(), _contract()
    m = np.asarray(rsm.channel_mask())
    assert m.shape == (36, 2) and m.dtype == bool
    assert m[:, 0].all()
    off = {C.TRACKS[i] for i in np.flatnonzero(~m[:, 1])}
    assert off == {"ATAC-seq", "DNase-seq"}
    assert m[C.TRACK_INDEX["H3K27ac"], 1] and m[C.TRACK_INDEX[C.CONTROL], 1]


def test_unstranded_kernel_is_the_read_weighted_mix_of_the_two_rows():
    k = np.stack([_box(0, 3), _box(-3, 0)])
    got = np.asarray(_rsm().unstranded_kernel(k, np.array([300, 100])), np.float64)
    assert got.shape == (17,)
    np.testing.assert_allclose(got, 0.75 * k[0] + 0.25 * k[1])


# --------------------------------------------------------------------------- convert_nb, means

def test_box_kernel_mean_is_the_box_convolution_of_each_strand_summed():
    rng = np.random.default_rng(0)
    L = 64
    mu_sub = rng.uniform(0.1, 5.0, (L, 2))
    n_sub = rng.uniform(0.5, 20.0, (L, 2))
    kernel = np.stack([_box(0, 2), _box(-2, 0)])      # plus spans forward, minus backward
    mu, _ = _rsm().convert_nb(torch.tensor(mu_sub), torch.tensor(n_sub),
                              torch.tensor(kernel), True)
    want = _edge_conv(mu_sub, kernel).sum(axis=1)
    np.testing.assert_allclose(np.asarray(mu, np.float64), want, rtol=1e-5)


def test_mean_interior_bins_do_not_depend_on_the_edge_rule():
    L = 40
    mu_sub = np.zeros((L, 2))
    mu_sub[20, 0] = 3.0                                # one plus start
    mu_sub[25, 1] = 2.0                                # one minus start
    n_sub = np.full((L, 2), 10.0)
    kernel = np.stack([_box(0, 2), _box(-1, 1)])
    mu, _ = _rsm().convert_nb(torch.tensor(mu_sub), torch.tensor(n_sub),
                              torch.tensor(kernel), True)
    want = np.zeros(L)
    want[20:23] += 3.0
    want[24:27] += 2.0
    np.testing.assert_allclose(np.asarray(mu, np.float64), want, atol=1e-6)


def test_leading_batch_axes_are_carried():
    rng = np.random.default_rng(1)
    mu_sub = rng.uniform(0.1, 5.0, (3, 4, 32, 2))
    n_sub = rng.uniform(0.5, 20.0, (3, 4, 32, 2))
    kernel = np.stack([_box(0, 2), _box(-2, 0)])
    mu, n = _rsm().convert_nb(torch.tensor(mu_sub), torch.tensor(n_sub), torch.tensor(kernel), True)
    assert tuple(mu.shape) == (3, 4, 32) and tuple(n.shape) == (3, 4, 32)
    one, _ = _rsm().convert_nb(torch.tensor(mu_sub[2, 1]), torch.tensor(n_sub[2, 1]),
                               torch.tensor(kernel), True)
    np.testing.assert_allclose(np.asarray(mu[2, 1]), np.asarray(one), rtol=1e-6)


# --------------------------------------------------------------------------- convert_nb, size

def test_size_is_mu_squared_over_variance_minus_mu():
    C = _contract()
    rng = np.random.default_rng(2)
    L = 48
    mu_sub = rng.uniform(0.5, 4.0, (L, 2))
    n_sub = rng.uniform(1.0, 8.0, (L, 2))
    kernel = np.stack([_box(0, 3) * 0.9, _box(-3, 0) * 0.7])   # non-box weights: thinning
    _, n = _rsm().convert_nb(torch.tensor(mu_sub), torch.tensor(n_sub), torch.tensor(kernel), True)
    _, want = _expected_nb(mu_sub, n_sub, kernel)
    want = np.clip(want, C.N_MIN, C.N_MAX)
    np.testing.assert_allclose(np.asarray(n, np.float64), want, rtol=1e-4)


def test_poisson_strand_means_give_the_upper_size_clamp():
    C = _contract()
    L = 32
    mu_sub = np.full((L, 2), 2.0)
    n_sub = np.full((L, 2), 1e12)                      # NB with a huge size is a Poisson
    kernel = np.stack([_box(0, 2), _box(-2, 0)])
    _, n = _rsm().convert_nb(torch.tensor(mu_sub), torch.tensor(n_sub), torch.tensor(kernel), True)
    np.testing.assert_allclose(np.asarray(n, np.float64), float(C.N_MAX), rtol=1e-6)


def test_size_stays_inside_the_contract_clamp():
    C = _contract()
    rng = np.random.default_rng(3)
    mu_sub = rng.uniform(1e-4, 50.0, (128, 2))
    n_sub = rng.uniform(1e-3, 1e6, (128, 2))
    kernel = np.stack([_box(0, 4), _box(-4, 0)])
    _, n = _rsm().convert_nb(torch.tensor(mu_sub), torch.tensor(n_sub), torch.tensor(kernel), True)
    n = np.asarray(n, np.float64)
    assert np.all(np.isfinite(n))
    assert n.min() >= float(C.N_MIN) * (1 - 1e-6) and n.max() <= float(C.N_MAX) * (1 + 1e-6)


# --------------------------------------------------------------------------- unstranded tracks

def test_unstranded_conversion_ignores_channel_one():
    rng = np.random.default_rng(4)
    L = 40
    mu_sub = rng.uniform(0.1, 5.0, (L, 2))
    n_sub = rng.uniform(0.5, 20.0, (L, 2))
    row = _box(-1, 2)
    kernel = torch.tensor(np.stack([row, row]))
    rsm = _rsm()
    mu_a, n_a = rsm.convert_nb(torch.tensor(mu_sub), torch.tensor(n_sub), kernel, False)
    mu_sub2, n_sub2 = mu_sub.copy(), n_sub.copy()
    mu_sub2[:, 1] = rng.uniform(10.0, 90.0, L)
    n_sub2[:, 1] = rng.uniform(0.01, 0.1, L)
    mu_b, n_b = rsm.convert_nb(torch.tensor(mu_sub2), torch.tensor(n_sub2), kernel, False)
    np.testing.assert_allclose(np.asarray(mu_a), np.asarray(mu_b), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(n_a), np.asarray(n_b), rtol=1e-6)
    only0 = np.stack([mu_sub[:, 0], np.zeros(L)], axis=1)
    want = _edge_conv(only0, np.stack([row, row]))[:, 0]
    np.testing.assert_allclose(np.asarray(mu_a, np.float64), want, rtol=1e-5)


# --------------------------------------------------------------------------- stitching

def _metas(n, L, start=1000, chrom="chr1", cell="s0", targets=("m_t",)):
    return [{"cell": cell, "chrom": chrom, "start_bin": start + i * L, "targets": list(targets)}
            for i in range(n)]


def _runs(rsm, metas, L):
    return [list(map(int, r)) for r in rsm.stitch(metas, L)]


def test_sixteen_consecutive_windows_stitch_into_one_run():
    assert _runs(_rsm(), _metas(16, 64), 64) == [list(range(16))]


def test_a_gap_a_new_chromosome_or_new_targets_break_the_run():
    rsm, L = _rsm(), 64
    m = _metas(6, L)
    m[3] = dict(m[3], start_bin=m[3]["start_bin"] + 1)           # gap before window 3
    runs = _runs(rsm, m, L)
    assert sorted(i for r in runs for i in r) == list(range(6))
    assert [0, 1, 2] in runs and all(3 not in r or r[0] == 3 for r in runs)
    m = _metas(4, L) + _metas(4, L, start=1000 + 4 * L, chrom="chr2")
    assert len(_runs(rsm, m, L)) == 2
    m = _metas(4, L) + _metas(4, L, start=1000 + 4 * L, targets=("other",))
    assert len(_runs(rsm, m, L)) == 2


def test_conversion_of_stitched_windows_equals_conversion_of_their_concatenation():
    """A shuffled batch: one run of 5 consecutive chr1 windows, a run of 3 after a gap, and one
    chr2 window placed where it would extend the first run. Each stitched run is converted as one
    array and scattered back; the reference is the moment match written out in numpy on each
    run's genomic concatenation."""
    C = _contract()
    rsm = _rsm()
    rng = np.random.default_rng(5)
    L, s0 = 16, 1000
    metas = ([{"cell": "s0", "chrom": "chr1", "start_bin": s0 + i * L, "targets": ["m"]}
              for i in range(5)]
             + [{"cell": "s0", "chrom": "chr1", "start_bin": s0 + 5 * L + 7 + i * L,
                 "targets": ["m"]} for i in range(3)]
             + [{"cell": "s0", "chrom": "chr2", "start_bin": s0 + 5 * L, "targets": ["m"]}])
    groups = [list(range(5)), [5, 6, 7], [8]]
    perm = rng.permutation(len(metas))                 # batch position j holds window perm[j]
    pos = {int(w): j for j, w in enumerate(perm)}
    batch_metas = [metas[int(w)] for w in perm]
    B = len(metas)
    mu_sub = rng.uniform(0.1, 5.0, (B, L, 2))
    n_sub = rng.uniform(0.5, 20.0, (B, L, 2))
    kernel_np = np.stack([_box(0, 3) * 0.8, _box(-3, 0)])
    kernel = torch.tensor(kernel_np)

    runs = _runs(rsm, batch_metas, L)
    assert sorted(i for r in runs for i in r) == list(range(B))
    assert sorted(sorted(r) for r in runs) == sorted(sorted(pos[w] for w in g) for g in groups)

    mu_got = np.zeros((B, L))
    n_got = np.zeros((B, L))
    for r in runs:
        r = sorted(r, key=lambda j: batch_metas[j]["start_bin"])
        mu, n = rsm.convert_nb(torch.tensor(mu_sub[r].reshape(-1, 2)),
                               torch.tensor(n_sub[r].reshape(-1, 2)), kernel, True)
        mu_got[r] = np.asarray(mu, np.float64).reshape(len(r), L)
        n_got[r] = np.asarray(n, np.float64).reshape(len(r), L)

    for g in groups:
        idx = [pos[w] for w in g]                      # genomic order
        m_cat, n_cat = mu_sub[idx].reshape(-1, 2), n_sub[idx].reshape(-1, 2)
        mu_want, n_want = _expected_nb(m_cat, n_cat, kernel_np)
        n_want = np.clip(n_want, C.N_MIN, C.N_MAX)
        np.testing.assert_allclose(mu_got[idx].reshape(-1), mu_want, rtol=1e-5)
        np.testing.assert_allclose(n_got[idx].reshape(-1), n_want, rtol=1e-4)
    # one window converted alone differs from its stitched value at its inner edges
    j = pos[1]
    mu_1, _ = rsm.convert_nb(torch.tensor(mu_sub[j]), torch.tensor(n_sub[j]), kernel, True)
    assert not np.allclose(np.asarray(mu_1, np.float64), mu_got[j])


# --------------------------------------------------------------------------- strand_output

def test_strand_output_uses_channel_zero_alone_for_unstranded_tracks():
    rsm, C = _rsm(), _contract()
    rng = np.random.default_rng(6)
    B, L, A = 2, 16, C.A
    mu_sub = torch.tensor(rng.uniform(0.1, 5.0, (B, L, A, 2)), dtype=torch.float32)
    n_sub = torch.tensor(rng.uniform(0.5, 20.0, (B, L, A, 2)), dtype=torch.float32)
    out = rsm.strand_output({"mu_sub": mu_sub, "n_sub": n_sub})
    assert tuple(out["mu"].shape) == (B, L, A) and tuple(out["n"].shape) == (B, L, A)
    for t in ("ATAC-seq", "DNase-seq"):
        i = C.TRACK_INDEX[t]
        np.testing.assert_allclose(out["mu"][..., i].numpy(), mu_sub[..., i, 0].numpy(), rtol=1e-6)
        np.testing.assert_allclose(out["n"][..., i].numpy(), n_sub[..., i, 0].numpy(), rtol=1e-5)
    i = C.TRACK_INDEX["H3K27ac"]                       # stranded: the pooled mean is the sum
    np.testing.assert_allclose(out["mu"][..., i].numpy(), mu_sub[..., i, :].sum(-1).numpy(),
                               rtol=1e-5)


def test_strand_output_pooled_size_of_a_stranded_track_is_pool_nb():
    from minicandi.losses import pool_nb
    rsm, C = _rsm(), _contract()
    rng = np.random.default_rng(7)
    mu_sub = torch.tensor(rng.uniform(0.1, 5.0, (2, 8, C.A, 2)), dtype=torch.float32)
    n_sub = torch.tensor(rng.uniform(0.5, 20.0, (2, 8, C.A, 2)), dtype=torch.float32)
    out = rsm.strand_output({"mu_sub": mu_sub, "n_sub": n_sub})
    mu_p, n_p = pool_nb(mu_sub, n_sub, 2, dim=-1)
    for a in ("H3K27ac", "H3K4me3", C.CONTROL):
        i = C.TRACK_INDEX[a]
        torch.testing.assert_close(out["n"][..., i], n_p[..., i, 0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(out["mu"][..., i], mu_p[..., i, 0], rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- read_start_loss

def _loss_case(seed=8, B=2, L=16):
    C = _contract()
    rng = np.random.default_rng(seed)
    y_sub = torch.tensor(rng.poisson(2.0, (B, L, C.A, 2)), dtype=torch.int32)
    for a in ("ATAC-seq", "DNase-seq"):                # the dataset's unstranded layout
        t = C.TRACK_INDEX[a]
        y_sub[..., t, 0] += y_sub[..., t, 1]
        y_sub[..., t, 1] = 0
    batch = {"y_counts_sub": y_sub, "y_counts": y_sub.sum(-1).to(torch.int32),
             "y_kind": torch.full((B, C.A), C.KIND_IMPUTE, dtype=torch.int64),
             "y_avail": torch.ones((B, C.A), dtype=torch.bool),
             "x_avail": torch.zeros((B, C.A), dtype=torch.bool),
             "x_bin_mask": torch.zeros((B, L, C.A), dtype=torch.bool)}
    out = {"mu_sub": torch.tensor(rng.uniform(0.5, 4.0, (B, L, C.A, 2)), dtype=torch.float32),
           "n_sub": torch.tensor(rng.uniform(1.0, 10.0, (B, L, C.A, 2)), dtype=torch.float32)}
    return batch, out


def _loss(batch, out):
    rsm = _rsm()
    full = dict(out)
    full.update(rsm.strand_output(out))
    loss, _ = rsm.read_start_loss(full, batch, {})
    return float(loss)


def test_read_start_loss_ignores_channel_one_of_unstranded_tracks():
    C = _contract()
    batch, out = _loss_case()
    base = _loss(batch, out)
    assert np.isfinite(base)
    b2 = dict(batch)
    o2 = {k: v.clone() for k, v in out.items()}
    b2["y_counts_sub"] = batch["y_counts_sub"].clone()
    for a in ("ATAC-seq", "DNase-seq"):
        t = C.TRACK_INDEX[a]
        o2["mu_sub"][..., t, 1] = 37.0
        o2["n_sub"][..., t, 1] = 0.05
        b2["y_counts_sub"][..., t, 1] = 9
    assert _loss(b2, o2) == pytest.approx(base, rel=1e-6, abs=1e-7)


def test_read_start_loss_sees_channel_one_of_a_chip_track():
    C = _contract()
    batch, out = _loss_case()
    base = _loss(batch, out)
    o2 = {k: v.clone() for k, v in out.items()}
    o2["mu_sub"][..., C.TRACK_INDEX["H3K27ac"], 1] *= 3.0
    assert _loss(batch, o2) != pytest.approx(base, rel=1e-6)
