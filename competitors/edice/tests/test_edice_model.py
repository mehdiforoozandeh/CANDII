"""Synthetic-tensor checks on the PyTorch eDICE.

Run from `competitors/edice/`:

    PYTHONPATH=. pytest tests/ -q

Deliberately NOT under the repo's top-level `tests/`: `competitors/` is not importable from
`candi` and must not enter the core gate (`pytest tests/ -q` + `tools/golden.py`).

What these can and cannot prove. They cannot prove numeric agreement with TensorFlow -- that needs
weights we will not produce, since RIVALS_PLAN §7.3 says read their code, do not run it. What they
do prove is that every structural claim the reimplementation makes is true: the parameter count the
paper published, the masking rule, the per-node averaging, the target read-out, and the invariances
the architecture is supposed to have. Numeric agreement is the Roadmap gate's job.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from edice_torch.data import FixedTargetSampler, TrainSampler
from edice_torch.metrics import per_track_mse, per_track_pearson, summarise
from edice_torch.model import CellAssayCrossFactoriser, SignalEmbedder, dense

ROADMAP = dict(n_cells=127, n_assays=24, embed_dim=256, decoder_hidden=2048, decoder_layers=2)


def make_inputs(b=8, s=30, t=5, n_cells=7, n_assays=4, seed=0):
    """A bin batch whose (cell, assay) support pairs are unique, as real tracks are."""
    g = torch.Generator().manual_seed(seed)
    pairs = [(c, a) for c in range(n_cells) for a in range(n_assays)]
    assert s <= len(pairs)
    chosen = [pairs[i] for i in torch.randperm(len(pairs), generator=g)[:s].tolist()]
    sc = torch.tensor([[c for c, _ in chosen]] * b)
    sa = torch.tensor([[a for _, a in chosen]] * b)
    vals = torch.randn(b, s, generator=g)
    tc = torch.randint(0, n_cells, (b, t), generator=g)
    ta = torch.randint(0, n_assays, (b, t), generator=g)
    return vals, sc, sa, tc, ta


def small_model(**kw):
    cfg = dict(n_cells=7, n_assays=4, embed_dim=16, intermediate_fc_dim=8, decoder_hidden=12,
               decoder_layers=2, n_attn_heads=4)
    cfg.update(kw)
    torch.manual_seed(0)
    return CellAssayCrossFactoriser(**cfg)


# --- the published parameter count -------------------------------------------------------------

def test_roadmap_parameter_count_matches_the_paper():
    """Supplementary Table 1 says eDICE needs "~ 6M" parameters. We land at 5,985,025.

    This is the one end-to-end check that the two towers, the four attention projections, the
    128-wide interspersed FFNs and the 512->2048->2048->1 decoder are all the widths they should
    be: get any of them wrong and the total moves by more than rounding.
    """
    model = CellAssayCrossFactoriser(**ROADMAP)
    n = sum(p.numel() for p in model.parameters())
    assert n == 5_985_025
    assert 5.5e6 < n < 6.5e6, "the paper's '~ 6M'"


def test_decoder_dominates_and_towers_are_symmetric():
    """127 cells x 24 assays: the two towers must come to the same size (24*256+256+127*256 both
    ways), and the decoder must be ~88% of the model -- the shape of Supplementary Table 1's claim
    that eDICE is small because it carries no per-position parameters."""
    m = CellAssayCrossFactoriser(**ROADMAP)
    cell = sum(p.numel() for p in m.cell_embedder.parameters())
    assay = sum(p.numel() for p in m.assay_embedder.parameters())
    dec = sum(p.numel() for p in m.decoder.parameters())
    assert cell == assay == 368_000
    assert dec == 5_249_025


# --- shapes and gradients ----------------------------------------------------------------------

def test_forward_shape_and_loss_backward():
    m = small_model()
    vals, sc, sa, tc, ta = make_inputs(b=8, s=20, t=5)
    out = m(vals, sc, sa, tc, ta)
    assert out.shape == (8, 5)
    assert torch.isfinite(out).all()
    loss = torch.nn.functional.mse_loss(out, torch.randn(8, 5))
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) == len(list(m.parameters()))
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_target_count_is_free():
    """Training draws 120 targets; evaluation asks for however many the split holds. The model must
    not care -- that is the whole point of reading targets out by id."""
    m = small_model().eval()
    vals, sc, sa, _, _ = make_inputs(b=4, s=20, t=1)
    for t in (1, 3, 28):
        tc = torch.randint(0, 7, (4, t))
        ta = torch.randint(0, 4, (4, t))
        assert m(vals, sc, sa, tc, ta).shape == (4, t)


def test_support_count_is_free():
    m = small_model().eval()
    for s in (1, 5, 28):
        vals, sc, sa, tc, ta = make_inputs(b=4, s=s, t=3)
        assert torch.isfinite(m(vals, sc, sa, tc, ta)).all()


# --- the semantics that are easy to get subtly wrong --------------------------------------------

def test_support_order_does_not_change_the_output():
    """The bin's observations are a SET. The reference permutes them every batch, so a model whose
    answer depended on their order would be learning noise."""
    m = small_model().eval()
    vals, sc, sa, tc, ta = make_inputs(b=4, s=20, t=5)
    a = m(vals, sc, sa, tc, ta)
    perm = torch.randperm(20)
    b = m(vals[:, perm], sc[:, perm], sa[:, perm], tc, ta)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_targets_are_independent_of_each_other():
    """Cross-attention queries share keys but not queries, so predicting a target alone must equal
    predicting it inside a batch of targets. If this fails the read-out is leaking."""
    m = small_model().eval()
    vals, sc, sa, tc, ta = make_inputs(b=4, s=20, t=5)
    full = m(vals, sc, sa, tc, ta)
    for j in range(5):
        one = m(vals, sc, sa, tc[:, j:j + 1], ta[:, j:j + 1])
        torch.testing.assert_close(full[:, j:j + 1], one, rtol=1e-5, atol=1e-5)


def test_bins_are_independent_of_each_other():
    """One example is one bin. Nothing may cross the batch axis -- there is no positional structure
    in eDICE at all, which is exactly why it needs no per-position parameters."""
    m = small_model().eval()
    vals, sc, sa, tc, ta = make_inputs(b=6, s=20, t=3)
    full = m(vals, sc, sa, tc, ta)
    solo = m(vals[2:3], sc[2:3], sa[2:3], tc[2:3], ta[2:3])
    torch.testing.assert_close(full[2:3], solo, rtol=1e-5, atol=1e-5)


def test_node_features_are_means_not_sums():
    """`NodeInputMasker` divides each node's row by that node's observation count. Two cells with
    the same assay value must therefore embed identically even when one carries more assays."""
    emb = SignalEmbedder(n_nodes=3, n_feats=2, embed_dim=5, add_global_embedding=False)
    vals = torch.tensor([[1.0, 3.0, 1.0]])
    node = torch.tensor([[0, 0, 1]])
    feat = torch.tensor([[0, 1, 0]])
    out, missing = emb(vals, node, feat)
    # node 0 sees (1, 3) over two assays -> mean row (0.5, 1.5); node 1 sees (1, -) -> (1.0, 0)
    expected = torch.relu(emb.nodewise_hidden(torch.tensor([[[0.5, 1.5], [1.0, 0.0], [0.0, 0.0]]])))
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(missing, torch.tensor([[0.0, 0.0, 1.0]]))


def test_unobserved_nodes_are_masked_out_of_attention():
    """A node with no observations must not enter the softmax normalisation as a KEY.

    It may still appear as a QUERY -- that is how a target cell with nothing observed at this bin
    still gets a prediction, out of its global embedding alone -- so the invariance is tested on
    empty nodes that no target names.
    """
    m = small_model().eval()
    torch.manual_seed(3)
    # cells 0..2 carry supports; cells 3..6 carry none, and no target names them.
    sc = torch.tensor([[0, 0, 1, 1, 2, 2]] * 4)
    sa = torch.tensor([[0, 1, 0, 2, 1, 3]] * 4)
    vals = torch.randn(4, 6)
    tc = torch.tensor([[0, 1, 2]] * 4)
    ta = torch.tensor([[3, 0, 2]] * 4)
    empty = [3, 4, 5, 6]

    before = m(vals, sc, sa, tc, ta)
    weights = m.cell_embedder.cross_block.attn_weights   # (B, H, T, n_cells)
    assert weights.shape[-1] == 7
    assert weights[:, :, :, empty].abs().max() < 1e-6

    with torch.no_grad():
        m.cell_embedder.signal_embedder.global_embeddings[empty] += 100.0
    after = m(vals, sc, sa, tc, ta)
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-5)


def test_all_nodes_missing_stays_finite():
    """A bin where every value is masked would give NaN under a -inf mask. The reference uses -1e9
    for exactly this reason, and the Roadmap h5 does contain all-zero bins."""
    m = small_model().eval()
    vals = torch.zeros(2, 4)
    sc = torch.tensor([[0, 1, 2, 3]] * 2)
    sa = torch.tensor([[0, 1, 2, 3]] * 2)
    tc = torch.tensor([[0]] * 2)
    ta = torch.tensor([[0]] * 2)
    # zeros are still OBSERVED here; force the harder case by hand
    emb = m.cell_embedder.signal_embedder
    embedded, missing = emb(vals, sc, sa)
    all_missing = torch.ones_like(missing)
    out = m.cell_embedder.cross_block(embedded, tc, mask=all_missing)
    assert torch.isfinite(out).all()


def test_dense_uses_keras_initialisation():
    """glorot_uniform kernel, zero bias -- not PyTorch's kaiming default."""
    layer = dense(512, 2048)
    limit = math.sqrt(6.0 / (512 + 2048))
    assert float(layer.weight.abs().max()) <= limit + 1e-6
    assert float(layer.weight.abs().max()) > 0.8 * limit
    assert torch.count_nonzero(layer.bias) == 0


def test_global_embeddings_use_keras_uniform_scale():
    emb = SignalEmbedder(n_nodes=200, n_feats=4, embed_dim=64)
    assert float(emb.global_embeddings.abs().max()) <= 0.05
    assert float(emb.global_embeddings.abs().max()) > 0.04


def test_dropout_is_active_in_train_and_off_in_eval():
    m = small_model(decoder_dropout=0.5, transformer_dropout=0.5)
    vals, sc, sa, tc, ta = make_inputs(b=4, s=20, t=3)
    m.eval()
    torch.testing.assert_close(m(vals, sc, sa, tc, ta), m(vals, sc, sa, tc, ta))
    m.train()
    torch.manual_seed(1)
    a = m(vals, sc, sa, tc, ta)
    torch.manual_seed(2)
    b = m(vals, sc, sa, tc, ta)
    assert not torch.allclose(a, b)


def test_it_can_learn_a_trivial_rule():
    """Loss sanity: a panel where every track equals a fixed per-assay constant is learnable.

    Scored with dropout OFF at both ends, because dropout 0.3 on a 12-wide toy decoder puts a floor
    under the training loss that has nothing to do with whether the model learned the rule.
    """
    torch.manual_seed(0)
    m = small_model()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    vals, sc, sa, tc, ta = make_inputs(b=32, s=20, t=4, seed=5)
    per_assay = torch.tensor([0.0, 1.0, 2.0, 3.0])
    vals = per_assay[sa].float()
    y = per_assay[ta].float()

    def scored():
        m.eval()
        with torch.no_grad():
            return float(torch.nn.functional.mse_loss(m(vals, sc, sa, tc, ta), y))

    first = scored()
    for _ in range(300):
        m.train()
        loss = torch.nn.functional.mse_loss(m(vals, sc, sa, tc, ta), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    last = scored()
    assert last < 0.05 * first, f"{first:.4f} -> {last:.4f}"


# --- the sampler -------------------------------------------------------------------------------

def test_train_sampler_partitions_every_bin():
    vals = np.arange(40, dtype=np.float32).reshape(4, 10)
    cells = np.arange(10) % 3
    assays = np.arange(10) % 4
    s = TrainSampler(vals, cells, assays, n_targets=3, batch_size=4, shuffle=False,
                     rng=np.random.default_rng(0))
    (sup, sc, sa, tc, ta, y), = list(s)
    assert sup.shape == (4, 7) and y.shape == (4, 3)
    for i in range(4):
        seen = np.concatenate([sup[i], y[i]])
        assert sorted(seen.tolist()) == sorted(vals[i].tolist()), "a partition, not a sample"
    # ids travel with their values
    for i in range(4):
        for v, c, a in zip(np.concatenate([sup[i], y[i]]),
                           np.concatenate([sc[i], tc[i]]),
                           np.concatenate([sa[i], ta[i]])):
            col = int(v) % 10
            assert c == cells[col] and a == assays[col]


def test_train_sampler_permutes_bins_independently():
    """Per-BIN permutations, not one per batch -- `TrainInMemGenerator` loops over rows."""
    vals = np.tile(np.arange(12, dtype=np.float32), (64, 1))
    s = TrainSampler(vals, np.arange(12) % 3, np.arange(12) % 4, n_targets=4, batch_size=64,
                     shuffle=False, rng=np.random.default_rng(1))
    (_, _, _, _, _, y), = list(s)
    assert len({tuple(r) for r in y.tolist()}) > 1


def test_train_sampler_rejects_a_full_mask():
    with pytest.raises(ValueError):
        TrainSampler(np.zeros((4, 5), np.float32), np.arange(5), np.arange(5), n_targets=5)


def test_fixed_sampler_covers_every_bin_once():
    sup = np.arange(50, dtype=np.float32).reshape(10, 5)
    s = FixedTargetSampler(sup, np.arange(5), np.arange(5), np.array([0, 1]), np.array([1, 2]),
                           batch_size=4)
    blocks = [b[0] for b in s]
    assert len(blocks) == 3
    np.testing.assert_array_equal(np.concatenate(blocks, axis=0), sup)


# --- metrics -----------------------------------------------------------------------------------

def test_metrics_are_per_track_then_averaged():
    """`models/metrics.py` keeps per-dimension state and averages in `result()`. On a panel where
    one track is much noisier, the mean of per-track MSEs differs from the pooled MSE -- this is
    the difference Supplementary Table 2 is quoted in."""
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(500, 3))
    pred = truth.copy()
    pred[:, 0] += rng.normal(scale=3.0, size=500)
    per = per_track_mse(truth, pred)
    assert per.shape == (3,)
    assert summarise(per)["n_tracks"] == 3
    assert abs(summarise(per)["mean"] - per.mean()) < 1e-12


def test_pearson_is_nan_for_a_constant_track_and_is_dropped():
    truth = np.stack([np.arange(100.0), np.ones(100)], axis=1)
    pred = truth + 0.1
    r = per_track_pearson(truth, pred)
    assert np.isclose(r[0], 1.0)
    assert np.isnan(r[1])
    assert summarise(r)["n_tracks"] == 1
