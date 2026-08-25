"""The Lavawizard port — `competitors/lavawizard/`, `RIVALS_PLAN.md` §7.4.

Everything here runs on synthetic tensors in seconds. The heavy check that cannot live in a unit
test — loading one of *their* real 700 MB checkpoints and matching Keras numerically — is
`competitors/lavawizard/parity_keras.py`, run on Fir.

The synthetic `.h5` in `keras_h5` is not a mock of their file. It is a Keras 2.2.4 archive built to
the layout read off a real one (`cruxvault/results/t53/SPIKE_MEMO.md` §3): same group nesting, same
`layer_names` / `weight_names` attributes, same `<layer>/<weight>:0` naming, same shapes. If their
Synapse files load, it is because this layout is right.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "competitors"))

from lavawizard import dataset3, emit, features, keras_weights             # noqa: E402
from lavawizard.model import Guacamole, Precamole                          # noqa: E402

# A model small enough to build thousands of times, with the real factor widths kept so every
# concatenation width (225 into dense_1, 2050 into dense_3) is the production one.
SMALL = dict(n_celltypes=6, n_assays=4, n_positions=1000)
UNITS = 2048


# ---------------------------------------------------------------------------
# a Keras 2.2.4 archive, built by hand
# ---------------------------------------------------------------------------

def _put(group, layer: str, tensors: dict) -> None:
    g = group.create_group(layer)
    names = []
    for short, arr in tensors.items():
        full = f"{layer}/{short}:0"
        g.create_dataset(full, data=np.asarray(arr, dtype=np.float32))
        names.append(full.encode())
    g.attrs["weight_names"] = names


def _dense(rng, n_in, n_out):
    return {"kernel": rng.normal(0, 0.05, (n_in, n_out)), "bias": rng.normal(0, 0.05, n_out)}


def _bn(rng, n):
    return {"gamma": rng.uniform(0.8, 1.2, n), "beta": rng.normal(0, 0.1, n),
            "moving_mean": rng.normal(0, 0.3, n), "moving_variance": rng.uniform(0.5, 1.5, n)}


@pytest.fixture
def keras_h5(tmp_path):
    """A stage-2 (`Guacamole`) checkpoint in Keras 2.2.4 layout. Returns `(path, tensors)`."""
    rng = np.random.default_rng(7)
    t = {
        "celltype_embedding": {"embeddings": rng.normal(0, 0.1, (SMALL["n_celltypes"], 45))},
        "assay_embedding": {"embeddings": rng.normal(0, 0.1, (SMALL["n_assays"], 65))},
        "genome_25bp_embedding": {"embeddings": rng.normal(0, 0.1, (SMALL["n_positions"], 25))},
        "genome_250bp_embedding": {"embeddings": rng.normal(0, 0.1, (SMALL["n_positions"] // 10 + 1, 30))},
        "genome_5kbp_embedding": {"embeddings": rng.normal(0, 0.1, (SMALL["n_positions"] // 200 + 1, 60))},
        "dense_1": _dense(rng, 225, UNITS), "dense_1_ac": {"alpha": rng.uniform(0, 0.3, UNITS)},
        "dense_1_bn": _bn(rng, UNITS),
        "dense_2": _dense(rng, UNITS, UNITS), "dense_2_ac": {"alpha": rng.uniform(0, 0.3, UNITS)},
        "dense_2_bn": _bn(rng, UNITS),
        "dense_3": _dense(rng, UNITS + 2, UNITS), "dense_3_ac": {"alpha": rng.uniform(0, 0.3, UNITS)},
        "dense_3_bn": _bn(rng, UNITS),
        "y_pred": _dense(rng, UNITS, 1),
    }
    # The weightless layers are present in a real file and must not confuse the reader.
    order = ["celltype_input", "assay_input", "genome_25bp_input", "genome_250bp_input",
             "genome_5kbp_input", "celltype_embedding", "assay_embedding", "genome_25bp_embedding",
             "genome_250bp_embedding", "genome_5kbp_embedding", "flatten_1", "concatenate_1",
             "dense_1", "dense_1_ac", "dense_1_bn", "dense_1_dp", "dense_2", "dense_2_ac",
             "dense_2_bn", "variance_value_input", "average_value_input", "dense_2_dp",
             "concat_last", "dense_3", "dense_3_ac", "dense_3_bn", "dense_3_dp", "y_pred", "add_1"]
    path = tmp_path / "model_v4_guacamole6_chrTest.h5"
    with h5py.File(path, "w") as f:
        f.attrs["keras_version"] = b"2.2.4"
        f.attrs["backend"] = b"tensorflow"
        mw = f.create_group("model_weights")
        mw.attrs["layer_names"] = [n.encode() for n in order]
        for layer in order:
            if layer in t:
                _put(mw, layer, t[layer])
            else:
                mw.create_group(layer).attrs["weight_names"] = []
        f.create_group("optimizer_weights")
    return path, t


# ---------------------------------------------------------------------------
# the loader
# ---------------------------------------------------------------------------

def test_layer_names_round_trip(keras_h5):
    path, _ = keras_h5
    names = keras_weights.keras_layer_names(path)
    assert names[0] == "celltype_input" and names[-1] == "add_1"
    assert "dense_3" in names, "a stage-2 file must carry dense_3; that is how it is told from stage 1"


def test_read_layer_weights_drops_weightless_layers(keras_h5):
    path, tensors = keras_h5
    got = keras_weights.read_layer_weights(path)
    assert set(got) == set(tensors), "weightless layers must be absent, not empty"
    assert got["dense_1"]["kernel"].shape == (225, UNITS)


def test_load_moves_every_tensor(keras_h5):
    path, tensors = keras_h5
    model = Guacamole(**SMALL)
    report = keras_weights.load_keras_h5(model, path)
    assert report["layers"] == 5 + 3 * 3 + 1                 # embeddings + three blocks + y_pred
    assert report["tensors"] == 5 * 1 + 3 * (2 + 1 + 4) + 2  # embeds + (dense, prelu, bn) x3 + head
    assert report["parameters"] == sum(int(np.prod(a.shape))
                                       for layer in tensors.values() for a in layer.values())
    assert not model.training, "a loaded checkpoint must be in eval mode: the BN running stats are the point"


def test_dense_kernel_is_transposed(keras_h5):
    """The conversion that is shape-legal but wrong on the square kernel, checked by value.

    Compared at float32 because that is what the file holds; the fixture keeps numpy's float64
    originals, and casting them here is the honest comparison rather than a loosened tolerance.
    """
    path, tensors = keras_h5
    model = Guacamole(**SMALL)
    keras_weights.load_keras_h5(model, path)

    def f32(a):
        return np.asarray(a, dtype=np.float32)

    np.testing.assert_array_equal(model.block2.dense.weight.detach().numpy(),
                                  f32(tensors["dense_2"]["kernel"]).T)
    np.testing.assert_array_equal(model.block1.bn.running_var.detach().numpy(),
                                  f32(tensors["dense_1_bn"]["moving_variance"]))
    np.testing.assert_array_equal(model.block3.ac.weight.detach().numpy(),
                                  f32(tensors["dense_3_ac"]["alpha"]))


def test_stage1_file_into_stage2_module_is_refused(keras_h5, tmp_path):
    """A `Precamole` checkpoint has no `dense_3`. Loading it as `Guacamole` must name the gap."""
    path, _ = keras_h5
    stripped = tmp_path / "stage1.h5"
    with h5py.File(path, "r") as src, h5py.File(stripped, "w") as dst:
        mw = dst.create_group("model_weights")
        keep = [n for n in keras_weights.keras_layer_names(path) if not n.startswith("dense_3")]
        mw.attrs["layer_names"] = [n.encode() for n in keep]
        for n in keep:
            src["model_weights"].copy(n, mw)
    with pytest.raises(keras_weights.KerasWeightError, match="dense_3"):
        keras_weights.load_keras_h5(Guacamole(**SMALL), stripped)


def test_shape_mismatch_names_the_tensor(keras_h5):
    """A checkpoint for a different chromosome has a different 25 bp table. Refuse, do not resize."""
    path, _ = keras_h5
    wrong = Guacamole(**{**SMALL, "n_positions": SMALL["n_positions"] + 5})
    with pytest.raises(keras_weights.KerasWeightError, match="genome_25bp_embedding"):
        keras_weights.load_keras_h5(wrong, path)


# ---------------------------------------------------------------------------
# the architecture
# ---------------------------------------------------------------------------

def _inputs(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        celltype=torch.randint(0, SMALL["n_celltypes"], (n,), generator=g),
        assay=torch.randint(0, SMALL["n_assays"], (n,), generator=g),
        pos25=torch.randint(0, SMALL["n_positions"], (n,), generator=g),
        average=torch.rand(n, generator=g) * 3,
        variance=torch.rand(n, generator=g),
    )


def test_forward_shape_and_finiteness():
    model = Guacamole(**SMALL).eval()
    out = model(**_inputs())
    assert out.shape == (32,)
    assert torch.isfinite(out).all()


def test_average_is_an_additive_base():
    """`y = y_pred(...) + average`. Zero the head and the model must return the average exactly.

    This is the property the whole method rests on: the cross-cell average is the base prediction
    and the network only learns a correction. Nulling `y_pred` isolates the skip from the average's
    second path through `dense_3`.
    """
    model = Guacamole(**SMALL).eval()
    with torch.no_grad():
        model.y_pred.weight.zero_()
        model.y_pred.bias.zero_()
    x = _inputs()
    torch.testing.assert_close(model(**x), x["average"], rtol=0, atol=0)


def test_batchnorm_uses_keras_epsilon():
    """Keras defaults to 1e-3; torch to 1e-5. Their trained statistics assume the former."""
    model = Guacamole(**SMALL)
    for block in (model.block1, model.block2, model.block3):
        assert block.bn.eps == pytest.approx(1e-3)
        assert block.bn.momentum == pytest.approx(0.01)
        assert block.ac.weight.numel() == UNITS, "PReLU alpha is per-unit, not shared"


def test_block_order_is_dense_prelu_bn_dropout():
    """Their activation comes BEFORE the normalisation. Checked by recomputing the block by hand."""
    model = Guacamole(**SMALL).eval()
    b = model.block1
    x = torch.randn(8, 225)
    manual = b.bn(b.ac(b.dense(x)))
    torch.testing.assert_close(b(x), manual, rtol=0, atol=0)


def test_precamole_adds_the_average_onehot_to_logits():
    model = Precamole(**SMALL).eval()
    x = _inputs(n=16)
    onehot = torch.eye(3)[torch.randint(0, 3, (16,))]
    base = model(x["celltype"], x["assay"], x["pos25"], torch.zeros(16, 3))
    with_avg = model(x["celltype"], x["assay"], x["pos25"], onehot)
    torch.testing.assert_close(with_avg - base, onehot, rtol=0, atol=1e-6)


def test_from_precamole_transfers_the_trunk_and_drops_the_head():
    pre = Precamole(**SMALL)
    gua = Guacamole(**SMALL).from_precamole(pre)
    torch.testing.assert_close(gua.block2.dense.weight, pre.block2.dense.weight, rtol=0, atol=0)
    torch.testing.assert_close(gua.factors.assay_embedding.weight,
                               pre.factors.assay_embedding.weight, rtol=0, atol=0)
    assert not hasattr(gua, "y_pred_dense"), "the stage-1 head must not survive the transfer"


# ---------------------------------------------------------------------------
# the features, and the switch the PI asked for
# ---------------------------------------------------------------------------

POOL = ["T_A", "T_B", "T_C", "T_X", "V_X"]
HAS = {("T_A", "H3K4me3"), ("T_B", "H3K4me3"), ("T_C", "H3K4me3"),
       ("T_X", "H3K4me3"), ("V_X", "H3K4me3"), ("T_A", "ATAC-seq")}


def _carries(bs, assay):
    return (bs, assay) in HAS


def test_cell_type_strips_the_split_prefix():
    assert features.cell_type("T_DND-41") == "DND-41"
    assert features.cell_type("V_X") == features.cell_type("B_X") == "X"
    assert features.cell_type("no-prefix") == "no-prefix"


def test_loo_excludes_by_cell_type_not_by_id():
    """T_X must drop out when the target is V_X — same cell measured twice (§5)."""
    got = features.contributors(POOL, "H3K4me3", "V_X", carries=_carries, mode="loo")
    assert got == ["T_A", "T_B", "T_C"]
    assert "T_X" not in got and "V_X" not in got


def test_upstream_mode_reproduces_their_leak():
    """Their average includes the target itself. Available on demand, never by default."""
    got = features.contributors(POOL, "H3K4me3", "V_X", carries=_carries, mode="upstream")
    assert got == POOL, "upstream keeps everyone, target included — that is the defect being reproduced"
    assert features.contributors(POOL, "H3K4me3", "V_X", carries=_carries) == ["T_A", "T_B", "T_C"], \
        "the DEFAULT must be the leak-free one"


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode must be one of"):
        features.contributors(POOL, "H3K4me3", "V_X", carries=_carries, mode="whatever")


def test_moments_are_arcsinh_space_and_population_variance():
    tracks = {"T_A": np.array([0.0, 1.0, 4.0]), "T_B": np.array([0.0, 3.0, 8.0])}

    def read(bs, assay, start, end):
        return tracks[bs][start:end]

    avg, var, k = features.cross_cell_moments(read, ["T_A", "T_B"], "chr21", 0, 3, assay="H3K4me3")
    assert k == 2
    want = np.arcsinh(np.stack([tracks["T_A"], tracks["T_B"]]))
    np.testing.assert_allclose(avg, want.mean(0), rtol=1e-6)
    np.testing.assert_allclose(var, want.var(0), rtol=1e-5, atol=1e-7)   # ddof=0, as upstream


def test_transform_none_leaves_the_reader_value_alone():
    """The Dataset-3 path: `dataset3` already applied arcsinh per base, so it must not go on twice."""
    tracks = {"T_A": np.array([0.0, 1.0, 4.0])}

    def read(bs, assay, start, end):
        return tracks[bs][start:end]

    avg, _, _ = features.cross_cell_moments(read, ["T_A"], "chr21", 0, 3,
                                            assay="a", transform="none")
    np.testing.assert_allclose(avg, tracks["T_A"], rtol=1e-6)
    twice, _, _ = features.cross_cell_moments(read, ["T_A"], "chr21", 0, 3, assay="a")
    np.testing.assert_allclose(twice, np.arcsinh(tracks["T_A"]), rtol=1e-6)
    assert not np.allclose(avg, twice), "the two orders must be distinguishable, else the flag is a lie"


def test_unknown_transform_is_refused():
    with pytest.raises(ValueError, match="transform must be"):
        features.cross_cell_moments(lambda *a: np.zeros(3), [], "chr21", 0, 3,
                                    assay="a", transform="log")


# ---------------------------------------------------------------------------
# Dataset 3 — the challenge's own grid and binning
# ---------------------------------------------------------------------------

def test_upstream_grid_is_ceil_not_floor():
    """One bin apart on chr21, and their released embedding table proves the ceil is theirs."""
    chr21 = 46_709_983
    assert dataset3.upstream_n_bins(chr21) == 1_868_400
    assert chr21 // 25 == 1_868_399, "our store's floor grid — deliberately different"
    assert dataset3.upstream_n_bins(1000) == 40 and dataset3.upstream_n_bins(1001) == 41


def test_bin_arcsinh_matches_upstream_order_and_padding():
    """arcsinh per BASE then mean, NaN to zero, zero-padded to a whole final bin."""
    length = 60                                   # 2 whole bins + 10 bp -> 3 bins
    v = np.arange(length, dtype=float)
    v[3] = np.nan
    got = dataset3.bin_arcsinh(v, length)
    assert got.shape == (3,)

    base = np.arcsinh(np.nan_to_num(v))
    np.testing.assert_allclose(got[0], base[0:25].mean(), rtol=1e-6)
    np.testing.assert_allclose(got[1], base[25:50].mean(), rtol=1e-6)
    # the tail bin is padded with zeros to 25, not averaged over the 10 real bases
    np.testing.assert_allclose(got[2], np.append(base[50:60], np.zeros(15)).mean(), rtol=1e-6)
    assert got[2] != pytest.approx(base[50:60].mean()), "padding must dilute the partial bin"


def test_bin_arcsinh_applies_the_transform_before_the_mean():
    """arcsinh is concave, so the two orders differ — this pins which one we do."""
    length = 25
    v = np.array([0.0] * 24 + [100.0])
    got = dataset3.bin_arcsinh(v, length)[0]
    assert got == pytest.approx(np.arcsinh(v).mean(), rel=1e-6)
    assert got != pytest.approx(np.arcsinh(v.mean()), rel=1e-3)


def test_bin_arcsinh_refuses_a_wrong_length():
    with pytest.raises(ValueError, match="expected 100 base pairs"):
        dataset3.bin_arcsinh(np.zeros(99), 100)


def test_parse_name_and_track_path():
    assert dataset3.parse_name("C05M17.bigwig") == ("C05", "M17")
    assert dataset3.parse_name("/a/b/C31M29.bigwig") == ("C31", "M29")
    assert dataset3.track_path("/d", "C05", "M17").name == "C05M17.bigwig"
    with pytest.raises(ValueError, match="not a C##M##"):
        dataset3.parse_name("Average.bigwig")


def test_chroms_are_the_23_upstream_trains_on():
    assert len(dataset3.CHROMS) == 23
    assert dataset3.CHROMS[0] == "chr1" and dataset3.CHROMS[-1] == "chrX"
    assert "chrY" not in dataset3.CHROMS


def test_hyperparams_cover_all_23_and_vary():
    assert set(dataset3.UPSTREAM_HYPERPARAMS) == set(dataset3.CHROMS)
    # The point of the table: there is no single shared setting.
    assert dataset3.factor_sizes("chr1")["n_25bp_factors"] == 10
    assert dataset3.factor_sizes("chr21")["n_25bp_factors"] == 25
    assert dataset3.factor_sizes("chr21")["n_5kbp_factors"] == 60
    assert dataset3.factor_sizes("chrX")["n_5kbp_factors"] == 45
    assert dataset3.schedule("chr1")["batch_size"] == 21000
    assert dataset3.schedule("chr21")["train_epochs"] == 800


def test_factor_sizes_build_the_chr21_model_that_matched_their_checkpoint():
    """chr21's row must reproduce the 61,772,797-weight model parity was measured on.

    Keras `count_params()` counts BatchNorm's `moving_mean`/`moving_variance` as non-trainable
    weights; torch calls them buffers and leaves them out of `.parameters()`. The difference is
    exactly `3 blocks x 2 buffers x 2048 = 12,288`, so the comparable total is parameters plus
    buffers — which is also what `load_keras_h5` reports, since it counts the tensors in the file.
    """
    m = Guacamole(n_celltypes=51, n_assays=35,
                  n_positions=dataset3.upstream_n_bins(46_709_983),
                  **dataset3.factor_sizes("chr21"))
    params = sum(p.numel() for p in m.parameters())
    buffers = sum(b.numel() for b in m.buffers() if b.dtype.is_floating_point)
    assert buffers == 3 * 2 * 2048
    assert params + buffers == 61_772_797


def test_unknown_chromosome_fails_by_name():
    for fn in (dataset3.factor_sizes, dataset3.schedule):
        with pytest.raises(KeyError, match="chrY"):
            fn("chrY")


def test_contributor_pool_splits(tmp_path):
    meta = tmp_path / "m.tsv"
    meta.write_text(
        "Cell_ID\tMark_ID\tCellType\tAssay\tTraining(T),Validation(V),Blind-test(B)\tfileName\n"
        "C01\tM16\tadipose\tH3K27ac\tT\tC01M16.bigwig\n"
        "C02\tM16\tadrenal\tH3K27ac\tV\tC02M16.bigwig\n"
        "C03\tM16\tbrain\tH3K27ac\tB\tC03M16.bigwig\n"
        "C04\tM01\tliver\tATAC-seq\tT\tC04M01.bigwig\n", encoding="utf-8")
    rows = dataset3.read_meta(meta)
    assert rows[0]["DataType"] == "T" and "Training(T),Validation(V),Blind-test(B)" not in rows[0]
    assert dataset3.contributor_pool(rows, "M16") == ["C01"]
    assert dataset3.contributor_pool(rows, "M16", splits=("T", "V")) == ["C01", "C02"]
    assert dataset3.contributor_pool(rows, "M99") == []


def test_moments_with_no_contributors_are_zero_not_nan():
    avg, var, k = features.cross_cell_moments(lambda *a: np.zeros(3), [], "chr21", 0, 3, assay="a")
    assert k == 0 and np.all(avg == 0) and np.all(var == 0)


def test_moments_refuse_a_short_contributor():
    def read(bs, assay, start, end):
        return np.zeros(2)

    with pytest.raises(ValueError, match="gave 2 bins"):
        features.cross_cell_moments(read, ["T_A"], "chr21", 0, 3, assay="a")


def test_blocks_tile_exactly():
    assert list(features.blocks(10, 4)) == [(0, 4), (4, 8), (8, 10)]
    assert sum(e - s for s, e in features.blocks(2_500_000, 100_000)) == 2_500_000


# ---------------------------------------------------------------------------
# the §4.1 emitter, checked against the reader that enforces the contract
# ---------------------------------------------------------------------------

def test_inversion_is_clip_then_sinh():
    x = np.array([-3.0, -1e-9, 0.0, 1.0, 2.5], dtype=np.float32)
    got = emit.invert_arcsinh(x)
    np.testing.assert_allclose(got[:3], 0.0, atol=0)
    np.testing.assert_allclose(got[3:], np.sinh([1.0, 2.5]), rtol=1e-6)


def test_write_track_round_trips_through_the_bench_reader(tmp_path):
    """The emitter and `candi.bench.external` must agree on the contract, not merely resemble it."""
    from candi.bench.external import read_track_arrays, track_dirname
    from candi.bench.harness import Pair

    pair, assay, chrom, n = Pair("T_X", "V_X"), "H3K4me3", "chr21", 64
    raw = np.linspace(-1.0, 2.0, n).astype(np.float32)
    emit.write_track(tmp_path, pair, assay, chrom, raw, n_bins=n)

    arrays = read_track_arrays(tmp_path / track_dirname(pair, assay), [chrom], {chrom: n})
    got = arrays[chrom]
    assert set(got) == {"signal_mu"}, "point-only: no invented count head (B1b), no self-fitted sigma"
    np.testing.assert_allclose(got["signal_mu"], emit.invert_arcsinh(raw), rtol=1e-6)


def test_write_track_refuses_a_wrong_length(tmp_path):
    from candi.bench.harness import Pair
    with pytest.raises(ValueError, match="grid wants 64"):
        emit.write_track(tmp_path, Pair("T_X", "V_X"), "H3K4me3", "chr21",
                         np.zeros(63, np.float32), n_bins=64)


def test_manifest_satisfies_the_bench_reader_and_records_the_switch(tmp_path):
    from candi.bench.external import read_manifest

    emit.write_manifest(tmp_path, version="0.1.0", generated_by="tests",
                        contributor_mode="loo", weights="ported-retrain",
                        sparse_assays=["H2AFZ"])
    obj = read_manifest(tmp_path)
    assert obj["method"] == "Lavawizard" and obj["arms"] == ["pval"]
    assert obj["contributor_mode"] == "loo"
    assert obj["signal_inversion"] == emit.ARCSINH_INVERSION
    assert obj["sparse_assays"] == ["H2AFZ"]
    assert json.loads((tmp_path / "manifest.json").read_text())["upstream"].endswith("@d638b204")
