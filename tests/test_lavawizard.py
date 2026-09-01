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
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "competitors"))

from lavawizard import dataset3, emit, features, keras_weights, preprocess  # noqa: E402
from lavawizard.model import Guacamole, Precamole                          # noqa: E402
from lavawizard.train import Sampler                                       # noqa: E402

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


def _total_weights(m):
    """Parameters plus float buffers — the count comparable to Keras `count_params()`.

    Keras counts BatchNorm's `moving_mean`/`moving_variance` as non-trainable weights; torch calls
    them buffers and leaves them out of `.parameters()`. The gap is exactly
    `3 blocks x 2 buffers x 2048 = 12,288`. `load_keras_h5` reports the same total, since it counts
    the tensors in the file.
    """
    return (sum(p.numel() for p in m.parameters())
            + sum(b.numel() for b in m.buffers() if b.dtype.is_floating_point))


def test_factor_sizes_reproduce_the_smoke_artifact_parity_was_measured_on():
    """61,772,797 is the SMOKE checkpoint, whose 35 assays were our choice, not their metadata.

    The parity run in `README.md` used a Keras artifact produced by their code but configured by
    us. It validates the loader — which reads every shape from the file and hardcodes nothing — and
    it does NOT establish the dimensions of their released files. See the next test for those.
    """
    m = Guacamole(n_celltypes=51, n_assays=35,
                  n_positions=dataset3.upstream_n_bins(46_709_983),
                  **dataset3.factor_sizes("chr21"))
    assert sum(b.numel() for b in m.buffers() if b.dtype.is_floating_point) == 3 * 2 * 2048
    assert _total_weights(m) == 61_772_797


def test_released_checkpoints_should_carry_23_assays_not_35(tmp_path):
    """Their own filter drops assays that appear only in training, and that reaches the embedding.

    `02_guacamole6_pretrain.py:49-65` builds the assay list from the metadata AFTER dropping
    training-only assays, so a released chr21 file should have 23 assay rows, not 35 — a 780-weight
    difference (12 assays x 65 factors). Recorded as an expectation to check the moment their
    weights arrive; the loader itself needs no change either way, since it takes the shapes from
    the file.
    """
    meta = _meta(tmp_path)
    _, cells, marks = preprocess.training_tracks(meta)
    assert "M99" not in marks                                  # the T-only assay is gone

    m23 = Guacamole(n_celltypes=51, n_assays=23,
                    n_positions=dataset3.upstream_n_bins(46_709_983),
                    **dataset3.factor_sizes("chr21"))
    assert _total_weights(m23) == 61_772_017
    assert 61_772_797 - 61_772_017 == 12 * 65


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


def test_track_dirname_agrees_with_the_bench_reader():
    """The port implements the §4.1 naming rule itself so it can run without `candi` installed.

    That freedom is only safe if the two implementations cannot drift, so this pins them together
    character for character. If it ever fails, `candi.bench.external` is the authority and
    `emit.track_dirname` is the bug.
    """
    from candi.bench.external import track_dirname as canonical
    from candi.bench.harness import Pair as HarnessPair

    for a, b, assay in [("T_X", "V_X", "H3K4me3"), ("T_DND-41", "B_BE2C", "ATAC-seq"),
                        ("T_a/b", "V_c", "H3K27ac"), ("C05", "C05", "M17")]:
        assert emit.track_dirname(emit.Pair(a, b), assay) == canonical(HarnessPair(a, b), assay)
    # and the real harness Pair must work through the local function too, since anchor may pass one
    assert emit.track_dirname(HarnessPair("T_X", "V_X"), "H3K4me3") == "T_X__V_X__H3K4me3"


def test_emit_needs_no_candi_import():
    """A rival generator has to run wherever the rival's data is — Fir had no `candi` on the path."""
    src = (Path(__file__).resolve().parents[1] / "competitors" / "lavawizard" / "emit.py").read_text()
    assert "from candi" not in src and "import candi" not in src
    anchor_src = (Path(__file__).resolve().parents[1] / "competitors" / "lavawizard"
                  / "anchor.py").read_text()
    assert "from candi" not in anchor_src and "import candi" not in anchor_src


def test_write_track_round_trips_through_the_bench_reader(tmp_path):
    """The emitter and `candi.bench.external` must agree on the contract, not merely resemble it."""
    from candi.bench.external import read_track_arrays, track_dirname

    pair, assay, chrom, n = emit.Pair("T_X", "V_X"), "H3K4me3", "chr21", 64
    raw = np.linspace(-1.0, 2.0, n).astype(np.float32)
    emit.write_track(tmp_path, pair, assay, chrom, raw, n_bins=n)

    arrays = read_track_arrays(tmp_path / track_dirname(pair, assay), [chrom], {chrom: n})
    got = arrays[chrom]
    assert set(got) == {"signal_mu"}, "point-only: no invented count head (B1b), no self-fitted sigma"
    np.testing.assert_allclose(got["signal_mu"], emit.invert_arcsinh(raw), rtol=1e-6)


def test_write_track_refuses_a_wrong_length(tmp_path):
    with pytest.raises(ValueError, match="grid wants 64"):
        emit.write_track(tmp_path, emit.Pair("T_X", "V_X"), "H3K4me3", "chr21",
                         np.zeros(63, np.float32), n_bins=64)


def test_manifest_satisfies_the_bench_reader_and_records_the_switch(tmp_path):
    from candi.bench.external import read_manifest

    emit.write_manifest(tmp_path, version="0.1.0", generated_by="tests",
                        contributor_mode="loo", weights="ported-retrain", clip=True,
                        sparse_assays=["H2AFZ"])
    obj = read_manifest(tmp_path)
    assert obj["method"] == "Lavawizard" and obj["arms"] == ["pval"]
    assert obj["contributor_mode"] == "loo"
    assert obj["signal_inversion"] == emit.ARCSINH_INVERSION
    assert obj["sparse_assays"] == ["H2AFZ"]
    assert json.loads((tmp_path / "manifest.json").read_text())["upstream"].endswith("@d638b204")


# ---------------------------------------------------------------------------
# the cap on the output (PI ruling, 2026-08-26)
# ---------------------------------------------------------------------------

def test_clip_caps_the_prediction_in_minus_log10_p_space(tmp_path):
    """The cap is applied AFTER the inversion, so it bounds `signal_mu` exactly as written."""
    from candi.bench.external import read_track_arrays, track_dirname

    pair, assay, chrom, n = emit.Pair("T_X", "V_X"), "H3K4me3", "chr21", 8
    raw = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 15.5, 3.0, 0.5], dtype=np.float32)
    emit.write_track(tmp_path, pair, assay, chrom, raw, n_bins=n, clip_max=100.0)
    mu = read_track_arrays(tmp_path / track_dirname(pair, assay), [chrom], {chrom: n})[chrom]
    assert mu["signal_mu"].max() == pytest.approx(100.0)
    uncapped = emit.invert_arcsinh(raw)
    below = uncapped <= 100.0
    np.testing.assert_allclose(mu["signal_mu"][below], uncapped[below], rtol=1e-6)


def test_clip_off_is_the_faithful_port(tmp_path):
    """Default `clip_max=None` leaves the anchor's arithmetic untouched, blowup and all."""
    raw = np.array([15.5], dtype=np.float32)
    p = emit.write_track(tmp_path, emit.Pair("T_X", "V_X"), "H3K4me3", "chr21", raw, n_bins=1)
    assert float(np.load(p)["signal_mu"][0]) > 2.0e6, "sinh(15.5) is the chr17 defect, unguarded"


def test_a_zero_or_nonfinite_cap_is_refused(tmp_path):
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="finite and positive"):
            emit.write_track(tmp_path, emit.Pair("T_X", "V_X"), "H3K4me3", "chr21",
                             np.ones(4, np.float32), n_bins=4, clip_max=bad)


def test_manifest_states_the_cap_either_way(tmp_path):
    """`clip` is required, not defaulted: absent and false must not look alike to a reader."""
    import inspect

    par = inspect.signature(emit.write_manifest).parameters["clip"]
    assert par.default is inspect.Parameter.empty, "a defaulted clip flag can be forgotten silently"
    for flag in (True, False):
        d = tmp_path / str(flag)
        emit.write_manifest(d, version="0.1.0", generated_by="tests",
                            contributor_mode="loo", weights="ported-retrain", clip=flag)
        obj = json.loads((d / "manifest.json").read_text())
        assert obj["clip"] is flag
        assert obj["clip_rule"] == emit.CLIP_RULE == "training_max_per_mark_per_chrom"


# ---------------------------------------------------------------------------
# preprocessing and the sampler
# ---------------------------------------------------------------------------

META_ROWS = [
    # cell, mark, celltype, assay, split
    ("C01", "M16", "adipose", "H3K27ac", "T"), ("C02", "M16", "adrenal", "H3K27ac", "V"),
    ("C03", "M16", "brain", "H3K27ac", "B"), ("C01", "M17", "adipose", "H3K4me3", "T"),
    ("C02", "M17", "adrenal", "H3K4me3", "T"), ("C03", "M17", "brain", "H3K4me3", "B"),
    ("C01", "M99", "adipose", "OnlyInTraining", "T"),   # dropped: T-only assay
    ("C02", "M99", "adrenal", "OnlyInTraining", "T"),
]


def _meta(tmp_path):
    p = tmp_path / "meta.tsv"
    head = "Cell_ID\tMark_ID\tCellType\tAssay\tTraining(T),Validation(V),Blind-test(B)\tfileName\n"
    body = "".join(f"{c}\t{m}\t{ct}\t{a}\t{s}\t{c}{m}.bigwig\n" for c, m, ct, a, s in META_ROWS)
    p.write_text(head + body, encoding="utf-8")
    return dataset3.read_meta(p)


def test_training_tracks_drops_training_only_assays(tmp_path):
    tracks, cells, marks = preprocess.training_tracks(_meta(tmp_path))
    assert "M99" not in marks, "an assay that is never a target must not get an embedding"
    assert all(m != "M99" for _, m in tracks)


def test_training_tracks_keep_T_and_V_but_not_B(tmp_path):
    tracks, _, _ = preprocess.training_tracks(_meta(tmp_path))
    assert ("C01", "M16") in tracks and ("C02", "M16") in tracks     # T and V both train
    assert ("C03", "M16") not in tracks                              # B never trains


def test_blind_cells_still_get_an_embedding_row(tmp_path):
    """The tensor-factorisation mechanism: C03's row trains through the assays C03 does have."""
    tracks, cells, _ = preprocess.training_tracks(_meta(tmp_path))
    assert "C03" in cells, "a blind cell must be in the embedding table even though it never trains"
    assert all(c != "C03" for c, _ in tracks)


def test_terciles_are_equal_count_and_break_ties_by_position():
    x = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 9.0])
    t = preprocess._terciles(x)
    assert t.dtype == np.int8
    assert sorted(np.bincount(t, minlength=3).tolist()) == [2, 2, 2], "three equal-count groups"
    assert t[5] == 2 and t[0] == 2, "the two largest values land in the top tercile"
    # The four tied zeros are split across terciles by position, not lumped together — upstream's
    # rank(method='first') does exactly this, and it is arbitrary by design.
    assert len(set(t[1:5].tolist())) > 1


def _fake_cache(tmp_path, n_tracks=6, n_bins=40, n_marks=2, seed=3):
    """A CachedChrom on disk without touching a bigwig."""
    rng = np.random.default_rng(seed)
    root = tmp_path / "cache"
    d = root / "chrT"
    d.mkdir(parents=True)
    values = rng.uniform(0, 3, (n_tracks, n_bins)).astype(np.float32)
    np.save(d / "tracks.npy", values)
    np.save(d / "tercile.npy", np.stack([preprocess._terciles(v) for v in values]))
    marks = [f"M{j:02d}" for j in range(n_marks)]
    cells = [f"C{i:02d}" for i in range(n_tracks)]
    tracks = [(cells[i], marks[i % n_marks]) for i in range(n_tracks)]
    mark_of = np.array([i % n_marks for i in range(n_tracks)])
    sums = np.stack([values[mark_of == j].sum(0) for j in range(n_marks)]).astype(np.float32)
    sumsq = np.stack([(values[mark_of == j] ** 2).sum(0) for j in range(n_marks)]).astype(np.float32)
    np.save(d / "sums.npy", sums)
    np.save(d / "sumsq.npy", sumsq)
    (d / "index.json").write_text(json.dumps({
        "chrom": "chrT", "n_bins": n_bins, "chrom_length": n_bins * 25, "grid": "upstream_ceil",
        "tracks": [list(t) for t in tracks], "cells": cells, "marks": marks,
        "mark_counts": {m: int((mark_of == j).sum()) for j, m in enumerate(marks)},
    }), encoding="utf-8")
    return root, values, mark_of


def test_moments_upstream_matches_a_direct_computation(tmp_path):
    root, values, mark_of = _fake_cache(tmp_path)
    c = preprocess.CachedChrom(root, "chrT")
    tix = np.array([0, 1, 4]); pos = np.array([3, 7, 11])
    x = c.values[tix, pos].astype(np.float32)
    avg, var = c.moments(tix, pos, x, "upstream")
    for i, (t, p) in enumerate(zip(tix, pos)):
        col = values[mark_of == mark_of[t], p]
        assert avg[i] == pytest.approx(col.mean(), rel=1e-5)
        assert var[i] == pytest.approx(col.var(), rel=1e-4, abs=1e-6)   # ddof=0


def test_moments_loo_excludes_the_target_itself(tmp_path):
    root, values, mark_of = _fake_cache(tmp_path)
    c = preprocess.CachedChrom(root, "chrT")
    tix = np.array([0, 3]); pos = np.array([5, 9])
    x = c.values[tix, pos].astype(np.float32)
    avg, var = c.moments(tix, pos, x, "loo")
    for i, (t, p) in enumerate(zip(tix, pos)):
        keep = (mark_of == mark_of[t]) & (np.arange(len(values)) != t)
        col = values[keep, p]
        assert avg[i] == pytest.approx(col.mean(), rel=1e-4)
        assert var[i] == pytest.approx(col.var(), rel=1e-3, abs=1e-6)
    up, _ = c.moments(tix, pos, x, "upstream")
    assert not np.allclose(avg, up), "the two modes must differ, else the switch is decorative"


def test_moments_reject_an_unknown_mode(tmp_path):
    root, _, _ = _fake_cache(tmp_path)
    c = preprocess.CachedChrom(root, "chrT")
    with pytest.raises(ValueError, match="upstream.*loo"):
        c.moments(np.array([0]), np.array([0]), np.zeros(1, np.float32), "sideways")


def test_sampler_walks_positions_contiguously_and_covers_an_epoch(tmp_path):
    """An epoch is ceil(n_bins/bs) steps because the position cursor sweeps the chromosome once."""
    root, _, _ = _fake_cache(tmp_path, n_bins=40)
    c = preprocess.CachedChrom(root, "chrT")
    s = Sampler(c, batch_size=8, mode="upstream", seed=1)
    assert s.steps_per_epoch == 5
    seen = []
    for _ in range(s.steps_per_epoch):
        _, pos, _, _, _ = s._batch()
        seen.append(pos)
    assert np.array_equal(seen[0], np.arange(0, 8))
    assert np.array_equal(seen[1], np.arange(8, 16))
    assert sorted(np.concatenate(seen).tolist()) == list(range(40)), "one epoch covers every bin"


def test_sampler_stage_batches_have_the_shapes_the_models_want(tmp_path):
    root, _, _ = _fake_cache(tmp_path)
    c = preprocess.CachedChrom(root, "chrT")
    s = Sampler(c, batch_size=16, mode="upstream", seed=2)
    dev = torch.device("cpu")

    d1, y1 = s.stage1(dev)
    assert set(d1) == {"celltype", "assay", "pos25", "average_onehot"}
    assert d1["average_onehot"].shape == (16, 3)
    assert y1.shape == (16,) and y1.min() >= 0 and y1.max() <= 2
    small = dict(n_celltypes=len(c.cells), n_assays=len(c.marks), n_positions=c.n_bins)
    assert Precamole(**small)(**d1).shape == (16, 3)

    d2, y2 = s.stage2(dev)
    assert set(d2) == {"celltype", "assay", "pos25", "average", "variance"}
    assert y2.shape == (16,)
    assert Guacamole(**small)(**d2).shape == (16,)


def test_stage2_target_is_the_track_value_itself(tmp_path):
    """Regression target is the observed arcsinh signal — not the average, not the tercile."""
    root, values, _ = _fake_cache(tmp_path)
    c = preprocess.CachedChrom(root, "chrT")
    s = Sampler(c, batch_size=6, mode="upstream", seed=5)
    tix, pos, x, _, _ = s._batch()
    np.testing.assert_allclose(x, values[tix, pos], rtol=1e-6)


# ---------------------------------------------------------------------------
# the anchor comparison
# ---------------------------------------------------------------------------

def test_bin_values_none_is_the_same_grid_without_the_transform():
    length = 50
    v = np.arange(length, dtype=float)
    raw = dataset3.bin_values(v, length, transform="none")
    arc = dataset3.bin_values(v, length, transform="arcsinh")
    assert raw.shape == arc.shape == (2,)
    np.testing.assert_allclose(raw[0], v[:25].mean(), rtol=1e-6)
    np.testing.assert_allclose(arc[0], np.arcsinh(v[:25]).mean(), rtol=1e-6)
    assert not np.allclose(raw, arc)
    with pytest.raises(ValueError, match="transform must be"):
        dataset3.bin_values(v, length, transform="log")


def test_compare_computes_the_three_pairings():
    from lavawizard import anchor
    rng = np.random.default_rng(11)
    truth = rng.uniform(0, 5, 500)
    ours = truth + rng.normal(0, 0.5, 500)
    theirs = truth + rng.normal(0, 1.0, 500)
    r = anchor.compare(ours, theirs, truth)
    assert r["n_bins"] == 500
    np.testing.assert_allclose(r["ours_vs_truth_mse"], np.mean((ours - truth) ** 2), rtol=1e-9)
    np.testing.assert_allclose(r["theirs_vs_truth_mse"], np.mean((theirs - truth) ** 2), rtol=1e-9)
    # the deliberately-better predictor must come out better, and the ratio must say so
    assert r["ours_vs_truth_mse"] < r["theirs_vs_truth_mse"]
    assert r["mse_ratio_ours_over_theirs"] < 1.0
    assert r["ours_vs_truth_pearson"] > r["theirs_vs_truth_pearson"]


def test_compare_truncates_to_the_shortest_input():
    from lavawizard import anchor
    r = anchor.compare(np.zeros(10), np.zeros(8), np.zeros(12))
    assert r["n_bins"] == 8


def test_compare_survives_a_constant_track():
    """A flat prediction has no correlation to report; it must be NaN, not an exception."""
    from lavawizard import anchor
    r = anchor.compare(np.ones(50), np.arange(50.0), np.arange(50.0))
    assert np.isnan(r["ours_vs_truth_pearson"])
    assert np.isfinite(r["ours_vs_truth_mse"])


def test_summarise_carries_the_caveat_where_it_cannot_be_missed():
    from lavawizard import anchor
    rows = [{"cell": "C05", "mark": "M17", "ours_vs_truth_mse": 1.0, "theirs_vs_truth_mse": 2.0,
             "ours_vs_truth_pearson": 0.8, "theirs_vs_truth_pearson": 0.7,
             "ours_vs_theirs_pearson": 0.9},
            {"cell": "C06", "mark": "M17", "ours_vs_truth_mse": 3.0, "theirs_vs_truth_mse": 4.0,
             "ours_vs_truth_pearson": 0.6, "theirs_vs_truth_pearson": 0.5,
             "ours_vs_theirs_pearson": 0.7}]
    s = anchor.summarise(rows)
    assert s["n_tracks"] == 2
    assert s["macro_ours_vs_truth_mse"] == pytest.approx(2.0)
    assert s["macro_theirs_vs_truth_mse"] == pytest.approx(3.0)
    assert "001 vendored EIC scorer" in s["caveat"]
    assert "Dataset-2" in s["caveat"], "the no-cross-dataset rule must travel with the number"


def test_predict_track_uses_pooled_moments_and_names_an_unknown_cell(tmp_path):
    """A blind cell contributes nothing to its mark's sum, so pooled IS leave-one-out."""
    from lavawizard import anchor
    root, values, mark_of = _fake_cache(tmp_path, n_tracks=6, n_bins=40, n_marks=2)
    c = preprocess.CachedChrom(root, "chrT")
    obj = {"cells": c.cells + ["C99"], "marks": c.marks}
    model = Guacamole(n_celltypes=len(obj["cells"]), n_assays=len(c.marks),
                      n_positions=c.n_bins).eval()
    # Null the head so the output is exactly the average that was fed in.
    with torch.no_grad():
        model.y_pred.weight.zero_(); model.y_pred.bias.zero_()
    got = anchor.predict_track(model, c, obj, "C99", c.marks[0], batch=16)
    pooled = values[mark_of == 0].mean(0)
    np.testing.assert_allclose(got, pooled, rtol=1e-5, atol=1e-6)

    with pytest.raises(ValueError, match="no embedding row"):
        anchor.predict_track(model, c, obj, "C_nope", c.marks[0])
    with pytest.raises(ValueError, match="no embedding row"):
        anchor.predict_track(model, c, obj, "C99", "M_nope")


def _fake_pybigwig(monkeypatch, opener):
    """Inject a stand-in `pyBigWig` module. `read_binned` imports it inside the function, so this
    works without pyBigWig installed — these tests stay dependency-free like the rest of the file."""
    import types
    mod = types.ModuleType("pyBigWig")
    mod.open = opener
    monkeypatch.setitem(sys.modules, "pyBigWig", mod)


class _FakeBW:
    def __init__(self, chroms): self._c = chroms
    def chroms(self): return self._c
    def values(self, c, a, b): return list(np.arange(b - a, dtype=float))
    def close(self): pass


def test_read_binned_retries_a_transport_error_then_succeeds(monkeypatch):
    """A sick Lustre client must not end a multi-hour anchor run (observed on fc30560)."""
    calls = {"n": 0}

    def opener(path):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError(108, "Cannot send after transport endpoint shutdown")
        return _FakeBW({"chr21": 50})

    _fake_pybigwig(monkeypatch, opener)
    got = dataset3.read_binned("/whatever.bigwig", "chr21", transform="none", backoff=0.0)
    assert calls["n"] == 3, "it must actually have retried, not succeeded first time"
    assert got.shape == (2,)


def test_read_binned_does_not_retry_a_real_content_error(monkeypatch):
    """A missing chromosome is not transient; retrying it only burns the backoff."""
    calls = {"n": 0}

    def opener(path):
        calls["n"] += 1
        return _FakeBW({"chr1": 10})

    _fake_pybigwig(monkeypatch, opener)
    with pytest.raises(ValueError, match="has no chr21"):
        dataset3.read_binned("/whatever.bigwig", "chr21", backoff=0.0)
    assert calls["n"] == 1, "a content error must fail on the first attempt"


def test_read_binned_gives_up_with_a_message_that_names_the_suspect(monkeypatch):
    def opener(path):
        raise OSError(108, "transport endpoint")

    _fake_pybigwig(monkeypatch, opener)
    with pytest.raises(OSError, match="Lustre mount is the suspect"):
        dataset3.read_binned("/whatever.bigwig", "chr21", attempts=2, backoff=0.0)


# ---------------------------------------------------------------------------
# the anchor verdict roll-up
# ---------------------------------------------------------------------------

def _anchor_file(tmp_path, chrom, rows):
    from lavawizard import anchor
    obj = anchor.summarise(rows)
    obj["chrom"] = chrom
    p = tmp_path / f"anchor_{chrom}.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _row(cell, mark, ours, theirs, n_bins=1000):
    return {"cell": cell, "mark": mark, "n_bins": n_bins,
            "ours_vs_truth_mse": ours, "theirs_vs_truth_mse": theirs,
            "ours_vs_truth_pearson": 0.7, "theirs_vs_truth_pearson": 0.72,
            "ours_vs_theirs_pearson": 0.9}


def test_verdict_bands_are_monotone_and_cover_everything():
    from lavawizard import anchor_report as R
    edges = [e for e, _ in R.VERDICT_BANDS]
    assert edges == sorted(edges) and edges[-1] == float("inf")
    assert R.verdict_for(1.00).startswith("APPROACHES")
    assert R.verdict_for(1.09).startswith("APPROACHES")
    assert R.verdict_for(1.20).startswith("NEAR")
    assert R.verdict_for(1.50).startswith("SHORT")
    assert R.verdict_for(9.90).startswith("FAILS")
    assert R.verdict_for(float("nan")).startswith("UNDEFINED")


def test_ratio_below_one_means_we_beat_their_submission():
    from lavawizard import anchor_report as R
    assert R.verdict_for(0.85).startswith("APPROACHES"), "better than theirs is not a failure"


def test_report_rolls_up_macro_pooled_and_per_mark(tmp_path):
    from lavawizard import anchor_report as R
    f1 = _anchor_file(tmp_path, "chr21", [_row("C05", "M17", 1.0, 1.0), _row("C05", "M18", 3.0, 2.0)])
    f2 = _anchor_file(tmp_path, "chr22", [_row("C05", "M17", 2.0, 1.0), _row("C05", "M18", 2.0, 2.0)])
    rec = R.build_report([f1, f2])
    assert rec["n_chromosomes"] == 2 and rec["n_track_chromosomes"] == 4
    assert rec["macro_ours_mse"] == pytest.approx(2.0)
    assert rec["macro_theirs_mse"] == pytest.approx(1.5)
    assert rec["macro_ratio_ours_over_theirs"] == pytest.approx(2.0 / 1.5)
    # M17 ratios are 1.0 and 2.0 -> median 1.5; M18 are 1.5 and 1.0 -> median 1.25
    assert rec["by_mark_median_ratio"]["M17"] == pytest.approx(1.5)
    assert rec["by_mark_median_ratio"]["M18"] == pytest.approx(1.25)
    # ratios are 1.0, 1.5, 2.0, 1.0 -> two are STRICTLY worse. Parity is not "worse".
    assert rec["worse_than_theirs_fraction"] == pytest.approx(0.5)
    assert "001 vendored EIC scorer" in rec["caveat"]


def test_pooled_weights_by_bin_count_and_macro_does_not(tmp_path):
    """A huge chromosome must move pooled but not macro — that is why both are reported."""
    from lavawizard import anchor_report as R
    big = _anchor_file(tmp_path, "chr1", [_row("C05", "M17", 10.0, 1.0, n_bins=1_000_000)])
    small = _anchor_file(tmp_path, "chr21", [_row("C05", "M17", 1.0, 1.0, n_bins=1_000)])
    rec = R.build_report([big, small])
    assert rec["macro_ratio_ours_over_theirs"] == pytest.approx((10.0 + 1.0) / 2 / 1.0)
    assert rec["pooled_ratio_ours_over_theirs"] > 9.9, "pooled must follow the big chromosome"
    assert rec["pooled_ratio_ours_over_theirs"] != pytest.approx(
        rec["macro_ratio_ours_over_theirs"])


def test_verdict_uses_the_median_track_ratio_not_the_ratio_of_means(tmp_path):
    """One huge-MSE track must not decide the verdict for a thousand others.

    Track MSE spans orders of magnitude on this data (C06M16 ~100 vs C05M18 ~0.4), so a ratio of
    means is set by dynamic range rather than by method quality.
    """
    from lavawizard import anchor_report as R
    rows = [_row("C05", "M18", 1.0, 1.0) for _ in range(20)]          # 20 tracks at parity
    rows.append(_row("C06", "M16", 200.0, 100.0))                      # one loud track at 2.0
    f = _anchor_file(tmp_path, "chr16", rows)
    rec = R.build_report([f])
    assert rec["median_track_ratio"] == pytest.approx(1.0)
    assert rec["macro_ratio_ours_over_theirs"] > 1.8, "the mean is captured by the loud track"
    assert rec["verdict"].startswith("APPROACHES"), "the verdict must follow the median"
    assert rec["headline_disagreement"] != "none"
    assert "High-MSE tracks are pulling the mean" in rec["headline_disagreement"]


def test_no_disagreement_flag_when_both_aggregations_agree(tmp_path):
    from lavawizard import anchor_report as R
    f = _anchor_file(tmp_path, "chr21", [_row("C05", "M17", 1.0, 1.0), _row("C05", "M18", 2.0, 2.0)])
    rec = R.build_report([f])
    assert rec["headline_disagreement"] == "none"
    assert rec["verdict"].startswith("APPROACHES")


# ---------------------------------------------------------------------------
# the our-EIC side: store_eic
# ---------------------------------------------------------------------------

def _regime(tmp_path, *, train=("T_A", "T_B", "T_C"), pairs=(("T_A", "V_A"),),
            assays=("H3K4me3", "H3K27ac"), train_chroms=("chr19",), regions=None):
    p = tmp_path / "regime.json"
    obj = {
        "store": str(tmp_path / "store"), "assays": list(assays),
        "biosamples": {"train": list(train), "eval": [b for _, b in pairs]},
        "eval_pairs": [list(x) for x in pairs],
        "train_chroms": list(train_chroms), "eval_chroms": ["chr21"],
    }
    if regions is not None:
        obj["regions"] = regions
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _bed(tmp_path, intervals, name="regions.bed"):
    """A BED4 plus the `{bed, sha256, policy}` block D32 requires, pinned on its own bytes."""
    import hashlib

    p = tmp_path / name
    text = "".join(f"{c}\t{s}\t{e}\tR{i}\n" for i, (c, s, e) in enumerate(intervals))
    p.write_text(text, encoding="utf-8")
    return {"bed": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "policy": "contain"}


class _FakeBiosample:
    def __init__(self, panel, values):
        self._panel, self._values = list(panel), values

    def assays(self, kind=None):
        return list(self._panel)

    def pval(self, chrom, start, end, assays=None):
        names = list(assays) if assays else self._panel
        return np.stack([self._values[n][start:end] for n in names], axis=1)


class _FakeCorpus:
    """Just enough `CorpusStore` for the cache builder and the predictor."""

    def __init__(self, bios, n_bins):
        self._bios, self._n = bios, n_bins

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getitem__(self, name):
        return self._bios[name]

    def n_bins(self, chrom=None):
        return self._n


def _install_fake_corpus(monkeypatch, bios, n_bins):
    import candi.store.reader as R
    monkeypatch.setattr(R, "CorpusStore", lambda *a, **k: _FakeCorpus(bios, n_bins))


def test_cell_index_gives_a_pair_one_embedding_row(tmp_path):
    from lavawizard import store_eic

    names, ix = store_eic.cell_index(store_eic.load_regime(_regime(tmp_path)))
    assert names == ["T_A", "T_B", "T_C"], "cells are named in T_ terms and sorted"
    assert ix["V_A"] == ix["T_A"], "a V_ target with no embedding row cannot be imputed at all"
    assert "V_A" not in names


def test_cell_index_refuses_a_pair_whose_input_is_not_a_training_cell(tmp_path):
    from lavawizard import store_eic

    r = store_eic.load_regime(_regime(tmp_path, pairs=(("T_Z", "V_Z"),)))
    with pytest.raises(store_eic.FairnessError, match="no fitted embedding"):
        store_eic.cell_index(r)


def test_train_columns_raises_rather_than_filtering_a_leaked_target(tmp_path):
    """§6.2 must be auditable: a rule that silently drops a column cannot be checked afterwards."""
    from lavawizard import store_eic

    r = store_eic.load_regime(_regime(tmp_path, train=("T_A", "V_A"), pairs=(("T_A", "V_A"),)))
    bios = {b: _FakeBiosample(["H3K4me3"], {"H3K4me3": np.zeros(4, np.float32)})
            for b in ("T_A", "V_A")}
    with pytest.raises(store_eic.FairnessError, match="TARGETS"):
        store_eic.train_columns(r, _FakeCorpus(bios, 4))


def test_train_columns_take_only_declared_train_biosamples_and_assays(tmp_path):
    from lavawizard import store_eic

    r = store_eic.load_regime(_regime(tmp_path))
    v = {n: np.zeros(4, np.float32) for n in ("H3K4me3", "H3K27ac", "DNase-seq")}
    bios = {"T_A": _FakeBiosample(["H3K4me3", "DNase-seq"], v),   # DNase-seq is not declared
            "T_B": _FakeBiosample(["H3K4me3", "H3K27ac"], v),
            "T_C": _FakeBiosample(["H3K27ac"], v)}
    cols = store_eic.train_columns(r, _FakeCorpus(bios, 4))
    assert cols == [("T_A", "H3K4me3"), ("T_B", "H3K4me3"), ("T_B", "H3K27ac"),
                    ("T_C", "H3K27ac")]
    assert all(a in r["assays"] for _, a in cols)


def _store_cache(tmp_path, monkeypatch, n_bins=64):
    """Build a real cache through `store_eic` over a fake corpus. Returns (regime path, values)."""
    from lavawizard import store_eic

    rng = np.random.default_rng(0)
    vals = {b: {a: rng.random(n_bins).astype(np.float32) * 5.0
                for a in ("H3K4me3", "H3K27ac")} for b in ("T_A", "T_B", "T_C", "V_A")}
    bios = {b: _FakeBiosample(["H3K4me3", "H3K27ac"], vals[b]) for b in vals}
    _install_fake_corpus(monkeypatch, bios, n_bins)
    rp = _regime(tmp_path)
    store_eic.build_cache_from_store(rp, "chr21", tmp_path / "cache", verbose=False)
    return rp, vals


def test_store_cache_is_arcsinh_of_the_stores_own_bins(tmp_path, monkeypatch):
    """No binning: the store's grid is already 25 bp, so the only transform is `arcsinh`."""
    _, vals = _store_cache(tmp_path, monkeypatch)
    c = preprocess.CachedChrom(tmp_path / "cache", "chr21")
    i = c.tracks.index(("T_B", "H3K27ac"))
    np.testing.assert_allclose(c.values[i], np.arcsinh(vals["T_B"]["H3K27ac"]), rtol=1e-6)
    assert c.cells == ["T_A", "T_B", "T_C"] and c.marks == ["H3K4me3", "H3K27ac"]
    assert json.loads((c.dir / "index.json").read_text())["grid"] == "store_floor"


def test_cache_records_the_per_mark_training_max(tmp_path, monkeypatch):
    _, vals = _store_cache(tmp_path, monkeypatch)
    c = preprocess.CachedChrom(tmp_path / "cache", "chr21")
    for j, mark in enumerate(c.marks):
        want = max(np.arcsinh(vals[b][mark]).max() for b in vals)
        assert c.mark_max[j] == pytest.approx(want, rel=1e-6)


def test_a_cache_without_mark_max_reports_none_rather_than_a_guess(tmp_path, monkeypatch):
    _store_cache(tmp_path, monkeypatch)
    d = preprocess.cache_dir(tmp_path / "cache", "chr21")
    obj = json.loads((d / "index.json").read_text())
    obj.pop("mark_max")
    (d / "index.json").write_text(json.dumps(obj), encoding="utf-8")
    assert preprocess.CachedChrom(tmp_path / "cache", "chr21").mark_max is None


def test_prediction_excludes_the_pairs_input_cell_from_the_average(tmp_path, monkeypatch):
    """§6.2 at predict time: T_A must not be in the average used to impute V_A.

    Checked on the model's own input rather than on its output, because the additive skip makes
    the average the dominant term — a leak here would look like a good score, not like a bug.
    """
    import torch
    from lavawizard import store_eic

    rp, vals = _store_cache(tmp_path, monkeypatch)
    cache = preprocess.CachedChrom(tmp_path / "cache", "chr21")

    seen = {}

    class _Spy:
        def __call__(self, *, celltype, assay, pos25, average, variance):
            seen.setdefault((int(celltype[0]), int(assay[0])), []).append(average.numpy().copy())
            return torch.zeros_like(average)

    meta = {"cells": cache.cells, "marks": cache.marks, "stage": "genome"}
    monkeypatch.setattr(store_eic, "load_checkpoint", lambda *a, **k: (_Spy(), meta),
                        raising=False)
    import lavawizard.anchor as A
    monkeypatch.setattr(A, "load_checkpoint", lambda *a, **k: (_Spy(), meta))

    store_eic.predict_chrom(rp, "chr21", tmp_path / "cache", tmp_path / "ck.pt",
                            tmp_path / "pred", clip=False, verbose=False)

    got = np.concatenate(seen[(0, 0)])                     # cell T_A, mark H3K4me3
    excl = np.mean([np.arcsinh(vals[b]["H3K4me3"]) for b in ("T_B", "T_C")], axis=0)
    incl = np.mean([np.arcsinh(vals[b]["H3K4me3"]) for b in ("T_A", "T_B", "T_C")], axis=0)
    np.testing.assert_allclose(got, excl, rtol=1e-5)
    assert not np.allclose(got, incl), "the pooled average would be the leak §6.2 forbids"


def test_predict_refuses_a_cap_from_a_cache_that_never_measured_one(tmp_path, monkeypatch):
    from lavawizard import store_eic

    rp, _ = _store_cache(tmp_path, monkeypatch)
    d = preprocess.cache_dir(tmp_path / "cache", "chr21")
    obj = json.loads((d / "index.json").read_text())
    obj.pop("mark_max")
    (d / "index.json").write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(ValueError, match="Rebuild"):
        store_eic.predict_chrom(rp, "chr21", tmp_path / "cache", tmp_path / "ck.pt",
                                tmp_path / "pred", clip=True, verbose=False)


def test_only_the_our_store_module_may_import_candi():
    """`dataset3`/`emit`/`anchor` run on Fir without `candi`; `store_eic` cannot and must not try."""
    d = Path(__file__).resolve().parents[1] / "competitors" / "lavawizard"
    for name in ("dataset3.py", "emit.py", "anchor.py", "model.py", "preprocess.py", "train.py"):
        src = (d / name).read_text()
        assert "import candi" not in src and "from candi" not in src, name
    assert "from candi.store.reader import CorpusStore" in (d / "store_eic.py").read_text()


def test_loo_sampler_skips_a_mark_with_no_leave_one_out_pool(tmp_path, monkeypatch):
    """Our EIC carries five single-track marks; Dataset 3 carries none. §5 says skip and list."""
    from lavawizard.train import Sampler

    _store_cache(tmp_path, monkeypatch)
    cache = preprocess.CachedChrom(tmp_path / "cache", "chr21")
    # make H3K27ac a single-contributor mark by hand, the way our store presents one
    j = cache.marks.index("H3K27ac")
    cache.mark_count[j] = 1

    s = Sampler(cache, 32, "loo", seed=0)
    assert s.skipped_marks == ["H3K27ac"]
    drawn = {cache.tracks[i][1] for i in np.unique(s._batch()[0])}
    assert drawn == {"H3K4me3"}, "a skipped mark must never reach a batch"

    keeps_everything = Sampler(cache, 32, "upstream", seed=0)
    assert keeps_everything.skipped_marks == [], "upstream keeps the target in its own average"
    assert keeps_everything.eligible.size == cache.n_tracks


def test_a_cache_of_only_thin_marks_refuses_rather_than_looping(tmp_path, monkeypatch):
    from lavawizard.train import Sampler

    _store_cache(tmp_path, monkeypatch)
    cache = preprocess.CachedChrom(tmp_path / "cache", "chr21")
    cache.mark_count[:] = 1
    with pytest.raises(ValueError, match="nothing to train on"):
        Sampler(cache, 32, "loo", seed=0)


# ---------------------------------------------------------------------------
# BENCHMARK_DESIGN.md §5 — the V_ selection panel
# ---------------------------------------------------------------------------

REGIMES = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("name", ["regime.eic_19.json", "regime.eic_pilot.json"])
def test_the_derived_selection_panel_holds_no_B_target_at_all(tmp_path, name):
    """§5, PI ruling 2026-08-31: B_ is not merely kept out of selection, it is never READ.

    Both live regimes declare all 38 pairs inside them, so a selection pass that took the regime
    verbatim would open B_ at every check and spend the one touch the design allows.
    """
    from lavawizard import store_eic

    src = REGIMES / name
    if not src.exists():                       # pragma: no cover - the shipped configs are tracked
        pytest.skip(f"{name} is not in this checkout")
    dst = store_eic.write_v_only_regime(src, tmp_path / "vsel.json")
    out = json.loads(dst.read_text())
    targets = [t for _, t in out["eval_pairs"]]
    assert len(targets) == 26, "the V_ panel is 26 T_->V_ cell pairs (§5)"
    assert [t for t in targets if t.startswith("B_")] == []
    assert all(t.startswith("V_") for t in targets)
    assert sorted(out["biosamples"]["eval"]) == sorted(set(targets)), \
        "leaving the B_ cells in biosamples.eval would be a false claim about the split"
    # Everything that is not the panel is carried through unchanged — a derived regime that also
    # moved the corpus, the assay order or the eval scope would select on a different exam.
    keep = json.loads(src.read_text())
    for k in ("store", "assays", "eval_chroms", "train_chroms", "context_bins", "seed"):
        if k in keep:
            assert out[k] == keep[k], k


def test_the_derived_selection_regime_points_at_the_bed_it_was_pinned_on(tmp_path):
    """A derived copy lands elsewhere, and `regions.bed` resolves against the regime's OWN dir."""
    from lavawizard import store_eic

    src = REGIMES / "regime.eic_pilot.json"
    if not src.exists():                       # pragma: no cover
        pytest.skip("the pilot regime is not in this checkout")
    dst = store_eic.write_v_only_regime(src, tmp_path / "deep" / "vsel.json")
    bed = json.loads(dst.read_text())["regions"]["bed"]
    assert Path(bed).is_absolute() and Path(bed).is_file()
    # The hash gate is the point of the rewrite: it must still pass from the new location.
    from candi.store.regime import Regime
    r = Regime.from_file(dst)
    assert r.regions is not None, "the derived copy must still carry the D32 scope"
    assert all(t.startswith("V_") for _, t in r.eval_pairs)


def test_a_regime_with_no_V_pair_is_refused_rather_than_selecting_on_B(tmp_path):
    from lavawizard import store_eic

    r = store_eic.load_regime(_regime(tmp_path, pairs=(("T_A", "B_A"),)))
    with pytest.raises(store_eic.ScopeError, match="no V_ eval pair"):
        store_eic.derive_v_only(r)


def test_the_selection_metric_must_be_a_number_the_scorer_actually_produced():
    """A metric the run cannot compute selects nothing, so it raises instead of scoring +inf."""
    from lavawizard import store_eic

    result = {"macro": {"pval": {"mse": 0.25, "gwcorr": 0.8, "n_tracks": 3}, "count": {}}}
    assert store_eic._macro_value(result, "pval:mse") == 0.25
    with pytest.raises(ValueError, match="no `crps` on the count arm"):
        store_eic._macro_value(result, "count:crps")
    with pytest.raises(ValueError, match="<arm>:<key>"):
        store_eic._macro_value(result, "mse")


# ---------------------------------------------------------------------------
# the selection loop inside training
# ---------------------------------------------------------------------------

def _shrink(monkeypatch, TR, *, pretrain, train, batch=8):
    """Upstream's per-chromosome schedule, shrunk to something a laptop runs in seconds.

    Both `schedule` and `factor_sizes` are keyed on the 23 real chromosome names and raise
    `KeyError` for anything else (`dataset3.py`) — which is the behaviour `BENCHMARK_DESIGN.md` §2
    cites as the reason Rule 2 exempts per-position adaptation, so it is patched here rather than
    weakened there.
    """
    monkeypatch.setattr(TR.dataset3, "schedule",
                        lambda c: {"batch_size": batch, "pretrain_epochs": pretrain,
                                   "train_epochs": train})
    monkeypatch.setattr(TR.dataset3, "factor_sizes",
                        lambda c: {"n_25bp_factors": 25, "n_250bp_factors": 30,
                                   "n_5kbp_factors": 60})


def _tiny_train(tmp_path, monkeypatch, values, **kw):
    """`train_chromosome` on the fake cache, with upstream's schedule shrunk to something local.

    The chromosome is `chr21` so `dataset3` has hyperparameters for it; `epoch_scale` and the batch
    size are what make it seconds rather than hours.
    """
    from lavawizard import train as TR

    root, _, _ = _fake_cache(tmp_path, n_tracks=6, n_bins=40, n_marks=2)
    _shrink(monkeypatch, TR, pretrain=1, train=8)
    calls = {"n": 0}

    def select_fn(model, epoch, step):
        v = values[min(calls["n"], len(values) - 1)]
        calls["n"] += 1
        return float(v)

    rec = TR.train_chromosome(root, "chrT", tmp_path / "out", contributor_mode="upstream",
                              device="cpu", seed=0, select_fn=select_fn, select_metric="pval:mse",
                              **kw)
    return rec, calls


def test_the_best_weights_are_written_the_moment_the_metric_improves(tmp_path, monkeypatch):
    """A run killed by walltime must still leave a SELECTED checkpoint, not just the last one."""
    from lavawizard import train as TR

    root, _, _ = _fake_cache(tmp_path, n_tracks=6, n_bins=40, n_marks=2)
    _shrink(monkeypatch, TR, pretrain=1, train=4)
    out = tmp_path / "out"
    best = out / "guacamole_chrT.best.pt"
    seen = []

    def select_fn(model, epoch, step):
        # Read the file's existence BEFORE returning, so the assertion is about ordering: the
        # improving check at epoch 0 must have put a file on disk before the next check runs.
        seen.append(best.exists())
        return [0.9, 0.4, 0.7, 0.8][len(seen) - 1]

    TR.train_chromosome(root, "chrT", out, contributor_mode="upstream", device="cpu",
                        select_fn=select_fn, select_every=1, select_metric="pval:mse")
    assert seen == [False, True, True, True], "the first improvement writes before the next check"
    assert best.exists()
    rec = json.loads((out / "train_chrT.json").read_text())
    sel = rec["selection"]
    assert sel["best"]["epoch"] == 1 and sel["best"]["value"] == pytest.approx(0.4)
    assert [c["improved"] for c in sel["curve"]] == [True, True, False, False]
    assert sel["metric"] == "pval:mse"


def test_the_selected_checkpoint_is_not_the_last_one_when_the_metric_worsened(tmp_path, monkeypatch):
    from lavawizard.anchor import load_checkpoint

    rec, _ = _tiny_train(tmp_path, monkeypatch, [0.5, 0.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
                         select_every=1)
    out = tmp_path / "out"
    best, last = out / "guacamole_chrT.best.pt", out / "guacamole_chrT.pt"
    assert best.exists() and last.exists()
    a, _ = load_checkpoint(best)
    b, _ = load_checkpoint(last)
    same = all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert not same, "the run kept training after the best epoch, so the two files must differ"
    assert rec["selection"]["best"]["epoch"] == 1


def test_a_run_with_no_selection_writes_no_best_checkpoint_and_says_so(tmp_path, monkeypatch):
    """`select_every=0` is the anchor path: it trains exactly as it did before selection existed."""
    from lavawizard import train as TR

    root, _, _ = _fake_cache(tmp_path, n_tracks=6, n_bins=40, n_marks=2)
    _shrink(monkeypatch, TR, pretrain=1, train=2)
    rec = TR.train_chromosome(root, "chrT", tmp_path / "out", contributor_mode="upstream",
                              device="cpu")
    assert not (tmp_path / "out" / "guacamole_chrT.best.pt").exists()
    assert rec["selection"]["best"] is None and rec["selection"]["every_epochs"] == 0


def test_selection_stops_a_run_that_has_not_improved_for_the_patience_in_epochs(tmp_path, monkeypatch):
    """Patience is counted in EPOCHS, the unit `candi.train --early-stop-epochs` counts it in."""
    rec, calls = _tiny_train(tmp_path, monkeypatch, [0.5] + [0.9] * 10,
                             select_every=1, early_stop_epochs=2)
    assert rec["stage2"]["early_stopped"] is True
    assert rec["stage2"]["steps"] < rec["stage2"]["planned_steps"]
    # best at epoch 0; the stop fires at the first epoch MORE than 2 after it, which is epoch 3.
    assert rec["selection"]["best"]["epoch"] == 0
    assert calls["n"] == 4


def test_a_non_finite_check_never_selects(tmp_path, monkeypatch):
    """NaN is not an improvement, and a run that only ever saw NaN selected nothing."""
    rec, _ = _tiny_train(tmp_path, monkeypatch, [float("nan")] * 8, select_every=1)
    assert rec["selection"]["best"] is None
    assert not (tmp_path / "out" / "guacamole_chrT.best.pt").exists()


def test_the_selection_pass_is_kept_out_of_the_step_rate(tmp_path, monkeypatch):
    """A cadence that inflated ms/step would make two runs at different cadences incomparable."""
    def slow(model, epoch, step):
        time.sleep(0.5)
        return 0.5 - epoch * 0.01

    from lavawizard import train as TR

    root, _, _ = _fake_cache(tmp_path, n_tracks=6, n_bins=40, n_marks=2)
    _shrink(monkeypatch, TR, pretrain=1, train=3)
    rec = TR.train_chromosome(root, "chrT", tmp_path / "out", contributor_mode="upstream",
                              device="cpu", select_fn=slow, select_every=1,
                              select_metric="pval:mse")
    s2 = rec["stage2"]
    assert s2["eval_seconds"] >= 1.5, "three checks at 0.5 s each were timed as selection"
    # 1.5 s of selection over 15 steps is 100 ms/step of it. The step rate must be well under that,
    # and the stage's own `seconds` must not carry it either.
    assert s2["ms_per_step"] < 100 and s2["seconds"] < 1.5


def test_the_selector_scores_the_V_panel_with_the_bench_scorer(tmp_path, monkeypatch):
    """The wiring: a live model in, one number out, and the panel it scored carries no B_.

    `score_external` itself is stubbed — it needs the real store and the annotation assets — but
    everything around it is the production path: the derived regime, the declared track list, the
    §4.1 root, and the macro key that comes back.
    """
    from lavawizard import store_eic

    rng = np.random.default_rng(0)
    n_bins = 64
    names = ("T_A", "T_B", "T_C", "V_A", "B_A")
    vals = {b: {a: rng.random(n_bins).astype(np.float32) * 5.0
                for a in ("H3K4me3", "H3K27ac")} for b in names}
    bios = {b: _FakeBiosample(["H3K4me3", "H3K27ac"], vals[b]) for b in names}
    _install_fake_corpus(monkeypatch, bios, n_bins)
    rp = _regime(tmp_path, pairs=(("T_A", "V_A"), ("T_B", "B_A")))
    store_eic.build_cache_from_store(rp, "chr21", tmp_path / "cache", verbose=False)
    cache = preprocess.CachedChrom(tmp_path / "cache", "chr21", mmap=True)

    opened = {}
    import candi.bench.external as X
    import candi.bench.harness as H
    monkeypatch.setattr(H, "open_source", lambda **kw: opened.update(kw) or "SOURCE")
    monkeypatch.setattr(X, "score_external",
                        lambda source, root, **kw: {"macro": {"pval": {"mse": 0.125}}})

    fn, info = store_eic.selector(rp, "chr21", cache, tmp_path / "work", clip=False,
                                  verbose=False)
    assert info["pairs"] == 1 and info["b_pairs"] == 0, "B_A was declared and must be dropped"
    assert opened["chroms"] == ("chr21",), "a per-chromosome model is only scored on its own chrom"

    model = Guacamole(n_celltypes=len(cache.cells), n_assays=len(cache.marks),
                      n_positions=cache.n_bins)
    assert fn(model, 0, 0) == pytest.approx(0.125)
    root = Path(info["pred_root"])
    assert (root / "manifest.json").exists()
    written = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert written == ["T_A__V_A__H3K27ac", "T_A__V_A__H3K4me3"], \
        "only the V_ pair's tracks are written, and B_A's truth is never opened"


# ---------------------------------------------------------------------------
# D32 — the BED-restricted training scope
# ---------------------------------------------------------------------------

def test_the_pilot_regions_contain_the_training_bins_the_design_pins():
    """§3.1 pins 1,023,489 contained 25 bp bins for `eic.pilot`'s training scope.

    That is a CONTAINMENT count and not `bp // 25` — 25,588,197 bp is not divisible by 25. It is
    also not §3.1's other number: 1,294 windows / 993,792 bins is the 768-bin WINDOW count CANDI's
    sampler plans, and Guacamole samples single bins.
    """
    from lavawizard import store_eic

    src = REGIMES / "regime.eic_pilot.json"
    if not src.exists():                       # pragma: no cover
        pytest.skip("the pilot regime is not in this checkout")
    r = store_eic.load_regime(src)
    total = sum(store_eic.contained_bins(r, c, base=src.parent).size for c in r["train_chroms"])
    assert total == 1_023_489
    assert store_eic.contained_bins(
        store_eic.load_regime(REGIMES / "regime.eic_19.json"), "chr19", base=src.parent) is None, \
        "a regime with no `regions` key restricts nothing"


def test_a_training_locus_counts_only_if_it_lies_wholly_inside_a_region(tmp_path):
    """Containment, on the chromosome bin grid, anchored at bin 0 — never re-anchored per region."""
    from lavawizard import store_eic

    # 30..80 bp is bins 2 (50..75) alone: bin 1 starts at 25 and ends at 50, inside; bin 0 is not.
    reg = _bed(tmp_path, [("chr21", 30, 80), ("chr21", 200, 260)])
    r = store_eic.load_regime(_regime(tmp_path, regions=reg))
    got = store_eic.contained_bins(r, "chr21", base=tmp_path)
    # 30..80: the first whole bin starts at ceil(30/25)=2, the last ends at 80//25=3 -> bin 2.
    # 200..260: ceil(200/25)=8 to 260//25=10 -> bins 8, 9.
    np.testing.assert_array_equal(got, [2, 8, 9])
    assert store_eic.contained_bins(r, "chr22", base=tmp_path).size == 0, \
        "a chromosome the BED never names has no training scope"


def _regions_cache(tmp_path, monkeypatch, reg, *, n_bins=64, chrom="chr21"):
    from lavawizard import store_eic

    rng = np.random.default_rng(1)
    vals = {b: {a: rng.random(n_bins).astype(np.float32) * 5.0
                for a in ("H3K4me3", "H3K27ac")} for b in ("T_A", "T_B", "T_C", "V_A")}
    bios = {b: _FakeBiosample(["H3K4me3", "H3K27ac"], vals[b]) for b in vals}
    _install_fake_corpus(monkeypatch, bios, n_bins)
    rp = _regime(tmp_path, regions=reg, train_chroms=(chrom,))
    store_eic.build_cache_from_store(rp, chrom, tmp_path / "cache", verbose=False)
    return rp


def test_under_a_regions_regime_every_training_locus_lies_inside_the_bed(tmp_path, monkeypatch):
    """The requirement: with a `regions` BED declared, training samples nothing outside it."""
    from lavawizard.train import Sampler

    reg = _bed(tmp_path, [("chr21", 250, 500), ("chr21", 1000, 1200)])
    _regions_cache(tmp_path, monkeypatch, reg)
    cache = preprocess.CachedChrom(tmp_path / "cache", "chr21")
    inside = set(range(10, 20)) | set(range(40, 48))       # 250..500 -> 10..19, 1000..1200 -> 40..47
    assert set(cache.train_bins.tolist()) == inside

    s = Sampler(cache, batch_size=4, mode="upstream", seed=0)
    drawn = set()
    for _ in range(4 * s.steps_per_epoch):
        drawn.update(s._batch()[1].tolist())
    assert drawn <= inside, "a bin outside the BED reached a training batch"
    assert drawn == inside, "and every bin inside it is reachable"


def test_the_bed_scope_sets_the_epoch_length_rather_than_repeating_the_chromosome(tmp_path, monkeypatch):
    """An epoch is the training scope once. A 40x smaller scope must buy 40x fewer steps."""
    from lavawizard.train import Sampler

    reg = _bed(tmp_path, [("chr21", 250, 500)])            # 10 of the chromosome's 64 bins
    _regions_cache(tmp_path, monkeypatch, reg)
    restricted = preprocess.CachedChrom(tmp_path / "cache", "chr21")
    assert restricted.train_bins.size == 10
    assert Sampler(restricted, 4, "upstream", seed=0).steps_per_epoch == 3      # ceil(10/4)
    assert restricted.train_scope["n_bins"] == 10 and restricted.train_scope["policy"] == "contain"


def test_a_cache_with_no_bed_still_trains_on_the_whole_chromosome(tmp_path, monkeypatch):
    """Absent means every bin — every regime without `regions`, and every Dataset-3 cache."""
    from lavawizard.train import Sampler

    _store_cache(tmp_path, monkeypatch)
    cache = preprocess.CachedChrom(tmp_path / "cache", "chr21")
    assert cache.train_bins is None and cache.train_scope is None
    s = Sampler(cache, 8, "upstream", seed=0)
    assert s.steps_per_epoch == 8 and s.positions.size == cache.n_bins


def test_a_bed_that_no_bin_of_the_chromosome_falls_inside_is_refused(tmp_path, monkeypatch):
    from lavawizard import store_eic

    reg = _bed(tmp_path, [("chr21", 3, 20)])               # inside one bin, containing none
    with pytest.raises(store_eic.ScopeError, match="no training scope"):
        _regions_cache(tmp_path, monkeypatch, reg)


def test_an_eval_chromosome_caches_whole_even_under_a_regions_regime(tmp_path, monkeypatch):
    """The BED restricts where TRANSFERABLE parameters are fit, and nothing else.

    Under `eic.pilot` the BED's chr21 regions are exactly the ones `train_chroms` CUT, so the old
    per-chromosome fit had to be refused there. The genome stage fits position tables alone, which
    §2 Rule 2 counts as inference and allows anywhere, so an eval chromosome now caches whole and
    carries no training restriction at all — the restriction lives on the shared cache instead.
    """
    from lavawizard import store_eic

    reg = _bed(tmp_path, [("chr21", 250, 500)])
    rng = np.random.default_rng(1)
    vals = {b: {a: rng.random(64).astype(np.float32) * 5.0 for a in ("H3K4me3", "H3K27ac")}
            for b in ("T_A", "T_B", "T_C", "V_A")}
    _install_fake_corpus(monkeypatch, {b: _FakeBiosample(["H3K4me3", "H3K27ac"], vals[b])
                                       for b in vals}, 64)
    rp = _regime(tmp_path, regions=reg, train_chroms=("chr19",))
    store_eic.build_cache_from_store(rp, "chr21", tmp_path / "cache", verbose=False)
    cache = preprocess.CachedChrom(tmp_path / "cache", "chr21")
    assert cache.train_bins is None and cache.train_scope is None
    assert cache.n_bins == 64, "prediction needs every bin, whatever the training scope is"
    np.testing.assert_allclose(cache.values[cache.tracks.index(("T_B", "H3K27ac"))],
                               np.arcsinh(vals["T_B"]["H3K27ac"]), rtol=1e-6)


# ---------------------------------------------------------------------------
# §2 Rule 2 — the transferable stage
# ---------------------------------------------------------------------------

def test_the_transferable_stage_fits_no_parameter_on_an_eval_chromosome(tmp_path, monkeypatch):
    """The requirement, stated as a diff: the genome stage must move the position tables and
    nothing else.

    §2 Rule 2 binds the cell factors, the assay factors and the dense network — every parameter that
    is not indexed by position. Those arrive from the shared fit and must come out of an eval
    chromosome's stage bit for bit unchanged, or the chromosome the method is scored on has entered
    a parameter the regime claims was fit elsewhere.
    """
    from lavawizard import train as TR
    from lavawizard.anchor import load_checkpoint

    root, _, _ = _fake_cache(tmp_path, n_tracks=6, n_bins=40, n_marks=2)
    _shrink(monkeypatch, TR, pretrain=1, train=3)
    shared = tmp_path / "shared"
    TR.train_chromosome(root, "chrT", shared, contributor_mode="upstream", device="cpu",
                        stage="shared")
    before, meta = load_checkpoint(shared / "guacamole_chrT.pt")
    assert meta["stage"] == "shared"

    TR.train_chromosome(root, "chrT", tmp_path / "gen", contributor_mode="upstream", device="cpu",
                        stage="genome", init=shared / "guacamole_chrT.pt")
    after, gmeta = load_checkpoint(tmp_path / "gen" / "guacamole_chrT.pt")
    assert gmeta["stage"] == "genome"

    kept = before.transferable_state()
    for k, v in after.transferable_state().items():
        assert torch.equal(v, kept[k]), f"{k} took a gradient on the chromosome it is scored on"
    moved = [k for k in after.state_dict()
             if k.startswith("factors.genome_") and
             not torch.equal(after.state_dict()[k], before.state_dict()[k])]
    assert sorted(moved) == ["factors.genome_250bp_embedding.weight",
                             "factors.genome_25bp_embedding.weight",
                             "factors.genome_5kbp_embedding.weight"], \
        "the position tables are the only thing this stage is allowed to fit, and it must fit them"


def test_a_run_that_never_saw_a_shared_fit_is_refused_rather_than_started(tmp_path, monkeypatch):
    """Falling back to a fresh init would fit the cell and assay factors on the eval chromosome."""
    from lavawizard import store_eic, train as TR

    with pytest.raises(ValueError, match="needs an --init"):
        TR.train_chromosome(tmp_path / "nope", "chr21", tmp_path / "out", stage="genome")
    with pytest.raises(store_eic.ScopeError, match="Rule 2"):
        store_eic.train_chrom(_regime(tmp_path), "chr21", tmp_path / "cache", tmp_path / "out")


def test_a_shared_checkpoint_is_never_predicted_from(tmp_path, monkeypatch):
    """Its position tables address a packed axis, so every array it wrote would be plausible junk."""
    from lavawizard import store_eic, train as TR

    root, _, _ = _fake_cache(tmp_path, n_tracks=6, n_bins=40, n_marks=2)
    _shrink(monkeypatch, TR, pretrain=1, train=1)
    TR.train_chromosome(root, "chrT", tmp_path / "ck", contributor_mode="upstream", device="cpu",
                        stage="shared")
    rp, _ = _store_cache(tmp_path, monkeypatch)
    with pytest.raises(store_eic.ScopeError, match="only a `genome` checkpoint"):
        store_eic.predict_chrom(rp, "chr21", tmp_path / "cache",
                                tmp_path / "ck" / "guacamole_chrT.pt", tmp_path / "pred",
                                clip=False, verbose=False)


def test_the_two_live_regimes_fit_the_transferable_half_on_different_loci(tmp_path):
    """THE POINT OF THE WHOLE STAGE. Before it, every key this method read was identical between
    `eic.19` and `eic.pilot`, so the two rows were one run under two labels.

    `shared_layout` is now the only reader of `train_chroms` and `regions`, and it answers
    differently for the two — different chromosomes, different bin counts, no overlap. And neither
    answer names an eval chromosome, which is what Rule 2 asks of it.
    """
    from lavawizard import store_eic

    if not (REGIMES / "regime.eic_pilot.json").exists():   # pragma: no cover
        pytest.skip("the live regimes are not in this checkout")
    scopes = {}
    for name in ("regime.eic_19.json", "regime.eic_pilot.json"):
        src = REGIMES / name
        r = store_eic.load_regime(src)
        spans, n_slots = store_eic.shared_layout(
            r, lambda c: 2_344_704, base=src.parent)       # chr19's bin count, for the no-BED case
        scopes[name] = (spans, n_slots, {c for c, _, _, _ in spans},
                        sum(b - a for _, a, b, _ in spans))
        assert not (scopes[name][2] & set(r["eval_chroms"])), \
            f"{name}: the transferable stage would touch an eval chromosome"

    a, b = scopes["regime.eic_19.json"], scopes["regime.eic_pilot.json"]
    assert a[2] == {"chr19"} and a[3] == 2_344_704, "eic.19's transferable scope is chr19 whole"
    assert len(b[2]) == 18, "eic.pilot's is the Pilot Regions over eighteen train chromosomes"
    assert b[3] == 1_023_489, "and it is §3.1's own contained-bin count, not a fraction of it"
    assert a[3] != b[3], "two regimes that trained on the same loci would be one run twice"


def test_the_two_regimes_produce_different_shared_factors(tmp_path, monkeypatch):
    """The axis is real only if the same code on the two scopes yields two different fits.

    Same seed, same schedule, same index space, same tracks — only the training loci differ. If the
    cell and assay factors came out equal, the regime label would still be decorative.
    """
    from lavawizard import train as TR

    root, values, _ = _fake_cache(tmp_path, n_tracks=6, n_bins=80, n_marks=2)
    _shrink(monkeypatch, TR, pretrain=1, train=4)
    fits = {}
    for label, bins in (("left", np.arange(0, 40)), ("right", np.arange(40, 80))):
        np.save(root / "chrT" / "train_bins.npy", bins.astype(np.int64))
        rec = TR.train_chromosome(root, "chrT", tmp_path / label, contributor_mode="upstream",
                                  device="cpu", seed=0, stage="shared")
        assert rec["n_train_bins"] == 40
        from lavawizard.anchor import load_checkpoint
        fits[label] = load_checkpoint(tmp_path / label / "guacamole_chrT.pt")[0]

    for name in ("factors.celltype_embedding.weight", "factors.assay_embedding.weight"):
        x, y = fits["left"].state_dict()[name], fits["right"].state_dict()[name]
        assert not torch.equal(x, y), f"{name} did not move when the training loci moved"


def test_the_packed_axis_cuts_the_coarse_factors_where_the_chromosome_would(tmp_path):
    """A slot's 250 bp and 5 kbp factors must be the ones its absolute chromosome bin would have.

    Packing regions end to end is what lets the shared fit hold eighteen chromosomes at all, and it
    is only sound if it does not re-anchor the grid: a region starting at chromosome bin 4,321 must
    land on a slot congruent to 4,321 modulo the coarsest stride, or its coarse factors are cut at
    coordinates that exist nowhere in the genome. Two regions must also never share one.
    """
    from lavawizard import store_eic

    reg = _bed(tmp_path, [("chr19", 25 * 4321, 25 * 4400), ("chr19", 25 * 9000, 25 * 9100),
                          ("chr20", 25 * 77, 25 * 500)])
    r = store_eic.load_regime(_regime(tmp_path, regions=reg, train_chroms=("chr19", "chr20")))
    spans, n_slots = store_eic.shared_layout(r, lambda c: 10_000, base=tmp_path)

    stride = store_eic.COARSE_STRIDE
    for _, first, _, slot0 in spans:
        assert slot0 % stride == first % stride, "the coarse grid was re-anchored on this region"
    coarse = [set(range(s0 // stride, (s0 + (b - a) - 1) // stride + 1))
              for _, a, b, s0 in spans]
    for i in range(len(coarse)):
        for j in range(i + 1, len(coarse)):
            assert not (coarse[i] & coarse[j]), "two regions share a 5 kbp factor"
    assert n_slots >= sum(b - a for _, a, b, _ in spans)


def test_under_the_pilot_regime_the_shared_fit_lies_inside_the_bed(tmp_path, monkeypatch):
    """Every slot the transferable stage may sample maps back to a bin inside a BED region, on a
    chromosome the regime declares it trains on. That is what makes `eic.pilot` legitimate."""
    from lavawizard import store_eic
    from lavawizard.train import Sampler

    reg = _bed(tmp_path, [("chr19", 250, 500), ("chr20", 1000, 1200)])
    rng = np.random.default_rng(2)
    vals = {b: {a: rng.random(64).astype(np.float32) * 5.0 for a in ("H3K4me3", "H3K27ac")}
            for b in ("T_A", "T_B", "T_C", "V_A")}
    _install_fake_corpus(monkeypatch, {b: _FakeBiosample(["H3K4me3", "H3K27ac"], vals[b])
                                       for b in vals}, 64)
    rp = _regime(tmp_path, regions=reg, train_chroms=("chr19", "chr20"))
    store_eic.build_shared_cache_from_store(rp, tmp_path / "cache", verbose=False)
    cache = preprocess.CachedChrom(tmp_path / "cache", store_eic.SHARED_STEM)

    r = store_eic.load_regime(rp)
    spans, _ = store_eic.shared_layout(r, lambda c: 64, base=tmp_path)
    allowed = {c: set(store_eic.contained_bins(r, c, base=tmp_path).tolist())
               for c in r["train_chroms"]}
    assert allowed["chr19"] == {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
    assert allowed["chr20"] == set(range(40, 48))

    slot_to = {}
    for c, a, b, s0 in spans:
        slot_to.update({s0 + k: (c, a + k) for k in range(b - a)})
    s = Sampler(cache, batch_size=4, mode="upstream", seed=0)
    drawn = set()
    for _ in range(6 * s.steps_per_epoch):
        drawn.update(s._batch()[1].tolist())
    assert drawn == set(slot_to), "the fit must reach every contained bin and no other slot"
    for slot in drawn:
        c, bin_ = slot_to[slot]
        assert bin_ in allowed[c], f"slot {slot} is {c}:{bin_}, outside the BED"
        assert c not in r["eval_chroms"]


def test_the_transferable_stage_does_not_select_in_either_regime(tmp_path):
    """Selection attaches to the genome stage alone — see `store_eic.train_shared`.

    Under `eic.pilot` the shared scope is not a chromosome and there is no panel to score; under
    `eic.19` it is one and there would be. Selecting in one regime and not the other would put a
    difference into the ablation that is not the regime, so it selects in neither.
    """
    from lavawizard import store_eic

    with pytest.raises(SystemExit):
        store_eic.main(["train", "--regime", str(_regime(tmp_path)), "--stage", "shared",
                        "--cache", str(tmp_path / "c"), "--out", str(tmp_path / "o"),
                        "--select-every", "50"])
    with pytest.raises(SystemExit):
        store_eic.main(["train", "--regime", str(_regime(tmp_path)), "--stage", "shared",
                        "--chrom", "chr19", "--cache", str(tmp_path / "c"),
                        "--out", str(tmp_path / "o"), "--select-every", "0"])


def test_a_shared_scope_the_hyperparameter_table_cannot_name_is_refused(tmp_path):
    """The packed stem borrows the eval chromosomes' UPSTREAM_HYPERPARAMS row, because the row sets
    the factor widths and those set `dense_1`'s input width — a transferable tensor. Chromosomes
    that disagree have no row to borrow, and guessing one would break the transfer silently."""
    from lavawizard import store_eic

    r = store_eic.load_regime(_regime(tmp_path))
    r["eval_chroms"] = ["chr20", "chr21", "chr22"]
    assert store_eic.shared_hparams_chrom(r) == "chr20"
    r["eval_chroms"] = ["chr1", "chr21"]                   # chr1 is (10, 10, 45), chr21 is (25, 30, 60)
    with pytest.raises(store_eic.ScopeError, match="do not share one"):
        store_eic.shared_hparams_chrom(r)


def test_the_deferred_whole_genome_scope_is_refused_by_name(tmp_path):
    """Packing 22 whole chromosomes end to end is `eic.gw`, which §3 defers — and ~129 GiB."""
    from lavawizard import store_eic

    r = store_eic.load_regime(_regime(tmp_path, train_chroms=("chr1", "chr2", "chr3")))
    with pytest.raises(store_eic.ScopeError, match="whole-genome regime"):
        store_eic.shared_layout(r, lambda c: 10_000, base=tmp_path)
