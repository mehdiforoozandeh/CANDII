"""`StoreDataset`: thinning (D6), the eval RNG (D22), the declared order (D14), the batch contract.

The batch contract test derives its expectation from a **real** `CandiKitH5Dataset` batch rather
than a hand-typed key list, so the two cannot drift apart without this file going red — which is
the only thing standing between the store and a `train.py` that needs edits.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from candi._vendored import CLOZE, MISSING
from candi.store import layout as L
from candi.store.layout import StoreError
from candi.store.dataset import META_ROWS, StoreDataset, stable_hash, thin_counts
from candi.store.reader import CorpusStore
from candi.store.regime import Regime

from tests.test_store_reader import ASSAYS, N_BINS, make_store
from tests.test_store_regime import CTX, regime_dict


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Path:
    return make_store(tmp_path_factory.mktemp("dsds"))


def make_regime(store: Path, **over) -> Regime:
    return Regime.from_dict(regime_dict(store, **over))


def make_ds(store: Path, *, regime_over=None, **kw) -> StoreDataset:
    kw.setdefault("batch_size", 2)
    return StoreDataset(make_regime(store, **(regime_over or {})), **kw)


def batches(ds: StoreDataset, num_workers: int = 0):
    return list(DataLoader(ds, batch_size=None, num_workers=num_workers))


# ---------------------------------------------------------------------------------------------
# §7.2 — thinning is exactly Binomial(c, 1/d)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("d", [2, 4, 8])
def test_thinning_matches_the_binomial_mean_and_variance(d):
    """D6 — the DSF ladder is not stored; it is generated. If this drifts, every DSF number does.

    One vectorized call of 1e6 draws, so the whole check is milliseconds.
    """
    c, n = 50, 1_000_000
    rng = np.random.default_rng(0)
    out = thin_counts(np.full(n, c, dtype=np.int64), d, rng)
    p = 1.0 / d
    assert abs(out.mean() / c - p) < 0.001, (d, out.mean() / c, p)
    assert abs(out.var() - c * p * (1 - p)) / (c * p * (1 - p)) < 0.02


def test_thinning_never_invents_a_read():
    rng = np.random.default_rng(1)
    c = rng.integers(0, 500, size=200_000).astype(np.int64)
    for d in (1, 2, 4, 8, 3.7):
        assert np.all(thin_counts(c, d, rng) <= c)
    assert np.array_equal(thin_counts(c, 1, rng), c)          # DSF1 is the identity, not a draw
    with pytest.raises(StoreError, match="invent"):
        thin_counts(c, 0.5, rng)


def test_the_stable_hash_is_not_pythons_salted_one():
    """D22 hangs on this: `hash()` is salted per process, so a `hash()`-seeded eval is per-worker."""
    import subprocess
    import sys

    code = (
        "from candi.store.dataset import stable_hash; print(stable_hash('T_DND-41'))"
    )
    outs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PYTHONHASHSEED": str(s), "PATH": "/usr/bin:/bin"}).stdout.strip()
        for s in (0, 1, 2)
    }
    assert outs == {str(stable_hash("T_DND-41"))}
    assert len(outs) == 1


# ---------------------------------------------------------------------------------------------
# §7.3 — eval determinism (D22)
# ---------------------------------------------------------------------------------------------


def _by_window(batch_list):
    """`(biosample, window_idx) -> x_data row`, so the comparison is per WINDOW, not per batch."""
    out = {}
    for b in batch_list:
        for j, wi in enumerate(b["window_idx"].tolist()):
            out[(b["biosample_name"], int(wi))] = (
                b["x_data"][j].clone(), b["y_data"][j].clone(), b["x_dsf"][j].clone()
            )
    return out


def test_eval_windows_are_identical_across_worker_counts_and_batch_orders(store):
    """Two independent reads with different worker counts AND shuffling must agree bit for bit.

    `seed` orders the pool; `run_seed` seeds the thinning. They are separate arguments precisely
    so this comparison is possible: same run, different shuffle, same numbers.
    """
    a = _by_window(batches(make_ds(store, train=False, shuffle=False, run_seed=7), num_workers=0))
    b = _by_window(
        batches(make_ds(store, train=False, shuffle=True, seed=99, run_seed=7), num_workers=2)
    )
    assert a and set(a) == set(b)
    for key, (xa, ya, da) in a.items():
        xb, yb, db = b[key]
        assert torch.equal(da, db), f"{key}: the DSF draw itself moved with the batch order"
        assert torch.equal(xa, xb), f"{key}: x_data differs across worker count / order"
        assert torch.equal(ya, yb), f"{key}: y_data differs across worker count / order"


def test_eval_determinism_is_not_vacuous(store):
    """The counter-based RNG must be doing the work — a free-running one gives a different answer.

    Same windows, same order, only the stream seed changes. Free-running: the numbers move.
    Counter-based: they do not, which is the whole of D22.
    """
    free = [
        _by_window(batches(make_ds(store, train=False, shuffle=False, seed=s,
                                   deterministic=False)))
        for s in (1, 2)
    ]
    assert any(not torch.equal(free[0][k][0], free[1][k][0]) for k in free[0])
    det = [
        _by_window(batches(make_ds(store, train=False, shuffle=False, seed=s, run_seed=7)))
        for s in (1, 2)
    ]
    assert all(torch.equal(det[0][k][0], det[1][k][0]) for k in det[0])


def test_the_eval_seed_tuple_is_the_one_the_plan_specifies(store):
    """`SeedSequence([run_seed, h(bios), h(assay), chrom_id, window_start, dsf_milli])`."""
    from candi.store.dataset import draw_seed, dsf_milli

    seq = draw_seed(42, "T_aa", "H3K4me3", "chr2", 448, dsf_milli(4))
    assert list(seq.entropy) == [
        42, stable_hash("T_aa"), stable_hash("H3K4me3"), stable_hash("chr2"), 448, 4000
    ]
    # a different run_seed, window or dsf is a different stream; the same tuple is the same stream
    assert list(draw_seed(42, "T_aa", "H3K4me3", "chr2", 448, 4000).entropy) == list(seq.entropy)
    assert list(draw_seed(43, "T_aa", "H3K4me3", "chr2", 448, 4000).entropy) != list(seq.entropy)


def test_a_different_run_seed_gives_different_eval_noise(store):
    a = _by_window(batches(make_ds(store, train=False, run_seed=1)))
    b = _by_window(batches(make_ds(store, train=False, run_seed=2)))
    assert any(not torch.equal(a[k][0], b[k][0]) for k in a)


# ---------------------------------------------------------------------------------------------
# §7.5 — the declared order, at the level that matters
# ---------------------------------------------------------------------------------------------


def test_a_permuted_assay_list_permutes_the_batch_columns(store):
    """`dsf_sampling="off"` so the comparison is of columns, not of two different noise draws.

    (Under a free-running train RNG the per-assay DSF draws come off one stream in column order,
    so permuting the declaration legitimately re-pairs assays with draws.)
    """
    kw = dict(train=True, shuffle=False, dsf_sampling="off")
    forward = batches(make_ds(store, **kw))[0]
    rev = list(reversed(ASSAYS))
    permuted = batches(make_ds(store, regime_over={"assays": rev}, **kw))[0]
    assert forward["x_data"].shape == permuted["x_data"].shape
    for i, assay in enumerate(ASSAYS):
        j = rev.index(assay)
        assert torch.equal(forward["x_data"][..., i], permuted["x_data"][..., j])
        assert torch.equal(forward["y_pval"][..., i], permuted["y_pval"][..., j])
    # and the values really moved: column 0 is a different assay in the two batches
    assert not torch.equal(forward["x_data"][..., 0], permuted["x_data"][..., 0])
    # the assay_id metadata row follows the declaration, not the storage order
    assert forward["x_meta"][0, 1, :].tolist() == [0.0, 1.0, 2.0]


def test_an_assay_absent_from_the_store_raises_naming_it(store):
    with pytest.raises(StoreError, match="H3K27ac"):
        make_ds(store, regime_over={"assays": list(ASSAYS) + ["H3K27ac"]})


# ---------------------------------------------------------------------------------------------
# §7.8 — the batch dict contract
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kit_batch(tmp_path_factory):
    """One real `CandiKitH5Dataset` batch — the contract this dataset has to match."""
    from candi.dataset import CandiKitH5Dataset

    from tests.test_bake_gates import write_v2_h5

    h5 = write_v2_h5(tmp_path_factory.mktemp("kit") / "v2.h5")
    ds = CandiKitH5Dataset(h5, "type1", train=True, batch_size=2, dsf_sampling="uniform",
                           shuffle=False, h5_cache_ram=False)
    return next(iter(ds))


def test_store_dataset_emits_exactly_the_kit_key_set(store, kit_batch):
    got = batches(make_ds(store, train=True, shuffle=False))[0]
    assert set(got) == set(kit_batch), (
        f"only in StoreDataset: {sorted(set(got) - set(kit_batch))}; "
        f"only in CandiKitH5Dataset: {sorted(set(kit_batch) - set(got))}"
    )


def test_every_shared_key_has_the_same_rank_and_dtype(store, kit_batch):
    got = batches(make_ds(store, train=True, shuffle=False))[0]
    for key, want in kit_batch.items():
        mine = got[key]
        if isinstance(want, torch.Tensor):
            assert isinstance(mine, torch.Tensor), key
            assert mine.ndim == want.ndim, (key, mine.shape, want.shape)
            assert mine.dtype == want.dtype, (key, mine.dtype, want.dtype)
        else:
            assert type(mine) is type(want), key


def test_the_shapes_are_the_documented_ones(store):
    ds = make_ds(store, train=True, shuffle=False, batch_size=3)
    b = batches(ds)[0]
    B, Lb, F, R = 3, CTX, len(ASSAYS), 4
    assert b["x_data"].shape == b["y_data"].shape == (B, Lb, F)
    assert b["y_pval"].shape == b["y_peaks"].shape == (B, Lb, F)
    assert b["x_meta"].shape == b["y_meta"].shape == (B, R, F)
    assert b["x_avail"].shape == b["y_avail"].shape == (B, F)
    assert b["x_dna"].shape == (B, Lb * ds.resolution, 4)
    assert b["control_data"].shape == (B, Lb, 1)
    assert b["control_meta"].shape == (B, R, 1)
    assert b["control_avail"].shape == (B, 1)
    assert b["region_type"].shape == (B,) and b["window_idx"].shape == (B,)
    assert b["x_dsf"].shape == b["y_dsf"].shape == (B, F)
    assert b["control_x_dsf"].tolist() == [1, 1, 1]        # the control is never thinned
    assert b["biosample_name"] == "T_aa"
    assert set(b["region_type"].tolist()) == {255}         # bake.py::REGION_TILE


def test_missing_semantics_match_the_old_path(store):
    """`V_aa` has no ATAC-seq: MISSING data, MISSING meta, zero availability — never a zero fill."""
    b = batches(make_ds(store, train=False, shuffle=False))[0]
    assert b["biosample_name"] == "V_aa"
    i = ASSAYS.index("ATAC-seq")
    assert torch.all(b["x_data"][..., i] == MISSING) and torch.all(b["y_data"][..., i] == MISSING)
    assert torch.all(b["x_meta"][:, :, i] == MISSING)
    assert torch.all(b["x_avail"][:, i] == 0.0) and torch.all(b["y_avail"][:, i] == 0.0)
    for j, assay in enumerate(ASSAYS):
        if assay != "ATAC-seq":
            assert torch.all(b["x_data"][..., j] >= 0.0)
            assert torch.all(b["x_avail"][:, j] == 1.0)


def test_y_pval_reaches_the_batch_as_raw_minus_log10_p(store):
    """The second link of the spaces contract (D26), pinned where the model actually sees it.

    `tests/test_store_reader.py` pins that `BiosampleStore.pval` decodes the arcsinh codec. This
    pins that NOTHING between the reader and the batch bends it again: `y_pval` is the reader's own
    array, value for value, so the ONLY place a p-value changes space on the way to the loss is
    `train.py::_apply_signal_target_transform` (D30) — and the only place it changes back is
    `bench.harness.score_track`. A second transform hiding in the loader would make both of those
    wrong while every shape and dtype check still passed.
    """
    ds = make_ds(store, regime_over={"kinds": ["counts", "peaks", "pval"]},
                 train=True, shuffle=False, batch_size=2)
    b = batches(ds)[0]
    bs = CorpusStore(store)[b["biosample_name"]]
    checked = 0
    for j, wi in enumerate(b["window_idx"].tolist()):
        chrom, start = ds._windows[wi]
        for assay in ASSAYS:
            i = ds.assays.index(assay)
            # An assay the cell does not have is a zero column here, not a read (`want_pval` gates
            # on availability); the MISSING semantics of that column are pinned elsewhere.
            if not bs.has(assay, "pval") or float(b["y_avail"][j, i]) != 1.0:
                continue
            want = bs.pval(chrom, start, start + CTX, assays=[assay])[:, 0]
            got = b["y_pval"][j, :, i].numpy()
            np.testing.assert_array_equal(got, want)
            checked += 1
    assert checked, "no pval track was compared — the fixture stopped carrying one"
    # And it is the DECODED value, not the stored code: the fixture's tracks run well past 1.0,
    # where the fixed point and the arcsinh of it are both visibly different numbers.
    assert float(b["y_pval"].max()) > 1.0


def test_the_loader_never_emits_cloze(store):
    """CLOZE is the masker's mark. A loader that writes it is masking behind the masker's back."""
    for b in batches(make_ds(store, train=True, shuffle=False))[:3]:
        for key in ("x_data", "y_data", "x_meta", "y_meta", "control_data", "control_meta"):
            assert not bool((b[key] == CLOZE).any()), key


def test_availability_agrees_with_the_signal(store):
    """`encoder.py::_prepare_signal` raises when meta-availability and signal-availability differ."""
    for b in batches(make_ds(store, train=False, shuffle=False)):
        from_meta = (b["x_meta"][:, :4, :] != MISSING).all(dim=1).float()
        assert torch.equal(from_meta, b["x_avail"])
        from_signal = (b["x_data"] != MISSING).any(dim=1).float()
        assert torch.equal(from_signal, b["x_avail"])


def test_control_is_a_column_of_counts_not_a_kind(store):
    b = batches(make_ds(store, train=True, shuffle=False))[0]
    assert torch.all(b["control_data"] >= 0) and b["control_data"].sum() > 0
    assert torch.all(b["control_avail"] == 1.0)
    assert b["control_meta"][0, 1, 0] == float(len(ASSAYS))   # the control's own assay_id
    # V_aa has no control column at all — MISSING in BOTH data and meta (t15), never 0. A 0-filled
    # control column is a real, very shallow control as far as `encoder._prepare_signal` can tell:
    # it ignores `control_avail` and infers availability from the signal and the metadata, and
    # all-zero agrees with all-zero that the column is present.
    v = batches(make_ds(store, train=False, shuffle=False))[0]
    assert torch.all(v["control_data"] == float(MISSING))
    assert torch.all(v["control_meta"] == float(MISSING))
    assert torch.all(v["control_avail"] == 0.0)


def test_a_control_the_manifest_does_not_describe_is_missing_not_half_filled(store, capsys):
    """t15 / D19 — present on disk but incomplete is the SAME state as absent, for the control too.

    The one combination that must never be emitted is real coverage beside `-1` metadata:
    `encoder._prepare_signal` raises when signal and metadata disagree about availability, so a
    half-filled control would turn a data-quality problem into a mid-training crash.
    """
    from candi.store import layout as L

    mpath = L.manifest_path(store)
    m = json.loads(mpath.read_text(encoding="utf-8"))
    for rec in m["biosamples"]["T_aa"]["tracks"]:
        if rec["assay"] == L.CONTROL_TRACK:
            rec["depth"] = None                     # what an undescribed track looks like
    mpath.write_text(json.dumps(m), encoding="utf-8")

    ds = make_ds(store, train=True, shuffle=False)
    assert any(g.endswith(L.CONTROL_TRACK) for g in ds._gaps), "the gap must be reported, not silent"
    b = batches(ds)[0]
    assert torch.all(b["control_data"] == float(MISSING))
    assert torch.all(b["control_meta"] == float(MISSING))
    assert torch.all(b["control_avail"] == 0.0)


# ---------------------------------------------------------------------------------------------
# the depth covariate has to move with the thinning
# ---------------------------------------------------------------------------------------------


def test_the_depth_row_tracks_the_dsf(store):
    """`prep/bake.py`'s F4 gate: `meta_dsf{d}[0] == meta_dsf1[0] - log2(d)`, now at load time."""
    b = batches(make_ds(store, train=True, shuffle=False))[0]
    base = np.log2(20_000_000)
    for j in range(b["x_meta"].shape[0]):
        for i in range(len(ASSAYS)):
            d = float(b["x_dsf"][j, i])
            assert b["x_meta"][j, 0, i].item() == pytest.approx(base - np.log2(d), abs=1e-4)
            assert b["x_meta"][j, 2, i].item() == 36.0        # read_length
            assert b["x_meta"][j, 3, i].item() == 0.0         # single-ended


def test_dsf_off_leaves_the_counts_untouched(store):
    raw = make_ds(store, train=True, shuffle=False, dsf_sampling="off")
    b = batches(raw)[0]
    corpus = CorpusStore(store)
    chrom, start = raw._windows[int(b["window_idx"][0])]
    want = corpus["T_aa"].counts(chrom, start, start + CTX, assays=list(ASSAYS))
    assert np.array_equal(b["x_data"][0].numpy().astype(np.int64), want.astype(np.int64))
    assert torch.all(b["x_dsf"] == 1)
    corpus.close()


def test_x_eq_y_matches_the_depth_but_not_the_noise(store):
    """The mode equalises `dsf`, and `dsf` alone -- see `dataset.py::_sample_xy_dsf`.

    Identical VALUES were never part of its definition; they were what the old bake's materialised
    ladder did, because equal `d` there meant reading one stored array twice. The store generates
    its ladder, so equal depth is two independent binomial draws -- which is also what two real
    experiments sequenced to the same depth look like. The perfect-copy control is
    `dsf_sampling="off"`, where there is no thinning at all and x genuinely is y.
    """
    b = batches(make_ds(store, train=False, shuffle=False, dsf_sampling="x_eq_y"))[0]
    assert torch.equal(b["x_dsf"], b["y_dsf"])
    thinned = (b["x_dsf"] > 1).any()
    if thinned:
        assert not torch.equal(b["x_data"], b["y_data"])


def test_upsample_only_never_lets_y_be_deeper_than_x(store):
    for b in batches(make_ds(store, train=False, shuffle=False, dsf_sampling="upsample_only")):
        assert torch.all(b["y_dsf"] >= b["x_dsf"])


# ---------------------------------------------------------------------------------------------
# D19 — nothing is fabricated
# ---------------------------------------------------------------------------------------------


def test_a_track_with_incomplete_metadata_is_emitted_as_missing(tmp_path):
    """No `read_length` -> no honest metadata column -> MISSING, never `read_length = 50`."""
    corpus_root = make_store(tmp_path / "gap", drop_meta=[("T_aa", "H3K4me3")])
    ds = StoreDataset(make_regime(corpus_root), train=True, shuffle=False, batch_size=2)
    b = batches(ds)[0]
    i = ASSAYS.index("H3K4me3")
    assert torch.all(b["x_avail"][:, i] == 0.0)
    assert torch.all(b["x_data"][..., i] == MISSING)
    assert "T_aa/H3K4me3" in ds._gaps
    with pytest.raises(StoreError, match="meta_missing='error'"):
        StoreDataset(make_regime(corpus_root), train=True, meta_missing="error")


def test_a_store_without_a_manifest_says_what_is_missing(tmp_path):
    corpus_root = make_store(tmp_path / "nomani")
    L.manifest_path(corpus_root).unlink()
    with pytest.raises(StoreError, match="build-manifest"):
        StoreDataset(make_regime(corpus_root), train=True)


def test_cell_cond_is_refused_rather_than_guessed(store):
    with pytest.raises(StoreError, match="D16"):
        make_ds(store, train=True, cell_cond="id")


# ---------------------------------------------------------------------------------------------
# plumbing the rest of the kit relies on
# ---------------------------------------------------------------------------------------------


def test_the_dataset_exposes_the_scale_train_py_reads(store):
    ds = make_ds(store, train=True)
    assert ds.assays == list(ASSAYS) and ds.num_assays == ds.signal_dim == 3
    assert ds.context_bins == CTX and ds.resolution == 25
    assert ds.train_chroms == ("chr1",) and ds.eval_chroms == ("chr2",)
    assert ds.dsf_list == (1, 2, 4, 8)
    assert ds.estimate_steps_per_epoch() == len(ds) > 0


def test_workers_shard_the_windows_without_overlap_or_loss(store):
    one = _by_window(batches(make_ds(store, train=False, shuffle=False), num_workers=0))
    two = batches(make_ds(store, train=False, shuffle=False), num_workers=2)
    seen = [(b["biosample_name"], int(w)) for b in two for w in b["window_idx"].tolist()]
    assert sorted(seen) == sorted(one)
    assert len(seen) == len(set(seen))


def test_a_batch_is_one_biosample(store):
    for b in batches(make_ds(store, train=True, shuffle=True)):
        assert isinstance(b["biosample_name"], str)


def test_windows_come_from_the_regime_plan_not_the_store(store):
    ds768 = make_ds(store, train=True, regime_over={"context_bins": 128,
                                                   "window_plan": {"type": "tile",
                                                                   "stride_bins": 128}})
    b = batches(ds768)[0]
    assert b["x_data"].shape[1] == 128
    assert b["x_dna"].shape[1] == 128 * 25
    assert len(ds768._windows) == len(range(0, N_BINS["chr1"] - 128 + 1, 128))


# ---------------------------------------------------------------------------------------------
# t14 / D31 — the eval-only imputation keys
# ---------------------------------------------------------------------------------------------
# `eval.py` reads y_data_imp / y_pval_imp / y_peaks_imp / y_meta_imp / imp_biosample_name through
# `batch.get(...)`, so a loader that omits them makes an imputation eval score nothing WITHOUT
# raising. The synthetic store is built for this: `T_aa` has all three assays and a control, `V_aa`
# has only DNase-seq and H3K4me3.

IMP_KEYS = ("y_data_imp", "y_pval_imp", "y_peaks_imp", "y_meta_imp", "imp_biosample_name")


def test_without_eval_pairs_nothing_changes(store):
    """Additive: a regime written before t14 gets exactly the batch it always got."""
    b = batches(make_ds(store, train=False, shuffle=False))[0]
    for k in IMP_KEYS:
        assert k not in b, f"{k} must not appear without a declared pair"


def test_declared_eval_pairs_emit_all_five_keys(store):
    ds = make_ds(store, train=False, shuffle=False,
                 regime_over={"eval_pairs": [["T_aa", "V_aa"]]})
    b = batches(ds)[0]
    B, Lb, F = 2, CTX, len(ASSAYS)
    assert b["biosample_name"] == "T_aa" and b["imp_biosample_name"] == "V_aa"
    assert b["y_data_imp"].shape == (B, Lb, F)
    assert b["y_pval_imp"].shape == (B, Lb, F)
    assert b["y_peaks_imp"].shape == (B, Lb, F)
    assert b["y_meta_imp"].shape == (B, META_ROWS, F)
    # The fills are the bake's, and they are NOT the same value for all three: `eval.py` builds its
    # imputation mask as `(y_avail <= 0) & (y_data_imp != -1)`, so -1 in y_data_imp means "the
    # target does not have this assay" while 0 in y_pval_imp/y_peaks_imp is just an empty track.
    atac = ASSAYS.index("ATAC-seq")                       # V_aa does not have it
    dnase = ASSAYS.index("DNase-seq")                     # V_aa does
    assert torch.all(b["y_data_imp"][:, :, atac] == float(MISSING))
    assert torch.all(b["y_meta_imp"][:, :, atac] == float(MISSING))
    assert torch.all(b["y_data_imp"][:, :, dnase] >= 0)
    assert float(b["y_data_imp"][:, :, dnase].sum()) > 0
    assert float(b["y_meta_imp"][0, 0, dnase]) != float(MISSING)


def test_the_imputation_target_is_the_partner_not_the_prompt(store):
    """The whole point of the pair: y_data_imp must be the TARGET's counts, not the input's."""
    ds = make_ds(store, train=False, shuffle=False, dsf_sampling="off",
                 regime_over={"eval_pairs": [["T_aa", "V_aa"]]})
    b = batches(ds)[0]
    dnase = ASSAYS.index("DNase-seq")
    v = CorpusStore(store)["V_aa"]
    for j, wi in enumerate(b["window_idx"].tolist()):
        chrom, start = ds._windows[wi]
        want = v.counts(chrom, start, start + CTX, assays=["DNase-seq"])[:, 0]
        assert np.allclose(b["y_data_imp"][j, :, dnase].numpy(), want)
    # and it is genuinely a different array from the prompt's own target
    assert not torch.equal(b["y_data"][:, :, dnase], b["y_data_imp"][:, :, dnase])


def test_the_eval_mask_eval_py_builds_is_non_empty_for_a_real_pair(store):
    """`(y_avail <= 0) & (y_data_imp != -1)` — an assay the prompt lacks and the target has.

    Prompting with `V_aa` (no ATAC-seq) against `T_aa` (has it) is the direction that produces an
    imputation target at all, and it is what makes this key set worth emitting.
    """
    ds = make_ds(store, train=False, shuffle=False,
                 regime_over={"biosamples": {"train": [], "eval": ["T_aa"]},
                              "eval_pairs": [["V_aa", "T_aa"]]})
    b = batches(ds)[0]
    imp = (b["y_avail"] <= 0).unsqueeze(1).expand_as(b["y_data_imp"]) & (b["y_data_imp"] != -1)
    assert int(imp.sum()) > 0
    atac = ASSAYS.index("ATAC-seq")
    assert bool(imp[:, :, atac].all()), "ATAC-seq is exactly the assay V_aa lacks and T_aa has"


def test_every_declared_pair_gets_its_own_batches(store):
    ds = make_ds(store, train=False, shuffle=False,
                 regime_over={"biosamples": {"train": [], "eval": ["T_aa", "V_aa"]},
                              "eval_pairs": [["T_aa", "V_aa"], ["V_aa", "T_aa"]]})
    seen = {(b["biosample_name"], b["imp_biosample_name"]) for b in batches(ds)}
    assert seen == {("T_aa", "V_aa"), ("V_aa", "T_aa")}


def test_the_eval_surface_eval_py_reads_off_the_dataset(store):
    """`build_eval_units` does its slot arithmetic through these three names."""
    ds = make_ds(store, train=False, shuffle=False,
                 regime_over={"biosamples": {"train": [], "eval": ["V_aa"]},
                              "eval_pairs": [["T_aa", "V_aa"]]})
    assert ds._bios_candidates() == ["T_aa"]
    assert ds._all_imp_biosamples("T_aa") == ["V_aa"]
    assert ds._all_imp_biosamples("V_aa") == []
    assert len(ds._eval_indices) == len(ds._windows) > 0


def test_a_target_is_read_but_never_prompted_with(store):
    """It must not enter `biosample_pool`, or `depth_center` would average the held-out split in."""
    ds = make_ds(store, train=False, shuffle=False,
                 regime_over={"eval_pairs": [["T_aa", "V_aa"]]})
    assert ds.biosample_pool == ["T_aa"] and ds.imp_targets == ["V_aa"]
    assert "V_aa" in ds._meta, "the target still needs metadata, for y_meta_imp"


# --- the x/y term in the thinning seed (the identity-copy leak) ---------------------------------
# `_thin` is called twice per column per window -- once for the input, once for the target -- off
# ONE entropy tuple. Before the `side` term that tuple could not tell the two calls apart, so at
# `x_dsf == y_dsf` the deterministic RNG built the same generator twice and the target came back
# bit-identical to the input. `tests/test_dsf_leak.py` measures the rate; these pin the mechanism.


def test_omitting_the_side_reproduces_the_pre_fix_seed_tuple(store):
    """`candi.bench.harness` thins through `draw_seed` and must not move. This is that promise.

    The bench sites share a tuple exactly as `_thin`'s two calls did, and whether that is a paired
    depth sweep or the same leak is an open measurement question. Until it is answered, omitting
    `side` has to give the OLD six-element tuple, byte for byte.
    """
    from candi.store.dataset import draw_seed

    assert list(draw_seed(42, "T_aa", "H3K4me3", "chr2", 448, 4000).entropy) == [
        42, stable_hash("T_aa"), stable_hash("H3K4me3"), stable_hash("chr2"), 448, 4000
    ]


def test_the_side_is_the_seventh_element_and_x_is_not_y(store):
    from candi.store.dataset import draw_seed

    x = list(draw_seed(42, "T_aa", "H3K4me3", "chr2", 448, 4000, "x").entropy)
    y = list(draw_seed(42, "T_aa", "H3K4me3", "chr2", 448, 4000, "y").entropy)
    bare = list(draw_seed(42, "T_aa", "H3K4me3", "chr2", 448, 4000).entropy)
    assert x[:6] == y[:6] == bare                      # only the new element differs
    assert len(x) == len(y) == 7 and x[6] != y[6]
    assert x[6] == stable_hash("x") and y[6] == stable_hash("y")


def test_thin_refuses_to_be_called_without_a_side(store):
    """Keyword-only, no default. The bug was invisible because both calls read identically."""
    ds = make_ds(store)
    with pytest.raises(TypeError):
        ds._thin(np.array([10, 20], dtype=np.int64), 2.0, "T_aa", "H3K4me3", "chr2", 0,
                 np.random.default_rng(0))


def test_the_deterministic_rng_no_longer_hands_the_target_back_as_the_input(store):
    """The fix itself: same column, same dsf, same window -- and now two different draws.

    Counts are large enough that two independent binomial thinnings colliding by chance is not a
    thing that happens; at dsf 2 over 512 bins the probability is astronomically small.
    """
    ds = make_ds(store, deterministic=True)
    col = np.random.default_rng(7).integers(20, 200, size=512).astype(np.int64)
    free = np.random.default_rng(0)
    x = ds._thin(col, 2.0, "T_aa", "H3K4me3", "chr2", 448, free, side="x")
    y = ds._thin(col, 2.0, "T_aa", "H3K4me3", "chr2", 448, free, side="y")
    assert not np.array_equal(x, y), "the target is still the input"


def test_the_fix_does_not_cost_determinism(store):
    """Two runs of the same side must still agree -- D22 is the reason the seed is built at all."""
    ds = make_ds(store, deterministic=True)
    col = np.random.default_rng(7).integers(20, 200, size=512).astype(np.int64)
    a = ds._thin(col, 4.0, "T_aa", "H3K4me3", "chr2", 448, np.random.default_rng(1), side="x")
    b = ds._thin(col, 4.0, "T_aa", "H3K4me3", "chr2", 448, np.random.default_rng(2), side="x")
    assert np.array_equal(a, b)


def test_no_thinning_still_means_x_equals_y(store):
    """Not this bug, and deliberately left alone: at dsf 1 there is no draw to make."""
    ds = make_ds(store, deterministic=True)
    col = np.arange(64, dtype=np.int64)
    free = np.random.default_rng(0)
    x = ds._thin(col, 1.0, "T_aa", "H3K4me3", "chr2", 0, free, side="x")
    y = ds._thin(col, 1.0, "T_aa", "H3K4me3", "chr2", 0, free, side="y")
    assert np.array_equal(x, y)
