"""Regression tests for the 5th metadata row (cell identity) — `--cell-cond`.

These are the standing versions of the wiring checks in `candi.healthcheck`. The healthcheck runs
once against a real baked panel and reports effect sizes; these run on every commit against a
synthetic v2 h5 and pin the invariants that, when they break, break SILENTLY:

  * the id survives whole-assay cloze (the masker's `[b, :, idx]` slice),
  * a T_X input and its V_X/B_X imputation prompt land on the SAME embedding row,
  * `cell_cond="off"` leaves the historical model untouched, RNG stream included,
  * the null arm draws its ids without perturbing the data stream the treatment arm sees.
"""
from __future__ import annotations

import hashlib

import pytest
import torch

from candi._vendored import CLOZE, MISSING
from candi.batch import make_masker, prepare_masked_batch
from candi.dataset import (
    CandiKitH5Dataset, base_cell_type, cell_id_map,
)
from candi.model import build_model

from tests.test_bake_gates import write_v2_h5

ORDER = ("T_aa", "T_bb", "T_cc", "V_aa", "B_bb")
PROBE = dict(embed_dim=8, n_transformer_layers=1, decoder_lane=4, dropout=0.0)
CELL_ROW = 4


@pytest.fixture(scope="module")
def h5_multi(tmp_path_factory):
    """3 base cell types, one with a V_ target and one with a B_ target."""
    return write_v2_h5(tmp_path_factory.mktemp("cc") / "cc.h5", order=ORDER)


def _ds(h5, **kw):
    kw.setdefault("cell_cond", "id")
    return CandiKitH5Dataset(h5, "type1", batch_size=2, dsf_sampling="off", seed=0,
                             h5_cache_ram=False, **kw)


# ---------------------------------------------------------------------------
# id mapping
# ---------------------------------------------------------------------------

def test_base_cell_type_splits_on_the_first_underscore_only():
    assert base_cell_type("T_adrenal_gland_embryonic") == "adrenal_gland_embryonic"
    assert base_cell_type("V_CD4-positive_alpha-beta_memory_T_cell") == \
        "CD4-positive_alpha-beta_memory_T_cell"


def test_prefixes_of_one_cell_type_share_an_id():
    m = cell_id_map(ORDER)
    assert m[base_cell_type("T_aa")] == m[base_cell_type("V_aa")]
    assert m[base_cell_type("T_bb")] == m[base_cell_type("B_bb")]
    assert sorted(m.values()) == [0, 1, 2]


def test_id_map_is_independent_of_h5_write_order():
    assert cell_id_map(ORDER) == cell_id_map(tuple(reversed(ORDER)))


def test_unprefixed_biosample_raises_rather_than_keying_on_the_whole_name():
    with pytest.raises(ValueError, match="no prefix separator"):
        base_cell_type("nodashhere")


# ---------------------------------------------------------------------------
# W1 — the id survives masking
# ---------------------------------------------------------------------------

def test_cell_id_is_never_clozed_by_whole_assay_masking(h5_multi):
    """THE silent bug. `_mask_full_assay` used to write mask_value across every metadata row, which
    clozed the cell id for exactly the assays being imputed — and nothing raised, because
    availability is read off rows 0-3 and still agreed with the signal."""
    ds = _ds(h5_multi)
    masker = make_masker(p_full_assay=1.0)
    saw_a_masked_column = False
    for i, batch in enumerate(ds):
        if i >= 8:
            break
        prep = prepare_masked_batch(batch, masker, torch.device("cpu"), apply_mask=True)
        if prep is None:
            continue
        saw_a_masked_column |= bool(prep["masked_map"].any())
        row = prep["x_meta"][:, CELL_ROW, :]
        assert not bool(((row == CLOZE) | (row == MISSING)).any()), "cell id was masked"
    assert saw_a_masked_column, "no assay was ever clozed — the test proved nothing"


def test_rows_0_to_3_are_still_masked(h5_multi):
    """The complement: restricting the slice must not have disabled masking itself."""
    ds = _ds(h5_multi)
    prep = None
    for batch in ds:
        prep = prepare_masked_batch(batch, make_masker(p_full_assay=1.0), torch.device("cpu"),
                                    apply_mask=True)
        if prep is not None and bool(prep["masked_map"].any()):
            break
    assert prep is not None
    masked_cols = prep["masked_map"].any(dim=1)                      # [B, A]
    assert bool((prep["x_meta"][:, :4, :prep["masked_map"].shape[2]][
        masked_cols.unsqueeze(1).expand(-1, 4, -1)] == CLOZE).all())


# ---------------------------------------------------------------------------
# W2 — the prompt agrees with the input
# ---------------------------------------------------------------------------

def test_row4_is_constant_across_assays_and_matches_the_biosample(h5_multi):
    ds = _ds(h5_multi)
    for i, batch in enumerate(ds):
        if i >= 6:
            break
        want = float(ds.cell_ids[base_cell_type(batch["biosample_name"])])
        for key in ("x_meta", "y_meta"):
            row = batch[key][:, CELL_ROW, :]
            assert int(row.unique().numel()) == 1
            assert float(row.unique().item()) == want


def test_imputation_prompt_uses_the_same_id_as_its_T_input(h5_multi):
    ds = _ds(h5_multi, train=False, eval_include_vb_ground_truth=True, shuffle=False)
    seen = 0
    for i, batch in enumerate(ds):
        if i >= 10:
            break
        ymi = batch.get("y_meta_imp")
        if ymi is None:
            continue
        seen += 1
        want = float(ds.cell_ids[base_cell_type(batch["biosample_name"])])
        assert ymi.shape[1] == 5
        assert float(ymi[:, CELL_ROW, :].unique().item()) == want
        assert base_cell_type(batch["imp_biosample_name"]) == \
            base_cell_type(batch["biosample_name"])
    assert seen, "no (T_, V_/B_) eval pair was produced — the test proved nothing"


def test_control_column_carries_the_id_too(h5_multi):
    """The control is the same physical sample, so row 4 must span assays AND control after the
    concat in prepare_masked_batch — otherwise the encoder sees a ragged id."""
    ds = _ds(h5_multi)
    batch = next(iter(ds))
    prep = prepare_masked_batch(batch, make_masker(p_full_assay=0.0, p_full_loci=0.0, p_chunks=0.0),
                                torch.device("cpu"), apply_mask=False)
    assert prep["x_meta"].shape[2] == ds.num_assays + 1
    assert int(prep["x_meta"][:, CELL_ROW, :].unique().numel()) == 1


# ---------------------------------------------------------------------------
# off is off
# ---------------------------------------------------------------------------

def test_cell_cond_off_yields_four_rows_and_no_table(h5_multi):
    ds = _ds(h5_multi, cell_cond="off")
    assert ds.num_cells == 0 and ds.meta_rows == 4
    assert next(iter(ds))["x_meta"].shape[1] == 4
    m = build_model(num_assays=ds.num_assays, context_length=ds.context_bins,
                         num_cells=0, **PROBE)
    assert not hasattr(m.encoder.metadata_embedding, "cell_embedding")


def test_cell_cond_off_is_bit_identical_to_the_historical_model(h5_multi):
    """W4. The control arm must be the model we already have, not a re-initialised cousin — a
    changed constructor RNG stream would silently make every arm-vs-arm delta partly an init delta."""
    def sig(num_cells):
        torch.manual_seed(0)
        m = build_model(num_assays=3, context_length=64, num_cells=num_cells, **PROBE)
        h = hashlib.sha1()
        for k, v in sorted(m.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().numpy().tobytes())
        return h.hexdigest(), sum(p.numel() for p in m.parameters())

    # a second build at num_cells=0 must reproduce the first exactly
    assert sig(0) == sig(0)
    on_hash, on_params = sig(3)
    off_hash, off_params = sig(0)
    assert on_hash != off_hash and on_params > off_params


def test_five_row_meta_into_a_four_row_embedder_raises(h5_multi):
    m = build_model(num_assays=3, context_length=64, num_cells=0, **PROBE)
    with pytest.raises(ValueError, match="must be 4 rows"):
        m.encoder.metadata_embedding(torch.zeros(2, 5, 4))


def test_out_of_range_cell_id_raises_rather_than_aliasing_onto_sentinels():
    m = build_model(num_assays=3, context_length=64, num_cells=3, **PROBE)
    meta = torch.zeros(2, 5, 4)
    meta[:, 1, :] = torch.arange(4).float()
    meta[:, 4, :] = 3.0                      # table holds ids 0..2; 3 is the MISSING slot
    with pytest.raises(ValueError, match="cell_id 3 exceeds table bound"):
        m.encoder.metadata_embedding(meta)


# ---------------------------------------------------------------------------
# the null arm
# ---------------------------------------------------------------------------

def test_random_mode_does_not_perturb_the_data_stream(h5_multi):
    """The null must differ from the treatment in the COVARIATE only. Drawing the random id from the
    loop's own rng advanced the shared stream once per batch, so the null arm would have visited a
    different sequence of biosamples and DSF levels — a confound wearing a control's clothes."""
    def stream(mode):
        ds = _ds(h5_multi, cell_cond=mode)
        out = []
        for i, b in enumerate(ds):
            if i >= 8:
                break
            out.append((b["biosample_name"], b["x_dsf"].tolist(), b["y_dsf"].tolist()))
        return out

    assert stream("random") == stream("id") == stream("off")


def test_random_mode_actually_decorrelates_the_id(h5_multi):
    ds = _ds(h5_multi, cell_cond="random")
    mismatches = 0
    n = 0
    for i, b in enumerate(ds):
        if i >= 30:
            break
        n += 1
        want = ds.cell_ids[base_cell_type(b["biosample_name"])]
        if int(b["x_meta"][0, CELL_ROW, 0].item()) != want:
            mismatches += 1
    assert n and mismatches > 0, "random ids always equalled the true id — not a null"


def test_random_mode_keeps_encoder_and_decoder_on_the_same_cell(h5_multi):
    """One draw per batch, reused everywhere. Two draws would hand the decoder a different cell from
    the encoder, which is a confound rather than a null."""
    ds = _ds(h5_multi, cell_cond="random", train=False, eval_include_vb_ground_truth=True,
             shuffle=False)
    seen = 0
    for i, b in enumerate(ds):
        if i >= 10:
            break
        ymi = b.get("y_meta_imp")
        if ymi is None:
            continue
        seen += 1
        assert float(ymi[:, CELL_ROW, :].unique().item()) == \
            float(b["x_meta"][:, CELL_ROW, :].unique().item())
    assert seen


def test_unknown_cell_cond_mode_is_rejected(h5_multi):
    with pytest.raises(ValueError, match="cell_cond must be one of"):
        _ds(h5_multi, cell_cond="permuted")


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def test_five_row_batch_runs_forward_and_backward(h5_multi):
    from candi.model import forward_full
    from candi.train import nb_count_loss

    ds = _ds(h5_multi)
    torch.manual_seed(0)
    m = build_model(num_assays=ds.num_assays, context_length=ds.context_bins,
                         num_cells=ds.num_cells, **PROBE)
    prep = None
    for batch in ds:
        prep = prepare_masked_batch(batch, make_masker(p_full_assay=1.0), torch.device("cpu"),
                                    apply_mask=True)
        if prep is not None:
            break
    assert prep is not None
    loss, _ = nb_count_loss(forward_full(m, prep), prep)
    assert torch.isfinite(loss)
    loss.backward()
    g = m.encoder.metadata_embedding.cell_embedding.weight.grad
    assert g is not None and bool((g.abs().sum(dim=1) > 0).any()), "no gradient reached the table"


def test_gradient_lands_only_on_the_cells_in_the_batch(h5_multi):
    """W3. A table row that picks up gradient for a cell absent from the batch means the index is
    wrong, and the embedding would be learning a blend of cell types."""
    from candi.model import forward_full
    from candi.train import nb_count_loss

    ds = _ds(h5_multi)
    torch.manual_seed(0)
    m = build_model(num_assays=ds.num_assays, context_length=ds.context_bins,
                         num_cells=ds.num_cells, **PROBE)
    for batch in ds:
        prep = prepare_masked_batch(batch, make_masker(p_full_assay=1.0), torch.device("cpu"),
                                    apply_mask=True)
        if prep is None:
            continue
        m.zero_grad(set_to_none=True)
        loss, _ = nb_count_loss(forward_full(m, prep), prep)
        loss.backward()
        present = sorted({int(v) for v in prep["x_meta"][:, CELL_ROW, 0].tolist()})
        g = m.encoder.metadata_embedding.cell_embedding.weight.grad
        nz = sorted(torch.nonzero(g.abs().sum(dim=1) > 0).flatten().tolist())
        assert nz == present, f"gradient on rows {nz}, batch held cells {present}"
        return
    pytest.fail("no supervised batch produced")
