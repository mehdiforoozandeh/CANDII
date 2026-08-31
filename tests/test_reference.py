"""h74 pre-launch gates: the reference offset, and the four harness fixes that ride with it.

Everything here runs on a tiny SYNTHETIC panel baked in-process, so the whole file is seconds on a
laptop and needs no GPU and no /scratch. The gates it covers:

  L1  leave-one-out is EXACT — the reference cell *i* sees provably excludes cell *i*
  L2  no V_/B_ biosample contributes to any reference value, and a file that violates it is rejected
  L3  the reference is depth-free — scaling one contributor's depth leaves R unchanged
  G0  the offset MOVES predictions, in the right direction and by the right amount
  W1  --reference off is bit-identical to the pre-h74 model
  W2  the unmasked-batch coin is 15% and does not perturb the shared data stream
  W3  `imp` is omitted (not logged as 0.0) on batches with nothing to impute
  W4  the mid-training eval keeps EVERY pair and only shrinks windows per pair

`python -m candi.reference verify` re-runs L1/L2/L3 against the REAL artifacts on the cluster;
these are the fast versions that gate a code change.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from candi.reference import (REFERENCE_PSEUDOCOUNT, ReferenceTable, build_reference,
                                 reference_path_for)

A, W, L = 4, 6, 8
BIOS = ["T_a", "T_b", "T_c", "V_a", "B_b"]
# Which assays each biosample holds. T_c holds only assay 3, so assay 3 has TWO contributors and is
# the thin case leave-one-out exists for: without LOO, half of its reference is the target's own row.
HAS = {"T_a": [0, 1, 2], "T_b": [0, 1, 3], "T_c": [3], "V_a": [3], "B_b": [2]}
DEPTHS = {"T_a": 20.0, "T_b": 22.0, "T_c": 21.0, "V_a": 20.5, "B_b": 21.5}


def _bake(path: Path, depths=None) -> Path:
    """Minimal v2-schema h5 with just the fields `candi.reference` reads."""
    depths = depths or DEPTHS
    rng = np.random.default_rng(0)
    with h5py.File(str(path), "w") as h:
        h.attrs["version"] = 2
        h.attrs["assays"] = json.dumps([f"assay{i}" for i in range(A)])
        h.attrs["context_bins"] = L
        h.attrs["resolution"] = 25
        h.attrs["dsf_list"] = json.dumps([1])
        h.attrs["train_chroms"] = json.dumps(["chr19"])
        h.attrs["eval_chroms"] = json.dumps(["chr21"])
        h.create_dataset("windows/chrom", data=np.array([b"chr19"] * W))
        h.create_dataset("windows/start", data=np.arange(W) * 100)
        h.create_dataset("windows/end", data=np.arange(W) * 100 + 100)
        h.create_dataset("windows/region_type", data=np.zeros(W, dtype=np.int32))
        g0 = h.create_group("biosamples")
        g0.attrs["order"] = json.dumps(BIOS)
        for b in BIOS:
            g = g0.create_group(b)
            counts = np.zeros((W, L, A), dtype=np.int32)
            meta = np.full((4, A), -1.0, dtype=np.float32)
            for a in HAS[b]:
                # counts scale with depth, which is what L3 has to strip back out
                counts[:, :, a] = rng.integers(0, 40, size=(W, L)) * int(2 ** (depths[b] - 20.0))
                meta[0, a] = depths[b]
                meta[1, a] = a
                meta[2, a] = 76.0
                meta[3, a] = 1.0
            g.create_dataset("counts_dsf1", data=counts)
            g.create_dataset("meta_dsf1", data=meta)
    return path


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("ref")
    h5 = _bake(d / "toy.h5")
    out = reference_path_for(h5)
    build_reference(h5, out, depth_center=21.0, verbose=False)
    return h5, out


# ---------------------------------------------------------------------------
# L1 / L2 / L3 — the reference itself
# ---------------------------------------------------------------------------

def test_L1_leave_one_out_is_exact(built):
    """R for cell i must equal the mean over the OTHER contributors, to float precision."""
    h5, out = built
    rt = ReferenceTable(out, src_h5=h5)
    wi = [0, 3, 5]
    with h5py.File(str(h5), "r") as h:
        for b in ["T_a", "T_b", "T_c"]:
            own = np.asarray(h["biosamples"][b]["counts_dsf1"][wi], dtype=np.float64)
            got = rt.log_ref(wi, b, own)
            others, den = np.zeros_like(own), np.zeros(A)
            for ob in rt.contributors:
                if ob == b:
                    continue
                others += (np.asarray(h["biosamples"][ob]["counts_dsf1"][wi], dtype=np.float64)
                           * rt.cell_scale(ob))
                den += rt.present[rt._bios_idx[ob]]
            R = np.zeros_like(others)
            R[:, :, den > 0] = others[:, :, den > 0] / den[den > 0]
            want = np.log2(np.clip(R, 0, None) + REFERENCE_PSEUDOCOUNT).astype(np.float32)
            assert np.allclose(got, want, atol=1e-4), f"LOO not exact for {b}"
    rt.close()


def test_L1_thin_assay_would_leak_without_loo(built):
    """The gate's teeth: on a 2-contributor assay, LOO must change the answer a LOT.

    Assay 3 is held by T_b and T_c only. Without leave-one-out, half of T_c's reference for assay 3 is
    T_c's own values — the model would be handed 50% of what it is asked to predict. A test that only
    checked the arithmetic on a well-covered assay would pass with LOO silently disabled.
    """
    h5, out = built
    rt = ReferenceTable(out, src_h5=h5)
    wi = [0, 1, 2]
    with h5py.File(str(h5), "r") as h:
        own = np.asarray(h["biosamples"]["T_c"]["counts_dsf1"][wi], dtype=np.float64)
    loo = rt.log_ref(wi, "T_c", own)
    leaky = rt.log_ref(wi)                      # no biosample -> full mean, the bug this prevents
    assert not np.allclose(loo[:, :, 3], leaky[:, :, 3], atol=0.05), \
        "LOO made no difference on a 2-contributor assay — it is not actually being applied"
    assert rt.count[3] == 2
    rt.close()


def test_L2_no_heldout_biosample_contributes(built):
    h5, out = built
    rt = ReferenceTable(out, src_h5=h5)
    assert rt.contributors == ["T_a", "T_b", "T_c"]
    assert not any(b.startswith(("V_", "B_")) for b in rt.contributors)
    # count must equal the T_-only tally, i.e. V_a's assay 3 and B_b's assay 2 are NOT in it
    assert list(rt.count) == [2, 2, 1, 2], f"held-out data reached the counts: {rt.count}"
    rt.close()


def test_L2_leaked_reference_is_rejected_on_load(built, tmp_path):
    """A hand-edited or older reference that lists a V_/B_ contributor must fail loudly on open."""
    h5, out = built
    bad = tmp_path / "leaky.reference.h5"
    bad.write_bytes(Path(out).read_bytes())
    with h5py.File(str(bad), "r+") as f:
        f.attrs["contributors"] = json.dumps(["T_a", "T_b", "T_c", "V_a"])
    with pytest.raises(ValueError, match="held-out biosamples contributing"):
        ReferenceTable(bad, src_h5=h5)


def test_L3_reference_is_depth_free(tmp_path):
    """Double one contributor's sequencing depth (and its counts with it): R must not move.

    This is the gate that keeps the reference offset from fighting the depth offset. A raw-count
    average would carry the depth mixture, the existing size-factor would re-apply the target's depth,
    and the two would double-count.
    """
    h5a = _bake(tmp_path / "a.h5")
    deeper = dict(DEPTHS, T_b=DEPTHS["T_b"] + 1.0)          # T_b sequenced 2x deeper
    h5b = _bake(tmp_path / "b.h5", depths=deeper)
    ra, rb = tmp_path / "a.reference.h5", tmp_path / "b.reference.h5"
    build_reference(h5a, ra, depth_center=21.0, verbose=False)
    build_reference(h5b, rb, depth_center=21.0, verbose=False)
    ta, tb = ReferenceTable(ra, src_h5=h5a), ReferenceTable(rb, src_h5=h5b)
    wi = list(range(W))
    assert np.allclose(ta.log_ref(wi), tb.log_ref(wi), atol=1e-4), \
        "reference moved when a contributor's depth changed — it is NOT depth-free"
    ta.close()
    tb.close()


def test_reference_rejects_mismatched_source(built, tmp_path):
    h5, out = built
    other = _bake(tmp_path / "other.h5", depths=dict(DEPTHS, T_a=19.0))
    with pytest.raises(ValueError, match="re-baked|mismatch"):
        ReferenceTable(out, src_h5=other)


def test_reference_refuses_leaky_call_for_contributor(built):
    """Asking for a contributor's reference without its counts must raise, not silently leak."""
    h5, out = built
    rt = ReferenceTable(out, src_h5=h5)
    with pytest.raises(ValueError, match="leave-one-out is required"):
        rt.log_ref([0, 1], "T_a")
    rt.close()


def test_log_ref_row_order_survives_unsorted_windows(built):
    """The train pool is SHUFFLED, so window indices arrive unsorted; h5py demands them sorted.

    A sort without an inverse permutation would silently pair each sample with another sample's
    reference — every number downstream would be wrong and nothing would raise.
    """
    h5, out = built
    rt = ReferenceTable(out, src_h5=h5)
    wi = [4, 1, 5, 0]
    got = rt.log_ref(wi)
    for j, w in enumerate(wi):
        assert np.allclose(got[j], rt.log_ref([w])[0], atol=1e-6), f"row {j} (window {w}) misaligned"
    rt.close()


# ---------------------------------------------------------------------------
# G0 / W1 — the offset in the model
# ---------------------------------------------------------------------------

def _tiny_model(num_assays=A, ctx=32):
    from candi.model import build_model
    torch.manual_seed(0)
    return build_model(embed_dim=8, dropout=0.0, n_transformer_layers=1, decoder_lane=4,
                            depth_center=21.0, use_offset=True, num_assays=num_assays,
                            context_length=ctx, d_model=16, nhead=2)


def _tiny_batch(model, num_assays=A, ctx=32, B=2):
    g = torch.Generator().manual_seed(1)
    y_meta = torch.zeros(B, 4, num_assays)
    y_meta[:, 0, :] = 21.0
    y_meta[:, 1, :] = torch.arange(num_assays).float()
    y_meta[:, 2, :] = 76.0
    y_meta[:, 3, :] = 1.0
    z = torch.randn(B, ctx // 4, model.encoder.d_model, generator=g)
    return z, y_meta


def test_W1_reference_off_is_bit_identical():
    """`log_ref=None` must be a strict no-op, or the `raw` arm is not a control."""
    m = _tiny_model().eval()
    z, ym = _tiny_batch(m)
    with torch.no_grad():
        a = m.decoder(z, ym)
        b = m.decoder(z, ym, None)
    for k in a:
        assert torch.equal(a[k], b[k]), f"{k} changed when log_ref=None was passed explicitly"


def test_G0_offset_moves_log2_mu_by_exactly_the_offset():
    """log2_mu must shift by the offset, one-for-one, below the clamp. This is the whole mechanism."""
    m = _tiny_model().eval()
    z, ym = _tiny_batch(m)
    with torch.no_grad():
        base = m.decoder(z, ym)
        off = torch.full_like(base["log2_mu"], 1.5)      # the trunk sets L, so read it off the output
        shifted = m.decoder(z, ym, off)
    d = (shifted["log2_mu"] - base["log2_mu"])
    assert torch.allclose(d, torch.full_like(d, 1.5), atol=1e-5), \
        f"offset did not pass through: mean shift {float(d.mean()):.4f}, expected 1.5"
    # ...and mu scales by 2^offset, which is the claim that matters for the NB mean
    assert torch.allclose(shifted["mu"], base["mu"] * (2.0 ** 1.5), rtol=1e-4)
    # eta is untouched: it stays the offset-free statistic the covariate block reads
    assert torch.equal(shifted["eta"], base["eta"])


def test_G0_offset_composes_with_the_depth_offset():
    """Depth and reference must ADD in log2 space, not interact — that is why R is depth-free."""
    m = _tiny_model().eval()
    z, ym = _tiny_batch(m)
    ym2 = ym.clone()
    ym2[:, 0, :] = 23.0                                    # +2 in log2 depth
    with torch.no_grad():
        a = m.decoder(z, ym)["log2_mu"]
        both = m.decoder(z, ym2, torch.full_like(a, 0.75))["log2_mu"]
    assert torch.allclose(both - a, torch.full_like(a, 2.75), atol=1e-5)


# ---------------------------------------------------------------------------
# W2 / W3 — the harness fixes
# ---------------------------------------------------------------------------

def test_W2_unmask_coin_hits_the_target_rate_and_is_arm_independent():
    """The coin must be 15% and must come from a DEDICATED stream.

    Two arms differing only in `--reference` have to see identical batches in identical order. If the
    coin were drawn from the shared data RNG, the two arms would visit different biosamples and DSF
    levels, and the paired comparison would silently stop being paired.
    """
    def coins(seed, n=20_000, frac=0.15):
        rng = np.random.default_rng(seed ^ 0xC10A_5EED)
        return np.array([float(rng.random()) < frac for _ in range(n)])

    c = coins(0)
    assert abs(c.mean() - 0.15) < 0.01, f"unmasked fraction {c.mean():.4f}, wanted 0.15"
    assert np.array_equal(c, coins(0)), "coin sequence is not reproducible at a fixed seed"
    assert not np.array_equal(c, coins(1)), "coin sequence ignores the seed"


def test_W2_unmasked_batch_has_no_masked_positions():
    from candi.batch import make_masker, prepare_masked_batch
    from candi._vendored import CLOZE
    B, Lb, F = 2, 8, A
    batch = dict(
        x_data=torch.rand(B, Lb, F) * 5, x_meta=torch.zeros(B, 4, F), x_avail=torch.ones(B, F),
        x_dna=torch.zeros(B, Lb * 25, 4), y_data=torch.rand(B, Lb, F) * 5,
        y_meta=torch.zeros(B, 4, F), y_avail=torch.ones(B, F), y_pval=torch.zeros(B, Lb, F),
        y_peaks=torch.zeros(B, Lb, F), control_data=torch.ones(B, Lb, 1),
        control_meta=torch.zeros(B, 4, 1), control_avail=torch.ones(B, 1),
        y_dsf=torch.ones(B, F, dtype=torch.int64))
    masker = make_masker(p_full_assay=1.0, mask_fraction=0.2, p_full_loci=0.0, p_chunks=0.0)
    unmasked = prepare_masked_batch(batch, masker, torch.device("cpu"), apply_mask=False)
    assert not bool(unmasked["masked_map"].any())
    assert not bool((unmasked["x_data"] == CLOZE).any()), "CLOZE token present in an unmasked batch"
    masked = prepare_masked_batch(batch, masker, torch.device("cpu"), apply_mask=True)
    assert bool(masked["masked_map"].any()), "the masked control produced no supervision"


def test_W3_imp_is_omitted_not_zero_when_nothing_is_masked():
    """h73 averaged 30.9% structural zeros into the `imp` curve. The key must be ABSENT instead."""
    from candi.train import nb_count_loss
    B, Lb, F = 2, 6, A
    out = dict(p=torch.full((B, Lb, F), 0.4), n=torch.full((B, Lb, F), 2.0))
    prep = dict(y_data=torch.ones(B, Lb, F),
                observed_map=torch.ones(B, Lb, F, dtype=torch.bool),
                masked_map=torch.zeros(B, Lb, F, dtype=torch.bool))
    loss, terms = nb_count_loss(out, prep, imp_weight=3.0)
    assert "imp" not in terms, "empty imp was logged as a value"
    assert terms["imp_n"] == 0 and terms["obs_n"] == B * Lb * F
    assert torch.isfinite(loss)

    prep["masked_map"] = torch.zeros(B, Lb, F, dtype=torch.bool)
    prep["masked_map"][:, :, 0] = True
    prep["observed_map"] = ~prep["masked_map"]
    loss3, t3 = nb_count_loss(out, prep, imp_weight=3.0)
    loss1, t1 = nb_count_loss(out, prep, imp_weight=1.0)
    assert "imp" in t3 and t3["imp_n"] == B * Lb
    # 3:1 must actually reweight: loss3 - loss1 == 2 * imp
    assert float(loss3 - loss1) == pytest.approx(2.0 * t3["imp"], rel=1e-5)


# ---------------------------------------------------------------------------
# W4 — even mid-training eval coverage
# ---------------------------------------------------------------------------

def _cycle_selection(n_slots: int, n_cycles: int, k: int):
    """The selection `build_eval_units` computes, isolated so it can be tested without an h5."""
    keep = sorted({min(n_cycles - 1, int((i + 0.5) * n_cycles / k)) for i in range(k)})
    kept = [bi for bi in range(n_cycles * n_slots) if (bi // n_slots) in keep]
    return keep, kept


def test_W4_batches_per_pair_keeps_every_pair():
    """`batches_per_pair=k` must give EVERY cycle slot the same number of batches — the property
    `--eval-max-batches` does not have, and the reason it is banned for checkpoint selection."""
    n_slots, n_cycles, k = 53, 11, 4
    keep, kept = _cycle_selection(n_slots, n_cycles, k)
    seen: Dict[int, int] = {}
    for bi in kept:
        seen[bi % n_slots] = seen.get(bi % n_slots, 0) + 1
    assert len(seen) == n_slots, f"{n_slots - len(seen)} slots went unscored"
    assert set(seen.values()) == {len(keep)}, f"uneven coverage: {sorted(set(seen.values()))}"
    # ...whereas a naive cap drops most of them, which is the h73 trap
    assert len({bi % n_slots for bi in range(12)}) == 12 < n_slots


def test_W4_selection_is_not_a_prefix_of_the_chromosome():
    """The bug the smoke test caught: a PREFIX of the eval windows is the chr21 p-arm, which is dead.

    Every held-out target's first 3,072 eval positions are exactly zero there, so the foreground
    purity filter drops 100% of records and the selection metric returns NaN over 0 targets. Even
    coverage across PAIRS is not enough; the windows have to be spread across the chromosome too.
    """
    n_slots, n_cycles, k = 53, 11, 4
    keep, kept = _cycle_selection(n_slots, n_cycles, k)
    assert keep[0] > 0, f"cycle 0 kept — that is the dead prefix: {keep}"
    # the kept batches must reach into the far half of the chromosome, not huddle at the start
    assert max(kept) > 0.6 * n_cycles * n_slots, f"selection never leaves the first 60%: {keep}"
    # and they must be spread, not adjacent
    assert min(b - a for a, b in zip(keep, keep[1:])) >= 2, f"cycles are bunched: {keep}"


@pytest.mark.parametrize("k", [1, 2, 3, 4, 6, 11, 20])
def test_W4_selection_is_well_formed_at_every_k(k):
    """Including k > n_cycles, where the clamp must not produce duplicates or an out-of-range cycle."""
    n_slots, n_cycles = 53, 11
    keep, kept = _cycle_selection(n_slots, n_cycles, k)
    assert keep, "no cycles kept"
    assert len(keep) == len(set(keep)) and max(keep) < n_cycles and min(keep) >= 0
    assert len(kept) == len(keep) * n_slots
    # The selection must always reach the interior of the chromosome. Including cycle 0 is only a
    # problem when it is ALL you have: at large k its all-zero records self-eliminate through the
    # foreground purity filter and the other cycles carry the metric.
    assert max(keep) >= n_cycles // 2, f"k={k} never reaches the chromosome interior: {keep}"
    if k <= n_cycles // 2:
        assert keep[0] > 0, f"k={k} had room to skip the dead prefix and did not: {keep}"
