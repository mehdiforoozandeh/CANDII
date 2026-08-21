"""t18 — the pinned E-block scoring assets are the organizers' exact bytes, and stay that way.

These are checksum tests, which usually earn their keep only when someone edits a file by accident.
Here they earn it for a sharper reason: swapping `gencode.v29` for a newer GENCODE, or the 38-region
stub for the real ENCODE Exclusion list, is a *tempting improvement* that silently destroys the one
property the E-block exists to have — comparability to the published EIC table. The failure would
be invisible in every other test. So it is asserted here, loudly, with the reason attached.
"""
from __future__ import annotations

import gzip

import numpy as np
import pytest

from candi.bench import annotations as A


def test_every_pinned_asset_matches_its_checksum() -> None:
    got = A.verify_assets()
    assert set(got) == set(A.ASSET_SHA256)
    for name, digest in got.items():
        assert digest == A.ASSET_SHA256[name], (
            f"{name} is no longer the bytes the challenge scored with. Restore the file; "
            f"updating the checksum makes every E-block number uncomparable to the published table."
        )


def test_a_missing_asset_names_the_reason_not_just_the_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(A, "ASSET_DIR", tmp_path)
    with pytest.raises(A.AssetError, match="do not re-download|Do not re-download"):
        A.verify_assets(["gencode.v29.genes.gtf.bed.gz"])


def test_a_changed_asset_refuses_rather_than_re_pinning(tmp_path, monkeypatch) -> None:
    fake = tmp_path / "gencode.v29.genes.gtf.bed.gz"
    fake.write_bytes(b"not the real bed")
    monkeypatch.setattr(A, "ASSET_DIR", tmp_path)
    with pytest.raises(A.AssetError, match="do not update the checksum"):
        A.verify_assets(["gencode.v29.genes.gtf.bed.gz"])


@pytest.mark.parametrize("loader,name,n_lines,n_cols", [
    (A.gene_annotations, "gencode.v29.genes.gtf.bed.gz", 58_721, 6),
    (A.enhancer_annotations, "F5.hg38.enhancers.bed.gz", 63_285, 12),
    (A.eic_blacklist_lines, "hg38.blacklist.bed.gz", 38, 3),
])
def test_bed_shape_is_what_the_metrics_unpack(loader, name, n_lines, n_cols) -> None:
    """The metrics unpack by fixed arity, so a column-count drift raises inside the metric."""
    lines = loader()
    assert len(lines) == n_lines
    assert all(len(ln.split()) == n_cols for ln in lines[:1000])
    assert A.ASSET_LINES[name] == n_lines
    assert A.ASSET_COLUMNS[name] == n_cols


def test_gene_bed_is_parseable_the_way_mseprom_parses_it() -> None:
    """`chrom_, start, end, _, _, strand = line.split()` must hold for every line, not just the head."""
    for line in A.gene_annotations():
        chrom_, start, end, _, _, strand = line.split()
        assert chrom_.startswith("chr")
        assert int(start) >= 0 and int(end) >= int(start)
        assert strand in ("+", "-")


def test_enhancer_bed_is_parseable_the_way_mseenh_parses_it() -> None:
    """A 12-way unpack that discards nine fields. Arity is the only thing that matters."""
    for line in A.enhancer_annotations():
        chrom_, start, end, _, _, _, _, _, _, _, _, _ = line.split()
        assert chrom_.startswith("chr")
        assert int(end) >= int(start)


def test_the_shipped_blacklist_is_a_stub_not_the_encode_exclusion_list() -> None:
    """The finding that decides what the E-block does with a blacklist: nothing.

    17,040 bp over 9 chromosomes, against the real v2 list's 227,162,400 bp over 24. Asserted so
    that if someone ever swaps in the real list, this test says why that is wrong before any score
    moves. The other half of the argument — that the npy scoring path returns before the blacklist
    branch is reached at all — is in assets/PROVENANCE.md and cannot be asserted from here.
    """
    lines = A.eic_blacklist_lines()
    chroms = {ln.split()[0] for ln in lines}
    covered = sum(int(ln.split()[2]) - int(ln.split()[1]) for ln in lines)
    assert len(lines) == 38
    assert len(chroms) == 9
    assert covered == 17_040, (
        f"the shipped blacklist covers {covered} bp; it covered 17,040 when pinned. The real "
        f"ENCODE v2 list (cruxvault/results/t4) covers 227,162,400 bp over 24 chromosomes."
    )


def test_real_v2_blacklist_is_the_one_that_is_three_orders_larger() -> None:
    """Guards the comparison above by checking the other side of it is still what we think."""
    from pathlib import Path
    v2 = Path(__file__).resolve().parent.parent / "cruxvault/results/t4/hg38-blacklist.v2.bed"
    if not v2.exists():                       # raw/results are not always present locally
        pytest.skip("t4 blacklist not present in this checkout")
    rows = [ln.split("\t") for ln in v2.read_text().splitlines() if ln.strip()]
    covered = sum(int(r[2]) - int(r[1]) for r in rows)
    assert len(rows) == 636
    assert covered == 227_162_400
    assert covered / 17_040 > 10_000


def test_load_bed_lines_matches_the_reference_loader_on_both_paths(tmp_path) -> None:
    """`load_bed` reads gz in binary and decodes ascii; plain files are read as text."""
    payload = "chr1\t100\t200\n" "chr2\t300\t400\n"
    plain = tmp_path / "a.bed"
    plain.write_text(payload)
    gz = tmp_path / "a.bed.gz"
    with gzip.open(gz, "wb") as fh:
        fh.write(payload.encode("ascii"))
    assert A.load_bed_lines(plain) == A.load_bed_lines(gz) == ["chr1\t100\t200\n", "chr2\t300\t400\n"]


def test_prom_loc_and_window_size_are_the_challenge_defaults() -> None:
    """80 bins x 25 bp = the 2 kb promoter the paper describes. Both are score.py's defaults."""
    assert A.PROM_LOC == 80
    assert A.WINDOW_SIZE == 25
    assert A.PROM_LOC * A.WINDOW_SIZE == 2000


def test_variance_pool_rejects_the_shapes_that_would_break_msevar() -> None:
    import numpy as np
    ok = dict(corpus="eic", assay="H3K4me3", chrom="chr21", space="pval",
              biosamples=("a", "b"), values=np.ones(4, dtype=np.float32))
    assert A.VariancePool(**ok).n_biosamples == 2
    with pytest.raises(ValueError, match="space must be"):
        A.VariancePool(**{**ok, "space": "arcsinh"})
    with pytest.raises(ValueError, match="1-D"):
        A.VariancePool(**{**ok, "values": np.ones((2, 2), dtype=np.float32)})
    with pytest.raises(ValueError, match="non-finite"):
        A.VariancePool(**{**ok, "values": np.array([1.0, np.nan], dtype=np.float32)})
    with pytest.raises(ValueError, match="negative"):
        A.VariancePool(**{**ok, "values": np.array([-1.0, 1.0], dtype=np.float32)})


def test_a_missing_variance_pool_raises_rather_than_scoring_zero(tmp_path) -> None:
    """`score_metrics.py::msevar` returns a bare 0.0 with no variance vector. We refuse instead.

    Their 0.0 is a sentinel that is indistinguishable from a perfect score in any table it lands in.
    """
    with pytest.raises(A.AssetError, match="silent 0.0"):
        A.load_variance_pool(tmp_path, "eic", "H3K4me3", "chr21")


# ---------------------------------------------------------------------------------------------
# the builder (D7) — against a real store written by candi.store.writer
# ---------------------------------------------------------------------------------------------

def _pool_store(tmp_path):
    """The t8 synthetic store. Two biosamples, so a two-member pool is the largest possible."""
    from tests.test_store_reader import make_store
    return make_store(tmp_path)


def test_the_builder_produces_a_pool_the_loader_can_read(tmp_path) -> None:
    """The end-to-end shape. This test exists because the builder had never once been run.

    Every reader accessor takes `start` as a REQUIRED argument, and the builder called
    `pval(chrom)`, so the first real invocation would have died on a `TypeError` after the job had
    already queued, loaded a 435 GB store and burned its allocation. A builder with no test is a
    builder that does not work.
    """
    corpus_root = _pool_store(tmp_path / "s")
    out = tmp_path / "varpool"
    meta = A.build_variance_pools(corpus_root, out, corpus="eic", chroms=["chr1", "chr2"],
                                  train_biosamples=["T_aa", "V_aa"], space="pval")
    assert meta.exists()

    pool = A.load_variance_pool(out, "eic", "H3K4me3", "chr1")
    assert pool.values.shape == (2_000,)          # CHROM_SIZES chr1 // 25
    assert pool.space == "pval"
    assert pool.n_biosamples == 2
    assert sorted(pool.biosamples) == ["T_aa", "V_aa"]


def test_an_assay_only_one_biosample_carries_is_skipped_not_written_as_zeros(tmp_path) -> None:
    """`ATAC-seq` is in `T_aa` alone. A one-member pool is zero everywhere, and `msevar` divides
    by `var.sum()`, so writing it would turn a missing weight into a silent `0/0`."""
    corpus_root = _pool_store(tmp_path / "s")
    out = tmp_path / "varpool"
    A.build_variance_pools(corpus_root, out, corpus="eic", chroms=["chr1"],
                           train_biosamples=["T_aa", "V_aa"], space="pval")
    with pytest.raises(A.AssetError, match="no variance vector for assay"):
        A.load_variance_pool(out, "eic", "ATAC-seq", "chr1")


def test_the_block_size_cannot_move_a_variance(tmp_path) -> None:
    """The claim the blocked read rests on, asserted rather than argued.

    Variance is per bin, so a block boundary can only change which read returned a column, never
    the column. `37` is deliberately coprime with the chromosome length, so blocks land unaligned
    and a boundary bug would show. Bit-exact, not `approx`.
    """
    corpus_root = _pool_store(tmp_path / "s")
    whole, chunked = tmp_path / "whole", tmp_path / "chunked"
    for root, bs in ((whole, 10_000), (chunked, 37)):
        A.build_variance_pools(corpus_root, root, corpus="eic", chroms=["chr1"],
                               train_biosamples=["T_aa", "V_aa"], space="pval", block_bins=bs)
    a = A.load_variance_pool(whole, "eic", "H3K4me3", "chr1").values
    b = A.load_variance_pool(chunked, "eic", "H3K4me3", "chr1").values
    assert np.array_equal(a, b)


def test_the_variance_is_the_population_variance_the_organizers_used(tmp_path) -> None:
    """`score_metrics.py` uses `std(y_all, axis=0) ** 2` — ddof=0. At n=2 the sample variance is
    twice that, so getting this wrong doubles every `msevar` and would still look plausible."""
    from candi.store.reader import CorpusStore

    corpus_root = _pool_store(tmp_path / "s")
    out = tmp_path / "varpool"
    A.build_variance_pools(corpus_root, out, corpus="eic", chroms=["chr1"],
                           train_biosamples=["T_aa", "V_aa"], space="pval")
    store = CorpusStore(corpus_root)
    stack = np.stack([np.asarray(store[b]["H3K4me3"].pval("chr1", 0), dtype=np.float64)
                      for b in ("T_aa", "V_aa")], axis=0)
    want = (np.std(stack, axis=0) ** 2).astype(np.float32)
    got = A.load_variance_pool(out, "eic", "H3K4me3", "chr1").values
    assert np.array_equal(got, want)
    assert not np.array_equal(got, (np.var(stack, axis=0, ddof=1)).astype(np.float32))
