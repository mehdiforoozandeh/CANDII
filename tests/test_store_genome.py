"""The genome layer's contract, checked against a synthetic FASTA and a synthetic blacklist.

No corpus and no hg38 are reachable from a laptop, so every test builds its own FASTA and BED in
`tmp_path` and reads `dna.h5` / `mask.h5` back with **plain h5py** — never through `GenomeLayer` —
because a round-trip through one module's own reader proves nothing about what is on disk.

Covers `STORE_PLAN.md` §7 item 4 (no eligible window is below `min_valid_frac`; none is all-N)
plus the rules whose failure modes are silent: the case/`N` folding and the IUPAC decision (D10),
blacklist overlap at an interval boundary (D11), the `floor` bin count (D13), the eligibility
primitive against a brute-force loop (D12), and the `fasta_sha256` mismatch being loud.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from candi.store import genome as G
from candi.store import layout as L
from candi.store.layout import StoreError

RES = 25


# ---------------------------------------------------------------------------------------------
# synthetic inputs
# ---------------------------------------------------------------------------------------------


def write_fasta(path: Path, seqs: dict, line_width: int = 60) -> Path:
    """A FASTA with one record per `seqs` entry, wrapped at `line_width`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as fh:
        for name, seq in seqs.items():
            fh.write(f">{name} synthetic\n")
            for i in range(0, len(seq), line_width):
                fh.write(seq[i : i + line_width] + "\n")
    return path


def write_bed(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{c}\t{s}\t{e}\tSynthetic Region\n" for c, s, e in rows), encoding="utf-8"
    )
    return path


def random_seq(rng: np.random.Generator, n: int) -> str:
    return "".join(rng.choice(list("ACGT"), size=n))


@pytest.fixture()
def tiny_genome(tmp_path):
    """A two-chromosome genome with N runs, soft masking and a blacklist that straddles bins."""
    rng = np.random.default_rng(0)
    # chr1: 400 bins worth + 7 leftover bp (D13 truncation), an N run, a soft-masked stretch.
    seq1 = list(random_seq(rng, 400 * RES + 7))
    seq1[1000:1500] = list("N" * 500)          # bins 40..59 all-N
    seq1[2000:2003] = list("nnn")              # bin 80 has N, the rest of it is real
    seq1[3000:3200] = [c.lower() for c in seq1[3000:3200]]   # soft-masked, still valid
    # chr2: shorter, one N at the very last bp of bin 5
    seq2 = list(random_seq(rng, 120 * RES))
    seq2[5 * RES + RES - 1] = "N"
    seqs = {"chr1": "".join(seq1), "chr2": "".join(seq2)}
    fasta = write_fasta(tmp_path / "tiny.fa", seqs)
    sizes = {c: len(s) for c, s in seqs.items()}
    (tmp_path / "genome").mkdir(exist_ok=True)
    (tmp_path / "genome" / "chrom_sizes.json").write_text(json.dumps(sizes), encoding="utf-8")
    # one interval that overlaps bin 100 by a single bp and bin 101 fully
    bed = write_bed(
        tmp_path / "bl.bed",
        [("chr2", 10 * RES, 12 * RES), ("chr1", 100 * RES + RES - 1, 102 * RES)],
    )
    return {"root": tmp_path, "fasta": fasta, "bed": bed, "sizes": sizes, "seqs": seqs}


def build(tiny, **kw):
    return G.build_genome(
        tiny["root"], tiny["fasta"], tiny["bed"], chrom_sizes=tiny["sizes"], **kw
    )


# ---------------------------------------------------------------------------------------------
# D10 — dna.h5
# ---------------------------------------------------------------------------------------------


def test_dna_codes_case_and_n_folding(tmp_path):
    seq = "acgtACGTnN"
    fasta = write_fasta(tmp_path / "x.fa", {"chr1": seq}, line_width=3)
    out = tmp_path / "dna.h5"
    G.build_dna(fasta, out, {"chr1": len(seq)})
    with h5py.File(out, "r") as f:
        got = f["chr1"][:]
        assert f.attrs["build"] == "GRCh38"
        assert json.loads(f.attrs["codes"]) == {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    # soft-masked acgt are real bases; n and N are both the N code
    assert got.tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 4, 4]
    assert got.dtype == np.uint8


def test_iupac_folds_to_n_and_is_counted(tmp_path):
    seq = "ACGTRYSWKMBDHVacgtry"
    fasta = write_fasta(tmp_path / "x.fa", {"chr1": seq})
    out = tmp_path / "dna.h5"
    summary = G.build_dna(fasta, out, {"chr1": len(seq)})
    with h5py.File(out, "r") as f:
        got = f["chr1"][:]
        counts = json.loads(f.attrs["iupac_counts"])
    assert got[4:14].tolist() == [G.N_CODE] * 10           # RYSWKMBDHV -> N
    assert got[18:20].tolist() == [G.N_CODE, G.N_CODE]     # lowercase r, y -> N
    assert counts["R"] == 2 and counts["Y"] == 2 and counts["S"] == 1
    assert summary["per_chrom"]["chr1"]["n_ambiguous_folded"] == 12
    assert summary["per_chrom"]["chr1"]["n_softmasked"] == 4


def test_dna_length_disagreement_is_loud(tmp_path):
    fasta = write_fasta(tmp_path / "x.fa", {"chr1": "ACGT" * 10})
    with pytest.raises(StoreError, match="different builds"):
        G.build_dna(fasta, tmp_path / "dna.h5", {"chr1": 41})


def test_dna_refuses_a_wrong_fasta_sha256(tmp_path):
    fasta = write_fasta(tmp_path / "x.fa", {"chr1": "ACGT" * 10})
    with pytest.raises(StoreError, match="fasta_sha256 mismatch"):
        G.build_dna(fasta, tmp_path / "dna.h5", {"chr1": 40}, fasta_sha256="deadbeef")
    # the right one is accepted, and lands in the attrs verbatim
    sha = G.sha256_file(fasta)
    G.build_dna(fasta, tmp_path / "dna.h5", {"chr1": 40}, fasta_sha256=sha)
    with h5py.File(tmp_path / "dna.h5", "r") as f:
        assert f.attrs["fasta_sha256"] == sha


def test_dna_refuses_to_overwrite(tiny_genome):
    build(tiny_genome, what="dna")
    with pytest.raises(StoreError, match="already exists"):
        build(tiny_genome, what="dna")
    build(tiny_genome, what="dna", overwrite=True)


# ---------------------------------------------------------------------------------------------
# D11 / D13 — mask.h5
# ---------------------------------------------------------------------------------------------


def test_mask_bin_count_is_floor(tiny_genome):
    build(tiny_genome)
    with h5py.File(L.mask_path(tiny_genome["root"]), "r") as f:
        for chrom, chrom_len in tiny_genome["sizes"].items():
            assert f[chrom].shape == (chrom_len // RES,)
            assert f[chrom].dtype == np.uint8
        assert json.loads(f.attrs["n_bins"]) == {
            c: n // RES for c, n in tiny_genome["sizes"].items()
        }
    # chr1 is 400 bins + 7 bp; the partial bin is dropped, not rounded up
    assert L.n_bins_for(tiny_genome["sizes"]["chr1"], RES) == 400


def test_mask_invalidates_any_n_in_a_bin(tiny_genome):
    build(tiny_genome)
    with h5py.File(L.mask_path(tiny_genome["root"]), "r") as f:
        m1 = f["chr1"][:]
        m2 = f["chr2"][:]
    assert m1[40:60].sum() == 0            # the 500 bp N run
    assert m1[80] == 0                     # three N in an otherwise real bin
    assert m1[120:126].all() == True       # soft-masked acgt stay valid
    assert m2[5] == 0                      # a single N in the last bp of the bin
    assert m2[4] == 1 and m2[6] == 1


def test_blacklist_overlap_of_one_bp_invalidates_the_bin(tiny_genome):
    build(tiny_genome)
    with h5py.File(L.mask_path(tiny_genome["root"]), "r") as f:
        m1 = f["chr1"][:]
        m2 = f["chr2"][:]
        attrs = dict(f.attrs)
    # chr1 interval is [100*25+24, 102*25): bin 100 overlaps by exactly one bp
    assert m1[99] == 1
    assert m1[100] == 0
    assert m1[101] == 0
    assert m1[102] == 1
    # chr2 interval [10*25, 12*25) is exactly bins 10 and 11 — a half-open BED end is exclusive
    assert m2[9] == 1 and m2[10] == 0 and m2[11] == 0 and m2[12] == 1
    assert attrs["blacklist_sha256"] == G.sha256_file(tiny_genome["bed"])
    assert "blacklist" in str(attrs["blacklist_source"])
    assert "no N base" in str(attrs["rule"])


def test_blacklist_reader_groups_and_merges_regardless_of_order():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        # lexicographic, interleaved, overlapping and touching — none of it may be relied on
        bed = write_bed(
            Path(td) / "b.bed",
            [("chr10", 500, 600), ("chr1", 100, 200), ("chr10", 550, 700),
             ("chr1", 200, 300), ("chr2", 0, 10), ("chr1", 50, 120)],
        )
        bl = G.read_blacklist(bed)
    assert bl["chr1"].tolist() == [[50, 300]]        # 50-120, 100-200 and 200-300 all merge
    assert bl["chr10"].tolist() == [[500, 700]]
    assert bl["chr2"].tolist() == [[0, 10]]


def test_blacklist_bin_flags_matches_a_brute_force_loop():
    rng = np.random.default_rng(7)
    n_bins = 500
    starts = np.sort(rng.integers(0, n_bins * RES, size=20))
    iv = np.stack([starts, starts + rng.integers(1, 300, size=20)], axis=1)
    iv = np.asarray([[s, e] for s, e in iv.tolist()], dtype=np.int64)
    got = G.blacklist_bin_flags(iv, n_bins, RES)
    want = np.zeros(n_bins, dtype=bool)
    for s, e in iv.tolist():
        for b in range(n_bins):
            if b * RES < e and (b + 1) * RES > s:
                want[b] = True
    assert got.tolist() == want.tolist()


def test_mask_summary_splits_n_blacklist_and_both(tiny_genome):
    res = build(tiny_genome)["mask"]
    c1 = res["per_chrom"]["chr1"]
    assert c1["n_bins"] == 400
    assert c1["invalid_n"] == 21                      # bins 40..59 plus bin 80
    assert c1["invalid_blacklist"] == 2               # bins 100, 101
    assert c1["invalid_both"] == 0
    assert c1["invalid_total"] == 23
    assert c1["n_valid"] == 400 - 23
    assert res["blacklist_intervals"] == 2
    assert 0.0 < res["valid_frac_genome"] < 1.0


# ---------------------------------------------------------------------------------------------
# D12 — window eligibility  (STORE_PLAN.md §7 test 4)
# ---------------------------------------------------------------------------------------------


def _brute_force_starts(mask, ctx, frac):
    return [
        s for s in range(len(mask) - ctx + 1) if mask[s : s + ctx].mean() >= frac
    ]


@pytest.mark.parametrize("ctx", [1, 3, 8, 64])
@pytest.mark.parametrize("frac", [0.0, 0.5, 0.9, 1.0])
def test_eligibility_matches_brute_force(ctx, frac):
    rng = np.random.default_rng(11)
    mask = (rng.random(600) > 0.15).astype(np.uint8)
    got = G.eligible_starts(mask, ctx, frac, stride=1).tolist()
    assert got == _brute_force_starts(mask, ctx, frac)
    assert G.count_eligible(mask, ctx, frac, stride=1) == len(got)


def test_no_eligible_window_is_below_the_threshold_or_all_n(tiny_genome):
    """`STORE_PLAN.md` §7 test 4, on a real built mask."""
    build(tiny_genome)
    with h5py.File(L.mask_path(tiny_genome["root"]), "r") as f:
        mask = f["chr1"][:]
    with h5py.File(L.dna_path(tiny_genome["root"]), "r") as f:
        dna = f["chr1"][:]
    for ctx in (8, 32, 128):
        for frac in (0.9, 1.0):
            for s in G.eligible_starts(mask, ctx, frac, stride=1):
                w = mask[s : s + ctx]
                assert w.mean() >= frac - 1e-9
                bases = dna[s * RES : (s + ctx) * RES]
                assert not np.all(bases == G.N_CODE)


def test_tiled_and_every_start_counts_differ_and_agree_with_slicing():
    rng = np.random.default_rng(3)
    mask = (rng.random(10_000) > 0.05).astype(np.uint8)
    ctx = 128
    every = G.eligible_starts(mask, ctx, 0.9, stride=1)
    tiled = G.eligible_starts(mask, ctx, 0.9)                # stride defaults to ctx
    assert set(tiled.tolist()) <= set(every.tolist())
    assert tiled.tolist() == [s for s in every.tolist() if s % ctx == 0]
    assert G.count_eligible(mask, ctx, 0.9) == len(tiled)
    assert G.count_eligible(mask, ctx, 0.9, stride=1) == len(every)
    offset = G.eligible_starts(mask, ctx, 0.9, stride=ctx, offset=17)
    assert all((s - 17) % ctx == 0 for s in offset.tolist())


def test_window_longer_than_the_chromosome_yields_nothing():
    mask = np.ones(10, dtype=np.uint8)
    assert G.window_valid_counts(mask, 11).size == 0
    assert G.eligible_starts(mask, 11).size == 0
    assert G.count_eligible(mask, 11) == 0
    with pytest.raises(StoreError):
        G.window_valid_counts(mask, 0)


def test_threshold_landing_on_an_integer_is_inclusive():
    mask = np.ones(100, dtype=np.uint8)
    mask[:10] = 0                       # exactly 90/100 valid
    assert G.eligible_window_mask(mask, 100, 0.9).tolist() == [True]
    mask[10] = 0                        # 89/100
    assert G.eligible_window_mask(mask, 100, 0.9).tolist() == [False]


# ---------------------------------------------------------------------------------------------
# the reader, and where the sha256 check lives
# ---------------------------------------------------------------------------------------------


def test_genome_layer_reads_dna_and_mask(tiny_genome):
    build(tiny_genome)
    with G.GenomeLayer(L.genome_dir(tiny_genome["root"])) as g:
        assert g.build == "GRCh38"
        assert g.chroms == ["chr1", "chr2"]
        assert g.n_bins["chr1"] == 400
        assert g.resolution == RES
        seq = g.dna("chr1", 1000, 1500)
        assert seq.tolist() == [G.N_CODE] * 500
        assert g.dna_for_bins("chr1", 40, 60).tolist() == [G.N_CODE] * 500
        assert g.mask("chr1")[40] == 0
        assert g.is_eligible("chr1", 0, 8) is True
        assert g.is_eligible("chr1", 40, 8) is False
        assert g.count_eligible("chr1", 8, 0.9, stride=1) > 0
        with pytest.raises(StoreError, match="outside"):
            g.dna("chr1", 0, 10**9)
        with pytest.raises(StoreError, match="no DNA"):
            g.dna("chrZ")


def test_a_fasta_sha256_mismatch_is_loud_at_open(tiny_genome):
    build(tiny_genome)
    gdir = L.genome_dir(tiny_genome["root"])
    real = G.sha256_file(tiny_genome["fasta"])
    G.GenomeLayer(gdir, fasta_sha256=real).close()          # the matching one opens
    with pytest.raises(StoreError, match="different builds"):
        G.GenomeLayer(gdir, fasta_sha256="0" * 64)
    with pytest.raises(StoreError, match="different blacklist"):
        G.GenomeLayer(gdir, blacklist_sha256="0" * 64)
    assert G.verify_genome(gdir, fasta_sha256="0" * 64)     # same check, as a problem list
    assert G.verify_genome(gdir, chrom_sizes=tiny_genome["sizes"]) == []


def test_a_mask_built_against_another_genome_is_caught(tiny_genome):
    build(tiny_genome)
    with h5py.File(L.mask_path(tiny_genome["root"]), "r+") as f:
        f.attrs["fasta_sha256"] = "1" * 64
    with pytest.raises(StoreError, match="different builds"):
        G.GenomeLayer(L.genome_dir(tiny_genome["root"]))


def test_genome_report_carries_the_numbers_t7_owes(tiny_genome):
    build(tiny_genome)
    rep = G.genome_report(L.genome_dir(tiny_genome["root"]), context_bins=(8, 32))
    assert rep["min_valid_frac"] == 0.9
    assert set(rep["per_chrom"]) == {"chr1", "chr2"}
    c1 = rep["per_chrom"]["chr1"]
    assert c1["n_bins"] == 400 and c1["n_valid"] == 377
    assert c1["eligible"]["8"]["tiled"] <= c1["eligible"]["8"]["every_start"]
    assert rep["genome"]["n_bins"] == 400 + 120
    assert rep["genome"]["valid_frac"] == rep["genome"]["n_valid"] / rep["genome"]["n_bins"]
    assert rep["fasta_sha256"] == G.sha256_file(tiny_genome["fasta"])


# ---------------------------------------------------------------------------------------------
# chrom_sizes.json, in both shapes
# ---------------------------------------------------------------------------------------------


def test_load_genome_chrom_sizes_accepts_the_flat_and_the_wrapped_form(tmp_path):
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({"chr1": 100, "chr2": 50}), encoding="utf-8")
    assert G.load_genome_chrom_sizes(flat) == {"chr1": 100, "chr2": 50}
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(
        json.dumps({"schema": 1, "build": "GRCh38", "resolution": 25,
                    "chrom_sizes": {"chr1": 100, "chr2": 50},
                    "n_bins": {"chr1": 4, "chr2": 2}}),
        encoding="utf-8",
    )
    assert G.load_genome_chrom_sizes(wrapped) == {"chr1": 100, "chr2": 50}
    tsv = tmp_path / "hg38.chrom.sizes"
    tsv.write_text("chr1\t100\nchr2\t50\n", encoding="utf-8")
    assert G.load_genome_chrom_sizes(tsv) == {"chr1": 100, "chr2": 50}


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def test_cli_build_genome_writes_both_files_and_a_report(tiny_genome, capsys):
    from candi.store.cli import main

    report = tiny_genome["root"] / "report.json"
    rc = main([
        "build-genome",
        "--store-root", str(tiny_genome["root"]),
        "--fasta", str(tiny_genome["fasta"]),
        "--blacklist", str(tiny_genome["bed"]),
        "--context-bins", "8,32",
        "--report", str(report),
    ])
    assert rc == 0
    assert L.dna_path(tiny_genome["root"]).is_file()
    assert L.mask_path(tiny_genome["root"]).is_file()
    rep = json.loads(report.read_text(encoding="utf-8"))
    assert rep["per_chrom"]["chr1"]["n_bins"] == 400
    assert "GENOME eligible L=8" in capsys.readouterr().out
