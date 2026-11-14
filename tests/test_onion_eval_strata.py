"""E3 region strata: rasterization on hand cases, deciles with ties, the foreground rule, and a tiny
end-to-end build against a synthetic genome layer (no network: the sources are pre-placed)."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import strata as S  # noqa: E402


def bins(*iv, n=10):
    s, e = zip(*iv) if iv else ((), ())
    return S.interval_bins(s, e, n).tolist()


# -- rasterization ---------------------------------------------------------------------------

def test_half_open_end_on_bin_edge_does_not_spill():
    # [0, 25) is exactly bin 0; [25, 50) is exactly bin 1
    assert bins((0, 25)) == [1] + [0] * 9
    assert bins((25, 50)) == [0, 1] + [0] * 8


def test_one_bp_overlap_marks_the_bin():
    assert bins((24, 25)) == [1] + [0] * 9          # last bp of bin 0
    assert bins((24, 26)) == [1, 1] + [0] * 8       # straddles the edge
    assert bins((49, 51)) == [0, 1, 1] + [0] * 7


def test_empty_and_reversed_intervals_mark_nothing():
    assert bins((30, 30)) == [0] * 10
    assert bins((40, 30)) == [0] * 10


def test_clip_to_floor_grid_and_overlaps():
    # n=4 bins = 100 bp; a chromosome of 110 bp: bp 100..109 have no bin
    assert S.interval_bins([90, 100], [120, 110], 4).tolist() == [0, 0, 0, 1]
    assert S.interval_bins([0, 10], [60, 30], 4).tolist() == [1, 1, 1, 0]   # overlapping intervals


def test_gtf_closed_to_half_open():
    # GTF [26, 50] (1-based closed) is bp 25..49 0-based: exactly bin 1
    s, e = S.gtf_to_half_open([26], [50])
    assert S.interval_bins(s, e, 4).tolist() == [0, 1, 0, 0]
    # GTF [1, 26] is bp 0..25: bins 0 and 1
    s, e = S.gtf_to_half_open([1], [26])
    assert S.interval_bins(s, e, 4).tolist() == [1, 1, 0, 0]


def test_gene_body_reader_spans_are_strand_free_tss_to_tes(tmp_path):
    p = tmp_path / "g.gtf"
    p.write_text(
        '##header\n'
        'chr1\tHAVANA\tgene\t1\t500\t.\t+\t.\tgene_type "protein_coding";\n'
        'chr1\tHAVANA\ttranscript\t26\t75\t.\t+\t.\tgene_type "protein_coding"; tag "basic";\n'
        'chr1\tHAVANA\ttranscript\t101\t150\t.\t-\t.\tgene_type "protein_coding"; tag "basic";\n'
        'chr1\tHAVANA\ttranscript\t201\t250\t.\t+\t.\tgene_type "lncRNA"; tag "basic";\n')
    iv, n = S.read_gencode_bodies(p)
    assert n == 2
    m = S.rasterize(iv, {"chr1": 12})["chr1"].tolist()
    # minus-strand transcript (TES at 101, TSS at 150) covers the same bins as its span
    assert m == [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0]


# -- deciles ---------------------------------------------------------------------------------

def test_deciles_distinct_values_are_balanced():
    d = S.deciles(np.arange(100.0)[::-1])
    assert np.bincount(d, minlength=10).tolist() == [10] * 10
    assert d[-1] == 0 and d[0] == 9


def test_deciles_ties_share_one_decile_and_match_scipy():
    from scipy.stats import rankdata
    rng = np.random.default_rng(0)
    x = rng.integers(0, 7, size=1000).astype(float)       # heavy ties
    x[::97] = np.nan
    d = S.deciles(x)
    ok = np.isfinite(x)
    assert (d[~ok] == -1).all()
    ref = np.minimum(((rankdata(x[ok], method="average") - 1) * 10 // ok.sum()).astype(int), 9)
    assert (d[ok] == ref).all()
    for v in np.unique(x[ok]):                             # every tie group in exactly one decile
        assert np.unique(d[x == v]).size == 1


def test_block_deciles_drop_bad_blocks_and_tail():
    n = {"chrA": 2 * S.BLOCK + 5, "chrB": 2 * S.BLOCK}
    m = {"chrA": np.r_[np.full(S.BLOCK, 0.1), np.full(S.BLOCK, 0.9), np.ones(5)],
         "chrB": np.r_[np.full(S.BLOCK, 0.5), np.full(S.BLOCK, 0.2)]}
    bad = {"chrA": np.zeros(n["chrA"], np.uint8), "chrB": np.zeros(n["chrB"], np.uint8)}
    bad["chrB"][S.BLOCK + 3] = 1                           # one bad bin drops the whole block
    dec, info = S.block_deciles(m, bad, n)
    assert info["n_blocks"] == 4 and info["n_dropped_blocks"] == 1
    a, b = dec["chrA"], dec["chrB"]
    assert (a[-5:] == -1).all()                            # tail past the last whole block
    assert (b[S.BLOCK:] == -1).all()
    # three kept blocks 0.1 < 0.5 < 0.9 -> deciles 0, 3, 6
    assert a[0] == 0 and b[0] == 3 and a[S.BLOCK] == 6


# -- foreground rule -------------------------------------------------------------------------

def test_foreground_zero_heavy_track_never_selects_everything():
    a = np.zeros(1000)
    a[:400] = np.arange(1, 401)                            # 60% zeros
    fg = S.foreground_mask(a, 0.02)
    assert fg.sum() == 20 and fg[380:400].all()
    assert S.foreground_mask(np.zeros(1000), 0.02).sum() == 0
    assert S.foreground_mask(a, 0.5).sum() == 400          # zeros are never foreground


def test_foreground_ties_at_cut_are_excluded_and_nan_ignored():
    a = np.array([5, 5, 5, 1, 0, np.nan, 2, 3])
    fg = S.foreground_mask(a, 0.3)                         # k = floor(0.3 * 7) = 2; cut = 5
    assert fg.sum() == 0
    fg = S.foreground_mask(a, 0.5)                         # k = 3; cut = 3 -> the three 5s
    assert fg.tolist() == [True, True, True, False, False, False, False, False]
    bg = S.background_mask(a, 0.5)
    assert bg.tolist() == [False, False, False, True, True, False, True, True]
    with pytest.raises(ValueError):
        S.foreground_mask(a, 0.0)


# -- end-to-end build on a synthetic genome layer ---------------------------------------------

def test_build_and_list_on_synthetic_sources(tmp_path, monkeypatch, capsys):
    h5py = pytest.importorskip("h5py")
    n_bins = {c: 2 * S.BLOCK for c in S.DECILE_CHROMS}
    n_bins["chrY"] = 50
    g = tmp_path / "genome"
    g.mkdir()
    with h5py.File(g / "mask.h5", "w") as f:
        f.attrs["fasta_sha256"] = "x"
        for c, n in n_bins.items():
            f.create_dataset(c, data=np.ones(n, np.uint8))
    (g / "hg38-blacklist.v2.bed").write_text("chr2\t0\t10\tHigh Signal Region\n")
    src = tmp_path / "src"
    src.mkdir()
    (src / "GRCh38-cCREs.bed").write_text(
        "chr1\t0\t50\tEH38D1\tEH38E1\tPLS\nchr1\t100\t125\tEH38D2\tEH38E2\tdELS\n"
        "chr1\t200\t225\tEH38D3\tEH38E3\tpELS\nchr1\t300\t325\tEH38D4\tEH38E4\tCA-CTCF\n")
    with gzip.open(src / "gencode.v40.basic.annotation.gtf.gz", "wt") as fh:
        fh.write('chr1\tH\ttranscript\t26\t50\t.\t-\t.\tgene_type "protein_coding";\n')
    with gzip.open(src / "hg38_genome_100_segments.bed.gz", "wt") as fh:
        for k in range(1, 10):
            fh.write(f"chr3\t{k * 25}\t{k * 25 + 25}\t{10 + k}_HET{k}\n")
        fh.write("chr3\t0\t25\t100_TSS2\n")
    for p in src.iterdir():
        p.with_name(p.name + ".date").write_text("2026-11-10\n")
    t146 = tmp_path / "t146"
    t146.mkdir()
    for i, c in enumerate(S.DECILE_CHROMS):
        np.save(t146 / f"map25__{c}.npy",
                np.r_[np.full(S.BLOCK, i / 100), np.full(S.BLOCK, 0.5 + i / 100)].astype(np.float32))
    umap = tmp_path / "k36.umap.bedgraph.gz"
    umap.write_bytes(b"x")
    monkeypatch.setattr(S, "fetch", lambda url, dest: (_ for _ in ()).throw(AssertionError(url))
                        if not dest.is_file() else "2026-11-10")
    out = tmp_path / "eval" / "strata" / "masks.h5"
    assert S.main(["build", "--genome", str(g), "--src", str(src), "--t146", str(t146),
                   "--umap", str(umap), "--out", str(out)]) == 0
    st = S.load_strata(out, ["promoters", "enhancers", "gene_bodies", "heterochromatin",
                             "mappability_d0", "mappability_d9"])
    assert st["promoters"]["chr1"][:3].tolist() == [True, True, False]
    assert st["enhancers"]["chr1"][[4, 8, 12]].tolist() == [True, True, False]
    assert st["gene_bodies"]["chr1"][:3].tolist() == [False, True, False]
    assert st["heterochromatin"]["chr3"][:11].tolist() == [False] + [True] * 9 + [False]
    assert not st["mappability_d0"]["chrY"].any()
    assert st["mappability_d0"]["chr1"][:S.BLOCK].all()
    assert not st["mappability_d0"]["chr2"].any()          # chr2 block 0 is blacklisted
    with h5py.File(out, "r") as f:
        prov = json.loads(f.attrs["sources"])
        assert {p["annotation"] for p in prov} == {"ccre", "gencode", "chromhmm", "umap_k36"}
        assert all(len(p["md5"]) == 32 and p["url"].startswith("https://") for p in prov)
        assert f["promoters"]["chr1"].chunks[0] <= 4096
        assert f["promoters"]["chr1"].shape == (n_bins["chr1"],)
    capsys.readouterr()
    S.main(["list", "--masks", str(out)])
    lines = capsys.readouterr().out.splitlines()
    assert any(l.startswith("promoters\t2\t") for l in lines)
    assert sum(l.startswith("mappability_d") for l in lines) == 10
