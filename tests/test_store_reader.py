"""The reader's contract: the OO tree, the upcast, the decode, and fork safety.

Everything here runs against a **real store built by `candi.store.writer`** in `tmp_path` — the
same code that will build EIC on Fir — so a test that passes here is a test against the on-disk
contract, not against a hand-rolled h5 that agrees with the reader by construction.

`make_store` below is the shared fixture for all three t8 test modules; `test_store_regime.py`
and `test_store_dataset.py` import it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from candi.store import layout as L
from candi.store.layout import StoreError
from candi.store.manifest import build_manifest, write_manifest
from candi.store.reader import CorpusStore, one_hot_dna
from candi.store.writer import build_biosample

RES = 25
CHROM_SIZES = {"chr1": 2_000 * RES + 7, "chr2": 800 * RES + 3}
N_BINS = {c: n // RES for c, n in CHROM_SIZES.items()}
ASSAYS = ("ATAC-seq", "DNase-seq", "H3K4me3")
#: `V_aa` deliberately lacks ATAC-seq and the control — the loader must emit those as MISSING
#: rather than the store pretending they are zero.
TRACKS = {
    "T_aa": ("ATAC-seq", "DNase-seq", "H3K4me3", L.CONTROL_TRACK),
    "V_aa": ("DNase-seq", "H3K4me3"),
}
BIOSAMPLES = tuple(TRACKS)
#: chr2 gets a blacklisted run so D12 has something to reject.
BLACKLIST_BINS = (100, 400)

CSV_HEADER = (
    "biosample_name,assay_name,bios_accession,exp_accession,file_accession,assembly,"
    "read_length,run_type,sequencing_platform,lab,depth\n"
)


# ---------------------------------------------------------------------------------------------
# the shared synthetic store
# ---------------------------------------------------------------------------------------------


def _npz(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, whatever=arr)


def _source_tree(root: Path, tracks_map=None) -> None:
    """`<root>/<BIOS>/<TRACK>/{signal_DSF1_res25,peaks_res25,signal_BW_res25}/chr*.npz`."""
    for bi, (bios, tracks) in enumerate((tracks_map or TRACKS).items()):
        for ti, track in enumerate(tracks):
            tdir = root / bios / track
            tdir.mkdir(parents=True, exist_ok=True)
            (tdir / "file_metadata.json").write_text(
                json.dumps(
                    {
                        "assay": {"1": track},
                        "accession": {"1": f"ENCFF-{bios}-{track}"},
                        "read_length": {"1": 36},
                        "run_type": {"1": "single-ended"},
                    }
                ),
                encoding="utf-8",
            )
            kinds = ("counts",) if track == L.CONTROL_TRACK else ("counts", "peaks", "pval")
            for kind in kinds:
                kdir = tdir / {
                    "counts": f"signal_DSF1_res{RES}",
                    "peaks": f"peaks_res{RES}",
                    "pval": f"signal_BW_res{RES}",
                }[kind]
                kdir.mkdir(parents=True, exist_ok=True)
                if kind == "counts":
                    (kdir / "metadata.json").write_text(
                        json.dumps({"depth": 20_000_000, "dsf": 1}), encoding="utf-8"
                    )
                for chrom, nb in N_BINS.items():
                    rng = np.random.default_rng(1000 * bi + 10 * ti + len(chrom) + ord(chrom[-1]))
                    if kind == "counts":
                        # A wide, high-count column: the thinning tests need counts big enough for
                        # a binomial mean to be measurable, not a field of zeros and ones.
                        arr = rng.integers(0, 200, size=nb + 1).astype(np.int64)  # ceil, as the real npz
                    elif kind == "peaks":
                        arr = (rng.random(nb) < 0.05).astype(np.int64)
                    else:
                        arr = (rng.random(nb) * 40.0).astype(np.float32)
                    _npz(kdir / f"{chrom}.npz", arr)


def _genome_layer(store_root: Path, *, seed: int = 7) -> None:
    """`genome/{chrom_sizes.json,dna.h5,mask.h5}` — t7's files, written directly.

    t7 owns `genome.py`; this only needs the *files* it produces, so it writes the documented
    shapes (D10: `(chr_len,)` uint8 codes; D11: `(n_bins,)` uint8 0/1) and nothing more.
    """
    gdir = L.genome_dir(store_root)
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "chrom_sizes.json").write_text(json.dumps(CHROM_SIZES), encoding="utf-8")
    rng = np.random.default_rng(seed)
    with h5py.File(L.dna_path(store_root), "w") as f:
        f.attrs["build"] = "SYNTH38"
        f.attrs["fasta_sha256"] = "0" * 64
        f.attrs["codes"] = json.dumps({"A": 0, "C": 1, "G": 2, "T": 3, "N": 4})
        for chrom, n in CHROM_SIZES.items():
            f.create_dataset(chrom, data=rng.integers(0, 4, size=n, dtype=np.uint8))
    with h5py.File(L.mask_path(store_root), "w") as f:
        f.attrs["rule"] = "invalid if any N or blacklist overlap"
        for chrom, nb in N_BINS.items():
            m = np.ones(nb, dtype=np.uint8)
            if chrom == "chr2":
                m[BLACKLIST_BINS[0]:BLACKLIST_BINS[1]] = 0
            f.create_dataset(chrom, data=m)


def make_store(tmp: Path, *, corpus: str = "eic", drop_meta=(), tracks=None) -> Path:
    """Build a whole CANDI_STORE — source tree, two biosamples, manifest, genome layer.

    Returns the **corpus root** (`…/CANDI_STORE/eic`), which is what `CorpusStore` takes.
    `drop_meta` is a list of `(biosample, assay)` whose CSV row omits `read_length`, so the
    D19 "incomplete metadata" path has something to work on.

    `tracks` overrides the `{biosample: assays}` layout. It defaults to `TRACKS`, where `V_aa` is a
    strict SUBSET of `T_aa` — the layout that exercises MISSING. An imputation eval needs the
    opposite (a prompt that lacks something the truth carries), and rather than hand-roll a second
    store it passes its own layout through the same real writer. See `tests/test_store_eval_units`.
    """
    layout = dict(tracks or TRACKS)
    src, store_root = tmp / "src", tmp / "CANDI_STORE"
    corpus_root = L.corpus_root(store_root, corpus)
    _source_tree(src, layout)
    for bios in layout:
        build_biosample(src, corpus_root, bios, chrom_sizes=CHROM_SIZES,
                        kinds=("counts", "peaks", "pval"))
    _genome_layer(store_root)

    rows = []
    for bios, tracks in layout.items():
        for track in tracks:
            readlen = "" if (bios, track) in drop_meta else "36"
            rows.append(
                f"{bios},{track},ENCBS1,ENCSR1,ENCFF-{bios}-{track},GRCh38,"
                f"{readlen},single-ended,Illumina HiSeq 2500,Synthetic Lab,20000000\n"
            )
    csv = tmp / "meta.csv"
    csv.write_text(CSV_HEADER + "".join(rows), encoding="utf-8")
    write_manifest(corpus_root, build_manifest(corpus_root, corpus, [csv], source_root=src))
    return corpus_root


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Path:
    return make_store(tmp_path_factory.mktemp("store"))


@pytest.fixture()
def corpus(store):
    c = CorpusStore(store)
    yield c
    c.close()


# ---------------------------------------------------------------------------------------------
# the OO tree (D3) — the API STORE_PLAN.md §5 spells out verbatim
# ---------------------------------------------------------------------------------------------


def test_the_documented_api_reads_exactly_what_the_plan_says(corpus):
    bs = corpus["T_aa"]
    counts = bs["H3K4me3"].counts("chr1", 100, 868)
    block = bs.counts("chr1", 100, 868, assays=list(ASSAYS))
    dna = corpus.genome.dna("chr1", 2500, 21_700)
    assert counts.shape == (768,) and counts.dtype == np.int32
    assert block.shape == (768, 3) and block.dtype == np.int32
    assert dna.shape == (19_200,) and dna.dtype == np.uint8
    assert np.array_equal(block[:, ASSAYS.index("H3K4me3")], counts)


def test_the_tree_is_python_not_hdf5_paths(corpus):
    """D3 — on disk there is one flat dataset per chromosome; the nesting is this API's."""
    with h5py.File(L.kind_path(corpus.root, "T_aa", "counts"), "r") as f:
        assert sorted(f.keys()) == ["chr1", "chr2"]        # no biosample/assay/kind groups
        assert f["chr1"].shape == (N_BINS["chr1"], 4)
    assert corpus["T_aa"]["H3K4me3"].kinds == ["counts", "peaks", "pval"]
    assert corpus["T_aa"][L.CONTROL_TRACK].kinds == ["counts"]


def test_biosample_and_track_structure(corpus):
    assert corpus.biosamples == ["T_aa", "V_aa"]
    assert len(corpus) == 2 and "T_aa" in corpus and "nope" not in corpus
    assert corpus["T_aa"].tracks("counts") == list(TRACKS["T_aa"])   # sorted, control last
    assert corpus["T_aa"].assays("counts") == list(ASSAYS)
    assert corpus["V_aa"].assays("counts") == ["DNase-seq", "H3K4me3"]
    assert corpus["T_aa"].control_col == 3 and corpus["V_aa"].control_col == -1
    assert corpus.assay_vocabulary == sorted(ASSAYS)
    assert corpus.n_bins() == N_BINS and corpus.resolution == RES


def test_an_absent_assay_raises_naming_it(corpus):
    with pytest.raises(StoreError, match="ATAC-seq"):
        corpus["V_aa"].counts("chr1", 0, 10, assays=["ATAC-seq"])
    with pytest.raises(StoreError, match="H3K27ac"):
        corpus["T_aa"]["H3K27ac"]


def test_a_window_off_the_end_of_a_chromosome_raises(corpus):
    with pytest.raises(StoreError, match="not inside"):
        corpus["T_aa"].counts("chr2", N_BINS["chr2"] - 10, N_BINS["chr2"] + 10)
    with pytest.raises(StoreError, match="no dataset for"):
        corpus["T_aa"].counts("chr17", 0, 10)


# ---------------------------------------------------------------------------------------------
# the declared column order (D14) at the reader level
# ---------------------------------------------------------------------------------------------


def test_the_reader_returns_columns_in_the_order_asked_for(corpus):
    bs = corpus["T_aa"]
    forward = bs.counts("chr1", 0, 50, assays=["ATAC-seq", "DNase-seq", "H3K4me3"])
    reversed_ = bs.counts("chr1", 0, 50, assays=["H3K4me3", "DNase-seq", "ATAC-seq"])
    assert np.array_equal(forward[:, ::-1], reversed_)
    assert not np.array_equal(forward, reversed_)          # the values really moved
    twice = bs.counts("chr1", 0, 50, assays=["H3K4me3", "H3K4me3"])
    assert np.array_equal(twice[:, 0], twice[:, 1])


# ---------------------------------------------------------------------------------------------
# codecs: the upcast (D7) and the decode (D9)
# ---------------------------------------------------------------------------------------------


def test_counts_are_upcast_so_uint16_and_uint32_stores_are_interchangeable(tmp_path):
    """D7 — the dtype is a storage fact; nothing downstream may be able to tell."""
    small = make_store(tmp_path / "a")
    with h5py.File(L.kind_path(small, "T_aa", "counts"), "r+") as f:
        assert f["chr1"].dtype == np.uint16
    c16 = CorpusStore(small)["T_aa"].counts("chr1", 0, 20)
    # rebuild the same source as uint32 and compare value for value
    big = L.corpus_root(tmp_path / "a" / "CANDI_STORE", "eic32")
    build_biosample(tmp_path / "a" / "src", big, "T_aa", chrom_sizes=CHROM_SIZES,
                    kinds=("counts",), counts_dtype="uint32")
    c32 = CorpusStore(big, check_manifest=False)["T_aa"].counts("chr1", 0, 20)
    assert c16.dtype == c32.dtype == np.int32
    assert np.array_equal(c16, c32)


def test_pval_comes_back_as_minus_log10_p_not_fixed_point(corpus):
    """D26 — the reader decodes; a caller must never see the uint16 OR the arcsinh.

    The codec is a storage detail and stops at this line: what comes out is ordinary `-log10 p`, in
    the same units the D9 store returned, which is why nothing above the reader had to change when
    the codec did. Any further transform is the objective's (`--signal-target-transform`, D30).
    """
    got = corpus["T_aa"].pval("chr1", 0, 64, assays=["H3K4me3"])[:, 0]
    with h5py.File(L.kind_path(corpus.root, "T_aa", "pval"), "r") as f:
        a = L.read_root_attrs(f)
        j = list(a[L.ATTR_TRACKS]).index("H3K4me3")
        raw = f["chr1"][0:64, j]
    assert a[L.ATTR_TRANSFORM] == "arcsinh" and a[L.ATTR_SCALE] == 2000
    assert got.dtype == np.float32
    assert np.allclose(got, np.sinh(raw / 2000.0), rtol=1e-6)
    assert got.max() > 1.0                                  # the synthetic tree goes up to ~40


def test_a_schema_1_pval_file_still_decodes_linearly(corpus, tmp_path):
    """D27, and the ONLY regression it exists to prevent.

    A schema-1 file carries `scale` and no `transform`. If the reader fell back to this package's
    CURRENT default instead of `"linear"`, it would return `sinh` of a number that was never
    compressed — 161.2 would read back as 1.4e70 — and nothing would raise. The fallback is four
    lines in `BiosampleStore.pval`; this is what those four lines are for.
    """
    import shutil

    root = tmp_path / "schema1"
    shutil.copytree(corpus.root, root)
    path = L.kind_path(root, "T_aa", "pval")
    with h5py.File(path, "r+") as f:
        tracks = list(L.read_root_attrs(f)[L.ATTR_TRACKS])
        j = tracks.index("H3K4me3")
        for chrom in ("chr1", "chr2"):
            block = f[chrom][...]
            f[chrom][...] = np.zeros_like(block)
            f[chrom][0:64, j] = np.rint(
                np.linspace(0.0, 161.2, 64) * L.PVAL_SCALE_LINEAR_V1).astype(block.dtype)
        del f.attrs[L.ATTR_TRANSFORM]                       # what a schema-1 file looks like
        f.attrs[L.ATTR_SCHEMA] = 1
        f.attrs[L.ATTR_SCALE] = L.PVAL_SCALE_LINEAR_V1

    got = CorpusStore(root)["T_aa"].pval("chr1", 0, 64, assays=["H3K4me3"])[:, 0]
    assert np.allclose(got, np.linspace(0.0, 161.2, 64), atol=0.005)


def test_peaks_are_the_zero_one_indicator(corpus):
    pk = corpus["T_aa"].peaks("chr1", 0, 200, assays=list(ASSAYS))
    assert pk.dtype == np.uint8 and set(np.unique(pk)) <= {0, 1}


def test_dna_one_hot_puts_n_at_all_zero():
    codes = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
    oh = one_hot_dna(codes)
    assert oh.shape == (5, 4) and oh.dtype == np.float32
    assert np.array_equal(oh[:4], np.eye(4, dtype=np.float32))
    assert oh[4].sum() == 0.0


# ---------------------------------------------------------------------------------------------
# the manifest is a convenience; the attrs are the record
# ---------------------------------------------------------------------------------------------


def test_a_manifest_that_disagrees_with_the_attrs_is_loud(tmp_path):
    corpus_root = make_store(tmp_path / "m")
    mpath = L.manifest_path(corpus_root)
    obj = json.loads(mpath.read_text(encoding="utf-8"))
    obj["biosamples"]["T_aa"]["control_col"] = 99
    obj["genome"]["n_bins"]["chr1"] = N_BINS["chr1"] + 1
    mpath.write_text(json.dumps(obj), encoding="utf-8")
    c = CorpusStore(corpus_root)
    with pytest.raises(StoreError, match="control_col"):
        c["T_aa"]


def test_a_store_without_a_manifest_still_reads(tmp_path):
    corpus_root = make_store(tmp_path / "n")
    L.manifest_path(corpus_root).unlink()
    c = CorpusStore(corpus_root)
    assert c.manifest is None
    assert c.assay_vocabulary == sorted(ASSAYS)
    assert c.n_bins() == N_BINS
    assert c.track_meta("T_aa", "H3K4me3") is None
    assert c["T_aa"].counts("chr1", 0, 8).shape == (8, 4)


def test_track_meta_comes_from_the_manifest(corpus):
    rec = corpus.track_meta("T_aa", "H3K4me3")
    assert rec["depth"] == 20_000_000 and rec["read_length"] == 36
    assert rec["run_type"] == "single-ended" and rec["col"] == 2
    assert corpus.track_meta("T_aa", "not-an-assay") == {}


def test_pointing_the_reader_at_candi_store_instead_of_a_corpus_says_so(tmp_path):
    corpus_root = make_store(tmp_path / "p")
    with pytest.raises(StoreError, match="no biosamples/"):
        CorpusStore(corpus_root.parent)


# ---------------------------------------------------------------------------------------------
# the genome layer
# ---------------------------------------------------------------------------------------------


def test_genome_is_a_sibling_of_the_corpus_and_reads_bp_and_bin_coordinates(corpus):
    assert corpus.genome.dir == (corpus.root.parent / "genome").resolve()
    assert corpus.genome.has_dna and corpus.genome.has_mask
    assert corpus.genome.chrom_sizes == CHROM_SIZES
    assert corpus.genome.mask("chr2", 0, N_BINS["chr2"]).sum() == (
        N_BINS["chr2"] - (BLACKLIST_BINS[1] - BLACKLIST_BINS[0])
    )
    assert corpus.genome.dna_onehot("chr1", 0, 100).shape == (100, 4)


def test_a_missing_genome_layer_is_named_not_guessed(tmp_path):
    corpus_root = make_store(tmp_path / "g")
    L.dna_path(corpus_root.parent).unlink()
    c = CorpusStore(corpus_root)
    assert not c.genome.has_dna
    with pytest.raises(StoreError, match="dna.h5"):
        c.genome.dna("chr1", 0, 10)


# ---------------------------------------------------------------------------------------------
# fork safety — the reason handles are opened lazily
# ---------------------------------------------------------------------------------------------


def test_handles_are_reopened_after_a_fork_and_never_shared(store):
    """h5py handles are not fork-safe: a child must open its own, and must not close the parent's.

    The parent reads first so a handle is definitely open and inherited, then a forked child reads
    the same window and reports whether it got the same bytes and a DIFFERENT file object.
    """
    if not hasattr(os, "fork"):                                  # pragma: no cover - not on macOS/Linux
        pytest.skip("no os.fork on this platform")
    c = CorpusStore(store)
    parent = c["T_aa"].counts("chr1", 0, 32)
    parent_id = id(c._pool._files[str(L.kind_path(store, "T_aa", "counts"))])
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:                                                 # pragma: no cover - child process
        try:
            os.close(r)
            child = c["T_aa"].counts("chr1", 0, 32)
            same = bool(np.array_equal(child, parent))
            reopened = id(c._pool._files[str(L.kind_path(store, "T_aa", "counts"))]) != parent_id
            os.write(w, b"1" if (same and reopened) else b"0")
            os.close(w)
        finally:
            os._exit(0)
    os.close(w)
    ok = os.read(r, 1)
    os.close(r)
    os.waitpid(pid, 0)
    assert ok == b"1"
    # the parent's handle survived the child: closing an inherited handle would have broken this
    assert np.array_equal(c["T_aa"].counts("chr1", 0, 32), parent)
    c.close()


def test_the_store_objects_pickle_for_spawn_workers(store):
    import pickle

    c = CorpusStore(store)
    want = c["T_aa"].counts("chr1", 0, 16)
    clone = pickle.loads(pickle.dumps(c))
    assert np.array_equal(clone["T_aa"].counts("chr1", 0, 16), want)
    assert np.array_equal(pickle.loads(pickle.dumps(c["T_aa"])).counts("chr1", 0, 16), want)
