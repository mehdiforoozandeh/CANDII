"""The store writer's contract, checked against a synthetic npz tree.

No real corpus is reachable from a laptop, so every test here builds its own ENCODE-shaped tree in
`tmp_path` and reads the result back with **plain h5py** — never through the writer's own helpers,
because a round-trip through one codec proves nothing about what is on disk.

Covers `STORE_PLAN.md` §7 items 1, 6 and 7, plus the four decisions whose failure modes are silent:
D13 (the ceil/floor truncation), D18 (`control_col`), D20 (the cross-check that must fail), and
D15 (no RNA-seq filter).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest

from candi.store import layout as L
from candi.store.layout import LengthMismatch, StoreError
from candi.store.writer import SourceLayout, build_biosample, discover_biosamples

RES = 25
CHROM_SIZES = {"chr1": 25_000 * RES + 7, "chr2": 4_000 * RES + 1}
N_BINS = {c: n // RES for c, n in CHROM_SIZES.items()}

CSV_HEADER = (
    "biosample_name,assay_name,bios_accession,exp_accession,file_accession,assembly,"
    "read_length,run_type,sequencing_platform,lab,depth\n"
)


# ---------------------------------------------------------------------------------------------
# a synthetic ENCODE-shaped npz tree
# ---------------------------------------------------------------------------------------------


def _write_npz(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # deliberately not the key the writer looks for — only files[0] may be read
    np.savez_compressed(path, whatever_key=arr)


def make_source_tree(
    root: Path,
    biosample: str = "T_SYNTH",
    tracks=("H3K4me3", "DNase-seq", "RNA-seq", "chipseq-control"),
    kinds=("counts", "peaks", "pval"),
    chroms=("chr1", "chr2"),
    *,
    counts_max: int = 73,
    pval_max: float = 161.2,
    seed: int = 0,
) -> dict:
    """Build `<root>/<BIOSAMPLE>/<TRACK>/{signal_DSF1_res25,peaks_res25,signal_BW_res25}/chr*.npz`.

    Counts get `n_bins + 1` (ceil, as the real corpus does); peaks and pval get `n_bins` (floor).
    Returns the ground-truth arrays already truncated to `n_bins`, keyed `(track, kind, chrom)`.
    """
    rng = np.random.default_rng(seed)
    truth = {}
    for track in tracks:
        tdir = root / biosample / track
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / "file_metadata.json").write_text(
            json.dumps(
                {
                    "assay": {"7": track},
                    "accession": {"7": f"ENCFF-{track}"},
                    "read_length": {"7": 36},
                    "run_type": {"7": "single-ended"},
                    "sequencing_platform": {"7": "Illumina HiSeq 2500"},
                    "lab": {"7": "Synthetic Lab"},
                }
            ),
            encoding="utf-8",
        )
        for kind in kinds:
            if kind in ("peaks", "pval") and track == L.CONTROL_TRACK:
                continue  # the control has counts only, exactly like the real corpus
            kdir = tdir / {
                "counts": f"signal_DSF1_res{RES}",
                "peaks": f"peaks_res{RES}",
                "pval": f"signal_BW_res{RES}",
            }[kind]
            if kind == "counts":
                (kdir).mkdir(parents=True, exist_ok=True)
                (kdir / "metadata.json").write_text(
                    json.dumps({"coverage": 0.31, "depth": 24_684_534, "dsf": 1}), encoding="utf-8"
                )
            for chrom in chroms:
                nb = N_BINS[chrom]
                if kind == "counts":
                    arr = rng.integers(0, counts_max + 1, size=nb + 1, dtype=np.int64)
                    arr[0] = counts_max
                    truth[(track, kind, chrom)] = arr[:nb].copy()
                elif kind == "peaks":
                    arr = (rng.random(nb) < 0.01).astype(np.int64)
                    truth[(track, kind, chrom)] = arr.copy()
                else:
                    arr = (rng.random(nb) * pval_max).astype(np.float32)
                    arr[0] = np.float32(pval_max)
                    truth[(track, kind, chrom)] = arr.copy()
                _write_npz(kdir / f"{chrom}.npz", arr)
    return truth


def write_csv(path: Path, rows, header: str = CSV_HEADER) -> Path:
    path.write_text(header + "".join(rows), encoding="utf-8")
    return path


def default_rows(biosample="T_SYNTH", tracks=("H3K4me3", "DNase-seq", "RNA-seq")):
    return [
        f"{biosample},{t},ENCBS001,ENCSR001,ENCFF-{t},GRCh38,"
        f"36,single-ended,Illumina HiSeq 2500,Synthetic Lab,24684534.0\n"
        for t in tracks
    ]


@pytest.fixture()
def built(tmp_path):
    """A synthetic source tree plus a store built from all three kinds."""
    src = tmp_path / "src"
    corpus = tmp_path / "store" / "eic"
    truth = make_source_tree(src)
    summary = build_biosample(
        src, corpus, "T_SYNTH", chrom_sizes=CHROM_SIZES, kinds=("counts", "peaks", "pval")
    )
    return {"src": src, "corpus": corpus, "truth": truth, "summary": summary}


# ---------------------------------------------------------------------------------------------
# §7.1 — round trip
# ---------------------------------------------------------------------------------------------


def test_round_trip_counts_and_peaks_are_exact(built):
    """counts and peaks must come back bit-identical; a lossy counts store is not a store."""
    for kind in ("counts", "peaks"):
        with h5py.File(L.kind_path(built["corpus"], "T_SYNTH", kind), "r") as f:
            tracks = json.loads(f.attrs[L.ATTR_TRACKS])
            for chrom in ("chr1", "chr2"):
                block = f[chrom][...]
                assert block.shape == (N_BINS[chrom], len(tracks))
                for j, track in enumerate(tracks):
                    want = built["truth"][(track, kind, chrom)]
                    assert np.array_equal(block[:, j], want.astype(block.dtype)), (kind, track, chrom)


def test_round_trip_pval_is_within_the_quantization_bound(built):
    """D25 — the codec's whole justification is a FLAT relative error of 1/(2*scale).

    That is what changed at t24 and it is why the bound below is not the D9 form's absolute 0.005.
    Rounding bounds the error IN ARCSINH SPACE at `eps`; `sinh` carries it back multiplied by the
    `cosh` Jacobian, which is `sqrt(1 + x^2)`. So the exact bound is `eps * hypot(1, x)` — an
    ABSOLUTE `eps` below 1, a RELATIVE `eps` above it, and one expression for both. Writing it this
    way rather than as a loose relative bound is the point: it is the identity the whole codec rests
    on, and a test that only checked "small" would not notice if `sinh` and `arcsinh` ever stopped
    cancelling.
    """
    eps = 1.0 / (2 * L.PVAL_SCALE)
    with h5py.File(L.kind_path(built["corpus"], "T_SYNTH", "pval"), "r") as f:
        assert int(f.attrs[L.ATTR_SCALE]) == L.PVAL_SCALE
        assert f.attrs[L.ATTR_TRANSFORM] == L.PVAL_TRANSFORM
        tracks = json.loads(f.attrs[L.ATTR_TRACKS])
        for chrom in ("chr1", "chr2"):
            got = L.decode_pval(f[chrom][...], L.PVAL_SCALE, L.PVAL_TRANSFORM).astype(np.float64)
            for j, track in enumerate(tracks):
                want = np.asarray(built["truth"][(track, "pval", chrom)], dtype=np.float64)
                err = np.abs(got[:, j] - want)
                assert np.all(err <= eps * np.hypot(1.0, want) * 1.01), (
                    track, chrom, float(np.max(err / (eps * np.hypot(1.0, want)))))


def test_the_npz_array_key_is_irrelevant(built):
    """Only `files[0]` is read — a tree keyed `whatever_key` must still build."""
    assert built["summary"]["kinds"]["counts"]["bins"] == sum(N_BINS.values())


def test_root_attrs_are_complete_and_self_describing(built):
    for kind in ("counts", "peaks", "pval"):
        with h5py.File(L.kind_path(built["corpus"], "T_SYNTH", kind), "r") as f:
            a = L.read_root_attrs(f)
        assert a[L.ATTR_SCHEMA] == L.SCHEMA_VERSION == 2      # D27
        assert a[L.ATTR_BIOSAMPLE] == "T_SYNTH"
        assert a[L.ATTR_KIND] == kind
        assert a[L.ATTR_RESOLUTION] == 25
        assert a[L.ATTR_N_BINS] == {c: N_BINS[c] for c in ("chr1", "chr2")}
        assert a[L.ATTR_SOURCE_ROOT] == str(built["src"])
        assert a[L.ATTR_KIT_VERSION] and a[L.ATTR_BUILT_UTC].endswith("Z")
        assert a[L.ATTR_DTYPE] == {"counts": "uint16", "peaks": "uint8", "pval": "uint16"}[kind]
    with h5py.File(L.kind_path(built["corpus"], "T_SYNTH", "counts"), "r") as f:
        assert f.attrs[L.ATTR_DSF] == 1                      # D6
        assert json.loads(f.attrs[L.ATTR_NPZ_DEPTH])[0] == 24_684_534
    # D27 — the units record. `scale` alone was what a consumer had to read units off before t24,
    # and it does not say which of two codecs produced the codes; the pair does.
    with h5py.File(L.kind_path(built["corpus"], "T_SYNTH", "pval"), "r") as f:
        assert f.attrs[L.ATTR_SCALE] == 2000
        assert f.attrs[L.ATTR_TRANSFORM] == "arcsinh"
    for kind in ("counts", "peaks"):
        with h5py.File(L.kind_path(built["corpus"], "T_SYNTH", kind), "r") as f:
            assert L.ATTR_TRANSFORM not in f.attrs, f"{kind} must not claim a pval codec"


def test_chunking_and_compression_are_the_declared_policy(built):
    """D2 — chunk `(1024, all tracks)`, gzip 4. lzf or a 512-bin chunk is a silent regression."""
    with h5py.File(L.kind_path(built["corpus"], "T_SYNTH", "counts"), "r") as f:
        ds = f["chr1"]
        assert ds.chunks == (1024, ds.shape[1])
        assert ds.compression == "gzip" and ds.compression_opts == 4


# ---------------------------------------------------------------------------------------------
# §7.6 — the dtype rule (D7)
# ---------------------------------------------------------------------------------------------


def test_counts_dtype_is_uint16_when_the_max_fits(built):
    with h5py.File(L.kind_path(built["corpus"], "T_SYNTH", "counts"), "r") as f:
        assert f["chr1"].dtype == np.uint16


def test_a_value_above_65535_forces_uint32_and_round_trips(tmp_path):
    """D7 — the dtype is per FILE and set by the biosample's global max, not per chromosome."""
    src, corpus = tmp_path / "src", tmp_path / "store" / "eic"
    truth = make_source_tree(src, tracks=("DNase-seq", "H3K4me3"), kinds=("counts",))
    big = 70_000
    path = src / "T_SYNTH" / "DNase-seq" / f"signal_DSF1_res{RES}" / "chr2.npz"
    arr = truth[("DNase-seq", "counts", "chr2")].copy()
    arr[5] = big
    _write_npz(path, np.append(arr, 0))
    summary = build_biosample(src, corpus, "T_SYNTH", chrom_sizes=CHROM_SIZES, kinds=("counts",))
    assert summary["counts_max"] == big
    with h5py.File(L.kind_path(corpus, "T_SYNTH", "counts"), "r") as f:
        assert f["chr1"].dtype == np.uint32 and f["chr2"].dtype == np.uint32   # per FILE
        assert f.attrs[L.ATTR_DTYPE] == "uint32"
        j = json.loads(f.attrs[L.ATTR_TRACKS]).index("DNase-seq")
        assert f["chr2"][5, j] == big


def test_counts_dtype_for_max_boundaries():
    assert L.counts_dtype_for_max(65535) == np.dtype(np.uint16)
    assert L.counts_dtype_for_max(65536) == np.dtype(np.uint32)
    assert L.counts_dtype_for_max(52_051) == np.dtype(np.uint16)   # B_DND-41's DNase peak bin
    with pytest.raises(StoreError):
        L.counts_dtype_for_max(2**33)
    with pytest.raises(StoreError):
        L.counts_dtype_for_max(-1)


# ---------------------------------------------------------------------------------------------
# §7.7 — the pval codec (D9, as amended by PVAL_CODEC_PLAN.md D25-D28)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0.0, 0.1, 1.005, 12.3456, 161.2, 655.35, 17731.0, 2.4e5])
def test_pval_codec_round_trips_within_the_bound(value):
    """`eps * hypot(1, x)`: absolute below 1, relative above it — one epsilon across seven decades.

    17,731 is the corpus maximum the teammate's report named and the value D25 exists to keep; under
    the D9 codec it stored as 655.35, a 96% error on a peak summit.
    """
    eps = 1.0 / (2 * L.PVAL_SCALE)
    enc, n_clipped = L.encode_pval(np.array([value], dtype=np.float64))
    assert n_clipped == 0
    got = float(L.decode_pval(enc)[0])
    assert abs(got - value) <= eps * math.hypot(1.0, value) * 1.01, (value, got)


def test_the_corpus_maximum_no_longer_clips():
    """The defect, stated as a test. 62 of 363 EIC tracks hit the old ceiling; this is why."""
    enc, n_clipped = L.encode_pval(np.array([17731.0], dtype=np.float64))
    assert n_clipped == 0
    assert int(enc[0]) < L.PVAL_UINT16_MAX
    assert L.pval_max_encodable() > 1e13
    # And the old codec, for contrast — kept reachable so the regression has a name.
    _, n_old = L.encode_pval(np.array([17731.0]), L.PVAL_SCALE_LINEAR_V1,
                             transform=L.PVAL_TRANSFORM_LINEAR)
    assert n_old == 1


def test_pval_above_the_ceiling_clips_and_is_counted():
    """`+inf` is the only thing that reaches it now — `p == 0` is real and genuinely infinite.

    It is also the reason this is an integer codec and not float16: float16 REPRESENTS inf, and an
    inf through `arcsinh` into a Gaussian NLL is a NaN loss with no file-level evidence of where it
    came from. A uint16 cannot carry the hazard.
    """
    top = L.pval_max_encodable()
    enc, n_clipped = L.encode_pval(np.array([0.0, 17731.0, np.inf], dtype=np.float64))
    assert enc.tolist()[:2] != [L.PVAL_UINT16_MAX, L.PVAL_UINT16_MAX]
    assert int(enc[2]) == L.PVAL_UINT16_MAX
    assert n_clipped == 1
    assert float(L.decode_pval(enc)[2]) == pytest.approx(top, rel=1e-6)
    # negatives clip at the floor: `-log10 p` is non-negative by construction
    enc, n_clipped = L.encode_pval(np.array([-1.0], dtype=np.float64))
    assert enc.tolist() == [0] and n_clipped == 1


def test_the_linear_codec_still_encodes_and_decodes_for_schema_1_files():
    """D27 — nothing WRITES linear any more, but round-tripping a schema-1 file is a real operation."""
    x = np.array([0.0, 161.2, 655.35, 655.36, 1e6], dtype=np.float64)
    enc, n_clipped = L.encode_pval(x, L.PVAL_SCALE_LINEAR_V1, transform=L.PVAL_TRANSFORM_LINEAR)
    assert enc.tolist() == [0, 16120, 65535, 65535, 65535]
    assert n_clipped == 2
    back = L.decode_pval(enc, L.PVAL_SCALE_LINEAR_V1, L.PVAL_TRANSFORM_LINEAR)
    assert float(back[1]) == pytest.approx(161.2, abs=0.005)
    assert L.pval_max_encodable(L.PVAL_SCALE_LINEAR_V1, L.PVAL_TRANSFORM_LINEAR) == 655.35


def test_decoding_with_the_wrong_transform_is_wrong_which_is_why_the_attr_exists():
    """The codes alone do not say which codec wrote them — hence D27's `transform` root attr."""
    enc, _ = L.encode_pval(np.array([100.0], dtype=np.float64))
    right = float(L.decode_pval(enc, L.PVAL_SCALE, L.PVAL_TRANSFORM)[0])
    wrong = float(L.decode_pval(enc, L.PVAL_SCALE, L.PVAL_TRANSFORM_LINEAR)[0])
    assert right == pytest.approx(100.0, rel=1e-3)
    assert abs(wrong - 100.0) > 90.0
    with pytest.raises(StoreError, match="unknown pval transform"):
        L.decode_pval(enc, L.PVAL_SCALE, "log1p")


def test_pval_clip_fraction_is_recorded_per_track(tmp_path):
    src, corpus = tmp_path / "src", tmp_path / "store" / "eic"
    make_source_tree(src, tracks=("H3K4me3", "DNase-seq"), kinds=("pval",), chroms=("chr2",))
    # every bin of chr2/DNase-seq goes over the ceiling; H3K4me3 stays under it. Built at the LINEAR
    # codec on purpose: under D25 the ceiling is 8.5e13 and no finite float32 reaches it, so the
    # only way to still test that the accounting works is to ask for the codec that can clip.
    nb = N_BINS["chr2"]
    _write_npz(
        src / "T_SYNTH" / "DNase-seq" / f"signal_BW_res{RES}" / "chr2.npz",
        np.full(nb, 700.0, dtype=np.float32),
    )
    build_biosample(src, corpus, "T_SYNTH", chrom_sizes=CHROM_SIZES,
                    kinds=("pval",), chroms=["chr2"],
                    pval_scale=L.PVAL_SCALE_LINEAR_V1, pval_transform=L.PVAL_TRANSFORM_LINEAR)
    with h5py.File(L.kind_path(corpus, "T_SYNTH", "pval"), "r") as f:
        tracks = json.loads(f.attrs[L.ATTR_TRACKS])
        frac = json.loads(f.attrs[L.ATTR_PVAL_CLIP_FRAC])
    assert frac[tracks.index("DNase-seq")] == pytest.approx(1.0)
    assert frac[tracks.index("H3K4me3")] == pytest.approx(0.0)


def test_the_default_codec_clips_nothing_which_is_what_d28_turns_into_an_alarm(tmp_path):
    """D28 — after the rebuild `pval_clip_frac` must read exactly 0.0 everywhere.

    The same 700.0 track that saturates the linear codec is nowhere near this one's ceiling, so a
    nonzero entry in a schema-2 file means a source value above ~8.5e13 — a fault in the SOURCE.
    """
    src, corpus = tmp_path / "src", tmp_path / "store" / "eic"
    make_source_tree(src, tracks=("H3K4me3", "DNase-seq"), kinds=("pval",), chroms=("chr2",))
    _write_npz(
        src / "T_SYNTH" / "DNase-seq" / f"signal_BW_res{RES}" / "chr2.npz",
        np.full(N_BINS["chr2"], 700.0, dtype=np.float32),
    )
    build_biosample(src, corpus, "T_SYNTH", chrom_sizes=CHROM_SIZES,
                    kinds=("pval",), chroms=["chr2"])
    with h5py.File(L.kind_path(corpus, "T_SYNTH", "pval"), "r") as f:
        assert json.loads(f.attrs[L.ATTR_PVAL_CLIP_FRAC]) == [0.0, 0.0]
        assert f.attrs[L.ATTR_TRANSFORM] == "arcsinh" and f.attrs[L.ATTR_SCALE] == 2000


def test_pval_refuses_non_finite_by_default_and_can_be_told_otherwise():
    x = np.array([1.0, np.nan, np.inf], dtype=np.float32)
    with pytest.raises(StoreError, match="non-finite"):
        L.encode_pval(x)
    enc, n_clipped = L.encode_pval(x, nan_policy="zero")
    assert enc.tolist() == [1763, 0, 65535]      # round(arcsinh(1)*2000), 0, the ceiling
    assert n_clipped == 2


# ---------------------------------------------------------------------------------------------
# D13 — the ceil/floor truncation
# ---------------------------------------------------------------------------------------------


def test_n_bins_is_floor_for_every_kind(built):
    for kind in ("counts", "peaks", "pval"):
        with h5py.File(L.kind_path(built["corpus"], "T_SYNTH", kind), "r") as f:
            for chrom in ("chr1", "chr2"):
                assert f[chrom].shape[0] == CHROM_SIZES[chrom] // RES


def test_the_extra_counts_bin_is_truncated_not_kept(built):
    """The counts npz is `ceil` and one bin longer. Keeping it would misalign every kind."""
    src = built["src"]
    raw = np.load(src / "T_SYNTH" / "H3K4me3" / f"signal_DSF1_res{RES}" / "chr1.npz")
    stored_len = N_BINS["chr1"]
    assert raw[raw.files[0]].shape[0] == stored_len + 1
    with h5py.File(L.kind_path(built["corpus"], "T_SYNTH", "counts"), "r") as f:
        assert f["chr1"].shape[0] == stored_len


def test_fit_to_n_bins_accepts_only_floor_or_ceil():
    assert L.fit_to_n_bins(np.arange(10), 10, "w").shape == (10,)
    assert L.fit_to_n_bins(np.arange(11), 10, "w").tolist() == list(range(10))
    with pytest.raises(LengthMismatch, match="chr1"):
        L.fit_to_n_bins(np.arange(12), 10, "bios/track/counts/chr1")
    with pytest.raises(LengthMismatch):
        L.fit_to_n_bins(np.arange(9), 10, "w")


def test_a_short_npz_fails_loudly_at_build_time(tmp_path):
    src, corpus = tmp_path / "src", tmp_path / "store" / "eic"
    make_source_tree(src, tracks=("H3K4me3",), kinds=("counts",), chroms=("chr2",))
    _write_npz(
        src / "T_SYNTH" / "H3K4me3" / f"signal_DSF1_res{RES}" / "chr2.npz",
        np.zeros(N_BINS["chr2"] - 3, dtype=np.int64),
    )
    with pytest.raises(LengthMismatch, match="T_SYNTH/H3K4me3/counts/chr2"):
        build_biosample(src, corpus, "T_SYNTH", chrom_sizes=CHROM_SIZES,
                        kinds=("counts",), chroms=["chr2"])


# ---------------------------------------------------------------------------------------------
# D18 — the control column
# ---------------------------------------------------------------------------------------------


def test_control_is_a_normal_counts_column_flagged_by_control_col(built):
    with h5py.File(L.kind_path(built["corpus"], "T_SYNTH", "counts"), "r") as f:
        tracks = json.loads(f.attrs[L.ATTR_TRACKS])
        col = int(f.attrs[L.ATTR_CONTROL_COL])
    assert tracks[-1] == "chipseq-control" and col == len(tracks) - 1
    assert tracks[:-1] == sorted(tracks[:-1])              # assays sorted, control last


def test_kinds_without_a_control_flag_minus_one(built):
    for kind in ("peaks", "pval"):
        with h5py.File(L.kind_path(built["corpus"], "T_SYNTH", kind), "r") as f:
            assert "chipseq-control" not in json.loads(f.attrs[L.ATTR_TRACKS])
            assert int(f.attrs[L.ATTR_CONTROL_COL]) == -1


def test_control_col_of_is_the_one_rule():
    assert L.control_col_of(["A", "B", "chipseq-control"]) == 2
    assert L.control_col_of(["A", "B"]) == -1
    assert L.order_tracks(["chipseq-control", "H3K4me3", "ATAC-seq"]) == [
        "ATAC-seq", "H3K4me3", "chipseq-control"
    ]


# ---------------------------------------------------------------------------------------------
# D15 / D16 — no filters, opaque names
# ---------------------------------------------------------------------------------------------


def test_rna_seq_is_stored_like_any_other_track(built):
    """D15 — the old `if "RNA-seq" in exps: exps.remove(...)` at three sites does not carry over."""
    with h5py.File(L.kind_path(built["corpus"], "T_SYNTH", "counts"), "r") as f:
        tracks = json.loads(f.attrs[L.ATTR_TRACKS])
        j = tracks.index("RNA-seq")
        assert np.array_equal(f["chr1"][:, j], built["truth"][("RNA-seq", "counts", "chr1")])


def test_biosample_names_are_opaque_ids(tmp_path):
    """D16 — no `T_`/`V_`/`B_` parsing anywhere; `vagina_nonrep` must build like `T_DND-41`."""
    src, corpus = tmp_path / "src", tmp_path / "store" / "eic"
    for name in ("vagina_nonrep", "BE2C_grp1_rep1"):
        make_source_tree(src, biosample=name, tracks=("H3K4me3",), kinds=("counts",),
                         chroms=("chr2",))
    assert discover_biosamples(src) == ["BE2C_grp1_rep1", "vagina_nonrep"]
    for name in discover_biosamples(src):
        build_biosample(src, corpus, name, chrom_sizes=CHROM_SIZES, kinds=("counts",),
                        chroms=["chr2"])
        assert L.kind_path(corpus, name, "counts").is_file()


# ---------------------------------------------------------------------------------------------
# writer guard rails
# ---------------------------------------------------------------------------------------------


def test_a_missing_chromosome_in_one_track_is_an_error(tmp_path):
    src, corpus = tmp_path / "src", tmp_path / "store" / "eic"
    make_source_tree(src, tracks=("H3K4me3", "DNase-seq"), kinds=("counts",))
    (src / "T_SYNTH" / "DNase-seq" / f"signal_DSF1_res{RES}" / "chr2.npz").unlink()
    with pytest.raises(StoreError, match="missing source npz"):
        build_biosample(src, corpus, "T_SYNTH", chrom_sizes=CHROM_SIZES, kinds=("counts",))


def test_rebuilding_without_overwrite_refuses(built):
    with pytest.raises(StoreError, match="already exists"):
        build_biosample(built["src"], built["corpus"], "T_SYNTH", chrom_sizes=CHROM_SIZES,
                        kinds=("counts",))
    build_biosample(built["src"], built["corpus"], "T_SYNTH", chrom_sizes=CHROM_SIZES,
                    kinds=("counts",), overwrite=True)


def test_no_temp_file_survives_a_successful_build(built):
    leftovers = list((built["corpus"] / "biosamples" / "T_SYNTH").glob("*.tmp"))
    assert leftovers == []


def test_peaks_above_255_fail_loudly(tmp_path):
    src, corpus = tmp_path / "src", tmp_path / "store" / "eic"
    make_source_tree(src, tracks=("H3K4me3",), kinds=("peaks",), chroms=("chr2",))
    _write_npz(
        src / "T_SYNTH" / "H3K4me3" / f"peaks_res{RES}" / "chr2.npz",
        np.full(N_BINS["chr2"], 300, dtype=np.int64),
    )
    with pytest.raises(StoreError, match="does not fit uint8"):
        build_biosample(src, corpus, "T_SYNTH", chrom_sizes=CHROM_SIZES, kinds=("peaks",),
                        chroms=["chr2"])


def test_source_layout_paths_match_the_documented_tree(tmp_path):
    src = SourceLayout(root=tmp_path, resolution=25, dsf=1)
    assert src.kind_dir("B", "T", "counts").name == "signal_DSF1_res25"
    assert src.kind_dir("B", "T", "peaks").name == "peaks_res25"
    assert src.kind_dir("B", "T", "pval").name == "signal_BW_res25"
    with pytest.raises(StoreError):
        src.kind_dir("B", "T", "signal")


# ---------------------------------------------------------------------------------------------
# layout odds and ends the other tasks will lean on
# ---------------------------------------------------------------------------------------------


def test_n_bins_rule_and_chrom_ordering():
    assert L.n_bins_for(9_958_256 * 25 + 24) == 9_958_256
    assert L.sort_chroms(["chrX", "chr2", "chr10", "chr1", "chrM", "chrUn_x"]) == [
        "chr1", "chr2", "chr10", "chrX", "chrM", "chrUn_x"
    ]
    with pytest.raises(StoreError):
        L.n_bins_for(0)


def test_chunk_shape_shrinks_on_a_short_chromosome():
    assert L.chunk_shape(100, 3) == (100, 3)
    assert L.chunk_shape(10_000, 3) == (1024, 3)


def test_load_chrom_sizes_reads_json_and_tsv(tmp_path):
    j = tmp_path / "chrom_sizes.json"
    j.write_text(json.dumps({"chr1": 100, "chr2": 50}), encoding="utf-8")
    t = tmp_path / "hg38.chrom.sizes"
    t.write_text("# comment\nchr1\t100\nchr2\t50\n", encoding="utf-8")
    assert L.load_chrom_sizes(j) == L.load_chrom_sizes(t) == {"chr1": 100, "chr2": 50}


def test_path_helpers_are_the_documented_layout(tmp_path):
    corpus = L.corpus_root(tmp_path, "eic")
    assert corpus == tmp_path / "eic"
    assert L.kind_path(corpus, "T_X", "counts") == corpus / "biosamples" / "T_X" / "counts.h5"
    assert L.manifest_path(corpus) == corpus / "manifest.json"
    assert L.corpus_genome_dir(corpus) == (tmp_path / "genome").resolve()
    assert L.dna_path(tmp_path) == tmp_path / "genome" / "dna.h5"
    assert L.mask_path(tmp_path) == tmp_path / "genome" / "mask.h5"


def test_load_chrom_sizes_reads_the_wrapped_genome_form(tmp_path):
    """`genome/chrom_sizes.json` is a provenance record, not a bare `{chrom: length}` map.

    `build-genome` writes the wrapper — `build`, `resolution`, `source` and a precomputed `n_bins`
    around the sizes — because a bare map cannot say which FASTA it came from. The fallback in
    `cli.py::_chrom_sizes_arg` points straight at that file, so a loader that only understood the
    bare form made `build-biosample` die with
    `ValueError: invalid literal for int() with base 10: 'GRCh38'`.
    """
    wrapped = tmp_path / "chrom_sizes.json"
    wrapped.write_text(json.dumps({
        "schema": 1,
        "build": "GRCh38",
        "resolution": 25,
        "source": {"path": "/…/hg38.fa", "sha256": "0" * 64},
        "chrom_sizes": {"chr1": 248956422, "chr21": 46709983},
        "n_bins": {"chr1": 9958256, "chr21": 1868399},
    }))
    assert L.load_chrom_sizes(wrapped) == {"chr1": 248956422, "chr21": 46709983}

    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"chr1": 248956422}))
    assert L.load_chrom_sizes(bare) == {"chr1": 248956422}

    tsv = tmp_path / "hg38.chrom.sizes"
    tsv.write_text("chr1\t248956422\nchr21\t46709983\n")
    assert L.load_chrom_sizes(tsv) == {"chr1": 248956422, "chr21": 46709983}


def test_load_chrom_sizes_still_refuses_a_genuinely_broken_file(tmp_path):
    """Unwrapping must not turn a corrupt file into a silent empty map."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"build": "GRCh38", "resolution": 25}))
    with pytest.raises(ValueError):
        L.load_chrom_sizes(bad)
