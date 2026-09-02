"""`tools/challenge_bigwigs.py` — the bigwig -> §4.1 converter, tested without pyBigWig.

pyBigWig lives in the Fir venv and not on this machine, and the two things worth testing are the
two that have nothing to do with it: the BIN RULE, where an off-by-one bin or a nanmean would move
every number downstream, and the BRIDGE JOIN, where a wrong row scores one experiment against
another's truth. Both take an injected `reader(chrom, start, end) -> np.ndarray`, so this file
never imports pyBigWig and the module must not either.

The headline test is an end-to-end identity: feed the converter a reader that replays the synthetic
store's OWN pval layer at base resolution, and the root it writes must hold that layer back, bin for
bin. That is the whole claim — the converter changes the container and not the numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import tools.challenge_bigwigs as C
from candi.bench.external import score_external, stream_truth, track_dirname
from candi.bench.harness import open_source
from candi.store import layout as L

from tests.test_bench_external import (
    C_PAIRS, PAIRED_TRACKS, PAIRS, TARGET_ASSAY, write_root,
)
from tests.test_store_reader import CHROM_SIZES, make_store
from tests.test_store_regime import regime_dict

RES = 25


# ---------------------------------------------------------------------------
# 0 — the lazy import, which is the reason this tool is shaped the way it is
# ---------------------------------------------------------------------------

def test_the_module_imports_with_no_pybigwig_anywhere() -> None:
    """It runs on Fir and is written on a laptop that cannot open a bigwig at all. A top-level
    `import pyBigWig` would make every test in this file unrunnable here."""
    assert "pyBigWig" not in sys.modules
    src = Path(C.__file__).read_text(encoding="utf-8")
    top = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert not any("pyBigWig" in ln for ln in top), top


# ---------------------------------------------------------------------------
# 1 — the bin rule
# ---------------------------------------------------------------------------

def _reader_from_bins(values: np.ndarray, per_base=None):
    """A reader replaying `values` — one number per BIN, repeated across the bin's 25 bases."""
    base = np.repeat(np.asarray(values, dtype=np.float64), RES) if per_base is None else per_base

    def reader(chrom, start, end):
        return base[start:end]
    return reader


def test_a_bin_is_the_mean_of_its_own_25_bases() -> None:
    rng = np.random.default_rng(0)
    base = rng.gamma(1.0, 2.0, 40 * RES)
    got = C.bin_track(_reader_from_bins(None, per_base=base), "chr1", 40 * RES)
    want = base.reshape(40, RES).mean(axis=1).astype(np.float32)
    assert got.dtype == np.float32
    assert np.array_equal(got, want)


def test_the_grid_is_ours_floor_and_not_the_2019_ceil() -> None:
    """`competitors/entrants/vendor/fir_tracks.py` bins at `ceil` with a partial-window fix; D13
    says every kind here is `floor`. The tail bases past the last whole bin are DROPPED, which is
    what makes the partial-window fix unnecessary rather than merely omitted."""
    clen = 40 * RES + 7
    base = np.arange(clen, dtype=np.float64)
    got = C.bin_track(_reader_from_bins(None, per_base=base), "chr1", clen)
    assert len(got) == L.n_bins_for(clen, RES) == 40
    assert len(got) == clen // RES < -(-clen // RES)


def test_a_nan_is_zero_and_not_skipped() -> None:
    """A bigwig has no value where the signal is zero, so a half-covered bin is half zero. A
    nanmean would report the covered half's height as the whole bin's."""
    base = np.full(2 * RES, np.nan)
    base[:RES] = 4.0                       # bin 0 fully covered, bin 1 not covered at all
    base[RES:RES + 5] = 2.0                # ... except its first five bases
    got = C.bin_track(_reader_from_bins(None, per_base=base), "chr1", 2 * RES)
    assert got[0] == pytest.approx(4.0)
    assert got[1] == pytest.approx(5 * 2.0 / RES)


def test_chunking_cannot_change_the_answer() -> None:
    rng = np.random.default_rng(1)
    base = rng.gamma(1.0, 2.0, 97 * RES)
    whole = C.bin_track(_reader_from_bins(None, per_base=base), "chr1", 97 * RES)
    for chunk in (1, 7, 96, 97, 1000):
        part = C.bin_track(_reader_from_bins(None, per_base=base), "chr1", 97 * RES,
                           chunk_bins=chunk)
        assert np.array_equal(part, whole), chunk


def test_a_short_read_is_refused_rather_than_padded() -> None:
    def stingy(chrom, start, end):
        return np.zeros(end - start - 1)

    with pytest.raises(C.ConvertError, match="per-base values"):
        C.bin_track(stingy, "chr1", 10 * RES)


# ---------------------------------------------------------------------------
# 2 — the bridge join (D16: opaque ids, no string surgery)
# ---------------------------------------------------------------------------

BRIDGE_HEADER = "filename,cell_id,assay_id,split,assay_name,biosample_dir,exp_accession"


def _bridge(tmp_path: Path, rows) -> Path:
    p = tmp_path / "eic_bridge.csv"
    p.write_text("\n".join([BRIDGE_HEADER, *rows]) + "\n", encoding="utf-8")
    return p


def test_the_bridge_joins_on_biosample_and_assay_name(tmp_path) -> None:
    p = _bridge(tmp_path, ["C12M01,C12,M01,B,ATAC-seq,B_DND-41,ENCSR660WSB",
                           "C05M18,C05,M18,B,H3K36me3,B_BE2C,ENCSR000AAA"])
    table, sha = C.read_bridge(p)
    assert table[("B_DND-41", "ATAC-seq")] == "C12M01"
    assert table[("B_BE2C", "H3K36me3")] == "C05M18"
    import hashlib
    assert sha == hashlib.sha256(p.read_bytes()).hexdigest()


def test_two_rows_for_one_experiment_are_refused_not_resolved(tmp_path) -> None:
    p = _bridge(tmp_path, ["C12M01,C12,M01,B,ATAC-seq,B_DND-41,ENCSR660WSB",
                           "C99M01,C99,M01,B,ATAC-seq,B_DND-41,ENCSR000ZZZ"])
    with pytest.raises(C.ConvertError, match="twice"):
        C.read_bridge(p)


def test_a_renamed_bridge_column_stops_the_run(tmp_path) -> None:
    p = tmp_path / "b.csv"
    p.write_text("file,cell,assay\nC12M01,C12,ATAC-seq\n", encoding="utf-8")
    with pytest.raises(C.ConvertError, match="biosample_dir"):
        C.read_bridge(p)


def test_a_track_suffix_is_found_however_it_is_spelled(tmp_path) -> None:
    (tmp_path / "C12M01.bigwig").write_bytes(b"x")
    (tmp_path / "C05M18.bw").write_bytes(b"x")
    assert C.bigwig_path(tmp_path, "C12M01").name == "C12M01.bigwig"
    assert C.bigwig_path(tmp_path, "C05M18").name == "C05M18.bw"
    assert C.bigwig_path(tmp_path, "C99M99") is None


# ---------------------------------------------------------------------------
# 3 — a whole root, over a real store, scored by the real scorer
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Path:
    return make_store(tmp_path_factory.mktemp("cbwstore"), tracks=PAIRED_TRACKS)


@pytest.fixture(scope="module")
def regime_file(store, tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("cbwregime")
    obj = regime_dict(store, biosamples={"train": ["T_aa", "T_bb"], "eval": ["V_aa", "V_bb"]},
                      kinds=["counts", "peaks", "pval"],
                      eval_pairs=[["T_aa", "V_aa"], ["T_bb", "V_bb"]])
    p = d / "regime.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def chrom_sizes_file(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("cbwgenome") / "chrom_sizes.json"
    p.write_text(json.dumps(CHROM_SIZES), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def bridge_file(tmp_path_factory) -> Path:
    """`V_aa`/`V_bb` stand in for the challenge's `B_` cells; the join is the same two columns."""
    rows = [f"C0{i}M22,C0{i},M22,B,{TARGET_ASSAY},{b},ENCSR00000{i}"
            for i, b in enumerate(("V_aa", "V_bb"), start=1)]
    return _bridge(tmp_path_factory.mktemp("cbwbridge"), rows)


@pytest.fixture(scope="module")
def store_pval(regime_file):
    """The store's own pval layer for both declared tracks, `{dirname: {chrom: vector}}`.

    Read through `external.stream_truth`, which is the very walk the scorer performs — so the
    identity test below compares the converter's output against the truth the store path would
    have used, not against a second reading of the same h5.
    """
    src = open_source(store=regime_file)
    try:
        col = src.assays.index(TARGET_ASSAY)
        out = {}
        for pair in PAIRS:
            got = stream_truth(src, pair, [col])
            out[track_dirname(pair, TARGET_ASSAY)] = {
                c: np.asarray(v["pval"], dtype=np.float32) for c, v in got[col].items()}
        return out
    finally:
        src.close()


def _replaying_opener(store_pval):
    """`open_reader(path)` replaying the store's pval layer at BASE resolution.

    `C01M22` -> the `V_aa` track, `C02M22` -> `V_bb`, which is the same lookup the real bigwigs get
    through the bridge. Each bin's value is repeated across its 25 bases, so the converter's mean
    must return the bin back unchanged — anything else is the converter, not the data.
    """
    by_stem = {f"C0{i}M22": track_dirname(p, TARGET_ASSAY) for i, p in enumerate(PAIRS, start=1)}

    def opener(path: Path):
        vecs = store_pval[by_stem[Path(path).stem]]

        def reader(chrom, start, end):
            return np.repeat(vecs[chrom].astype(np.float64), RES)[start:end]
        return reader
    return opener


@pytest.fixture(scope="module")
def truth_root(tmp_path_factory, regime_file, chrom_sizes_file, bridge_file, store_pval) -> Path:
    bw = tmp_path_factory.mktemp("cbwbigwigs")
    for stem in ("C01M22", "C02M22"):
        (bw / f"{stem}.bigwig").write_bytes(b"not a real bigwig; the reader is injected")
    out = tmp_path_factory.mktemp("cbwtruth") / "B_"
    C.build_root(out, bigwig_dir=bw, bridge=bridge_file, regime=regime_file,
                 chrom_sizes=chrom_sizes_file, chroms=["chr2"], kind="truth",
                 open_reader=_replaying_opener(store_pval), progress=False)
    return out


def test_the_converter_returns_the_layer_it_was_given_bin_for_bin(truth_root, store_pval) -> None:
    """THE IDENTITY. The container changes; the numbers do not."""
    for dirname, vecs in store_pval.items():
        got = np.load(truth_root / dirname / "chr2.npz")["signal_mu"]
        assert got.dtype == np.float32
        assert np.array_equal(got, vecs["chr2"]), dirname


def test_the_truth_manifest_says_what_it_was_built_from(truth_root, bridge_file) -> None:
    import hashlib
    m = json.loads((truth_root / "manifest.json").read_text())
    assert m["kind"] == "truth" and m["truth"] == "challenge"
    assert m["bridge_sha256"] == hashlib.sha256(bridge_file.read_bytes()).hexdigest()
    assert m["declared_tracks"] == 2 and m["chroms"] == ["chr2"]
    assert m["skipped_tracks"] == []
    assert "floor(chrom_len/25)" in m["bin_rule"] and "NaN->0" in m["bin_rule"]
    assert m["generated_by"] == "tools/challenge_bigwigs.py"
    assert sorted(m["tracks"]) == sorted(track_dirname(p, TARGET_ASSAY) for p in PAIRS)


def test_a_converted_truth_root_is_what_the_scorer_asks_for(truth_root, regime_file,
                                                            store_pval, tmp_path) -> None:
    """End to end: the converter writes it, `bench.external --truth-root` opens it, and the pval
    numbers are the store path's own — because the root holds the store's own layer."""
    src = open_source(store=regime_file)
    try:
        recs = [type("R", (), {"pair": p, "assay": TARGET_ASSAY, "chroms": ("chr2",),
                               "signal_mu": {"chr2": store_pval[track_dirname(p, TARGET_ASSAY)]
                                             ["chr2"] * 0.5}})()
                for p in PAIRS]
        pred = write_root(tmp_path / "pred", recs, keep=("signal_mu",))
        got = score_external(src, pred, seed=0, c_index_pairs=C_PAIRS, truth_root=truth_root)
    finally:
        src.close()
    assert got["provenance"]["truth"]["source"] == "challenge"
    assert len(got["tracks"]) == 2
    for key, arms in got["per_track"].items():
        assert set(arms) == {"pval"}
        # half the truth, everywhere: the MSE is the mean of (y/2)^2 over the scored bins
        pair = next(p for p in PAIRS if key.startswith(f"{p.input_biosample}|"))
        y = store_pval[track_dirname(pair, TARGET_ASSAY)]["chr2"].astype(np.float64)
        assert arms["pval"]["mse"] == pytest.approx(float(((y - y * 0.5) ** 2).mean()), rel=1e-6)


# ---------------------------------------------------------------------------
# 4 — the refusals
# ---------------------------------------------------------------------------

def test_a_track_with_no_bridge_row_stops_the_run(tmp_path_factory, regime_file, chrom_sizes_file,
                                                  store_pval) -> None:
    """UIOWA_Michaelson never submitted C38M18 (`competitors/entrants/README.md` §7). A hole is
    fatal by default and recorded, never silently converted around."""
    d = tmp_path_factory.mktemp("cbwhole")
    bridge = _bridge(d, [f"C01M22,C01,M22,B,{TARGET_ASSAY},V_aa,ENCSR000001"])
    bw = d / "bw"
    bw.mkdir()
    (bw / "C01M22.bigwig").write_bytes(b"x")
    kw = dict(bigwig_dir=bw, bridge=bridge, regime=regime_file, chrom_sizes=chrom_sizes_file,
              chroms=["chr2"], kind="truth", open_reader=_replaying_opener(store_pval),
              progress=False)
    gone = track_dirname(PAIRS[1], TARGET_ASSAY)
    with pytest.raises(C.ConvertError, match=gone):
        C.build_root(d / "out1", **kw)

    m = C.build_root(d / "out2", allow_missing=True, **kw)
    assert [s["track"] for s in m["skipped_tracks"]] == [gone]
    assert sorted(m["tracks"]) == [track_dirname(PAIRS[0], TARGET_ASSAY)]
    assert m["declared_tracks"] == 2, "the DECLARED count is the panel, not what was written"


def test_a_bigwig_the_bridge_names_but_the_directory_lacks_stops_the_run(
        tmp_path_factory, regime_file, chrom_sizes_file, bridge_file, store_pval) -> None:
    d = tmp_path_factory.mktemp("cbwnofile")
    bw = d / "bw"
    bw.mkdir()
    (bw / "C01M22.bigwig").write_bytes(b"x")           # C02M22 is bridged but not present
    with pytest.raises(C.ConvertError, match="C02M22"):
        C.build_root(d / "out", bigwig_dir=bw, bridge=bridge_file, regime=regime_file,
                     chrom_sizes=chrom_sizes_file, chroms=["chr2"], kind="truth",
                     open_reader=_replaying_opener(store_pval), progress=False)


def test_a_pred_root_carries_the_entrant_manifest_section_4_1_asks_for(
        tmp_path_factory, regime_file, chrom_sizes_file, bridge_file, store_pval) -> None:
    d = tmp_path_factory.mktemp("cbwpred")
    bw = d / "bw"
    bw.mkdir()
    for stem in ("C01M22", "C02M22"):
        (bw / f"{stem}.bigwig").write_bytes(b"x")
    m = C.build_root(d / "root", bigwig_dir=bw, bridge=bridge_file, regime=regime_file,
                     chrom_sizes=chrom_sizes_file, chroms=["chr2"], kind="pred",
                     method="Guacamole", open_reader=_replaying_opener(store_pval), progress=False)
    assert m["method"] == "Guacamole" and m["version"] == C.ENTRANT_VERSION
    assert m["arms"] == ["pval"] and m["lineage"] == "entrant"
    assert "kind" not in m, "a prediction root is not a truth root and must not read as one"
    for field in ("generated_by", "date", "notes"):
        assert field in m, field


def test_a_pred_root_with_no_method_is_refused(tmp_path_factory, regime_file, chrom_sizes_file,
                                               bridge_file) -> None:
    d = tmp_path_factory.mktemp("cbwnomethod")
    (d / "bw").mkdir()
    with pytest.raises(C.ConvertError, match="method"):
        C.build_root(d / "root", bigwig_dir=d / "bw", bridge=bridge_file, regime=regime_file,
                     chrom_sizes=chrom_sizes_file, chroms=["chr2"], kind="pred", progress=False)


def test_a_chromosome_the_sizes_file_does_not_carry_is_refused(tmp_path_factory, regime_file,
                                                               chrom_sizes_file,
                                                               bridge_file) -> None:
    d = tmp_path_factory.mktemp("cbwbadchrom")
    (d / "bw").mkdir()
    with pytest.raises(C.ConvertError, match="chr99"):
        C.build_root(d / "root", bigwig_dir=d / "bw", bridge=bridge_file, regime=regime_file,
                     chrom_sizes=chrom_sizes_file, chroms=["chr99"], kind="truth", progress=False)


def test_the_cli_parses_both_verbs() -> None:
    p = C.build_parser()
    a = p.parse_args(["truth-root", "--bigwig-dir", "d", "--bridge", "b", "--regime", "r",
                      "--chrom-sizes", "s", "--chroms", "chr20,chr21", "--out", "o"])
    assert a.verb == "truth-root" and a.chroms == "chr20,chr21"
    b = p.parse_args(["pred-root", "--bigwig-dir", "d", "--bridge", "b", "--regime", "r",
                      "--chrom-sizes", "s", "--chroms", "chr20", "--out", "o",
                      "--method", "Guacamole", "--allow-missing"])
    assert b.verb == "pred-root" and b.method == "Guacamole" and b.allow_missing is True
