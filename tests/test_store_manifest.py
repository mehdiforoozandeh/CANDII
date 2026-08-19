"""`manifest.py` — the CSV authority (D20), the `null` policy (D19), and `verify`.

The cross-check is the point of these tests: a manifest that quietly prefers one source over the
other is worse than no manifest, because the covariates it feeds the model are then wrong with no
symptom. So the disagreement case must raise, and the missing case must write `null`.
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from candi.store import layout as L
from candi.store.layout import StoreError
from candi.store.manifest import (
    AMBIGUOUS,
    MetadataConflict,
    build_manifest,
    read_file_metadata,
    read_metadata_csvs,
    verify_store,
    write_manifest,
)
from candi.store.writer import build_biosample

from tests.test_store_writer import (
    CHROM_SIZES,
    N_BINS,
    RES,
    default_rows,
    make_source_tree,
    write_csv,
)


@pytest.fixture()
def store(tmp_path):
    src, corpus = tmp_path / "src", tmp_path / "store" / "eic"
    make_source_tree(src, kinds=("counts", "peaks", "pval"))
    build_biosample(src, corpus, "T_SYNTH", chrom_sizes=CHROM_SIZES,
                    kinds=("counts", "peaks", "pval"))
    csv_path = write_csv(tmp_path / "eic_metadata.csv", default_rows())
    return {"src": src, "corpus": corpus, "csv": csv_path, "tmp": tmp_path}


# ---------------------------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------------------------


def test_manifest_has_the_documented_shape(store):
    m = build_manifest(store["corpus"], "eic", [store["csv"]], source_root=store["src"])
    assert m["schema"] == 1 and m["corpus"] == "eic" and m["resolution"] == 25
    assert m["kinds"] == ["counts", "peaks", "pval"]
    assert m["genome"]["n_bins"] == N_BINS
    assert m["genome"]["build"] is None and m["genome"]["fasta_sha256"] is None  # no dna.h5 yet
    assert m["assay_vocabulary"] == ["DNase-seq", "H3K4me3", "RNA-seq"]          # control excluded
    assert m["built"]["utc"].endswith("Z") and m["built"]["kit_version"]
    entry = m["biosamples"]["T_SYNTH"]
    assert entry["dtype"] == "uint16"
    assert entry["control_col"] == 3
    tracks = {t["assay"]: t for t in entry["tracks"]}
    assert [t["col"] for t in entry["tracks"]] == [0, 1, 2, 3]
    h = tracks["H3K4me3"]
    assert h["depth"] == 24684534 and h["read_length"] == 36
    assert h["run_type"] == "single-ended" and h["assembly"] == "GRCh38"
    assert h["platform"] == "Illumina HiSeq 2500" and h["lab"] == "Synthetic Lab"
    assert h["pval_clip_frac"] == 0.0 and h["npz_depth"] == 24684534
    assert h["kinds"] == ["counts", "peaks", "pval"]
    assert tracks["chipseq-control"]["kinds"] == ["counts"]


def test_write_manifest_lands_where_the_reader_looks(store):
    m = build_manifest(store["corpus"], "eic", [store["csv"]], source_root=store["src"])
    out = write_manifest(store["corpus"], m)
    assert out == L.manifest_path(store["corpus"])
    assert json.loads(out.read_text(encoding="utf-8"))["corpus"] == "eic"
    assert not list(store["corpus"].glob("*.tmp"))


# ---------------------------------------------------------------------------------------------
# D20 — the cross-check must FAIL
# ---------------------------------------------------------------------------------------------


def test_a_disagreeing_file_metadata_json_fails_loudly(store):
    """The whole point of D20. A silent win for either side poisons the covariates."""
    fm = store["src"] / "T_SYNTH" / "H3K4me3" / "file_metadata.json"
    obj = json.loads(fm.read_text(encoding="utf-8"))
    obj["read_length"] = {"7": 100}          # CSV says 36
    fm.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(MetadataConflict, match="read_length"):
        build_manifest(store["corpus"], "eic", [store["csv"]], source_root=store["src"])


def test_a_disagreeing_run_type_fails_loudly(store):
    fm = store["src"] / "T_SYNTH" / "DNase-seq" / "file_metadata.json"
    obj = json.loads(fm.read_text(encoding="utf-8"))
    obj["run_type"] = {"7": "paired-ended"}  # CSV says single-ended
    fm.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(MetadataConflict, match="run_type"):
        build_manifest(store["corpus"], "eic", [store["csv"]], source_root=store["src"])


def test_no_strict_downgrades_the_conflict_to_a_gap(store):
    fm = store["src"] / "T_SYNTH" / "H3K4me3" / "file_metadata.json"
    obj = json.loads(fm.read_text(encoding="utf-8"))
    obj["accession"] = {"7": "ENCFFWRONG"}
    fm.write_text(json.dumps(obj), encoding="utf-8")
    m = build_manifest(store["corpus"], "eic", [store["csv"]], source_root=store["src"],
                       strict=False)
    assert any(g.get("reason") == "CSV/json conflict" for g in m["metadata_gaps"])


def test_the_cross_check_is_skipped_without_a_source_root(store):
    fm = store["src"] / "T_SYNTH" / "H3K4me3" / "file_metadata.json"
    obj = json.loads(fm.read_text(encoding="utf-8"))
    obj["read_length"] = {"7": 100}
    fm.write_text(json.dumps(obj), encoding="utf-8")
    m = build_manifest(store["corpus"], "eic", [store["csv"]])   # no source_root
    assert m["source_root"] is None


def test_more_than_one_replicate_key_raises_instead_of_picking_one(tmp_path):
    """D19 — the old path hard-coded `"2"`. Picking a key is a coin flip, so we refuse."""
    p = tmp_path / "file_metadata.json"
    p.write_text(json.dumps({"read_length": {"1": 36, "2": 100}}), encoding="utf-8")
    with pytest.raises(StoreError, match="replicate keys"):
        read_file_metadata(p)


def test_whatever_replicate_key_exists_is_used(tmp_path):
    p = tmp_path / "file_metadata.json"
    p.write_text(json.dumps({"read_length": {"17": 36}, "run_type": {"17": "single-ended"}}),
                 encoding="utf-8")
    assert read_file_metadata(p) == {"read_length": 36, "run_type": "single-ended"}
    assert read_file_metadata(tmp_path / "nope.json") is None


# ---------------------------------------------------------------------------------------------
# D19 — nothing fabricated
# ---------------------------------------------------------------------------------------------


def test_a_track_absent_from_every_csv_becomes_null_not_a_default(store):
    """`chipseq-control` is in no CSV until t5 lands. It must be `null`, never `read_length=50`."""
    m = build_manifest(store["corpus"], "eic", [store["csv"]], source_root=store["src"])
    ctrl = next(t for t in m["biosamples"]["T_SYNTH"]["tracks"] if t["assay"] == "chipseq-control")
    for field in ("depth", "read_length", "run_type", "file_accession", "lab"):
        assert ctrl[field] is None, field
    assert {"biosample": "T_SYNTH", "track": "chipseq-control", "field": "*",
            "reason": "no CSV row"} in m["metadata_gaps"]


def test_an_extra_control_csv_fills_the_control_in(store):
    """t5's `control_metadata.csv` is just one more `--metadata-csv`; no code path of its own."""
    ctrl_csv = write_csv(
        store["tmp"] / "control_metadata.csv",
        ["T_SYNTH,chipseq-control,ENCBS001,ENCSR009,ENCFF-chipseq-control,GRCh38,"
         "36,single-ended,Illumina HiSeq 2500,Synthetic Lab,9000000\n"],
    )
    m = build_manifest(store["corpus"], "eic", [store["csv"], ctrl_csv], source_root=store["src"])
    ctrl = next(t for t in m["biosamples"]["T_SYNTH"]["tracks"] if t["assay"] == "chipseq-control")
    assert ctrl["depth"] == 9_000_000 and ctrl["read_length"] == 36
    assert ctrl["file_accession"] == "ENCFF-chipseq-control"
    assert "chipseq-control" not in m["assay_vocabulary"]     # a column, not an assay (D18)


def test_an_empty_csv_cell_becomes_null_and_a_gap(store):
    write_csv(
        store["csv"],
        ["T_SYNTH,H3K4me3,ENCBS001,ENCSR001,ENCFF001,GRCh38,,,Illumina,Lab,24684534\n"]
        + default_rows(tracks=("DNase-seq", "RNA-seq")),
    )
    m = build_manifest(store["corpus"], "eic", [store["csv"]])
    h = next(t for t in m["biosamples"]["T_SYNTH"]["tracks"] if t["assay"] == "H3K4me3")
    assert h["read_length"] is None and h["run_type"] is None
    assert h["depth"] == 24684534
    reasons = {(g["track"], g["field"]) for g in m["metadata_gaps"]}
    assert ("H3K4me3", "read_length") in reasons and ("H3K4me3", "run_type") in reasons


def test_two_rows_that_disagree_make_the_field_ambiguous_not_first_wins(store):
    rows = default_rows() + [
        "T_SYNTH,H3K4me3,ENCBS001,ENCSR001,ENCFF001,GRCh38,"
        "100,single-ended,Illumina,Lab,24684534\n"
    ]
    write_csv(store["csv"], rows)
    table = read_metadata_csvs([store["csv"]])
    assert table[("T_SYNTH", "H3K4me3")]["read_length"] is AMBIGUOUS
    m = build_manifest(store["corpus"], "eic", [store["csv"]])
    h = next(t for t in m["biosamples"]["T_SYNTH"]["tracks"] if t["assay"] == "H3K4me3")
    assert h["read_length"] is None
    assert any(g["field"] == "read_length" and g["reason"] == "CSV rows disagree"
               for g in m["metadata_gaps"])


def test_an_unrecognized_run_type_is_null_not_a_guess(store):
    write_csv(
        store["csv"],
        ["T_SYNTH,H3K4me3,ENCBS001,ENCSR001,ENCFF001,GRCh38,36,mystery,Illumina,Lab,1\n"]
        + default_rows(tracks=("DNase-seq", "RNA-seq")),
    )
    m = build_manifest(store["corpus"], "eic", [store["csv"]])
    h = next(t for t in m["biosamples"]["T_SYNTH"]["tracks"] if t["assay"] == "H3K4me3")
    assert h["run_type"] is None


def test_run_type_and_read_length_normalize_the_same_on_both_sides(store):
    """`76.0` in merged_metadata.csv and `76` in the json must not read as a disagreement."""
    write_csv(
        store["csv"],
        ["T_SYNTH,H3K4me3,ENCBS001,ENCSR001,ENCFF-H3K4me3,GRCh38,"
         "36.0,single,Illumina,Lab,24684534.0\n"] + default_rows(tracks=("DNase-seq", "RNA-seq")),
    )
    m = build_manifest(store["corpus"], "eic", [store["csv"]], source_root=store["src"])
    h = next(t for t in m["biosamples"]["T_SYNTH"]["tracks"] if t["assay"] == "H3K4me3")
    assert h["read_length"] == 36 and h["run_type"] == "single-ended"


def test_a_csv_missing_its_key_columns_is_refused(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("assay,depth\nH3K4me3,1\n", encoding="utf-8")
    with pytest.raises(StoreError, match="missing required column"):
        read_metadata_csvs([p])


# ---------------------------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------------------------


def test_verify_passes_on_a_freshly_built_store(store):
    write_manifest(store["corpus"],
                   build_manifest(store["corpus"], "eic", [store["csv"]],
                                  source_root=store["src"]))
    assert verify_store(store["corpus"]) == []


def test_verify_reports_a_missing_manifest(store):
    problems = verify_store(store["corpus"])
    assert len(problems) == 1 and "build-manifest" in problems[0]


def test_verify_catches_a_deleted_kind_file(store):
    write_manifest(store["corpus"],
                   build_manifest(store["corpus"], "eic", [store["csv"]],
                                  source_root=store["src"]))
    L.kind_path(store["corpus"], "T_SYNTH", "peaks").unlink()
    problems = verify_store(store["corpus"])
    assert any("does not exist" in p for p in problems)


def test_a_store_built_against_two_chrom_size_files_is_refused(store):
    """Two biosamples whose n_bins disagree means two different chrom_sizes. Loud, not merged."""
    other = dict(CHROM_SIZES)
    other["chr2"] = other["chr2"] + 25 * 10
    make_source_tree(store["src"], biosample="T_OTHER", tracks=("H3K4me3",), kinds=("counts",),
                     chroms=("chr2",))
    # rebuild T_OTHER's chr2 npz at the longer length
    from tests.test_store_writer import _write_npz

    _write_npz(
        store["src"] / "T_OTHER" / "H3K4me3" / f"signal_DSF1_res{RES}" / "chr2.npz",
        np.zeros(other["chr2"] // RES + 1, dtype=np.int64),
    )
    build_biosample(store["src"], store["corpus"], "T_OTHER", chrom_sizes=other,
                    kinds=("counts",), chroms=["chr2"])
    with pytest.raises(StoreError, match="two different chrom_sizes"):
        build_manifest(store["corpus"], "eic", [store["csv"]])


def test_manifest_reads_the_genome_block_from_dna_h5(store):
    """t7 writes `genome/dna.h5`; the manifest must pick its build + sha up, not re-derive them."""
    gdir = L.corpus_genome_dir(store["corpus"])
    gdir.mkdir(parents=True, exist_ok=True)
    with h5py.File(gdir / "dna.h5", "w") as f:
        f.attrs["build"] = "GRCh38"
        f.attrs["fasta_sha256"] = "deadbeef"
    m = build_manifest(store["corpus"], "eic", [store["csv"]])
    assert m["genome"]["build"] == "GRCh38" and m["genome"]["fasta_sha256"] == "deadbeef"
