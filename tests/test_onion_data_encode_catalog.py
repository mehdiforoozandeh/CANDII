"""Tests for tools/onion_data/encode_catalog.py (t152 chunk D1). No network: the parser runs on
a fixture trimmed from the real portal JSON of ENCSR780XZD (fetched on Nibi 2026-11-10), and the
census runs on small synthetic tables with hand-counted pairs."""
from __future__ import annotations

import io
import sys
import urllib.error
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from onion_data import encode_catalog as ec  # noqa: E402

_DONOR = {"accession": "ENCDO000ACQ", "sex": "male", "age": "2", "age_units": "year",
          "life_stage": "child", "health_status": "unknown"}
_PIPES = ["Transcription factor ChIP-seq 2", "Histone ChIP-seq 2 (unreplicated)",
          "Histone ChIP-seq 2", "Transcription factor ChIP-seq 2 (unreplicated)"]
EXP = {
    "accession": "ENCSR780XZD", "assay_title": "Histone ChIP-seq", "assay_term_name": "ChIP-seq",
    "replication_type": "isogenic", "status": "released", "date_released": "2020-08-28",
    "internal_tags": ["RegulomeDB_2_2"], "target": {"label": "H3K27me3"},
    "lab": {"title": "Michael Snyder, Stanford"},
    "biosample_ontology": {"term_name": "BE2C", "term_id": "EFO:0005725",
                           "classification": "cell line"},
    "possible_controls": [{"accession": "ENCSR377FNA"}],
    "replicates": [
        {"biological_replicate_number": br, "technical_replicate_number": 1,
         "antibody": {"accession": "ENCAB155VEG"},
         "library": {"accession": lib, "biosample": {
             "accession": bs, "organism": {"scientific_name": "Homo sapiens"}, "donor": _DONOR}}}
        for br, lib, bs in ((1, "ENCLB660LDU", "ENCBS277GST"), (2, "ENCLB963JAP", "ENCBS161OUD"))
    ],
    "files": [
        {"accession": acc, "file_format": "fastq", "output_type": "reads", "assembly": None,
         "run_type": "paired-ended", "read_length": 100, "paired_end": end,
         "paired_with": f"/files/{mate}/", "flowcell_details": [{"lane": "4", "barcode": "ACAGTG"}],
         "biological_replicates": [2], "technical_replicates": ["2_1"], "file_size": size,
         "md5sum": "x", "status": "released", "read_count": 63700456,
         "href": f"/files/{acc}/@@download/{acc}.fastq.gz",
         "platform": {"title": "Illumina HiSeq 4000"},
         "cloud_metadata": {"url": f"https://encode-public.s3.amazonaws.com/{acc}.fastq.gz"},
         "replicate": {"library": "/libraries/ENCLB963JAP/"}, "quality_metrics": [],
         "analyses": []}
        for acc, mate, end, size in (("ENCFF522ZTU", "ENCFF491GRA", "1", 3922157559),
                                     ("ENCFF491GRA", "ENCFF522ZTU", "2", 4337947983))
    ] + [{
        "accession": "ENCFF598CZS", "file_format": "bam", "output_type": "alignments",
        "assembly": "GRCh38", "run_type": None, "read_length": None, "paired_end": None,
        "paired_with": None, "flowcell_details": [], "biological_replicates": [1],
        "technical_replicates": ["1_1"], "file_size": 7533859123, "md5sum": "y",
        "status": "released", "read_count": None, "platform": None,
        "href": "/files/ENCFF598CZS/@@download/ENCFF598CZS.bam", "cloud_metadata": {},
        "quality_metrics": [
            {"@type": ["ChipAlignmentQualityMetric"], "processing_stage": "unfiltered",
             "mapped_reads": 164620944},
            {"@type": ["ChipAlignmentQualityMetric"], "processing_stage": "filtered",
             "mapped_reads": 124702970},
            {"@type": ["ChipAlignmentEnrichmentQualityMetric"], "estimated_fragment_len": 210,
             "NSC": 1.024717, "RSC": 0.7882723},
            {"@type": ["ChipLibraryQualityMetric"], "NRF": 0.865308, "PBC1": 0.863579,
             "PBC2": 7.228748},
            "/quality-metrics/not-embedded/"],
        "analysis_step_version": {"analysis_step": {"pipelines": [{"title": t} for t in _PIPES]}},
        "analyses": [{"pipeline_version": "1.5.1", "title": "ENCODE4 v1.5.1 GRCh38"}],
    }],
}


def test_parse_experiment_row_and_libraries():
    p = ec.parse_experiment(EXP)
    e = p["experiment"]
    assert e["experiment"] == "ENCSR780XZD" and e["assay"] == "H3K27me3"
    assert e["target"] == "H3K27me3" and e["biosample_term"] == "BE2C"
    assert e["biosample_type"] == "cell line" and e["replication_type"] == "isogenic"
    assert e["lab"] == "Michael Snyder, Stanford" and e["control_experiments"] == "ENCSR377FNA"
    assert e["internal_tags"] == "RegulomeDB_2_2" and e["organism"] == "Homo sapiens"
    assert set(e) == set(ec.EXPERIMENT_COLS)
    libs = p["libraries"]
    assert [(l["library"], l["bio_rep"], l["tech_rep"], l["donor"], l["antibody"]) for l in libs] == [
        ("ENCLB660LDU", "1", "1", "ENCDO000ACQ", "ENCAB155VEG"),
        ("ENCLB963JAP", "2", "1", "ENCDO000ACQ", "ENCAB155VEG")]
    assert libs[0]["biosample"] == "ENCBS277GST"
    assert [d["donor"] for d in p["donors"]] == ["ENCDO000ACQ"]
    assert p["donors"][0]["sex"] == "male" and p["donors"][0]["organism"] == ""


def test_parse_experiment_files():
    files = {f["file"]: f for f in ec.parse_experiment(EXP)["files"]}
    fq = files["ENCFF522ZTU"]
    assert fq["library"] == "ENCLB963JAP"                 # replicate.library given as an @id
    assert fq["paired_with"] == "ENCFF491GRA" and fq["lane"] == "4" and fq["flowcell"] == ""
    assert fq["platform"] == "Illumina HiSeq 4000" and fq["read_length"] == "100"
    assert fq["url"].startswith("https://encode-public.s3") and fq["file_size"] == "3922157559"
    bam = files["ENCFF598CZS"]
    assert bam["library"] == "ENCLB660LDU"                # mapped through technical_replicates
    assert bam["mapped_reads"] == "124702970"             # the filtered stage wins
    assert (bam["nsc"], bam["rsc"], bam["frag_len"]) == ("1.024717", "0.7882723", "210")
    assert (bam["nrf"], bam["pbc1"], bam["pbc2"]) == ("0.865308", "0.863579", "7.228748")
    assert bam["pipeline"] == "Histone ChIP-seq 2" and bam["pipeline_version"] == "1.5.1"
    assert bam["url"].endswith("/files/ENCFF598CZS/@@download/ENCFF598CZS.bam")
    assert bam["bio_reps"] == "1" and bam["tech_reps"] == "1_1"


def test_assay_of():
    assert ec.assay_of({"assay_title": "Control ChIP-seq"}) == ec.CONTROL
    assert ec.assay_of({"assay_title": "Control Mint-ChIP-seq"}) == ec.CONTROL
    assert ec.assay_of({"assay_title": "ATAC-seq"}) == "ATAC-seq"
    assert ec.assay_of({"assay_title": "Mint-ChIP-seq", "target": {"label": "H3K4me3"}}) == "H3K4me3"
    assert ec.assay_of({"assay_title": "Histone ChIP-seq", "target": {"label": "H3K9me2"}}) == "H3K9me2"
    assert ec.assay_of({"assay_title": "Histone ChIP-seq", "target": {"label": "H4K16ac"}}) == ""
    assert ec.assay_of({"assay_title": "TF ChIP-seq", "target": {"label": "CTCF"}}) == ""
    assert len(ec.PANEL) == 35 and len(ec.CENSUS_ASSAYS) == 36


# ------------------------------------------------------------------ census on synthetic tables

def _exp(acc, term, assay="H3K4me3", status="released"):
    return {"experiment": acc, "assay": assay, "biosample_term": term, "biosample_term_id": term,
            "status": status, "organism": "Homo sapiens"}


def _lib(exp, lib, br, tr, donor):
    return {"experiment": exp, "library": lib, "bio_rep": str(br), "tech_rep": str(tr),
            "donor": donor, "biosample": "", "antibody": ""}


def _fq(exp, lib, acc, fc="", lane="", mate="", size=1e9, status="released"):
    return {"file": acc, "experiment": exp, "library": lib, "file_format": "fastq",
            "flowcell": fc, "lane": lane, "paired_with": mate, "file_size": str(int(size)),
            "status": status}


def _tables():
    exps = [_exp("A", "T1"), _exp("B", "T1"), _exp("C", "T2"), _exp("D", "T2"),
            _exp("E", "T3", assay="ATAC-seq")]
    libs = [_lib("A", "L1", 1, 1, "D1"), _lib("A", "L2", 1, 2, "D1"), _lib("A", "L3", 2, 1, "D2"),
            _lib("B", "L4", 1, 1, "D3"), _lib("C", "L5", 1, 1, "D4"), _lib("D", "L6", 1, 1, "D4"),
            _lib("E", "L7", 1, 1, "D5"), _lib("E", "L8", 2, 1, "D5")]
    files = [
        # L1: two runs with known, distinct keys; each a mate pair -> 1 strict pair
        _fq("A", "L1", "f1", "FC1", "1", mate="f2"), _fq("A", "L1", "f2", "FC1", "1", mate="f1"),
        _fq("A", "L1", "f3", "FC1", "2", mate="f4"), _fq("A", "L1", "f4", "FC1", "2", mate="f3"),
        # L2: two unlabelled single-end fastqs -> 0 strict, 1 loose pair
        _fq("A", "L2", "f5"), _fq("A", "L2", "f6"),
        _fq("A", "L3", "f7", "FC2", "3"),
        _fq("B", "L4", "f8"), _fq("C", "L5", "f9"),
        _fq("D", "L6", "f10", status="deleted"),        # D has no reads: excluded everywhere
        {**_fq("A", "L1", "b1"), "file_format": "bam"},  # not a read file
        _fq("E", "L7", "f11"), _fq("E", "L8", "f12"),
    ]
    return exps, libs, files


def _get(rows, rung, assay):
    return next(r for r in rows if r["rung"] == rung and r["assay"] == assay)


def test_census_hand_counted_pairs():
    rows = ec.census(*_tables())
    assert len(rows) == len(ec.RUNGS) * (len(ec.CENSUS_ASSAYS) + 1)
    k = {r: _get(rows, r, "H3K4me3") for r in ec.RUNGS}
    assert k["pseudoreplicate"]["n_pairs"] == 5               # L1 L2 L3 L4 L5
    assert (k["sequencing"]["n_pairs"], k["sequencing"]["n_pairs_loose"]) == (1, 2)
    assert k["technical"]["n_pairs"] == 1                     # L1-L2
    assert k["isogenic"]["n_pairs"] == 0
    assert k["anisogenic"]["n_pairs"] == 1                    # A br1 (D1) vs br2 (D2)
    assert k["other_experiment"]["n_pairs"] == 1              # A-B in T1
    assert k["other_cell_type"]["n_pairs"] == 2               # A-C, B-C
    assert k["other_experiment"]["n_experiments"] == 2 and k["other_experiment"]["n_cells"] == 1
    assert k["other_cell_type"]["n_cells"] == 2
    assert k["pseudoreplicate"]["fastq_gb"] == "9.0"          # A 7 + B 1 + C 1
    a = {r: _get(rows, r, "ATAC-seq")["n_pairs"] for r in ec.RUNGS}
    assert a["isogenic"] == 1 and a["anisogenic"] == 0 and a["pseudoreplicate"] == 2
    assert _get(rows, "isogenic", "ALL")["n_pairs"] == 1
    assert _get(rows, "pseudoreplicate", "ALL")["n_pairs"] == 7
    assert _get(rows, "technical", "DNase-seq")["n_pairs"] == 0   # zero rows are kept


def test_census_counts_a_library_listed_twice_once():
    exps, libs, files = _tables()
    libs = libs + [_lib("B", "L4", 1, 3, "D3")]          # same library, second replicate entry
    rows = ec.census(exps, libs, files)
    assert _get(rows, "pseudoreplicate", "H3K4me3")["n_pairs"] == 5
    assert _get(rows, "technical", "H3K4me3")["n_pairs"] == 1


def test_census_keep_restricts_cross_experiment_rungs():
    rows = ec.census(*_tables(), keep={"A", "C"})
    assert _get(rows, "other_experiment", "H3K4me3")["n_pairs"] == 0
    assert _get(rows, "other_cell_type", "H3K4me3")["n_pairs"] == 1
    assert _get(rows, "pseudoreplicate", "H3K4me3")["n_pairs"] == 4


def test_summarize_counts_panel_assays_over_threshold():
    rows = ec.census(*_tables())
    s = {r["rung"]: r for r in ec.summarize(rows, min_pairs=2)}
    assert s["pseudoreplicate"]["panel_assays_ge2"] == 2     # H3K4me3 (5) and ATAC-seq (2)
    assert s["other_cell_type"]["panel_assays_ge2"] == 1
    assert s["sequencing"]["n_pairs_loose"] == 2


def test_run_census_from_tables(tmp_path):
    exps, libs, files = _tables()
    exps = [{**e, **{c: e.get(c, "") for c in ec.EXPERIMENT_COLS}} for e in exps]
    exps[1]["status"] = "archived"                           # B leaves the portal census
    ec.write_tsv(str(tmp_path / "experiments.tsv"), exps, ec.EXPERIMENT_COLS)
    ec.write_tsv(str(tmp_path / "libraries.tsv"), libs, ec.LIBRARY_COLS)
    ec.write_tsv(str(tmp_path / "files.tsv"), files, ec.FILE_COLS)
    (tmp_path / "eic.tsv").write_text("experiment\tassay\tsplit\tcell\nA\tH3K4me3\tT\tx\n"
                                      "B\tH3K4me3\tV\ty\n")
    summ = tmp_path / "summ"
    assert ec.main(["census", "--out", str(tmp_path), "--eic-list", str(tmp_path / "eic.tsv"),
                    "--summary-dir", str(summ)]) == 0
    portal = ec.read_tsv(str(tmp_path / "ladder_census.tsv"))
    eic = ec.read_tsv(str(tmp_path / "eic_census.tsv"))
    assert list(portal[0])[:6] == ["rung", "assay", "n_pairs", "n_cells", "n_experiments",
                                   "fastq_gb"]
    assert _get(portal, "other_experiment", "H3K4me3")["n_pairs"] == "0"
    assert _get(eic, "other_experiment", "H3K4me3")["n_pairs"] == "1"
    assert _get(eic, "other_cell_type", "H3K4me3")["n_pairs"] == "0"
    assert {r["rung"] for r in eic} == set(ec.RUNGS)
    for name in ("census_summary.tsv", "eic_census_summary.tsv"):
        assert len(ec.read_tsv(str(summ / name))) == len(ec.RUNGS)


# ------------------------------------------------------------------ fetcher (no network)

def test_fetcher_caches_and_retries(tmp_path):
    calls = []

    def opener(url):
        calls.append(url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(url, 503, "busy", {}, io.BytesIO(b""))
        return b'{"accession": "ENCSR1"}'

    f = ec.Fetcher(str(tmp_path), rate=10, opener=opener, backoff=0.0)
    assert f.get("exp_ENCSR1", "u") == {"accession": "ENCSR1"}
    assert len(calls) == 2 and f.n_net == 1
    assert f.get("exp_ENCSR1", "u") == {"accession": "ENCSR1"}     # skip-if-exists
    assert len(calls) == 2 and f.n_cache == 1
    g = ec.Fetcher(str(tmp_path), rate=10, opener=opener, backoff=0.0)
    assert g.get("exp_ENCSR1", "u") == {"accession": "ENCSR1"} and len(calls) == 2


def test_fetcher_gives_up_on_404(tmp_path):
    def opener(url):
        raise urllib.error.HTTPError(url, 404, "gone", {}, io.BytesIO(b""))

    with pytest.raises(urllib.error.HTTPError):
        ec.Fetcher(str(tmp_path), rate=10, opener=opener, backoff=0.0).get("x", "u")
    assert not list(tmp_path.glob("x.json.gz"))


def test_user_agent_is_generic():
    assert "@" not in ec.USER_AGENT and "mforooz" not in ec.USER_AGENT.lower()
