"""Parser tests for tools/onion_data/portal_catalogs.py (chunk D2, task t157). No network.

Fixtures are small excerpts in the real formats seen on 2026-11-10: EpiATLAS readme CSVs and
ChIP-Atlas experimentList.tab lines (tab separated, no header).
"""
import csv
import io
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "tools",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from onion_data import portal_catalogs as pc  # noqa: E402

EXP_CSV = """epirr_id,status,data_generating_centre,assay_type,experiment_type,uuid,inputs,inputs_ctl,original_read_length,original_read_length_ctl,trimmed_read_length,trimmed_read_length_ctl,software_version,chastity_passed,paired_end,analyzed_as_stranded,epirr_id_without_version
IHECRE00000001.4,complete,CEEHRC_[CEMT],ChIP-Seq,H3K27ac,8e4a8d6c-26e0-4b8d-aa08-7875973212d4,EGAR00001176281;EGAR00001332451,EGAR00001176282;EGAR00001332466,75;75,75;75,75;75,75;75,ihec-chipseq-containerv1.1.4,true,true,nan,IHECRE00000001
IHECRE00000001.4,complete,CEEHRC_[CEMT],ChIP-Seq,H3K27me3,7fa161be-cf7b-4429-a672-c1d72473e2c1,EGAR00001176284,EGAR00001176282;EGAR00001332466,75,75;75,75,75;75,ihec-chipseq-containerv1.1.4,true,true,nan,IHECRE00000001
IHECRE00000300.2,partial,NIH_Roadmap_Epigenomics,ChIP-Seq,H3K4me3,aaaa0000-0000-0000-0000-000000000001,SRR899583;SRR899584,SRR899600,36,36,36,36,ihec-chipseq-containerv1.1.4,true,false,nan,IHECRE00000300
IHECRE00000301.1,partial,NIH_Roadmap_Epigenomics,ChIP-Seq,H3K9me3,aaaa0000-0000-0000-0000-000000000002,SRR2172780,SRR899600,36,36,36,36,ihec-chipseq-containerv1.1.4,true,false,nan,IHECRE00000301
IHECRE00000400.1,partial,ENCODE,ChIP-Seq,H3K36me3,aaaa0000-0000-0000-0000-000000000003,ENCFF001AAA,ENCFF001CTL,100,100,100,100,ihec-chipseq-containerv1.1.4,true,false,nan,IHECRE00000400
IHECRE00000001.4,complete,CEEHRC_[CEMT],RNA-Seq,mRNA-Seq,bbbb0000-0000-0000-0000-000000000004,EGAR00001176290,nan,75,nan,75,nan,ihec-rnaseq-containerv1.1.0,true,true,true,IHECRE00000001
"""
IMP_CSV = """mark,epirr_id,data_file_path
H3K27ac,IHECRE00000001.4,incoming/ChIP-Seq_imputed/H3K27ac/impute_IHECRE00000001.4_H3K27ac.pval.bw
DNAMethyl,IHECRE00000001.4,incoming/DNA_Methylation_imputed/impute_IHECRE00000001.4_DNAMethyl.bw
H3K4me1,IHECRE00000300.3,incoming/ChIP-Seq_imputed/H3K4me1/impute_IHECRE00000300.3_H3K4me1.pval.bw
"""
SAMPLE_CSV = """EpiRR,project,harmonized_sample_label,automated_harmonized_sample_ontology_term,harmonized_donor_id,epirr_id_without_version
IHECRE00000001.5,CEEHRC,Breast,breast epithelium,DONOR_A,IHECRE00000001
IHECRE00000300.3,NIH Roadmap Epigenomics,Brain,brain,,IHECRE00000300
IHECRE00000301.1,NIH Roadmap Epigenomics,Brain,brain,DONOR_B,IHECRE00000301
"""

CA_LINES = [
    "DRX013177\thg38\tHistone\tH3K27ac\tDigestive tract\tDLD-1\tPrimary Tissue=Colon\t22725271,59.1,16.9,4391\tDLD1 H3K27Ac\tsample_name=DRS012976\tdonor id=P7",
    "SRX000001\thg38\tHistone\tH2A.Z\tBlood\tK-562\tNA\t1000,90.0,5.0,10\tt\tdonor_id=NA\tindividual=IND9",
    "SRX000002\thg38\tHistone\tH3.3\tBlood\tK-562\tNA\t1000,90.0,5.0,10\tt",
    "SRX000003\thg38\tHistone\tH3K27me1\tBlood\tK-562\tNA\t1000,90.0,5.0,10\tt",
    "DRX127553\thg38\tATAC-Seq\tATAC-Seq\tLung\tA549\tNA\t22046197,96.9,46.0,13311\tt\tcell_line=A549",
    "SRX000004\thg38\tDNase-seq\tDNase-Seq\tLung\tA549\tNA\t5,1,1,1\tt",
    "DRX000744\thg38\tInput control\tInput control\tBlood\tRamos\tNA\t33590293,84.4,26.5,471\tRamos WCE",
    "SRX000005\thg19\tHistone\tH3K27ac\tBlood\tK-562\tNA\t1,1,1,1\tt",
    "SRX000006\thg38\tTFs and others\tCTCF\tBlood\tK-562\tNA\t1,1,1,1\tt",
    "SRX000007\thg38\tBisulfite-Seq\tBisulfite-Seq\tBlood\tK-562\tNA\t1,1,1,1\tt",
]


def _rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def _epi(public=frozenset({"SRR899583", "SRR899584", "SRR899600", "ENCFF001AAA", "ENCFF001CTL"})):
    return pc.parse_epiatlas(_rows(EXP_CSV), _rows(IMP_CSV), _rows(SAMPLE_CSV), set(public))


def test_panel_is_35_assays():
    assert len(pc.PANEL) == 35 and len(set(pc.PANEL)) == 35
    assert pc.CONTROL not in pc.PANEL


def test_epiatlas_observed_rows_and_columns():
    rows = _epi()
    obs = [r for r in rows if r["is_imputed"] == 0 and r["assay"] != pc.CONTROL]
    assert [r["assay"] for r in obs] == ["H3K27ac", "H3K27me3", "H3K4me3", "H3K9me3", "H3K36me3"]
    assert all(set(r) == set(pc.COLUMNS) for r in rows)
    r0 = obs[0]
    assert r0["experiment"] == "8e4a8d6c-26e0-4b8d-aa08-7875973212d4"
    assert r0["biosample_term"] == "breast epithelium"  # joined across EpiRR versions .4 / .5
    assert r0["donor"] == "DONOR_A"
    assert r0["genome"] == "GRCh38"
    assert r0["native_signal_url"] == (
        "https://ihec-epigenomes.org/epiatlas/index/base_data/IHECRE00000001/H3K27ac/"
        "ihec.chipseq.ihec-chipseq-containerv1.1.4.IHECRE00000001.4."
        "8e4a8d6c-26e0-4b8d-aa08-7875973212d4.pval.signal.bigwig")
    assert r0["native_pipeline"] == "ihec-chipseq-containerv1.1.4"


def test_epiatlas_reads_public_rule():
    obs = {r["assay"]: r for r in _epi() if r["is_imputed"] == 0 and r["assay"] != pc.CONTROL}
    assert obs["H3K27ac"]["reads_public"] == 0  # EGA: controlled
    assert obs["H3K4me3"]["reads_public"] == 1  # both SRA runs public
    assert obs["H3K9me3"]["reads_public"] == 0  # SRA run without files at ENA
    assert obs["H3K36me3"]["reads_public"] == 1  # released ENCODE file
    assert obs["H3K4me3"]["reads_accession"] == "SRR899583;SRR899584"
    assert pc.reads_public_for([], set()) == 0


def test_epiatlas_imputed_flagged_and_off_panel_dropped():
    imp = [r for r in _epi() if r["is_imputed"] == 1]
    assert [r["assay"] for r in imp] == ["H3K27ac", "H3K4me1"]  # DNAMethyl is off-panel
    assert all(r["reads_public"] == 0 and r["reads_accession"] == "" for r in imp)
    assert imp[0]["native_signal_url"].endswith(
        "/post_processed_data/ChIP-Seq_imputed/H3K27ac/impute_IHECRE00000001.4_H3K27ac.pval.bw")
    assert imp[0]["native_pipeline"].startswith("ChromImpute")


def test_epiatlas_controls_dedup_and_rna_skipped():
    rows = _epi()
    ctl = [r for r in rows if r["assay"] == pc.CONTROL]
    # IHECRE00000001.4 shares one input set across 2 marks; 300 and 301 each have their own
    assert len(ctl) == 4
    assert not any(r["assay"] == "mRNA-Seq" for r in rows)


def test_chipatlas_parse_filters_and_aliases():
    rows = [pc.parse_chipatlas_line(x) for x in CA_LINES]
    kept = [r for r in rows if r]
    assert [r["assay"] for r in kept] == ["H3K27ac", "H2AFZ", "H3F3A", "ATAC-seq", "DNase-seq",
                                          pc.CONTROL]
    r0 = kept[0]
    assert set(r0) == set(pc.COLUMNS)
    assert r0["experiment"] == "DRX013177" and r0["reads_accession"] == "DRX013177"
    assert r0["biosample_term"] == "DLD-1" and r0["donor"] == "P7"
    assert r0["native_signal_url"] == "https://chip-atlas.dbcls.jp/data/hg38/eachData/bw/DRX013177.bw"
    assert r0["reads_public"] == 1 and r0["is_imputed"] == 0 and r0["genome"] == "hg38"
    assert kept[1]["donor"] == "IND9"  # "NA" placeholder skipped, next key used
    assert kept[2]["donor"] == ""
    assert pc.ca_reads(CA_LINES[0]) == 22725271


def test_ena_and_encode_parsers():
    tsv = ("run_accession\tfastq_ftp\tsra_ftp\nSRR1\tftp.x/SRR1.fastq.gz\t\nSRR2\t\t\n"
           "SRR3\t\tftp.x/SRR3\n")
    assert pc.ena_parse_public(tsv) == {"SRR1", "SRR3"}
    js = {"@graph": [{"accession": "ENCFF1", "status": "released"},
                     {"accession": "ENCFF2", "status": "revoked"},
                     {"accession": "ENCFF3", "status": "released", "restricted": True}]}
    assert pc.encode_parse_public(js) == {"ENCFF1"}


def test_write_summarize_and_report(tmp_path):
    cat = tmp_path / "catalog"
    (cat / "epiatlas").mkdir(parents=True)
    (cat / "chipatlas").mkdir(parents=True)
    epi = _epi()
    ca = [r for r in (pc.parse_chipatlas_line(x) for x in CA_LINES) if r]
    pc.write_tsv(cat / "epiatlas" / "experiments.tsv", epi)
    pc.write_tsv(cat / "chipatlas" / "experiments.tsv", ca)
    back = pc.read_tsv(cat / "chipatlas" / "experiments.tsv")
    assert list(back[0].keys()) == pc.COLUMNS and len(back) == len(ca)
    (cat / "epiatlas" / "provenance.json").write_text(json.dumps(
        {"fetched": "t", "run_check": {}, "url_check": {}}))
    (cat / "chipatlas" / "provenance.json").write_text(json.dumps(
        {"fetched": "t", "file": {"bytes": 1, "lines": 10, "last_modified": "x"},
         "ena_check": {"sampled_experiments": 6, "with_public_files": 6}, "url_check": {}}))
    md, counts = tmp_path / "r.md", tmp_path / "c.tsv"
    pc.build_report(cat, md, counts, encode={"H3K27ac": 500, "H2AFZ": 30, "ATAC-seq": 400})
    text = md.read_text()
    assert "imputed" in text and "never enter a training or truth set" in text
    tab = list(csv.DictReader(open(counts), delimiter="\t"))
    k27 = [r for r in tab if r["source"] == "epiatlas" and r["assay"] == "H3K27ac"][0]
    assert k27["n_observed"] == "1" and k27["n_imputed"] == "1"
    assert any(r["source"] == "encode_portal" and r["assay"] == "H2AFZ" for r in tab)
    s = {(r["source"], r["assay"]): r for r in pc.summarize(epi)}
    assert s[("epiatlas", "H3K4me3")]["n_reads_public"] == 1
