#!/usr/bin/env python
"""EpiATLAS and ChIP-Atlas catalogs for the 35-assay panel (t152 programme, chunk D2, task t157).

Catalog only: it downloads metadata tables, never signal or reads. Public data only (PI ruling 5).

    python tools/onion_data/portal_catalogs.py epiatlas  --catalog <ONION>/catalog
    python tools/onion_data/portal_catalogs.py chipatlas --catalog <ONION>/catalog
    python tools/onion_data/portal_catalogs.py report    --catalog <ONION>/catalog
    python tools/onion_data/portal_catalogs.py all       --catalog <ONION>/catalog

Writes `<catalog>/{epiatlas,chipatlas}/experiments.tsv` (columns `COLUMNS`), a `provenance.json`
beside each, `<catalog>/source_report.md` and `<catalog>/counts_by_source.tsv`.

Source facts, verified 2026-11-10 against the portals (not from memory):
- EpiATLAS metadata: https://ihec-epigenomes.org/epiatlas/index/readme/ (listed in
  `info_metadata_qc.txt`). Observed files: `index/base_data/<EpiRR>/<mark>/ihec.chipseq.
  <software_version>.<epirr_id>.<uuid>.{pval.signal.bigwig,fc.signal.bigwig,raw.bigwig,noseq.bam}`,
  open HTTP listing. Imputed files (ChromImpute, -log10 p, 25 bp): `EpiATLAS_imputed_experiment_
  metadata.csv`, served under `index/post_processed_data/ChIP-Seq_imputed/<mark>/`. Genome:
  GRCh38_no_alt_analysis_set (BAM header, chr1 length 248956422; bwa). Reads: run accessions in
  `inputs` (EGAR/EGAX = EGA, JGAR = JGA: controlled; ENCFF = ENCODE; SRR = SRA).
- ChIP-Atlas: https://chip-atlas.dbcls.jp/data/metadata/experimentList.tab, no header; columns per
  https://github.com/inutano/chip-atlas/wiki: 1 experiment id (SRX/ERX/DRX), 2 genome, 3 track
  type class, 4 track type, 5 cell type class, 6 cell type, 7 cell type description, 8 processing
  log "reads,%mapped,%duplicates,peaks", 9 title, 10+ key=value metadata. bigWig at
  https://chip-atlas.dbcls.jp/data/hg38/eachData/bw/<SRX>.bw. Bowtie2 2.2.2, MACS2 2.1.0, coverage
  in RPM; all runs of an SRX concatenated. ChIP-Atlas processes public SRA only.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

COLUMNS = ["source", "experiment", "assay", "biosample_term", "donor", "reads_public",
           "reads_accession", "native_signal_url", "native_pipeline", "genome", "is_imputed"]

# The 35 assays of CANDI_STORE/eic/manifest.json::assay_vocabulary, then the control (track 35).
PANEL = ["ATAC-seq", "DNase-seq", "H2AFZ", "H2AK5ac", "H2AK9ac", "H2BK120ac", "H2BK12ac",
         "H2BK15ac", "H2BK20ac", "H2BK5ac", "H3F3A", "H3K14ac", "H3K18ac", "H3K23ac", "H3K23me2",
         "H3K27ac", "H3K27me3", "H3K36me3", "H3K4ac", "H3K4me1", "H3K4me2", "H3K4me3", "H3K56ac",
         "H3K79me1", "H3K79me2", "H3K9ac", "H3K9me1", "H3K9me2", "H3K9me3", "H3T11ph", "H4K12ac",
         "H4K20me1", "H4K5ac", "H4K8ac", "H4K91ac"]
CONTROL = "chipseq-control"
PANEL_SET = set(PANEL)

EPI_BASE = "https://ihec-epigenomes.org/epiatlas/index"
EPI_META = {
    "experiments": f"{EPI_BASE}/readme/EpiATLAS_experiment_metadata.csv",
    "imputed": f"{EPI_BASE}/readme/EpiATLAS_imputed_experiment_metadata.csv",
    "samples": f"{EPI_BASE}/readme/IHEC_sample_metadata_harmonization.v2.0_extended.csv",
}
EPI_GENOME = "GRCh38"
EPI_IMPUTED_PIPELINE = "ChromImpute (-log10 p-value, 25 bp)"

CA_LIST = "https://chip-atlas.dbcls.jp/data/metadata/experimentList.tab"
CA_BW = "https://chip-atlas.dbcls.jp/data/hg38/eachData/bw/{}.bw"
CA_PIPELINE = "ChIP-Atlas: Bowtie2 2.2.2, MACS2 2.1.0, RPM coverage, runs concatenated"
CA_GENOME = "hg38"
# ChIP-Atlas track type -> panel name, where the spelling differs.
CA_ALIAS = {"H2A.Z": "H2AFZ", "H3.3": "H3F3A", "ATAC-Seq": "ATAC-seq", "DNase-Seq": "DNase-seq",
            "DNase-seq": "DNase-seq", "Input control": CONTROL}
CA_CLASSES = {"Histone", "ATAC-Seq", "DNase-seq", "Input control"}
# metadata keys (lower case) that name a donor, in order of preference
DONOR_KEYS = ["donor_id", "donor id", "donor", "individual", "subject id", "patient id", "patient"]

ENA_SEARCH = "https://www.ebi.ac.uk/ena/portal/api/search"
ENCODE_SEARCH = "https://www.encodeproject.org/search/"
UA = {"User-Agent": "onion-catalog/1.0 (research metadata crawl)"}


# ---------------------------------------------------------------- network helpers
def http_get(url: str, data: bytes | None = None, headers: dict | None = None,
             tries: int = 4, timeout: int = 300) -> bytes:
    h = dict(UA, **(headers or {}))
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=h),
                                        timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - retry any transport error
            if k == tries - 1:
                raise
            print(f"[retry {k + 1}] {url[:120]}: {e}", file=sys.stderr)
            time.sleep(5 * (k + 1))
    raise AssertionError


def http_head_ok(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def ena_public_runs(runs: list[str], chunk: int = 100) -> set[str]:
    """Runs (SRR/ERR/DRR) for which ENA serves a fastq or an SRA file."""
    public = set()
    for i in range(0, len(runs), chunk):
        part = runs[i:i + chunk]
        q = " OR ".join(f'run_accession="{r}"' for r in part)
        body = urllib.parse.urlencode({"result": "read_run", "query": q, "format": "tsv",
                                       "fields": "run_accession,fastq_ftp,sra_ftp"}).encode()
        txt = http_get(ENA_SEARCH, data=body,
                       headers={"Content-Type": "application/x-www-form-urlencoded"}).decode()
        public |= ena_parse_public(txt)
        time.sleep(0.3)
    return public


def ena_parse_public(tsv_text: str, key: str = "run_accession") -> set[str]:
    out = set()
    for row in csv.DictReader(io.StringIO(tsv_text), delimiter="\t"):
        if (row.get("fastq_ftp") or "").strip() or (row.get("sra_ftp") or "").strip():
            out.add(row[key])
    return out


def ena_public_experiments(exps: list[str], chunk: int = 100) -> set[str]:
    public = set()
    for i in range(0, len(exps), chunk):
        q = " OR ".join(f'experiment_accession="{e}"' for e in exps[i:i + chunk])
        body = urllib.parse.urlencode({"result": "read_run", "query": q, "format": "tsv",
                                       "fields": "experiment_accession,fastq_ftp,sra_ftp"}).encode()
        txt = http_get(ENA_SEARCH, data=body,
                       headers={"Content-Type": "application/x-www-form-urlencoded"}).decode()
        public |= ena_parse_public(txt, key="experiment_accession")
        time.sleep(0.3)
    return public


def encode_public_files(files: list[str], chunk: int = 50) -> set[str]:
    """ENCODE files that are released or archived and not restricted."""
    public = set()
    for i in range(0, len(files), chunk):
        qs = [("type", "File"), ("limit", "all"), ("format", "json"), ("field", "accession"),
              ("field", "status"), ("field", "restricted"), ("field", "no_file_available")]
        qs += [("accession", a) for a in files[i:i + chunk]]
        js = json.loads(http_get(ENCODE_SEARCH + "?" + urllib.parse.urlencode(qs),
                                 headers={"Accept": "application/json"}))
        public |= encode_parse_public(js)
        time.sleep(0.2)
    return public


def encode_parse_public(js: dict) -> set[str]:
    return {g["accession"] for g in js.get("@graph", [])
            if g.get("status") in ("released", "archived") and not g.get("restricted")
            and not g.get("no_file_available")}


def encode_counts() -> dict[str, int]:
    """Released human experiments per panel assay on the ENCODE portal (facet counts)."""
    out = {}
    base = [("type", "Experiment"), ("status", "released"), ("limit", "0"), ("format", "json"),
            ("replicates.library.biosample.donor.organism.scientific_name", "Homo sapiens")]
    for title in ("ATAC-seq", "DNase-seq"):
        js = json.loads(http_get(ENCODE_SEARCH + "?" + urllib.parse.urlencode(
            base + [("assay_title", title)]), headers={"Accept": "application/json"}))
        out[title] = int(js.get("total", 0))
    js = json.loads(http_get(ENCODE_SEARCH + "?" + urllib.parse.urlencode(
        base + [("assay_title", "Histone ChIP-seq")]), headers={"Accept": "application/json"}))
    for f in js.get("facets", []):
        if f.get("field") == "target.label":
            for t in f.get("terms", []):
                if t["key"] in PANEL_SET:
                    out[t["key"]] = int(t["doc_count"])
    js = json.loads(http_get(ENCODE_SEARCH + "?" + urllib.parse.urlencode(
        base + [("assay_title", "Control ChIP-seq")]), headers={"Accept": "application/json"}))
    out[CONTROL] = int(js.get("total", 0))
    return out


# ---------------------------------------------------------------- EpiATLAS
def run_accessions(field: str) -> list[str]:
    return [a.strip() for a in (field or "").split(";") if a.strip() and a.strip() != "nan"]


def reads_public_for(runs: list[str], public_runs: set[str]) -> int:
    """1 when every run is public: EGA and JGA are controlled; ENCFF and SRA runs are public only
    if the archive check put them in `public_runs`."""
    if not runs:
        return 0
    for r in runs:
        if r.startswith(("EGA", "JGA")) or r not in public_runs:
            return 0
    return 1


def epi_signal_url(row: dict) -> str:
    epi = row["epirr_id"].split(".")[0]
    return (f"{EPI_BASE}/base_data/{epi}/{row['experiment_type']}/ihec.chipseq."
            f"{row['software_version']}.{row['epirr_id']}.{row['uuid']}.pval.signal.bigwig")


def epi_imputed_url(row: dict) -> str:
    return (f"{EPI_BASE}/post_processed_data/ChIP-Seq_imputed/{row['mark']}/"
            f"{row['data_file_path'].rsplit('/', 1)[-1]}")


def parse_epiatlas(exp_rows: list[dict], imp_rows: list[dict], sample_rows: list[dict],
                   public_runs: set[str]) -> list[dict]:
    """Panel rows: observed ChIP-Seq marks, one control row per distinct input set, imputed
    marks flagged is_imputed=1. Sample metadata joins on the EpiRR id without version."""
    samples = {s["epirr_id_without_version"] or s["EpiRR"].split(".")[0]: s for s in sample_rows}

    def term_donor(epirr: str) -> tuple[str, str]:
        s = samples.get(epirr.split(".")[0], {})
        term = s.get("automated_harmonized_sample_ontology_term") or s.get("harmonized_sample_label", "")
        return term, s.get("harmonized_donor_id", "")

    out, ctl_seen = [], set()
    for r in exp_rows:
        if r["assay_type"] != "ChIP-Seq" or r["experiment_type"] not in PANEL_SET:
            continue
        term, donor = term_donor(r["epirr_id"])
        runs = run_accessions(r["inputs"])
        out.append(dict(source="epiatlas", experiment=r["uuid"], assay=r["experiment_type"],
                        biosample_term=term, donor=donor,
                        reads_public=reads_public_for(runs, public_runs),
                        reads_accession=";".join(runs), native_signal_url=epi_signal_url(r),
                        native_pipeline=r["software_version"], genome=EPI_GENOME, is_imputed=0))
        ctl = run_accessions(r.get("inputs_ctl", ""))
        key = (r["epirr_id"], tuple(ctl))
        if ctl and key not in ctl_seen:
            ctl_seen.add(key)
            h = hashlib.sha1(";".join(ctl).encode()).hexdigest()[:10]
            out.append(dict(source="epiatlas", experiment=f"ctl.{r['epirr_id']}.{h}", assay=CONTROL,
                            biosample_term=term, donor=donor,
                            reads_public=reads_public_for(ctl, public_runs),
                            reads_accession=";".join(ctl), native_signal_url="",
                            native_pipeline=r["software_version"], genome=EPI_GENOME, is_imputed=0))
    for r in imp_rows:
        if r["mark"] not in PANEL_SET:
            continue
        term, donor = term_donor(r["epirr_id"])
        out.append(dict(source="epiatlas", experiment=f"imputed.{r['epirr_id']}.{r['mark']}",
                        assay=r["mark"], biosample_term=term, donor=donor, reads_public=0,
                        reads_accession="", native_signal_url=epi_imputed_url(r),
                        native_pipeline=EPI_IMPUTED_PIPELINE, genome=EPI_GENOME, is_imputed=1))
    return out


def read_csv_bytes(b: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(b.decode("utf-8"))))


def build_epiatlas(cat: Path, n_head: int = 200) -> None:
    d = cat / "epiatlas"
    (d / "source").mkdir(parents=True, exist_ok=True)
    prov = {"fetched": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "files": {}}
    raw = {}
    for k, url in EPI_META.items():
        b = http_get(url)
        (d / "source" / url.rsplit("/", 1)[-1]).write_bytes(b)
        prov["files"][k] = {"url": url, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
        raw[k] = read_csv_bytes(b)
    exps = [r for r in raw["experiments"] if r["assay_type"] == "ChIP-Seq"]
    runs = sorted({a for r in exps for f in ("inputs", "inputs_ctl") for a in run_accessions(r[f])})
    sra = [a for a in runs if re.match(r"^[SED]RR\d+$", a)]
    encff = [a for a in runs if a.startswith("ENCFF")]
    public = ena_public_runs(sra) | encode_public_files(encff)
    prov["run_check"] = {"sra_runs": len(sra), "sra_public": len(public & set(sra)),
                         "encode_files": len(encff), "encode_public": len(public & set(encff)),
                         "ega_or_jga_runs": sum(a.startswith(("EGA", "JGA")) for a in runs)}
    rows = parse_epiatlas(raw["experiments"], raw["imputed"], raw["samples"], public)
    write_tsv(d / "experiments.tsv", rows)
    prov["url_check"] = head_check(rows, n_head)
    (d / "provenance.json").write_text(json.dumps(prov, indent=1))
    print(f"epiatlas: {len(rows)} rows -> {d / 'experiments.tsv'}; {json.dumps(prov['run_check'])}")


# ---------------------------------------------------------------- ChIP-Atlas
def ca_donor(meta_fields: list[str]) -> str:
    kv = {}
    for f in meta_fields:
        if "=" in f:
            k, v = f.split("=", 1)
            kv.setdefault(k.strip().lower(), v.strip())
    for k in DONOR_KEYS:
        v = kv.get(k, "")
        if v and v.upper() not in ("NA", "N/A", "NONE", "UNKNOWN", "NOT APPLICABLE", "MISSING"):
            return v
    return ""


def parse_chipatlas_line(line: str) -> dict | None:
    """One experimentList.tab line -> a panel row, or None when off-panel or not hg38."""
    f = line.rstrip("\n").split("\t")
    if len(f) < 9 or f[1] != CA_GENOME or f[2] not in CA_CLASSES:
        return None
    # the class decides for non-histone rows; the track type names the histone mark
    assay = CA_ALIAS[f[2]] if f[2] != "Histone" else CA_ALIAS.get(f[3], f[3])
    if assay not in PANEL_SET and assay != CONTROL:
        return None
    return dict(source="chipatlas", experiment=f[0], assay=assay, biosample_term=f[5],
                donor=ca_donor(f[9:]), reads_public=1, reads_accession=f[0],
                native_signal_url=CA_BW.format(f[0]), native_pipeline=CA_PIPELINE,
                genome=CA_GENOME, is_imputed=0)


def ca_reads(line: str) -> int | None:
    f = line.split("\t")
    try:
        return int(f[7].split(",")[0])
    except (IndexError, ValueError):
        return None


def build_chipatlas(cat: Path, n_head: int = 200, n_ena: int = 300) -> None:
    d = cat / "chipatlas"
    d.mkdir(parents=True, exist_ok=True)
    sha, nbytes, nlines, rows, depth = hashlib.sha256(), 0, 0, [], {}
    enc_like = set()
    req = urllib.request.Request(CA_LIST, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r:
        last_mod = r.headers.get("Last-Modified", "")
        for raw in r:
            sha.update(raw)
            nbytes += len(raw)
            nlines += 1
            line = raw.decode("utf-8", "replace")
            row = parse_chipatlas_line(line)
            if row is None:
                continue
            rows.append(row)
            depth[row["experiment"]] = ca_reads(line)
            if re.search(r"ENC(SR|BS|LB|DO|AB)\d{3}[A-Z]{3}", line):
                enc_like.add(row["experiment"])
    write_tsv(d / "experiments.tsv", rows)
    with open(d / "aux.tsv", "w") as fh:  # per-experiment depth and ENCODE-accession flag
        fh.write("experiment\treads\tmentions_encode_accession\n")
        for row in rows:
            e = row["experiment"]
            fh.write(f"{e}\t{'' if depth[e] is None else depth[e]}\t{int(e in enc_like)}\n")
    rng = random.Random(0)
    sample = rng.sample([r["experiment"] for r in rows], min(n_ena, len(rows)))
    pub = ena_public_experiments(sample)
    prov = {"fetched": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "file": {"url": CA_LIST, "bytes": nbytes, "lines": nlines, "last_modified": last_mod,
                     "sha256": sha.hexdigest()},
            "ena_check": {"sampled_experiments": len(sample), "with_public_files": len(pub),
                          "seed": 0},
            "url_check": head_check(rows, n_head)}
    (d / "provenance.json").write_text(json.dumps(prov, indent=1))
    print(f"chipatlas: {len(rows)} of {nlines} lines -> {d / 'experiments.tsv'}; "
          f"ENA {len(pub)}/{len(sample)}")


# ---------------------------------------------------------------- shared
def write_tsv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in COLUMNS})


def read_tsv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def head_check(rows: list[dict], n: int) -> dict:
    """HEAD a seeded sample of signal URLs per is_imputed value; report how many answer 200."""
    out = {}
    rng = random.Random(0)
    for imp in ("0", "1"):
        urls = [r["native_signal_url"] for r in rows
                if str(r["is_imputed"]) == imp and r["native_signal_url"]]
        if not urls:
            continue
        s = rng.sample(urls, min(n, len(urls)))
        failed = [u for u in s if not http_head_ok(u)]
        out[f"is_imputed={imp}"] = {"sampled": len(s), "http_200": len(s) - len(failed),
                                    "failed_examples": failed[:3]}
    return out


def summarize(rows: list[dict]) -> list[dict]:
    """counts_by_source rows: per source and assay, experiments observed and imputed, public reads,
    distinct biosample terms and donors, and anisogenic groups (same term, at least 2 donors)."""
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["source"], r["assay"])].append(r)
    out = []
    order = {a: i for i, a in enumerate(PANEL + [CONTROL])}
    for (src, assay), rs in sorted(by.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 99))):
        obs = [r for r in rs if str(r["is_imputed"]) == "0"]
        terms = collections.defaultdict(set)
        nexp = collections.Counter()
        for r in obs:
            nexp[r["biosample_term"]] += 1
            if r["donor"]:
                terms[r["biosample_term"]].add(r["donor"])
        out.append(dict(source=src, assay=assay, n_observed=len(obs),
                        n_imputed=len(rs) - len(obs),
                        n_reads_public=sum(str(r["reads_public"]) == "1" for r in obs),
                        n_biosample_terms=len(nexp),
                        n_donors=len({r["donor"] for r in obs if r["donor"]}),
                        n_terms_with_2plus_experiments=sum(v >= 2 for v in nexp.values()),
                        n_terms_with_2plus_donors=sum(len(v) >= 2 for v in terms.values())))
    return out


# ---------------------------------------------------------------- report
def _pct(a: int, b: int) -> str:
    return f"{a} of {b} ({100.0 * a / b:.1f}%)" if b else f"{a} of 0"


def build_report(cat: Path, out_md: Path, out_counts: Path, encode: dict | None = None) -> None:
    epi = read_tsv(cat / "epiatlas" / "experiments.tsv")
    ca = read_tsv(cat / "chipatlas" / "experiments.tsv")
    pe = json.loads((cat / "epiatlas" / "provenance.json").read_text())
    pc = json.loads((cat / "chipatlas" / "provenance.json").read_text())
    if encode is None:
        try:
            encode = encode_counts()
        except Exception as e:  # noqa: BLE001
            print(f"encode counts unavailable: {e}", file=sys.stderr)
            encode = {}
    counts = summarize(epi + ca)
    for a in PANEL + [CONTROL]:
        if a in encode:
            counts.append(dict(source="encode_portal", assay=a, n_observed=encode[a], n_imputed=0,
                               n_reads_public="", n_biosample_terms="", n_donors="",
                               n_terms_with_2plus_experiments="", n_terms_with_2plus_donors=""))
    keys = list(counts[0].keys())
    with open(out_counts, "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for c in counts:
            fh.write("\t".join(str(c[k]) for k in keys) + "\n")

    epi_obs = [r for r in epi if r["is_imputed"] == "0" and r["assay"] != CONTROL]
    epi_ctl = [r for r in epi if r["assay"] == CONTROL]
    epi_imp = [r for r in epi if r["is_imputed"] == "1"]
    epi_pub = [r for r in epi_obs if r["reads_public"] == "1"]
    enc_reads = [r for r in epi_obs if r["reads_accession"].startswith("ENCFF")]
    sra_reads = [r for r in epi_obs if re.match(r"^[SED]RR", r["reads_accession"])]
    ctrl_reads = [r for r in epi_obs if r["reads_accession"].startswith(("EGA", "JGA"))]
    ca_obs = [r for r in ca if r["assay"] != CONTROL]
    ca_ctl = [r for r in ca if r["assay"] == CONTROL]
    aux = {}
    p = cat / "chipatlas" / "aux.tsv"
    if p.exists():
        aux = {r["experiment"]: r for r in read_tsv(p)}
    ca_enc = sum(aux.get(r["experiment"], {}).get("mentions_encode_accession") == "1" for r in ca_obs)
    ca_depth = sorted(int(aux[r["experiment"]]["reads"]) for r in ca_obs
                      if aux.get(r["experiment"], {}).get("reads"))

    def q(xs, f):
        return xs[int(f * (len(xs) - 1))] if xs else 0

    epi_by = collections.Counter(r["assay"] for r in epi_obs)
    ca_by = collections.Counter(r["assay"] for r in ca_obs)
    rare = [a for a in PANEL if a not in ("ATAC-seq", "DNase-seq")
            and 0 < encode.get(a, 0) < 50]
    ca_rare = sum(ca_by.get(a, 0) for a in rare)
    enc_rare = sum(encode.get(a, 0) for a in rare)
    panel_hit_ca = sum(1 for a in PANEL if ca_by.get(a, 0) > 0)

    g = collections.defaultdict(set)
    for r in epi_obs:
        if r["donor"]:
            g[(r["biosample_term"], r["assay"])].add(r["donor"])
    epi_aniso, epi_groups = sum(len(v) >= 2 for v in g.values()), len(g)
    ca_multi = collections.Counter((r["biosample_term"], r["assay"]) for r in ca_obs)
    ca_multi2 = sum(v >= 2 for v in ca_multi.values())
    rc = pe.get("run_check", {})
    L = []
    L.append("# EpiATLAS and ChIP-Atlas: what each adds over ENCODE\n")
    L.append(f"Task t157 (chunk D2 of t152). Catalog only: metadata tables, no signal and no reads. "
             f"Public data only (PI ruling 5). Fetched {pe.get('fetched', '')} (EpiATLAS) and "
             f"{pc.get('fetched', '')} (ChIP-Atlas) on Nibi compute nodes.\n")
    L.append("## Sources and access\n")
    L.append("| | EpiATLAS (IHEC) | ChIP-Atlas |\n|---|---|---|")
    L.append(f"| metadata | `EpiATLAS_experiment_metadata.csv`, `EpiATLAS_imputed_experiment_metadata.csv`, "
             f"`IHEC_sample_metadata_harmonization.v2.0_extended.csv` ({EPI_BASE}/readme/) | "
             f"`experimentList.tab` ({pc['file']['bytes']} bytes, {pc['file']['lines']} lines, "
             f"Last-Modified {pc['file']['last_modified']}) |")
    L.append(f"| genome | GRCh38 no-alt analysis set (bwa; read from a `noseq.bam` header) | hg38 rows only |")
    L.append("| panel assays | 6 histone marks + input controls | histone marks, ATAC-seq, DNase-seq, input controls |")
    L.append("| processing | IHEC `integrative_analysis_chip` container (ENCODE-style; MACS2) | Bowtie2 2.2.2, MACS2 2.1.0, RPM coverage, all runs of an SRX concatenated |")
    L.append("| raw reads | EGA / JGA (controlled) for most centres; ENCODE (ENCFF) and SRA runs for ENCODE and Roadmap data | public SRA / ENA / DDBJ by construction |")
    L.append("| processed files | open HTTP: `pval.signal.bigwig`, `fc.signal.bigwig`, `raw.bigwig`, peaks, `qc.json`, and `noseq.bam` (alignments without sequences) | open HTTP: per-SRX bigWig and peaks |")
    L.append("| imputed tracks | yes, ChromImpute -log10 p at 25 bp, in a separate directory and table | none |\n")
    L.append("## Counts\n")
    L.append(f"- EpiATLAS observed panel experiments: {len(epi_obs)} "
             f"({', '.join(f'{a} {n}' for a, n in sorted(epi_by.items()))}), plus {len(epi_ctl)} distinct input-control sets.")
    L.append(f"- EpiATLAS imputed panel tracks: {len(epi_imp)}. They are flagged `is_imputed = 1` and "
             f"never enter a training or truth set. Share of all EpiATLAS panel rows that are imputed: "
             f"{_pct(len(epi_imp), len(epi_imp) + len(epi_obs))}.")
    L.append(f"- EpiATLAS observed experiments with public reads: {_pct(len(epi_pub), len(epi_obs))}. "
             f"By read source: ENCODE files {len(enc_reads)}, SRA runs {len(sra_reads)}, EGA/JGA {len(ctrl_reads)}. "
             f"Archive check: {rc.get('sra_public', '?')} of {rc.get('sra_runs', '?')} SRA runs had a file at ENA; "
             f"{rc.get('encode_public', '?')} of {rc.get('encode_files', '?')} ENCODE files were released and unrestricted.")
    L.append(f"- ChIP-Atlas panel experiments (hg38): {len(ca_obs)} non-control, {len(ca_ctl)} input controls; "
             f"{panel_hit_ca} of 35 panel assays present. Reads public by construction; an ENA spot check of "
             f"{pc['ena_check']['sampled_experiments']} random panel SRX (seed 0) found public files for "
             f"{pc['ena_check']['with_public_files']}.")
    L.append(f"- ChIP-Atlas rows that carry an ENCODE accession (ENCSR/ENCBS/ENCLB/ENCDO/ENCAB) in their metadata: "
             f"{_pct(ca_enc, len(ca_obs))}. This is a lower bound on the ENCODE overlap; ENCODE data deposited "
             f"through GEO without these accessions is not caught.")
    if ca_depth:
        L.append(f"- ChIP-Atlas read depth (reads before mapping, processing log): median {q(ca_depth, .5):,}, "
                 f"10th percentile {q(ca_depth, .1):,}, 90th percentile {q(ca_depth, .9):,}.")
    if encode:
        L.append(f"- ENCODE portal, released human experiments in the same query day: "
                 + ", ".join(f"{a} {encode[a]}" for a in PANEL + [CONTROL] if a in encode) + ".")
    for s in ("is_imputed=0", "is_imputed=1"):
        for name, pv in (("EpiATLAS", pe), ("ChIP-Atlas", pc)):
            u = pv.get("url_check", {}).get(s)
            if u:
                L.append(f"- {name} signal URL check ({s}): {u['http_200']} of {u['sampled']} sampled URLs answered HTTP 200.")
    L.append("\nPer-assay numbers are in `counts_by_source.tsv`.\n")
    L.append("## What each source adds over ENCODE, per goal\n")
    L.append("### EpiATLAS\n")
    L.append(f"- **Measurement counterfactuals.** {len(enc_reads) + len(sra_reads)} observed experiments are "
             f"ENCODE or Roadmap data re-run through the IHEC pipeline. Each gives a pipeline counterfactual: the same reads, "
             f"two pipelines (ENCODE's own processing against the IHEC container). This needs no new reads. "
             f"The other {len(ctrl_reads)} experiments have controlled reads, so no FASTQ-level arm can be built from them. "
             f"Their `noseq.bam` files are public and keep positions and flags (read sequences removed), so counts, depth thinning "
             f"and pseudoreplicate halves can be built from alignments without controlled access.")
    L.append(f"- **Balance along the measurement axis.** The non-ENCODE centres (BLUEPRINT, CEEHRC, DEEP, GIS, AMED-CREST, "
             f"KNIH, EpiHK) add labs and protocols that ENCODE lacks, for the 6 core marks only. All of it is one pipeline, so it "
             f"adds lab and protocol spread at a fixed pipeline. It adds nothing for ATAC-seq, DNase-seq or the other 29 histone marks.")
    L.append(f"- **Replicate rungs.** EpiATLAS has one experiment per epigenome and mark, so no isogenic or technical rung. "
             f"(biosample term, mark) groups with at least 2 donors (anisogenic candidates): {epi_aniso} of {epi_groups}. "
             f"`noseq.bam` has no read groups, so no sequencing-run rung from public files.")
    L.append(f"- **Donor.** {_pct(sum(1 for r in epi_obs if r['donor']), len(epi_obs))} observed experiments carry a harmonized donor id (decision D2).\n")
    L.append("### ChIP-Atlas\n")
    L.append(f"- **Measurement counterfactuals.** All reads are public, so every FASTQ-level arm (depth, crop, single-end, "
             f"dedup, mapq) and the uniform ENCODE-DCC base can be built for any experiment. The ChIP-Atlas track itself is a "
             f"pipeline counterfactual (Bowtie2 + RPM, runs concatenated) against the uniform base. For the at least {ca_enc} rows "
             f"that are ENCODE experiments, it is a second-pipeline view of ENCODE reads.")
    L.append(f"- **Balance along the measurement axis.** It brings many labs, read lengths, platforms and depths that ENCODE "
             f"does not have. For the {len(rare)} panel histone marks with fewer than 50 released ENCODE experiments "
             f"({', '.join(rare)}), ChIP-Atlas holds {ca_rare} experiments against ENCODE's {enc_rare}.")
    L.append(f"- **Replicate rungs.** ChIP-Atlas does not record replicate structure. (cell type, assay) groups with at least "
             f"2 experiments: {ca_multi2} of {len(ca_multi)}. These are candidates for the isogenic, anisogenic and "
             f"other-experiment (cross-lab) rungs, an upper bound; GEO/SRA sample metadata must confirm each pair. The cross-lab rung "
             f"is the one EIC cannot populate. A sequencing-run rung needs the per-run SRA files, which are public.")
    L.append(f"- **Donor.** {_pct(sum(1 for r in ca_obs if r['donor']), len(ca_obs))} experiments carry a donor field in their "
             f"free-text metadata (keys {', '.join(DONOR_KEYS)}); the values are submitter labels, not harmonized ids.\n")
    L.append("## Caveats\n")
    L.append("- ChIP-Atlas cell-type names are ChIP-Atlas's own curation, not ontology terms; joining to ENCODE terms needs a map.")
    L.append("- EpiATLAS experiment metadata is dated 2024-03 (Last-Modified); the sample harmonization v2.0 is 2025-10.")
    L.append("- No controlled-access application was made (ruling 5).")
    out_md.write_text("\n".join(L) + "\n")
    print(f"report -> {out_md}; counts -> {out_counts}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["epiatlas", "chipatlas", "report", "all"])
    ap.add_argument("--catalog", required=True, type=Path)
    a = ap.parse_args(argv)
    a.catalog.mkdir(parents=True, exist_ok=True)
    if a.cmd in ("epiatlas", "all"):
        build_epiatlas(a.catalog)
    if a.cmd in ("chipatlas", "all"):
        build_chipatlas(a.catalog)
    if a.cmd in ("report", "all"):
        build_report(a.catalog, a.catalog / "source_report.md", a.catalog / "counts_by_source.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
