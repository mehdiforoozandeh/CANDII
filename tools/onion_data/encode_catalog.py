"""ENCODE catalog and replicate-ladder census (t152 programme, chunk D1, crux task t156).

Crawls the ENCODE portal REST API (JSON) for the 35-assay EIC panel (ATAC-seq, DNase-seq and 33
histone marks, as Histone ChIP-seq or Mint-ChIP-seq) in human, status released, plus every ChIP
control that those experiments name in `possible_controls`, plus every EIC experiment by
accession (any status). It writes the catalog tables and two censuses of replicate pairs:

    <out>/experiments.tsv libraries.tsv files.tsv donors.tsv   catalog tables
    <out>/ladder_census.tsv   pairs per rung and assay, whole portal (status released)
    <out>/eic_census.tsv      the same, restricted to the EIC experiments (any status)

Usage (on the cluster; compute nodes have internet):

    python tools/onion_data/encode_catalog.py crawl  --out DIR --cache DIR --eic-list TSV
    python tools/onion_data/encode_catalog.py census --out DIR --eic-list TSV [--summary-dir D]

`crawl` fetches, writes the tables and runs `census`. `census` recomputes the censuses from the
tables only. The crawl is polite and resumable: at most `--rate` requests per second (default 5;
the portal's documented limit is 10), a generic User-Agent, retries with exponential backoff
(Retry-After honoured), and every response cached as gzip JSON under `--cache` and never
refetched (skip-if-exists).

Rungs, in ladder order. A unit counts only when it has reads: a fastq file with status released
or archived.
- pseudoreplicate: one pair per library with reads (the two halves of its reads).
- sequencing: pairs of sequencing runs of one library. A run is one fastq (mates collapsed via
  `paired_with`) keyed by the (flowcell, lane) set of `flowcell_details`; files that share a key
  are one run. `n_pairs` counts only runs with a known, distinct key; `n_pairs_loose` also
  counts each unlabelled fastq as its own run.
- technical: pairs of libraries with the same biological replicate number in one experiment.
- isogenic: pairs of biological replicates of one experiment whose donor sets are equal.
- anisogenic: pairs of biological replicates of one experiment whose donor sets are disjoint.
- other_experiment: pairs of experiments with the same assay and the same biosample term.
- other_cell_type: pairs of experiments with the same assay and different biosample terms.
`n_cells` counts distinct biosample terms and `n_experiments` the experiments that contribute at
least one pair; `fastq_gb` sums the fastq bytes (1e9) of those experiments.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations

PORTAL = "https://www.encodeproject.org"
USER_AGENT = "onion-catalog/1.0 (python-urllib; research metadata crawler)"
HUMAN = "replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens"

PANEL_HISTONES = (
    "H2AFZ H2AK5ac H2AK9ac H2BK120ac H2BK12ac H2BK15ac H2BK20ac H2BK5ac H3F3A H3K14ac H3K18ac "
    "H3K23ac H3K23me2 H3K27ac H3K27me3 H3K36me3 H3K4ac H3K4me1 H3K4me2 H3K4me3 H3K56ac H3K79me1 "
    "H3K79me2 H3K9ac H3K9me1 H3K9me2 H3K9me3 H3T11ph H4K12ac H4K20me1 H4K5ac H4K8ac H4K91ac"
).split()
PANEL = ["ATAC-seq", "DNase-seq"] + PANEL_HISTONES
CONTROL = "chipseq-control"
CENSUS_ASSAYS = PANEL + [CONTROL]
RUNGS = ["pseudoreplicate", "sequencing", "technical", "isogenic", "anisogenic",
         "other_experiment", "other_cell_type"]
READ_STATUSES = {"released", "archived"}

# Search listings: (cache name, query). Histone marks are filtered to the panel after the fetch.
SEARCHES = [
    ("atac", "assay_title=ATAC-seq"),
    ("dnase", "assay_title=DNase-seq"),
    ("histone", "assay_title=Histone+ChIP-seq"),
    ("mint", "assay_title=Mint-ChIP-seq"),
]

EXPERIMENT_COLS = ["experiment", "assay", "target", "biosample_term", "biosample_type",
                   "replication_type", "lab", "status", "date_released", "internal_tags",
                   "control_experiments", "assay_title", "biosample_term_id", "organism"]
LIBRARY_COLS = ["experiment", "library", "biosample", "donor", "bio_rep", "tech_rep", "antibody"]
FILE_COLS = ["file", "experiment", "library", "file_format", "output_type", "assembly",
             "run_type", "read_length", "platform", "paired_with", "flowcell", "lane",
             "bio_reps", "tech_reps", "pipeline", "pipeline_version", "mapped_reads",
             "file_size", "md5", "url", "nsc", "rsc", "nrf", "pbc1", "pbc2", "frip", "frag_len",
             "status", "paired_end", "read_count"]
DONOR_COLS = ["donor", "sex", "age", "age_units", "life_stage", "health_status", "organism"]
CENSUS_COLS = ["rung", "assay", "n_pairs", "n_cells", "n_experiments", "fastq_gb",
               "n_pairs_loose"]


# ---------------------------------------------------------------- small helpers

def _acc(x) -> str:
    """Accession of an embedded object or of an @id string like '/files/ENCFF1/'."""
    if isinstance(x, dict):
        return str(x.get("accession") or _acc(x.get("@id", "")))
    if isinstance(x, str):
        parts = [p for p in x.split("/") if p]
        return parts[-1] if parts else ""
    return ""


def _s(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return ";".join(_s(v) for v in x)
    return str(x)


def _title(x) -> str:
    if isinstance(x, dict):
        return _s(x.get("title") or x.get("label") or x.get("name") or _acc(x))
    return _s(x)


def assay_of(exp: dict) -> str:
    """Panel assay name of an experiment, CONTROL for a ChIP control, '' when outside the panel."""
    title = exp.get("assay_title") or ""
    if title.startswith("Control"):
        return CONTROL
    if title in ("ATAC-seq", "DNase-seq"):
        return title
    if exp.get("assay_term_name") in ("ChIP-seq", "Mint-ChIP-seq") or "ChIP" in title:
        target = exp.get("target")
        label = target.get("label") if isinstance(target, dict) else _acc(target).split("-")[0]
        return label if label in PANEL_HISTONES else ""
    return ""


def _qc(file: dict) -> dict:
    """QC numbers from a file's embedded quality_metrics (empty when they are not embedded)."""
    out: dict = {}
    mapped_filtered = mapped_any = None
    for q in file.get("quality_metrics") or []:
        if not isinstance(q, dict):
            continue
        for key, col in (("NSC", "nsc"), ("RSC", "rsc"), ("NRF", "nrf"), ("PBC1", "pbc1"),
                         ("PBC2", "pbc2"), ("frip", "frip"),
                         ("estimated_fragment_len", "frag_len")):
            if key in q and col not in out and not isinstance(q[key], (dict, list)):
                out[col] = q[key]
        if "mapped_reads" in q:
            if q.get("processing_stage") == "filtered" and mapped_filtered is None:
                mapped_filtered = q["mapped_reads"]
            elif mapped_any is None:
                mapped_any = q["mapped_reads"]
    out["mapped_reads"] = mapped_filtered if mapped_filtered is not None else mapped_any
    return out


def _pipeline(file: dict, keyword: str = "", unreplicated: bool = False) -> tuple[str, str]:
    """(pipeline title, pipeline version). A step serves several pipelines (TF and histone,
    replicated and not), so the title is the first that contains `keyword` and whose
    '(unreplicated)' suffix matches the experiment; else the first that contains `keyword`."""
    name = version = ""
    asv = file.get("analysis_step_version")
    if isinstance(asv, dict):
        step = asv.get("analysis_step")
        if isinstance(step, dict):
            titles = [_s(p.get("title")) for p in step.get("pipelines") or [] if isinstance(p, dict)]
            hits = [t for t in titles if keyword.lower() in t.lower()] or titles
            best = [t for t in hits if ("unreplicated" in t.lower()) == unreplicated]
            name = (best or hits or [""])[0]
    for a in file.get("analyses") or []:
        if isinstance(a, dict) and a.get("pipeline_version"):
            version = _s(a.get("pipeline_version"))
            if not name:
                name = _s(a.get("title"))
            break
    return name, version


# ---------------------------------------------------------------- parser

def parse_experiment(exp: dict) -> dict:
    """One embedded experiment JSON -> {'experiment': row, 'libraries': [...], 'files': [...],
    'donors': [...]} with the column sets of the catalog tables."""
    acc = exp["accession"]
    onto = exp.get("biosample_ontology") or {}
    target = exp.get("target")
    lab = exp.get("lab")
    controls = [_acc(c) for c in exp.get("possible_controls") or []]
    organisms = set()
    libraries, donors, lib_by_reps = [], {}, {}
    for rep in exp.get("replicates") or []:
        if not isinstance(rep, dict):
            continue
        lib = rep.get("library") if isinstance(rep.get("library"), dict) else {}
        bios = lib.get("biosample") if isinstance(lib.get("biosample"), dict) else {}
        donor = bios.get("donor")
        org = bios.get("organism")
        if isinstance(org, dict) and org.get("scientific_name"):
            organisms.add(org["scientific_name"])
        if isinstance(donor, dict) and donor.get("accession"):
            dorg = donor.get("organism")
            donors[donor["accession"]] = {
                "donor": donor["accession"], "sex": _s(donor.get("sex")),
                "age": _s(donor.get("age")), "age_units": _s(donor.get("age_units")),
                "life_stage": _s(donor.get("life_stage")),
                "health_status": _s(donor.get("health_status")),
                "organism": _s(dorg.get("scientific_name")) if isinstance(dorg, dict) else _acc(dorg),
            }
        br, tr = rep.get("biological_replicate_number"), rep.get("technical_replicate_number")
        lib_acc = _acc(rep.get("library"))
        lib_by_reps[f"{br}_{tr}"] = lib_acc
        libraries.append({
            "experiment": acc, "library": lib_acc, "biosample": _acc(lib.get("biosample")),
            "donor": _acc(donor), "bio_rep": _s(br), "tech_rep": _s(tr),
            "antibody": _acc(rep.get("antibody")) if rep.get("antibody") else "",
        })
    title = exp.get("assay_title") or ""
    keyword = next((k for k in ("Histone", "ATAC", "DNase") if k.lower() in title.lower()),
                   "ChIP")
    unrep = exp.get("replication_type") == "unreplicated"
    files = []
    for f in exp.get("files") or []:
        if not isinstance(f, dict):
            continue
        rep = f.get("replicate")
        lib = _acc(rep.get("library")) if isinstance(rep, dict) else ""
        tech = [str(t) for t in f.get("technical_replicates") or []]
        if not lib and len(tech) == 1:
            lib = lib_by_reps.get(tech[0], "")
        fc = [d for d in f.get("flowcell_details") or [] if isinstance(d, dict)]
        cloud = f.get("cloud_metadata") or {}
        pipe, pver = _pipeline(f, keyword, unrep)
        qc = _qc(f)
        files.append({
            "file": f.get("accession") or _acc(f), "experiment": acc, "library": lib,
            "file_format": _s(f.get("file_format")), "output_type": _s(f.get("output_type")),
            "assembly": _s(f.get("assembly")), "run_type": _s(f.get("run_type")),
            "read_length": _s(f.get("read_length")), "platform": _title(f.get("platform")),
            "paired_with": _acc(f.get("paired_with")) if f.get("paired_with") else "",
            "flowcell": ";".join(sorted({_s(d.get("flowcell")) for d in fc if d.get("flowcell")})),
            "lane": ";".join(sorted({_s(d.get("lane")) for d in fc if d.get("lane")})),
            "bio_reps": _s(f.get("biological_replicates")), "tech_reps": ";".join(tech),
            "pipeline": pipe, "pipeline_version": pver,
            "mapped_reads": _s(qc.get("mapped_reads")), "file_size": _s(f.get("file_size")),
            "md5": _s(f.get("md5sum")),
            "url": _s(cloud.get("url")) or (PORTAL + f["href"] if f.get("href") else ""),
            **{k: _s(qc.get(k)) for k in ("nsc", "rsc", "nrf", "pbc1", "pbc2", "frip",
                                          "frag_len")},
            "status": _s(f.get("status")), "paired_end": _s(f.get("paired_end")),
            "read_count": _s(f.get("read_count")),
        })
    row = {
        "experiment": acc, "assay": assay_of(exp),
        "target": _s(target.get("label")) if isinstance(target, dict) else "",
        "biosample_term": _s(onto.get("term_name")),
        "biosample_type": _s(onto.get("classification")),
        "replication_type": _s(exp.get("replication_type")),
        "lab": _title(lab), "status": _s(exp.get("status")),
        "date_released": _s(exp.get("date_released")),
        "internal_tags": _s(exp.get("internal_tags")),
        "control_experiments": ";".join(c for c in controls if c),
        "assay_title": _s(exp.get("assay_title")),
        "biosample_term_id": _s(onto.get("term_id")),
        "organism": ";".join(sorted(organisms)),
    }
    return {"experiment": row, "libraries": libraries, "files": files,
            "donors": list(donors.values())}


# ---------------------------------------------------------------- census

def _runs(fastqs: list[dict]) -> tuple[int, int]:
    """(known distinct run keys, loose run count) of one library's fastq rows."""
    seen, known, loose_unknown = set(), set(), 0
    for f in sorted(fastqs, key=lambda r: r["file"]):
        if f["file"] in seen:
            continue
        seen.add(f["file"])
        if f.get("paired_with"):
            seen.add(f["paired_with"])          # mates are one run
        key = (f.get("flowcell", ""), f.get("lane", ""))
        if key != ("", ""):
            known.add(key)
        else:
            loose_unknown += 1
    return len(known), len(known) + loose_unknown


def _c2(n: int) -> int:
    return n * (n - 1) // 2


def census(experiments: list[dict], libraries: list[dict], files: list[dict],
           keep: set[str] | None = None) -> list[dict]:
    """Pairs per rung and assay over the experiments in `keep` (all when None). Returns rows with
    CENSUS_COLS, one per (rung, assay in CENSUS_ASSAYS) plus one 'ALL' row per rung; zeros kept."""
    exps = {e["experiment"]: e for e in experiments
            if e.get("assay") in CENSUS_ASSAYS and (keep is None or e["experiment"] in keep)}
    fq_by_lib, fq_bytes = defaultdict(list), Counter()
    for f in files:
        if f["experiment"] not in exps or f.get("file_format") != "fastq":
            continue
        if f.get("status") not in READ_STATUSES:
            continue
        fq_bytes[f["experiment"]] += int(float(f["file_size"])) if f.get("file_size") else 0
        if f.get("library"):
            fq_by_lib[(f["experiment"], f["library"])].append(f)
    libs = defaultdict(list)      # experiment -> library rows with reads, one per library
    seen_libs = set()             # (some old libraries sit under two replicate entries)
    for lib in libraries:
        key = (lib["experiment"], lib["library"])
        if key in fq_by_lib and key not in seen_libs:
            seen_libs.add(key)
            libs[lib["experiment"]].append(lib)

    # per (rung, assay): pairs, loose pairs, contributing experiments
    pairs, loose, contrib = Counter(), Counter(), defaultdict(set)

    def add(rung, exp_acc, n, n_loose=None):
        assay = exps[exp_acc]["assay"]
        if n:
            contrib[(rung, assay)].add(exp_acc)
        pairs[(rung, assay)] += n
        loose[(rung, assay)] += n if n_loose is None else n_loose

    for acc, rows in libs.items():
        add("pseudoreplicate", acc, len(rows))
        seq_k = seq_l = 0
        for lib in rows:
            k, l = _runs(fq_by_lib[(acc, lib["library"])])
            seq_k += _c2(k)
            seq_l += _c2(l)
        add("sequencing", acc, seq_k, seq_l)
        by_br = defaultdict(list)
        for lib in rows:
            by_br[lib["bio_rep"]].append(lib)
        add("technical", acc, sum(_c2(len({l["library"] for l in v})) for v in by_br.values()))
        donor_sets = [frozenset(l["donor"] for l in v if l["donor"]) for v in by_br.values()]
        iso = aniso = 0
        for a, b in combinations(donor_sets, 2):
            if a and a == b:
                iso += 1
            elif a and b and not (a & b):
                aniso += 1
        add("isogenic", acc, iso)
        add("anisogenic", acc, aniso)

    by_assay = defaultdict(lambda: defaultdict(list))   # assay -> term -> experiments with reads
    for acc in libs:
        e = exps[acc]
        by_assay[e["assay"]][e.get("biosample_term_id") or e["biosample_term"]].append(acc)
    for assay, terms in by_assay.items():
        total = sum(len(v) for v in terms.values())
        same = sum(_c2(len(v)) for v in terms.values())
        for v in terms.values():
            if len(v) > 1:
                contrib[("other_experiment", assay)].update(v)
        pairs[("other_experiment", assay)] += same
        loose[("other_experiment", assay)] += same
        diff = _c2(total) - same
        pairs[("other_cell_type", assay)] += diff
        loose[("other_cell_type", assay)] += diff
        if len(terms) > 1:
            contrib[("other_cell_type", assay)].update(a for v in terms.values() for a in v)

    def term(acc):
        e = exps[acc]
        return e.get("biosample_term_id") or e["biosample_term"]

    out = []
    for rung in RUNGS:
        all_exps: set = set()
        tot = tot_loose = 0
        for assay in CENSUS_ASSAYS:
            ex = contrib[(rung, assay)]
            all_exps |= ex
            tot += pairs[(rung, assay)]
            tot_loose += loose[(rung, assay)]
            out.append(_census_row(rung, assay, pairs[(rung, assay)], loose[(rung, assay)],
                                   ex, term, fq_bytes))
        out.append(_census_row(rung, "ALL", tot, tot_loose, all_exps, term, fq_bytes))
    return out


def _census_row(rung, assay, n, n_loose, exps, term, fq_bytes) -> dict:
    return {"rung": rung, "assay": assay, "n_pairs": n, "n_cells": len({term(a) for a in exps}),
            "n_experiments": len(exps), "fastq_gb": f"{sum(fq_bytes[a] for a in exps) / 1e9:.1f}",
            "n_pairs_loose": n_loose}


def summarize(rows: list[dict], min_pairs: int = 20) -> list[dict]:
    """Per rung: total pairs, loose pairs, and the count of panel assays (controls excluded) with
    at least `min_pairs` pairs."""
    out = []
    for rung in RUNGS:
        rr = [r for r in rows if r["rung"] == rung]
        tot = next(r for r in rr if r["assay"] == "ALL")
        panel = [r for r in rr if r["assay"] in PANEL]
        out.append({"rung": rung, "n_pairs": tot["n_pairs"], "n_pairs_loose": tot["n_pairs_loose"],
                    f"panel_assays_ge{min_pairs}": sum(int(r["n_pairs"]) >= min_pairs for r in panel),
                    "panel_assays_gt0": sum(int(r["n_pairs"]) > 0 for r in panel),
                    "n_cells": tot["n_cells"], "n_experiments": tot["n_experiments"],
                    "fastq_gb": tot["fastq_gb"]})
    return out


# ---------------------------------------------------------------- I/O

def write_tsv(path: str, rows: list[dict], cols: list[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: _s(r.get(c, "")) for c in cols})
    os.replace(tmp, path)


def read_tsv(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def read_eic_list(path: str) -> dict[str, dict]:
    """experiment -> row of the EIC list (columns experiment, assay, split, cell)."""
    out: dict[str, dict] = {}
    for r in read_tsv(path):
        out.setdefault(r["experiment"], r)
    return out


class Fetcher:
    """Rate-limited, retrying, caching JSON fetcher. A cached response is never refetched."""

    def __init__(self, cache: str, rate: float = 5.0, retries: int = 6, opener=None,
                 backoff: float = 2.0):
        self.cache, self.min_gap, self.retries = cache, 1.0 / rate, retries
        self.backoff = backoff
        self.opener = opener or self._urlopen
        self._lock, self._next = threading.Lock(), 0.0
        self.n_net = self.n_cache = 0
        os.makedirs(cache, exist_ok=True)

    @staticmethod
    def _urlopen(url: str) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read()

    def _wait(self):
        with self._lock:
            now = time.monotonic()
            t = max(now, self._next)
            self._next = t + self.min_gap
        if t > now:
            time.sleep(t - now)

    def get(self, name: str, url: str) -> dict:
        path = os.path.join(self.cache, name + ".json.gz")
        if os.path.exists(path):
            self.n_cache += 1
            with gzip.open(path, "rt") as fh:
                return json.load(fh)
        delay = self.backoff
        for attempt in range(self.retries + 1):
            self._wait()
            try:
                body = self.opener(url)
                data = json.loads(body)
                break
            except urllib.error.HTTPError as e:
                if e.code == 404 or attempt == self.retries or e.code not in (429, 500, 502, 503, 504):
                    raise
                ra = e.headers.get("Retry-After") if e.headers else None
                wait = float(ra) if ra and ra.isdigit() else delay
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
                if attempt == self.retries:
                    raise
                wait = delay
            time.sleep(wait + random.uniform(0, min(1.0, self.backoff)))
            delay = min(delay * 2, 120)
        tmp = path + f".tmp{os.getpid()}.{threading.get_ident()}"
        with gzip.open(tmp, "wt") as fh:
            json.dump(data, fh)
        os.replace(tmp, path)
        self.n_net += 1
        return data


def search_url(query: str) -> str:
    return (f"{PORTAL}/search/?type=Experiment&{HUMAN}&status=released&{query}"
            "&format=json&limit=all&field=accession&field=assay_title&field=target.label")


def exp_url(acc: str) -> str:
    return f"{PORTAL}/experiments/{urllib.parse.quote(acc)}/?frame=embedded&format=json"


def crawl(args) -> int:
    out, fetch = args.out, Fetcher(args.cache, rate=args.rate)
    os.makedirs(out, exist_ok=True)
    eic = read_eic_list(args.eic_list)
    wanted: dict[str, str] = {}
    for name, q in SEARCHES:
        d = fetch.get(f"search_{name}", search_url(q))
        graph = d.get("@graph", [])
        kept = [g["accession"] for g in graph if assay_of(g)]
        print(f"[search] {name}: total={d.get('total')} listed={len(graph)} in_panel={len(kept)}",
              flush=True)
        for a in kept:
            wanted.setdefault(a, name)
    for a in eic:
        wanted.setdefault(a, "eic")
    if args.limit:
        wanted = dict(list(wanted.items())[: args.limit])
    print(f"[crawl] {len(wanted)} experiments to read (cache {args.cache})", flush=True)

    parsed: dict[str, dict] = {}
    failed: list[str] = []

    def run(accs):
        done = 0
        with ThreadPoolExecutor(args.threads) as pool:
            futs = {pool.submit(fetch.get, f"exp_{a}", exp_url(a)): a for a in accs}
            for fut in as_completed(futs):
                a = futs[fut]
                try:
                    parsed[a] = parse_experiment(fut.result())
                except Exception as e:  # noqa: BLE001 -- record and carry on
                    failed.append(a)
                    print(f"[fail] {a}: {type(e).__name__}: {e}", flush=True)
                done += 1
                if done % 250 == 0 or done == len(accs):
                    print(f"[progress] {done}/{len(accs)} net={fetch.n_net} cache={fetch.n_cache}"
                          f" failed={len(failed)} {time.strftime('%H:%M:%S')}", flush=True)

    run(list(wanted))
    controls = sorted({c for p in parsed.values()
                       if p["experiment"]["assay"] in PANEL_HISTONES and
                       (p["experiment"]["status"] == "released" or p["experiment"]["experiment"] in eic)
                       for c in p["experiment"]["control_experiments"].split(";") if c} - set(parsed))
    print(f"[crawl] {len(controls)} control experiments named by panel experiments", flush=True)
    if args.limit:
        controls = controls[: args.limit]
    run(controls)

    rows = [p["experiment"] for p in parsed.values()]
    # a fetched control enters only as a ChIP control; searched and EIC experiments always enter
    rows = [r for r in rows if r["experiment"] in wanted or r["experiment"] in eic
            or r["assay"] == CONTROL]
    keep = {r["experiment"] for r in rows}
    donors = {}
    for p in parsed.values():
        if p["experiment"]["experiment"] in keep:
            for d in p["donors"]:
                donors[d["donor"]] = d
    exp_rows = sorted(rows, key=lambda r: r["experiment"])
    lib_rows = [l for a in sorted(keep) for l in parsed[a]["libraries"]]
    file_rows = [f for a in sorted(keep) for f in parsed[a]["files"]]
    write_tsv(os.path.join(out, "experiments.tsv"), exp_rows, EXPERIMENT_COLS)
    write_tsv(os.path.join(out, "libraries.tsv"), lib_rows, LIBRARY_COLS)
    write_tsv(os.path.join(out, "files.tsv"), file_rows, FILE_COLS)
    write_tsv(os.path.join(out, "donors.tsv"), sorted(donors.values(), key=lambda d: d["donor"]),
              DONOR_COLS)
    print(f"[tables] experiments={len(exp_rows)} libraries={len(lib_rows)} files={len(file_rows)}"
          f" donors={len(donors)} failed={len(failed)}", flush=True)
    if failed:
        print("[failed] " + " ".join(sorted(failed)), flush=True)
    return run_census(out, args.eic_list, args.summary_dir)


def run_census(out: str, eic_list: str, summary_dir: str | None = None) -> int:
    exps = read_tsv(os.path.join(out, "experiments.tsv"))
    libs = read_tsv(os.path.join(out, "libraries.tsv"))
    files = read_tsv(os.path.join(out, "files.tsv"))
    eic = read_eic_list(eic_list)
    released = {e["experiment"] for e in exps
                if e["status"] == "released" and "Homo sapiens" in e.get("organism", "")}
    portal = census(exps, libs, files, released)
    eic_rows = census(exps, libs, files, set(eic))
    write_tsv(os.path.join(out, "ladder_census.tsv"), portal, CENSUS_COLS)
    write_tsv(os.path.join(out, "eic_census.tsv"), eic_rows, CENSUS_COLS)

    by_acc = {e["experiment"]: e for e in exps}
    missing = sorted(a for a in eic if a not in by_acc)
    mismatch = sorted(a for a in eic if a in by_acc and by_acc[a]["assay"] != eic[a]["assay"])
    status = Counter(by_acc[a]["status"] for a in eic if a in by_acc)
    print(f"[eic] listed={len(eic)} found={len(eic) - len(missing)} missing={len(missing)} "
          f"assay_mismatch={len(mismatch)} status={dict(status)}", flush=True)
    for a in mismatch:
        print(f"[eic-mismatch] {a} list={eic[a]['assay']} portal={by_acc[a]['assay']}", flush=True)
    for name, rows in (("portal", portal), ("eic", eic_rows)):
        for s in summarize(rows):
            print(f"[census-{name}] " + " ".join(f"{k}={v}" for k, v in s.items()), flush=True)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
        sp, se = summarize(portal), summarize(eic_rows)
        write_tsv(os.path.join(summary_dir, "census_summary.tsv"), sp, list(sp[0]))
        write_tsv(os.path.join(summary_dir, "eic_census_summary.tsv"), se, list(se[0]))
        write_tsv(os.path.join(summary_dir, "ladder_census.tsv"), portal, CENSUS_COLS)
        write_tsv(os.path.join(summary_dir, "eic_census.tsv"), eic_rows, CENSUS_COLS)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crawl", help="fetch, write the catalog tables, then the censuses")
    c.add_argument("--out", required=True)
    c.add_argument("--cache", required=True)
    c.add_argument("--eic-list", required=True)
    c.add_argument("--rate", type=float, default=5.0, help="requests per second (portal max 10)")
    c.add_argument("--threads", type=int, default=4)
    c.add_argument("--limit", type=int, default=0, help="read only the first N (smoke test)")
    c.add_argument("--summary-dir", default=None)
    s = sub.add_parser("census", help="recompute the censuses from the catalog tables")
    s.add_argument("--out", required=True)
    s.add_argument("--eic-list", required=True)
    s.add_argument("--summary-dir", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "crawl":
        if not 0 < args.rate <= 10:
            ap.error("--rate must be in (0, 10]")
        return crawl(args)
    return run_census(args.out, args.eic_list, args.summary_dir)


if __name__ == "__main__":
    sys.exit(main())
