#!/usr/bin/env python3
"""Second experiments for EIC tracks (t257, chunk OXs of the onion plan, Amendment 4; PI ruling 33).

    other_experiments.py select --census CENSUS.tsv --catalog DIR --ledger L --build-list B
                                --out OUT.tsv --dropped DROPPED.tsv [--summary SELECTION.md]
    other_experiments.py pair-smoke --pairs <a dir>:<b dir>[,...] --evalset EV.json --out OUT.md

`select` resolves the candidates of the other-experiment census (`best_candidate` and
`other_accessions`; no new census is made) into one first choice and one fallback per EIC track.
It reads the D1 ENCODE catalog (`experiments.tsv`, `files.tsv`, `libraries.tsv`), the EIC ledger
and the EIC build list, asks the ENCODE portal for status, audits and perturbations (one search per
40 accessions, generic User-Agent, metadata only), and sends one HTTP HEAD per FASTQ as
`build_list.probe` does. It downloads no data file. Run it on the cluster.

Checks per (track, candidate), written to the column `checks`:
  1 released: portal status released and no audit of level ERROR.
  2 unperturbed: not `perturbed`, no treatment, no genetic modification, and the same biosample
    term as the EIC experiment.
  3 not EIC: not an EIC ledger experiment, not an EIC control, no library shared with the EIC
    experiment.
  4 source: at least one FASTQ served (HTTP 200), mates kept together
    (`build_list.available_fastqs`); `source = fastq`. Else (PI ruling 16, as for the EIC
    experiments) a released GRCh38 "redacted unfiltered alignments" BAM per biological replicate
    served by the portal (`build_list.native_bams`, the `native-redacted-bam` route);
    `source = redacted_bam`, its depth = flagstats `mapped` of the unfiltered BAM, halved when
    paired-end (fragments). Among candidates equal on lab and donor, `fastq` goes first.
  5 tech: run type, read length, platform, biological-replicate count (recorded).
  6 depth: fragments >= the 5th percentile of the fragments of the EIC training base experiments
    of the same assay class. A paired-end read pair is one fragment (the R1 read counts);
    `catalog_reads` sums every FASTQ (both mates) and is the figure for the build cost.
  7 control: the candidate's own control, served or not, and whether the EIC build holds it
    (recorded only; decision 4).
A candidate passes when checks 1, 2, 3, 4 and 6 pass. Order of the passing candidates: other lab
first, then same donor, other donor, unknown donor, then source fastq before redacted_bam, then the
smallest |log2 depth ratio| to the EIC experiment (fragments), then the accession. Assignment: tracks in the order validation, blind,
training, then `eic_track`; first every track takes its best passing candidate that no earlier
track holds, then, in the same order, every track takes its next free candidate as the fallback.
An experiment serves at most one track. A blind track gets no second experiment (foreman decision
of 2026-11-12): it goes to the dropped table with the reason `blind_track_excluded`, because a second
experiment of a blind track could become its input before the single ledgered blind pass. A track
with no free passing candidate goes to the dropped table with the failed checks of every candidate.

`pair-smoke` reads `counts.h5` of both measurement directories of each pair on the eval set's
regions, depth-matches them by binomial thinning as `metrics.variance_ratio` does (declared depths
from the `depth_reads` attribute, seed crc32("h7:<a>:<b>")), and prints sum(a)/sum(b) and whether it
lies outside 1 +- 0.05; such a pair gives an Undefined headline-7 ratio.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections import Counter, defaultdict
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from onion_data import build_list as B  # noqa: E402

PORTAL = "https://www.encodeproject.org"
USER_AGENT = "onion-catalog/1.0 (python-urllib; research metadata crawler)"
SPLIT_ORDER = {"val": 0, "blind": 1, "train": 2}
DONOR_RANK = {"same": 0, "other": 1, "unknown": 2}
FLOOR_PCT = 5.0
TOTAL_TOL = 0.05
BLIND_REASON = "blind_track_excluded"
#: CPU-h per million catalog reads (Amendment 4, "What was verified": t180 rates).
RATES = {"chip_atac": ((0.160, 0.177), 0.027), "accessibility_dnase": ((0.04, 0.04), 0.016)}
OUT_COLUMNS = ("eic_track", "split", "cell", "assay", "assay_class", "eic_experiment", "eic_lab",
               "eic_donor", "experiment", "lab", "donor", "biosample", "pair_class",
               "same_biosample", "run_type", "read_length", "platform", "tech_diff", "n_bio_reps",
               "catalog_reads", "fragments", "log2_depth_ratio", "fastq_files", "fastq_gb",
               "own_control", "own_control_state", "checks", "fallback", "fallback_pair_class",
               "fallback_fastq_files", "fallback_fastq_gb", "source", "alignment_files",
               "alignment_bytes", "alignment_md5", "alignment_gb", "fallback_source",
               "fallback_alignment_files", "fallback_alignment_gb")
DROPPED_COLUMNS = ("eic_track", "split", "cell", "assay", "assay_class", "eic_experiment",
                   "n_candidates", "reason", "candidates")


# ------------------------------------------------------------------------------------ network

def _get_json(url: str, retries: int = 5) -> dict:
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {"@graph": []}
            if attempt == retries or e.code not in (429, 500, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries:
                raise
        time.sleep(delay)
        delay *= 2
    raise RuntimeError("unreachable")


def portal_meta(accessions, batch: int = 40, gap: float = 0.3) -> dict:
    """accession -> {status, errors, perturbed, treated, modified, term} from the portal search."""
    fields = ("accession", "status", "perturbed", "audit", "biosample_ontology.term_name",
              "replicates.library.biosample.treatments",
              "replicates.library.biosample.genetic_modifications")
    accs, out = sorted(set(accessions)), {}
    for i in range(0, len(accs), batch):
        q = [("type", "Experiment")] + [("accession", a) for a in accs[i:i + batch]] + \
            [("field", f) for f in fields] + [("format", "json"), ("limit", "all")]
        d = _get_json(f"{PORTAL}/search/?{urllib.parse.urlencode(q)}")
        for g in d.get("@graph", []):
            bios = [((r.get("library") or {}).get("biosample") or {})
                    for r in g.get("replicates") or [] if isinstance(r, dict)]
            out[g["accession"]] = {
                "status": g.get("status", ""),
                "errors": sorted({str(x.get("category", "")) for x in
                                  (g.get("audit") or {}).get("ERROR", [])}),
                "perturbed": bool(g.get("perturbed")),
                "treated": any(b.get("treatments") for b in bios),
                "modified": any(b.get("genetic_modifications") for b in bios),
                "term": (g.get("biosample_ontology") or {}).get("term_name", ""),
            }
        time.sleep(gap)
    return out


def bam_metrics(accessions, batch: int = 40, gap: float = 0.3) -> dict:
    """file accession -> {mapped, total, paired} of the portal's samtools flagstats of that BAM
    (the `unfiltered` processing stage when the file has several)."""
    accs, out = sorted(set(accessions)), {}
    for i in range(0, len(accs), batch):
        q = [("type", "File")] + [("accession", a) for a in accs[i:i + batch]] + \
            [("field", "accession"), ("field", "quality_metrics"), ("format", "json"),
             ("limit", "all")]
        d = _get_json(f"{PORTAL}/search/?{urllib.parse.urlencode(q)}")
        for g in d.get("@graph", []):
            qs = [x for x in g.get("quality_metrics") or [] if isinstance(x, dict)
                  and "SamtoolsFlagstatsQualityMetric" in (x.get("@type") or []) and "mapped" in x]
            qs = [x for x in qs if x.get("processing_stage") == "unfiltered"] or qs
            if qs:
                x = qs[0]
                out[g["accession"]] = {"mapped": int(x["mapped"]), "total": int(x.get("total") or 0),
                                       "paired": int(x.get("paired") or x.get("read1") or 0)}
        time.sleep(gap)
    return out


def head_probe(files, threads: int = 16) -> dict:
    """file accession -> HTTP status (str) of one HEAD, as `build_list.probe` sends it."""
    def head(f):
        ext = "bam" if f.get("file_format") == "bam" else "fastq.gz"
        url = f.get("url") or (f"{PORTAL}/files/{f['file']}/@@download/{f['file']}.{ext}")
        last = ""
        for _ in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(
                        url, method="HEAD", headers={"User-Agent": USER_AGENT}), timeout=30) as r:
                    return f["file"], str(r.status)
            except urllib.error.HTTPError as e:
                last = str(e.code)
                if e.code in (401, 403, 404):
                    break
            except Exception as e:  # noqa: BLE001  (network errors are recorded, not raised)
                last = type(e).__name__
        return f["file"], last
    with cf.ThreadPoolExecutor(threads) as ex:
        return dict(ex.map(head, list(files)))


# ------------------------------------------------------------------------------------ catalog

def load_catalog(catalog: Path, wanted: set) -> dict:
    exps = {r["experiment"]: r for r in B.read_tsv(catalog / "experiments.tsv")
            if r["experiment"] in wanted}
    libs = defaultdict(list)
    for r in B.read_tsv(catalog / "libraries.tsv"):
        if r["experiment"] in wanted:
            libs[r["experiment"]].append(r)
    files = defaultdict(list)
    with open(catalog / "files.tsv", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["experiment"] in wanted and (r.get("file_format") == "fastq" or B.is_native_bam(r)):
                files[r["experiment"]].append(r)
    return {"exps": exps, "libs": libs, "files": files}


def released_fastqs(files: list[dict]) -> list[dict]:
    """Released FASTQs, or the archived ones when none is released (the kits' rule)."""
    fq = [f for f in files if f.get("file_format", "fastq") == "fastq"]
    return [f for f in fq if f.get("status") == "released"] or \
           [f for f in fq if f.get("status") == "archived"]


def is_r1(f: dict) -> bool:
    return f.get("paired_end", "") in ("", "1")


def depth_of(files: list[dict]) -> tuple[int | None, int | None]:
    """(catalog reads over every FASTQ, fragments = reads of R1/single-end FASTQs); None when a
    FASTQ has no catalog read count."""
    if not files or any(not (f.get("read_count") or "").strip() for f in files):
        return None, None
    reads = sum(int(float(f["read_count"])) for f in files)
    frags = sum(int(float(f["read_count"])) for f in files if is_r1(f))
    return reads, frags


def tech_of(files: list[dict]) -> dict:
    return {"run_type": ";".join(sorted({f.get("run_type", "") for f in files} - {""})),
            "read_length": ";".join(str(x) for x in sorted(
                {int(float(f["read_length"])) for f in files if f.get("read_length")})),
            "platform": ";".join(sorted({f.get("platform", "") for f in files} - {""})),
            "n_bio_reps": str(len({B.bio_of(f) for f in files if f.get("bio_reps")}) or "")}


def donors_of(libs: list[dict]) -> set:
    return {l["donor"] for l in libs if l.get("donor")}


def donor_relation(a: set, b: set) -> str:
    if not a or not b:
        return "unknown"
    return "same" if a & b else "other"


def pair_class(same_lab: bool, donor: str) -> str:
    parts = ["experiment"] + ([] if same_lab else ["lab"])
    if donor == "other":
        parts.append("donor")
    elif donor == "unknown":
        parts.append("donor?")
    return ";".join(parts)


def percentile(xs, q: float) -> float:
    """Linear-interpolation percentile (numpy's default), without numpy."""
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def eic_sets(ledger: list[dict], build_rows: list[dict]) -> dict:
    """Every accession the EIC side holds: ledger experiments (any column naming one), EIC controls
    (ledger control rows, `encode_possible_controls`, build-list control rows)."""
    exps, ctls = set(), set()
    for r in ledger:
        accs = {r.get(k, "") for k in ("experiment", "published_experiment", "file_experiment")} - {""}
        (ctls if r.get("assay_class") == B.CONTROL_CLASS else exps).update(accs)
        ctls.update(B.split_acc(r.get("encode_possible_controls", "")))
    for r in build_rows:
        (ctls if r.get("assay_class") == B.CONTROL_CLASS else exps).add(r["experiment"])
        ctls.update(B.split_acc(r.get("control_experiment", "")))
    ctls -= exps
    source = {r["experiment"]: "redacted_bam" for r in build_rows
              if r.get("pipeline") == B.NATIVE_PIPELINE}
    return {"experiments": exps, "controls": ctls, "source": source,
            "build": {r["experiment"] for r in build_rows}}


def depth_floors(build_rows: list[dict], cat: dict) -> dict:
    """assay class -> (5th percentile of fragments, n experiments) over the EIC training base
    experiments of the build list."""
    by_cls = defaultdict(dict)
    for r in build_rows:
        if r.get("role") == "train" and r.get("arm") == "base" and r.get("unit") == "pooled" \
                and r.get("assay_class") != B.CONTROL_CLASS:
            _, fr = depth_of(released_fastqs(cat["files"].get(r["experiment"], [])))
            if fr:
                by_cls[r["assay_class"]][r["experiment"]] = fr
    return {c: (percentile(list(d.values()), FLOOR_PCT), len(d)) for c, d in by_cls.items()}


# ------------------------------------------------------------------------------------- select

def candidates_of(row: dict) -> list[str]:
    out = []
    for a in [row.get("best_candidate", "")] + B.split_acc(row.get("other_accessions", "")):
        if a and a not in out:
            out.append(a)
    return out


def bam_depth(bams: list[dict], metrics: dict) -> tuple[int | None, int | None]:
    """(reads, fragments) of redacted BAMs: `total` and `mapped` of the unfiltered flagstats;
    `mapped` counts mates separately, so a paired-end BAM (flagstats `paired` > 0) gives
    fragments = mapped / 2. None when a BAM has no flagstats."""
    if not bams or any(f["file"] not in metrics for f in bams):
        return None, None
    reads = sum(metrics[f["file"]]["total"] for f in bams)
    frags = sum(metrics[f["file"]]["mapped"] // (2 if metrics[f["file"]]["paired"] > 0 else 1)
                for f in bams)
    return reads, frags


def evaluate(track: dict, acc: str, cat: dict, portal: dict, probe: dict, eic: dict,
             floors: dict, metrics: dict | None = None) -> dict:
    """All seven checks of one candidate for one track."""
    E = track["eic_experiment"]
    ex = cat["exps"].get(acc)
    fails, notes = [], {}
    pm = portal.get(acc)
    # 1
    if pm is None:
        fails.append("1 not on the portal")
    elif pm["status"] != "released":
        fails.append(f"1 status {pm['status']}")
    elif pm["errors"]:
        fails.append("1 audit ERROR " + ",".join(pm["errors"]))
    # 2
    eic_term = (cat["exps"].get(E) or {}).get("biosample_term") or track.get("encode_term", "")
    term = (ex or {}).get("biosample_term") or (pm or {}).get("term", "")
    if pm is not None and (pm["perturbed"] or pm["treated"] or pm["modified"]):
        fails.append("2 perturbed (" + ",".join(k for k in ("perturbed", "treated", "modified")
                                               if pm[k]) + ")")
    if term != eic_term:
        fails.append(f"2 biosample term {term!r} != {eic_term!r}")
    # 3
    libs, eic_libs = cat["libs"].get(acc, []), cat["libs"].get(E, [])
    shared = {l["library"] for l in libs} & {l["library"] for l in eic_libs} - {""}
    if acc in eic["experiments"]:
        fails.append("3 EIC ledger experiment")
    if acc in eic["controls"]:
        fails.append("3 EIC control")
    if shared:
        fails.append("3 shares library " + ",".join(sorted(shared)))
    # 4
    rel = released_fastqs(cat["files"].get(acc, []))
    pdict = {f: {"http": h} for f, h in probe.items()}
    kept, _ = B.available_fastqs(rel, pdict)
    bams, source = [], "fastq"
    if not kept:
        bams = B.native_bams(cat["files"].get(acc, []), pdict)
        if bams:
            source = "redacted_bam"
        else:
            fails.append(f"4 no FASTQ served (0/{len(rel)}) and no redacted GRCh38 BAM served")
    # 5
    tech = tech_of(kept or rel)
    eic_files = released_fastqs(cat["files"].get(E, []))
    eic_tech = tech_of(eic_files)
    # 6
    reads, frags = depth_of(kept) if kept else bam_depth(bams, metrics or {})
    _, eic_frags = depth_of(eic_files)
    floor = floors.get(track["assay_class"], (float("nan"), 0))[0]
    if (kept or bams) and frags is None:
        fails.append("6 a FASTQ has no catalog read count" if kept else
                     "6 a redacted BAM has no flagstats mapped count")
    elif (kept or bams) and not (frags >= floor):
        fails.append(f"6 fragments {frags} < floor {floor:.0f}")
    ratio = (math.log2(frags / eic_frags) if frags and eic_frags else float("nan"))
    # 7
    ctl = B.split_acc((ex or {}).get("control_experiments", ""))
    states = []
    for c in ctl:
        cf_ = released_fastqs(cat["files"].get(c, []))
        n_ok = sum(1 for f in cf_ if probe.get(f["file"]) == "200")
        states.append(f"{c}:{'served' if n_ok else 'not_served'}"
                      f"{',in_eic_build' if c in eic['build'] else ''}")
    same_lab = (ex or {}).get("lab", "") == track["eic_lab"]
    donor = donor_relation(donors_of(libs), donors_of(eic_libs))
    ok = not fails
    checks = ";".join([
        "1:" + ("fail" if any(f.startswith("1") for f in fails) else "ok"),
        "2:" + ("fail" if any(f.startswith("2") for f in fails) else "ok"),
        "3:" + ("fail" if any(f.startswith("3") for f in fails) else "ok"),
        (f"4:ok({len(kept)}/{len(rel)} served)" if kept else
         f"4:ok(redacted_bam {len(bams)} BAMs; 0/{len(rel)} FASTQ served)" if bams else
         f"4:fail(0/{len(rel)} served)"),
        "5:recorded",
        f"6:{'fail' if any(f.startswith('6') for f in fails) else 'ok'}"
        f"({frags if frags is not None else 'NA'}>={floor:.0f})",
        f"7:{'recorded' if ctl else 'none'}"])
    return {
        "experiment": acc, "ok": ok, "fails": fails, "checks": checks,
        "lab": (ex or {}).get("lab", ""), "donor": ";".join(sorted(donors_of(libs))),
        "biosample": ";".join(sorted({l["biosample"] for l in libs if l.get("biosample")})),
        "same_biosample": "1" if ({l["biosample"] for l in libs} &
                                  {l["biosample"] for l in eic_libs} - {""}) else "0",
        "same_lab": same_lab, "donor_rel": donor, "pair_class": pair_class(same_lab, donor),
        **tech,
        "tech_diff": ";".join([k for k in ("run_type", "read_length", "platform")
                               if tech[k] != eic_tech[k]] +
                              (["source"] if source != eic["source"].get(E, "fastq") else [])),
        "source": source, "alignment_files": ";".join(f["file"] for f in bams),
        "alignment_bytes": ";".join(str(f.get("file_size", "")) for f in bams),
        "alignment_md5": ";".join(f.get("md5", "") for f in bams),
        "alignment_gb": f"{B.gb(bams):.3f}",
        "catalog_reads": "" if reads is None else str(reads),
        "fragments": "" if frags is None else str(frags),
        "log2_depth_ratio": "" if math.isnan(ratio) else f"{ratio:.3f}", "_ratio": ratio,
        "fastq_files": ";".join(f["file"] for f in kept),
        "fastq_gb": f"{B.gb(kept):.3f}",
        "own_control": ";".join(ctl), "own_control_state": ";".join(states),
    }


def order_key(c: dict):
    r = abs(c["_ratio"]) if not math.isnan(c["_ratio"]) else float("inf")
    return (c["same_lab"], DONOR_RANK[c["donor_rel"]], c["source"] != "fastq", r, c["experiment"])


def select(census: list[dict], cat: dict, ledger: list[dict], build_rows: list[dict],
           portal: dict, probe: dict, metrics: dict | None = None) -> dict:
    eic = eic_sets(ledger, build_rows)
    floors = depth_floors(build_rows, cat)
    tracks = sorted((r for r in census if candidates_of(r)),
                    key=lambda r: (SPLIT_ORDER.get(r["split"], 9), r["eic_track"]))
    blind = {t["eic_track"] for t in tracks if t["split"] == "blind"}
    evals = {}
    for t in tracks:
        evals[t["eic_track"]] = [evaluate(t, a, cat, portal, probe, eic, floors, metrics)
                                 for a in candidates_of(t)]
    # foreman decision 2026-11-12: a blind track gets no second experiment (it could become an
    # input of that track before the single ledgered blind pass); it holds no candidate either
    passing = {k: ([] if k in blind else sorted((c for c in v if c["ok"]), key=order_key))
               for k, v in evals.items()}
    held, first, fall = {}, {}, {}
    for t in tracks:                       # pass 1: first choices
        k = t["eic_track"]
        for c in passing[k]:
            if c["experiment"] not in held:
                first[k], held[c["experiment"]] = c, k
                break
    for t in tracks:                       # pass 2: fallbacks
        k = t["eic_track"]
        if k not in first:
            continue
        for c in passing[k]:
            if c["experiment"] not in held:
                fall[k], held[c["experiment"]] = c, k
                break
    out, dropped = [], []
    for t in tracks:
        k = t["eic_track"]
        eic_libs = cat["libs"].get(t["eic_experiment"], [])
        base = {"eic_track": k, "split": t["split"], "cell": t["cell"], "assay": t["assay"],
                "assay_class": t["assay_class"], "eic_experiment": t["eic_experiment"]}
        if k in first:
            c, f = first[k], fall.get(k, {})
            out.append({**base, "eic_lab": t["eic_lab"],
                        "eic_donor": ";".join(sorted(donors_of(eic_libs))),
                        **{col: c.get(col, "") for col in OUT_COLUMNS
                           if col not in base and col not in ("eic_lab", "eic_donor")},
                        "fallback": f.get("experiment", ""),
                        "fallback_pair_class": f.get("pair_class", ""),
                        "fallback_fastq_files": f.get("fastq_files", ""),
                        "fallback_fastq_gb": f.get("fastq_gb", ""),
                        "fallback_source": f.get("source", ""),
                        "fallback_alignment_files": f.get("alignment_files", ""),
                        "fallback_alignment_gb": f.get("alignment_gb", "")})
        else:
            cands = []
            for c in evals[k]:
                why = "; ".join(c["fails"]) if c["fails"] else (
                    "passes; blind track" if k in blind else f"held by {held.get(c['experiment'])}")
                cands.append(f"{c['experiment']}: {why}")
            reason = (BLIND_REASON if k in blind else
                      "no candidate passes" if not passing[k] else
                      "every passing candidate is held by an earlier track")
            dropped.append({**base, "n_candidates": str(len(evals[k])), "reason": reason,
                            "candidates": " | ".join(cands)})
    return {"rows": out, "dropped": dropped, "floors": floors, "evals": evals, "tracks": tracks}


# ------------------------------------------------------------------------------------ summary

def build_cost(rows: list[dict]) -> tuple[float, float, float]:
    """(million reads, CPU-h low, CPU-h high) of the first choices at the t180 rates. A FASTQ row
    pays alignment plus views. A redacted-BAM row is not aligned: its low figure is the views rate
    only, its high figure the full rate (no rate of the native filter step is measured)."""
    m = lo = hi = 0.0
    for r in rows:
        mr = int(r["catalog_reads"] or 0) / 1e6
        (a_lo, a_hi), views = RATES["accessibility_dnase" if r["assay_class"] ==
                                    "accessibility_dnase" else "chip_atac"]
        native = r.get("source") == "redacted_bam"
        m, lo, hi = m + mr, lo + mr * ((0 if native else a_lo) + views), hi + mr * (a_hi + views)
    return m, lo, hi


def summary_md(res: dict, n_heads: int, n_served: int, census_path: str) -> str:
    rows, dropped, floors = res["rows"], res["dropped"], res["floors"]
    n_tracks = len(res["tracks"])
    n_cand = sum(len(v) for v in res["evals"].values())
    n_pass = sum(1 for v in res["evals"].values() for c in v if c["ok"])
    fail_counts = Counter(f.split(" ")[0] for v in res["evals"].values() for c in v
                          for f in c["fails"])
    L = ["# Selection of the second experiments (t257, chunk OXs)", "",
         f"Census: `{Path(census_path).name}` (`cruxvault/results/pi_notes_2026-11-12/`). The census matched candidates on the exact ENCODE biosample "
         "term and assay or target of each EIC track. A cell type that ENCODE files under another "
         "term is missed, so the census undercounts the candidates; this selection makes no new "
         "census.", "",
         f"The census lists {n_tracks} tracks with at least one candidate and {n_cand} "
         f"(track, candidate) entries. {n_pass} entries passed checks 1, 2, 3, 4 and 6. "
         f"The selection kept {len(rows)} tracks and dropped {len(dropped)}. "
         f"The probe sent {n_heads} HTTP HEAD requests; {n_served} FASTQs answered 200.", "",
         "Failed checks over all entries (one entry can fail several): " +
         (", ".join(f"check {k}: {v}" for k, v in sorted(fail_counts.items())) or "none") + ".", "",
         "## Depth floor per assay class", "",
         "The floor is the 5th percentile of the fragments (paired-end pairs counted once) of the "
         "EIC training base experiments of the class, from the catalog read counts.", "",
         "| assay class | EIC training experiments | floor (fragments) |", "|---|---|---|"]
    for c in sorted(floors):
        L.append(f"| {c} | {floors[c][1]} | {floors[c][0]:,.0f} |")

    def table(title, key, extra=None):
        L.extend(["", f"## Kept tracks per {title}", "", f"| {title} | kept | other lab | same lab |"
                  + (" dropped |" if extra else ""), "|---|---|---|---|" + ("---|" if extra else "")])
        keys = sorted({r[key] for r in rows} | ({d[key] for d in dropped} if extra else set()))
        for k in keys:
            rs = [r for r in rows if r[key] == k]
            ol = sum(1 for r in rs if ";lab" in r["pair_class"])
            line = f"| {k} | {len(rs)} | {ol} | {len(rs) - ol} |"
            if extra:
                line += f" {sum(1 for d in dropped if d[key] == k)} |"
            L.append(line)
    table("split", "split", True)
    table("assay_class", "assay_class", True)
    L.extend(["", "## Kept tracks per pair class", "", "| pair class | first choice | fallback |",
              "|---|---|---|"])
    pc, fc = Counter(r["pair_class"] for r in rows), Counter(r["fallback_pair_class"] for r in rows
                                                             if r["fallback"])
    for k in sorted(set(pc) | set(fc)):
        L.append(f"| {k} | {pc.get(k, 0)} | {fc.get(k, 0)} |")
    L.append(f"| (no fallback) | | {sum(1 for r in rows if not r['fallback'])} |")
    L.extend(["", "## Kept tracks per source", "",
              "A candidate with no served FASTQ is accepted from ENCODE's released redacted "
              "unfiltered GRCh38 BAMs (PI ruling 16, the build's `native-redacted-bam` route). Its "
              "depth is the `mapped` count of the portal's samtools flagstats of the unfiltered BAM, "
              "halved for a paired-end BAM, so it is compared with the floor in fragments.", "",
              "| split | assay class | fastq | redacted_bam |", "|---|---|---|---|"])
    for k in sorted({(r["split"], r["assay_class"]) for r in rows}):
        rs = [r for r in rows if (r["split"], r["assay_class"]) == k]
        nb = sum(1 for r in rs if r["source"] == "redacted_bam")
        L.append(f"| {k[0]} | {k[1]} | {len(rs) - nb} | {nb} |")
    nb = sum(1 for r in rows if r["source"] == "redacted_bam")
    L.append(f"| all | all | {len(rows) - nb} | {nb} |")
    m, lo, hi = build_cost(rows)
    gb = sum(float(r["fastq_gb"] or 0) for r in rows)
    agb = sum(float(r["alignment_gb"] or 0) for r in rows)
    fgb = sum(float(r["fallback_fastq_gb"] or 0) + float(r["fallback_alignment_gb"] or 0)
              for r in rows)
    td = Counter(r["tech_diff"] or "(none)" for r in rows)
    L.extend(["", "## Size and build cost of the first choices", "",
              f"- Staging: {gb + agb:,.1f} GB over {len(rows)} experiments ({gb:,.1f} GB of FASTQ, "
              f"{agb:,.1f} GB of redacted BAM; fallbacks: {fgb:,.1f} GB).",
              f"- Reads (FASTQ: catalog read counts, both mates; BAM: flagstats total): "
              f"{m:,.0f} million.",
              f"- Build cost at the t180 rates (ChIP-seq and ATAC-seq 0.160 to 0.177 CPU-h per "
              f"million reads plus 0.027 for the views; DNase-seq 0.04 plus 0.016; a redacted-BAM row "
              f"at the views rate only for the low figure and at the full rate for the high one): "
              f"{lo:,.0f} to {hi:,.0f} CPU-h, against the planned 2,190 CPU-h.",
              "- Technical differences from the EIC experiment: " +
              ", ".join(f"{k} {v}" for k, v in sorted(td.items())) + ".",
              "- Own controls are recorded only (decision 4); no second experiment is built with a "
              "control or with signal products."])
    if dropped:
        L.extend(["", "## Dropped tracks", "", "| track | split | assay | reason |", "|---|---|---|---|"])
        for d in dropped:
            L.append(f"| {d['eic_track']} | {d['split']} | {d['assay']} | {d['reason']} |")
        L.append("")
        L.append("The failed checks of every candidate are in `dropped.tsv`.")
    L.extend(["", "## Check commands", "",
              "Every census track with a candidate is in exactly one of the two tables, and no "
              "experiment is chosen twice. `FNR>1` skips the header of each file (`NR>1` would "
              "count the header of the second file as a track):", "",
              "```",
              "awk -F'\\t' 'FNR>1' other_experiments.tsv dropped.tsv | cut -f1 | sort -u | wc -l",
              "awk -F'\\t' 'FNR>1{print $9; if ($28 != \"\") print $28}' other_experiments.tsv "
              "| sort | uniq -d | wc -l", "```"])
    return "\n".join(L) + "\n"


def cmd_select(a) -> int:
    census = B.read_tsv(a.census)
    ledger = B.read_tsv(a.ledger)
    build_rows = B.read_tsv(a.build_list)
    cands = {x for r in census for x in candidates_of(r)}
    eic_exps = {r["eic_experiment"] for r in census if candidates_of(r)}
    train_base = {r["experiment"] for r in build_rows if r.get("role") == "train"}
    print(f"[select] {len(cands)} distinct candidates, {len(eic_exps)} EIC experiments", flush=True)
    cat0 = load_catalog(Path(a.catalog), cands | eic_exps | train_base)
    ctls = {c for x in cands for c in B.split_acc(cat0["exps"].get(x, {}).get("control_experiments", ""))}
    cat = load_catalog(Path(a.catalog), cands | eic_exps | train_base | ctls)
    print(f"[select] catalog: {len(cat['exps'])} experiments, {len(ctls)} own controls; "
          f"{len(cands - set(cat['exps']))} candidates missing from experiments.tsv", flush=True)
    portal = portal_meta(cands)
    print(f"[select] portal: {len(portal)} of {len(cands)} candidates returned", flush=True)
    to_head = [f for x in sorted(cands | ctls) for f in released_fastqs(cat["files"].get(x, []))]
    probe = head_probe(to_head, a.threads)
    n_ok = sum(1 for v in probe.values() if v == "200")
    print(f"[select] HEAD: {len(probe)} FASTQs, {n_ok} served; "
          f"codes {dict(Counter(probe.values()))}", flush=True)
    # PI ruling 16 for second experiments: a candidate with no served FASTQ may use ENCODE's
    # redacted unfiltered GRCh38 BAMs (the build's native-redacted-bam route)
    pd = {f: {"http": h} for f, h in probe.items()}
    no_fq = sorted(x for x in cands if not B.available_fastqs(
        released_fastqs(cat["files"].get(x, [])), pd)[0])
    bam_rows = [f for x in no_fq for f in cat["files"].get(x, []) if B.is_native_bam(f)]
    bprobe = head_probe(bam_rows, a.threads)
    probe = {**probe, **bprobe}
    served_bams = [f for x in no_fq for f in B.native_bams(cat["files"].get(x, []),
                                                           {**pd, **{k: {"http": v} for k, v in bprobe.items()}})]
    metrics = bam_metrics([f["file"] for f in served_bams])
    print(f"[select] {len(no_fq)} candidates with no served FASTQ; {len(bam_rows)} redacted BAMs, "
          f"{sum(1 for v in bprobe.values() if v == '200')} served, {len(served_bams)} usable, "
          f"{len(metrics)} with flagstats", flush=True)
    res = select(census, cat, ledger, build_rows, portal, probe, metrics)
    for c, (v, n) in sorted(res["floors"].items()):
        print(f"[select] depth floor {c}: {v:,.0f} fragments (5th percentile of {n})", flush=True)
    B.write_tsv(a.out, OUT_COLUMNS, res["rows"])
    B.write_tsv(a.dropped, DROPPED_COLUMNS, res["dropped"])
    summ = Path(a.summary) if a.summary else Path(a.out).parent / "SELECTION.md"
    summ.write_text(summary_md(res, len(probe), n_ok, a.census))
    print(f"[select] kept {len(res['rows'])}, dropped {len(res['dropped'])}; wrote {a.out}, "
          f"{a.dropped}, {summ}", flush=True)
    return 0


# --------------------------------------------------------------------------------- pair smoke

def _read_regions(path: Path, regions, rb: int):
    import h5py
    import numpy as np
    with h5py.File(path, "r") as f:
        parts = []
        for c, s in regions:
            out = np.zeros(rb, dtype=np.int64)
            if c in f:
                ds = f[c]
                end = min(s + rb, ds.shape[0])
                if end > s:
                    out[: end - s] = ds[s:end]
            parts.append(out)
        depth = int(f.attrs.get("depth_reads", -1))
    return np.concatenate(parts).astype(np.float64), depth


def pair_smoke(pairs: list[tuple[str, str]], evalset: dict) -> list[dict]:
    import numpy as np
    rb = int(evalset.get("region_bins", 2 ** 14))
    regions = [(c, int(s)) for c, s in evalset["regions"]]
    res = []
    for da, db in pairs:
        a, Da = _read_regions(Path(da) / "counts.h5", regions, rb)
        b, Db = _read_regions(Path(db) / "counts.h5", regions, rb)
        ida, idb = Path(da).name, Path(db).name
        row = {"a": ida, "b": idb, "depth_a": Da, "depth_b": Db}
        if not (Da > 0 and Db > 0):
            res.append({**row, "ratio": float("nan"), "outside": True, "note": "no declared depth"})
            continue
        r = Da / Db
        rng = np.random.default_rng(zlib.crc32(f"h7:{ida}:{idb}".encode()))
        if r < 1.0:
            b = rng.binomial(b.astype(np.int64), r).astype(np.float64)
        elif r > 1.0:
            a = rng.binomial(a.astype(np.int64), 1.0 / r).astype(np.float64)
        sa, sb = float(a.sum()), float(b.sum())
        ratio = sa / sb if sb > 0 else float("nan")
        outside = not (sa > 0 and sb > 0) or abs(ratio - 1.0) > TOTAL_TOL
        res.append({**row, "sum_a": sa, "sum_b": sb, "ratio": ratio, "outside": outside,
                    "note": ""})
    return res


def cmd_pair_smoke(a) -> int:
    pairs = []
    for tok in a.pairs.split(","):
        x, _, y = tok.partition(":")
        if not (x and y):
            raise SystemExit(f"bad pair {tok!r}: want <a dir>:<b dir>")
        pairs.append((x, y))
    ev = json.loads(Path(a.evalset).read_text())
    res = pair_smoke(pairs, ev)
    n_out = sum(1 for r in res if r["outside"])
    L = ["# Pair smoke: replicate totals after depth matching", "",
         f"Eval set `{a.evalset}`, {len(ev['regions'])} regions of {ev.get('region_bins', 2 ** 14)} "
         f"bins. A pair outside 1 +- {TOTAL_TOL} gives an Undefined headline-7 ratio "
         "(`metrics.variance_ratio`).", "",
         "| a | b | depth a | depth b | sum(a)/sum(b) | outside |", "|---|---|---|---|---|---|"]
    for r in res:
        L.append(f"| {r['a']} | {r['b']} | {r['depth_a']} | {r['depth_b']} | "
                 f"{r['ratio']:.4f} | {'yes' if r['outside'] else 'no'} |")
    L.extend(["", f"{n_out} of {len(res)} pairs are outside the tolerance."])
    Path(a.out).write_text("\n".join(L) + "\n")
    for r in res:
        print(f"{r['a']}\t{r['b']}\t{r['ratio']:.4f}\t{'OUTSIDE' if r['outside'] else 'ok'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--census", required=True)
    s.add_argument("--catalog", required=True)
    s.add_argument("--ledger", required=True)
    s.add_argument("--build-list", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--dropped", required=True)
    s.add_argument("--summary", help="default: SELECTION.md beside --out")
    s.add_argument("--threads", type=int, default=16)
    p = sub.add_parser("pair-smoke")
    p.add_argument("--pairs", required=True)
    p.add_argument("--evalset", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return cmd_select(a) if a.cmd == "select" else cmd_pair_smoke(a)


if __name__ == "__main__":
    sys.exit(main())
