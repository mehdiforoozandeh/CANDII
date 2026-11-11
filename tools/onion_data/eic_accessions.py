#!/usr/bin/env python3
"""EIC accession ledger and its hard check against the challenge's own published list (t158, D8).

The published list is Additional file 5 of Schreiber et al., Genome Biology (2023),
doi:10.1186/s13059-023-02915-y: 363 tracks (C##M##) with their role and ENCODE experiment. It is
committed as `tests/fixtures/eic_published_accessions.tsv` with its URL and md5 in the header. The
check never derives the published side from our own store.

    eic_accessions.py build --manifest <eic manifest.json> [--metadata <eic_metadata.t78p3.csv>]
        --published <fixture tsv> --out <ledger tsv> [--encode-cache <json>] [--testbed <dir>]
    eic_accessions.py build --store <CANDI_STORE/eic> ...            (manifest = <store>/manifest.json)
    eic_accessions.py check --ledger <ledger tsv> [--published <fixture tsv>]
    eic_accessions.py imbalance --ledger <ledger tsv> --out <eic_imbalance.tsv>

`build` joins the store manifest to the published list on (role, cell, assay), and adds the
ENCODE portal facts of every experiment (status, replicates, libraries, biosamples, possible
controls), of every store file (pipeline) and of every experiment's FASTQs. The portal answers are
cached in `--encode-cache`: an existing cache is read, never refetched, so a rebuild is offline.
`check` re-reads the published list and fails (exit 1, one line per mismatch) when a published
track is missing, a ledger track is not published, a role or experiment differs, the store's
file belongs to another ENCODE experiment than the published one, or a control is not a possible
control of a ChIP experiment of its cell in the same role. It ends with `mismatches: N`.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
DEFAULT_PUBLISHED = HERE.parents[2] / "tests" / "fixtures" / "eic_published_accessions.tsv"

ROLE_OF_DATA_TYPE = {"training_data": "train", "validation_data": "val", "blind_data": "blind"}
ROLE_OF_PREFIX = {"T": "train", "V": "val", "B": "blind"}
ROLES = ("train", "val", "blind")
CONTROL_ASSAY = "chipseq-control"
CONTROL_MARK = "CTL"   # ledger eic_track of a control row is C##CTL (EIC published no controls)

# Pinned interface B. ENCODE's histone standards page (read 2026-11-10) lists as broad: H3F3A,
# H3K27me3, H3K36me3, H3K4me1, H3K79me2, H3K79me3, H3K9me1, H3K9me2, H4K20me1; H3K9me3 as an
# exception. The plan's list adds H3K79me1 and H3K9me3; it is kept as pinned (see LEDGER_CHECK.md).
BROAD_HISTONE = {"H3F3A", "H3K27me3", "H3K36me3", "H3K4me1", "H3K79me1", "H3K79me2", "H3K9me1",
                 "H3K9me2", "H3K9me3", "H4K20me1"}

LEDGER_COLUMNS = [
    # pinned by the plan, in order
    "eic_track", "role", "cell", "assay", "assay_class", "experiment", "biosample_accession",
    "published_experiment", "match", "run_type", "read_length", "platform", "lab", "depth",
    "replication_type", "n_bio_reps", "n_libraries", "control_experiment", "in_testbed", "note",
    # additive: what D4 and the imbalance table need
    "store_biosample", "file_accession", "file_experiment", "file_bio_reps", "store_bios_accession", "pipeline",
    "encode_status", "encode_target", "encode_possible_controls", "fastq_run_types",
    "fastq_read_lengths", "fastq_platforms", "n_fastq",
]

ENCODE = "https://www.encodeproject.org/search/"
EXP_FIELDS = ["accession", "status", "assay_term_name", "target.label",
              "biosample_ontology.term_name", "lab.title", "replication_type",
              "replicates.biological_replicate_number", "replicates.technical_replicate_number",
              "replicates.library.accession", "replicates.library.biosample.accession",
              "possible_controls.accession", "superseded_by"]
FILE_FIELDS = ["accession", "status", "dataset", "output_type", "file_format", "assembly",
               "biological_replicates", "technical_replicates", "lab.title",
               "analysis_step_version.analysis_step.pipelines.title"]
FASTQ_FIELDS = ["accession", "status", "dataset", "run_type", "read_length",
                "platform.term_name", "read_count", "replicate.biological_replicate_number",
                "replicate.technical_replicate_number"]


def assay_class(assay: str) -> str:
    if assay == "DNase-seq":
        return "accessibility_dnase"
    if assay == "ATAC-seq":
        return "accessibility_atac"
    if assay == CONTROL_ASSAY:
        return "control"
    return "broad_histone" if assay in BROAD_HISTONE else "narrow_histone"


# ---------------------------------------------------------------- published list

def load_published(path) -> list[dict]:
    with open(path) as fh:
        lines = [ln for ln in fh if not ln.startswith("#") and ln.strip()]
    rows = list(csv.DictReader(lines, delimiter="\t"))
    for r in rows:
        r["role"] = ROLE_OF_DATA_TYPE[r["data_type"]]
        r["eic_track"] = r["cell_type_id"] + r["mark_id"]
    return rows


def published_maps(pub: list[dict]):
    """cell name -> C##, assay -> M##, (role, track) -> row."""
    cell_id, mark_id = {}, {}
    for r in pub:
        cell_id[r["cell_type"]] = r["cell_type_id"]
        mark_id[r["assay"]] = r["mark_id"]
    by_key = {(r["role"], r["eic_track"]): r for r in pub}
    return cell_id, mark_id, by_key


# ---------------------------------------------------------------- ENCODE portal

def _search(params: list[tuple[str, str]], retries: int = 4) -> list[dict]:
    url = ENCODE + "?" + urllib.parse.urlencode(params + [("format", "json"), ("limit", "all")])
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "eic-accession-ledger/1"})
    for k in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp).get("@graph", [])
        except urllib.error.HTTPError as e:
            if e.code == 404:          # the portal answers 404 for an empty search
                return []
            err = e
        except Exception as e:  # noqa: BLE001  (network trouble: retry)
            err = e
        time.sleep(2 * (k + 1))
    raise RuntimeError(f"ENCODE search failed after {retries} tries: {err}")


def _batched(kind, key, values, fields, extra=(), size=40):
    out = []
    values = sorted(set(values))
    for i in range(0, len(values), size):
        params = [("type", kind), ("status", "*")] + list(extra)
        params += [(key, v) for v in values[i:i + size]] + [("field", f) for f in fields]
        out += _search(params)
    return out


def fetch_encode(experiments, files) -> dict:
    exps = _batched("Experiment", "accession", experiments, EXP_FIELDS)
    fls = _batched("File", "accession", files, FILE_FIELDS)
    fq = _batched("File", "dataset", [f"/experiments/{e}/" for e in experiments], FASTQ_FIELDS,
                  extra=[("file_format", "fastq")], size=25)
    fastqs = defaultdict(list)
    for f in fq:
        fastqs[f["dataset"].strip("/").split("/")[-1]].append(f)
    return {"retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "experiments": {e["accession"]: e for e in exps},
            "files": {f["accession"]: f for f in fls},
            "fastqs": dict(fastqs)}


def _pipeline(fobj: dict | None, assay: str) -> str:
    if not fobj:
        return ""
    titles = [p.get("title", "") for p in
              fobj.get("analysis_step_version", {}).get("analysis_step", {}).get("pipelines", [])]
    titles = sorted({t.replace(" (unreplicated)", "") for t in titles if t})
    if not titles:
        return "none recorded"
    want = "Histone" if assay not in ("DNase-seq", "ATAC-seq", CONTROL_ASSAY) else assay.split("-")[0]
    pick = [t for t in titles if want.lower() in t.lower()]
    return (pick or titles)[0]


def _joined(values) -> str:
    return ";".join(str(v) for v in sorted({v for v in values if v not in (None, "")}, key=str))


# ---------------------------------------------------------------- build

def store_tracks(manifest: dict):
    for bios, b in sorted(manifest["biosamples"].items()):
        prefix, cell = bios.split("_", 1)
        for t in b["tracks"]:
            yield bios, ROLE_OF_PREFIX[prefix], cell, t


def build_ledger(manifest: dict, pub: list[dict], encode: dict, testbed: set[str],
                 metadata: list[dict] | None = None) -> list[dict]:
    cell_id, mark_id, pub_by_key = published_maps(pub)
    exps, files, fastqs = encode["experiments"], encode["files"], encode["fastqs"]
    meta_exp = {(m["biosample_name"], m["assay_name"]): m["exp_accession"] for m in metadata or []}
    rows, seen = [], set()
    chip_by_cell = defaultdict(set)      # (role, cell) -> store ChIP experiments
    controls = []
    for bios, role, cell, t in store_tracks(manifest):
        assay = t["assay"]
        cid = cell_id.get(cell, "")
        notes = []
        if not cid:
            notes.append(f"cell '{cell}' not in published list")
        if assay == CONTROL_ASSAY:
            controls.append((bios, role, cell, cid, t))
            continue
        mid = mark_id.get(assay, "")
        track = (cid or "C??") + (mid or "M??")
        p = pub_by_key.get((role, track))
        exp = t["exp_accession"]
        if (bios, assay) in meta_exp and meta_exp[(bios, assay)] != exp:
            notes.append(f"metadata csv names {meta_exp[(bios, assay)]}")
        if p is None:
            other = [r for (rl, tr), r in pub_by_key.items() if tr == track]
            notes.append(f"published under role {other[0]['role']}" if other
                         else "track not in published list")
            match = 0
        else:
            seen.add((role, track))
            match = int(p["experiment"] == exp)
            if not match:
                notes.append(f"store experiment {exp} differs from published {p['experiment']}")
        if assay != "DNase-seq" and assay != "ATAC-seq":
            chip_by_cell[(role, cell)].add(exp)
        rows.append(_row(track, role, cell, assay, exp, p["experiment"] if p else "", match,
                         t, bios, exps, files, fastqs, testbed, notes))
    for (role, track), p in sorted(pub_by_key.items()):
        if (role, track) not in seen:
            rows.append({c: "" for c in LEDGER_COLUMNS} | {
                "eic_track": track, "role": role, "cell": p["cell_type"], "assay": p["assay"],
                "assay_class": assay_class(p["assay"]), "published_experiment": p["experiment"],
                "match": 0, "in_testbed": int(track in testbed),
                "note": "published track missing from the store"})
    for bios, role, cell, cid, t in controls:
        exp = t["exp_accession"]
        possible = set()
        for chip in chip_by_cell[(role, cell)]:
            possible |= {c["accession"] for c in exps.get(chip, {}).get("possible_controls", [])}
        notes = ["EIC published no controls; checked against ENCODE possible_controls of the "
                 "cell's ChIP experiments"]
        match = int(exp in possible)
        if not match:
            notes.append(f"not a possible control of {_joined(chip_by_cell[(role, cell)])}"
                         if possible else f"the {role} split of this cell holds no ChIP "
                                          "experiment with possible_controls")
            owners = sorted(f"{rl}:{chip}" for (rl, c), chips in chip_by_cell.items()
                            if c == cell and rl != role for chip in chips
                            if exp in {x["accession"] for x in
                                       exps.get(chip, {}).get("possible_controls", [])})
            if owners:
                roles = sorted({o.split(":")[0] for o in owners})
                notes.append(f"it is the control of {len(owners)} {'/'.join(roles)} ChIP "
                             f"experiment(s) of this cell; proposed fix: drop this {role} "
                             "control track")
        r = _row((cid or "C??") + CONTROL_MARK, role, cell, CONTROL_ASSAY, exp, "", match, t,
                 bios, exps, files, fastqs, testbed, notes)
        r["encode_possible_controls"] = _joined(possible)
        rows.append(r)
    # each ChIP row names the control the store pairs with its cell, when ENCODE agrees
    ctl_of = {(r["role"], r["cell"]): r["experiment"] for r in rows if r["assay"] == CONTROL_ASSAY}
    for r in rows:
        if r["assay_class"] in ("broad_histone", "narrow_histone") and r["experiment"]:
            possible = set(r["encode_possible_controls"].split(";")) - {""}
            store_ctl = ctl_of.get((r["role"], r["cell"]), "")
            if store_ctl and store_ctl in possible:
                r["control_experiment"] = store_ctl
            else:
                r["control_experiment"] = r["encode_possible_controls"]
                if store_ctl:
                    r["note"] = _add(r["note"], f"store control {store_ctl} is not among "
                                                "ENCODE possible_controls")
    rows.sort(key=lambda r: (r["eic_track"], ROLES.index(r["role"])))
    return rows


def _add(note: str, more: str) -> str:
    return f"{note}; {more}" if note else more


def _row(track, role, cell, assay, exp, pub_exp, match, t, bios, exps, files, fastqs, testbed,
         notes) -> dict:
    e = exps.get(exp)
    f = files.get(t["file_accession"])
    fq = fastqs.get(exp, [])
    reps = (e or {}).get("replicates", [])
    file_reps = set((f or {}).get("biological_replicates", []))
    bios_acc = {r["library"]["biosample"]["accession"] for r in reps
                if r.get("biological_replicate_number") in file_reps and "library" in r}
    if e is None:
        notes.append("experiment not found on the ENCODE portal")
    else:
        if e.get("status") != "released":
            notes.append(f"ENCODE status {e.get('status')}")
        target = (e.get("target") or {}).get("label") or e.get("assay_term_name", "")
        if assay not in (CONTROL_ASSAY,) and target and target != assay:
            notes.append(f"ENCODE target {target}")
    if f is None:
        notes.append(f"file {t['file_accession']} not found on the ENCODE portal")
    elif _acc(f.get("dataset", "")) != exp:
        notes.append(f"store file {t['file_accession']} belongs to {_acc(f.get('dataset', ''))}, "
                     f"not to {exp}; proposed fix: rebuild from the FASTQs of {exp}")
    if bios_acc and t.get("bios_accession") not in bios_acc:
        notes.append(f"store biosample {t.get('bios_accession')} is not the file's biosample")
    return {
        "eic_track": track, "role": role, "cell": cell, "assay": assay,
        "assay_class": assay_class(assay), "experiment": exp,
        "biosample_accession": _joined(bios_acc), "published_experiment": pub_exp,
        "match": match, "run_type": t.get("run_type", ""), "read_length": t.get("read_length", ""),
        "platform": t.get("platform", ""), "lab": t.get("lab", ""), "depth": t.get("depth", ""),
        "replication_type": (e or {}).get("replication_type", ""),
        "n_bio_reps": len({r.get("biological_replicate_number") for r in reps}) if e else "",
        "n_libraries": len({r["library"]["accession"] for r in reps if "library" in r}) if e else "",
        "control_experiment": "", "in_testbed": int(track in testbed), "note": "; ".join(notes),
        "store_biosample": bios, "file_accession": t["file_accession"],
        "file_experiment": _acc((f or {}).get("dataset", "")),
        "file_bio_reps": _joined(file_reps), "store_bios_accession": t.get("bios_accession", ""),
        "pipeline": _pipeline(f, assay), "encode_status": (e or {}).get("status", "missing"),
        "encode_target": ((e or {}).get("target") or {}).get("label", ""),
        "encode_possible_controls": _joined(c["accession"] for c in
                                            (e or {}).get("possible_controls", [])),
        "fastq_run_types": _joined(x.get("run_type") for x in fq),
        "fastq_read_lengths": _joined(x.get("read_length") for x in fq),
        "fastq_platforms": _joined((x.get("platform") or {}).get("term_name") for x in fq),
        "n_fastq": len(fq),
    }


def _acc(at_id: str) -> str:
    return at_id.strip("/").split("/")[-1] if at_id else ""


def write_tsv(rows: list[dict], path, columns) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, columns, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_tsv(path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


# ---------------------------------------------------------------- check

def check_ledger(ledger: list[dict], pub: list[dict]) -> list[str]:
    """One line per mismatch. The published side comes from `pub` only, never from the ledger."""
    out = []
    _, _, pub_by_key = published_maps(pub)
    pub_roles = defaultdict(set)
    for role, track in pub_by_key:
        pub_roles[track].add(role)
    led = defaultdict(list)
    for r in ledger:
        led[(r["role"], r["eic_track"])].append(r)
    for key, rs in sorted(led.items()):
        if len(rs) > 1:
            out.append(f"DUPLICATE {key[1]} role={key[0]}: {len(rs)} ledger rows")
    for (role, track), p in sorted(pub_by_key.items()):
        rs = led.get((role, track))
        if not rs:
            out.append(f"MISSING {track} role={role} published={p['experiment']}: "
                       "published track has no ledger row with an experiment")
        elif not rs[0]["experiment"]:
            out.append(f"MISSING {track} role={role} published={p['experiment']}: "
                       f"{rs[0]['note'] or 'no store track'}")
        elif rs[0]["experiment"] != p["experiment"]:
            out.append(f"EXPERIMENT {track} role={role} ledger={rs[0]['experiment']} "
                       f"published={p['experiment']}: {rs[0]['note']}")
    for (role, track), rs in sorted(led.items()):
        r = rs[0]
        if r.get("file_experiment") and r["experiment"] and r["file_experiment"] != r["experiment"]:
            out.append(f"FILE {track} role={role} experiment={r['experiment']} store file "
                       f"{r['file_accession']} belongs to {r['file_experiment']}")
        if r["assay"] == CONTROL_ASSAY:
            if r.get("match") != "1":
                out.append(f"CONTROL {track} role={role} experiment={r['experiment']}: "
                           f"{r['note']}")
            continue
        if (role, track) not in pub_by_key:
            if pub_roles.get(track):
                out.append(f"ROLE {track} ledger role={role} published role="
                           f"{_joined(pub_roles[track])} experiment={r['experiment']}")
            else:
                out.append(f"EXTRA {track} role={role} experiment={r['experiment']}: "
                           "not in the published list")
        elif str(r.get("match")) != str(int(r["experiment"] == pub_by_key[(role, track)]["experiment"])):
            out.append(f"MATCHFLAG {track} role={role}: ledger match={r.get('match')} "
                       "disagrees with the published list")
    return out


# ---------------------------------------------------------------- imbalance

def read_length_bin(v) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "unknown"
    return "<=36" if x <= 36 else ("37-75" if x <= 75 else ">=76")


def depth_terciles(rows: list[dict]) -> dict[int, str]:
    by_assay = defaultdict(list)
    for i, r in enumerate(rows):
        try:
            by_assay[r["assay"]].append((float(r["depth"]), i))
        except (TypeError, ValueError):
            pass
    out = {}
    for items in by_assay.values():
        items.sort()
        n = len(items)
        for rank, (_, i) in enumerate(items):
            out[i] = ("low", "mid", "high")[min(2, rank * 3 // n)]
    return out


COVARIATES = ("run_type", "read_length_bin", "depth_tercile_within_assay", "lab", "platform",
              "pipeline")


def imbalance(ledger: list[dict]):
    """Per covariate value: counts and fractions in train, val, blind (EIC tracks only), plus a
    summary with the total variation distance (TVD) of val and blind against train, raw and with
    train reweighted to the assay mix of the other split."""
    rows = [r for r in ledger if r["assay"] != CONTROL_ASSAY and r["experiment"]]
    terc = depth_terciles(rows)
    for i, r in enumerate(rows):
        r = rows[i] = dict(r)
        r["read_length_bin"] = read_length_bin(r["read_length"])
        r["depth_tercile_within_assay"] = terc.get(i, "unknown")
        r["pipeline"] = r.get("pipeline") or "unknown"
    n_role = Counter(r["role"] for r in rows)
    table, summary = [], []
    for cov in COVARIATES:
        cnt = {role: Counter(r[cov] for r in rows if r["role"] == role) for role in ROLES}
        values = sorted(set().union(*cnt.values()), key=lambda v: -sum(c[v] for c in cnt.values()))
        frac = {role: {v: cnt[role][v] / max(1, n_role[role]) for v in values} for role in ROLES}
        for v in values:
            table.append({"covariate": cov, "value": v,
                          **{f"n_{x}": cnt[x][v] for x in ROLES},
                          **{f"frac_{x}": f"{frac[x][v]:.3f}" for x in ROLES}})
        s = {"covariate": cov, "n_values": len(values)}
        for other in ("val", "blind"):
            s[f"tvd_train_vs_{other}"] = f"{_tvd(frac['train'], frac[other]):.3f}"
            s[f"tvd_train_vs_{other}_assay_matched"] = f"{_tvd(_reweighted(rows, cov, other), frac[other]):.3f}"
            only = [v for v in values if cnt[other][v] and not cnt["train"][v]]
            s[f"{other}_tracks_with_value_absent_in_train"] = sum(cnt[other][v] for v in only)
        summary.append(s)
    return table, summary


def _tvd(p: dict, q: dict) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def _reweighted(rows, cov, other) -> dict:
    """Train distribution of `cov` after weighting each train track so that train's assay mix
    equals `other`'s (assays absent from train are dropped from the target mix)."""
    tr = [r for r in rows if r["role"] == "train"]
    ot = Counter(r["assay"] for r in rows if r["role"] == other)
    tr_a = Counter(r["assay"] for r in tr)
    keep = {a: n for a, n in ot.items() if tr_a[a]}
    tot = sum(keep.values())
    out = Counter()
    for r in tr:
        if r["assay"] in keep:
            out[r[cov]] += keep[r["assay"]] / tot / tr_a[r["assay"]]
    return dict(out)


# ---------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    src = b.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest")
    src.add_argument("--store", help="corpus root; reads <store>/manifest.json")
    b.add_argument("--metadata", help="eic_metadata.t78p3.csv (cross-check of experiments)")
    b.add_argument("--published", default=str(DEFAULT_PUBLISHED))
    b.add_argument("--out", required=True)
    b.add_argument("--encode-cache", help="json of portal answers; fetched and written if absent")
    b.add_argument("--testbed", help="dir of testbed products named C##M##__...")
    c = sub.add_parser("check")
    c.add_argument("--ledger", required=True)
    c.add_argument("--published", default=str(DEFAULT_PUBLISHED))
    im = sub.add_parser("imbalance")
    im.add_argument("--ledger", required=True)
    im.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "build":
        manifest = json.loads(Path(a.manifest or Path(a.store) / "manifest.json").read_text())
        pub = load_published(a.published)
        metadata = None
        if a.metadata:
            with open(a.metadata) as fh:
                metadata = list(csv.DictReader(fh))
        cache = Path(a.encode_cache) if a.encode_cache else None
        if cache and cache.exists():
            encode = json.loads(cache.read_text())
        else:
            exps = {t["exp_accession"] for *_, t in store_tracks(manifest)}
            exps |= {r["experiment"] for r in pub}
            fls = {t["file_accession"] for *_, t in store_tracks(manifest)}
            encode = fetch_encode(exps, fls)
            if cache:
                cache.write_text(json.dumps(encode, sort_keys=True))
        testbed = set()
        if a.testbed:
            testbed = {p.name.split("__")[0] for p in Path(a.testbed).iterdir() if "__" in p.name}
        rows = build_ledger(manifest, pub, encode, testbed, metadata)
        write_tsv(rows, a.out, LEDGER_COLUMNS)
        print(f"ledger: {a.out} rows={len(rows)} "
              + " ".join(f"{k}={v}" for k, v in sorted(Counter(
                  (r['role'] + ('_control' if r['assay'] == CONTROL_ASSAY else ''))
                  for r in rows).items())))
        print(f"encode: experiments={len(encode['experiments'])} files={len(encode['files'])} "
              f"experiments_with_fastqs={len(encode['fastqs'])}")
        return 0
    if a.cmd == "check":
        lines = check_ledger(read_tsv(a.ledger), load_published(a.published))
        for ln in lines:
            print(ln)
        print(f"mismatches: {len(lines)}")
        return 1 if lines else 0
    if a.cmd == "imbalance":
        table, summary = imbalance(read_tsv(a.ledger))
        write_tsv(table, a.out, list(table[0]))
        spath = Path(a.out).with_name(Path(a.out).stem + "_summary.tsv")
        write_tsv(summary, spath, list(summary[0]))
        print(f"imbalance: {a.out} rows={len(table)}; summary: {spath}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
