"""t112 — the per-product JSON records (`covariates.json`, `provenance.json`) and `MANIFEST.tsv`.

Every counterfactual product lives in `<products>/<pid>/` and holds `counts25.npz`, `pval25.npz`,
`covariates.json`, `provenance.json`, and `control_counts25.npz` exactly when a control was used.
This module is the one place that says what a valid record is. The writers refuse an invalid
record (`ValueError` naming the key) so a bad file is never left on disk for a later chunk to find.

`covariates.json` carries CANDI's four experimental covariates under the names the loader uses
(`DATA.md` "Four covariate rows"): `depth` (row 0 is `log2(depth)`; `log2_depth` is stored beside
it), `assay` (row 1, `assay_id`, is derived from the assay name), `read_length`, and `run_type`
as the `file_metadata.json` string ("single-ended" / "paired-ended").

Deliberately stdlib only: `validate` runs on Nibi's login-node python3 (no numpy). Product ids,
biosample names and the rows-TSV header come from the sibling `arms.py`, never re-declared here.

    python3 records.py manifest [--products DIR] --out MANIFEST.tsv
    python3 records.py validate [--products DIR] [--rows ROWS_TSV] [--expect N]

`validate` prints one line per problem, then `OK <n>` (exit 0) or `FAIL <n>` (exit 1).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms  # noqa: E402  (sibling, stdlib only)

#: `$CF/products` — the default for `--products`, so `validate --rows R --expect N` and
#: `manifest --out M` work as the plan writes them.
DEFAULT_PRODUCTS = "/scratch/mforooz/t112_cf/products"

SCHEMA = 1

COUNT_RULE = ("store overlap rule: +1 to bins floor(start/25)..floor(end/25) inclusive, "
              "grid floor(len/25)")
WORDING_NOTE = ("t112 says read-start counts; products use the store overlap rule so the loader "
                "reads them unchanged")

COVARIATE_KEYS = ("schema", "pid", "biosample", "track", "cell", "assay", "arm", "level", "knob",
                  "knob_value", "depth", "log2_depth", "read_length", "run_type", "fraglen",
                  "control", "count_rule", "wording_note")

PROVENANCE_KEYS = ("schema", "pid", "route", "pipeline", "commands", "inputs", "outputs",
                   "subsample_seed", "patched_script", "caper", "slurm_job_ids", "code",
                   "created_utc")

#: repo -> (release, sif md5). md5s from `$EIC/sif/sif_md5.txt`, verified on Nibi 2026-09-17.
PIPELINES = {
    "ENCODE-DCC/chip-seq-pipeline2": ("v2.2.2", "f6e408f7e77bafde4556883f33552191"),
    "ENCODE-DCC/atac-seq-pipeline": ("v2.2.3", "04d9fa482d67cf633845653750f58cef"),
}

#: the rows-TSV `pipeline` column (arms.py) -> the repo its products must record.
PIPELINE_REPO = {"chip": "ENCODE-DCC/chip-seq-pipeline2", "atac": "ENCODE-DCC/atac-seq-pipeline"}

#: `store/dataset.py::_RUN_TYPE_ID` keys — the strings the loader maps to covariate row 3.
RUN_TYPES = ("single-ended", "paired-ended")
CONTROL_SOURCES = ("matched", "other")
ROUTES = ("bam", "fastq")

MANIFEST_COLUMNS = ("pid", "biosample", "track", "cell", "assay", "arm", "level", "route", "knob",
                    "knob_value", "depth", "read_length", "run_type", "fraglen",
                    "control_accession", "control_source", "pipeline_repo", "pipeline_release",
                    "sif_md5", "counts25_md5", "pval25_md5", "control_counts25_md5",
                    "slurm_job_ids", "product_dir")

_MD5 = re.compile(r"^[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{7,40}$")


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v))


def _is_str(v) -> bool:
    return isinstance(v, str) and v != ""


def _exact_keys(d, keys, where) -> list:
    """Missing and unexpected keys of dict `d`, each as one problem line."""
    if not isinstance(d, dict):
        return [f"{where}: must be an object"]
    out = [f"{where}: missing key {k!r}" for k in keys if k not in d]
    out += [f"{where}: unexpected key {k!r}" for k in d if k not in keys]
    return out


def covariates_problems(d) -> list:
    """Every way `d` fails the covariates.json schema, one line each. Empty list = valid."""
    probs = _exact_keys(d, COVARIATE_KEYS, "covariates")
    if not isinstance(d, dict):
        return probs
    g = d.get
    if "schema" in d and g("schema") != SCHEMA:
        probs.append(f"covariates.schema: expected {SCHEMA}, got {g('schema')!r}")
    for k in ("pid", "biosample", "track", "cell", "assay", "arm", "level", "knob"):
        if k in d and not _is_str(g(k)):
            probs.append(f"covariates.{k}: must be a non-empty string, got {g(k)!r}")
    if all(_is_str(g(k)) for k in ("pid", "track", "arm", "level")):
        want = arms.pid(g("track"), g("arm"), g("level"))
        if g("pid") != want:
            probs.append(f"covariates.pid: {g('pid')!r} != track__arm__level {want!r}")
    if all(_is_str(g(k)) for k in ("biosample", "cell", "arm", "level")):
        want = arms.biosample(g("cell"), g("arm"), g("level"))
        if g("biosample") != want:
            probs.append(f"covariates.biosample: {g('biosample')!r} != {want!r}")
    if "knob_value" in d and not (g("knob_value") is None or isinstance(g("knob_value"), bool)
                                  or _is_num(g("knob_value")) or isinstance(g("knob_value"), str)):
        probs.append(f"covariates.knob_value: must be null, bool, number or string, "
                     f"got {g('knob_value')!r}")
    for k in ("depth", "read_length", "fraglen"):
        if k in d and not (_is_int(g(k)) and g(k) > 0):
            probs.append(f"covariates.{k}: must be a positive integer, got {g(k)!r}")
    if "log2_depth" in d:
        if not _is_num(g("log2_depth")):
            probs.append(f"covariates.log2_depth: must be a finite number, got {g('log2_depth')!r}")
        elif _is_int(g("depth")) and g("depth") > 0 \
                and abs(g("log2_depth") - math.log2(g("depth"))) > 1e-6:
            probs.append(f"covariates.log2_depth: {g('log2_depth')!r} != log2(depth) "
                         f"{math.log2(g('depth'))!r}")
    if "run_type" in d and g("run_type") not in RUN_TYPES:
        probs.append(f"covariates.run_type: must be one of {RUN_TYPES}, got {g('run_type')!r}")
    if "control" in d and g("control") is not None:
        c = g("control")
        cp = _exact_keys(c, ("accession", "source", "reads"), "covariates.control")
        if isinstance(c, dict):
            if "accession" in c and not _is_str(c["accession"]):
                cp.append(f"covariates.control.accession: must be a non-empty string, "
                          f"got {c['accession']!r}")
            if "source" in c and c["source"] not in CONTROL_SOURCES:
                cp.append(f"covariates.control.source: must be one of {CONTROL_SOURCES}, "
                          f"got {c['source']!r}")
            if "reads" in c and not (_is_int(c["reads"]) and c["reads"] > 0):
                cp.append(f"covariates.control.reads: must be a positive integer, "
                          f"got {c['reads']!r}")
        probs += cp
    if "count_rule" in d and g("count_rule") != COUNT_RULE:
        probs.append(f"covariates.count_rule: must be the fixed string {COUNT_RULE!r}")
    if "wording_note" in d and g("wording_note") != WORDING_NOTE:
        probs.append(f"covariates.wording_note: must be the fixed string {WORDING_NOTE!r}")
    return probs


def _path_md5_list(v, where) -> list:
    if not isinstance(v, list) or not v:
        return [f"{where}: must be a non-empty list"]
    probs = []
    for i, e in enumerate(v):
        w = f"{where}[{i}]"
        probs += _exact_keys(e, ("path", "md5"), w)
        if isinstance(e, dict):
            if "path" in e and not _is_str(e["path"]):
                probs.append(f"{w}.path: must be a non-empty string, got {e['path']!r}")
            if "md5" in e and not (isinstance(e["md5"], str) and _MD5.match(e["md5"])):
                probs.append(f"{w}.md5: must be 32 lowercase hex, got {e['md5']!r}")
    return probs


def provenance_problems(d) -> list:
    """Every way `d` fails the provenance.json schema, one line each. Empty list = valid."""
    probs = _exact_keys(d, PROVENANCE_KEYS, "provenance")
    if not isinstance(d, dict):
        return probs
    g = d.get
    if "schema" in d and g("schema") != SCHEMA:
        probs.append(f"provenance.schema: expected {SCHEMA}, got {g('schema')!r}")
    if "pid" in d and not _is_str(g("pid")):
        probs.append(f"provenance.pid: must be a non-empty string, got {g('pid')!r}")
    if "route" in d and g("route") not in ROUTES:
        probs.append(f"provenance.route: must be one of {ROUTES}, got {g('route')!r}")

    if "pipeline" in d:
        p = g("pipeline")
        probs += _exact_keys(p, ("repo", "release", "sif", "sif_md5", "sif_sha256"),
                             "provenance.pipeline")
        if isinstance(p, dict):
            if "repo" in p and p["repo"] not in PIPELINES:
                probs.append(f"provenance.pipeline.repo: must be one of {tuple(PIPELINES)}, "
                             f"got {p['repo']!r}")
            elif "repo" in p:
                release, md5 = PIPELINES[p["repo"]]
                if "release" in p and p["release"] != release:
                    probs.append(f"provenance.pipeline.release: {p['repo']} is pinned at "
                                 f"{release}, got {p['release']!r}")
                if "sif_md5" in p and p["sif_md5"] != md5:
                    probs.append(f"provenance.pipeline.sif_md5: {p['repo']} SIF md5 is {md5}, "
                                 f"got {p['sif_md5']!r}")
            if "sif" in p and not _is_str(p["sif"]):
                probs.append(f"provenance.pipeline.sif: must be a non-empty string, "
                             f"got {p['sif']!r}")
            if "sif_sha256" in p and not (p["sif_sha256"] is None or (
                    isinstance(p["sif_sha256"], str) and _SHA256.match(p["sif_sha256"]))):
                probs.append(f"provenance.pipeline.sif_sha256: must be null or 64 lowercase hex, "
                             f"got {p['sif_sha256']!r}")

    if "commands" in d:
        c = g("commands")
        if not (isinstance(c, list) and c and all(_is_str(x) for x in c)):
            probs.append("provenance.commands: must be a non-empty list of non-empty strings")
    if "inputs" in d:
        probs += _path_md5_list(g("inputs"), "provenance.inputs")
    if "outputs" in d:
        probs += _path_md5_list(g("outputs"), "provenance.outputs")

    if "subsample_seed" in d:
        s = g("subsample_seed")
        if not isinstance(s, list):
            probs.append("provenance.subsample_seed: must be a list")
        else:
            for i, e in enumerate(s):
                w = f"provenance.subsample_seed[{i}]"
                probs += _exact_keys(e, ("file", "uncompressed_bytes"), w)
                if isinstance(e, dict):
                    if "file" in e and not _is_str(e["file"]):
                        probs.append(f"{w}.file: must be a non-empty string, got {e['file']!r}")
                    if "uncompressed_bytes" in e and not (_is_int(e["uncompressed_bytes"])
                                                          and e["uncompressed_bytes"] >= 0):
                        probs.append(f"{w}.uncompressed_bytes: must be a non-negative integer, "
                                     f"got {e['uncompressed_bytes']!r}")

    if "patched_script" in d and g("patched_script") is not None:
        s = g("patched_script")
        probs += _exact_keys(s, ("original_md5", "patched_md5", "diff"),
                             "provenance.patched_script")
        if isinstance(s, dict):
            for k in ("original_md5", "patched_md5"):
                if k in s and not (isinstance(s[k], str) and _MD5.match(s[k])):
                    probs.append(f"provenance.patched_script.{k}: must be 32 lowercase hex, "
                                 f"got {s[k]!r}")
            if "diff" in s and not _is_str(s["diff"]):
                probs.append("provenance.patched_script.diff: must be a non-empty string")

    if "caper" in d:
        c = g("caper")
        if c is not None:
            cp = _exact_keys(c, ("input_json", "workflow_id", "metadata_json"), "provenance.caper")
            if isinstance(c, dict):
                cp += [f"provenance.caper.{k}: must be a non-empty string, got {c[k]!r}"
                       for k in ("input_json", "workflow_id", "metadata_json")
                       if k in c and not _is_str(c[k])]
            probs += cp
        if g("route") == "fastq" and c is None:
            probs.append("provenance.caper: must not be null when route is 'fastq'")
        if g("route") == "bam" and c is not None:
            probs.append("provenance.caper: must be null when route is 'bam'")

    if "slurm_job_ids" in d:
        j = g("slurm_job_ids")
        if not (isinstance(j, list) and all(_is_str(x) for x in j)):
            probs.append(f"provenance.slurm_job_ids: must be a list of non-empty strings, got {j!r}")

    if "code" in d:
        c = g("code")
        probs += _exact_keys(c, ("snapshot_dir", "git_sha"), "provenance.code")
        if isinstance(c, dict):
            if "snapshot_dir" in c and not _is_str(c["snapshot_dir"]):
                probs.append(f"provenance.code.snapshot_dir: must be a non-empty string, "
                             f"got {c['snapshot_dir']!r}")
            if "git_sha" in c and not (isinstance(c["git_sha"], str)
                                       and _GIT_SHA.match(c["git_sha"])):
                probs.append(f"provenance.code.git_sha: must be 7-40 lowercase hex, "
                             f"got {c['git_sha']!r}")

    if "created_utc" in d:
        t = g("created_utc")
        try:
            datetime.datetime.fromisoformat(t)
        except (TypeError, ValueError):
            probs.append(f"provenance.created_utc: must be an ISO-8601 string, got {t!r}")
    return probs


def _write_json(path, d, probs):
    if probs:
        raise ValueError("; ".join(probs))
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(d, indent=2, sort_keys=False) + "\n")
    tmp.replace(path)


def _key_order(d, keys) -> dict:
    """`d` with the schema keys first, in schema order; unknown keys kept (the validator names them)."""
    out = {k: d[k] for k in keys if k in d}
    out.update((k, v) for k, v in d.items() if k not in keys)
    return out


def write_covariates(path, **kw) -> dict:
    """Validate and write covariates.json. `schema`, `count_rule`, `wording_note` default to their
    fixed values and `log2_depth` to `log2(depth)`; every other key must be given."""
    d = dict(kw)
    d.setdefault("schema", SCHEMA)
    d.setdefault("count_rule", COUNT_RULE)
    d.setdefault("wording_note", WORDING_NOTE)
    if "log2_depth" not in d and _is_int(d.get("depth")) and d["depth"] > 0:
        d["log2_depth"] = math.log2(d["depth"])
    d = _key_order(d, COVARIATE_KEYS)
    _write_json(path, d, covariates_problems(d))
    return d


def write_provenance(path, **kw) -> dict:
    """Validate and write provenance.json. `schema` defaults to 1 and `created_utc` to now (UTC);
    every other key must be given."""
    d = dict(kw)
    d.setdefault("schema", SCHEMA)
    d.setdefault("created_utc",
                 datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    d = _key_order(d, PROVENANCE_KEYS)
    _write_json(path, d, provenance_problems(d))
    return d


def md5_file(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_json(path, where):
    """(dict or None, problems)."""
    try:
        return json.loads(Path(path).read_text()), []
    except FileNotFoundError:
        return None, [f"{where}: missing"]
    except (OSError, ValueError) as e:
        return None, [f"{where}: unreadable JSON ({e})"]


def product_problems(pdir, row=None) -> list:
    """Every problem with one product dir, each line prefixed by its pid. `row` is its C1 rows-TSV
    row, if one names it; the covariates must then agree with the row."""
    pdir = Path(pdir)
    pid = pdir.name
    if not pdir.is_dir():
        return [f"{pid}: product dir missing: {pdir}"]
    probs = []
    for name in ("counts25.npz", "pval25.npz"):
        f = pdir / name
        if not f.is_file():
            probs.append(f"{name}: missing")
        elif f.stat().st_size == 0:
            probs.append(f"{name}: empty")
    cov, p = _load_json(pdir / "covariates.json", "covariates.json")
    probs += p
    if cov is not None:
        probs += covariates_problems(cov)
        if isinstance(cov, dict):
            if cov.get("pid") != pid:
                probs.append(f"covariates.pid: {cov.get('pid')!r} != dir name {pid!r}")
            has_ctl = (pdir / "control_counts25.npz").is_file()
            if cov.get("control") is not None and not has_ctl:
                probs.append("control_counts25.npz: missing but covariates.control is not null")
            if cov.get("control") is None and "control" in cov and has_ctl:
                probs.append("control_counts25.npz: present but covariates.control is null")
            if row is not None:
                for k in ("biosample", "track", "cell", "assay", "arm", "level", "knob"):
                    if k in row and row[k] != cov.get(k):
                        probs.append(f"covariates.{k}: {cov.get(k)!r} != rows TSV {row[k]!r}")
                if "knob_value" in row:
                    try:
                        want = json.loads(row["knob_value"])
                    except ValueError:
                        probs.append(f"rows TSV knob_value is not JSON: {row['knob_value']!r}")
                    else:
                        got = cov.get("knob_value")
                        if got != want or isinstance(got, bool) != isinstance(want, bool):
                            probs.append(f"covariates.knob_value: {got!r} != rows TSV {want!r}")
    prov, p = _load_json(pdir / "provenance.json", "provenance.json")
    probs += p
    if prov is not None:
        probs += provenance_problems(prov)
        if isinstance(prov, dict):
            if prov.get("pid") != pid:
                probs.append(f"provenance.pid: {prov.get('pid')!r} != dir name {pid!r}")
            if row is not None and "route" in row and row["route"] != prov.get("route"):
                probs.append(f"provenance.route: {prov.get('route')!r} != rows TSV {row['route']!r}")
            repo = prov.get("pipeline", {}).get("repo") if isinstance(prov.get("pipeline"),
                                                                       dict) else None
            if row is not None and "pipeline" in row and PIPELINE_REPO.get(row["pipeline"]) != repo:
                probs.append(f"provenance.pipeline.repo: {repo!r} does not match rows TSV "
                             f"pipeline {row['pipeline']!r}")
    return [f"{pid}: {x}" for x in probs]


def read_rows(path) -> list:
    """A rows TSV written by `arms.py rows` (header exactly `arms.HEADER`) as a list of dicts."""
    with open(path, newline="") as f:
        # QUOTE_NONE: knob_value is JSON text, so a string value arrives as "ENCFF337JNL" with its
        # quotes, and the default dialect would strip them.
        r = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        if tuple(r.fieldnames or ()) != arms.HEADER:
            raise ValueError(f"{path}: header {r.fieldnames} is not arms.HEADER {list(arms.HEADER)}")
        return list(r)


def validate(products, rows_tsv=None, expect=None) -> tuple:
    """(n products checked, problem lines). Without `rows_tsv`: every dir holding a covariates.json."""
    products = Path(products)
    probs = []
    if rows_tsv is not None:
        rows = read_rows(rows_tsv)
        seen = set()
        for r in rows:
            if r["pid"] in seen:
                probs.append(f"{r['pid']}: duplicate pid in rows TSV")
            seen.add(r["pid"])
            probs += product_problems(products / r["pid"], r)
        n = len(seen)
    else:
        dirs = sorted(p.parent for p in products.glob("*/covariates.json"))
        for pdir in dirs:
            probs += product_problems(pdir)
        n = len(dirs)
    if expect is not None and n != expect:
        probs.append(f"expected {expect} products, checked {n}")
    return n, probs


def manifest_rows(products) -> tuple:
    """(MANIFEST.tsv rows as dicts sorted by pid, problem lines). Refuses nothing itself."""
    products = Path(products)
    rows, probs = [], []
    for pdir in sorted((p.parent for p in products.glob("*/covariates.json")),
                       key=lambda p: p.name):
        p = product_problems(pdir)
        if p:
            probs += p
            continue
        cov = json.loads((pdir / "covariates.json").read_text())
        prov = json.loads((pdir / "provenance.json").read_text())
        ctl = cov["control"] or {}
        ctl_npz = pdir / "control_counts25.npz"
        rows.append({
            "pid": cov["pid"], "biosample": cov["biosample"], "track": cov["track"],
            "cell": cov["cell"], "assay": cov["assay"], "arm": cov["arm"], "level": cov["level"],
            "route": prov["route"], "knob": cov["knob"],
            "knob_value": "" if cov["knob_value"] is None else json.dumps(cov["knob_value"]),
            "depth": str(cov["depth"]), "read_length": str(cov["read_length"]),
            "run_type": cov["run_type"], "fraglen": str(cov["fraglen"]),
            "control_accession": ctl.get("accession", ""), "control_source": ctl.get("source", ""),
            "pipeline_repo": prov["pipeline"]["repo"],
            "pipeline_release": prov["pipeline"]["release"],
            "sif_md5": prov["pipeline"]["sif_md5"],
            "counts25_md5": md5_file(pdir / "counts25.npz"),
            "pval25_md5": md5_file(pdir / "pval25.npz"),
            "control_counts25_md5": md5_file(ctl_npz) if ctl_npz.is_file() else "",
            "slurm_job_ids": ",".join(prov["slurm_job_ids"]),
            "product_dir": str(pdir.resolve()),
        })
    return rows, probs


def write_manifest(products, out) -> tuple:
    """Write MANIFEST.tsv only if every product is valid. Returns (n rows, problem lines)."""
    rows, probs = manifest_rows(products)
    probs += [f"{r['pid']}: {k} holds a tab or newline: {v!r}" for r in rows
              for k, v in r.items() if "\t" in v or "\n" in v]
    if probs:
        return 0, probs
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    # plain joins, no csv quoting: knob_value is JSON text and must reach the file as written
    lines = ["\t".join(MANIFEST_COLUMNS)] + ["\t".join(r[k] for k in MANIFEST_COLUMNS) for r in rows]
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(out)
    return len(rows), []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest", help="assemble MANIFEST.tsv from DIR/*/covariates.json")
    m.add_argument("--products", default=DEFAULT_PRODUCTS)
    m.add_argument("--out", required=True)
    v = sub.add_parser("validate", help="check every product named by --rows (or every product)")
    v.add_argument("--products", default=DEFAULT_PRODUCTS)
    v.add_argument("--rows", default=None)
    v.add_argument("--expect", type=int, default=None)
    a = ap.parse_args(argv)

    if a.cmd == "manifest":
        n, probs = write_manifest(a.products, a.out)
        for p in probs:
            print(p)
        print(f"FAIL {len(probs)} problems, {a.out} not written" if probs else f"OK {n}")
        return 1 if probs else 0

    try:
        n, probs = validate(a.products, a.rows, a.expect)
    except (OSError, ValueError) as e:
        print(f"rows TSV unreadable: {e}")
        print("FAIL 0")
        return 1
    for p in probs:
        print(p)
    print(f"FAIL {n}" if probs else f"OK {n}")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
