"""Close-out D1 -- fill the lab, antibody and donor covariates and unify the platform vocabulary.

    python fill_covariates.py join --tables <tables dir> --catalog <catalog/encode dir> --out <side.tsv>
    python fill_covariates.py apply --base <tables dir> --side <side.tsv> --out <new tables dir>

`join` reads one tables directory and the ENCODE catalog tables (`encode_catalog.py`:
experiments.tsv, libraries.tsv, files.tsv) and writes the side table, one line per measurement
row (`SIDE_COLUMNS`). Per row:
  lab       `experiments.tsv` `lab` of the row's experiment.
  donor     the donor accessions of the row's libraries (`libraries.tsv`): unit `br<i>` takes the
            libraries with `bio_rep = i`, every other unit (pooled) all libraries of the experiment.
            One value when they agree, else `MIXED`.
  antibody  the same rule on the libraries' ENCODE antibody accessions, for ChIP-seq target rows
            only; a control or accessibility row stays empty (not applicable).
  platform  one vocabulary, the ENCODE platform term id (OBI or NTR code). A row whose build wrote
            codes keeps them (the codes of the FASTQs it aligned); a name (the testbed rows) is
            mapped through `PLATFORM_CODE`; an empty cell (the native layer, built from ENCODE's
            own alignments) takes the sorted codes of the experiment's released FASTQs
            (`files.tsv`, all FASTQs when none is released).
Every empty or mixed cell gets its cause in `note`. `PLATFORM_CODE` was read from the portal
(`/platforms/<id>/`, 2026-11-16) and agrees with every eic_v1 row that holds a single code.

`apply` writes a new tables directory: `measurements.tsv` with the four columns of the side table
written in (every other cell of every row byte-identical), `covariates.json` with fresh sorted
vocabularies (`build_tables.covariates_spec`), every other table file copied byte for byte,
`covariates_fill.tsv` (a copy of the side table; not hashed by `tables_sha256`), `CHANGES.md`
(coverage per split, causes, unseen ids per split) and `build_info.json` (`covariates_of`, the
base's own build_info under `base_build_info`). `--out` must not exist. Check the result with
`check_dataset.py --tables <new> --superset-of <base> --covariate-columns lab,antibody,donor,platform`.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from minicandi import contract as C  # noqa: E402
from minicandi.dataset import MEASUREMENT_COLUMNS, read_tsv  # noqa: E402
from onion_data import build_tables as BT  # noqa: E402

FILLED = ("lab", "antibody", "donor", "platform")
SIDE_COLUMNS = ("measurement_id", "experiment", "in_eic", "assay", "assay_class", "unit", "lab", "antibody",
                "donor", "platform", "platform_before", "libraries", "note")
MIXED = "mixed"
#: ENCODE platform title -> term id (portal /platforms/<id>/, read 2026-11-16)
PLATFORM_CODE = {"Illumina Genome Analyzer": "NTR:0004974",
                 "Illumina Genome Analyzer I": "OBI:0002128",
                 "Illumina Genome Analyzer II": "OBI:0000703",
                 "Illumina Genome Analyzer IIe": "OBI:0002027",
                 "Illumina Genome Analyzer IIx": "OBI:0002000",
                 "Illumina HiSeq 2000": "OBI:0002001",
                 "Illumina HiSeq 2500": "OBI:0002002",
                 "Illumina HiSeq 4000": "OBI:0002049",
                 "Illumina NextSeq 500": "OBI:0002021"}
#: assay classes with no antibody (a ChIP-seq control, DNase-seq, ATAC-seq)
NO_ANTIBODY_CLASSES = ("control", "accessibility_dnase", "accessibility_atac")
SIDE_NAME, CHANGES = "covariates_fill.tsv", "CHANGES.md"


def _tsv(path) -> list:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _one(values: set, what: str):
    """(value, note) for a set of library values."""
    v = {x for x in values if x}
    if not v:
        return "", f"{what}: no value on any library in the catalog"
    if len(v) == 1:
        return next(iter(v)), ""
    return MIXED, f"{what}: {MIXED} ({';'.join(sorted(v))})"


def _is_code(token: str) -> bool:
    return token[:4] in ("OBI:", "NTR:")


def platform_of(row: dict, fastqs: list):
    """(new platform cell, note)."""
    before = row.get("platform", "")
    if before:
        toks = [t for t in before.split(",") if t]
        unknown = [t for t in toks if not _is_code(t) and t not in PLATFORM_CODE]
        if unknown:
            return before, f"platform: name(s) with no code kept as written: {unknown}"
        codes = sorted({t if _is_code(t) else PLATFORM_CODE[t] for t in toks})
        return ",".join(codes), ""
    rel = [f for f in fastqs if f.get("status") == "released"] or fastqs
    names = {f.get("platform", "") for f in rel if f.get("platform")}
    if not names:
        return "", "platform: no FASTQ with a platform in the catalog"
    unknown = sorted(n for n in names if n not in PLATFORM_CODE)
    if unknown:
        return "", f"platform: catalog name(s) with no code: {unknown}"
    return ",".join(sorted({PLATFORM_CODE[n] for n in names})), "platform: from the catalog FASTQs"


def join_rows(rows: list, experiments: list, libraries: list, files: list) -> list:
    E = {e["experiment"]: e for e in experiments}
    L = defaultdict(list)
    for lib in libraries:
        L[lib["experiment"]].append(lib)
    F = defaultdict(list)
    for f in files:
        if f.get("file_format") == "fastq":
            F[f["experiment"]].append(f)
    out = []
    for r in rows:
        x, notes = r["experiment"], []
        e = E.get(x)
        lab = (e or {}).get("lab", "")
        if not e:
            notes.append("lab: experiment not in the catalog")
        elif not lab:
            notes.append("lab: empty in the catalog")
        if r.get("lab") and lab and r["lab"] != lab:
            notes.append(f"lab: the base row said {r['lab']!r}")
        libs = L.get(x, [])
        if r["unit"].startswith("br") and r.get("bio_rep"):
            libs = [lib for lib in libs if lib["bio_rep"] == r["bio_rep"]]
        donor, n = _one({lib["donor"] for lib in libs}, "donor")
        if n:
            notes.append(n)
        if not libs:
            notes.append("libraries: none for this unit in the catalog")
        antibody = ""
        if needs(r, "antibody"):
            antibody, n = _one({lib["antibody"] for lib in libs}, "antibody")
            if n:
                notes.append(n)
        platform, n = platform_of(r, F.get(x, []))
        if n:
            notes.append(n)
        out.append({"measurement_id": r["measurement_id"], "experiment": x, "in_eic": r["in_eic"],
                    "assay": r["assay"],
                    "assay_class": r["assay_class"], "unit": r["unit"], "lab": lab,
                    "antibody": antibody, "donor": donor, "platform": platform,
                    "platform_before": r.get("platform", ""),
                    "libraries": ";".join(sorted(lib["library"] for lib in libs)),
                    "note": " | ".join(notes)})
    return out


def cmd_join(a) -> int:
    T, K = Path(a.tables), Path(a.catalog)
    rows = read_tsv(T / BT.MEASUREMENTS, MEASUREMENT_COLUMNS)
    side = join_rows(rows, _tsv(K / "experiments.tsv"), _tsv(K / "libraries.tsv"),
                     _tsv(K / "files.tsv"))
    BT.write_text(Path(a.out), BT.tsv_text(SIDE_COLUMNS, side))
    print(f"wrote {a.out}: {len(side)} rows")
    for line in coverage_lines(side):
        print(line)
    return 0


# --------------------------------------------------------------------------------------------
# apply


def needs(r: dict, col: str) -> bool:
    """Whether a row should hold a value in `col` (antibody: ChIP-seq target rows only)."""
    return col != "antibody" or (r["assay_class"] not in NO_ANTIBODY_CLASSES
                                 and r["assay"] != C.CONTROL)


def coverage_lines(rows: list) -> list:
    out = ["| column | split | rows that need it | filled | share |", "|---|---|---|---|---|"]
    for col in FILLED:
        for s in BT.ROLE_ORDER:
            need = [r for r in rows if r["in_eic"] == s and needs(r, col)]
            n = sum(1 for r in need if r[col])
            out.append(f"| {col} | {s} | {len(need)} | {n} | "
                       f"{(n / len(need) if need else float('nan')):.3f} |")
    return out


def unseen_lines(rows: list) -> list:
    out = ["| column | split | rows with a value | rows whose value is not in training | "
           "values not in training |", "|---|---|---|---|---|"]
    for col in FILLED:
        train = {r[col] for r in rows if r["in_eic"] == "train" and r[col]}
        for s in ("val", "blind"):
            have = [r for r in rows if r["in_eic"] == s and r[col]]
            new = sorted({r[col] for r in have if r[col] not in train})
            out.append(f"| {col} | {s} | {len(have)} | "
                       f"{sum(1 for r in have if r[col] not in train)} | "
                       f"{len(new)}{(': ' + ', '.join(new)) if new else ''} |")
    return out


def cmd_apply(a) -> int:
    base, out, side_path = Path(a.base), Path(a.out), Path(a.side)
    if out.exists():
        print(f"refused: {out} exists (a tables version is never overwritten)")
        return 2
    rows = read_tsv(base / BT.MEASUREMENTS, MEASUREMENT_COLUMNS)
    base_text = (base / BT.MEASUREMENTS).read_text()
    if BT.tsv_text(MEASUREMENT_COLUMNS, rows) != base_text:
        print("refused: the base measurements.tsv does not round-trip through the table writer")
        return 1
    side = {s["measurement_id"]: s for s in read_tsv(side_path, SIDE_COLUMNS)}
    ids = [r["measurement_id"] for r in rows]
    if set(ids) != set(side) or len(ids) != len(side):
        print(f"refused: side table rows {len(side)} do not match the base rows {len(ids)}")
        return 1
    new_rows = []
    for r in rows:
        n = dict(r)
        for col in FILLED:
            n[col] = side[r["measurement_id"]][col]
        new_rows.append(n)
    texts = {BT.MEASUREMENTS: BT.tsv_text(MEASUREMENT_COLUMNS, new_rows),
             BT.COVARIATES: json.dumps(BT.covariates_spec(new_rows), indent=1) + "\n"}
    stage = out.with_name(out.name + ".part")
    if stage.exists():
        print(f"refused: {stage} exists")
        return 2
    stage.mkdir(parents=True)
    copied = []
    for p in sorted(base.iterdir()):
        if p.is_file() and p.name not in texts and p.name not in (BT.BUILD_INFO, CHANGES, SIDE_NAME):
            shutil.copyfile(p, stage / p.name)
            copied.append(p.name)
    for name, text in texts.items():
        BT.write_text(stage / name, text)
    shutil.copyfile(side_path, stage / SIDE_NAME)
    sp = Counter(r["in_eic"] for r in new_rows)
    notes = Counter(part.split(":")[0] + ": " + part.split(":", 1)[1].split("(")[0].strip()
                    for s in side.values() for part in s["note"].split(" | ") if part)
    vocab = json.loads(texts[BT.COVARIATES])["vocab"]
    blind_only = sorted(set(vocab["platform"]) - {r["platform"] for r in new_rows
                                                 if r["in_eic"] != "blind"})
    lines = [f"# Tables {out.name}: lab, antibody and donor filled; one platform vocabulary",
             "", f"Base: `{base}` (left untouched). Side table: `{SIDE_NAME}` "
             f"(sha256 {BT.sha256_bytes(side_path.read_bytes())}).",
             f"Rows: {len(new_rows)} (" + ", ".join(f"{s} {sp.get(s, 0)}" for s in BT.ROLE_ORDER)
             + "). Only the columns lab, antibody, donor and platform differ from the base.",
             f"Files copied byte for byte: {', '.join(copied)}.", "",
             "## Coverage (antibody: ChIP-seq target rows only)", "", *coverage_lines(new_rows),
             "", "## Empty, mixed or derived cells, by cause", "",
             *[f"- {k}: {v} rows" for k, v in sorted(notes.items())], "",
             "## Values of validation and blind rows that training rows never hold", "",
             *unseen_lines(new_rows), "",
             "## Vocabulary sizes (embedding table rows: 1024)", "",
             *[f"- {f}: {len(v)}" for f, v in vocab.items()], "",
             f"Platform ids that occur only in blind rows: {len(blind_only)}"
             + (f" ({', '.join(blind_only)})" if blind_only else "") + "."]
    BT.write_text(stage / CHANGES, "\n".join(lines) + "\n")
    hashed = sorted(set(copied) | set(texts))
    binfo = json.loads((base / BT.BUILD_INFO).read_text()) if (base / BT.BUILD_INFO).is_file() else {}
    info = {"schema": 1, "tool": "tools/onion_data/fill_covariates.py", "git_sha": BT.git_sha(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "covariates_of": str(base.resolve()), "covariate_columns": list(FILLED),
            "side_table_sha256": BT.sha256_bytes(side_path.read_bytes()),
            "files_sha256": {n: BT.sha256_bytes((stage / n).read_bytes()) for n in hashed},
            # keys check_dataset.py reads from build_info.json, carried from the base
            **{k: binfo[k] for k in ("ledger", "addon_of", "other_experiments") if k in binfo},
            "base_build_info": binfo}
    BT.write_text(stage / BT.BUILD_INFO, json.dumps(info, indent=1) + "\n")
    stage.rename(out)
    print("\n".join(lines))
    print(f"wrote {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("join")
    j.add_argument("--tables", required=True)
    j.add_argument("--catalog", required=True)
    j.add_argument("--out", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--base", required=True)
    p.add_argument("--side", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return cmd_join(a) if a.cmd == "join" else cmd_apply(a)


if __name__ == "__main__":
    sys.exit(main())
