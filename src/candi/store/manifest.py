"""Metadata CSVs + `file_metadata.json` cross-check -> `manifest.json`.

**D20** — the CSVs are the authority. `file_metadata.json` is a cross-check that fails loudly on
disagreement instead of quietly winning. **D19** — nothing is fabricated: a field that is missing,
unparseable or contradicted by a second CSV row is written as an explicit `null` and listed under
`metadata_gaps`, so the model's `readlen_missing_emb` / `depth_missing_emb` get the case they
exist for. The `read_length=50` and `run_type=single` fallbacks and the hard-coded replicate key
`"2"` of the old path do **not** appear here.

The store itself supplies the structural facts — track order, dtype, `control_col`, `n_bins`,
`pval_clip_frac` — read straight off the h5 root attrs, because the h5 is the record. The CSVs
supply only the per-track experimental metadata.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import h5py

from candi.store import layout as L
from candi.store.layout import StoreError

__all__ = [
    "CSV_COLUMNS",
    "MetadataConflict",
    "AMBIGUOUS",
    "read_metadata_csvs",
    "read_file_metadata",
    "build_manifest",
    "write_manifest",
    "verify_store",
]

#: The column set of `eic_metadata.csv` / `merged_metadata.csv`. `control_metadata.csv` (t5) has
#: the same columns; that is why the extra CSV is just another entry in the same list.
CSV_COLUMNS = (
    "biosample_name",
    "assay_name",
    "bios_accession",
    "exp_accession",
    "file_accession",
    "assembly",
    "read_length",
    "run_type",
    "sequencing_platform",
    "lab",
    "depth",
)

#: Fields copied into each manifest track record, in this order.
_TRACK_FIELDS = (
    "depth",
    "read_length",
    "run_type",
    "file_accession",
    "exp_accession",
    "bios_accession",
    "lab",
    "platform",
    "assembly",
)

#: Sentinel for "two CSV rows disagree" — becomes `null` in the manifest plus a gap entry (D19).
AMBIGUOUS = object()


class MetadataConflict(StoreError):
    """The CSVs and `file_metadata.json` disagree (D20). Lists every disagreement it found."""


# ---------------------------------------------------------------------------------------------
# normalization — one place, so the CSV side and the json side are compared like for like
# ---------------------------------------------------------------------------------------------


def _norm_str(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"na", "nan", "none", "null", "-"}:
        return None
    return s


def _norm_number(v):
    s = _norm_str(v)
    if s is None:
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    if f != f:  # NaN
        return None
    return int(f) if f.is_integer() else f


def _norm_run_type(v) -> Optional[str]:
    """Substring parse, as the old path did — but an unrecognized string is `null`, not a guess."""
    s = _norm_str(v)
    if s is None:
        return None
    low = s.lower()
    if "single" in low:
        return "single-ended"
    if "pair" in low:
        return "paired-ended"
    return None


_NORMALIZERS = {
    "read_length": _norm_number,
    "depth": _norm_number,
    "run_type": _norm_run_type,
}


def _norm_field(name: str, value):
    return _NORMALIZERS.get(name, _norm_str)(value)


# ---------------------------------------------------------------------------------------------
# the CSVs (D20 authority)
# ---------------------------------------------------------------------------------------------


def read_metadata_csvs(paths: Iterable[Path | str]) -> dict:
    """`(biosample_name, assay_name) -> {field: value | AMBIGUOUS}` over one or more CSVs.

    Several CSVs are merged as one table — that is how t5's `control_metadata.csv` joins the
    signal CSVs without a special case. Rows that agree collapse; rows that disagree on a field
    set that field to `AMBIGUOUS` (D19), never to the first one seen.
    """
    table: dict = {}
    for path in paths:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"metadata CSV not found: {p}")
        with p.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            missing = [c for c in ("biosample_name", "assay_name") if c not in (reader.fieldnames or [])]
            if missing:
                raise StoreError(f"{p}: metadata CSV is missing required column(s) {missing}")
            for row in reader:
                bios = _norm_str(row.get("biosample_name"))
                assay = _norm_str(row.get("assay_name"))
                if bios is None or assay is None:
                    continue
                rec = {c: _norm_field(c, row.get(c)) for c in CSV_COLUMNS if c in row}
                rec["_source"] = str(p)
                key = (bios, assay)
                if key not in table:
                    table[key] = rec
                    continue
                merged = table[key]
                for c, v in rec.items():
                    if c == "_source":
                        continue
                    old = merged.get(c)
                    if old is AMBIGUOUS:
                        continue
                    if old is None:
                        merged[c] = v
                    elif v is not None and v != old:
                        merged[c] = AMBIGUOUS
    return table


# ---------------------------------------------------------------------------------------------
# file_metadata.json (D20 cross-check)
# ---------------------------------------------------------------------------------------------


def read_file_metadata(path: Path | str) -> Optional[dict]:
    """Flatten `<TRACK>/file_metadata.json` to `{field: value}`, or None when it is absent.

    Nearly every value in that file is a dict keyed by an arbitrary replicate index. **Whatever key
    is there is used** — the old path hard-coded `"2"` for controls and read the wrong replicate
    everywhere else. More than one key under a field the manifest *consumes* (`_CROSS_CHECK`) is
    ambiguous and raises, per D19.

    Not every dict-valued field is replicate-keyed: `T_testis/H3K9me3` in `DATA_CANDI_EIC` carries
    a free-text `notes` dict keyed by `alternative_bigbed_used` / `original_bigbed` /
    `alternative_reason`. Nothing reads it, so it is passed through **unflattened** rather than
    resolved by a coin flip or made fatal for the whole corpus.
    """
    p = Path(path)
    if not p.is_file():
        return None
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise StoreError(f"{p}: expected a JSON object")
    out: dict = {}
    for field, value in obj.items():
        if isinstance(value, dict):
            keys = list(value.keys())
            if len(keys) != 1:
                if field in _CROSS_CHECK:
                    raise StoreError(
                        f"{p}: field {field!r} has {len(keys)} replicate keys {keys}; exactly one "
                        f"is required — picking one would be a coin flip written into the manifest."
                    )
                out[field] = value          # not replicate-keyed, and nothing reads it
                continue
            out[field] = value[keys[0]]
        else:
            out[field] = value
    return out


#: `file_metadata.json` field -> manifest/CSV field. Only these four are cross-checked; the rest
#: of the json (`sequencing_platform`, `lab`) is free text the CSVs already carry.
_CROSS_CHECK = {
    "assay": "assay_name",
    "accession": "file_accession",
    "read_length": "read_length",
    "run_type": "run_type",
}


# ---------------------------------------------------------------------------------------------
# the manifest
# ---------------------------------------------------------------------------------------------


def _open_kinds(corpus_root: Path, biosample: str) -> dict:
    out = {}
    for kind in L.KINDS:
        p = L.kind_path(corpus_root, biosample, kind)
        if p.is_file():
            with h5py.File(p, "r") as f:
                attrs = L.read_root_attrs(f)
                attrs["_chroms"] = sorted(f.keys())
                attrs["_shapes"] = {c: tuple(f[c].shape) for c in f.keys()}
                attrs["_dtypes"] = {c: f[c].dtype.name for c in f.keys()}
            out[kind] = attrs
    return out


def _genome_block(corpus_root: Path, n_bins: Mapping[str, int], genome: Optional[Path]) -> dict:
    """`build` and `fasta_sha256` come from `genome/dna.h5` (t7) when it exists, else `null`."""
    gdir = Path(genome) if genome is not None else L.corpus_genome_dir(corpus_root)
    build = fasta_sha256 = None
    dna = gdir / "dna.h5"
    if dna.is_file():
        with h5py.File(dna, "r") as f:
            a = L.read_root_attrs(f)
        build = a.get("build")
        fasta_sha256 = a.get("fasta_sha256")
    return {
        "build": build,
        "fasta_sha256": fasta_sha256,
        "n_bins": {c: int(n) for c, n in sorted(n_bins.items(), key=lambda kv: L.sort_chroms([kv[0]]))},
    }


def build_manifest(
    corpus_root: Path | str,
    corpus: str,
    metadata_csvs: Sequence[Path | str],
    *,
    source_root: Optional[Path | str] = None,
    genome: Optional[Path | str] = None,
    biosamples: Optional[Sequence[str]] = None,
    strict: bool = True,
) -> dict:
    """Assemble `manifest.json` for one corpus store.

    `metadata_csvs` is a list, so t5's `control_metadata.csv` is passed as one more entry
    (`--metadata-csv a.csv --metadata-csv b.csv`) and needs no code change here. `source_root`
    is where `file_metadata.json` lives; omit it and the cross-check is skipped and reported as
    skipped. `strict=False` downgrades a CSV/json disagreement from an exception to a gap entry —
    it exists for triage, not for building a store you intend to train on.
    """
    corpus_root = Path(corpus_root)
    if not L.biosamples_dir(corpus_root).is_dir():
        raise FileNotFoundError(f"no biosamples/ under {corpus_root}")
    names = (
        list(biosamples)
        if biosamples is not None
        else sorted(p.name for p in L.biosamples_dir(corpus_root).iterdir() if p.is_dir())
    )
    if not names:
        raise StoreError(f"{corpus_root}: no biosamples to describe")

    table = read_metadata_csvs(metadata_csvs)
    gaps: list = []
    conflicts: list = []
    n_bins_all: dict = {}
    resolution = None
    kinds_seen: set = set()
    vocabulary: set = set()
    out_biosamples: dict = {}

    for name in names:
        kinds = _open_kinds(corpus_root, name)
        if not kinds:
            raise StoreError(f"{name}: no {'/'.join(L.KINDS)}.h5 under {corpus_root}")
        kinds_seen.update(kinds)
        # counts is canonical for column order and dtype; peaks/pval fall back only if it is absent.
        canonical = next(k for k in L.KINDS if k in kinds)
        base = kinds[canonical]
        tracks = list(base[L.ATTR_TRACKS])
        res = int(base[L.ATTR_RESOLUTION])
        resolution = res if resolution is None else resolution
        if res != resolution:
            raise StoreError(f"{name}/{canonical}: resolution {res} != {resolution} elsewhere")
        for kind, attrs in kinds.items():
            for chrom, nb in attrs[L.ATTR_N_BINS].items():
                prev = n_bins_all.setdefault(chrom, int(nb))
                if prev != int(nb):
                    raise StoreError(
                        f"{name}/{kind}: n_bins[{chrom}] = {nb} but another file says {prev}. "
                        f"The store was built against two different chrom_sizes."
                    )
        clip = kinds.get("pval", {}).get(L.ATTR_PVAL_CLIP_FRAC)
        clip_tracks = list(kinds.get("pval", {}).get(L.ATTR_TRACKS, []))
        # D25/D27 — the codec, carried through per track so the manifest ALONE answers "what are the
        # units of this pval track". An absent `transform` attr is a schema-1 file and means
        # `"linear"`, the same default `reader.BiosampleStore.pval` applies.
        pval_scale = kinds.get("pval", {}).get(L.ATTR_SCALE)
        pval_transform = kinds.get("pval", {}).get(
            L.ATTR_TRANSFORM, L.PVAL_TRANSFORM_LINEAR if "pval" in kinds else None)
        npz_depth = kinds.get("counts", {}).get(L.ATTR_NPZ_DEPTH)
        counts_tracks = list(kinds.get("counts", {}).get(L.ATTR_TRACKS, []))

        records = []
        for col, track in enumerate(tracks):
            if track != L.CONTROL_TRACK:
                vocabulary.add(track)
            csv_rec = table.get((name, track))
            rec: dict = {"assay": track, "col": col}
            if csv_rec is None:
                for f in _TRACK_FIELDS:
                    rec[f] = None
                gaps.append({"biosample": name, "track": track, "field": "*", "reason": "no CSV row"})
            else:
                for f in _TRACK_FIELDS:
                    src_key = "sequencing_platform" if f == "platform" else f
                    v = csv_rec.get(src_key)
                    if v is AMBIGUOUS:
                        rec[f] = None
                        gaps.append(
                            {"biosample": name, "track": track, "field": f,
                             "reason": "CSV rows disagree"}
                        )
                    elif v is None:
                        rec[f] = None
                        gaps.append(
                            {"biosample": name, "track": track, "field": f, "reason": "absent"}
                        )
                    else:
                        rec[f] = v
            # cross-check (D20)
            if source_root is not None:
                fm = read_file_metadata(Path(source_root) / name / track / "file_metadata.json")
                if fm is None:
                    gaps.append(
                        {"biosample": name, "track": track, "field": "file_metadata.json",
                         "reason": "absent"}
                    )
                else:
                    for jkey, csv_key in _CROSS_CHECK.items():
                        if jkey not in fm:
                            continue
                        jv = _norm_field(csv_key, fm[jkey])
                        # `assay_name` is checked against the track DIRECTORY name — the one thing
                        # the CSV cannot be wrong about, because it is what we stored.
                        cv = track if csv_key == "assay_name" else rec.get(csv_key)
                        if jv is None or cv is None:
                            continue
                        if jv != cv:
                            conflicts.append(
                                {"biosample": name, "track": track, "field": csv_key,
                                 "csv": cv, "file_metadata": jv}
                            )
            rec["npz_depth"] = (
                npz_depth[counts_tracks.index(track)]
                if npz_depth and track in counts_tracks
                else None
            )
            rec["pval_clip_frac"] = (
                clip[clip_tracks.index(track)] if clip and track in clip_tracks else None
            )
            has_pval = track in clip_tracks
            rec["pval_scale"] = int(pval_scale) if has_pval and pval_scale is not None else None
            rec["pval_transform"] = str(pval_transform) if has_pval and pval_transform else None
            rec["kinds"] = [k for k in L.KINDS if k in kinds and track in list(kinds[k][L.ATTR_TRACKS])]
            records.append(rec)

        out_biosamples[name] = {
            "dtype": base[L.ATTR_DTYPE] if canonical == "counts" else kinds.get(
                "counts", {}).get(L.ATTR_DTYPE),
            "control_col": int(base[L.ATTR_CONTROL_COL]),
            "kinds": sorted(kinds, key=L.KINDS.index),
            "chroms": L.sort_chroms(base["_chroms"]),
            "tracks": records,
        }

    if conflicts:
        head = "\n".join(
            f"  {c['biosample']}/{c['track']}.{c['field']}: CSV={c['csv']!r} "
            f"file_metadata.json={c['file_metadata']!r}"
            for c in conflicts[:20]
        )
        msg = (
            f"{len(conflicts)} CSV vs file_metadata.json disagreement(s) (D20 — the CSV is the "
            f"authority and the json must agree with it):\n{head}"
        )
        if strict:
            raise MetadataConflict(msg)
        print(f"WARNING: {msg}", flush=True)
        gaps.extend({**c, "field": c["field"], "reason": "CSV/json conflict"} for c in conflicts)

    manifest = {
        "schema": L.SCHEMA_VERSION,
        "corpus": corpus,
        "resolution": int(resolution),
        "genome": _genome_block(corpus_root, n_bins_all, Path(genome) if genome else None),
        "assay_vocabulary": sorted(vocabulary),
        "kinds": [k for k in L.KINDS if k in kinds_seen],
        "biosamples": out_biosamples,
        "source_root": str(source_root) if source_root is not None else None,
        "built": {"kit_version": L.kit_version(), "utc": L.utc_now()},
        "metadata_csvs": [str(p) for p in metadata_csvs],
        "metadata_gaps": gaps,
    }
    return manifest


def write_manifest(corpus_root: Path | str, manifest: Mapping[str, Any]) -> Path:
    """Write `manifest.json`. Generated, never hand-edited — regenerate it instead."""
    out = L.manifest_path(corpus_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(out)
    return out


# ---------------------------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------------------------


def verify_store(corpus_root: Path | str) -> list:
    """Structural check of a built corpus against its own manifest. Returns a list of problems.

    Empty list means the store is self-consistent: every file the manifest names exists, carries a
    SUPPORTED schema, agrees on resolution and track order, and every chromosome dataset has exactly
    the `(n_bins, n_tracks)` shape and the dtype its attrs claim.

    "Supported", not "the current version": D27 bumped the schema for the pval codec alone, and t25
    rebuilds only `pval.h5`, so a correct corpus is mixed by construction. What IS required of a
    schema-2 pval file is that it name its `transform` — a file whose units cannot be recovered is a
    problem whatever its version says.
    """
    corpus_root = Path(corpus_root)
    problems: list = []
    mpath = L.manifest_path(corpus_root)
    if not mpath.is_file():
        return [f"{mpath}: missing — run `build-manifest` first"]
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("schema") not in L.SUPPORTED_SCHEMA_VERSIONS:
        problems.append(f"{mpath}: schema {manifest.get('schema')} is not one of "
                        f"{list(L.SUPPORTED_SCHEMA_VERSIONS)}")
    n_bins = {c: int(n) for c, n in manifest.get("genome", {}).get("n_bins", {}).items()}
    resolution = manifest.get("resolution")

    for name, entry in manifest.get("biosamples", {}).items():
        track_names = [t["assay"] for t in entry["tracks"]]
        for kind in entry.get("kinds", []):
            p = L.kind_path(corpus_root, name, kind)
            if not p.is_file():
                problems.append(f"{name}/{kind}: manifest names it but {p} does not exist")
                continue
            with h5py.File(p, "r") as f:
                a = L.read_root_attrs(f)
                # D27 — MEMBERSHIP, not equality. t25 rebuilds only the pval layer, so a valid
                # corpus carries schema-2 pval.h5 beside schema-1 counts.h5 and peaks.h5. An
                # equality check here would have made a codec bump a whole-corpus rebuild of every
                # kind, which is a much larger and entirely unnecessary operation.
                if a.get(L.ATTR_SCHEMA) not in L.SUPPORTED_SCHEMA_VERSIONS:
                    problems.append(f"{p}: schema {a.get(L.ATTR_SCHEMA)} is not one of "
                                    f"{list(L.SUPPORTED_SCHEMA_VERSIONS)}")
                if kind == "pval" and a.get(L.ATTR_SCHEMA) == 2 and not a.get(L.ATTR_TRANSFORM):
                    problems.append(f"{p}: schema 2 pval with no {L.ATTR_TRANSFORM!r} attr — the "
                                    f"units are unrecoverable (D27)")
                if a.get(L.ATTR_RESOLUTION) != resolution:
                    problems.append(f"{p}: resolution {a.get(L.ATTR_RESOLUTION)} != {resolution}")
                if a.get(L.ATTR_BIOSAMPLE) != name:
                    problems.append(f"{p}: biosample attr {a.get(L.ATTR_BIOSAMPLE)!r} != {name!r}")
                tracks = list(a.get(L.ATTR_TRACKS, []))
                if [t for t in tracks if t not in track_names]:
                    problems.append(f"{p}: tracks not in the manifest: "
                                    f"{[t for t in tracks if t not in track_names]}")
                if a.get(L.ATTR_CONTROL_COL) != L.control_col_of(tracks):
                    problems.append(f"{p}: control_col {a.get(L.ATTR_CONTROL_COL)} disagrees with "
                                    f"tracks {tracks}")
                for chrom in f.keys():
                    ds = f[chrom]
                    want = (n_bins.get(chrom), len(tracks))
                    if want[0] is not None and tuple(ds.shape) != want:
                        problems.append(f"{p}:/{chrom}: shape {tuple(ds.shape)} != {want}")
                    if ds.dtype.name != a.get(L.ATTR_DTYPE):
                        problems.append(
                            f"{p}:/{chrom}: dtype {ds.dtype.name} != attr {a.get(L.ATTR_DTYPE)}"
                        )
    return problems
