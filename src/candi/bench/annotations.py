"""Pinned scoring assets for the E-block: the annotation beds, and the `msevar` variance pools.

Two very different kinds of thing live here, and the difference is the point.

The **beds** are frozen bytes taken from the organizers' own repository (`assets/PROVENANCE.md`).
They are not rebuilt from upstream GENCODE or FANTOM5, and rebuilding them would be a regression:
a newer, better annotation makes `mseprom`/`msegene`/`mseenh` incomparable to the published table,
which is the one thing the E-block exists to be comparable to (`EVAL_PLAN.md` D5, D16).

The **variance pools** are ours. `msevar` weights each squared error by the cross-cell-type
variance of that assay at that position, and no such vector was ever published — the organizers
built theirs with `build_var_npy.py` over their training experiments. D7 settles what ours is: the
store's own training biosamples, in pval space for EIC (chosen to sit as close to the original 267
training experiments as the store allows) and in the corpus's own terms for MERGED. The pool
membership is written out beside the numbers so the difference from 267 stays visible.

The loaders return **raw lines**, not parsed records, and that is deliberate. `score_metrics.py`
does its own positional `line.split()` inside each metric with a fixed arity, so a parser here
would either have to reproduce that arity exactly or hand the metrics a different object than the
reference implementation sees. Returning lines keeps `eic.py` a transcription of the reference
rather than an interpretation of it.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

ASSET_DIR = Path(__file__).resolve().parent / "assets"

# The authority. `assets/PROVENANCE.md` restates these for a human reading the directory; when the
# two disagree, this dict is right and the markdown is the bug.
ASSET_SHA256: Dict[str, str] = {
    "gencode.v29.genes.gtf.bed.gz":
        "f6294b10c03d9ed45c9ffdc151f4a55cf05a4b0283b55624e87263950b19e560",
    "F5.hg38.enhancers.bed.gz":
        "85562823f7a394aad2a1ede17dadcca63772b5aac20922dd802850ec3eb606ee",
    "hg38.blacklist.bed.gz":
        "7e3af7a6d572c8447ef8c0513830d23974a47a36a29b58646fab455903c6f6d2",
    "gtf_to_bed.sh":
        "b59d9682a95743618deac8233f8a24a13de4de8bc20fc21f7ff1813daee4bdae",
}

# Observed line counts. A checksum tells you the file changed; these tell you *how* at a glance,
# which is what you want in a failure message.
ASSET_LINES: Dict[str, int] = {
    "gencode.v29.genes.gtf.bed.gz": 58_721,
    "F5.hg38.enhancers.bed.gz": 63_285,
    "hg38.blacklist.bed.gz": 38,
}

# Field arity, per metric. `mseenh` unpacks twelve names and discards nine of them; a bed with
# eleven or thirteen columns raises inside the metric rather than at load, so we check it here.
ASSET_COLUMNS: Dict[str, int] = {
    "gencode.v29.genes.gtf.bed.gz": 6,
    "F5.hg38.enhancers.bed.gz": 12,
    "hg38.blacklist.bed.gz": 3,
}

PROM_LOC = 80          # bins upstream of a gene start = 2 kb at 25 bp. score.py's default.
WINDOW_SIZE = 25       # the challenge's evaluation resolution, not its data resolution.


class AssetError(RuntimeError):
    """A pinned asset is missing, truncated, or no longer the bytes we verified."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_assets(names: Optional[Iterable[str]] = None) -> Dict[str, str]:
    """Re-check every pinned asset's sha256. Returns {name: digest}; raises on any mismatch.

    Called at the top of every public loader below rather than once at import. Import-time
    verification would hash 2.4 MB on `import candi.bench`, and — worse — would make a corrupted
    asset a failure the user sees while doing something unrelated.
    """
    want = list(names) if names is not None else list(ASSET_SHA256)
    got: Dict[str, str] = {}
    for name in want:
        path = ASSET_DIR / name
        if not path.exists():
            raise AssetError(
                f"pinned asset {name} is missing from {ASSET_DIR}. It is tracked in git; a clean "
                f"checkout has it. Do not re-download from upstream GENCODE/FANTOM5 — the E-block "
                f"needs the organizers' exact bytes (see assets/PROVENANCE.md)."
            )
        digest = _sha256(path)
        if digest != ASSET_SHA256[name]:
            raise AssetError(
                f"pinned asset {name} has changed: sha256 {digest} != {ASSET_SHA256[name]}. "
                f"Every E-block number computed against it is now uncomparable to the published "
                f"table. Restore the file; do not update the checksum."
            )
        got[name] = digest
    return got


def load_bed_lines(path: Path | str) -> List[str]:
    """Read a gzipped or plain BED into a list of raw lines.

    A transcription of `bw_to_npy.py::load_bed`, including its one visible quirk: gzip is opened in
    binary and each line is `.decode("ascii")`, so a non-ASCII byte anywhere in the file raises
    rather than being replaced. Ours would too.
    """
    path = Path(path)
    out: List[str] = []
    if str(path).endswith("gz"):
        with gzip.open(path, "r") as fh:
            for line in fh:
                out.append(line.decode("ascii"))
    else:
        with open(path, "r") as fh:
            for line in fh:
                out.append(line)
    return out


def _load_checked(name: str) -> List[str]:
    verify_assets([name])
    lines = load_bed_lines(ASSET_DIR / name)
    n = len(lines)
    if name in ASSET_LINES and n != ASSET_LINES[name]:
        raise AssetError(f"{name}: {n} lines, expected {ASSET_LINES[name]}")
    ncol = ASSET_COLUMNS.get(name)
    if ncol is not None and lines:
        got = len(lines[0].split())
        if got != ncol:
            raise AssetError(
                f"{name}: first line has {got} fields, expected {ncol}. The metrics unpack by "
                f"fixed arity, so this raises inside the metric rather than here."
            )
    return lines


def gene_annotations() -> List[str]:
    """GENCODE v29 genes as 6-column bed lines: `chrom start end gene_id score strand`.

    Consumed by both `mseprom` and `msegene`. Note there is no `feature == "gene"` filter anywhere
    in the organizers' `gtf_to_bed.sh` — the row count is an observation about the GTF they fed it,
    not a guarantee about what the rows are.
    """
    return _load_checked("gencode.v29.genes.gtf.bed.gz")


def enhancer_annotations() -> List[str]:
    """FANTOM5 permissive enhancers as 12-column bed12 lines. `mseenh` reads the first three."""
    return _load_checked("F5.hg38.enhancers.bed.gz")


def eic_blacklist_lines() -> List[str]:
    """The 38-region file shipped in the organizers' `annot/hg38/`.

    Pinned for the record, and **not used by the E-block**. Two independent reasons, both in
    `assets/PROVENANCE.md`: the scoring path reads npy and returns before the blacklist branch is
    reached, and the branch itself *deletes* bins rather than masking them, which shifts every
    downstream index out of register with the annotation coordinates the metrics compute. It is
    also not the ENCODE Exclusion list — 17,040 bp against that list's 227,162,400.

    Exposed so a reader who finds `--blacklist-file` in the organizers' CLI can check for
    themselves rather than assuming it was applied.
    """
    return _load_checked("hg38.blacklist.bed.gz")


# ---------------------------------------------------------------------------
# msevar variance pools (D7)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VariancePool:
    """A per-assay, per-chromosome cross-biosample variance vector, plus who is in it.

    `values` is `(n_bins,)` float32 and is what `msevar` weights by. `biosamples` is the pool
    membership, carried alongside because a variance vector with no membership is unfalsifiable —
    the whole reason D7 insists the EIC pool's size be reported against the original 267.
    """
    corpus: str
    assay: str
    chrom: str
    space: str                      # "pval" | "count"
    biosamples: Sequence[str]
    values: np.ndarray

    @property
    def n_biosamples(self) -> int:
        return len(self.biosamples)

    def __post_init__(self) -> None:
        if self.space not in ("pval", "count"):
            raise ValueError(f"space must be 'pval' or 'count', got {self.space!r}")
        if self.values.ndim != 1:
            raise ValueError(f"variance vector must be 1-D, got shape {self.values.shape}")
        if not np.all(np.isfinite(self.values)):
            raise ValueError(f"{self.corpus}/{self.assay}/{self.chrom}: non-finite variance")
        if np.any(self.values < 0):
            raise ValueError(f"{self.corpus}/{self.assay}/{self.chrom}: negative variance")


def varpool_path(root: Path | str, corpus: str, chrom: str) -> Path:
    """`<root>/<corpus>/var_<chrom>.npz` — one file per corpus per chromosome.

    These are NOT in git and must not be. chr21 alone is 1,868,399 bins; at float32 across 35
    assays that is 262 MB for one chromosome of one corpus. They live where the store lives (Fir),
    the CLI takes `--varpool <root>`, and only the membership JSON is committed.
    """
    return Path(root) / corpus / f"var_{chrom}.npz"


def varpool_meta_path(root: Path | str, corpus: str) -> Path:
    """`<root>/<corpus>/membership.json` — the small, committable half."""
    return Path(root) / corpus / "membership.json"


def load_variance_pool(root: Path | str, corpus: str, assay: str, chrom: str) -> VariancePool:
    """Read one assay's variance vector, with its pool membership attached."""
    npz = varpool_path(root, corpus, chrom)
    if not npz.exists():
        raise AssetError(
            f"no variance pool at {npz}. `msevar` cannot be computed without it, and the "
            f"organizers' own code returns a bare 0.0 in this situation rather than raising "
            f"(score_metrics.py: `if var is None and y_all is None: return 0.0`). We raise, "
            f"because a silent 0.0 that looks like a score is worse than a missing number."
        )
    meta_path = varpool_meta_path(root, corpus)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    with np.load(npz) as z:
        if assay not in z:
            raise AssetError(f"{npz} has no variance vector for assay {assay!r}; "
                             f"has {sorted(z.files)[:8]}...")
        values = np.asarray(z[assay], dtype=np.float32)
    entry = meta.get(assay, {})
    return VariancePool(
        corpus=corpus, assay=assay, chrom=chrom,
        space=entry.get("space", "pval"),
        biosamples=tuple(entry.get("biosamples", ())),
        values=values,
    )


def build_variance_pools(store_root: Path | str, out_root: Path | str, *, corpus: str,
                         chroms: Sequence[str], train_biosamples: Sequence[str],
                         space: str = "pval", assays: Optional[Sequence[str]] = None) -> Path:
    """Build and write the variance pools for one corpus. **Runs where the store is** (Fir).

    The definition follows `score_metrics.py::msevar` exactly: `var = std(y_all, axis=0) ** 2`
    over the biosamples in the pool — a population variance (ddof=0), not a sample variance. The
    difference is a factor of n/(n-1), which at n=267 is 0.4% and at a small per-assay pool is not.
    We take theirs, because the point is comparability, not correctness.

    `train_biosamples` is passed in rather than derived, because "which biosamples are training"
    is a property of the regime file, and a builder that guessed it would silently disagree with
    whatever the run was actually trained on.
    """
    from candi.store.reader import CorpusStore    # lazy: drags in h5py, and only Fir needs it

    store = CorpusStore(store_root)
    out_dir = Path(out_root) / corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = "pval" if space == "pval" else "counts"

    membership: Dict[str, dict] = {}
    for chrom in chroms:
        vectors: Dict[str, np.ndarray] = {}
        pool_assays = assays if assays is not None else sorted(
            {a for b in train_biosamples for a in store[b].assays(kind)}
        )
        for assay in pool_assays:
            members = [b for b in train_biosamples if store[b].has(assay, kind)]
            if len(members) < 2:
                # One track has zero variance everywhere; a weight vector of zeros makes msevar
                # a 0/0. Skip the assay and say so, rather than emit a vector that divides by zero.
                continue
            stack = np.stack([
                np.asarray(getattr(store[b][assay], kind)(chrom), dtype=np.float64)
                for b in members
            ], axis=0)
            vectors[assay] = (np.std(stack, axis=0) ** 2).astype(np.float32)
            membership[assay] = {"space": space, "biosamples": members, "n": len(members)}
        np.savez_compressed(varpool_path(out_root, corpus, chrom), **vectors)

    meta_path = varpool_meta_path(out_root, corpus)
    meta_path.write_text(json.dumps(membership, indent=2, sort_keys=True))
    return meta_path
