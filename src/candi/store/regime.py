"""The regime file — the authority that replaced `h5.attrs` (t8).

The old bake froze window size, context length, the DSF ladder, the chromosome split and the assay
column order into the h5, so changing any of them meant a re-bake. The store holds none of it.
A regime file holds all of it, is read at load time, and is recorded verbatim (plus its sha256)
in `run.json`, so one store serves every regime.

```json
{
  "store": "/…/CANDI_STORE/eic",
  "assays": ["ATAC-seq", "DNase-seq", "H3K4me3"],
  "biosamples": {"train": ["T_…"], "eval": ["V_…", "B_…"]},
  "eval_pairs": [["T_…", "V_…"], ["T_…", "B_…"]],
  "context_bins": 768,
  "train_chroms": ["chr19"],
  "eval_chroms": ["chr21"],
  "window_plan": {"type": "tile", "stride_bins": 768, "min_valid_frac": 0.9},
  "dsf": {"policy": "discrete", "levels": [1, 2, 4, 8]},
  "kinds": ["counts", "peaks"],
  "seed": 42
}
```

Two decisions carry the weight:

* **D14 — `assays` IS the column order.** Declared, never derived. The old
  `sort_values(ascending=False)` ordered columns by *availability*, so adding one biosample
  permuted every column in the panel and every checkpoint trained before it became
  uninterpretable; the bijection asserts that guarded it are gone with it.
  `Regime.validate_against` checks each declared assay exists in the store and **names** the one
  that does not.
* **D12 — window eligibility is `mask[s:s+L].mean() >= min_valid_frac`**, default 0.9,
  overridable per regime. The primitive that computes it belongs to `genome.py` (t7); this module
  imports it lazily so a store without a genome layer still parses, and falls back to its own
  cumulative-sum implementation when `genome.py` is not importable yet.
* **D23 — the default DSF policy is discrete `{1, 2, 4, 8}`**, for continuity with the existing
  configs. `{"policy": "loguniform", "min": 1, "max": 8}` is the continuous version.
* **D31 (t14) — `eval_pairs` is DECLARED, never inferred.** An imputation evaluation prompts with
  one biosample and scores against a *different* one that holds the held-out assays. The old bake
  found the second by string surgery on the first (`T_X` → `V_X` / `B_X`,
  `dataset.py::_all_imp_biosamples`), which D16 forbids here — store biosample names are opaque ids
  and nothing may parse them. So the pairing is a list of `[input, target]` in the regime. It is
  OPTIONAL: without it the eval split iterates `biosamples.eval` exactly as it did before t14 and
  emits no imputation keys, which is the training-only mode `train.py` already documents.
* **D32 (t79) — a regime MAY restrict its TRAIN windows to a BED, by strict containment.**
  `eic.pilot` trains on the 44 ENCODE Pilot Regions, which a chromosome list cannot express, so
  the optional `regions` key names a BED4, its sha256 and a policy:
  `{"bed": "regions/encode_pilot_hg38.bed", "sha256": "13e1…", "policy": "contain"}`. Three
  things are deliberate. **The sha256 is required and checked at load**: the regime is recorded
  verbatim plus its own hash in `run.json`, but a BED referenced by path sits OUTSIDE that hash,
  so without the pin the training scope could change while the regime hash swore nothing did —
  the exact failure the regime file exists to prevent. **`bed` resolves against the regime file's
  own directory**, because the regime is copied into `run.json` and read back from elsewhere.
  **Containment is a SECOND rule, not an edit to D12**: `eligible_starts` runs unchanged, then a
  start is dropped unless the whole window `[s, s+context_bins)` lies inside one region. ANDing
  the region indicator into the mask was rejected — it lets a window straddling a region edge
  pass on 90 % coverage, admitting ~1.5 Mb of non-pilot sequence, ~6 % of the scope. Bin
  *validity* and region *membership* stay two separate rules. The key applies to the **train**
  split only: Rule 2 is about training loci, and eval is whole chromosomes. Absent, nothing
  changes — `to_dict()` omits it, so a re-serialised regime is byte-identical to today's.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from candi.store import layout as L
from candi.store.layout import StoreError

__all__ = [
    "DEFAULT_MIN_VALID_FRAC",
    "DEFAULT_DSF_LEVELS",
    "RegimeError",
    "DsfPolicy",
    "WindowPlan",
    "RegionSet",
    "Regime",
    "eligible_starts",
]

#: D12 — a window is eligible when at least this fraction of its bins are valid.
DEFAULT_MIN_VALID_FRAC = 0.9
#: D23 — the discrete ladder the existing configs use.
DEFAULT_DSF_LEVELS: Tuple[int, ...] = (1, 2, 4, 8)

_SPLITS = ("train", "eval")


class RegimeError(StoreError):
    """A malformed regime file, or one that does not match the store it points at."""


def _parse_eval_pairs(obj: Any) -> Tuple[Tuple[str, str], ...]:
    """D31 — `[[input, target], …]`, or `[{"input": …, "target": …}, …]`. Both spellings, one shape.

    A pair whose two halves are the same name is refused rather than ignored: scoring a biosample
    against itself is an identity copy, and it is the exact mistake this list exists to make
    impossible to arrive at by accident.
    """
    if obj is None:
        return ()
    if not isinstance(obj, (list, tuple)):
        raise RegimeError("regime.eval_pairs must be a list of [input, target] pairs")
    out: List[Tuple[str, str]] = []
    for i, item in enumerate(obj):
        if isinstance(item, Mapping):
            missing = [k for k in ("input", "target") if k not in item]
            if missing:
                raise RegimeError(f"regime.eval_pairs[{i}] is missing key(s) {missing}")
            a, b = str(item["input"]), str(item["target"])
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            a, b = str(item[0]), str(item[1])
        else:
            raise RegimeError(
                f"regime.eval_pairs[{i}] must be [input, target] or "
                f'{{"input": …, "target": …}}; got {item!r}'
            )
        if a == b:
            raise RegimeError(
                f"regime.eval_pairs[{i}] pairs {a!r} with itself. The target supplies the ground "
                f"truth for assays the input does not have; the same biosample supplies neither."
            )
        if (a, b) in out:
            raise RegimeError(f"regime.eval_pairs[{i}] repeats the pair ({a!r}, {b!r})")
        out.append((a, b))
    return tuple(out)


# ---------------------------------------------------------------------------------------------
# D12 — window eligibility
# ---------------------------------------------------------------------------------------------


def eligible_starts(mask, context_bins: int, min_valid_frac: float = DEFAULT_MIN_VALID_FRAC,
                    stride: int = 1) -> np.ndarray:
    """Start bins of every eligible window (D12). The rule itself lives in `genome.py`.

    This is a thin delegation on purpose. D12 is one invariant, so it gets exactly one
    implementation — `genome.py::eligible_starts` — and this module holds no copy of the
    cumulative-sum arithmetic to drift away from it. The import is lazy only to keep
    `import candi.store.regime` off h5py; `genome.py` is a sibling and is always present.
    """
    from candi.store.genome import eligible_starts as _real  # noqa: WPS433 (lazy on purpose)

    try:
        return np.asarray(
            _real(mask, context_bins, min_valid_frac, stride=stride), dtype=np.int64
        )
    except TypeError as exc:  # pragma: no cover - only on a t7/t8 signature drift
        raise RegimeError(
            f"candi.store.genome.eligible_starts does not take "
            f"(mask, context_bins, min_valid_frac, stride=): {exc}"
        ) from exc


# ---------------------------------------------------------------------------------------------
# the pieces
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DsfPolicy:
    """D23 — how a downsampling factor is drawn. Discrete by default; loguniform on request."""

    policy: str = "discrete"
    levels: Tuple[float, ...] = DEFAULT_DSF_LEVELS
    dsf_min: float = 1.0
    dsf_max: float = 8.0

    @classmethod
    def from_obj(cls, obj: Optional[Mapping[str, Any]]) -> "DsfPolicy":
        obj = dict(obj or {})
        policy = str(obj.get("policy", "discrete"))
        if policy not in ("discrete", "loguniform", "off"):
            raise RegimeError(
                f"dsf.policy must be 'discrete', 'loguniform' or 'off', got {policy!r}"
            )
        levels = tuple(float(x) for x in obj.get("levels", DEFAULT_DSF_LEVELS))
        lo = float(obj.get("min", obj.get("dsf_min", 1.0)))
        hi = float(obj.get("max", obj.get("dsf_max", 8.0)))
        if policy == "discrete" and (not levels or any(x < 1 for x in levels)):
            raise RegimeError(f"dsf.levels must all be >= 1, got {list(levels)}")
        if policy == "loguniform" and not (1.0 <= lo <= hi):
            raise RegimeError(f"dsf: need 1 <= min <= max, got min={lo} max={hi}")
        return cls(policy=policy, levels=levels, dsf_min=lo, dsf_max=hi)

    def to_dict(self) -> dict:
        if self.policy == "loguniform":
            return {"policy": self.policy, "min": self.dsf_min, "max": self.dsf_max}
        return {"policy": self.policy, "levels": [int(x) if float(x).is_integer() else x
                                                  for x in self.levels]}

    def sample(self, rng: np.random.Generator) -> float:
        """One DSF draw. Always `>= 1`: thinning can only remove reads, never invent them."""
        if self.policy == "off":
            return 1.0
        if self.policy == "discrete":
            return float(self.levels[int(rng.integers(len(self.levels)))])
        return float(np.exp(rng.uniform(np.log(self.dsf_min), np.log(self.dsf_max))))

    @property
    def is_trivial(self) -> bool:
        return self.policy == "off" or (self.policy == "discrete" and tuple(self.levels) == (1.0,))


@dataclass(frozen=True)
class WindowPlan:
    """How windows are laid over a chromosome. `tile` is the only plan the store needs."""

    type: str = "tile"
    stride_bins: Optional[int] = None            # defaults to `context_bins` — non-overlapping
    min_valid_frac: float = DEFAULT_MIN_VALID_FRAC

    @classmethod
    def from_obj(cls, obj: Optional[Mapping[str, Any]]) -> "WindowPlan":
        obj = dict(obj or {})
        t = str(obj.get("type", "tile"))
        if t != "tile":
            raise RegimeError(
                f"window_plan.type {t!r} is not supported; the store tiles ('tile') and lets the "
                f"stride do the rest — an overlapping plan is stride < context_bins."
            )
        stride = obj.get("stride_bins")
        frac = float(obj.get("min_valid_frac", DEFAULT_MIN_VALID_FRAC))
        if not (0.0 <= frac <= 1.0):
            raise RegimeError(f"window_plan.min_valid_frac must be in [0, 1], got {frac}")
        if stride is not None and int(stride) <= 0:
            raise RegimeError(f"window_plan.stride_bins must be positive, got {stride}")
        return cls(type=t, stride_bins=None if stride is None else int(stride), min_valid_frac=frac)

    def to_dict(self) -> dict:
        return {"type": self.type, "stride_bins": self.stride_bins,
                "min_valid_frac": self.min_valid_frac}

    def stride(self, context_bins: int) -> int:
        return int(self.stride_bins or context_bins)


# ---------------------------------------------------------------------------------------------
# D32 — restricting the train split to a BED
# ---------------------------------------------------------------------------------------------


def _parse_bed4(text: str) -> Tuple[Tuple[str, int, int, str], ...]:
    """BED4 → `((chrom, start_bp, end_bp, name), …)`. Half-open, 0-based, as BED is."""
    out: List[Tuple[str, int, int, str]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith(("#", "track ", "browser ")):
            continue
        f = line.split()
        if len(f) < 3:
            raise RegimeError(f"regions BED line {i} has {len(f)} field(s); BED needs chrom start end")
        try:
            s, e = int(f[1]), int(f[2])
        except ValueError as exc:
            raise RegimeError(f"regions BED line {i}: start/end are not integers — {line!r}") from exc
        if e <= s:
            raise RegimeError(f"regions BED line {i} is empty or reversed: {line!r}")
        out.append((f[0], s, e, f[3] if len(f) > 3 else ""))
    if not out:
        raise RegimeError("regions BED has no intervals")
    return tuple(out)


@dataclass(frozen=True)
class RegionSet:
    """D32 — the BED the train split is restricted to, its pinned hash, and its parsed intervals."""

    bed: str                                       # as declared, verbatim — what `to_dict` writes
    sha256: str                                    # of the BED's bytes, checked at load
    policy: str = "contain"
    resolved: Optional[str] = None                 # the path the declaration resolved to
    intervals: Tuple[Tuple[str, int, int, str], ...] = ()

    @classmethod
    def from_obj(cls, obj: Optional[Mapping[str, Any]], *, base: Optional[Path] = None
                 ) -> Optional["RegionSet"]:
        """Parse, resolve and hash-check. `base` is the regime file's own directory."""
        if obj is None:
            return None
        if not isinstance(obj, Mapping):
            raise RegimeError('regime.regions must be an object: {"bed": …, "sha256": …}')
        missing = [k for k in ("bed", "sha256") if k not in obj]
        if missing:
            raise RegimeError(
                f"regime.regions is missing key(s) {missing}. `sha256` is REQUIRED (D32): the BED "
                f"is outside the regime's own hash, so without it the training scope could change "
                f"while the regime hash stayed identical."
            )
        policy = str(obj.get("policy", "contain"))
        if policy != "contain":
            raise RegimeError(
                f"regime.regions.policy {policy!r} is not supported; 'contain' is the only rule — "
                f"the whole window must lie inside one region (D32)."
            )
        bed = str(obj["bed"])
        p = Path(bed)
        if not p.is_absolute():
            p = (Path(base) if base is not None else Path.cwd()) / p
        if not p.is_file():
            raise RegimeError(f"regime.regions.bed {bed!r} resolves to {p}, which is not a file")
        data = p.read_bytes()
        got = hashlib.sha256(data).hexdigest()
        want = str(obj["sha256"]).strip().lower()
        if got != want:
            raise RegimeError(
                f"regime.regions.bed {p} has sha256 {got}, but the regime declares {want}. The BED "
                f"defines the training scope and is pinned by hash (D32) — a mismatch means the "
                f"scope moved under a regime whose own hash did not."
            )
        return cls(bed=bed, sha256=want, policy=policy, resolved=str(p),
                   intervals=_parse_bed4(data.decode("utf-8")))

    @classmethod
    def from_bed(cls, path: Path | str) -> "RegionSet":
        """A region set read straight off a BED, hashing it rather than checking a declared hash.

        `from_obj` is the REGIME's constructor and demands a `sha256` because the BED sits outside
        the regime's own hash — without one the training scope could move while the regime hash
        stayed identical. This one is for a scope named on a COMMAND LINE (t89's eval scope), where
        there is no declaration to check against; the hash is computed here and travels in the run's
        provenance instead, which is what makes two runs comparable or not.
        """
        p = Path(path)
        if not p.is_file():
            raise RegimeError(f"region BED {p} is not a file")
        data = p.read_bytes()
        return cls(bed=str(p), sha256=hashlib.sha256(data).hexdigest(), policy="contain",
                   resolved=str(p), intervals=_parse_bed4(data.decode("utf-8")))

    def to_dict(self) -> dict:
        return {"bed": self.bed, "sha256": self.sha256, "policy": self.policy}

    @property
    def chroms(self) -> Tuple[str, ...]:
        return tuple(L.sort_chroms(c for c, _, _, _ in self.intervals))

    def bin_spans(self, chrom: str, resolution: int) -> Tuple[Tuple[int, int], ...]:
        """`[first_bin, end_bin)` per region on `chrom`, over bins lying WHOLLY inside it.

        A region boundary need not sit on the bin grid — the hg38 Pilot Regions do not — so the
        bin count is a containment count, never `bp // resolution`.
        """
        res = int(resolution)
        spans = [(-(-s // res), e // res) for c, s, e, _ in self.intervals if c == chrom]
        return tuple(sorted((a, b) for a, b in spans if b > a))

    def contained_starts(self, chrom: str, starts, context_bins: int, resolution: int) -> np.ndarray:
        """The subset of `starts` whose whole window `[s, s+context_bins)` is inside ONE region."""
        s = np.asarray(starts, dtype=np.int64)
        spans = self.bin_spans(chrom, resolution)
        if s.size == 0 or not spans:
            return s[:0]
        keep = np.zeros(s.shape, dtype=bool)
        for a, b in spans:
            keep |= (s >= a) & (s + int(context_bins) <= b)
        return s[keep]


# ---------------------------------------------------------------------------------------------
# the regime
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Regime:
    """A parsed, validated regime file. Immutable; `sha256` is over the bytes it was read from."""

    store: str
    assays: Tuple[str, ...]                       # D14 — THE column order
    context_bins: int
    train_biosamples: Tuple[str, ...] = ()
    eval_biosamples: Tuple[str, ...] = ()
    #: D31 — `((input, target), …)`. Empty means "no imputation evaluation", not "derive it".
    eval_pairs: Tuple[Tuple[str, str], ...] = ()
    train_chroms: Tuple[str, ...] = ()
    eval_chroms: Tuple[str, ...] = ()
    window_plan: WindowPlan = field(default_factory=WindowPlan)
    #: D32 — optional. Restricts the TRAIN split's windows to a BED, by strict containment.
    regions: Optional[RegionSet] = None
    dsf: DsfPolicy = field(default_factory=DsfPolicy)
    kinds: Tuple[str, ...] = ("counts", "peaks")
    seed: int = 42
    path: Optional[str] = None
    sha256: Optional[str] = None
    raw: Optional[str] = None                     # the file verbatim, for `run.json`

    # -- parsing ------------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any], *, path: Optional[Path | str] = None,
                  raw: Optional[str] = None, sha256: Optional[str] = None) -> "Regime":
        if not isinstance(obj, Mapping):
            raise RegimeError("a regime must be a JSON object")
        missing = [k for k in ("store", "assays", "context_bins") if k not in obj]
        if missing:
            raise RegimeError(f"regime is missing required key(s) {missing}")

        assays = [str(a) for a in obj["assays"]]
        if not assays:
            raise RegimeError("regime.assays is empty; there is nothing to model")
        dupes = sorted({a for a in assays if assays.count(a) > 1})
        if dupes:
            raise RegimeError(f"regime.assays has duplicate entries {dupes}; a column may appear once")
        if L.CONTROL_TRACK in assays:
            raise RegimeError(
                f"regime.assays contains {L.CONTROL_TRACK!r}. The control is a column of "
                f"counts.h5 flagged by `control_col` (D18) and rides its own batch keys — it is "
                f"never one of the modelled assays."
            )

        bios = obj.get("biosamples") or {}
        if not isinstance(bios, Mapping):
            raise RegimeError("regime.biosamples must be an object with 'train' / 'eval' lists")
        unknown = [k for k in bios if k not in _SPLITS]
        if unknown:
            raise RegimeError(f"regime.biosamples has unknown split(s) {unknown}; expected {list(_SPLITS)}")

        pairs = _parse_eval_pairs(obj.get("eval_pairs"))

        ctx = int(obj["context_bins"])
        if ctx <= 0:
            raise RegimeError(f"regime.context_bins must be positive, got {ctx}")

        kinds = tuple(str(k) for k in obj.get("kinds", ("counts", "peaks")))
        bad_kinds = [k for k in kinds if k not in L.KINDS]
        if bad_kinds:
            raise RegimeError(f"regime.kinds has unknown kind(s) {bad_kinds}; expected {list(L.KINDS)}")
        if "counts" not in kinds:
            raise RegimeError("regime.kinds must include 'counts' — it is the modelled quantity")

        return cls(
            store=str(obj["store"]),
            assays=tuple(assays),
            context_bins=ctx,
            train_biosamples=tuple(str(b) for b in bios.get("train", ())),
            eval_biosamples=tuple(str(b) for b in bios.get("eval", ())),
            eval_pairs=pairs,
            train_chroms=tuple(str(c) for c in obj.get("train_chroms", ())),
            eval_chroms=tuple(str(c) for c in obj.get("eval_chroms", ())),
            window_plan=WindowPlan.from_obj(obj.get("window_plan")),
            regions=RegionSet.from_obj(
                obj.get("regions"), base=None if path is None else Path(path).parent
            ),
            dsf=DsfPolicy.from_obj(obj.get("dsf")),
            kinds=kinds,
            seed=int(obj.get("seed", 42)),
            path=None if path is None else str(path),
            raw=raw,
            sha256=sha256,
        )

    @classmethod
    def from_file(cls, path: Path | str) -> "Regime":
        p = Path(path)
        if not p.is_file():
            raise RegimeError(f"regime file not found: {p}")
        raw = p.read_text(encoding="utf-8")
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RegimeError(f"{p}: not valid JSON — {exc}") from exc
        sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return cls.from_dict(obj, path=p, raw=raw, sha256=sha)

    def to_dict(self) -> dict:
        """The regime as JSON, round-trippable through `from_dict`."""
        out = {
            "store": self.store,
            "assays": list(self.assays),
            "biosamples": {"train": list(self.train_biosamples), "eval": list(self.eval_biosamples)},
            "eval_pairs": [list(p) for p in self.eval_pairs],
            "context_bins": self.context_bins,
            "train_chroms": list(self.train_chroms),
            "eval_chroms": list(self.eval_chroms),
            "window_plan": self.window_plan.to_dict(),
            "dsf": self.dsf.to_dict(),
            "kinds": list(self.kinds),
            "seed": self.seed,
        }
        # D32 — OMITTED when unset. A `"regions": null` in `run.json` would be a claim about the
        # training scope that the run never made.
        if self.regions is not None:
            out["regions"] = self.regions.to_dict()
        return out

    # -- derived ------------------------------------------------------------------------------

    @property
    def num_assays(self) -> int:
        return len(self.assays)

    def biosamples(self, split: str) -> Tuple[str, ...]:
        if split not in _SPLITS:
            raise RegimeError(f"unknown split {split!r}; expected one of {list(_SPLITS)}")
        return self.train_biosamples if split == "train" else self.eval_biosamples

    def chroms(self, split: str) -> Tuple[str, ...]:
        if split not in _SPLITS:
            raise RegimeError(f"unknown split {split!r}; expected one of {list(_SPLITS)}")
        return self.train_chroms if split == "train" else self.eval_chroms

    @property
    def has_eval_pairs(self) -> bool:
        """D31 — whether this regime declares an imputation evaluation at all."""
        return bool(self.eval_pairs)

    @property
    def eval_inputs(self) -> Tuple[str, ...]:
        """The prompt biosamples, deduplicated, in declaration order.

        This is the eval-split pool when `eval_pairs` is set: the model is prompted with these and
        scored against their partners. Without pairs the pool is `biosamples.eval`, as before t14.
        """
        seen: List[str] = []
        for a, _ in self.eval_pairs:
            if a not in seen:
                seen.append(a)
        return tuple(seen)

    @property
    def eval_targets(self) -> Tuple[str, ...]:
        """The ground-truth biosamples, deduplicated, in declaration order."""
        seen: List[str] = []
        for _, b in self.eval_pairs:
            if b not in seen:
                seen.append(b)
        return tuple(seen)

    def assay_columns(self, tracks: Sequence[str]) -> List[int]:
        """D14 — the declared assay order as column indices into a file's storage order.

        The one place a name becomes a column. Raises naming every declared assay the file lacks,
        because a silently dropped column shifts every column after it.
        """
        tracks = list(tracks)
        missing = [a for a in self.assays if a not in tracks]
        if missing:
            raise RegimeError(
                f"declared assay(s) {missing} are not in this file's tracks {tracks}"
            )
        return [tracks.index(a) for a in self.assays]

    # -- validation ---------------------------------------------------------------------------

    def validate_against(self, corpus) -> "Regime":
        """D14 — every declared assay, biosample and chromosome must exist. Raise naming it.

        `corpus` is a `reader.CorpusStore`. Assays are checked against the union over the declared
        biosamples: an assay no biosample measures is a typo in the regime, while an assay only
        *some* biosamples have is the normal case the loader fills with `MISSING`.
        """
        names = list(self.train_biosamples) + list(self.eval_biosamples)
        for a, b in self.eval_pairs:
            names.extend((a, b))
        if not names:
            names = list(corpus.biosamples)
        absent = [b for b in names if b not in corpus]
        if absent:
            raise RegimeError(
                f"regime names biosample(s) not in {corpus.root}: {absent}. Names are opaque ids "
                f"used verbatim (D16)."
            )
        available: set = set()
        for b in names:
            for kind in self.kinds:
                if kind in corpus[b].kinds:
                    available.update(corpus[b].assays(kind))
        missing = [a for a in self.assays if a not in available]
        if missing:
            raise RegimeError(
                f"declared assay(s) {missing} are in no biosample of {corpus.root}. "
                f"The store has {sorted(available)}. The regime's `assays` list is the column "
                f"order (D14) — a name that is not in the store is a typo, not an empty column."
            )
        store_chroms = set(corpus.n_bins())
        bad = [c for c in list(self.train_chroms) + list(self.eval_chroms) if c not in store_chroms]
        if bad:
            raise RegimeError(
                f"regime names chromosome(s) not in the store: {bad}. The store has "
                f"{L.sort_chroms(store_chroms)}."
            )
        overlap = set(self.train_chroms) & set(self.eval_chroms)
        if overlap:
            raise RegimeError(
                f"train_chroms and eval_chroms share {sorted(overlap)} — an eval window that also "
                f"trained is not an evaluation."
            )
        # D31 — a target that also trains is not held out. The INPUT is allowed to be a training
        # biosample and normally is: the imputation prompt is exactly a biosample the model knows.
        leaked = sorted(set(self.eval_targets) & set(self.train_biosamples))
        if leaked:
            raise RegimeError(
                f"eval_pairs name {leaked} as imputation target(s), but they are also in "
                f"biosamples.train. The target holds the ground truth being scored; a target the "
                f"model trained on is not an imputation."
            )
        for kind in self.kinds:
            without = [b for b in names if kind not in corpus[b].kinds]
            if without:
                raise RegimeError(
                    f"regime.kinds asks for {kind!r} but {without[:5]} "
                    f"{'…' if len(without) > 5 else ''}have no {kind}.h5. pval is built last "
                    f"(D24); drop it from `kinds` until it exists."
                )
        return self

    # -- the window plan ------------------------------------------------------------------------

    def windows(self, corpus, split: str, *, chroms: Optional[Sequence[str]] = None
                ) -> List[Tuple[str, int]]:
        """`[(chrom, start_bin), …]` for one split, in `sort_chroms` then ascending-start order.

        Eligibility is D12 against `genome/mask.h5`. **When there is no mask layer yet, every
        window is eligible** — the mask is t7's and a store mid-build legitimately has none. That
        is stated rather than silent: `mask_available()` says which case you are in, and the
        `StoreDataset` prints it once.

        D32 adds a SECOND, independent rule on the **train** split only: with `regions` declared,
        a start survives only if its whole window lies inside one region. D12 is untouched — a
        window inside a region can still fail it on blacklist or `N`.
        """
        chrom_list = list(chroms if chroms is not None else self.chroms(split))
        if not chrom_list:
            raise RegimeError(
                f"regime has no {split}_chroms, so there are no {split} windows to plan"
            )
        n_bins = corpus.n_bins()
        stride = self.window_plan.stride(self.context_bins)
        have_mask = self.mask_available(corpus)
        regions = self.regions if split == "train" else None
        out: List[Tuple[str, int]] = []
        for chrom in L.sort_chroms(chrom_list):
            n = int(n_bins[chrom])
            if have_mask:
                mask = corpus.genome.mask(chrom, 0, n)
            else:
                mask = np.ones(n, dtype=np.uint8)
            starts = eligible_starts(
                mask, self.context_bins, self.window_plan.min_valid_frac, stride
            )
            if regions is not None:
                starts = regions.contained_starts(
                    chrom, starts, self.context_bins, corpus.resolution
                )
            out.extend((chrom, int(s)) for s in starts)
        if not out:
            because = "" if regions is None else (
                f" No window fits wholly inside a region of {regions.resolved} "
                f"({len(regions.intervals)} regions, D32), which is checked after the mask."
            )
            raise RegimeError(
                f"no eligible {split} window on {chrom_list} at context_bins={self.context_bins}, "
                f"stride={stride}, min_valid_frac={self.window_plan.min_valid_frac}. Either the "
                f"context is longer than the chromosome or the mask rejects everything.{because}"
            )
        return out

    def mask_available(self, corpus) -> bool:
        try:
            return bool(corpus.genome.has_mask)
        except StoreError:  # pragma: no cover - genome dir missing entirely
            return False
