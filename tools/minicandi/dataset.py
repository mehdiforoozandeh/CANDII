"""miniCANDI table-driven dataset (plan pinned interfaces B and C).

`OnionDataset(tables_dir, mode, config, seed)` reads a tables directory
(`measurements.tsv`, `withheld.tsv`, `covariates.json`; `pairs.tsv` is not needed to sample).
Iterating it yields collated contract batches of `config["batch_size"]` windows (default 8; in
val and blind modes the last batch may be partial). `window(...)` builds one single window (no B
axis) and is the explicit route for any other layout; `contract.collate` stacks windows. Use it
in a DataLoader with `batch_size=None`.

Roles. EIC splits per TRACK, so one cell holds training, validation and blind tracks. Each mode
may read only some `in_eic` roles; any other row raises `RoleError`:
  train  train                  (a validation or blind row is never read, even by request)
  val    train, val
  blind  train, val, blind      (and only when `config["allow_blind"]` is true)

Training mode, per window: one cell; the candidate tracks are that cell's TRAINING tracks only;
a window of `context_bins` bins on one chromosome (the genome mask must mark at least half the
bins valid, 8 tries); a random set of tracks is masked whole (kind 1: no input, its C hidden); for
every other track one input view and one target view are drawn among the views of that track
(same view -> kind 3 with a random `x_bin_mask`; another view -> kind 2). A (source view, target
view) combination listed in `withheld.tsv` is never sampled: every candidate is checked against
the input and target views already accepted in the window, on the same track and across tracks,
and is skipped when it completes a withheld combination. A track with no acceptable candidate
becomes an imputation target alone, or is dropped.

Validation and blind modes are deterministic from `evalset.json` (pinned interface D), over every
region tiled into windows of `context_bins`, for the items of the mode's split:
  imputation      inputs = the evalset's inputs (the same cell's training tracks), target kind 1
  correction      PI ruling 6 floor layout: the source is read from its `pr1` half and the target
                  from its `pr2` half, both at C with the depth halved. Ruling (foreman, item 12
                  of the M1 review): for SCORED correction pairs the source view is built from
                  the pr1 half of the reads and the target is the pr2 half of the target arm, so
                  the two share no reads; the tables builders must make the halves that way.
  reconstruction  every imputation target is also its own input, kind 3, with a bin mask at
                  `bin_mask_frac` seeded by the item index, plus the same cell's training tracks
                  as context (so the copy-leak guardrail has same-view entries on validation)

C and C' follow the file read: a `pr1`/`pr2` half is described at half depth (`input_depth_scale`
and `target_depth_scale` default to 0.5 for a half, 1 otherwise). An imputed target is decoded
with the target's own C'. `meta` holds `cell, chrom, start_bin, inputs, targets` and
`measurements_in`, `measurements_out` (lists of length A, "" where a track has none).

Config keys read here: context_bins, batch_size, qc_on_input, average_input, average_h5,
control_input, predict_control, two_step, base_depth, allow_blind, dna, evalset, genome_dir and
the sampler's own keys in `DEFAULTS`.
  control_input = false    the control (track 35) is never an input (x_avail, c_in stay 0);
                           a control target then becomes kind 1
  predict_control = false  the control is never a target (y_avail, c_out stay 0)
  two_step = true          adds the optional keys `c_base` float32 [B, A, K_META] and `y_base`
                           int32 [B, L, A] (an addition to pinned interface C, decision D15): for
                           every track with a target, C_base of the track's base view (uniform
                           layer preferred, pooled unit preferred) and its counts. Under
                           `base_depth = fixed`, a base deeper than the class reference depth
                           `FIXED_BASE_LOG2_DEPTH` is binomially thinned to it and C_base states
                           the thinned depth. A track without a base view has c_base = 0 (column
                           0, log2 depth, is never 0 for a real view, so it is the mask).
  scramble_covariates      NOT applied here: the train loop must call
                           `covariates.scramble_covariates` on c_in and c_out of each batch.

Leave-one-cell-out average (`average_input`): with `config["average_h5"]` (built once by
`build_average_h5`) it reads one per-track sum of training-cell rates and subtracts the window's
own cell; without it, it reads every other training cell directly (small tables only).

`counts_path` forms:
  h5:<abs path to counts.h5>           pr1.h5 / pr2.h5 are its siblings; writers should chunk
                                       datasets at <= 4096 bins (whole-chromosome gzip chunks
                                       would make every window decompress a chromosome)
  store:<corpus root>:<biosample>:<assay>
      read through `candi.store.reader.CorpusStore`; a row with `arm_knob = depth` is thinned
      read-free from the store track by `candi.store.dataset.thin_counts` with a generator fixed
      by (measurement id, chrom, start bin). For the pilot only: overlapping windows are thinned
      independently, so it is not one consistent thinned genome.

Open h5 handles are LRU-bounded (`config["max_open_files"]`, default 256).

Reused by import: `CorpusStore`, `GenomeView`, `_HandlePool` (`candi.store.reader`),
`thin_counts`, `stable_hash` (`candi.store.dataset`).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from candi.store.dataset import stable_hash, thin_counts  # noqa: E402
from candi.store.reader import CorpusStore, GenomeView, _HandlePool  # noqa: E402
from minicandi import contract as C  # noqa: E402
from minicandi.covariates import CovariateEncoder, load_covariates_json  # noqa: E402

MEASUREMENT_COLUMNS = (
    "measurement_id", "experiment", "assay", "assay_class", "cell", "eic_track", "in_eic", "unit",
    "bio_rep", "tech_rep", "library", "layer", "arm_knob", "arm_value", "parent_measurement_id",
    "reads_subset_of", "depth_reads", "depth_fragments", "run_type", "read_length", "platform",
    "lab", "pipeline", "pipeline_version", "image_digest", "dedup", "mapq_threshold", "antibody",
    "donor", "control_measurement_id", "has_pseudoreps", "has_signal", "nsc", "rsc", "nrf",
    "pbc1", "pbc2", "frip", "frag_len", "counts_path", "sha256")
PAIR_COLUMNS = ("pair_id", "a", "b", "rung", "assay", "cell", "knobs_differing", "subset_pair",
                "split", "never_trained")
WITHHELD_COLUMNS = ("source_knob", "source_value", "target_knob", "target_value", "assay_class")
TABLE_FILES = ("measurements.tsv", "pairs.tsv", "withheld.tsv", "covariates.json",
               "scorable.tsv", "missing.tsv")
MODES = ("train", "val", "blind")
ROLES = {"train": ("train",), "val": ("train", "val"), "blind": ("train", "val", "blind")}
FILES = ("counts", "pr1", "pr2")
HALVES = ("pr1", "pr2")
DEFAULTS = {"context_bins": 1024, "batch_size": 8, "mask_prob_range": (0.2, 0.8),
            "same_view_prob": 0.5, "bin_mask_frac": 0.25, "region_bins": 16384,
            "min_valid_frac": 0.5, "exclude_chroms": ("chrY", "chrM"), "max_open_files": 256,
            "control_input": True, "predict_control": True, "two_step": False,
            "base_depth": "native"}
#: D15 arm "fixed reference depth": log2 reads of the base per assay class (agent decision)
FIXED_BASE_LOG2_DEPTH = {"accessibility_dnase": 23.0, "accessibility_atac": 23.0,
                         "broad_histone": 23.0, "narrow_histone": 23.0, "control": 23.0}
LAYER_PREF = {"uniform": 0, "native": 1, "testbed": 2}


class RoleError(PermissionError):
    """A row whose `in_eic` role the dataset's mode may not read."""


class WithheldError(PermissionError):
    """A training window asked for a combination listed in `withheld.tsv`."""


def read_tsv(path, columns: tuple | None = None) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        if columns is not None and tuple(rd.fieldnames or ()) != columns:
            raise ValueError(f"{path}: columns {rd.fieldnames}, want {list(columns)}")
        return [dict(r) for r in rd]


def tables_sha256(tables_dir) -> str:
    """sha256 over the name and bytes of every table file present, in `TABLE_FILES` order."""
    h = hashlib.sha256()
    for name in TABLE_FILES:
        p = Path(tables_dir) / name
        if p.is_file():
            h.update(name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def view_of(row: dict) -> tuple[str, str]:
    """(knob, value) of a measurement; a base measurement is ("base", "")."""
    knob = (row.get("arm_knob") or "").strip()
    if knob in ("", "base"):
        return ("base", "")
    return (knob, (row.get("arm_value") or "").strip())


def _same_value(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def _base_rank(row: dict) -> tuple:
    return (LAYER_PREF.get(row.get("layer", ""), 9), 0 if row.get("unit") == "pooled" else 1,
            row["measurement_id"])


class Withheld:
    """`withheld.tsv`: (source knob and value, target knob and value, assay class) rows.

    An empty or `*` assay_class applies to every class. A row applies to a (source, target)
    pair when its class is the source's or the target's class.
    """

    def __init__(self, rows: list[dict]):
        self.rows = [{k: (r.get(k) or "").strip() for k in WITHHELD_COLUMNS} for r in rows]
        for r in self.rows:
            if r["source_knob"] == "":
                r["source_knob"] = "base"
            if r["target_knob"] == "":
                r["target_knob"] = "base"

    def __len__(self) -> int:
        return len(self.rows)

    def blocks(self, src: dict, tgt: dict) -> bool:
        if not self.rows:
            return False
        sk, sv = view_of(src)
        tk, tv = view_of(tgt)
        classes = {src.get("assay_class", ""), tgt.get("assay_class", "")}
        for w in self.rows:
            if (w["source_knob"] == sk and _same_value(w["source_value"], sv)
                    and w["target_knob"] == tk and _same_value(w["target_value"], tv)
                    and (w["assay_class"] in ("", "*") or w["assay_class"] in classes)):
                return True
        return False


class OnionDataset(IterableDataset):
    """See the module docstring. `mode` in train | val | blind."""

    def __init__(self, tables_dir, mode: str, config: dict | None = None, seed: int = 0):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} not in {MODES}")
        self.config = dict(DEFAULTS)
        self.config.update({k: v for k, v in (config or {}).items() if v is not None})
        if mode == "blind" and not self.config.get("allow_blind"):
            raise RoleError("mode = blind needs config['allow_blind'] = true (gate G4)")
        if self.config["base_depth"] not in ("native", "fixed"):
            raise ValueError(f"base_depth {self.config['base_depth']!r} not in (native, fixed)")
        self.mode, self.seed = mode, int(seed)
        self.tables_dir = Path(tables_dir)
        self.L = int(self.config["context_bins"])
        self.qc_on_input = bool(self.config.get("qc_on_input"))
        self.K = C.k_input(self.config)

        rows = read_tsv(self.tables_dir / "measurements.tsv", MEASUREMENT_COLUMNS)
        self.rows = {}
        for r in rows:
            if r["measurement_id"] in self.rows:
                raise ValueError(f"duplicate measurement_id {r['measurement_id']!r}")
            if r["assay"] not in C.TRACK_INDEX:
                raise ValueError(f"{r['measurement_id']}: assay {r['assay']!r} not in the panel")
            if r["in_eic"] not in ("train", "val", "blind"):
                raise ValueError(f"{r['measurement_id']}: in_eic {r['in_eic']!r}")
            if not r["assay_class"]:
                r["assay_class"] = C.assay_class(r["assay"])
            self.rows[r["measurement_id"]] = r
        wp = self.tables_dir / "withheld.tsv"
        self.withheld = Withheld(read_tsv(wp, WITHHELD_COLUMNS) if wp.is_file() else [])
        self.encoder = CovariateEncoder(load_covariates_json(self.tables_dir / "covariates.json"))

        self.allowed = {m for m, r in self.rows.items() if r["in_eic"] in ROLES[mode]}
        #: training tracks: (cell, assay) -> sorted measurement ids (train rows only)
        self.train_tracks = self.group_tracks(("train",))
        self.train_cells = sorted({c for c, _ in self.train_tracks})
        #: the training-track base view of each (cell, assay), for the average
        self.train_base = {k: min(v, key=lambda m: (view_of(self.rows[m])[0] != "base",
                                                     _base_rank(self.rows[m])))
                           for k, v in self.train_tracks.items()}

        gdir = self.config.get("genome_dir") or (self.tables_dir.parent.parent / "genome")
        self._pool = _HandlePool()
        self.genome = GenomeView(Path(gdir), self._pool)
        if self.config.get("dna", "own") != "off" and not self.genome.has_dna:
            raise FileNotFoundError(f"{self.genome.dna_path}: missing (set dna = off to run "
                                    f"without sequence)")
        sizes = self.genome.chrom_sizes
        chroms = self.config.get("chroms") or [c for c in sizes
                                               if c not in self.config["exclude_chroms"]]
        self.n_bins = {c: int(sizes[c]) // C.RES for c in chroms}
        self.chroms = [c for c in chroms if self.n_bins[c] >= self.L]
        if not self.chroms:
            raise ValueError(f"no chromosome holds a {self.L}-bin window")
        self._h5: OrderedDict = OrderedDict()
        self._stores: dict = {}
        self._pid = os.getpid()
        self._rng = np.random.default_rng([self.seed, 0])

    def group_tracks(self, roles: tuple) -> dict:
        """(cell, assay) -> sorted measurement ids of the rows whose `in_eic` is in `roles`."""
        out: dict[tuple, list] = {}
        for m in sorted(self.rows):
            r = self.rows[m]
            if r["in_eic"] in roles:
                out.setdefault((r["cell"], r["assay"]), []).append(m)
        return out

    # ---------------------------------------------------------------- reading

    def check_row(self, mid: str) -> dict:
        """The row of `mid`, or RoleError when this mode may not read it."""
        if mid not in self.rows:
            raise KeyError(f"measurement {mid!r} not in measurements.tsv")
        if mid not in self.allowed:
            raise RoleError(f"{mid}: in_eic = {self.rows[mid]['in_eic']} is not readable in "
                            f"mode {self.mode!r}")
        return self.rows[mid]

    def _close_all(self) -> None:
        for f in self._h5.values():
            try:
                f.close()
            except Exception:  # pragma: no cover - best effort
                pass
        self._h5 = OrderedDict()

    def _fresh(self) -> None:
        if os.getpid() != self._pid:            # forked DataLoader worker: drop parent handles
            self._h5, self._stores, self._pid = OrderedDict(), {}, os.getpid()
            self._pool = _HandlePool()
            self.genome = GenomeView(self.genome.dir, self._pool)

    def __getstate__(self) -> dict:
        """Pickle without open handles (DataLoader workers under spawn)."""
        state = dict(self.__dict__)
        state["_h5"], state["_stores"], state["_pid"] = OrderedDict(), {}, -1
        state["_pool"] = _HandlePool()
        state["genome"] = GenomeView(self.genome.dir, state["_pool"])
        return state

    def _h5file(self, path: Path):
        f = self._h5.get(path)
        if f is not None:
            self._h5.move_to_end(path)
            return f
        while len(self._h5) >= int(self.config["max_open_files"]):
            _, old = self._h5.popitem(last=False)
            old.close()
        f = self._h5[path] = h5py.File(path, "r")
        return f

    def read(self, mid: str, chrom: str, start: int, L: int | None = None,
             which: str = "counts") -> np.ndarray:
        """int32 [L] counts of one measurement; bins past the chromosome end are 0."""
        row = self.check_row(mid)
        if which not in FILES:
            raise ValueError(f"which {which!r} not in {FILES}")
        self._fresh()
        L = self.L if L is None else int(L)
        path = row["counts_path"]
        out = np.zeros(L, dtype=np.int32)
        if path.startswith("h5:"):
            p = Path(path[3:])
            if which != "counts":
                p = p.parent / f"{which}.h5"
            ds = self._h5file(p)[chrom]
            end = min(start + L, ds.shape[0])
            if end > start:
                out[: end - start] = ds[start:end]
            return out
        if path.startswith("store:"):
            if which != "counts":
                raise ValueError(f"{mid}: a store: path has no pseudoreplicate halves")
            root, bios, assay = path[len("store:"):].rsplit(":", 2)
            corpus = self._stores.get(root)
            if corpus is None:
                corpus = self._stores[root] = CorpusStore(root)
            bs = corpus[bios]
            end = min(start + L, bs.n_bins(chrom))
            if end > start:
                if assay == C.CONTROL:
                    v = bs.control(chrom, start, end)[:, 0]
                else:
                    v = bs[assay].counts(chrom, start, end)
                out[: end - start] = v
            knob, value = view_of(row)
            if knob == "depth":
                rng = np.random.default_rng(np.random.SeedSequence(
                    [stable_hash(mid), stable_hash(chrom), int(start)]))
                out = thin_counts(out, 1.0 / float(value), rng).astype(np.int32)
            return out
        raise ValueError(f"{mid}: counts_path {path!r} is neither h5: nor store:")

    def dna(self, chrom: str, start: int, L: int | None = None) -> np.ndarray:
        L = self.L if L is None else int(L)
        out = np.full(L * C.RES, 4, dtype=np.uint8)
        end_bp = min((start + L) * C.RES, self.genome.chrom_sizes[chrom]) \
            if self.genome.has_dna else 0
        if end_bp > start * C.RES:
            self._fresh()
            v = self.genome.dna(chrom, start * C.RES, end_bp)
            out[: v.shape[0]] = v
        return out

    # ---------------------------------------------------------------- withheld

    def _compatible(self, cand_in, cand_out, accepted) -> bool:
        rows = self.rows
        ri = rows[cand_in] if cand_in else None
        ro = rows[cand_out]
        if ri is not None and self.withheld.blocks(ri, ro):
            return False
        for a_in, a_out in accepted:
            if ri is not None and self.withheld.blocks(ri, rows[a_out]):
                return False
            if a_in is not None and self.withheld.blocks(rows[a_in], ro):
                return False
        return True

    def window_violations(self, inputs: list, targets: list) -> list:
        """Every (input, target) pair of a window that `withheld.tsv` lists."""
        return [(i, t) for i in inputs for t in targets
                if self.withheld.blocks(self.rows[i], self.rows[t])]

    # ---------------------------------------------------------------- training plan

    def _draw_locus(self, rng) -> tuple[str, int]:
        self._fresh()
        w = np.asarray([self.n_bins[c] - self.L + 1 for c in self.chroms], dtype=np.float64)
        best = None
        for _ in range(8):
            chrom = self.chroms[int(rng.choice(len(self.chroms), p=w / w.sum()))]
            start = int(rng.integers(0, self.n_bins[chrom] - self.L + 1))
            if not self.genome.has_mask:
                return chrom, start
            frac = float(self.genome.mask(chrom, start, start + self.L).mean())
            if frac >= self.config["min_valid_frac"]:
                return chrom, start
            if best is None or frac > best[0]:
                best = (frac, chrom, start)
        return best[1], best[2]

    def plan(self, rng: np.random.Generator) -> dict:
        """One training window's choices, without reading counts."""
        if self.mode != "train":
            raise RuntimeError("plan() is the training sampler; use window() or iterate")
        cell = self.train_cells[int(rng.integers(len(self.train_cells)))]
        chrom, start = self._draw_locus(rng)
        assays = sorted(a for c, a in self.train_tracks if c == cell)
        order = [assays[i] for i in rng.permutation(len(assays))]
        lo, hi = self.config["mask_prob_range"]
        p_mask = float(rng.uniform(lo, hi))
        masked = {a for a in order if rng.random() < p_mask}
        if len(assays) > 1 and len(masked) == len(assays):
            masked.discard(order[0])
        if len(assays) == 1:
            masked = set()
        accepted, entries = [], []
        for a in order:
            views = self.train_tracks[(cell, a)]
            for v_in, v_out in self._candidates(a in masked, views, rng):
                if self._compatible(v_in, v_out, accepted):
                    accepted.append((v_in, v_out))
                    kind = (C.KIND_IMPUTE if v_in is None
                            else C.KIND_SAME if v_in == v_out else C.KIND_CORRECT)
                    entries.append((a, v_in, v_out, kind))
                    break
        return {"cell": cell, "chrom": chrom, "start_bin": start, "entries": entries}

    def _candidates(self, masked: bool, views: list, rng) -> list:
        vs = [views[i] for i in rng.permutation(len(views))]
        impute = [(None, v) for v in vs]
        if masked:
            return impute
        first_in = vs[0]
        others = [v for v in vs if v != first_in]
        same = rng.random() < self.config["same_view_prob"] or not others
        first = (first_in, first_in) if same else (first_in, others[0])
        rest = [(i, o) for i in vs for o in vs if (i, o) != first]
        rest = [rest[j] for j in rng.permutation(len(rest))] if rest else []
        return [first] + rest + impute

    # ---------------------------------------------------------------- materialise

    def _base_view(self, row: dict) -> str | None:
        """The base view of a target's track among the rows this mode may read."""
        cands = [m for m in self.allowed
                 if self.rows[m]["cell"] == row["cell"] and self.rows[m]["assay"] == row["assay"]
                 and view_of(self.rows[m])[0] == "base"]
        return min(cands, key=lambda m: _base_rank(self.rows[m])) if cands else None

    def _base_leg(self, mid: str, chrom: str, start: int, L: int) -> tuple:
        """(c_base float32[K_META], y_base int32[L]) for a target's track (D15)."""
        r = self.rows[mid]
        base = self._base_view(r)
        if base is None:
            return np.zeros(C.K_META, np.float32), np.zeros(L, np.int32)
        br = self.rows[base]
        c = self.encoder.c_out(br)
        y = self.read(base, chrom, start, L)
        if self.config["base_depth"] == "fixed":
            ref = FIXED_BASE_LOG2_DEPTH[br["assay_class"]]
            if c[C.LOG2_DEPTH_COL] > ref:
                q = 2.0 ** (ref - float(c[C.LOG2_DEPTH_COL]))
                rng = np.random.default_rng(np.random.SeedSequence(
                    [stable_hash(base), stable_hash(chrom), int(start), 15]))
                y = thin_counts(y, 1.0 / q, rng).astype(np.int32)
                c = c.copy()
                c[C.LOG2_DEPTH_COL] = ref
        return c, y

    def window(self, chrom: str, start_bin: int, inputs: list, targets: list, *,
               target_file: str = "counts", target_depth_scale: float | None = None,
               input_file: str = "counts", input_depth_scale: float | None = None,
               x_bin_mask: np.ndarray | None = None, avg: bool | None = None,
               extra_meta: dict | None = None) -> dict:
        """One explicit window (no B axis). Kind per target: 1 when its track is not an input,
        3 when the input is the same measurement read from the same file, else 2. In training
        mode a withheld combination raises WithheldError. `x_bin_mask` [L, A] is applied to
        available tracks. A depth scale left None is 0.5 for a pr1/pr2 half and 1 otherwise.
        `control_input` / `predict_control` false drop the control from inputs / targets.
        """
        L = self.L
        if input_depth_scale is None:
            input_depth_scale = 0.5 if input_file in HALVES else 1.0
        if target_depth_scale is None:
            target_depth_scale = 0.5 if target_file in HALVES else 1.0
        rows_in = [self.check_row(m) for m in inputs]
        rows_out = [self.check_row(m) for m in targets]
        cells = {r["cell"] for r in rows_in + rows_out}
        if len(cells) != 1:
            raise ValueError(f"a window holds one cell; got {sorted(cells)}")
        cell = cells.pop()
        if self.mode == "train":
            bad = self.window_violations(inputs, targets)
            if bad:
                raise WithheldError(f"withheld combinations requested in training: {bad}")
        if not self.config["control_input"]:
            keep = [i for i, r in enumerate(rows_in) if r["assay"] != C.CONTROL]
            inputs, rows_in = [inputs[i] for i in keep], [rows_in[i] for i in keep]
        if not self.config["predict_control"]:
            keep = [i for i, r in enumerate(rows_out) if r["assay"] != C.CONTROL]
            targets, rows_out = [targets[i] for i in keep], [rows_out[i] for i in keep]
        x_counts = np.zeros((L, C.A), np.int32)
        y_counts = np.zeros((L, C.A), np.int32)
        x_avail = np.zeros(C.A, bool)
        y_avail = np.zeros(C.A, bool)
        y_kind = np.zeros(C.A, np.uint8)
        c_in = np.zeros((C.A, self.K), np.float32)
        c_out = np.zeros((C.A, C.K_META), np.float32)
        m_in, m_out = [""] * C.A, [""] * C.A
        two = bool(self.config.get("two_step"))
        if two:
            c_base = np.zeros((C.A, C.K_META), np.float32)
            y_base = np.zeros((L, C.A), np.int32)
        for m, r in zip(inputs, rows_in):
            t = C.TRACK_INDEX[r["assay"]]
            if x_avail[t]:
                raise ValueError(f"two inputs for track {r['assay']}")
            x_avail[t] = True
            m_in[t] = m
            x_counts[:, t] = self.read(m, chrom, start_bin, L, input_file)
            c_in[t] = self.encoder.c_in(r, self.qc_on_input, input_depth_scale)
        for m, r in zip(targets, rows_out):
            t = C.TRACK_INDEX[r["assay"]]
            if y_avail[t]:
                raise ValueError(f"two targets for track {r['assay']}")
            y_avail[t] = True
            m_out[t] = m
            same = m_in[t] == m and target_file == input_file
            y_kind[t] = (C.KIND_IMPUTE if not m_in[t]
                         else C.KIND_SAME if same else C.KIND_CORRECT)
            y_counts[:, t] = x_counts[:, t] if same else self.read(m, chrom, start_bin, L,
                                                                   target_file)
            c_out[t] = self.encoder.c_out(r, target_depth_scale)
            if two:
                c_base[t], y_base[:, t] = self._base_leg(m, chrom, start_bin, L)
        bmask = np.zeros((L, C.A), bool)
        if x_bin_mask is not None:
            bmask = np.asarray(x_bin_mask, bool) & x_avail[None, :]
        x_counts[bmask] = 0
        win = {
            "x_counts": torch.from_numpy(x_counts), "x_avail": torch.from_numpy(x_avail),
            "x_bin_mask": torch.from_numpy(bmask),
            "x_dna": torch.from_numpy(self.dna(chrom, start_bin, L)),
            "c_in": torch.from_numpy(c_in), "c_out": torch.from_numpy(c_out),
            "y_counts": torch.from_numpy(y_counts), "y_avail": torch.from_numpy(y_avail),
            "y_kind": torch.from_numpy(y_kind),
        }
        if two:
            win["c_base"] = torch.from_numpy(c_base)
            win["y_base"] = torch.from_numpy(y_base)
        if avg if avg is not None else self.config.get("average_input"):
            win["avg"] = torch.from_numpy(self.average(cell, chrom, start_bin, L))
        meta = {"cell": cell, "chrom": chrom, "start_bin": int(start_bin),
                "inputs": list(inputs), "targets": list(targets),
                "measurements_in": m_in, "measurements_out": m_out,
                "input_file": input_file, "target_file": target_file}
        meta.update(extra_meta or {})
        win["meta"] = meta
        return win

    def materialize(self, plan: dict, rng: np.random.Generator) -> dict:
        inputs = [e[1] for e in plan["entries"] if e[1] is not None]
        targets = [e[2] for e in plan["entries"]]
        bm = np.zeros((self.L, C.A), bool)
        for a, v_in, v_out, kind in plan["entries"]:
            if kind == C.KIND_SAME:
                bm[:, C.TRACK_INDEX[a]] = self._bin_mask(rng)
        return self.window(plan["chrom"], plan["start_bin"], inputs, targets, x_bin_mask=bm,
                           extra_meta={"kinds": {e[0]: int(e[3]) for e in plan["entries"]}})

    def _bin_mask(self, rng) -> np.ndarray:
        col = rng.random(self.L) < float(self.config["bin_mask_frac"])
        if not col.any():
            col[int(rng.integers(self.L))] = True
        return col

    # ---------------------------------------------------------------- average

    def _rate(self, mid: str, chrom: str, start: int, L: int) -> np.ndarray | None:
        r = self.rows[mid]
        depth = float(r["depth_reads"]) if r["depth_reads"] else 0.0
        if depth <= 0:
            return None
        return self.read(mid, chrom, start, L).astype(np.float64) * (1e6 / depth)

    def average(self, cell: str, chrom: str, start: int, L: int | None = None) -> np.ndarray:
        """Leave-one-cell-out per-position average rate, float32 [L, A]: over the training cells
        other than `cell` that hold the track as a TRAINING track, the mean of counts x 1e6 /
        depth_reads of each cell's base view. 0 where no other training cell holds the track.
        With `config["average_h5"]` the sum over all training cells is read once and the
        window's own cell is subtracted."""
        L = self.L if L is None else int(L)
        out = np.zeros((L, C.A), np.float32)
        path = self.config.get("average_h5")
        if path:
            f = self._h5file(Path(path))
            for track, g in f.items():
                t = C.TRACK_INDEX[track]
                n = int(g.attrs["n_cells"])
                if chrom not in g:
                    continue
                ds = g[chrom]
                s = np.zeros(L, np.float64)
                end = min(start + L, ds.shape[0])
                if end > start:
                    s[: end - start] = ds[start:end]
                own = self.train_base.get((cell, track))
                if own is not None:
                    rate = self._rate(own, chrom, start, L)
                    if rate is not None:
                        s -= rate
                        n -= 1
                if n > 0:
                    out[:, t] = np.maximum(s, 0.0) / n
            return out
        n = np.zeros(C.A, np.int64)
        acc = np.zeros((L, C.A), np.float64)
        for (c, a), m in self.train_base.items():
            if c == cell:
                continue
            rate = self._rate(m, chrom, start, L)
            if rate is None:
                continue
            t = C.TRACK_INDEX[a]
            acc[:, t] += rate
            n[t] += 1
        nz = n > 0
        out[:, nz] = (acc[:, nz] / n[nz]).astype(np.float32)
        return out

    # ---------------------------------------------------------------- validation items

    def evalset_items(self) -> list[dict]:
        """Deterministic list of explicit windows for val / blind mode, from evalset.json."""
        if self.mode == "train":
            raise RuntimeError("evalset_items() is for val and blind modes")
        path = self.config.get("evalset")
        if path is None:
            path = (self.tables_dir.parent.parent / "eval" / self.tables_dir.name
                    / "evalset.json")
        ev = json.loads(Path(path).read_text(encoding="utf-8"))
        split = self.mode
        jobs = []
        for t in ev.get("imputation_targets", []):
            if t.get("split") == split:
                jobs.append({"inputs": list(t["inputs"]), "targets": [t["target"]],
                             "task": "imputation", "regime": t.get("regime")})
        for p in ev.get("correction_pairs", []):
            if p.get("split") == split:
                jobs.append({"inputs": [p["source"]], "targets": [p["target"]],
                             "task": "correction", "input_file": "pr1", "target_file": "pr2",
                             "never_trained": p.get("never_trained"),
                             "subset_pair": p.get("subset_pair")})
        seen = set()
        for t in ev.get("imputation_targets", []):
            if t.get("split") == split and t["target"] not in seen:
                seen.add(t["target"])
                ctx = [m for m in t["inputs"] if m != t["target"]]
                jobs.append({"inputs": [t["target"]] + ctx, "targets": [t["target"]],
                             "task": "reconstruction"})
        rb = int(self.config["region_bins"])
        per_task: dict[str, list] = {}
        for j in jobs:
            for chrom, start in ev.get("regions", []):
                for k in range(max(rb // self.L, 1)):
                    s0 = int(start) + k * self.L
                    if s0 + self.L > self.n_bins.get(chrom, 0):
                        break                       # tiles stop at the chromosome end
                    per_task.setdefault(j["task"], []).append(dict(j, chrom=chrom,
                                                                   start_bin=s0))
        # round-robin over tasks, so the first batches already hold every task (the guardrail
        # reads only the first few batches and needs same-view entries)
        lists = [per_task[t] for t in ("imputation", "correction", "reconstruction")
                 if t in per_task]
        items = [lst[i] for i in range(max((len(x) for x in lists), default=0))
                 for lst in lists if i < len(lst)]
        for i, it in enumerate(items):
            it["item"] = i
        return items

    def item_window(self, it: dict) -> dict:
        """The window of one evalset item (deterministic)."""
        bm = None
        if it["task"] == "reconstruction":
            rng = np.random.default_rng([self.seed, int(it["item"]), 3])
            bm = np.zeros((self.L, C.A), bool)
            bm[:, C.TRACK_INDEX[self.rows[it["targets"][0]]["assay"]]] = self._bin_mask(rng)
        meta = {k: v for k, v in it.items()
                if k not in ("inputs", "targets", "chrom", "start_bin", "input_file",
                             "target_file")}
        return self.window(it["chrom"], it["start_bin"], it["inputs"], it["targets"],
                           input_file=it.get("input_file", "counts"),
                           target_file=it.get("target_file", "counts"),
                           x_bin_mask=bm, extra_meta=meta)

    # ---------------------------------------------------------------- iteration

    def __iter__(self):
        """Collated batches of `config["batch_size"]` windows (pinned interface C layout)."""
        info = get_worker_info()
        wid = 0 if info is None else info.id
        nw = 1 if info is None else info.num_workers
        bs = int(self.config["batch_size"])
        if self.mode == "train":
            rng = np.random.default_rng([self.seed, wid + 1])
            while True:
                yield self.batch(bs, rng)
        else:
            items = self.evalset_items()
            buf = []
            for b0 in range(0, len(items), bs):
                if (b0 // bs) % nw != wid:
                    continue
                buf = [self.item_window(it) for it in items[b0:b0 + bs]]
                yield C.collate(buf)

    def batch(self, batch_size: int, rng: np.random.Generator | None = None) -> dict:
        """A collated training batch from the dataset's own generator (or `rng`)."""
        rng = self._rng if rng is None else rng
        return C.collate([self.materialize(self.plan(rng), rng) for _ in range(batch_size)])


def build_average_h5(tables_dir, out_path, config: dict | None = None) -> Path:
    """Write the per-track sum of training-cell rates (counts x 1e6 / depth_reads of each
    training cell's base view) per chromosome, float32, chunks of 4096 bins, with the number of
    cells as the group attr `n_cells`. `OnionDataset` subtracts the window's own cell."""
    cfg = dict(config or {})
    cfg.setdefault("context_bins", 1)
    ds = OnionDataset(tables_dir, "train", cfg)
    per_track: dict[str, list] = {}
    for (c, a), m in sorted(ds.train_base.items()):
        if ds.rows[m]["depth_reads"]:
            per_track.setdefault(a, []).append(m)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        for a, mids in per_track.items():
            g = f.create_group(a)
            g.attrs["n_cells"] = len(mids)
            for chrom, nb in ds.n_bins.items():
                s = np.zeros(nb, np.float64)
                for k in range(0, nb, 1 << 20):
                    L = min(1 << 20, nb - k)
                    for m in mids:
                        s[k:k + L] += ds._rate(m, chrom, k, L)
                g.create_dataset(chrom, data=s.astype(np.float32),
                                 chunks=(min(4096, nb),), compression="gzip")
    return out_path
