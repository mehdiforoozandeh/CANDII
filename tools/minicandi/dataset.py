"""miniCANDI table-driven dataset (plan pinned interfaces B and C).

`OnionDataset(tables_dir, mode, config, seed)` reads a tables directory
(`measurements.tsv`, `withheld.tsv`, `covariates.json`; `pairs.tsv` is not needed to sample) and
yields single windows in the contract layout; `contract.collate` stacks them into a batch.

Roles. Each mode may read only some `in_eic` roles; any other row raises `RoleError`:
  train  train                  (a validation or blind row is never read, even by request)
  val    train, val
  blind  train, val, blind      (and only when `config["allow_blind"]` is true)

Training mode, per window: one training cell; a window of `context_bins` bins on one chromosome
(the genome mask must mark at least half the bins valid, 8 tries); a random set of the cell's
tracks is masked whole (kind 1: no input, its C hidden); for every other track one input view and
one target view are drawn among the views of that track (same view -> kind 3 with a random
`x_bin_mask`; another view -> kind 2). A (source view, target view) combination listed in
`withheld.tsv` is never sampled: every candidate is checked against the input and target views
already accepted in the window, on the same track and across tracks, and is skipped when it
completes a withheld combination. A track with no acceptable candidate becomes an imputation
target alone, or is dropped.

Validation and blind modes are deterministic from `evalset.json` (pinned interface D): every
imputation target and correction pair of the mode's split, over every region, tiled into windows
of `context_bins`. `window(...)` builds one explicit window and is the route E6 uses for any other
scoring layout (pseudoreplicate halves, depth-halved C', hidden bins).

An imputed target is decoded with the target's own C' (c_out from the target row).

`counts_path` forms:
  h5:<abs path to counts.h5>           pr1.h5 / pr2.h5 are its siblings
  store:<corpus root>:<biosample>:<assay>
      read through `candi.store.reader.CorpusStore`; a row with `arm_knob = depth` is thinned
      read-free from the store track by `candi.store.dataset.thin_counts` with a generator fixed
      by (measurement id, chrom, start bin). For the pilot only: overlapping windows are thinned
      independently, so it is not one consistent thinned genome.

Reused by import: `CorpusStore`, `GenomeView`, `_HandlePool` (`candi.store.reader`),
`thin_counts`, `stable_hash` (`candi.store.dataset`).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
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
DEFAULTS = {"context_bins": 1024, "mask_prob_range": (0.2, 0.8), "same_view_prob": 0.5,
            "bin_mask_frac": 0.25, "region_bins": 16384, "min_valid_frac": 0.5,
            "exclude_chroms": ("chrY", "chrM")}


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


class Withheld:
    """`withheld.tsv`: (source knob and value, target knob and value, assay class) rows.

    An empty or `*` assay_class applies to every class. A row applies to a (source, target)
    pair when its class is the source's or the target's class.
    """

    def __init__(self, rows: list[dict]):
        self.rows = [{k: (r.get(k) or "").strip() for k in WITHHELD_COLUMNS} for r in rows]
        for r in self.rows:
            if r["source_knob"] in ("",):
                r["source_knob"] = "base"
            if r["target_knob"] in ("",):
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
        self.config.update(config or {})
        if mode == "blind" and not self.config.get("allow_blind"):
            raise RoleError("mode = blind needs config['allow_blind'] = true (gate G4)")
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

        allowed = ROLES[mode]
        self.allowed = {m for m, r in self.rows.items() if r["in_eic"] in allowed}
        #: training tracks: (cell, assay) -> sorted measurement ids (train rows only)
        self.train_tracks: dict[tuple, list] = {}
        for m in sorted(self.rows):
            r = self.rows[m]
            if r["in_eic"] == "train":
                self.train_tracks.setdefault((r["cell"], r["assay"]), []).append(m)
        self.train_cells = sorted({c for c, _ in self.train_tracks})

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
        self._h5: dict = {}
        self._stores: dict = {}
        self._pid = os.getpid()
        self._rng = np.random.default_rng([self.seed, 0])

    # ---------------------------------------------------------------- reading

    def check_row(self, mid: str) -> dict:
        """The row of `mid`, or RoleError when this mode may not read it."""
        if mid not in self.rows:
            raise KeyError(f"measurement {mid!r} not in measurements.tsv")
        if mid not in self.allowed:
            raise RoleError(f"{mid}: in_eic = {self.rows[mid]['in_eic']} is not readable in "
                            f"mode {self.mode!r}")
        return self.rows[mid]

    def _fresh(self) -> None:
        if os.getpid() != self._pid:            # forked DataLoader worker: drop parent handles
            self._h5, self._stores, self._pid = {}, {}, os.getpid()
            self._pool = _HandlePool()
            self.genome = GenomeView(self.genome.dir, self._pool)

    def __getstate__(self) -> dict:
        """Pickle without open handles (DataLoader workers under spawn)."""
        state = dict(self.__dict__)
        state["_h5"], state["_stores"], state["_pid"] = {}, {}, -1
        state["_pool"] = _HandlePool()
        state["genome"] = GenomeView(self.genome.dir, state["_pool"])
        return state

    def _h5file(self, path: Path):
        f = self._h5.get(path)
        if f is None:
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
            cands = self._candidates(a in masked, views, rng)
            for v_in, v_out in cands:
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

    def window(self, chrom: str, start_bin: int, inputs: list, targets: list, *,
               target_file: str = "counts", target_depth_scale: float = 1.0,
               input_file: str = "counts", x_bin_mask: np.ndarray | None = None,
               avg: bool | None = None, extra_meta: dict | None = None) -> dict:
        """One explicit window (no B axis). Kind per target: 1 when its track is not an input,
        3 when the input is the same measurement, else 2. In training mode a withheld
        combination raises WithheldError. `x_bin_mask` [L, A] is applied to available tracks.
        """
        L = self.L
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
        x_counts = np.zeros((L, C.A), np.int32)
        y_counts = np.zeros((L, C.A), np.int32)
        x_avail = np.zeros(C.A, bool)
        y_avail = np.zeros(C.A, bool)
        y_kind = np.zeros(C.A, np.uint8)
        c_in = np.zeros((C.A, self.K), np.float32)
        c_out = np.zeros((C.A, C.K_META), np.float32)
        bmask = np.zeros((L, C.A), bool)
        in_by_track = {}
        for m, r in zip(inputs, rows_in):
            t = C.TRACK_INDEX[r["assay"]]
            if x_avail[t]:
                raise ValueError(f"two inputs for track {r['assay']}")
            x_avail[t] = True
            in_by_track[t] = m
            x_counts[:, t] = self.read(m, chrom, start_bin, L, input_file)
            c_in[t] = self.encoder.c_in(r, self.qc_on_input)
        for m, r in zip(targets, rows_out):
            t = C.TRACK_INDEX[r["assay"]]
            if y_avail[t]:
                raise ValueError(f"two targets for track {r['assay']}")
            y_avail[t] = True
            src = in_by_track.get(t)
            y_kind[t] = (C.KIND_IMPUTE if src is None
                         else C.KIND_SAME if (src == m and target_file == input_file)
                         else C.KIND_CORRECT)
            if src == m and target_file == input_file:
                y_counts[:, t] = x_counts[:, t]
            else:
                y_counts[:, t] = self.read(m, chrom, start_bin, L, target_file)
            c_out[t] = self.encoder.c_out(r, target_depth_scale)
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
        if avg if avg is not None else self.config.get("average_input"):
            win["avg"] = torch.from_numpy(self.average(cell, chrom, start_bin, L))
        meta = {"cell": cell, "chrom": chrom, "start_bin": int(start_bin),
                "inputs": list(inputs), "targets": list(targets)}
        meta.update(extra_meta or {})
        win["meta"] = meta
        return win

    def materialize(self, plan: dict, rng: np.random.Generator) -> dict:
        inputs = [e[1] for e in plan["entries"] if e[1] is not None]
        targets = [e[2] for e in plan["entries"]]
        bm = np.zeros((self.L, C.A), bool)
        frac = float(self.config["bin_mask_frac"])
        for a, v_in, v_out, kind in plan["entries"]:
            if kind == C.KIND_SAME:
                col = rng.random(self.L) < frac
                if not col.any():
                    col[int(rng.integers(self.L))] = True
                bm[:, C.TRACK_INDEX[a]] = col
        return self.window(plan["chrom"], plan["start_bin"], inputs, targets, x_bin_mask=bm,
                           extra_meta={"kinds": {e[0]: int(e[3]) for e in plan["entries"]}})

    def average(self, cell: str, chrom: str, start: int, L: int | None = None) -> np.ndarray:
        """Leave-one-cell-out per-position average rate, float32 [L, A]: over the training cells
        other than `cell` that hold the track, the mean of counts x 1e6 / depth_reads of each
        cell's base view (the first base measurement in id order, else its first view). 0 where
        no other training cell holds the track."""
        L = self.L if L is None else int(L)
        out = np.zeros((L, C.A), np.float32)
        n = np.zeros(C.A, np.int64)
        for (c, a), views in self.train_tracks.items():
            if c == cell:
                continue
            base = [v for v in views if view_of(self.rows[v])[0] == "base"]
            m = (base or views)[0]
            r = self.rows[m]
            depth = float(r["depth_reads"]) if r["depth_reads"] else 0.0
            if depth <= 0:
                continue
            t = C.TRACK_INDEX[a]
            out[:, t] += self.read(m, chrom, start, L) * (1e6 / depth)
            n[t] += 1
        nz = n > 0
        out[:, nz] /= n[nz]
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
                             "task": "correction", "never_trained": p.get("never_trained"),
                             "subset_pair": p.get("subset_pair")})
        rb = int(self.config["region_bins"])
        items = []
        for j in jobs:
            for chrom, start in ev.get("regions", []):
                for k in range(max(rb // self.L, 1)):
                    s0 = int(start) + k * self.L
                    if s0 + self.L > self.n_bins.get(chrom, 0):
                        break                       # tiles stop at the chromosome end
                    items.append(dict(j, chrom=chrom, start_bin=s0))
        return items

    # ---------------------------------------------------------------- iteration

    def __iter__(self):
        info = get_worker_info()
        wid = 0 if info is None else info.id
        nw = 1 if info is None else info.num_workers
        if self.mode == "train":
            rng = np.random.default_rng([self.seed, wid + 1])
            while True:
                yield self.materialize(self.plan(rng), rng)
        else:
            for i, it in enumerate(self.evalset_items()):
                if i % nw == wid:
                    meta = {k: v for k, v in it.items()
                            if k not in ("inputs", "targets", "chrom", "start_bin")}
                    yield self.window(it["chrom"], it["start_bin"], it["inputs"],
                                      it["targets"], extra_meta=meta)

    def batch(self, batch_size: int, rng: np.random.Generator | None = None) -> dict:
        """A collated training batch from the dataset's own generator (or `rng`)."""
        rng = self._rng if rng is None else rng
        return C.collate([self.materialize(self.plan(rng), rng) for _ in range(batch_size)])
