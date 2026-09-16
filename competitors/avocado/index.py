#!/usr/bin/env python3
"""The (cell, assay) index space Avocado factorises over, built from the regime alone.

Avocado is a factorisation: a prediction for (cell, assay, position) is a function of a cell
embedding, an assay embedding and that position's genomic factors.  So the one thing this module
has to get right is *what counts as the same cell*.

`T_K562` and `V_K562` are two views of one biosample -- the tracks that are available as input and
the tracks held out as truth.  Avocado must give them ONE embedding, or a V_ pair has no cell
representation at all and there is nothing to impute from.  But the embedding must be fitted from
the T_ side only (RIVALS_PLAN.md 6.2), which is what `train_columns` below enforces.

The identification comes from the regime's own `eval_pairs`, which declares `[T_X, V_X]` verbatim.
Nothing here splits a biosample name apart to discover that two ids are related -- STORE.md's D16
says a biosample name is an opaque id, and it is treated as one.  A regime that declared
`[T_A, V_B]` would be honoured exactly as written.

Two more index spaces live here because they are the same kind of thing -- a set derived from the
regime and from nothing else, so there is one place to audit:

* `select_pairs` / `write_select_regime` -- the V_-only panel the checkpoint-selection loop scores
  against (BENCHMARK_DESIGN.md 5).
* `region_layout` -- the compact coordinate the shared fit trains in when the regime declares a
  `regions` BED (D32).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


class FairnessError(RuntimeError):
    """A column that RIVALS_PLAN.md 6.2 forbids reached the training matrix."""


class RegionScopeError(RuntimeError):
    """A `regions` regime whose BED leaves the shared fit nothing to train on."""


def load_regime(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cell_index(regime: dict) -> Tuple[List[str], Dict[str, int]]:
    """`(cell names, biosample id -> cell index)`.

    One index per biosample the regime declares, EXCEPT that an eval biosample declared as the
    partner of a train biosample shares the train biosample's index.  The train biosample's own id
    is the class name, so a checkpoint written by this code names cells in T_ terms only.
    """
    train = list(regime["biosamples"]["train"])
    names = sorted(train)
    ix = {c: i for i, c in enumerate(names)}
    for row in regime.get("eval_pairs", []):
        src, tgt = str(row[0]), str(row[1])
        if src not in ix:
            raise FairnessError(
                f"eval_pairs names `{src}` as an input biosample but it is not in "
                f"biosamples.train; Avocado would have no fitted embedding for it.")
        ix[tgt] = ix[src]
    return names, ix


def assay_index(regime: dict) -> Tuple[List[str], Dict[str, int]]:
    """The regime's `assays` list is the column order (D14) -- declared, never derived."""
    names = list(regime["assays"])
    return names, {a: i for i, a in enumerate(names)}


def train_columns(regime: dict, corpus) -> List[Tuple[str, str]]:
    """`[(biosample, assay), ...]` -- every observed track Avocado is allowed to see.

    **This function is where RIVALS_PLAN.md 6.2 is enforced.**  The column list is drawn from
    `regime["biosamples"]["train"]` and from nowhere else, and any biosample that also appears as an
    eval TARGET raises rather than being filtered quietly: a fairness rule that silently drops a
    column is a fairness rule nobody can audit.  Every other stage of this pipeline -- binning,
    training, the sigma fit -- reads its columns from here, so there is one place to check.
    """
    train = list(regime["biosamples"]["train"])
    targets = {str(row[1]) for row in regime.get("eval_pairs", [])}
    leaked = sorted(set(train) & targets)
    if leaked:
        raise FairnessError(
            f"biosamples.train contains {leaked}, which eval_pairs also names as imputation "
            f"TARGETS. Training on a target biosample's tracks is the leak RIVALS_PLAN.md 6.2 "
            f"forbids outright.")
    declared = set(regime["assays"])
    cols: List[Tuple[str, str]] = []
    for b in sorted(train):
        have = [a for a in regime["assays"] if a in set(corpus[b].assays()) & declared]
        cols.extend((b, a) for a in have)
    if not cols:
        raise FairnessError("no training columns: the regime declares no train biosample with a "
                            "declared assay.")
    return cols


def write_tracks(path: str | Path, cols: Sequence[Tuple[str, str]], regime: dict) -> None:
    """`tracks.csv` -- the column order of every `<chrom>.npy`, plus its index-space translation.

    Written through a per-process temp and `os.replace`, because all 23 binning array tasks write
    it and they write it concurrently. The content is identical from every task, so an atomic
    rename makes the race harmless; a plain `write_text` would let one reader see a half-written
    file and mis-map every column.
    """
    import os
    _, cix = cell_index(regime)
    _, aix = assay_index(regime)
    lines = ["col,biosample,assay,cell_idx,assay_idx"]
    for i, (b, a) in enumerate(cols):
        lines.append(f"{i},{b},{a},{cix[b]},{aix[a]}")
    path = Path(path)
    tmp = path.with_suffix(f".csv.{os.getpid()}.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_tracks(path: str | Path) -> List[Tuple[int, str, str, int, int]]:
    rows = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines()[1:]:
        if not ln.strip():
            continue
        col, b, a, ci, ai = ln.split(",")
        rows.append((int(col), b, a, int(ci), int(ai)))
    return rows


# ---------------------------------------------------------------------------------------------
# the selection panel -- BENCHMARK_DESIGN.md 5
# ---------------------------------------------------------------------------------------------

#: The panel prefix a selection loop may read.  Both live regimes declare all 38 pairs inside one
#: file (26 V_ + 12 B_), so a loop handed the shipped regime would score B_ at every check.  The PI
#: ruling of 2026-08-31 is absolute and is not "keep B_ out of the argmax": *"never ever we use B_
#: in training -- V_ is only for checkpoint selection and monitoring, not training"*.  B_ is not
#: read.
SELECT_PREFIX = "V_"


def _pair_ends(row) -> Tuple[str, str]:
    """`(input, target)` from either declared spelling -- a 2-list, or an object with both keys."""
    if isinstance(row, dict):
        return str(row["input"]), str(row["target"])
    return str(row[0]), str(row[1])


def select_pairs(regime: dict, prefix: str = SELECT_PREFIX):
    """`(kept, dropped)` -- the declared eval pairs whose TARGET is on the selection panel.

    This reads a prefix off a biosample name, which D16 forbids a *loader* to do.  The licence is
    `tools/declare_eval_pairs.py split`'s: a person may parse a name deliberately, once, somewhere
    it can be audited, and record the result.  This is that one place for Avocado, it is called at
    the top of a run, never inside the data path, and the derived list is written to disk beside the
    checkpoint so a reviewer can count the pairs instead of trusting this docstring.
    """
    pairs = [_pair_ends(r) for r in regime.get("eval_pairs", [])]
    kept = [[i, t] for i, t in pairs if t.startswith(prefix)]
    dropped = [[i, t] for i, t in pairs if not t.startswith(prefix)]
    return kept, dropped


def write_select_regime(src: str | Path, dst: str | Path, prefix: str = SELECT_PREFIX) -> dict:
    """Derive the V_-only regime the selection loop scores against, and write it to `dst`.

    Derived rather than shipped as a second config, for the reason `slurm/t81_train_candi.sh` gives
    of CANDI's copy: a second 340-line file drifts from its original and nothing notices.

    `regions.bed` RESOLVES AGAINST THE REGIME FILE'S OWN DIRECTORY (`store/regime.py`), so the
    derived copy carries an ABSOLUTE BED path -- it lands next to a checkpoint on /scratch, and a
    relative path there fails D32's sha256 gate rather than merely missing the file.
    """
    src, dst = Path(src), Path(dst)
    d = json.loads(src.read_text(encoding="utf-8"))
    kept, dropped = select_pairs(d, prefix)
    if not kept:
        raise FairnessError(
            f"{src} declares no eval pair whose target starts with {prefix!r}; there is no "
            f"selection panel to score and BENCHMARK_DESIGN.md 5's uniform rule cannot be met.")
    d["eval_pairs"] = kept
    # `eval_pairs` being set makes `biosamples.eval` inert, but leaving all 38 cells in a V_-only
    # config would be a false claim about the eval split.
    d["biosamples"]["eval"] = sorted({t for _, t in kept})
    if d.get("regions"):
        d["regions"]["bed"] = str((src.parent / d["regions"]["bed"]).resolve())
    d["_comment"] = (
        f"DERIVED by competitors/avocado/index.py::write_select_regime from {src.name} -- "
        f"eval_pairs filtered to {prefix} targets, so the selection loop never reads B_ "
        f"(BENCHMARK_DESIGN.md 5). {len(kept)} kept, {len(dropped)} dropped. "
        f"Do not edit; edit the source.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    return d


# ---------------------------------------------------------------------------------------------
# the BED training scope -- D32, at Avocado's own unit
# ---------------------------------------------------------------------------------------------


def region_layout(regime_path: str | Path, chroms: Sequence[str], resolution: int = 25,
                  coarse_stride: int = 200):
    """`([(chrom, first_bin, end_bin, slot0), ...], n_slots)` -- the shared fit's compact coordinate.

    D32's rule is CONTAINMENT, and CANDI applies it to a 768-bin window: a window counts only if it
    lies wholly inside one region.  Avocado is per-position, so the same rule applies at its own
    unit -- **a 25 bp bin counts only if the bin lies wholly inside one region** -- which is exactly
    `RegionSet.bin_spans`, the primitive CANDI's window filter is built on.  Reusing it rather than
    re-deriving the arithmetic is why the two scopes cannot drift: the BED, its sha256 pin and the
    half-open containment all come from one place.  Over `configs/regime.eic_pilot.json`'s 18 train
    chromosomes that is 40 regions and 1,023,489 bins, the figure BENCHMARK_DESIGN.md 3.1 pins.

    Avocado cannot hold genomic factors for whole chromosomes it will never predict on -- 18 of them
    is ~11 G parameters -- so the contained bins are PACKED into one compact axis and the model's
    `n_bins` is that axis.  The packing preserves the one property 3.1 insists on, that the grid is
    anchored at chromosome bin 0 and never re-anchored per region:

    * each region's `slot0` is chosen so that `slot0 - first_bin` is a multiple of `coarse_stride`,
      so `slot // 10` and `slot // 200` -- the 250 bp and 5 kbp factor grids of `vendor/avocado.py`
      -- cut the genome at exactly the absolute chromosome coordinates they would have cut it at on
      a whole-chromosome fit;
    * regions are separated to the next `coarse_stride` boundary, so no coarse factor is shared by
      two regions, or by two chromosomes.

    The alignment costs at most `coarse_stride - 1` unused slots per region (<= 8 k over the pilot
    scope, 0.8 %).  Those slots carry no data and are never trained on: the caller trains on
    `region_slots`, not on `arange(n_slots)`.
    """
    from candi.store.regime import RegionSet             # lazy: bin_store's binning array imports
    regime_path = Path(regime_path)                      # this module before torch is available
    obj = json.loads(regime_path.read_text(encoding="utf-8")).get("regions")
    if not obj:
        raise RegionScopeError(f"{regime_path} declares no `regions` block")
    rs = RegionSet.from_obj(obj, base=regime_path.parent)

    spans, slot = [], 0
    for c in chroms:
        for a, b in rs.bin_spans(c, resolution):
            slot0 = slot + ((a - slot) % coarse_stride)
            spans.append((c, int(a), int(b), int(slot0)))
            slot = -(-(slot0 + (b - a)) // coarse_stride) * coarse_stride
    if not spans:
        raise RegionScopeError(
            f"{rs.resolved} has no region on any of {list(chroms)}, so the shared fit has no "
            f"training scope. Rule 2 cuts regions by the regime's chromosome list, not by the BED.")
    n_slots = max(s0 + (b - a) for _, a, b, s0 in spans)
    return spans, int(n_slots)


def region_slots(spans) -> "object":
    """Every REAL slot of a `region_layout`, ascending -- the positions the shared fit trains on."""
    import numpy as np
    return np.concatenate([np.arange(s0, s0 + (b - a), dtype=np.int64) for _, a, b, s0 in spans])


def write_layout(path: str | Path, spans, n_slots: int) -> None:
    """`regions_layout.csv` -- the compact axis, in the same shape and spirit as `tracks.csv`."""
    lines = [f"# n_slots={n_slots}", "chrom,first_bin,end_bin,slot0"]
    lines += [f"{c},{a},{b},{s0}" for c, a, b, s0 in spans]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_layout(path: str | Path):
    """`(spans, n_slots)` back off disk."""
    text = Path(path).read_text(encoding="utf-8").splitlines()
    n_slots = int(text[0].split("=", 1)[1])
    spans = []
    for ln in text[2:]:
        if ln.strip():
            c, a, b, s0 = ln.split(",")
            spans.append((c, int(a), int(b), int(s0)))
    return spans, n_slots
