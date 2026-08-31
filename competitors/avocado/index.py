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
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


class FairnessError(RuntimeError):
    """A column that RIVALS_PLAN.md 6.2 forbids reached the training matrix."""


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
