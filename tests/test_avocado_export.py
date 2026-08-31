"""`tools/avocado_export.py` — the data contract between our store and vendored Avocado.

Two things here can fail silently and score well anyway, which is why they are tested and the file
format is not:

1. A scored track reaching the training matrix. Avocado would fit it and impute it back, and every
   downstream number would look excellent. Nothing later in the pipeline can see it.
2. A blind cell not sharing its `cell_id` with its training counterpart. Avocado would give it a
   fresh, untrained cell factor, and it would be predicted badly for a reason nobody could name.
"""
from __future__ import annotations

import csv
import importlib.util as _u
import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_store_reader import ASSAYS, make_store

_spec = _u.spec_from_file_location(
    "avocado_export", Path(__file__).resolve().parent.parent / "tools" / "avocado_export.py")
AX = _u.module_from_spec(_spec)
_spec.loader.exec_module(AX)


def _regime(store: Path, tmp: Path, **over) -> Path:
    obj = {
        "store": str(store),
        "assays": list(ASSAYS),
        "context_bins": 64,
        "biosamples": {"train": ["T_aa"], "eval": ["V_aa"]},
        "eval_pairs": [["T_aa", "V_aa"]],
        "train_chroms": ["chr1"],
        "eval_chroms": ["chr2"],
        "window_plan": {"type": "tile", "stride_bins": 64, "min_valid_frac": 0.9},
        "dsf": {"policy": "discrete", "levels": [1]},
        "kinds": ["counts", "peaks", "pval"],
        "seed": 42,
    }
    obj.update(over)
    p = tmp / "regime.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Path:
    return make_store(tmp_path_factory.mktemp("avocadostore"))


def test_a_prompt_and_its_target_share_one_cell_id(store, tmp_path) -> None:
    """The blind cell's factor is learned from its training counterpart's tracks, or not at all.

    The identity comes from the DECLARED pairing. `T_aa` and `V_aa` look like an obvious pair to a
    reader and are one only because `eval_pairs` says so (D16, D31).
    """
    ids = AX.cell_ids(AX.Regime.from_file(_regime(store, tmp_path)))
    assert ids["T_aa"] == ids["V_aa"]


def test_a_regime_with_no_declared_pairing_is_refused_not_guessed(store, tmp_path) -> None:
    r = _regime(store, tmp_path, eval_pairs=[], biosamples={"train": ["T_aa"], "eval": []})
    with pytest.raises(SystemExit, match="declares no `eval_pairs`"):
        AX.cell_ids(AX.Regime.from_file(r))


def test_the_matrix_carries_training_columns_only(store, tmp_path) -> None:
    reg = AX.Regime.from_file(_regime(store, tmp_path))
    cols = AX.columns(AX.CorpusStore(reg.store), reg)
    assert {b for b, _ in cols} == {"T_aa"}, "a scored cell reached Avocado's training matrix"


def test_a_scored_cell_in_the_train_split_is_an_error_not_a_filter(store, tmp_path) -> None:
    """Dropping it quietly would let a mis-declared regime export a different matrix and look fine.

    The failure has to be loud here because it is invisible everywhere downstream: Avocado fits the
    leaked track, imputes it back, and every number it produces looks excellent.
    """
    reg = AX.Regime.from_file(
        _regime(store, tmp_path, biosamples={"train": ["T_aa", "V_aa"], "eval": ["V_aa"]}))
    with pytest.raises(SystemExit, match="Rule 1 forbids"):
        AX.columns(AX.CorpusStore(reg.store), reg)


def test_the_export_is_the_three_files_the_trainer_reads(store, tmp_path) -> None:
    out = tmp_path / "d"
    AX.main(["--regime", str(_regime(store, tmp_path)), "--out", str(out), "--chroms", "chr1"])

    names = (out / "tracks.txt").read_text().split()
    Y = np.load(out / "chr1.npy")
    assert Y.shape == (AX.CorpusStore(str(store)).n_bins("chr1"), len(names))
    assert Y.dtype == np.float32

    rows = list(csv.DictReader((out / "bridge.csv").open()))
    assert [r["filename"] for r in rows] == names
    assert {r["assay_id"] for r in rows} <= set(ASSAYS)
    assert len({r["cell_id"] for r in rows}) == 1        # every column is T_aa's cell type

    rec = json.loads((out / "export.json").read_text())
    assert rec["layer"] == "pval" and rec["n_tracks"] == len(names)
