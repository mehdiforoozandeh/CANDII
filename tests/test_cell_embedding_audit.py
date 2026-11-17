"""Close-out A6: the zero-shot trap checks of the cell-embedding arm
(`tools/minicandi/cell_embedding_audit.py`). On the synthetic tables: the index equals
`extra_data.cell_index`; the eval set passes; a scored cell with no training row and a scored
training row each fail; after real AdamW steps (weight decay on), the rows a batch reached pass and
the rows no batch reached fail, although weight decay changed their values."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import cell_embedding_audit as AU  # noqa: E402
from minicandi import registry as R  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi import train as T  # noqa: E402
from minicandi.dataset import OnionDataset  # noqa: E402
from minicandi.extra_data import cell_index  # noqa: E402

L = 64


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> tuple[Path, Path]:
    td = synth.make_synthetic_tables(tmp_path_factory.mktemp("a6_tables"))
    ev = synth.make_evalset(td, n_regions=2, region_bins=4 * L)
    return td, ev


def _cfg(n_cells: int) -> dict:
    base = json.loads((_ROOT / "tools/minicandi/configs/ablations/cell_embedding/rung2/"
                       "cell_embedding.json").read_text())
    return {**base, "context_bins": L, "batch_size": 2, "n_cells": n_cells,
            "cell_state_dropout": 0.0}


def test_index_is_extra_data_cell_index(tables):
    td, _ = tables
    assert AU.index_of(AU.read_rows(td)) == cell_index(td)


def test_synthetic_eval_set_passes(tables):
    td, ev = tables
    rep = AU.table_checks(td, ev)
    assert rep["checks"]["served"]["pass"] and rep["checks"]["no_leak"]["pass"]
    assert rep["scored_cells"] and all(v["id"] > 0 for v in rep["scored_cells"].values())
    assert AU.main(["tables", "--tables", str(td), "--evalset", str(ev)]) == 0


def test_unknown_cell_and_scored_training_row_fail(tables, tmp_path):
    td, ev = tables
    e = json.loads(Path(ev).read_text())
    rows = AU.read_rows(td)
    train_mid = next(m for m, r in rows.items() if r["in_eic"] == "train" and r["cell"])
    e["imputation_targets"].append({"target": train_mid, "inputs": [], "cell": "never_seen",
                                    "split": "val"})
    bad = tmp_path / "evalset_bad.json"
    bad.write_text(json.dumps(e))
    rep = AU.table_checks(td, bad)
    assert rep["checks"]["served"]["unknown_cells"] == ["never_seen"]
    assert rep["checks"]["no_leak"]["scored_train_rows"] == [train_mid]
    assert AU.main(["tables", "--tables", str(td), "--evalset", str(bad)]) == 1


def _train(cfg: dict, td: Path, ids_per_step, seed: int = 3, steps: int = 6):
    torch.manual_seed(seed)                     # as train.py: right before build_model
    model = R.build_model(cfg)
    ds = OnionDataset(td, "train", cfg, 0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=T.WEIGHT_DECAY)
    rng = np.random.default_rng(1)
    model.train()
    for s in range(steps):
        b = ds.batch(2, rng)
        b["cell_id"] = torch.tensor(ids_per_step(s), dtype=torch.int64)
        loss = T.step_loss(model, b, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model, seed


def test_untrained_rows_fail_and_reached_rows_pass(tables):
    td, ev = tables
    table = AU.table_checks(td, ev)
    n = table["n_cells"]
    cfg = _cfg(n)
    first = min(v["id"] for v in table["scored_cells"].values())
    model, seed = _train(cfg, td, lambda s: [first, first])   # one row only
    info = {"config": cfg, "seed": seed, "tables_sha256": table["evalset_tables_sha256"]}
    sd = model.state_dict()
    w0 = AU.initial_embedding(cfg, seed)
    w = sd[AU._emb_key(sd)]
    assert not torch.equal(w, w0)               # weight decay moved every row ...
    share = AU.orthogonal_share(w, w0)
    assert share[first] >= AU.TRAINED_MIN       # ... the reached row left its direction ...
    assert max(s for i, s in enumerate(share) if i != first) < 1e-6   # ... the others did not
    rep = AU.run_checks("synthetic", table, state=sd, info=info)
    assert rep["checks"]["config"]["pass"]
    tr = rep["checks"]["trained"]
    assert not tr["pass"]
    assert "0 (unknown)" in tr["untrained_rows"]
    assert not any(r.startswith(f"{first} ") for r in tr["untrained_rows"])


def test_every_row_reached_passes(tables):
    td, ev = tables
    table = AU.table_checks(td, ev)
    n = table["n_cells"]
    cfg = _cfg(n)
    model, seed = _train(cfg, td, lambda s: [(2 * s) % (n + 1), (2 * s + 1) % (n + 1)],
                         steps=2 * (n + 1))
    info = {"config": cfg, "seed": seed, "tables_sha256": table["evalset_tables_sha256"]}
    rep = AU.run_checks("synthetic", table, state=model.state_dict(), info=info)
    assert rep["checks"]["trained"]["pass"], rep["checks"]["trained"]


def test_config_mismatch_fails(tables):
    td, ev = tables
    table = AU.table_checks(td, ev)
    cfg = _cfg(table["n_cells"] + 1)
    torch.manual_seed(0)
    sd = R.build_model(cfg).state_dict()
    info = {"config": cfg, "seed": 0, "tables_sha256": "not_the_eval_set"}
    rep = AU.run_checks("synthetic", table, state=sd, info=info)
    probs = rep["checks"]["config"]["problems"]
    assert not rep["checks"]["config"]["pass"] and len(probs) == 2
