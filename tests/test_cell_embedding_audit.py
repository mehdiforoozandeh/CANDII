"""Close-out A6: the zero-shot trap checks of the cell-embedding arm
(`tools/minicandi/cell_embedding_audit.py`), the initial table and the window counts that train.py
writes under `cell_embedding`, the zero-shot scoring switch `cell_id_eval` of the dataset, and
`derive-unknown`. On the synthetic tables: the index equals `extra_data.cell_index`; the eval set
passes; a scored cell with no training row and a scored training row each fail; after real AdamW
steps (weight decay on), the rows a batch reached pass and the rows no batch reached fail,
although weight decay changed their values; without the saved initial table the check fails."""
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
    sd0 = model.state_dict()
    key = AU._emb_key(sd0)
    model.init = {"key": key, "seed": seed, "weight": sd0[key].detach().clone()}
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
    w0 = model.init["weight"]
    w = sd[AU._emb_key(sd)]
    assert not torch.equal(w, w0)               # weight decay moved every row ...
    share = AU.orthogonal_share(w, w0)
    assert share[first] >= AU.TRAINED_MIN       # ... the reached row left its direction ...
    assert max(s for i, s in enumerate(share) if i != first) < 1e-6   # ... the others did not
    rep = AU.run_checks("synthetic", table, state=sd, info=info, init=model.init)
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
    rep = AU.run_checks("synthetic", table, state=model.state_dict(), info=info, init=model.init)
    assert rep["checks"]["trained"]["pass"], rep["checks"]["trained"]
    # without the saved initial table the same run cannot be verified, so it fails
    rep = AU.run_checks(Path("no_such_run"), table, state=model.state_dict(), info=info)
    assert not rep["checks"]["trained"]["pass"] and "unverifiable" in rep["checks"]["trained"]
    bad = {**model.init, "seed": seed + 1}
    rep = AU.run_checks("synthetic", table, state=model.state_dict(), info=info, init=bad)
    assert not rep["checks"]["trained"]["pass"]


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


# ---- train.py: the initial table and the windows per row; dataset switch; derive-unknown

def _small(cfg: dict) -> dict:
    out = {**cfg, "windows_seen": 12, "param_budget": 1.4e5}
    out.pop("heldout_loss", None)
    return out


@pytest.fixture(scope="module")
def trained_run(tables, tmp_path_factory):
    td, _ = tables
    cfg = _small({**_cfg(len(cell_index(td))), "cell_state_dropout": 0.1})
    out = tmp_path_factory.mktemp("a6_run") / "seed0"
    T.train(cfg, td, 0, out)
    return cfg, out


def test_names_agree():
    assert AU.INIT_NAME == T.INIT_CELL_EMB_NAME


def test_train_saves_initial_table_and_windows_per_row(tables, trained_run):
    td, ev = tables
    cfg, out = trained_run
    init = torch.load(out / T.INIT_CELL_EMB_NAME, weights_only=False)
    full = T.full_config(cfg)
    torch.manual_seed(0)
    sd = R.build_model(full).state_dict()
    assert torch.equal(init["weight"], sd[init["key"]]) and init["seed"] == 0
    info = json.loads((out / "run.json").read_text())
    assert sum(info["cell_id_windows"].values()) == info["windows_seen"]
    rep = AU.run_checks(out, AU.table_checks(td, ev))
    # the synthetic eval set records no tables_sha256, so only that problem may appear
    assert all("tables_sha256" in x for x in rep["checks"]["config"]["problems"])
    assert "unverifiable" not in rep["checks"]["trained"]
    assert sum(rep["windows_per_row"].values()) == info["windows_seen"]


def test_train_without_the_arm_writes_neither(tables, tmp_path):
    td, _ = tables
    cfg = _small({**_cfg(1), "cell_embedding": False})
    cfg.pop("n_cells"), cfg.pop("cell_state_dropout")
    T.train(cfg, td, 0, tmp_path / "off")
    assert not (tmp_path / "off" / T.INIT_CELL_EMB_NAME).exists()
    assert "cell_id_windows" not in json.loads((tmp_path / "off" / "run.json").read_text())
    assert "cell_id_windows" not in torch.load(tmp_path / "off" / T.STATE_NAME,
                                               weights_only=False)


def test_cell_id_eval_switch(tables):
    td, ev = tables
    n = len(cell_index(td))
    cfg = {**_cfg(n), "evalset": str(ev)}
    own = OnionDataset(td, "val", {**cfg, "cell_id_eval": "own"}, 0)
    absent = OnionDataset(td, "val", cfg, 0)
    unk = OnionDataset(td, "val", {**cfg, "cell_id_eval": "unknown"}, 0)
    cell = absent.train_cells[-1]
    keys = [d._extra_keys(cell, "chr1", 0, L) for d in (absent, own, unk)]
    assert int(keys[0]["cell_id"]) > 0
    assert int(keys[0]["cell_id"]) == int(keys[1]["cell_id"])     # "own" = the switch off
    assert int(keys[2]["cell_id"]) == 0
    with pytest.raises(ValueError):
        OnionDataset(td, "train", {**cfg, "cell_id_eval": "unknown"}, 0)
    with pytest.raises(ValueError):
        OnionDataset(td, "val", {**cfg, "cell_embedding": False, "cell_id_eval": "unknown"}, 0)
    with pytest.raises(ValueError):
        OnionDataset(td, "val", {**cfg, "cell_id_eval": "other"}, 0)


def test_derive_unknown(tables, trained_run, tmp_path):
    td, ev = tables
    _, run = trained_run
    (run / "DONE").exists() or (run / "DONE").write_text("")
    out = tmp_path / "zs" / "seed0"
    assert AU.main(["derive-unknown", "--run", str(run), "--out", str(out)]) == 0
    model, config = R.load_run(out)
    m0, c0 = R.load_run(run)
    assert config == {**c0, "cell_id_eval": "unknown"}
    for (k, v), (k0, v0) in zip(model.state_dict().items(), m0.state_dict().items()):
        assert k == k0 and torch.equal(v, v0)
    ds = OnionDataset(td, "val", {**config, "evalset": str(ev)}, 0)
    assert int(ds._extra_keys(ds.train_cells[-1], "chr1", 0, L)["cell_id"]) == 0
    with pytest.raises(FileExistsError):
        AU.derive_unknown(run, out)
    rep = AU.run_checks(out, AU.table_checks(td, ev))
    assert any("cell_id_eval" in x for x in rep["checks"]["config"]["problems"])   # not a trained run
