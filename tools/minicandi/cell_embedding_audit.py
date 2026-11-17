"""The zero-shot trap checks of the learned cell-type embedding arm (close-out item A6).

    python tools/minicandi/cell_embedding_audit.py tables --tables T --evalset E [--json OUT]
    python tools/minicandi/cell_embedding_audit.py runs --tables T --evalset E --run R [--run R ...]
        [--json OUT]

The arm `cell_embedding` (`extra_inputs.CellState`) looks up row `cell_id` of an `nn.Embedding`
of `n_cells + 1` rows. `extra_data.cell_index` numbers the cells of the `in_eic = train` rows
from 1; any other cell gets row 0 (unknown), which training reaches through the per-sample
dropout `cell_state_dropout` and through windows of the shared-control pseudo-cell (cell "").
A cell that the eval set scores but that no training row holds would read row 0, and a row that
no training window reached keeps its random initial value. Either one makes the arm's score
mean something else than "the cell is known". These checks catch both, and a scored track that
trained its own cell's row (leakage).

`tables` (before training; needs only the tables and the eval set; standard library only):
  index     the cell index of the tables: sorted training cells, numbered from 1.
  served    every cell of the scored items (imputation targets, control targets, correction pairs)
            has an index > 0. FAIL lists the cells that would read the unknown row.
  no_leak   no scored measurement (imputation and control target, correction source and target)
            is an `in_eic = train` row, so no scored track trains its own cell's row. FAIL lists
            them.
`runs` (after training; also needs torch): `tables`, plus per run
  config    run.json `config` has `cell_embedding` true and `n_cells` = the index size, and
            run.json `tables_sha256` equals the eval set's.
  trained   every row a scored cell reads, and row 0, left the direction of its initial value.
            The initial value is rebuilt as train.py builds it (`torch.manual_seed(seed)` right
            before `registry.build_model(config)`). AdamW weight decay scales a row that got no
            gradient by a positive factor and keeps its direction, so the measure is the share of
            the row orthogonal to its initial value, |w - proj_w0(w)| / |w|. A row is trained
            when the share is at least TRAINED_MIN. FAIL lists the untrained rows.
Exit code 0 when every check passes, 1 when one fails, 2 on bad input.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

#: a row whose orthogonal share is below this has kept its initial direction (never trained)
TRAINED_MIN = 1e-3
#: eval-set item lists and the measurement keys of each item that are scored
SCORED_KEYS = {"imputation_targets": ("target",), "control_targets": ("target",),
               "correction_pairs": ("source", "target")}
EMB_SUFFIX = "cell_state.emb.weight"


def read_rows(tables) -> dict[str, dict]:
    """measurement_id -> row of `<tables>/measurements.tsv`."""
    with open(Path(tables) / "measurements.tsv", newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for col in ("measurement_id", "cell", "in_eic"):
            if col not in (rd.fieldnames or ()):
                raise ValueError(f"{tables}/measurements.tsv lacks column {col!r}")
        return {r["measurement_id"]: r for r in rd}


def index_of(rows: dict[str, dict]) -> dict[str, int]:
    """`extra_data.cell_index` from rows already read (the same rule; a test holds them equal)."""
    cells = sorted({r["cell"] for r in rows.values() if r["in_eic"] == "train" and r["cell"]})
    return {c: i + 1 for i, c in enumerate(cells)}


def table_checks(tables, evalset) -> dict:
    rows = read_rows(tables)
    ev = json.loads(Path(evalset).read_text(encoding="utf-8"))
    idx = index_of(rows)
    served: dict[str, int] = {}
    unknown, missing, leaks = set(), [], []
    for key, fields in SCORED_KEYS.items():
        for item in ev.get(key, []):
            cell = item.get("cell")
            if cell is None and item.get("target") in rows:   # older eval sets: the target's cell
                cell = rows[item["target"]]["cell"]
            if cell is not None:
                served[cell] = served.get(cell, 0) + 1
                if idx.get(cell, 0) == 0:
                    unknown.add(cell)
            for f in fields:
                mid = item.get(f)
                if mid is None:
                    continue
                r = rows.get(mid)
                if r is None:
                    missing.append(mid)
                elif r["in_eic"] == "train":
                    leaks.append(mid)
    train_tracks = {}
    for r in rows.values():
        if r["in_eic"] == "train" and r["cell"]:
            train_tracks.setdefault(r["cell"], set()).add(r.get("assay", ""))
    checks = {
        "served": {"pass": not unknown and not missing,
                   "unknown_cells": sorted(unknown), "missing_measurements": sorted(set(missing))},
        "no_leak": {"pass": not leaks, "scored_train_rows": sorted(set(leaks))},
    }
    return {"tables": str(Path(tables).resolve()), "evalset": str(Path(evalset).resolve()),
            "evalset_tables_sha256": ev.get("tables_sha256"),
            "n_cells": len(idx), "index": idx,
            "scored_cells": {c: {"id": idx.get(c, 0), "scored_items": n,
                                 "training_assays": len(train_tracks.get(c, ()))}
                             for c, n in sorted(served.items())},
            "checks": checks}


def orthogonal_share(w, w0) -> list[float]:
    """Per row of `w` [N, H]: |w - proj_w0(w)| / |w| (0 for a row parallel to its initial value)."""
    import torch
    w, w0 = w.double(), w0.double()
    coef = (w * w0).sum(1, keepdim=True) / (w0 * w0).sum(1, keepdim=True).clamp_min(1e-300)
    resid = (w - coef * w0).norm(dim=1)
    return (resid / w.norm(dim=1).clamp_min(1e-300)).tolist()


def _emb_key(state: dict) -> str:
    keys = [k for k in state if k.endswith(EMB_SUFFIX)]
    if len(keys) != 1:
        raise ValueError(f"want one '*{EMB_SUFFIX}' in the state dict, found {keys}")
    return keys[0]


def initial_embedding(config: dict, seed: int):
    """The embedding table as train.py initializes it for (config, seed)."""
    import torch
    from minicandi.registry import build_model
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = build_model(config)
    sd = model.state_dict()
    return sd[_emb_key(sd)].detach().clone()


def run_checks(run, table: dict, state: dict | None = None, info: dict | None = None) -> dict:
    """The `config` and `trained` checks of one run directory (`run.json`, `ckpt.pt`). `state`
    and `info` replace the files (tests)."""
    import torch
    run = Path(run)
    if info is None:
        info = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if state is None:
        state = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)["state_dict"]
    cfg, seed = dict(info["config"]), int(info["seed"])
    problems = []
    if cfg.get("cell_embedding") is not True:
        problems.append("config: cell_embedding is not true")
    if cfg.get("n_cells") != table["n_cells"]:
        problems.append(f"config: n_cells {cfg.get('n_cells')!r} != {table['n_cells']} cells")
    if info.get("tables_sha256") != table["evalset_tables_sha256"]:
        problems.append("tables_sha256 of the run differs from the eval set's")
    out = {"run": str(run), "seed": seed, "checks": {"config": {"pass": not problems,
                                                                 "problems": problems}}}
    if cfg.get("cell_embedding") is not True:
        out["checks"]["trained"] = {"pass": False, "untrained_rows": [], "share": {}}
        return out
    w = state[_emb_key(state)].detach().cpu()
    w0 = initial_embedding(cfg, seed)
    if w.shape != w0.shape:
        raise ValueError(f"{run}: embedding {tuple(w.shape)} != initial {tuple(w0.shape)}")
    share = orthogonal_share(w, w0)
    rows = {0: "unknown"} | {v["id"]: c for c, v in table["scored_cells"].items() if v["id"] > 0}
    untrained = sorted(f"{i} ({rows[i]})" for i in rows if share[i] < TRAINED_MIN)
    out["checks"]["trained"] = {
        "pass": not untrained, "untrained_rows": untrained,
        "share": {rows[i]: round(share[i], 6) for i in sorted(rows)},
        "share_min_all_rows": round(min(share), 6)}
    return out


def _passed(rep: dict) -> bool:
    return all(c["pass"] for c in rep["checks"].values())


def _print(rep: dict, label: str) -> None:
    for name, c in rep["checks"].items():
        extra = {k: v for k, v in c.items() if k not in ("pass", "share") and v}
        print(f"{'PASS' if c['pass'] else 'FAIL'} {label} {name}"
              + (f": {json.dumps(extra)}" if extra else ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tables", "runs"):
        p = sub.add_parser(name)
        p.add_argument("--tables", required=True)
        p.add_argument("--evalset", required=True)
        p.add_argument("--json", default=None, help="write the full report here")
        if name == "runs":
            p.add_argument("--run", action="append", required=True, help="a run's seed directory")
    a = ap.parse_args(argv)
    try:
        table = table_checks(a.tables, a.evalset)
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"cells in the index: {table['n_cells']}; scored cells: {len(table['scored_cells'])}")
    _print(table, "tables")
    report, ok = {"tables": table}, _passed(table)
    if a.cmd == "runs":
        report["runs"] = []
        for r in a.run:
            rr = run_checks(r, table)
            report["runs"].append(rr)
            _print(rr, r)
            ok = ok and _passed(rr)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
