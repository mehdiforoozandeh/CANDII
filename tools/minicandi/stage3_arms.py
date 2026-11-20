"""The stage-3 arm configs (chunk S3K; plan Amendment 3, section B, plus the foreman's arm
`loss_bin_equal`).

    python tools/minicandi/stage3_arms.py write --incumbent <design.json> --out <dir>
        [--rna-h5-dir D] [--expression-tsv F] [--tables T | --n-cells N]
    python tools/minicandi/stage3_arms.py evalset-subset --evalset <json> --cells <rna_cells.tsv>
        --out <json>

`write` puts `<arm>.json` in `--out` for every arm that applies to the incumbent (the design), and
`arms.tsv` in the columns of `stage2_arms.ARMS_TSV_COLUMNS`. Each arm is one change from the
incumbent and is compared with it. No arm is a D14 condition 1 comparison. `simpler_than_base` is
false for the three input arms (`rna_track`, `rna_vector`, `cell_embedding`) and null for the loss
arms and `aug_revcomp`.

An arm gets no json, and its `status` column gives the reason, when (1) building it through
`registry.build_model` raises `NotApplicable` (e.g. `rna_track` under `dna = off`), (2) the
incumbent already holds the arm's values, or (3) its key cannot act on the incumbent
(`loss_bin_equal` needs `loss = nb_nll`; `aug_revcomp` is refused with a frozen DNA embedding).

Input values. `needs` names what an arm reads: `rna_h5_dir` (rna_track), `expression_tsv`
(rna_vector; `genes.tsv` beside it) and `n_cells` (cell_embedding: the number of training cells of
the tables, `extra_data.cell_index`). A value given on the command line (or `n_cells` already in
the incumbent) is written into the arm's json; without it the json lacks the key and the submitter
must add it. The applicability check of `cell_embedding` builds with `n_cells = 1` when it is
unknown.

`joins_design` is `no` for the two RNA-seq arms (PI ruling 27) and for `cell_embedding` (it cannot
serve a cell type with no training track): `NEVER_IN_DESIGN`.

`evalset-subset` keeps, of an eval set built by `onion_eval/evalset.py`, the imputation targets,
control targets, correction pairs and replicate pairs whose cell is listed in the `cell` column of
`rna_cells.tsv` (the cells with RNA-seq; chunk XRf). Regions and the `scorable` lists are kept as
they are, so a subset score is comparable across the arms scored on it. The replicate pairs are
read from `pairs.tsv` of the eval set's `tables_dir`. The key `subset` records the source and the
counts.
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

from minicandi import stage2_arms as S2  # noqa: E402

ARMS_TSV_COLUMNS = S2.ARMS_TSV_COLUMNS
#: the per-sample dropout of the cell-state inputs (pinned interface E, `cell_state_dropout`)
CELL_STATE_DROPOUT = 0.1

#: ARMS[arm] = (group, {key: value}): the change of each arm from the incumbent, in table order
ARMS = {
    "rna_track": ("RNA-seq input (ruling 27)", {"rna_track": True}),
    "rna_vector": ("RNA-seq input (ruling 27)", {"cell_vector": True,
                                                 "cell_state_dropout": CELL_STATE_DROPOUT}),
    "cell_embedding": ("cell embedding", {"cell_embedding": True,
                                          "cell_state_dropout": CELL_STATE_DROPOUT}),
    "loss_shape_x5": ("loss shape", {"loss_shape_weight": 5}),
    "loss_assay_equal": ("assay weighting", {"loss_assay_weight": "equal"}),
    "loss_assay_learned": ("assay weighting", {"loss_assay_weight": "learned"}),
    "loss_multiscale": ("multi-scale loss", {"loss_multiscale": [4, 16]}),
    "loss_bin_equal": ("bin weighting", {"loss_bin_equal": True}),
    "aug_revcomp": ("augmentation", {"aug_revcomp": True}),
}
INPUT_ARMS = ("rna_track", "rna_vector", "cell_embedding")
#: simpler_than_base per arm: False for the input arms, None otherwise
SIMPLER = {a: (False if a in INPUT_ARMS else None) for a in ARMS}
#: arm -> the input key it needs
NEEDS = {"rna_track": "rna_h5_dir", "rna_vector": "expression_tsv", "cell_embedding": "n_cells"}
#: arms that never join a design, with the reason
NEVER_IN_DESIGN = {
    "rna_track": "PI ruling 27: RNA-seq input is an ablation only, reported in a side table",
    "rna_vector": "PI ruling 27: RNA-seq input is an ablation only, reported in a side table",
    "cell_embedding": "it cannot serve a cell type with no training track",
}
#: arms whose key acts only under an incumbent condition: arm -> (predicate, reason)
_ACTS_ONLY = {
    "loss_bin_equal": (lambda inc: inc.get("loss", "nb_nll") == "nb_nll",
                       "loss_bin_equal needs loss = nb_nll"),
    "aug_revcomp": (lambda inc: inc.get("dna") != "precomputed" and not inc.get("dna_embedding_h5"),
                    "aug_revcomp is refused with a frozen DNA embedding (dna = precomputed)"),
}


def arm_change(arm: str, values: dict | None = None) -> dict:
    """The declared keys of `arm`, with its input value when given. `values` maps `rna_h5_dir`,
    `expression_tsv`, `n_cells` to their values (None = not given)."""
    values = values or {}
    change = json.loads(json.dumps(ARMS[arm][1]))
    need = NEEDS.get(arm)
    if need and values.get(need) is not None:
        change[need] = int(values[need]) if need == "n_cells" else str(values[need])
    return change


def arm_config(incumbent: dict, arm: str, values: dict | None = None) -> dict:
    """The incumbent with the arm's change and its `simpler_than_base`."""
    return {**incumbent, **arm_change(arm, values), "simpler_than_base": SIMPLER[arm]}


def _build_check(cfg: dict) -> str | None:
    """`stage2_arms._build_check`; a `cell_embedding` config without `n_cells` is checked with
    `n_cells = 1` (applicability does not depend on the table size)."""
    if cfg.get("cell_embedding") and cfg.get("n_cells") is None:
        cfg = {**cfg, "n_cells": 1}
    return S2._build_check(cfg)


def status(incumbent: dict, arm: str, cfg: dict, build_check=_build_check) -> str:
    """`write`, or `not_applicable: <reason>`, or `no_change: <reason>`."""
    if arm in _ACTS_ONLY and not _ACTS_ONLY[arm][0](incumbent):
        return f"not_applicable: {_ACTS_ONLY[arm][1]}"
    declared = set(ARMS[arm][1])
    if not (declared & set(S2.changed_keys(incumbent, cfg))):
        return "no_change: the incumbent already holds " + ", ".join(
            f"{k} = {S2._json(cfg[k])}" for k in sorted(declared))
    why = build_check(cfg)
    if why is not None:
        return f"not_applicable: {why}"
    return "write"


def plan(incumbent: dict, values: dict | None = None, build_check=_build_check) -> list[dict]:
    """One row per arm (ARMS_TSV_COLUMNS) plus `config` (None when no json is written)."""
    values = dict(values or {})
    if values.get("n_cells") is None and incumbent.get("n_cells") is not None:
        values["n_cells"] = incumbent["n_cells"]
    rows = []
    for arm, (group, _) in ARMS.items():
        cfg = arm_config(incumbent, arm, values)
        st = status(incumbent, arm, cfg, build_check)
        need = NEEDS.get(arm, "")
        if need and need in cfg:
            need = f"{need}={cfg[need]}"
        rows.append({
            "arm": arm,
            "keys changed": ";".join(f"{k}={S2._cell(cfg[k])}"
                                     for k in S2.changed_keys(incumbent, cfg)),
            "comparison incumbent": "incumbent",
            "d14_condition_1": "false",
            "needs": need,
            "group": group,
            "simpler_than_base": S2._json(SIMPLER[arm]),
            "joins_design": ("no: " + NEVER_IN_DESIGN[arm]) if arm in NEVER_IN_DESIGN else "yes",
            "status": st,
            "config": cfg if st == "write" else None,
        })
    return rows


def write(incumbent: dict, out, values: dict | None = None, build_check=_build_check) -> list[dict]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = plan(incumbent, values, build_check)
    for r in rows:
        path = out / f"{r['arm']}.json"
        if r["config"] is not None:
            path.write_text(json.dumps(r["config"], indent=1) + "\n", encoding="utf-8")
        elif path.exists():
            path.unlink()
    lines = ["\t".join(ARMS_TSV_COLUMNS)]
    lines += ["\t".join(str(r[c]).replace("\t", " ").replace("\n", " ")
                        for c in ARMS_TSV_COLUMNS) for r in rows]
    (out / "arms.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


# --------------------------------------------------------------------------- eval-set subset


def _read_tsv(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def evalset_subset(evalset, cells_tsv, out) -> dict:
    """The eval set restricted to the cells of `cells_tsv` (module docstring); writes `out`."""
    evalset = Path(evalset)
    ev = json.loads(evalset.read_text(encoding="utf-8"))
    cells = sorted({r["cell"] for r in _read_tsv(cells_tsv) if r.get("cell")})
    if not cells:
        raise ValueError(f"{cells_tsv}: no cell listed in column `cell`")
    keep = set(cells)
    sub = dict(ev)
    counts = {}
    for key in ("imputation_targets", "control_targets", "correction_pairs"):
        items = ev.get(key, [])
        sub[key] = [t for t in items if t["cell"] in keep]
        counts[key] = [len(sub[key]), len(items)]
    kept_targets = {t["target"] for t in sub["imputation_targets"]}
    sub["targets_without_halves"] = [t for t in ev.get("targets_without_halves", [])
                                     if t in kept_targets]
    rep = ev.get("replicate_pairs")
    if rep:
        pairs = Path(ev["tables_dir"]) / "pairs.tsv"
        cell_of = {p["pair_id"]: p["cell"] for p in _read_tsv(pairs)}
        missing = [p for p in rep if p not in cell_of]
        if missing:
            raise ValueError(f"{pairs}: {len(missing)} replicate pairs of the eval set are absent "
                             f"(first: {missing[0]})")
        sub["replicate_pairs"] = [p for p in rep if cell_of[p] in keep]
        counts["replicate_pairs"] = [len(sub["replicate_pairs"]), len(rep)]
    in_tables = {t["cell"] for k in ("imputation_targets", "control_targets", "correction_pairs")
                 for t in ev.get(k, [])}
    sub["subset"] = {"parent_evalset": str(evalset.resolve()),
                     "cells_tsv": str(Path(cells_tsv).resolve()), "cells": cells,
                     "cells_not_in_evalset": sorted(keep - in_tables),
                     "kept_of_total": counts}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sub, indent=1), encoding="utf-8")
    return sub


# ----------------------------------------------------------------------------------- CLI


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--incumbent", required=True, help="the design config json")
    w.add_argument("--out", required=True)
    w.add_argument("--rna-h5-dir", default=None)
    w.add_argument("--expression-tsv", default=None)
    g = w.add_mutually_exclusive_group()
    g.add_argument("--tables", default=None, help="tables dir: n_cells = its training cells")
    g.add_argument("--n-cells", type=int, default=None)
    e = sub.add_parser("evalset-subset")
    e.add_argument("--evalset", required=True)
    e.add_argument("--cells", required=True, help="rna_cells.tsv (column `cell`)")
    e.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "evalset-subset":
        s = evalset_subset(args.evalset, args.cells, args.out)
        print(f"wrote {args.out}: " + "; ".join(
            f"{k} {a} of {b}" for k, (a, b) in s["subset"]["kept_of_total"].items()))
        if s["subset"]["cells_not_in_evalset"]:
            print("  cells with no target or pair in the eval set: "
                  + ", ".join(s["subset"]["cells_not_in_evalset"]))
        return 0
    inc = json.loads(Path(args.incumbent).read_text(encoding="utf-8"))
    n_cells = args.n_cells
    if args.tables:
        from minicandi.extra_data import cell_index
        n_cells = len(cell_index(args.tables))
    values = {"rna_h5_dir": args.rna_h5_dir, "expression_tsv": args.expression_tsv,
              "n_cells": n_cells}
    rows = write(inc, args.out, values)
    n = sum(r["config"] is not None for r in rows)
    print(f"wrote {n} arm configs and arms.tsv ({len(rows)} rows) under {args.out}")
    for r in rows:
        if r["config"] is None:
            print(f"  {r['arm']}: {r['status']}")
        elif r["arm"] in NEEDS and NEEDS[r["arm"]] not in r["config"]:
            print(f"  {r['arm']}: written without {NEEDS[r['arm']]}; the submitter must add it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
