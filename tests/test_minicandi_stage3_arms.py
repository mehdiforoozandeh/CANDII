"""Stage-3 arm configs (chunk S3K; plan Amendment 3, section B, and the arm `loss_bin_equal`).

(a) The writer: 9 arms, each one change from the incumbent, with `arms.tsv` in the columns of
    `stage2_arms.ARMS_TSV_COLUMNS`, the pinned `simpler_than_base`, the arms that never join a
    design, input values written into the json, and no json for an arm that does not apply.
(b) Integration, on `synth.py` tables with synthetic RNA-seq files (the writer of chunk XRd) and a
    64-bin context: on rungs 2 and 5 every arm is written, built through the registry by
    `train.train` and trained 2 steps with finite losses.
(c) `evalset-subset` keeps the targets and pairs of the listed cells.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import contract as C  # noqa: E402
from minicandi import registry as R  # noqa: E402
from minicandi import stage2_arms as S2  # noqa: E402
from minicandi import stage3_arms as S3  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi import train as T  # noqa: E402
from minicandi.extra_data import cell_index  # noqa: E402
from onion_data import rna_inputs as RI  # noqa: E402

L = 64
BS = 2
STEPS = 2
N_GENES_TEST = 5
SMALL = {2: {"param_budget": 1e5}, 5: {"param_budget": 1e5}}
RUNGS = (2, 5)
CASES = [(r, a) for r in RUNGS for a in S3.ARMS]
PLAN_ARMS = {"rna_track", "rna_vector", "cell_embedding", "loss_shape_x5", "loss_assay_equal",
             "loss_assay_learned", "loss_multiscale", "aug_revcomp"}


def _incumbent(r: int, **over) -> dict:
    cfg = json.loads((_ROOT / f"tools/minicandi/configs/arms/rung{r}/base.json").read_text())
    cfg.update(context_bins=L, batch_size=BS, windows_seen=STEPS * BS, n_genes=N_GENES_TEST,
               **SMALL[r])
    cfg.update(over)
    return cfg


def _tsv(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:]]


def _no_build(cfg):
    return None


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Synthetic tables with an eval set, and RNA-seq tracks and expression for every cell but
    the last (so the unavailable path is exercised)."""
    root = tmp_path_factory.mktemp("stage3")
    td = synth.make_synthetic_tables(root)
    sizes = {c: int(n) // C.RES for c, n in
             json.loads((root / "genome" / "chrom_sizes.json").read_text()).items()}
    cells = sorted({r["cell"] for r in RI.read_tsv(td / "measurements.tsv")})
    with_rna = cells[:-1]
    rng = np.random.default_rng(7)
    tdir = root / "tracks"
    for c in with_rna:
        plus, _ = RI.strand_arrays({ch: rng.gamma(1.0, 3.0, n) for ch, n in sizes.items()}, 1.0)
        minus, _ = RI.strand_arrays({ch: -rng.gamma(1.0, 2.0, n) for ch, n in sizes.items()}, 1.0)
        RI.write_track(tdir / f"{c}.h5", c, plus, minus, {"stranded": 1})
    genes = [f"ENSG{i:011d}" for i in range(12)]
    gt = root / "gene_types.tsv"
    gt.write_text("gene_id\tgene_type\n" + "".join(
        f"{g}.3\t{'protein_coding' if i < 10 else 'lncRNA'}\n" for i, g in enumerate(genes)))
    files, rows = root / "files", []
    for k, c in enumerate(with_rna):
        exp, acc = f"ENCSRX{k:05d}", f"ENCFFX{k:05d}"
        (files / exp).mkdir(parents=True)
        (files / exp / f"{acc}.tsv").write_text("gene_id\tTPM\n" + "".join(
            f"{g}.1\t{rng.gamma(1.0, 10.0):.4f}\n" for g in genes))
        rows.append({"cell": c, "experiment": exp, "gene_quant": acc,
                     "gene_quant_url": f"https://x/{acc}.tsv", "gencode": "V29"})
    adir = root / "aux"
    RI.build_vector(rows, gt, adir, files, n=N_GENES_TEST)
    values = {"rna_h5_dir": str(tdir), "expression_tsv": str(adir / "expression.tsv"),
              "n_cells": len(cell_index(td))}
    return {"root": root, "tables": td, "values": values, "cells": cells, "with_rna": with_rna,
            "written": {}}


def _written(env, r: int) -> dict:
    if r not in env["written"]:
        out = env["root"] / f"arms_rung{r}"
        S3.write(_incumbent(r), out, env["values"])
        env["written"][r] = {row["arm"]: row for row in _tsv(out / "arms.tsv")}
        env["written"][r]["_dir"] = out
    return env["written"][r]


# ------------------------------------------------------------------------------ (a) the writer


def test_the_table_holds_the_plan_arms_and_loss_bin_equal():
    assert set(S3.ARMS) == PLAN_ARMS | {"loss_bin_equal"}
    assert S3.ARMS_TSV_COLUMNS is S2.ARMS_TSV_COLUMNS
    assert S3.ARMS["loss_shape_x5"][1] == {"loss_shape_weight": 5}
    assert S3.ARMS["loss_multiscale"][1] == {"loss_multiscale": [4, 16]}
    assert S3.ARMS["rna_vector"][1]["cell_vector"] is True


def test_simpler_design_flags_and_needs():
    for arm in S3.ARMS:
        want = False if arm in ("rna_track", "rna_vector", "cell_embedding") else None
        assert S3.SIMPLER[arm] is want, arm
    assert set(S3.NEVER_IN_DESIGN) == {"rna_track", "rna_vector", "cell_embedding"}
    rows = {r["arm"]: r for r in S3.plan(_incumbent(2), build_check=_no_build)}
    assert {a for a, r in rows.items() if r["joins_design"] != "yes"} == set(S3.NEVER_IN_DESIGN)
    assert "ruling 27" in rows["rna_track"]["joins_design"]
    assert all(r["d14_condition_1"] == "false" and r["comparison incumbent"] == "incumbent"
               for r in rows.values())
    assert {a: r["needs"] for a, r in rows.items() if r["needs"]} == {
        "rna_track": "rna_h5_dir", "rna_vector": "expression_tsv", "cell_embedding": "n_cells"}
    assert rows["rna_track"]["simpler_than_base"] == "false"
    assert rows["aug_revcomp"]["simpler_than_base"] == "null"


@pytest.mark.parametrize("r", RUNGS)
def test_each_json_is_one_change_from_the_incumbent(env, r):
    inc = _incumbent(r)
    rows = _written(env, r)
    out = rows["_dir"]
    assert sorted(p.stem for p in out.glob("*.json")) == sorted(S3.ARMS)
    assert list(_tsv(out / "arms.tsv")[0]) == list(S2.ARMS_TSV_COLUMNS)
    extra = {"rna_track": {"rna_h5_dir"}, "rna_vector": {"expression_tsv"},
             "cell_embedding": {"n_cells"}}
    for arm, (_, change) in S3.ARMS.items():
        assert rows[arm]["status"] == "write", (r, arm, rows[arm]["status"])
        cfg = json.loads((out / f"{arm}.json").read_text())
        diff = {k for k in set(cfg) | set(inc) if cfg.get(k) != inc.get(k)} - {"simpler_than_base"}
        assert diff == {k for k, v in change.items() if inc.get(k) != v} | extra.get(arm, set())
        assert cfg["simpler_than_base"] is S3.SIMPLER[arm]
    v = env["values"]
    assert json.loads((out / "rna_track.json").read_text())["rna_h5_dir"] == v["rna_h5_dir"]
    assert rows["rna_vector"]["needs"] == f"expression_tsv={v['expression_tsv']}"
    assert json.loads((out / "cell_embedding.json").read_text())["n_cells"] == v["n_cells"]


def test_not_applicable_and_no_change(tmp_path):
    rows = {r["arm"]: r for r in S3.write(
        _incumbent(2, loss="nb_crps", loss_assay_weight="equal"), tmp_path / "a")}
    assert rows["loss_bin_equal"]["status"] == "not_applicable: loss_bin_equal needs loss = nb_nll"
    assert rows["loss_assay_equal"]["status"].startswith("no_change")
    assert rows["loss_assay_learned"]["status"] == "write"
    assert not (tmp_path / "a" / "loss_bin_equal.json").exists()
    rows = {r["arm"]: r for r in S3.plan(
        _incumbent(2, dna="precomputed", dna_embedding_dim=8), build_check=_no_build)}
    assert rows["aug_revcomp"]["status"].startswith("not_applicable: aug_revcomp is refused")
    # rna_track adds to the DNA features: NotApplicable under dna = off (chunk XI)
    rows = {r["arm"]: r for r in S3.write(_incumbent(2, dna="off"), tmp_path / "b")}
    assert rows["rna_track"]["status"].startswith("not_applicable")
    assert not (tmp_path / "b" / "rna_track.json").exists()


def test_without_values_the_json_lacks_the_input_key(tmp_path, capsys):
    inc = tmp_path / "design.json"
    inc.write_text(json.dumps(_incumbent(2)))
    assert S3.main(["write", "--incumbent", str(inc), "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert f"wrote {len(S3.ARMS)} arm configs" in out
    assert "cell_embedding: written without n_cells" in out
    ce = json.loads((tmp_path / "o" / "cell_embedding.json").read_text())
    assert "n_cells" not in ce and ce["cell_embedding"] is True
    assert "rna_h5_dir" not in json.loads((tmp_path / "o" / "rna_track.json").read_text())
    assert [r["arm"] for r in _tsv(tmp_path / "o" / "arms.tsv")] == list(S3.ARMS)


def test_cli_tables_fills_n_cells(env, tmp_path):
    inc = tmp_path / "design.json"
    inc.write_text(json.dumps(_incumbent(5)))
    v = env["values"]
    assert S3.main(["write", "--incumbent", str(inc), "--out", str(tmp_path / "o"),
                    "--rna-h5-dir", v["rna_h5_dir"], "--expression-tsv", v["expression_tsv"],
                    "--tables", str(env["tables"])]) == 0
    assert json.loads((tmp_path / "o" / "cell_embedding.json").read_text())["n_cells"] == \
        v["n_cells"]


# ------------------------------------------------------------------------- (b) integration


@pytest.mark.parametrize("r,arm", CASES, ids=[f"rung{r}-{a}" for r, a in CASES])
def test_arm_builds_and_trains_two_steps(env, r, arm):
    rows = _written(env, r)
    assert rows[arm]["status"] == "write", rows[arm]["status"]
    cfg = json.loads((rows["_dir"] / f"{arm}.json").read_text())
    run = env["root"] / "runs" / f"rung{r}_{arm}"
    torch.manual_seed(0)
    info = T.train(cfg, env["tables"], 0, run, log_every=1)
    assert info["windows_seen"] >= STEPS * BS
    losses = [json.loads(x)["loss"] for x in (run / "log.jsonl").read_text().splitlines()]
    assert len(losses) == STEPS and all(math.isfinite(x) for x in losses), losses
    model, _ = R.load_run(run)
    assert ("assay_log_w" in model.state_dict()) == (arm == "loss_assay_learned")


# ------------------------------------------------------------------------- (c) evalset-subset


def test_evalset_subset_keeps_the_listed_cells(env, tmp_path, capsys):
    from onion_eval import evalset as E
    ev_path = env["root"] / "eval" / "s3k" / "evalset.json"
    ev = E.build(env["tables"], ev_path, n_regions=2, n_fit_regions=1, region_bins=256, seed=0)
    keep = env["with_rna"][:2]
    cells = tmp_path / "rna_cells.tsv"
    cells.write_text("cell\tcell_id\texperiment\n" + "".join(
        f"{c}\tX\tENCSR{i}\n" for i, c in enumerate(keep)))
    out = tmp_path / "rna_cells" / "evalset.json"
    assert S3.main(["evalset-subset", "--evalset", str(ev_path), "--cells", str(cells),
                    "--out", str(out)]) == 0
    assert "wrote" in capsys.readouterr().out
    sub = json.loads(out.read_text())
    for key in ("imputation_targets", "control_targets", "correction_pairs"):
        want = [t for t in ev[key] if t["cell"] in keep]
        assert sub[key] == want, key
    assert ev["imputation_targets"] and sub["imputation_targets"]
    assert len(sub["imputation_targets"]) < len(ev["imputation_targets"])
    pc = {p["pair_id"]: p["cell"] for p in RI.read_tsv(env["tables"] / "pairs.tsv")}
    assert sub["replicate_pairs"] == [p for p in ev["replicate_pairs"] if pc[p] in keep]
    for k in ("regions", "fit_regions", "region_bins", "scorable", "tables_sha256"):
        assert sub[k] == ev[k], k
    assert sub["subset"]["cells"] == sorted(keep)


def test_evalset_subset_filters_replicate_pairs_by_cell(tmp_path):
    """The synthetic tables carry no replicate pair, so this path is checked on a crafted set."""
    td = tmp_path / "tables"
    td.mkdir()
    (td / "pairs.tsv").write_text("pair_id\tcell\trung\np1\tA\tisogenic\np2\tB\tisogenic\n"
                                  "p3\tA\tpseudoreplicate\n")
    ev = {"tables_dir": str(td), "regions": [["chr1", 0]], "scorable": {},
          "imputation_targets": [{"target": "m1", "cell": "A"}, {"target": "m2", "cell": "B"}],
          "control_targets": [], "correction_pairs": [{"source": "m3", "target": "m4", "cell": "B"}],
          "targets_without_halves": ["m1", "m2"], "replicate_pairs": ["p1", "p2", "p3"]}
    src = tmp_path / "evalset.json"
    src.write_text(json.dumps(ev))
    cells = tmp_path / "rna_cells.tsv"
    cells.write_text("cell\tcell_id\nA\tx\nZ\ty\n")
    sub = S3.evalset_subset(src, cells, tmp_path / "o" / "evalset.json")
    assert sub["replicate_pairs"] == ["p1", "p3"]
    assert [t["target"] for t in sub["imputation_targets"]] == ["m1"]
    assert sub["correction_pairs"] == [] and sub["targets_without_halves"] == ["m1"]
    assert sub["subset"]["cells_not_in_evalset"] == ["Z"]
    assert sub["subset"]["kept_of_total"]["replicate_pairs"] == [2, 3]
    ev["replicate_pairs"].append("p9")
    src.write_text(json.dumps(ev))
    with pytest.raises(ValueError, match="absent"):
        S3.evalset_subset(src, cells, tmp_path / "o2.json")
