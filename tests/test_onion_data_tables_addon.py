"""t152 chunk OXt (Amendment 4): the tables add-on of the evaluation-only second experiments, and
the dataset check that proves none can leak into training.

Tree: the tiny tree of `test_onion_data_build_tables.py` (its base tables are built with
`build_tables.py build`), plus second experiments built by `align_views.build` with role
`evalonly`, as the `run --row` path of the second-experiment build list writes them:
  SXA  other-lab second experiment of the validation track C01M01 (appended, split val)
  SXB  same-lab second experiment of the training track C02M01 (appended, split val)
  SXC  second experiment of the blind track C03M01, whose EIC partner is testbed only in the
       base (never a partner): it goes to `missing.tsv` of the add-on
  SXD  built and verified, but not in the reviewed list: it goes to `missing.tsv`
  SXA.depth0.50  a depth arm of SXA in the build list: not a pooled uniform base view, missing
  SXE  same-lab second experiment of the training track C01M02 built from ENCODE's redacted
       alignment (`$OXL` source `redacted_bam`, pipeline `native-redacted-bam`): its one view is
       `<x>.pooled.native.base` (appended, split val)
  SXF  the fallback of C02M01, in the build list at priority 6 and never built: missing

Digest of the `build` output. `BASE_DIGESTS` was written on the tree BEFORE this chunk's edits
(base commit 463dfb0, `python tests/test_onion_data_tables_addon.py --print`, on the laptop and
on the cluster). The digest covers every file `build` writes except `build_info.json` (which
holds the time and the paths), with the tree root (resolved, then as given) replaced by `<ROOT>`.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import shutil
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from onion_data import align_views as AV  # noqa: E402
from onion_data import build_list as BL  # noqa: E402
from onion_data import build_tables as BT  # noqa: E402
from onion_data import other_experiments as OX  # noqa: E402
from minicandi.dataset import (MEASUREMENT_COLUMNS, PAIR_COLUMNS, read_tsv,  # noqa: E402
                               tables_sha256)
from tests.test_onion_data_build_tables import (EXPS, TRN, TRN2, VAL, BLD, _ta,  # noqa: E402
                                                _write_tsv, build_argv, make_tree)

SXA, SXB, SXC, SXD, SXE, SXF = ("ENCSRSXA101", "ENCSRSXB102", "ENCSRSXC103", "ENCSRSXD104",
                                "ENCSRSXE105", "ENCSRSXF109")
#: second experiment -> (EIC experiment, lab, arms, layer)
SX_BUILT = {SXA: (VAL, "LabB", "base,depth0.50", "uniform"), SXB: (TRN, "LabA", "base", "uniform"),
            SXC: (BLD, "LabB", "base", "uniform"), SXD: (TRN, "LabC", "base", "uniform"),
            SXE: (TRN2, "LabA", "base,depth0.50", "native")}
LAYER = {x: v[3] for x, v in SX_BUILT.items()}
BUILD_FILES = ("measurements.tsv", "pairs.tsv", "pairs_shared_reads.tsv", "withheld.tsv",
               "covariates.json", "missing.tsv", "evaluation_only.tsv")

#: platform key -> digest of `build_digest` on the base commit 463dfb0
BASE_DIGESTS: dict[str, str] = {
    "Darwin-arm64": "a5897463e4340407b52805b6dd3657bfb8190e320a6b1cf37563428d299ed5b9",
    "Linux-x86_64": "a5897463e4340407b52805b6dd3657bfb8190e320a6b1cf37563428d299ed5b9",
}


def platform_key() -> str:
    return f"{platform.system()}-{platform.machine()}"


def build_digest(out: Path, root: Path) -> str:
    h = hashlib.sha256()
    for name in BUILD_FILES:
        text = (out / name).read_text(encoding="utf-8")
        text = text.replace(str(Path(root).resolve()), "<ROOT>").replace(str(root), "<ROOT>")
        h.update(name.encode() + b"\0" + text.encode() + b"\0")
    return h.hexdigest()


# ------------------------------------------------------------------------------- fixtures


def _sx_build(tmp: Path, exp: str, eic: str, lab: str, arms: str, layer: str, seed: int) -> None:
    assay, cell, track, *_ = EXPS[eic]
    src = _ta(tmp / "src" / f"{exp}_rep1.tagAlign.gz", 2800, seed)
    argv = ["build", "--experiment", exp, "--assay", assay, "--cell", cell, "--eic-track", track,
            "--role", "val", "--layer", layer, "--arms", arms, "--splitter", "python",
            "--work", str(tmp / "work"), "--out", str(tmp / "measurements"),
            "--chrom-sizes", str(tmp / "chrom_sizes.json"), "--tmp", str(tmp / "tmp"),
            "--read-length", "50", "--meta", f"lab={lab}", "--meta", "platform=Illumina",
            "--source", str(src)]
    orig = AV.build

    def patched(cfg):                       # the role of a second-experiment build-list row
        cfg["role"] = "evalonly"
        return orig(cfg)
    with mock.patch.object(AV, "build", patched):
        assert AV.main(argv) == 0


def _oxl_row(eic_exp: str, exp: str, lab: str, pair_class: str, tech_diff: str,
             fallback: str = "", fallback_class: str = "", source: str = "fastq") -> dict:
    assay, cell, track, role, *_ = EXPS[eic_exp]
    r = {c: "" for c in OX.OUT_COLUMNS}
    r.update({"eic_track": track, "split": role, "cell": f"cell_{cell}", "assay": assay,
              "assay_class": BT.C.assay_class(assay), "eic_experiment": eic_exp,
              "eic_lab": "LabA", "experiment": exp, "lab": lab, "pair_class": pair_class,
              "tech_diff": tech_diff, "fallback": fallback, "fallback_pair_class": fallback_class,
              "source": source})
    return r


def make_addon_tree(tmp: Path) -> dict:
    t = make_tree(tmp)
    root = t["root"]
    base = root / "tables" / "eic_v1"
    assert BT.main(build_argv(t, base)) == 0
    for i, (exp, (eic, lab, arms, layer)) in enumerate(sorted(SX_BUILT.items())):
        _sx_build(root, exp, eic, lab, arms, layer, 200 + i)
    oxl = [_oxl_row(VAL, SXA, "LabB", "experiment;lab", "read_length"),
           _oxl_row(TRN, SXB, "LabA", "experiment", "", SXF, "experiment;donor?"),
           _oxl_row(BLD, SXC, "LabB", "experiment;lab", "read_length"),
           _oxl_row(TRN2, SXE, "LabA", "experiment;donor?", "read_length",
                    source="redacted_bam")]
    _write_tsv(root / "other_experiments.tsv", OX.OUT_COLUMNS, oxl)
    bx = []
    toks = {SXA: [f"{SXA}.pooled.uniform.base", f"{SXA}.pooled.uniform.depth0.50"],
            SXB: [f"{SXB}.pooled.uniform.base"], SXC: [f"{SXC}.pooled.uniform.base"],
            SXD: [f"{SXD}.pooled.uniform.base"], SXF: [f"{SXF}.pooled.uniform.base"],
            SXE: [f"{SXE}.pooled.native.base"]}
    eic_of = {SXA: VAL, SXB: TRN, SXC: BLD, SXD: TRN, SXE: TRN2, SXF: TRN}
    for i, exp in enumerate(sorted(toks)):
        assay, _cell, track, *_ = EXPS[eic_of[exp]]
        fb = exp == SXF
        bx.append({"row": str((3001 if fb else 2001) + i), "priority": "6" if fb else "5",
                   "experiment": exp, "eic_track": track, "role": "evalonly", "assay": assay,
                   "assay_class": BT.C.assay_class(assay), "unit": "pooled", "arm": "base",
                   "pipeline": BL.NATIVE_PIPELINE if exp == SXE else "chip-seq-pipeline2 v2.2.2",
                   "keep_bam": "0",
                   "expected_measurements": ";".join(BL.token(m, BL.BASE_PRODUCTS)
                                                     for m in toks[exp])})
    _write_tsv(root / "other_build_list.tsv", BL.BUILD_LIST_COLUMNS, bx)
    t.update({"base": base, "oxl": root / "other_experiments.tsv",
              "bx": root / "other_build_list.tsv"})
    return t


def addon_argv(t: dict, out) -> list:
    return ["addon", "--base", str(t["base"]), "--other-build-list", str(t["bx"]),
            "--other-experiments", str(t["oxl"]), "--measurements", str(t["measurements"]),
            "--work", str(t["work"]), "--out", str(out)]


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    t = make_addon_tree(tmp_path_factory.mktemp("tree"))
    out = t["root"] / "tables" / "eic_v1x"
    assert BT.main(addon_argv(t, out)) == 0
    return t, out


def _rows(d):
    return {r["measurement_id"]: r for r in read_tsv(d / "measurements.tsv", MEASUREMENT_COLUMNS)}


# ------------------------------------------------------------------------------- the builder


def test_build_output_unchanged_against_base_commit(tree):
    t, _ = tree
    key = platform_key()
    if key not in BASE_DIGESTS:
        pytest.skip(f"no base digest recorded for {key}")
    assert build_digest(t["base"], t["root"]) == BASE_DIGESTS[key]


def test_addon_holds_the_base_byte_for_byte(tree):
    t, out = tree
    base = t["base"]
    for p in sorted(base.iterdir()):
        if p.name == "build_info.json":
            continue
        old, new = p.read_bytes(), (out / p.name).read_bytes()
        if p.name in BT.ADDON_APPENDED:
            assert new.startswith(old), p.name
        else:
            assert new == old, p.name
    info = json.loads((out / "build_info.json").read_text())
    assert info["addon_of"] == str(base.resolve())
    assert info["addon_of_tables_sha256"] == tables_sha256(base)
    assert info["other_experiments_sha256"] == hashlib.sha256(t["oxl"].read_bytes()).hexdigest()
    assert info["base_build_info"] == json.loads((base / "build_info.json").read_text())


def test_appended_rows_pairs_and_missing(tree):
    t, out = tree
    base_rows, rows = _rows(t["base"]), _rows(out)
    new = {m: r for m, r in rows.items() if m not in base_rows}
    assert sorted(new) == [f"{SXA}.pooled.uniform.base", f"{SXB}.pooled.uniform.base",
                           f"{SXE}.pooled.native.base"]
    a, b = new[f"{SXA}.pooled.uniform.base"], new[f"{SXB}.pooled.uniform.base"]
    assert a["in_eic"] == b["in_eic"] == "evalonly"
    assert (a["eic_track"], a["cell"], a["assay"]) == ("C01M01", "C01", "H3K27ac")
    assert (b["eic_track"], b["cell"], b["assay"]) == ("C02M01", "C02", "H3K27ac")
    assert a["control_measurement_id"] == "" and a["lab"] == "LabB"
    ev = {r["measurement_id"]: r["use"] for r in read_tsv(out / "evaluation_only.tsv")}
    assert ev[a["measurement_id"]] == ev[b["measurement_id"]] == BT.USE_OTHER_EXPERIMENT
    base_pairs = {p["pair_id"] for p in read_tsv(t["base"] / "pairs.tsv", PAIR_COLUMNS)}
    newp = [p for p in read_tsv(out / "pairs.tsv", PAIR_COLUMNS) if p["pair_id"] not in base_pairs]
    by_b = {p["b"]: p for p in newp}
    assert len(newp) == 3 and set(by_b) == set(new)
    pe = by_b[f"{SXE}.pooled.native.base"]
    assert (pe["a"], pe["knobs_differing"], pe["split"]) == (f"{TRN2}.pooled.uniform.base",
                                                             "experiment;donor?;read_length", "val")
    assert new[f"{SXE}.pooled.native.base"]["layer"] == "native"
    pa = by_b[a["measurement_id"]]
    assert (pa["a"], pa["rung"], pa["knobs_differing"], pa["split"], pa["subset_pair"],
            pa["never_trained"], pa["cell"]) == (f"{VAL}.pooled.uniform.base", "other_experiment",
                                                 "experiment;lab;read_length", "val", "0", "0",
                                                 "C01")
    pb = by_b[b["measurement_id"]]
    assert (pb["a"], pb["knobs_differing"], pb["split"]) == (f"{TRN}.pooled.uniform.base",
                                                             "experiment", "val")
    shared = read_tsv(out / "pairs_shared_reads.tsv", PAIR_COLUMNS)
    assert not any(p["a"].startswith("ENCSRSX") or p["b"].startswith("ENCSRSX") for p in shared)
    miss = {m["measurement_id"]: m["reason"] for m in read_tsv(out / "missing.tsv")}
    assert "EIC partner" in miss[f"{SXC}.pooled.uniform.base"]
    assert "reviewed list" in miss[f"{SXD}.pooled.uniform.base"]
    assert "pooled uniform base" in miss[f"{SXA}.pooled.uniform.depth0.50"]
    assert "not built" in miss[f"{SXF}.pooled.uniform.base"]


def test_addon_out_must_be_empty(tree):
    t, out = tree
    assert BT.main(addon_argv(t, out)) == 2


def test_build_still_sends_second_experiments_to_missing(tree, tmp_path):
    t, _ = tree
    # a build list that names a second experiment beside the EIC rows: `build` keeps it out
    rows = BL.read_tsv(t["build_list"]) + BL.read_tsv(t["bx"])
    bl = tmp_path / "bl.tsv"
    _write_tsv(bl, BL.BUILD_LIST_COLUMNS, [dict(r, priority="1") for r in rows])
    out = tmp_path / "tables"
    argv = build_argv(t, out, "1")
    argv[argv.index("--build-list") + 1] = str(bl)
    assert BT.main(argv) == 0
    assert not any(m.startswith("ENCSRSX") for m in _rows(out))
    miss = {m["measurement_id"]: m["reason"] for m in read_tsv(out / "missing.tsv")}
    assert "ledger" in miss[f"{SXA}.pooled.uniform.base"]


def test_dataset_reads_the_addon_only_for_evaluation(tree):
    from minicandi import dataset as D
    t, out = tree
    g = t["root"] / "genome"
    g.mkdir(exist_ok=True)
    (g / "chrom_sizes.json").write_text(json.dumps({"chr1": 409600, "chr21": 409600}))
    cfg = {"dna": "off", "genome_dir": str(g), "context_bins": 64}
    with pytest.raises(D.RoleError):
        D.OnionDataset(out, "train", cfg)
    ds = D.OnionDataset(out, "val", cfg)
    sx = {f"{SXA}.pooled.uniform.base", f"{SXB}.pooled.uniform.base", f"{SXE}.pooled.native.base"}
    assert not sx & {m for v in ds.train_tracks.values() for m in v}
    assert not sx & set(ds.train_base.values())


# ------------------------------------------------------------------------------- the check


def _check(t, tables, capsys, *extra):
    from onion_data import check_dataset as CD
    capsys.readouterr()
    code = CD.main(["--tables", str(tables), "--published", str(t["published"]), "--no-widths",
                    *extra])
    txt = capsys.readouterr().out
    counts = {m.group(1): int(m.group(2)) for m in re.finditer(r"^([^:\n]+): (\d+)\b", txt, re.M)}
    return code, counts, txt


NEW_RULES = ("second-experiment rows outside the reviewed list",
             "second-experiment rows not flagged evaluation-only",
             "second-experiment rows that are not a pooled uniform base view",
             "second-experiment rows in a training pair or outside the other_experiment rung",
             "evaluation-only rows readable in a training mode",
             "evalonly rows that disagree with evaluation_only.tsv",
             "rows of the base tables changed or missing")


def test_check_passes_on_the_addon(tree, capsys):
    t, out = tree
    code, c, txt = _check(t, out, capsys)
    assert code == 0, txt
    for k in NEW_RULES:
        assert c[k] == 0, k
    assert c["measurements outside ledger"] == 0
    assert c["accessions outside the published EIC list"] == 0
    assert "second-experiment rows: 3 (other lab 1, same lab 2)" in txt


def test_check_on_the_base_prints_the_new_rules_at_zero(tree, capsys):
    t, _ = tree
    code, c, txt = _check(t, t["base"], capsys)
    assert code == 0, txt
    for k in NEW_RULES:
        assert c[k] == 0, k
    assert "second-experiment rows: 0 (other lab 0, same lab 0)" in txt


def _plant(t, out, tmp_path, capsys, rule, plant, *extra):
    d = tmp_path / "tables" / "eic_v1x"
    shutil.copytree(out, d)
    plant(d)
    code, c, txt = _check(t, d, capsys, *extra)
    assert code == 1, txt
    assert c[rule] > 0, txt
    return c


def _edit(path, columns, fn):
    rows = fn(read_tsv(path, columns))
    _write_tsv(path, columns, rows)


def mid_of(x: str) -> str:
    return f"{x}.pooled.{LAYER[x]}.base"


EIC_OF = {SXA: (VAL, "C01", "H3K27ac"), SXE: (TRN2, "C01", "H3K4me3")}
SX_PARAM = pytest.mark.parametrize("sx", [SXA, SXE], ids=["uniform", "native"])


@SX_PARAM
def test_leak_evaluation_only_row_removed(tree, tmp_path, capsys, sx):
    t, out = tree
    SXA_MID = mid_of(sx)
    _plant(t, out, tmp_path, capsys, "evalonly rows that disagree with evaluation_only.tsv",
           lambda d: _edit(d / "evaluation_only.tsv", BT.EVAL_ONLY_COLUMNS,
                           lambda rows: [r for r in rows if r["measurement_id"] != SXA_MID]))


@SX_PARAM
def test_leak_in_eic_set_to_train(tree, tmp_path, capsys, sx):
    t, out = tree
    SXA_MID = mid_of(sx)

    def plant(d):
        def fn(rows):
            for r in rows:
                if r["measurement_id"] == SXA_MID:
                    r["in_eic"] = "train"
            return rows
        _edit(d / "measurements.tsv", MEASUREMENT_COLUMNS, fn)
    c = _plant(t, out, tmp_path, capsys, "second-experiment rows not flagged evaluation-only",
               plant)
    assert c["evaluation-only rows readable in a training mode"] > 0


@SX_PARAM
def test_leak_pair_with_split_train(tree, tmp_path, capsys, sx):
    t, out = tree
    SXA_MID = mid_of(sx)

    def plant(d):
        def fn(rows):
            for p in rows:
                if p["b"] == SXA_MID:
                    p["split"] = "train"
            return rows
        _edit(d / "pairs.tsv", PAIR_COLUMNS, fn)
    _plant(t, out, tmp_path, capsys,
           "second-experiment rows in a training pair or outside the other_experiment rung", plant)


@SX_PARAM
def test_leak_experiment_outside_the_reviewed_list(tree, tmp_path, capsys, sx):
    t, out = tree
    SXA_MID = mid_of(sx)
    oxl = tmp_path / "oxl.tsv"
    _write_tsv(oxl, OX.OUT_COLUMNS,
               [r for r in read_tsv(t["oxl"]) if r["experiment"] != sx])
    _plant(t, out, tmp_path, capsys, "second-experiment rows outside the reviewed list",
           lambda d: None, "--other-experiments", str(oxl))


@SX_PARAM
def test_leak_second_experiment_in_a_designed_pair(tree, tmp_path, capsys, sx):
    t, out = tree
    SXA_MID = mid_of(sx)

    def plant(d):
        eic, cell, assay = EIC_OF[sx]
        a = f"{eic}.pooled.uniform.base"
        _edit(d / "pairs.tsv", PAIR_COLUMNS, lambda rows: rows + [{
            "pair_id": BT.pair_id("designed", a, SXA_MID), "a": a, "b": SXA_MID,
            "rung": "designed", "assay": assay, "cell": cell, "knobs_differing": "",
            "subset_pair": "0", "split": "val", "never_trained": "0"}])
    _plant(t, out, tmp_path, capsys,
           "second-experiment rows in a training pair or outside the other_experiment rung", plant)


@SX_PARAM
def test_leak_second_experiment_depth_arm(tree, tmp_path, capsys, sx):
    t, out = tree
    SXA_MID = mid_of(sx)
    arm = f"{sx}.pooled.{LAYER[sx]}.depth0.50"

    def plant(d):
        m = json.loads((t["measurements"] / arm / "meta.json").read_text())
        row = {k: str(m.get(k, "")) for k in MEASUREMENT_COLUMNS}
        row.update({"in_eic": "evalonly", "cell": EIC_OF[sx][1], "assay": EIC_OF[sx][2],
                    "frip": ""})
        _edit(d / "measurements.tsv", MEASUREMENT_COLUMNS, lambda rows: rows + [row])
        _edit(d / "evaluation_only.tsv", BT.EVAL_ONLY_COLUMNS, lambda rows: rows + [
            {"measurement_id": arm, "use": BT.USE_OTHER_EXPERIMENT}])
    _plant(t, out, tmp_path, capsys,
           "second-experiment rows that are not a pooled uniform base view", plant)


@SX_PARAM
def test_leak_view_in_the_other_layer(tree, tmp_path, capsys, sx):
    """A native view is allowed only for a redacted-alignment second experiment, and a uniform
    view is not allowed for one."""
    t, out = tree
    SXA_MID = mid_of(sx)
    other = "uniform" if LAYER[sx] == "native" else "native"
    moved = f"{sx}.pooled.{other}.base"

    def plant(d):
        def fn(rows):
            for r in rows:
                if r["measurement_id"] == SXA_MID:
                    r["measurement_id"], r["layer"] = moved, other
            return rows
        _edit(d / "measurements.tsv", MEASUREMENT_COLUMNS, fn)
        _edit(d / "evaluation_only.tsv", BT.EVAL_ONLY_COLUMNS, lambda rows: [
            dict(r, measurement_id=moved) if r["measurement_id"] == SXA_MID else r for r in rows])
        _edit(d / "pairs.tsv", PAIR_COLUMNS, lambda rows: [
            dict(p, b=moved) if p["b"] == SXA_MID else p for p in rows])
    _plant(t, out, tmp_path, capsys,
           "second-experiment rows that are not a pooled uniform base view", plant)


def test_base_row_changed_in_the_addon(tree, tmp_path, capsys):
    t, out = tree

    def plant(d):
        lines = (d / "pairs.tsv").read_text().splitlines(keepends=True)
        del lines[1]
        (d / "pairs.tsv").write_text("".join(lines))
    _plant(t, out, tmp_path, capsys, "rows of the base tables changed or missing", plant)


if __name__ == "__main__" and "--print" in sys.argv:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tt = make_tree(Path(td))
        o = tt["root"] / "tables" / "v1"
        assert BT.main(build_argv(tt, o)) == 0
        print(platform_key(), build_digest(o, tt["root"]))
