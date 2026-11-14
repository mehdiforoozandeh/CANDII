"""STt2: `ccre_density`, the power tilt and the tercile classes (tools/onion_data/train_strata.py;
plan Amendment 7, amended section B.4). Synthetic worlds from test_onion_data_train_strata."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "tools", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from onion_data import train_strata as S  # noqa: E402
import test_onion_data_train_strata as T  # noqa: E402

#: digests of the synthetic world's tile outputs made by the code BEFORE STt2 (commit 20d026a),
#: with the `# sha256_inputs` line dropped (it hashes tmp paths) and, for the tile table, the
#: new `ccre_density` column dropped
GOLDEN = {
    "tile_table": "ca9201b74b32da5804842f38dffb48773d6970fbcc83a6787fc2c6ce4f910ad2",
    "tile_uniform": "be84d7710d7215cf5c0c488167773610fb48cd720e701211d713197c9d3baf26",
    "peakfreq_w": "d1818a646e01558fb2c7feb50167d94c81d5c5426fe0abd257e0bccdf3b15e26",
    "peakfreq_u": "524540b3c52b1793b2c149d43ac20602b84ce007531a8ba94b7e299254876d69",
    "ccre_like_w": "558d9e4e9b5224d157d4c48dcc0b3feeda69f34f53c07c32a4d217fa8a2a216c",
    "variance_w": "87725d341164064027d8673218a550fb2a4febcb99938bdff3c7b8dfe26464c7",
    "kcenter": "1a9ecc4ce2b0e3af284f7a2f877f324259b1737f9d010c6671f6799d62c10d82",
}


def _digest(path, drop_col=None):
    lines = [x for x in Path(path).read_text().splitlines() if not x.startswith("# sha256_inputs=")]
    if drop_col:
        h = [i for i, x in enumerate(lines) if not x.startswith("#")][0]
        cols = lines[h].split("\t")
        if drop_col in cols:
            k = cols.index(drop_col)
            lines = lines[:h] + ["\t".join(c for j, c in enumerate(x.split("\t")) if j != k)
                                 for x in lines[h:]]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("w")
    w = T.make_world(root)
    S.run_tiles(w["tables"], w["peaks"], w["peak_files"], w["average"], w["embedding"],
                w["genome"], w["evalset"], root / "o", k=6)
    w["out"] = root / "o"
    return w


def test_existing_lists_unchanged(world):
    for name, d in GOLDEN.items():
        assert _digest(world["out"] / f"{name}.tsv", "ccre_density") == d, name


def test_ccre_density_column(world):
    tt = T._tile_table(world["out"] / "tile_table.tsv")
    assert list(tt[0])[-1] == "ccre_density"
    by = {(r["chrom"], int(r["start_bin"])): float(r["ccre_density"]) for r in tt}
    # every training track peaks over bins 4..23 of PEAK_TILE (20 bins); 9 tracks, all of the
    # four assays (DNase-seq, H3K4me3, H3K27ac); the validation DNase-seq track is not read
    assert by[T.PEAK_TILE] == pytest.approx(20 / 2048)
    assert sum(v > 0 for v in by.values()) == 1


def test_region_terciles(world, tmp_path):
    out = tmp_path / "terc.tsv"
    rows = S.run_region_classes(world["out"] / "tile_table.tsv", world["evalset"], world["tables"],
                                world["peaks"], world["peak_files"], out, by="ccre_density_tercile",
                                genome=world["genome"])
    by = {(r["chrom"], r["start_bin"]): r for r in rows}
    assert by[T.EVAL[0]]["class"] == "t1_low" and by[T.EVAL[1]]["class"] == "t2_mid"
    # chr2 eval region: tracks k % 3 == 0 (3 of 9, all DNase-seq) peak over 12 bins (300 bp from a bin edge) of 16,384
    assert by[T.EVAL[1]]["ccre_density"] == pytest.approx(3 / 9 * 12 / 16384)
    assert out.read_text().splitlines()[0] == "chrom\tstart_bin\tclass\tccre_density"


def _table(path, n=600, seed=0, roles=True):
    rng = np.random.default_rng(seed)
    v = rng.integers(1850, 2049, n)
    s = rng.lognormal(-4, 1.2, n)
    s[:7] = 0.0                                          # tiles with no peak bin
    var = rng.gamma(2.0, 1.0, n)
    with open(path, "w") as fh:
        if roles:
            fh.write("# roles_read=train\n")
        fh.write("# sha256_inputs=" + "0" * 64 + "\n")
        fh.write("\t".join(S.TILE_COLUMNS) + "\n")
        for i in range(n):
            fh.write(f"chr1\t{i * 2048}\t{v[i]}\t0.5\tnone\t{var[i]}\t{s[i]}\n")
    return path


@pytest.mark.parametrize("score", ["ccre_density", "variance"])
@pytest.mark.parametrize("target", [0.5, 0.25])
def test_tilt_hits_target(tmp_path, score, target):
    tt = _table(tmp_path / "tile_table.tsv")
    res = S.run_tilt(tt, score, target, "x", tmp_path)
    assert abs(res["ess"] - target) <= 0.01
    head, rows = S.read_list(tmp_path / "x.tsv")
    assert head["roles_read"] == "train"
    q = np.array([r["sample_weight"] for r in rows])
    w = np.array([r["loss_weight"] for r in rows])
    assert (q > 0).all() and len(rows) == 600
    assert (q * w).sum() == pytest.approx(1.0, abs=1e-12) and q.sum() == pytest.approx(1.0, abs=1e-12)
    assert 1.0 / (q * w * w).sum() == pytest.approx(res["ess"], abs=1e-9)
    assert {r["stratum"] for r in rows} == {f"d{i:02d}" for i in range(1, 11)}
    md = (tmp_path / "STRATA.md").read_text()
    assert "## x (tilt)" in md and "alpha found" in md and "realized ESS" in md and "floor" in md


def test_unweighted_same_q_and_alpha_order(tmp_path):
    tt = _table(tmp_path / "tile_table.tsv")
    a50 = S.run_tilt(tt, "ccre_density", 0.5, "w50", tmp_path)
    S.run_tilt(tt, "ccre_density", 0.5, "u50", tmp_path, unweighted=True)
    a25 = S.run_tilt(tt, "ccre_density", 0.25, "w25", tmp_path)
    _, w = S.read_list(tmp_path / "w50.tsv")
    _, u = S.read_list(tmp_path / "u50.tsv")
    assert [r["sample_weight"] for r in w] == [r["sample_weight"] for r in u]
    assert all(r["loss_weight"] == 1.0 for r in u)
    assert a25["alpha"] > a50["alpha"] > 0                # a lower target needs a stronger tilt
    sc = np.array([float(r["ccre_density"]) for r in T._tile_table(tt)])
    _, w25 = S.read_list(tmp_path / "w25.tsv")
    top = sc >= np.quantile(sc, 0.9)
    q50 = np.array([r["sample_weight"] for r in w])
    q25 = np.array([r["sample_weight"] for r in w25])
    assert q25[top].sum() > q50[top].sum() > 0.1          # the dose: more mass on dense tiles


def test_alpha_zero_is_tile_uniform(tmp_path):
    tt = _table(tmp_path / "tile_table.tsv")
    res = S.run_tilt(tt, "variance", 1.0, "a0", tmp_path)
    assert res["alpha"] == 0.0 and res["ess"] == pytest.approx(1.0)
    _, rows = S.read_list(tmp_path / "a0.tsv")
    trows = T._tile_table(tt)
    units = [(r["chrom"], int(r["start_bin"])) for r in trows]
    v = np.array([int(r["valid_bins"]) for r in trows])
    uni = S.strata_rows(units, v, ["tile"] * len(units), {"tile": 1.0}, {"tile": 1.0}, 2048)
    assert [r["sample_weight"] for r in rows] == pytest.approx([r["sample_weight"] for r in uni],
                                                               rel=1e-12)
    assert all(r["loss_weight"] == pytest.approx(1.0) for r in rows)


def test_tilt_deterministic(tmp_path):
    tt = _table(tmp_path / "tile_table.tsv")
    for d in ("a", "b"):
        S.run_tilt(tt, "ccre_density", 0.25, "x", tmp_path / d)
    assert (tmp_path / "a" / "x.tsv").read_bytes() == (tmp_path / "b" / "x.tsv").read_bytes()


def test_tilt_refusals(tmp_path):
    tt = _table(tmp_path / "tile_table.tsv")
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            S.run_tilt(tt, "ccre_density", bad, "x", tmp_path)
    with pytest.raises(SystemExit):
        S.main(["tilt", "--tile-table", str(tt), "--score", "peak_fraction", "--target-ess", "0.5",
                "--name", "x", "--out", str(tmp_path)])
    nr = _table(tmp_path / "noroles.tsv", roles=False)
    with pytest.raises(S.LeakError):
        S.run_tilt(nr, "ccre_density", 0.5, "x", tmp_path)


def test_leak_plants_hold(tmp_path, monkeypatch):
    """The validation counts file is never opened and a validation peak file listed as train
    raises, with the ccre_density step in place."""
    import csv
    import h5py
    w = T.make_world(tmp_path)
    val = [r for r in w["rows"] if r["in_eic"] == "val"][0]
    opened, real = [], h5py.File

    def spy(name, *a, **k):
        opened.append(str(name))
        return real(name, *a, **k)
    monkeypatch.setattr(h5py, "File", spy)
    S.run_tiles(w["tables"], w["peaks"], w["peak_files"], w["average"], w["embedding"],
                w["genome"], w["evalset"], tmp_path / "o", k=6)
    assert val["counts_path"][3:] not in opened
    pf = list(csv.DictReader(open(w["peak_files"]), delimiter="\t"))
    [r for r in pf if r["role"] == "val"][0]["role"] = "train"
    T._write_tsv(w["peak_files"], pf)
    with pytest.raises(S.LeakError):
        S.run_tiles(w["tables"], w["peaks"], w["peak_files"], w["average"], w["embedding"],
                    w["genome"], w["evalset"], tmp_path / "o2", k=6)


def test_cli_tilt(tmp_path, capsys):
    tt = _table(tmp_path / "tile_table.tsv")
    assert S.main(["tilt", "--tile-table", str(tt), "--score", "ccre_density", "--target-ess",
                   "0.5", "--name", "ccredens_e50_w", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "ccredens_e50_w.tsv").is_file()
