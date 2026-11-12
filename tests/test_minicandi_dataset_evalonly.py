"""miniCANDI OnionDataset and evaluation-only second experiments (Amendment 4, chunk OXd).

A second experiment of an EIC track (PI ruling 33) sits in add-on tables with
`in_eic = evalonly`. The fixture adds one such row beside an EIC track of the `synth.py` tables.
Its measurement id sorts first (`ENCSR000000...`), so a base-view choice that did not exclude it
would pick it. The tests pin: mode `train` refuses the add-on tables in the constructor (three
configs), also when only `build_info.json::addon_of` marks them; in modes `val` and `blind` the
base view of every other row, `train_base`, every group and the position average are the same as
without the row; the second experiment as a target reads its own counts and is its own base
view; it is never an input; and no file under `tools/minicandi/` or `slurm/minicandi/` names an
add-on table set.

Digest of tables without the row. `BASE_DIGESTS` was written on the tree BEFORE this chunk's
edits (base commit 5bfad89, `python tests/test_minicandi_dataset_evalonly.py --print`, once on
the laptop and once on the cluster). `test_tables_without_the_row_are_bit_identical` demands the
same digest now. Float columns depend on the platform, so the dict holds one entry per platform.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import contract as C  # noqa: E402
from minicandi import dataset as D  # noqa: E402
from minicandi import synth  # noqa: E402

L = 64
X_MID = "ENCSR000000.pooled.uniform.base"
#: (cell index, assay) of the EIC track that gets a second experiment: a validation track and
#: a training track of `synth.VAL_TRACKS` / the default tables
TRACKS = ((0, "H3K4me3"), (0, "DNase-seq"))
TRAIN_CONFIGS = ({}, {"final_model": True}, {"train_on_validation": True, "final_model": True})

#: platform key -> digest of `plain_digest` on the base commit
BASE_DIGESTS: dict[str, str] = {
    "Darwin-arm64-torch2.6.0": "b747536caeea3c19aa0b188b78a54b376e3083b07a41cea38745720f479450ef",
    "Linux-x86_64-torch2.6.0": "b747536caeea3c19aa0b188b78a54b376e3083b07a41cea38745720f479450ef",
}


def platform_key() -> str:
    return f"{platform.system()}-{platform.machine()}-torch{torch.__version__}"


# ------------------------------------------------------------------------------- fixtures


def _read_rows(tdir: Path) -> list[dict]:
    with open(tdir / "measurements.tsv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def add_second_experiment(tdir: Path, cell_i: int, assay: str, seed: int = 7) -> dict:
    """Append one `in_eic = evalonly` row beside the EIC track (cell, assay), with its own
    counts (another NB draw at about twice the depth). Returns the row."""
    rows = _read_rows(tdir)
    cell = f"s{cell_i}"
    eic = next(r for r in rows if r["cell"] == cell and r["assay"] == assay
               and not r["arm_knob"])
    rng = np.random.default_rng(seed)
    root = tdir.parent.parent
    mdir = root / "measurements" / X_MID
    counts = {}
    with h5py.File(Path(eic["counts_path"][3:]), "r") as f:
        for chrom in f:
            v = f[chrom][:].astype(np.float64)
            counts[chrom] = rng.poisson(2.0 * v + 0.5).astype(np.uint32)
    depth = 2.0 * float(eic["depth_reads"])
    synth._write_counts(mdir / "counts.h5", counts, X_MID, depth)
    pr1 = {c: rng.binomial(v, 0.5) for c, v in counts.items()}
    synth._write_counts(mdir / "pr1.h5", pr1, X_MID, depth / 2)
    synth._write_counts(mdir / "pr2.h5", {c: counts[c] - pr1[c] for c in counts}, X_MID,
                        depth / 2)
    x = dict(eic)
    x.update({"measurement_id": X_MID, "experiment": "ENCSR000000", "in_eic": "evalonly",
              "lab": "labB" if eic["lab"] == "labA" else "labA",
              "depth_reads": f"{depth:.0f}", "depth_fragments": f"{depth / 2:.0f}",
              "counts_path": f"h5:{(mdir / 'counts.h5').resolve()}"})
    with open(tdir / "measurements.tsv", "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(D.MEASUREMENT_COLUMNS), delimiter="\t",
                           lineterminator="\n")
        w.writerow(x)
    return x


@pytest.fixture(scope="module")
def plain(tmp_path_factory) -> Path:
    td = synth.make_synthetic_tables(tmp_path_factory.mktemp("plain"))
    synth.make_evalset(td, n_regions=2, region_bins=4 * L)
    return td


@pytest.fixture(scope="module", params=TRACKS, ids=lambda t: f"s{t[0]}-{t[1]}")
def addon(request, tmp_path_factory) -> tuple[Path, dict]:
    td = synth.make_synthetic_tables(tmp_path_factory.mktemp("addon"))
    synth.make_evalset(td, n_regions=2, region_bins=4 * L)
    x = add_second_experiment(td, *request.param)
    (td / "build_info.json").write_text(json.dumps({"addon_of": "0" * 64}), encoding="utf-8")
    return td, x


def _cfg(**kw) -> dict:
    return {"context_bins": L, "region_bins": 4 * L, **kw}


def _sha(batch: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(batch):
        t = batch[k]
        if torch.is_tensor(t):
            t = t.detach().cpu().contiguous()
            h.update(f"{k}|{t.dtype}|{tuple(t.shape)}".encode() + t.numpy().tobytes())
    h.update(json.dumps(batch["meta"], sort_keys=True, default=str).encode())
    return h.hexdigest()


def plain_digest(tables: Path, tmp: Path) -> str:
    """sha256 over training batches (default; two-step with the average), every validation and
    blind batch (two-step with the average), and the average h5 of `build_average_h5`."""
    h = hashlib.sha256()
    ds = D.OnionDataset(tables, "train", _cfg(batch_size=2), seed=0)
    for _ in range(3):
        h.update(_sha(ds.batch(2)).encode())
    ds = D.OnionDataset(tables, "train", _cfg(batch_size=2, two_step=True, average_input=True),
                        seed=1)
    for _ in range(2):
        h.update(_sha(ds.batch(2)).encode())
    for mode, extra in (("val", {}), ("blind", {"allow_blind": True})):
        ds = D.OnionDataset(tables, mode, _cfg(batch_size=8, two_step=True, average_input=True,
                                               **extra), seed=0)
        for b in ds:
            h.update(_sha(b).encode())
    p = D.build_average_h5(tables, tmp / "avg.h5", {})
    with h5py.File(p, "r") as f:
        for track in sorted(f):
            g = f[track]
            h.update(track.encode() + str(int(g.attrs["n_cells"])).encode())
            h.update("|".join(str(x) for x in np.atleast_1d(g.attrs["measurement_ids"])).encode())
            for chrom in sorted(g):
                h.update(chrom.encode() + g[chrom][:].tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------- digest


def test_tables_without_the_row_are_bit_identical(plain, tmp_path):
    key = platform_key()
    if key not in BASE_DIGESTS:
        pytest.fail(f"no base digest recorded for platform {key!r}; recorded: "
                    f"{sorted(BASE_DIGESTS)}")
    assert plain_digest(plain, tmp_path) == BASE_DIGESTS[key]


# ------------------------------------------------------------------------------- train refused


def test_fixture_would_pick_the_second_experiment_without_the_guard(addon):
    """The second experiment ranks before the EIC base view, so the tests below can fail."""
    td, x = addon
    eic = next(r for r in _read_rows(td) if r["cell"] == x["cell"] and r["assay"] == x["assay"]
               and not r["arm_knob"] and r["in_eic"] != "evalonly")
    assert D._base_rank(x) < D._base_rank(eic)


@pytest.mark.parametrize("cfg", TRAIN_CONFIGS, ids=("default", "final", "final+val"))
def test_train_mode_refuses_an_evalonly_row(addon, cfg, tmp_path):
    td, _ = addon
    only_row = tmp_path / "tables" / "x"
    shutil.copytree(td, only_row)
    (only_row / "build_info.json").unlink()            # the row alone must suffice
    for t in (td, only_row):
        with pytest.raises(D.RoleError, match="evalonly|addon_of"):
            D.OnionDataset(t, "train", _cfg(**cfg))


@pytest.mark.parametrize("cfg", TRAIN_CONFIGS, ids=("default", "final", "final+val"))
def test_train_mode_refuses_addon_of_alone(plain, cfg, tmp_path):
    t = tmp_path / "tables" / "plain_addon"
    shutil.copytree(plain, t)
    (t / "build_info.json").write_text(json.dumps({"addon_of": "0" * 64, "x": 1}),
                                       encoding="utf-8")
    with pytest.raises(D.RoleError, match="addon_of"):
        D.OnionDataset(t, "train", _cfg(**cfg))
    with pytest.raises(D.RoleError):
        D.build_average_h5(t, tmp_path / "avg.h5", cfg)


def test_build_info_without_addon_of_is_accepted(plain, tmp_path):
    t = tmp_path / "tables" / "plain_info"
    shutil.copytree(plain, t)
    (t / "build_info.json").write_text(json.dumps({"tables_sha256": "0" * 64}), encoding="utf-8")
    D.OnionDataset(t, "train", _cfg(genome_dir=str(plain.parent.parent / "genome")))


def test_build_average_h5_refuses_addon_tables(addon, tmp_path):
    td, _ = addon
    with pytest.raises(D.RoleError):
        D.build_average_h5(td, tmp_path / "avg.h5", {})


def test_unknown_in_eic_still_refused(plain, tmp_path):
    t = tmp_path / "tables" / "bad"
    shutil.copytree(plain, t)
    text = (t / "measurements.tsv").read_text(encoding="utf-8")
    first = text.splitlines()[1].split("\t")
    i = D.MEASUREMENT_COLUMNS.index("in_eic")
    bad = first[:]
    bad[i] = "evaluation_only"
    text = text.replace("\t".join(first), "\t".join(bad), 1)
    (t / "measurements.tsv").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="in_eic"):
        D.OnionDataset(t, "val", _cfg())


# ------------------------------------------------------------------------------- val and blind


@pytest.mark.parametrize("mode", ("val", "blind"))
def test_other_rows_see_no_change(plain, addon, mode):
    td, x = addon
    cfg = _cfg(allow_blind=True) if mode == "blind" else _cfg()
    a = D.OnionDataset(plain, mode, cfg)
    b = D.OnionDataset(td, mode, cfg)
    assert set(b.rows) - set(a.rows) == {X_MID}
    assert b.allowed - a.allowed == {X_MID}
    for mid, row in a.rows.items():
        assert b._base_view(b.rows[mid]) == a._base_view(row), mid
    assert b.train_base == a.train_base
    assert b.train_tracks == a.train_tracks
    for roles in (("train",), ("train", "val"), ("train", "val", "blind"), ("evalonly",)):
        assert b.group_tracks(roles) == a.group_tracks(roles)
    assert not any(X_MID in v for v in b.group_tracks(("train", "val", "blind",
                                                         "evalonly")).values())
    chrom = a.chroms[0]
    for cell in sorted({r["cell"] for r in a.rows.values()}):
        assert np.array_equal(b.average(cell, chrom, 40), a.average(cell, chrom, 40)), cell


def test_base_leg_of_every_other_row_unchanged(plain, addon):
    td, x = addon
    cfg = _cfg(two_step=True)
    a = D.OnionDataset(plain, "val", cfg)
    b = D.OnionDataset(td, "val", cfg)
    chrom = a.chroms[0]
    for mid in sorted(a.allowed):
        ca, ya = a._base_leg(mid, chrom, 40, L)
        cb, yb = b._base_leg(mid, chrom, 40, L)
        assert np.array_equal(ca, cb) and np.array_equal(ya, yb), mid


def test_val_batches_unchanged_by_the_row(plain, addon):
    """The add-on tables with the frozen evalset give the same validation batches."""
    td, _ = addon
    cfg = _cfg(batch_size=8, two_step=True, average_input=True,
               evalset=str(plain.parent.parent / "eval" / plain.name / "evalset.json"))
    a = [_sha(bt) for bt in D.OnionDataset(plain, "val", cfg)]
    b = [_sha(bt) for bt in D.OnionDataset(td, "val", cfg)]
    assert a == b and len(a) > 0


def test_second_experiment_as_a_target_reads_its_own_counts(addon):
    td, x = addon
    ds = D.OnionDataset(td, "val", _cfg(two_step=True))
    assert ds.check_row(X_MID)["in_eic"] == "evalonly"
    assert ds._base_view(x) == X_MID
    eic_base = next(m for m in ds.rows if ds.rows[m]["cell"] == x["cell"]
                    and ds.rows[m]["assay"] == x["assay"] and not ds.rows[m]["arm_knob"]
                    and m != X_MID)
    inputs = [ds.train_base[(c, a)] for (c, a) in ds.train_base
              if c == x["cell"] and a != x["assay"]]
    chrom, t = ds.chroms[0], C.TRACK_INDEX[x["assay"]]
    win = ds.window(chrom, 40, inputs, [X_MID])
    with h5py.File(Path(x["counts_path"][3:]), "r") as f:
        own = f[chrom][40:40 + L].astype(np.int32)
    assert np.array_equal(win["y_counts"][:, t].numpy(), own)
    assert np.array_equal(win["y_base"][:, t].numpy(), own)
    assert int(win["y_kind"][t]) == C.KIND_IMPUTE
    assert not np.array_equal(own, ds.read(eic_base, chrom, 40))
    win2 = ds.window(chrom, 40, inputs, [X_MID], target_file="pr2")
    assert np.array_equal(win2["y_counts"][:, t].numpy(), ds.read(X_MID, chrom, 40,
                                                                 which="pr2"))


def test_second_experiment_is_never_an_input(addon):
    td, x = addon
    ds = D.OnionDataset(td, "val", _cfg())
    chrom = ds.chroms[0]
    with pytest.raises(D.RoleError, match="evalonly"):
        ds.window(chrom, 40, [X_MID], [X_MID])
    other = next(m for (c, a), m in ds.train_base.items() if c == x["cell"] and a != x["assay"])
    with pytest.raises(D.RoleError, match="evalonly"):
        ds.window(chrom, 40, [X_MID], [other])


def test_addon_evalset_items_hold_the_target_but_no_input_use(addon, tmp_path):
    """An add-on target gets imputation items and no reconstruction item (which would feed
    it as an input); every other item is the frozen evalset's."""
    td, x = addon
    frozen = json.loads((td.parent.parent / "eval" / td.name / "evalset.json").read_text())
    ev = json.loads(json.dumps(frozen))
    inputs = sorted(m for (c, a), m in D.OnionDataset(td, "val", _cfg()).train_base.items()
                    if c == x["cell"] and a != x["assay"])
    ev["imputation_targets"].append({"target": X_MID, "inputs": inputs,
                                     "regime": "leave_one_assay_out", "split": "val",
                                     "evaluation_only": True})
    p = tmp_path / "evalset_addon.json"
    p.write_text(json.dumps(ev), encoding="utf-8")
    ds = D.OnionDataset(td, "val", _cfg(evalset=str(p)))
    items = ds.evalset_items()
    mine = [it for it in items if X_MID in it["targets"] or X_MID in it["inputs"]]
    assert mine and {it["task"] for it in mine} == {"imputation"}
    assert all(X_MID not in it["inputs"] for it in items)
    for it in mine:
        ds.item_window(it)
    pf = tmp_path / "evalset_frozen.json"
    pf.write_text(json.dumps(frozen), encoding="utf-8")
    base = D.OnionDataset(td, "val", _cfg(evalset=str(pf))).evalset_items()
    strip = [{k: v for k, v in it.items() if k != "item"} for it in items if it not in mine]
    assert strip == [{k: v for k, v in it.items() if k != "item"} for it in base]


# ------------------------------------------------------------------------------- pinned rule


def test_no_minicandi_file_names_an_addon_table_set():
    hits = []
    for d in ("tools/minicandi", "slurm/minicandi"):
        root = _ROOT / d
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                text = p.read_bytes()
                for name in (b"eic_v1" + b"x", b"eic_v2" + b"x"):
                    if name in text:
                        hits.append(f"{p.relative_to(_ROOT)}: {name.decode()}")
    assert not hits, hits


if __name__ == "__main__":
    import tempfile
    if "--print" in sys.argv:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            td = synth.make_synthetic_tables(tmp / "plain")
            synth.make_evalset(td, n_regions=2, region_bins=4 * L)
            print(json.dumps({platform_key(): plain_digest(td, tmp)}))
