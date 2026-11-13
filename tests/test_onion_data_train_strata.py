"""STt: training-region lists and strata tables (tools/onion_data/train_strata.py; plan Amendment 7).

Synthetic genome, tables, peaks, average and embedding; hand cases with known answers."""
from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_data import train_strata as S  # noqa: E402

RB = 16384
NBLK = 12                       # blocks per chromosome
CHROMS = ("chr1", "chr2")
EVAL = [("chr1", 2 * RB), ("chr2", 5 * RB)]
FIT = [("chr1", 7 * RB)]
MASKED_BLOCK = ("chr2", 9 * RB)          # valid fraction 0.5: not a block
PART_TILE = ("chr1", 3 * RB + 2048)      # valid fraction 0.85: block stays, this tile is dropped
CELLS = ("C01", "C02", "C03")
ASSAYS = ("DNase-seq", "H3K4me3", "H3K27ac")
PEAK_TILE = ("chr1", 4 * RB)             # every training track has a peak here


def _write_tsv(path, rows):
    cols = list(rows[0])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def make_genome(root: Path) -> Path:
    g = root / "genome"
    g.mkdir()
    nb = NBLK * RB
    (g / "chrom_sizes.json").write_text(json.dumps({c: nb * 25 for c in CHROMS}))
    with h5py.File(g / "mask.h5", "w") as f:
        for c in CHROMS:
            m = np.ones(nb, bool)
            if c == MASKED_BLOCK[0]:
                s = MASKED_BLOCK[1]
                m[s:s + RB // 2] = False
            if c == PART_TILE[0]:
                s = PART_TILE[1]
                m[s:s + 308] = False              # 1740 / 2048 = 0.85 valid
            f.create_dataset(c, data=m)
    return g


def make_evalset(root: Path) -> Path:
    p = root / "evalset.json"
    p.write_text(json.dumps({"regions": [list(r) for r in EVAL], "fit_regions": [list(r) for r in FIT],
                             "region_bins": RB}))
    return p


def _peaks(path: Path, intervals):
    with gzip.open(path, "wt") as fh:
        for c, s, e in intervals:
            fh.write(f"{c}\t{s}\t{e}\t.\t0\t.\t1\t1\t1\t10\n")


def make_world(root: Path, rng_seed: int = 0) -> dict:
    """Tables with 3 training cells x 3 assays, one validation track (C04 DNase) and a control."""
    rng = np.random.default_rng(rng_seed)
    g = make_genome(root)
    ev = make_evalset(root)
    tables = root / "tables"
    tables.mkdir()
    peaks = root / "peaks"
    peaks.mkdir()
    counts = root / "counts"
    counts.mkdir()
    nb = NBLK * RB
    rows, prow = [], []
    tracks = [(c, a, "train") for c in CELLS for a in ASSAYS] + [("C04", "DNase-seq", "val"),
                                                                 ("C01", "chipseq-control", "train")]
    for k, (c, a, role) in enumerate(tracks):
        t = f"{c}M{k:02d}"
        cp = counts / f"{t}.h5"
        if role == "val":                                  # planted: opening it would fail loudly
            cp.write_bytes(b"validation counts: must never be opened")
        else:
            with h5py.File(cp, "w") as f:
                for ch in CHROMS:
                    f.create_dataset(ch, data=rng.poisson(1 + k % 4, nb).astype(np.int32))
        rows.append({"measurement_id": f"{t}.base", "eic_track": t, "cell": c, "assay": a,
                     "in_eic": role, "arm_knob": "base", "layer": "uniform", "unit": "pooled",
                     "depth_reads": "1000000", "counts_path": f"h5:{cp}"})
        if a == "chipseq-control":
            continue
        ivs = [(PEAK_TILE[0], PEAK_TILE[1] * 25 + 100, PEAK_TILE[1] * 25 + 600)]
        if k % 3 == 0:                                     # some tracks peak in a second tile
            ivs.append(("chr2", (5 * RB + 4096) * 25, (5 * RB + 4096) * 25 + 300))
        _peaks(peaks / f"{t}.narrowPeak.gz", ivs)
        prow.append({"eic_track": t, "role": role, "experiment": f"E{k}", "file": f"F{k}"})
    _write_tsv(tables / "measurements.tsv", rows)
    _write_tsv(root / "peak_files.tsv", prow)
    avg = root / "average.h5"
    with h5py.File(avg, "w") as f:
        for a in ASSAYS:
            gr = f.create_group(a)
            mids = [r["measurement_id"] for r in rows if r["assay"] == a and r["in_eic"] == "train"]
            gr.attrs["n_cells"] = len(mids)
            gr.attrs["measurement_ids"] = np.asarray(mids, dtype=h5py.string_dtype())
            for ch in CHROMS:
                gr.create_dataset(ch, data=rng.gamma(2.0, 1.0, nb).astype(np.float32))
    emb = root / "ntv2_pca"
    emb.mkdir()
    (emb / "meta.json").write_text(json.dumps({"source": "ntv2", "trained_on_encode_tracks": False,
                                               "model_id": "InstaDeepAI/nucleotide-transformer-v2"}))
    for ch in CHROMS:
        with h5py.File(emb / f"{ch}.h5", "w") as f:
            d = f.create_dataset("emb", data=rng.normal(size=(nb, 4)).astype(np.float16))
            d.attrs["source"] = "ntv2"
    return {"genome": g, "evalset": ev, "tables": tables, "peaks": peaks,
            "peak_files": root / "peak_files.tsv", "average": avg, "embedding": emb, "rows": rows}


def _qw(rows):
    return sum(r["sample_weight"] * r["loss_weight"] for r in rows), sum(r["sample_weight"] for r in rows)


def _overlaps_taken(rows, n_bins_key="n_bins"):
    taken = [(c, s, s + RB) for c, s in EVAL + FIT]
    for r in rows:
        for c, s, e in taken:
            if r["chrom"] == c and r["start_bin"] < e and r["start_bin"] + r[n_bins_key] > s:
                return True
    return False


# ------------------------------------------------------------------------------------ universe

def test_universe_lists(tmp_path):
    w = make_world(tmp_path)
    out = tmp_path / "lists"
    res = S.run_universe(w["evalset"], w["genome"], [0, 1, 2], out, sizes=(2, 4, 8))
    n_blocks = 2 * NBLK - 1                       # the half-masked block is no block
    assert res["U"] == n_blocks - len(EVAL) - len(FIT)
    for f in sorted(out.glob("*.tsv")):
        head, rows = S.read_list(f)
        assert head["roles_read"] == "train" and head["universe"] and len(head["sha256_inputs"]) == 64
        qw, q = _qw(rows)
        assert qw == pytest.approx(1.0, abs=1e-12) and q == pytest.approx(1.0, abs=1e-12)
        if f.name != "eval_heavy.tsv":
            assert not _overlaps_taken(rows), f.name
            assert all(r["loss_weight"] == 1.0 for r in rows)
    _, eh = S.read_list(out / "eval_heavy.tsv")
    assert sum(r["sample_weight"] for r in eh if r["stratum"] == "eval") == pytest.approx(0.5)
    assert {(r["chrom"], r["start_bin"]) for r in eh if r["stratum"] == "eval"} == set(EVAL)
    sets = {}
    for s in (0, 1, 2):
        prev = set()
        for R in (2, 4, 8):
            _, rows = S.read_list(out / f"terr_r{R}_s{s}.tsv")
            cur = {(r["chrom"], r["start_bin"]) for r in rows}
            assert len(cur) == R and prev <= cur          # nested within a seed
            prev = cur
        sets[s] = prev
    assert sets[0] != sets[1] and sets[1] != sets[2] and sets[0] != sets[2]
    with pytest.raises(ValueError):
        S.run_universe(w["evalset"], w["genome"], [0], tmp_path / "x", sizes=(4096,))


def test_universe_refuses_foreign_genome(tmp_path):
    w = make_world(tmp_path)
    ev = json.loads(w["evalset"].read_text())
    ev["regions"].append(list(MASKED_BLOCK))      # not a block of this mask
    w["evalset"].write_text(json.dumps(ev))
    with pytest.raises(ValueError, match="not blocks"):
        S.run_universe(w["evalset"], w["genome"], [0], tmp_path / "x", sizes=(2,))


# ------------------------------------------------------------------------------------ tiles

def _run_tiles(w, out, k=6):
    return S.run_tiles(w["tables"], w["peaks"], w["peak_files"], w["average"], w["embedding"],
                       w["genome"], w["evalset"], out, k=k)


def _tile_table(path):
    lines = [x for x in Path(path).read_text().splitlines() if not x.startswith("#")]
    cols = lines[0].split("\t")
    return [dict(zip(cols, x.split("\t"))) for x in lines[1:]]


def test_tiles_files(tmp_path):
    w = make_world(tmp_path)
    out = tmp_path / "strata"
    res = _run_tiles(w, out)
    n_u = 2 * NBLK - 1 - len(EVAL) - len(FIT)
    assert res["tiles"] == n_u * 8 - 1                      # one tile below 0.9 valid
    assert res["tracks"] == len(CELLS) * len(ASSAYS)        # the validation track is not counted
    tt = _tile_table(out / "tile_table.tsv")
    keys = {(r["chrom"], int(r["start_bin"])) for r in tt}
    assert PART_TILE not in keys
    assert all(int(r["valid_bins"]) >= 0.9 * 2048 for r in tt)
    pk = {(r["chrom"], int(r["start_bin"])): r for r in tt}
    assert float(pk[PEAK_TILE]["peak_fraction"]) == 1.0
    assert pk[PEAK_TILE]["ccre_like_class"] == "promoter_like"
    for name in ("tile_uniform", "peakfreq_w", "peakfreq_u", "ccre_like_w", "variance_w", "kcenter"):
        head, rows = S.read_list(out / f"{name}.tsv")
        assert head["roles_read"] == "train"
        qw, q = _qw(rows)
        assert qw == pytest.approx(1.0, abs=1e-12), name
        assert q == pytest.approx(1.0, abs=1e-12), name
        assert not _overlaps_taken(rows), name
        assert all(r["n_bins"] == 2048 for r in rows)
    _, tu = S.read_list(out / "tile_uniform.tsv")
    assert len(tu) == res["tiles"] and all(r["loss_weight"] == 1.0 for r in tu)
    assert {r["stratum"] for r in tu} == {"tile"}
    vb = {(r["chrom"], int(r["start_bin"])): int(r["valid_bins"]) for r in tt}
    tot = sum(vb.values())
    assert all(r["sample_weight"] == pytest.approx(vb[(r["chrom"], r["start_bin"])] / tot) for r in tu)
    _, pu = S.read_list(out / "peakfreq_u.tsv")
    assert all(r["loss_weight"] == 1.0 for r in pu)
    _, pw = S.read_list(out / "peakfreq_w.tsv")
    assert [r["sample_weight"] for r in pu] == [r["sample_weight"] for r in pw]
    assert any(r["loss_weight"] != 1.0 for r in pw)
    _, kc = S.read_list(out / "kcenter.tsv")
    assert len(kc) == 6
    _, vw = S.read_list(out / "variance_w.tsv")
    q = {}
    for r in vw:
        q[r["stratum"]] = q.get(r["stratum"], 0) + r["sample_weight"]
    w_ = {r["stratum"]: r["loss_weight"] for r in vw}
    assert w_["d10"] == pytest.approx(w_["d01"] / 2, rel=1e-12)
    md = (out / "STRATA.md").read_text()
    for name in ("peakfreq_w", "peakfreq_u", "ccre_like_w", "variance_w", "kcenter"):
        assert f"## {name}" in md
    assert "Peak-frequency denominator: 9 training tracks" in md
    assert "Effective sample size" in md and "Cluster sizes" in md
    assert "## tile_uniform" in md and "edge effect" in md
    assert f"1 of {n_u * 8} tiles of U" in md                # the tile below 0.9 valid gets q = 0
    n = res["tiles"]
    assert f"kcenter {(n - 6) / n:.1%}" in md and "biased coreset by design" in md
    assert "tile_uniform 0.0%" in md
    assert "cCRE-like denominators: 3 training cells" in md
    sec = md.split("## peakfreq_u")[1].split("## ccre_like_w")[0]
    g_rows = [x.split("|") for x in sec.splitlines() if x.startswith("| pf_")]
    gw = md.split("## peakfreq_w")[1].split("## peakfreq_u")[0]
    gw_rows = [x.split("|") for x in gw.splitlines() if x.startswith("| pf_")]
    assert [r[3] for r in g_rows] == [r[3] for r in gw_rows]       # same g_h in both arms
    assert sum(float(r[3]) for r in g_rows) == pytest.approx(1.0, abs=1e-5)


def test_region_classes(tmp_path):
    w = make_world(tmp_path)
    out = tmp_path / "strata"
    _run_tiles(w, out)
    rc = tmp_path / "classes.tsv"
    rows = S.run_region_classes(out / "tile_table.tsv", w["evalset"], w["tables"], w["peaks"],
                                w["peak_files"], rc)
    assert [(r["chrom"], r["start_bin"]) for r in rows] == EVAL
    by = {(r["chrom"], r["start_bin"]): r for r in rows}
    # chr2 eval region holds the second-tile peak of tracks k % 3 == 0 (3 of 9) in 1 of 8 tiles
    assert by[EVAL[1]]["peak_fraction"] == pytest.approx(3 / 9 / 8)
    assert by[EVAL[1]]["class"] == "pf_0-0.05"
    assert by[EVAL[0]]["class"] == "pf_0"
    assert rc.read_text().splitlines()[0] == "chrom\tstart_bin\tclass\tpeak_fraction"
    pf = list(csv.DictReader(open(w["peak_files"]), delimiter="\t"))
    pf[0]["file"] = ""                                  # another denominator
    _write_tsv(w["peak_files"], pf)
    with pytest.raises(ValueError, match="other peak tracks"):
        S.run_region_classes(out / "tile_table.tsv", w["evalset"], w["tables"], w["peaks"],
                             w["peak_files"], rc)


# ------------------------------------------------------------------------------------ guards

def test_validation_row_raises(tmp_path):
    w = make_world(tmp_path)
    rows = [dict(r) for r in w["rows"]]
    rows[0]["in_eic"] = "val"                           # a track whose peak file says train
    _write_tsv(w["tables"] / "measurements.tsv", rows)
    with pytest.raises(S.LeakError):
        S.peak_tracks(w["tables"], w["peaks"], w["peak_files"])
    with pytest.raises(S.LeakError):
        S.require_train(rows[0])


def test_validation_peak_file_raises(tmp_path):
    w = make_world(tmp_path)
    pf = list(csv.DictReader(open(w["peak_files"]), delimiter="\t"))
    val = [r for r in pf if r["role"] == "val"][0]
    val["role"] = "train"                               # the validation track's peak file
    _write_tsv(w["peak_files"], pf)
    with pytest.raises(S.LeakError):
        S.peak_tracks(w["tables"], w["peaks"], w["peak_files"])
    val["role"] = "evaluation"                          # an unknown role
    _write_tsv(w["peak_files"], pf)
    with pytest.raises(S.LeakError):
        S.peak_tracks(w["tables"], w["peaks"], w["peak_files"])


def test_validation_peak_file_not_read(tmp_path):
    w = make_world(tmp_path)
    val = [r for r in w["rows"] if r["in_eic"] == "val"][0]["eic_track"]
    (w["peaks"] / f"{val}.narrowPeak.gz").unlink()      # a val row's file is never opened
    tr = S.peak_tracks(w["tables"], w["peaks"], w["peak_files"])
    assert val not in {t for t, *_ in tr} and len(tr) == 9


def test_average_with_non_training_id_raises(tmp_path):
    w = make_world(tmp_path)
    val = [r for r in w["rows"] if r["in_eic"] == "val"][0]["measurement_id"]
    with h5py.File(w["average"], "a") as f:
        ids = [x.decode() if isinstance(x, bytes) else x for x in f["DNase-seq"].attrs["measurement_ids"]]
        f["DNase-seq"].attrs["measurement_ids"] = np.asarray(ids + [val], dtype=h5py.string_dtype())
    with pytest.raises(S.LeakError):
        S.check_average(w["average"], w["tables"])
    with pytest.raises(S.LeakError):
        _run_tiles(w, tmp_path / "o")
    with h5py.File(w["average"], "a") as f:
        f["DNase-seq"].attrs["measurement_ids"] = np.asarray(["not_a_row"], dtype=h5py.string_dtype())
    with pytest.raises(S.LeakError):
        S.check_average(w["average"], w["tables"])
    with h5py.File(w["average"], "a") as f:
        del f["DNase-seq"].attrs["measurement_ids"]
    with pytest.raises(S.LeakError):
        S.check_average(w["average"], w["tables"])


@pytest.mark.parametrize("where", ["meta_source", "meta_encode", "meta_encode_missing",
                                   "no_meta", "chrom_attr"])
def test_non_ntv2_embedding_raises(tmp_path, where):
    w = make_world(tmp_path)
    emb = w["embedding"]
    meta = json.loads((emb / "meta.json").read_text())
    if where == "meta_source":
        meta["source"] = "flashzoi"
    elif where == "meta_encode":
        meta["trained_on_encode_tracks"] = True
    elif where == "meta_encode_missing":
        del meta["trained_on_encode_tracks"]
    (emb / "meta.json").write_text(json.dumps(meta))
    if where == "no_meta":
        (emb / "meta.json").unlink()
    if where == "chrom_attr":
        with h5py.File(emb / "chr1.h5", "a") as f:
            f["emb"].attrs["source"] = "flashzoi"
    with pytest.raises(S.LeakError):
        S.check_embedding(emb)


def test_ntv2_embedding_passes(tmp_path):
    w = make_world(tmp_path)
    assert S.check_embedding(w["embedding"])["source"] == "ntv2"
    one = tmp_path / "one.h5"                                   # the single-file layout
    with h5py.File(one, "w") as f:
        f.attrs["source"] = "ntv2"
        f.attrs["trained_on_encode_tracks"] = False
        f.create_dataset("chr1", data=np.zeros((4, 2), np.float16))
    S.check_embedding(one)
    with h5py.File(one, "a") as f:
        f.attrs["trained_on_encode_tracks"] = True
    with pytest.raises(S.LeakError):
        S.check_embedding(one)


def test_validation_counts_never_opened(tmp_path, monkeypatch):
    w = make_world(tmp_path)
    val = [r for r in w["rows"] if r["in_eic"] == "val"][0]["counts_path"][3:]
    opened = []
    real = h5py.File

    def spy(name, *a, **k):
        opened.append(str(name))
        return real(name, *a, **k)
    monkeypatch.setattr(h5py, "File", spy)
    _run_tiles(w, tmp_path / "o")
    assert opened and val not in opened
    assert any("counts" in o for o in opened)                 # training counts were read


def test_second_experiment_is_removed(tmp_path):
    w = make_world(tmp_path)
    second = [r for r in w["rows"] if r["in_eic"] == "train" and r["assay"] == "H3K27ac"][0]
    _write_tsv(w["tables"] / "evaluation_only.tsv",
               [{"measurement_id": second["measurement_id"], "use": "other_experiment"}])
    trows, _ = S.train_tables(w["tables"])
    assert second["measurement_id"] not in {r["measurement_id"] for r in trows}
    with pytest.raises(S.LeakError):                          # the average sums it
        S.check_average(w["average"], w["tables"])


def test_no_ccre_or_chromhmm_option():
    with pytest.raises(SystemExit):
        S.main(["tiles", "--ccre", "x"])
    src = Path(S.__file__).read_text()
    assert "--ccre" not in src and "--chromhmm" not in src.lower()


# ------------------------------------------------------------------------------------ units

def test_kcenter_planted_clusters():
    rng = np.random.default_rng(1)
    centres = np.array([[0, 0], [50, 0], [0, 50], [50, 50], [100, 100]], float)
    sizes = [7, 3, 12, 5, 9]
    X = np.concatenate([c + rng.normal(scale=0.5, size=(n, 2)) for c, n in zip(centres, sizes)])
    lab = np.repeat(np.arange(5), sizes)
    for seed in range(4):
        picked, sz = S.kcenter(X, 5, seed=seed)
        assert sorted(lab[picked]) == [0, 1, 2, 3, 4]
        assert sorted(sz) == sorted(sizes)
        assert {int(lab[p]): int(n) for p, n in zip(picked, sz)} == dict(enumerate(sizes))


def test_ccre_like_hand_classes():
    tracks = [("a1", "A", "DNase-seq", None), ("a2", "A", "H3K4me3", None),
              ("a3", "A", "H3K27ac", None), ("a4", "A", "H3K4me1", None),
              ("b1", "B", "ATAC-seq", None), ("b2", "B", "H3K4me3", None),
              ("b3", "B", "H3K27me3", None), ("c1", "C", "DNase-seq", None),
              ("c2", "C", "H3K27ac", None)]
    # tiles: 0 promoter (A acc + A K4me3), 1 enhancer (A acc + A K27ac), 2 acc in A and K4me3 in B
    # only -> accessible_only, 3 only K27me3 -> other_marked, 4 nothing -> none,
    # 5 A acc + K4me1 + K4me3 -> promoter, 6 B ATAC + A K4me1 -> accessible_only,
    # 7 B ATAC + B K4me3 -> promoter, 8 A acc + K4me1 -> enhancer
    # 9: C (no H3K4me3 track) acc + K27ac -> accessible_only, never enhancer_like
    H = np.zeros((9, 10), bool)
    on = {0: [0, 1], 1: [0, 2], 2: [0, 5], 3: [6], 5: [0, 3, 1], 6: [4, 3], 7: [4, 5], 8: [0, 3],
          9: [7, 8]}
    for t, ks in on.items():
        H[ks, t] = True
    n = {}
    assert S.ccre_like(tracks, H, n) == ["promoter_like", "enhancer_like", "accessible_only",
                                         "other_marked", "none", "promoter_like", "accessible_only",
                                         "promoter_like", "enhancer_like", "accessible_only"]
    assert n == {"cells": 3, "accessibility": 3, "promoter_rule": 2, "enhancer_rule": 1}


def test_peakfreq_edges_and_deciles():
    f = np.array([0, 1e-9, 0.05, 0.0500001, 0.2, 0.21, 0.5, 0.51, 1.0])
    assert S.peakfreq_class(f) == ["pf_0", "pf_0-0.05", "pf_0-0.05", "pf_0.05-0.20",
                                   "pf_0.05-0.20", "pf_0.20-0.50", "pf_0.20-0.50", "pf_gt0.50",
                                   "pf_gt0.50"]
    d = S.decile_strata(np.arange(100, dtype=float)[::-1])
    assert d[0] == "d10" and d[-1] == "d01" and len(set(d)) == 10


def test_write_list_refuses_bad_sums(tmp_path):
    rows = [{"chrom": "chr1", "start_bin": 0, "n_bins": 2048, "stratum": "a",
             "sample_weight": 0.5, "loss_weight": 1.0}]
    with pytest.raises(ValueError):
        S.write_list(tmp_path / "x.tsv", rows, "u", "0" * 64)


def _strata_of(totals_by_assay):
    """Decile stratum of each tile from linear totals {assay: [cells, tiles]}."""
    return S.decile_strata(S.variance_score({a: np.log1p(v) for a, v in totals_by_assay.items()}))


def test_variance_equal_mean_different_spread():
    rng = np.random.default_rng(3)
    base = rng.uniform(1e3, 1e4, size=(4, 40)) * rng.uniform(0.5, 2.0, size=(4, 40))
    base[:, 0] = [1000, 1000, 1000, 1000]                     # mean 1000, no spread
    base[:, 1] = [100, 400, 1600, 1900]                       # mean 1000, wide spread
    st = _strata_of({"H3K4me3": base, "H3K27ac": base})
    assert st[0] == "d01" and st[1] == "d10"


def test_variance_equal_relative_spread_same_stratum():
    rng = np.random.default_rng(4)
    base = rng.uniform(1e3, 1e4, size=(4, 40)) * rng.uniform(0.5, 2.0, size=(4, 40))
    base[:, 0] = np.array([1, 2, 4, 8]) * 5e3                 # mean 18,750
    base[:, 1] = np.array([1, 2, 4, 8]) * 5e5                 # mean 100 times higher, same ratios
    st = _strata_of({"H3K4me3": base})
    assert st[0] == st[1]
    v = np.log1p(base[:, :2]).var(axis=0, ddof=1)             # log1p is ~log at these totals
    assert v[0] == pytest.approx(v[1], rel=1e-3)
