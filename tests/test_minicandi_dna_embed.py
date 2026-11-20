"""tools/minicandi/dna_embed.py with a stub model on a tiny synthetic genome (no download, no torch model)."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import contract as C  # noqa: E402
from minicandi import dna_embed as E  # noqa: E402

CHROMS = {"chr1": 3 * 4096 * 25 + 17, "chr2": 1500 * 25 + 24}     # lengths in bp, not multiples of 25


def make_genome(root: Path, seed: int = 0) -> Path:
    g = root / "genome"
    g.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    with h5py.File(g / "dna.h5", "w") as fd, h5py.File(g / "mask.h5", "w") as fm:
        for c, n in CHROMS.items():
            codes = rng.integers(0, 4, n).astype(np.uint8)
            codes[:300] = 4
            fd.create_dataset(c, data=codes)
            m = np.ones(n // 25, np.uint8)
            m[:12] = 0
            fm.create_dataset(c, data=m)
    return g


def composition(codes: np.ndarray, native_bp: int) -> np.ndarray:
    """Fractions of A, C, G, T, N per native bin: the stub's embedding, a pure function of the bases."""
    n = codes.shape[-1] // native_bp
    k = codes[..., : n * native_bp].reshape(*codes.shape[:-1], n, native_bp)
    return np.stack([(k == j).mean(-1) for j in range(5)], -1).astype(np.float32)


def stub(native_bp: int = 32, out_native: int = 64, crop_native: int = 16, calls=None) -> E.Source:
    crop = crop_native * native_bp

    def embed(x):
        if calls is not None:
            calls.append(x.shape)
        return composition(x[:, crop:crop + out_native * native_bp], native_bp)
    return E.Source("stub", native_bp=native_bp, input_bp=out_native * native_bp + 2 * crop,
                    out_bp=out_native * native_bp, d=5, embed=embed, meta={"model_id": "stub"})


def expected(genome: Path, chrom: str, native_bp: int) -> np.ndarray:
    with h5py.File(genome / "dna.h5") as f:
        codes = f[chrom][:]
    nb = CHROMS[chrom] // 25
    idx = E.native_index(nb, native_bp)
    pad = np.full((idx[-1] + 1) * native_bp, 4, np.uint8)
    pad[: min(pad.size, codes.size)] = codes[: pad.size]
    return composition(pad, native_bp)[idx]


@pytest.fixture()
def genome(tmp_path):
    return make_genome(tmp_path)


def test_res_matches_contract():
    assert E.RES == C.RES == 25


@pytest.mark.parametrize("native_bp", [32, 6, 25, 128, 7])
def test_native_index_holds_centre(native_bp):
    idx = E.native_index(5000, native_bp)
    centre = 25 * np.arange(5000) + 12.5
    assert np.all(idx * native_bp <= centre) and np.all(centre < (idx + 1) * native_bp)
    assert np.all(np.diff(idx) >= 0)


def test_native_index_hand_cases():
    assert E.native_index(6, 32).tolist() == [0, 1, 1, 2, 3, 4]       # centres 12.5 37.5 62.5 87.5 112.5 137.5
    assert E.native_index(3, 6).tolist() == [2, 6, 10]                # centres 12.5 37.5 62.5
    assert E.native_index(4, 25).tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("native_bp", [32, 6])
def test_precompute_values_shape_chunks(genome, tmp_path, native_bp):
    out = tmp_path / "emb"
    E.precompute(stub(native_bp), genome, out, ["chr1", "chr2"], batch=3, log=lambda s: None)
    for c in CHROMS:
        with h5py.File(out / f"{c}.h5") as h:
            ds = h["emb"]
            assert ds.dtype == np.float16 and ds.shape == (CHROMS[c] // 25, 5)
            assert ds.chunks == (min(4096, CHROMS[c] // 25), 5)
            np.testing.assert_allclose(ds[:].astype(np.float32), expected(genome, c, native_bp), atol=1e-3)
    meta = json.loads((out / "meta.json").read_text())
    for k in ("model_id", "native_bin_bp", "receptive_field_bp", "d", "projection", "date", "grid_rule"):
        assert k in meta
    assert meta["native_bin_bp"] == native_bp and meta["d"] == 5 and meta["projection"] is None


def test_batch_size_invariant_and_skip_existing(genome, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    E.precompute(stub(), genome, a, ["chr2"], batch=1, log=lambda s: None)
    E.precompute(stub(), genome, b, ["chr2"], batch=7, log=lambda s: None)
    with h5py.File(a / "chr2.h5") as fa, h5py.File(b / "chr2.h5") as fb:
        assert np.array_equal(fa["emb"][:], fb["emb"][:])
    calls = []
    E.precompute(stub(calls=calls), genome, a, ["chr2"], log=lambda s: None)
    assert calls == []                                               # existing file: no model call


def test_window_geometry(genome, tmp_path):
    calls = []
    E.precompute(stub(calls=calls), genome, tmp_path / "e", ["chr2"], batch=2, log=lambda s: None)
    nb = CHROMS["chr2"] // 25
    n_native = E.native_index(nb, 32)[-1] + 1
    assert sum(s[0] for s in calls) == -(-n_native // 64)
    assert all(s[1] == 64 * 32 + 2 * 16 * 32 for s in calls)


def test_wrong_embed_shape_raises(genome, tmp_path):
    s = stub()
    s.embed = lambda x: np.zeros((x.shape[0], 3, 5), np.float32)
    with pytest.raises(ValueError):
        E.precompute(s, genome, tmp_path / "e", ["chr2"], log=lambda s: None)
    assert not (tmp_path / "e" / "chr2.h5").exists()


def test_verify_ok_mismatch_nan(genome, tmp_path, capsys):
    out = tmp_path / "emb"
    E.precompute(stub(), genome, out, ["chr1", "chr2"], log=lambda s: None)
    assert E.verify(out) == (2, True, 0)
    assert capsys.readouterr().out.strip().endswith("chromosomes: 2, bins match genome layer: True, NaN: 0")
    assert E.main(["verify", "--dir", str(out), "--chroms", "chr2"]) == 0
    assert "chromosomes: 1, bins match genome layer: True, NaN: 0" in capsys.readouterr().out
    with h5py.File(out / "chr2.h5", "r+") as h:
        h["emb"][5, 1] = np.nan
    assert E.verify(out, ["chr2"]) == (1, True, 1)
    with h5py.File(out / "chr2.h5", "w") as h:
        h.create_dataset("emb", data=np.zeros((10, 5), np.float16))
    assert E.verify(out, ["chr2"])[1] is False
    assert E.verify(out, ["chr1", "chr7"])[1] is False               # a missing file


def test_project_pca(genome, tmp_path):
    out = tmp_path / "emb"
    E.precompute(stub(), genome, out, ["chr1", "chr2"], log=lambda s: None)
    with h5py.File(out / "chr2.h5") as h:
        raw2 = h["emb"][:].astype(np.float32)
    E.project(out, None, ["chr1"], 2, None, n_fit=5000, seed=1, log=lambda s: None)
    p = np.load(out / "projection.npz")
    assert p["components"].shape == (2, 5) and int(p["n_fit"]) == 5000
    np.testing.assert_allclose(p["components"] @ p["components"].T, np.eye(2), atol=1e-5)
    with h5py.File(out / "chr2.h5") as h:
        ds = h["emb"]
        assert ds.shape == (CHROMS["chr2"] // 25, 2) and ds.dtype == np.float16 and ds.chunks[0] == 1500
        np.testing.assert_allclose(ds[:].astype(np.float32), (raw2 - p["mean"]) @ p["components"].T, atol=2e-3)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["projection"]["dim"] == 2 and meta["projection"]["fit_chroms"] == ["chr1"] and meta["d"] == 2
    E.project(out, None, ["chr1"], 2, None, n_fit=5000, seed=1, log=lambda s: None)   # idempotent
    with h5py.File(out / "chr2.h5") as h:
        assert h["emb"].shape[1] == 2
    assert E.verify(out) == (2, True, 0)
    with pytest.raises(SystemExit):
        E.project(out, None, ["chr1"], 3, None, log=lambda s: None)


def test_pca_seed_and_no_labels(genome, tmp_path):
    out = tmp_path / "emb"
    E.precompute(stub(), genome, out, ["chr1"], log=lambda s: None)
    a = E.fit_pca(out, genome, ["chr1"], 2, 4000, seed=3)
    b = E.fit_pca(out, genome, ["chr1"], 2, 4000, seed=3)
    assert np.array_equal(a["components"], b["components"]) and a["n_fit"] == 4000


def test_nt_token_ids():
    letters = "ACGT"
    vocab = {"<unk>": 0, "<cls>": 3}
    for i in range(4 ** 6):
        k = "".join(letters[(i >> (2 * (5 - j))) & 3] for j in range(6))
        vocab[k] = 6 + i
    codes = np.array([[0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 1, 2, 4, 0, 0, 0, 2, 2, 2, 2, 2, 1]], np.uint8)
    ids = E.nt_token_ids(codes, vocab, unk_id=0, cls_id=3)
    assert ids.tolist() == [[3, vocab["AAAAAA"], vocab["TTTTTT"], 0, vocab["GGGGGC"]]]
    with pytest.raises(ValueError):
        E.nt_token_ids(codes[:, :20], vocab, 0, 3)


def test_model_pins_and_geometry():
    for s in ("flashzoi", "ntv2"):
        m = E.MODELS[s]
        assert len(m["revision"]) == 40 and len(m["sha256"]) == 64
    assert E.FLASHZOI_GEOM["input_bp"] - E.FLASHZOI_GEOM["out_bp"] == 2 * 163_840
    assert E.NTV2_GEOM["input_bp"] // 6 + 1 <= 2048                   # tokens + <cls> within model_max_length
    for g in (E.FLASHZOI_GEOM, E.NTV2_GEOM):
        E.Source("x", embed=None, **g).check()


def test_cli_precompute_with_stub_factory(genome, tmp_path, capsys):
    seen = {}

    def factory(source, device, weights, weights_dir):
        seen.update(source=source, device=device)
        return stub()
    out = tmp_path / "cli"
    assert E.main(["precompute", "--source", "ntv2", "--genome", str(genome), "--out", str(out),
                   "--chroms", "chr2", "--device", "cpu"], source_factory=factory) == 0
    assert seen == {"source": "ntv2", "device": "cpu"} and (out / "chr2.h5").is_file()
    assert not (out / "chr1.h5").exists()
    with pytest.raises(SystemExit):
        E.main(["precompute", "--source", "enformer", "--genome", str(genome), "--out", str(out)])
    assert E.genome_chroms(genome) == ["chr1", "chr2"]


def test_folds(tmp_path, capsys):
    rows = ["chr2\t100\t200\tfold3", "chr1\t500\t900\tfold0", "chr1\t50\t90\tfold4", "chrX\t1\t5\tfold3"]
    raw = gzip.compress(("\n".join(rows) + "\n").encode())
    src = tmp_path / "sequences_human.bed.gz"
    src.write_bytes(raw)
    rec = E.folds(str(tmp_path / "out" / "borzoi.bed"), str(src))
    assert rec["md5"] == hashlib.md5(raw).hexdigest() and rec["counts"] == {"train": 2, "test": 2}
    assert (tmp_path / "out" / "borzoi_test.bed").read_text() == "chr2\t100\t200\tfold3\nchrX\t1\t5\tfold3\n"
    assert (tmp_path / "out" / "borzoi_train.bed").read_text().splitlines()[0] == "chr1\t50\t90\tfold4"
    assert json.loads((tmp_path / "out" / "borzoi_source.json").read_text())["source"] == str(src)


def test_requirements_pinned():
    lines = [ln.split("#")[0].strip() for ln in (_ROOT / "requirements-dnaemb.txt").read_text().splitlines()]
    pins = [ln for ln in lines if ln]
    assert pins and all("==" in p for p in pins)
    names = {p.split("==")[0].lower() for p in pins}
    assert {"borzoi-pytorch", "transformers", "torch", "flash_attn", "h5py", "numpy"} <= names


def test_slurm_dry_run(tmp_path):
    sh = _ROOT / "slurm" / "minicandi" / "dna_embed.sh"
    env = {**os.environ, "DRY_RUN": "1", "SLURM_TMPDIR": str(tmp_path), "SLURM_ARRAY_TASK_ID": "1",
           "CHROMS_ARRAY": "chr1 chr21 chrX"}
    r = subprocess.run(["bash", str(sh), str(_ROOT), "precompute", "--source", "flashzoi", "--genome", "/g",
                        "--out", "/o"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = r.stdout.splitlines()
    assert out[0].startswith("pip install --no-index -r")
    assert "borzoi-pytorch==0.5.1 --hash=sha256:" in r.stdout
    assert out[-1].endswith("dna_embed.py precompute --source flashzoi --genome /g --out /o --chroms chr21")
    wh = [ln for ln in (tmp_path / "req_wheelhouse.txt").read_text().splitlines() if not ln.startswith("#")]
    assert "torch==2.9.1" in wh and not any("borzoi" in ln for ln in wh)
    py = [ln for ln in (tmp_path / "req_pypi.txt").read_text().splitlines() if ln.strip()]
    assert py == ["borzoi-pytorch==0.5.1 --hash=sha256:"
                  "a7ed251e013e1eed320a3998ddb6484e7aab7fdf81d52755ad880ae7da0f26fa"]


# ---------------------------------------------------------------- streamed fit and projection (chunk A5x)
def test_streamed_projection_equals_fit_then_project(genome, tmp_path):
    npz = tmp_path / "proj" / "projection.npz"
    E.fit(stub(), genome, npz, ["chr1"], 2, n_fit=3000, seed=2, per_window=40, batch=3, log=lambda s: None)
    with pytest.raises(SystemExit):                                   # never refitted in place
        E.fit(stub(), genome, npz, ["chr1"], 2, n_fit=3000, log=lambda s: None)
    two = tmp_path / "two"
    E.precompute(stub(), genome, two, ["chr1", "chr2"], batch=2, log=lambda s: None)
    (two / "projection.npz").write_bytes(npz.read_bytes())
    E.project(two, None, ["chr1"], 2, None, log=lambda s: None)       # reuses the copied projection
    one = tmp_path / "one"
    E.precompute(stub(), genome, one, ["chr1", "chr2"], batch=5, log=lambda s: None, projection=npz)
    for c in CHROMS:
        with h5py.File(one / f"{c}.h5") as a, h5py.File(two / f"{c}.h5") as b:
            assert a["emb"].shape == (CHROMS[c] // 25, 2) and a["emb"].dtype == np.float16
            np.testing.assert_allclose(a["emb"][:].astype(np.float32), b["emb"][:].astype(np.float32), atol=1e-3)
            assert a.attrs["projected_dim"] == 2 and a.attrs["projection_sha256"] == E.sha256_file(npz)
    assert sorted(p.name for p in one.iterdir()) == ["chr1.h5", "chr2.h5", "meta.json"]   # no intermediate
    meta = json.loads((one / "meta.json").read_text())
    assert meta["d"] == 2 and meta["d_model"] == 5 and meta["projection"]["n_fit"] == 3000
    assert E.verify(one) == (2, True, 0)
    E.precompute(stub(), genome, one, ["chr2"], log=lambda s: None, projection=npz)        # skip, same projection
    raw = tmp_path / "raw"                                            # an unprojected file is never mistaken
    E.precompute(stub(), genome, raw, ["chr2"], log=lambda s: None)
    with pytest.raises(SystemExit):
        E.precompute(stub(), genome, raw, ["chr2"], log=lambda s: None, projection=npz)


def test_streamed_fit_matches_exact_pca_and_respects_beds(genome, tmp_path):
    raw = tmp_path / "raw"
    E.precompute(stub(), genome, raw, ["chr1"], log=lambda s: None)
    with h5py.File(raw / "chr1.h5") as h, h5py.File(genome / "mask.h5") as m:
        x = h["emb"][:].astype(np.float64)[m["chr1"][:] > 0]
    p = E.fit_streamed(stub(), genome, ["chr1"], 2, n_fit=10 ** 9, per_window=10 ** 9, log=lambda s: None)
    assert p["n_fit"] == x.shape[0]                                   # every eligible bin, exactly once
    np.testing.assert_allclose(p["mean"], x.mean(0), atol=1e-6)
    ev, v = np.linalg.eigh(np.cov(x.T))
    top = v[:, np.argsort(ev)[::-1][:2]].T
    np.testing.assert_allclose(np.abs(p["components"] @ top.T), np.eye(2), atol=1e-4)
    nb = CHROMS["chr1"] // 25
    inc, exc = tmp_path / "inc.bed", tmp_path / "exc.bed"
    inc.write_text("chr1\t0\t100000\tfold0\nchr1\t50000\t150000\tfold1\n")      # overlapping intervals
    exc.write_text("chr1\t120000\t130000\tfold3\n")
    q = E.fit_streamed(stub(), genome, ["chr1"], 2, n_fit=10 ** 9, per_window=10 ** 9, fit_bed=str(inc),
                       exclude_bed=str(exc), log=lambda s: None)
    centre = 25 * np.arange(nb) + 12.5
    with h5py.File(genome / "mask.h5") as m:
        want = (m["chr1"][:] > 0) & (centre < 150000) & ~((centre >= 120000) & (centre < 130000))
    assert q["n_fit"] == int(want.sum()) and q["fit_bed_md5"] and q["exclude_bed_md5"]
    np.testing.assert_allclose(q["mean"], raw_mean(raw, want), atol=1e-6)
    r = E.fit_streamed(stub(), genome, ["chr1"], 2, n_fit=500, per_window=30, seed=4, log=lambda s: None)
    assert r["n_fit"] == 500 and r["n_windows"] == -(-500 // 30)


def raw_mean(raw: Path, rows: np.ndarray) -> np.ndarray:
    with h5py.File(raw / "chr1.h5") as h:
        return h["emb"][:].astype(np.float64)[rows].mean(0)


def test_bins_in_centre_rule():
    bed = {"chrA": (np.array([0, 30]), np.array([13, 60]))}
    # bin centres 12.5, 37.5, 62.5: [0,13) holds 12.5; [30,60) holds 37.5; nothing holds 62.5
    assert E.bins_in(bed, "chrA", 3).tolist() == [True, True, False]
    assert E.bins_in(bed, "chrB", 2).tolist() == [False, False]


def test_cli_fit_and_projected_precompute(genome, tmp_path, capsys):
    npz = tmp_path / "p.npz"
    assert E.main(["fit", "--source", "flashzoi", "--genome", str(genome), "--out", str(npz), "--fit-chroms", "chr1",
                   "--dim", "3", "--n-fit", "2000", "--device", "cpu"], source_factory=lambda *a: stub()) == 0
    out = tmp_path / "o"
    assert E.main(["precompute", "--source", "flashzoi", "--genome", str(genome), "--out", str(out), "--chroms",
                   "chr2", "--device", "cpu", "--projection", str(npz)], source_factory=lambda *a: stub()) == 0
    with h5py.File(out / "chr2.h5") as h:
        assert h["emb"].shape == (CHROMS["chr2"] // 25, 3)
