"""miniCANDI extra inputs in the dataset, the contract and the six rungs (Amendment 3, chunk XI).

Part 2 (below the digests): the extra inputs of pinned interface E. The RNA-seq files are written
by the writer of chunk XRd (`rna_inputs.write_track`, `strand_arrays`, `build_vector`) and read
back by `extra_data` (a round trip), the dataset emits each key only under its config key, every
rung builds and trains every arm or raises NotApplicable (the support matrix is printed), a model
with the keys starts as the model without them, and the copy-leak wiring check (guardrail part 1)
passes on every arm.

Part 1, the default digests. `tests/fixtures/minicandi_default_digests.json` was written on the
tree BEFORE this chunk's edits (`python tests/test_minicandi_extra_integration.py --write` in a
checkout of the base commit 5a975dc, once on the laptop and once on the cluster): per rung 1 to 6 under its `base.json` (data shape
`context_bins` = 64 and `batch_size` = 2, so the laptop runs it in seconds; no network key is
changed), on the `synth.py` tables, seed 0, it holds the sha256 of the fixed training batch, of
the initial `state_dict`, of the eval-mode forward output on that batch, and the loss of the
first 5 training steps (AdamW and clipping as `train.py`, constant lr), and the sha256 of the
forward output on the fixed batch after those steps (the initial output is the same for every rung,
since the NB head starts at zero; the trained output is not). `test_default_digest`
demands the same values now: exact for the digests, 0 difference for the losses. Float results
depend on the platform (BLAS kernels), so the fixture keeps one entry per platform signature and
the test reads the entry of the platform it runs on.
"""
from __future__ import annotations

import hashlib
import json
import platform
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
from minicandi import synth  # noqa: E402
from minicandi import train as T  # noqa: E402
from minicandi.components import NotApplicable  # noqa: E402
from minicandi.dataset import OnionDataset  # noqa: E402
from minicandi.extra_data import Expression, RnaTracks, cell_index  # noqa: E402
from onion_data import rna_inputs as RI  # noqa: E402
from onion_eval import guardrail as G  # noqa: E402

FIXTURE = _ROOT / "tests" / "fixtures" / "minicandi_default_digests.json"
DIGEST_DATA = {"context_bins": 64, "batch_size": 2}
DIGEST_SEED = 0
DIGEST_STEPS = 5
RUNGS = sorted(R.RUNGS)


def platform_key() -> str:
    return f"{platform.system()}-{platform.machine()}-torch{torch.__version__}"


def base_config(r: int) -> dict:
    return json.loads((_ROOT / f"tools/minicandi/configs/rung{r}/base.json").read_text())


def digest_config(r: int) -> dict:
    return {**base_config(r), **DIGEST_DATA}


def _sha_tensors(items) -> str:
    h = hashlib.sha256()
    for name, t in items:
        t = t.detach().cpu().contiguous()
        h.update(f"{name}|{t.dtype}|{tuple(t.shape)}".encode())
        h.update(t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.float().numpy().tobytes())
    return h.hexdigest()


def batch_sha(batch: dict) -> str:
    items = [(k, batch[k]) for k in sorted(batch) if torch.is_tensor(batch[k])]
    return _sha_tensors(items) + ":" + hashlib.sha256(
        json.dumps(batch["meta"], sort_keys=True, default=str).encode()).hexdigest()[:16]


def rung_digest(r: int, tables) -> dict:
    """The digest of rung r (module docstring), single-threaded."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = digest_config(r)
        bs = int(cfg["batch_size"])
        ds = OnionDataset(tables, "train", cfg, DIGEST_SEED)
        fixed = ds.batch(bs, np.random.default_rng([DIGEST_SEED, 2]))
        torch.manual_seed(DIGEST_SEED)
        model = R.build_model(cfg)
        sd = _sha_tensors(list(model.state_dict().items()))
        model.eval()
        with torch.no_grad():
            out = model(fixed)
        fwd = _sha_tensors([(k, out[k]) for k in sorted(out)])
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                                weight_decay=T.WEIGHT_DECAY)
        rng = np.random.default_rng([DIGEST_SEED, 1])
        losses = []
        for _ in range(DIGEST_STEPS):
            b = ds.batch(bs, rng)
            loss = T.step_loss(model, b, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), T.CLIP)
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            out = model(fixed)
        trained = _sha_tensors([(k, out[k]) for k in sorted(out)])
        return {"batch_sha256": batch_sha(fixed), "state_dict_sha256": sd,
                "forward_sha256": fwd, "forward_trained_sha256": trained,
                "param_count": model.param_count(), "losses": losses}
    finally:
        torch.set_num_threads(threads)


def all_digests(tmp) -> dict:
    tables = synth.make_synthetic_tables(Path(tmp))
    return {str(r): rung_digest(r, tables) for r in RUNGS}


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> Path:
    return synth.make_synthetic_tables(tmp_path_factory.mktemp("xi_digest"))


@pytest.mark.parametrize("r", RUNGS)
def test_default_digest(r, tables):
    fx = json.loads(FIXTURE.read_text(encoding="utf-8"))
    key = platform_key()
    if key not in fx["platforms"]:
        pytest.fail(f"no default digests recorded for platform {key!r}; recorded: "
                    f"{sorted(fx['platforms'])}")
    want = fx["platforms"][key][str(r)]
    got = rung_digest(r, tables)
    for k in ("batch_sha256", "state_dict_sha256", "forward_sha256", "forward_trained_sha256",
              "param_count"):
        assert got[k] == want[k], (r, k)
    diff = [abs(a - b) for a, b in zip(got["losses"], want["losses"])]
    assert len(got["losses"]) == len(want["losses"]) and max(diff) == 0.0, (r, diff)


def _write(argv) -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = all_digests(tmp)
    fx = (json.loads(FIXTURE.read_text(encoding="utf-8")) if FIXTURE.is_file()
          else {"doc": "see tests/test_minicandi_extra_integration.py", "data": DIGEST_DATA,
                "seed": DIGEST_SEED, "steps": DIGEST_STEPS, "platforms": {}})
    key = platform_key()
    if "--print" in argv:
        print(json.dumps({key: d}, indent=1, sort_keys=True))
        return 0
    fx["platforms"][key] = d
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fx, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {key} into {FIXTURE}")
    return 0


if __name__ == "__main__":
    sys.exit(_write(sys.argv[1:]))


# ------------------------------------------------------------------------- part 2: extra inputs

L = 64
N_GENES_TEST = 5
ARMS = {
    "rna_track": {"rna_track": True},
    "cell_vector": {"cell_vector": True},
    "cell_embedding": {"cell_embedding": True},
    "aug_revcomp": {"aug_revcomp": True},
    "loss_assay_learned": {"loss_assay_weight": "learned"},
}
ALL_EXTRA = {"rna_track": True, "cell_vector": True, "cell_embedding": True}


def _sizes(tables) -> dict:
    sz = json.loads((Path(tables).parent.parent / "genome" / "chrom_sizes.json").read_text())
    return {c: int(n) // C.RES for c, n in sz.items()}


def _all_cells(tables) -> list:
    rows = RI.read_tsv(Path(tables) / "measurements.tsv")
    return sorted({r["cell"] for r in rows})


@pytest.fixture(scope="module")
def extra(tmp_path_factory, xtables) -> dict:
    """Tracks and expression of the synthetic cells, written by the XRd writer; the last cell
    gets neither, so the unavailable path is exercised."""
    root = tmp_path_factory.mktemp("xi_extra")
    cells = _all_cells(xtables)
    with_rna = cells[:-1]
    rng = np.random.default_rng(7)
    raw = {}
    tdir = root / "tracks"
    for c in with_rna:
        p = {ch: rng.gamma(1.0, 3.0, n) for ch, n in _sizes(xtables).items()}
        m = {ch: -rng.gamma(1.0, 2.0, n) for ch, n in _sizes(xtables).items()}   # signed minus
        plus, _ = RI.strand_arrays(p, 1.0)
        minus, _ = RI.strand_arrays(m, 1.0)
        RI.write_track(tdir / f"{c}.h5", c, plus, minus, {"stranded": 1})
        raw[c] = (p, m)
    files = root / "files"
    genes = [f"ENSG{i:011d}" for i in range(12)]
    gt = root / "gene_types.tsv"
    gt.write_text("gene_id\tgene_type\n" + "".join(
        f"{g}.3\t{'protein_coding' if i < 10 else 'lncRNA'}\n" for i, g in enumerate(genes)))
    rows, tpm = [], {}
    for k, c in enumerate(with_rna):
        exp, acc = f"ENCSRX{k:05d}", f"ENCFFX{k:05d}"
        (files / exp).mkdir(parents=True)
        tpm[c] = {g: float(f"{rng.gamma(1.0, 10.0):.4f}") for g in genes}
        (files / exp / f"{acc}.tsv").write_text("gene_id\tTPM\n" + "".join(
            f"{g}.{k % 3 + 1}\t{tpm[c][g]:.4f}\n" for g in genes))
        rows.append({"cell": c, "experiment": exp, "gene_quant": acc,
                     "gene_quant_url": f"https://x/{acc}.tsv", "gencode": "V29"})
    adir = root / "aux"
    RI.build_vector(rows, gt, adir, files, n=N_GENES_TEST)
    return {"tracks": tdir, "expression": adir / "expression.tsv", "cells": cells,
            "with_rna": with_rna, "raw": raw, "tpm": tpm,
            "n_cells": len(cell_index(xtables))}


@pytest.fixture(scope="module")
def xtables(tmp_path_factory) -> Path:
    td = synth.make_synthetic_tables(tmp_path_factory.mktemp("xi_extra_tables"))
    synth.make_evalset(td, n_regions=2, region_bins=4 * L)
    return td


def _cfg(r: int, extra: dict, **over) -> dict:
    return {**base_config(r), "context_bins": L, "batch_size": 2,
            "rna_h5_dir": str(extra["tracks"]), "expression_tsv": str(extra["expression"]),
            "n_genes": N_GENES_TEST, "n_cells": extra["n_cells"], **over}


# ---- the round trip writer (XRd) -> reader (XD)

def test_rna_track_round_trip(extra, xtables):
    c = extra["with_rna"][0]
    p, m = extra["raw"][c]
    x, ok = RnaTracks(extra["tracks"]).window(c, "chr1", 10, 20)
    assert ok and x.shape == (20, 2)
    want_p = np.log1p(np.abs(p["chr1"][10:30]).astype(np.float16).astype(np.float32))
    want_m = np.log1p(np.abs(m["chr1"][10:30]).astype(np.float16).astype(np.float32))
    np.testing.assert_allclose(x[:, 0], want_p, rtol=1e-6)
    np.testing.assert_allclose(x[:, 1], want_m, rtol=1e-6)
    x, ok = RnaTracks(extra["tracks"]).window(extra["cells"][-1], "chr1", 0, 8)
    assert not ok and not x.any()


def test_expression_round_trip(extra):
    ex = Expression(extra["expression"], extra["expression"].parent / "genes.tsv")
    assert ex.n_genes == N_GENES_TEST
    cells = extra["with_rna"]
    mat = np.log1p(np.array([[extra["tpm"][c][g] for g in ex.genes] for c in cells]))
    sd = mat.std(0)
    z = (mat - mat.mean(0)) / np.where(sd > 0, sd, 1.0)
    for i, c in enumerate(cells):
        v, ok = ex.vector(c)
        assert ok
        np.testing.assert_allclose(v, z[i], atol=1e-4)      # the writer prints 6 digits
    v, ok = ex.vector(extra["cells"][-1])
    assert not ok and not v.any()


# ---- the dataset

def test_dataset_emits_extra_keys_only_when_set(extra, xtables):
    rng = lambda: np.random.default_rng([0, 9])  # noqa: E731
    plain = OnionDataset(xtables, "train", {**base_config(2), "context_bins": L}, 0).batch(4, rng())
    assert not set(C.EXTRA_DTYPES) & set(plain)
    cfg = _cfg(2, extra, **ALL_EXTRA)
    b = OnionDataset(xtables, "train", cfg, 0).batch(4, rng())
    C.validate_batch(b)
    assert set(C.EXTRA_DTYPES) <= set(b)
    for k in ("x_counts", "x_dna", "y_counts", "c_in", "c_out", "x_bin_mask"):
        assert torch.equal(b[k], plain[k]), k      # the extra keys change no other key
    ids = cell_index(xtables)
    for i, mt in enumerate(b["meta"]):
        has = mt["cell"] in extra["with_rna"]
        assert bool(b["x_rna_avail"][i]) == has and bool(b["x_cell_vec_avail"][i]) == has
        assert int(b["cell_id"][i]) == ids.get(mt["cell"], 0)
        if not has:
            assert not b["x_rna"][i].any() and not b["x_cell_vec"][i].any()
    vb = next(iter(OnionDataset(xtables, "val", cfg, 0)))
    assert set(C.EXTRA_DTYPES) <= set(vb)


def test_dataset_refuses_bad_extra_configs(extra, xtables):
    with pytest.raises(ValueError, match="n_cells"):
        OnionDataset(xtables, "train", _cfg(2, extra, cell_embedding=True, n_cells=99), 0)
    with pytest.raises(ValueError, match="n_genes"):
        OnionDataset(xtables, "train", _cfg(2, extra, cell_vector=True, n_genes=7), 0)
    with pytest.raises(ValueError, match="frozen"):
        OnionDataset(xtables, "train", _cfg(2, extra, aug_revcomp=True, dna="precomputed",
                                           dna_embedding_dim=8), 0)
    with pytest.raises(ValueError, match="reverse strand"):
        R.build_model(_cfg(2, extra, aug_revcomp=True, dna="precomputed", dna_embedding_dim=8))
    with pytest.raises(ValueError, match="loss_assay_weight"):
        R.build_model(_cfg(2, extra, loss_assay_weight="bogus"))


def test_aug_revcomp_in_training_mode_only(extra, xtables):
    cfg = _cfg(2, extra, rna_track=True, aug_revcomp=True)
    ds = OnionDataset(xtables, "train", cfg, 0)
    b = ds.batch(16, np.random.default_rng(5))
    flipped = [bool(m.get("revcomp")) for m in b["meta"]]
    assert any(flipped) and not all(flipped)
    for i, mt in enumerate(b["meta"]):
        dna = torch.from_numpy(ds.dna(mt["chrom"], mt["start_bin"], L))
        rna = torch.from_numpy(RnaTracks(extra["tracks"]).window(mt["cell"], mt["chrom"],
                                                                 mt["start_bin"], L)[0])
        y = torch.from_numpy(np.stack([ds.read(m, mt["chrom"], mt["start_bin"], L)
                                       for m in mt["targets"]], 1))
        cols = [C.TRACK_INDEX[ds.rows[m]["assay"]] for m in mt["targets"]]
        if flipped[i]:
            comp = torch.where(dna < 4, 3 - dna, dna).flip(0)
            assert torch.equal(b["x_dna"][i], comp)
            assert torch.equal(b["x_rna"][i], rna.flip(0).flip(1))
            assert torch.equal(b["y_counts"][i][:, cols], y.flip(0))
        else:
            assert torch.equal(b["x_dna"][i], dna) and torch.equal(b["x_rna"][i], rna)
            assert torch.equal(b["y_counts"][i][:, cols], y)
    plain = {**cfg, "aug_revcomp": False}
    va = next(iter(OnionDataset(xtables, "val", plain, 0)))
    vb = next(iter(OnionDataset(xtables, "val", cfg, 0)))
    for k in ("x_counts", "x_dna", "x_rna", "y_counts", "x_bin_mask"):
        assert torch.equal(va[k], vb[k]), k
    assert not any(m.get("revcomp") for m in vb["meta"])


# ---- the six rungs

def _support(r: int, extra: dict, arm: dict) -> str:
    try:
        torch.manual_seed(0)
        R.build_model(_cfg(r, extra, **arm))
        return "yes"
    except NotApplicable:
        return "NotApplicable"


def test_support_matrix(extra, capsys):
    rows = []
    for dna in ("own", "off"):
        for r in RUNGS:
            cells = [_support(r, extra, {**arm, "dna": dna}) for arm in ARMS.values()]
            rows.append(f"rung {r} dna={dna:<4} " + " ".join(f"{c:>14}" for c in cells))
    with capsys.disabled():
        print("\nsupport matrix (arms: " + ", ".join(ARMS) + ")\n" + "\n".join(rows))
    assert all("NotApplicable" not in row for row in rows if "dna=own" in row)


@pytest.mark.parametrize("r", RUNGS)
@pytest.mark.parametrize("arm", list(ARMS))
def test_every_rung_trains_every_arm(r, arm, extra, xtables):
    cfg = _cfg(r, extra, **ARMS[arm])
    torch.manual_seed(0)
    model = R.build_model(cfg)
    ds = OnionDataset(xtables, "train", cfg, 0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(1)
    model.train()
    for _ in range(2):
        b = ds.batch(2, rng)
        loss = T.step_loss(model, b, cfg)
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    out = model(b)
    if arm == "loss_assay_learned":
        assert out["assay_log_w"].shape == (C.A,) and "assay_log_w" in model.state_dict()
    else:
        assert "assay_log_w" not in out


@pytest.mark.parametrize("r", RUNGS)
def test_extra_inputs_start_as_the_model_without_them(r, extra, xtables):
    cfg0 = _cfg(r, extra)
    cfg1 = _cfg(r, extra, **ALL_EXTRA)
    b = OnionDataset(xtables, "train", cfg1, 0).batch(2, np.random.default_rng(3))
    torch.manual_seed(0)
    m0 = R.build_model(cfg0).eval()
    torch.manual_seed(0)
    m1 = R.build_model(cfg1).eval()
    sd0, sd1 = m0.state_dict(), m1.state_dict()
    for k, v in sd0.items():   # every default parameter is there, with the same value
        k1 = k if k in sd1 else k.replace(".embed.", ".embed.base.")
        assert torch.equal(v, sd1[k1]), k
    assert m1.param_count() > m0.param_count()
    with torch.no_grad():
        o0, o1 = m0(b), m1(b)
    assert torch.equal(o0["mu"], o1["mu"]) and torch.equal(o0["n"], o1["n"])


def _perturb(model, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model.eval()


@pytest.mark.parametrize("r", RUNGS)
@pytest.mark.parametrize("arm", list(ARMS) + ["all"])
def test_guardrail_part1_passes_on_every_arm(r, arm, extra, xtables):
    over = ALL_EXTRA if arm == "all" else ARMS[arm]
    cfg = _cfg(r, extra, **over)
    torch.manual_seed(0)
    model = _perturb(R.build_model(cfg))
    ds = OnionDataset(xtables, "train", cfg, 0)
    rng = np.random.default_rng(4)
    batches = [ds.batch(2, rng) for _ in range(2)]
    res = G.copy_leak(model, batches, seed=0)
    assert res["part1_wiring_pass"], res
    if arm in ("rna_track", "cell_vector", "cell_embedding"):   # the new input reaches the output
        b = batches[0]
        alt = dict(b)
        key = {"rna_track": "x_rna", "cell_vector": "x_cell_vec", "cell_embedding": "cell_id"}[arm]
        alt[key] = (b[key] + 1) if arm != "cell_embedding" else torch.ones_like(b[key])
        with torch.no_grad():
            assert not torch.equal(model(b)["mu"], model(alt)["mu"])
