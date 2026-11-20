"""Training regions, exclusion and per-window loss weights (pinned interface I, chunk TR).

All on `minicandi.synth` tables (chr1 2,048 bins, chr2 1,024 bins; mask 0 on the first 10 bins of
each chromosome). Without the two keys the batches, `meta`, the generator stream and the loss equal
the code before chunk TR: the digests and loss values below were computed on the base af4ee30.
"""
from __future__ import annotations

import hashlib
import json
import math
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
from minicandi import losses as LS  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi import train as T  # noqa: E402

L = 64
HEADER = "# roles_read=train\n# universe=test\n# sha256_inputs=0\n"

#: base af4ee30: sha256 of 40 batches (dataset seed 5, generator default_rng([5, 1])), and the
#: generator's next integer after them
REFERENCE = {
    "plain": ({"context_bins": L},
              "bd15e51ef7747ac39078161b7ad31a92e4e198121bd8f86f71747944e9ac67da",
              1100622906585187420),
    "two_step_revcomp": ({"context_bins": L, "batch_size": 4, "two_step": True,
                          "aug_revcomp": True},
                         "950ab06d0d4b6a7e3c4448e082c3e63f9a65448f89e91f6eb60596039ac9273a",
                         2956069215334489786),
}
#: base af4ee30: total_loss on the fixed output of `_fixed_out` (float.hex), per machine: the last
#: bit differs between the laptop (arm64, macOS) and Nibi (x86_64); each column was computed with
#: the base code on that machine (Nibi: kit TRbase, git af4ee30)
LOSS_CONFIGS = [{}, {"loss_shape_weight": 0.5}, {"loss_multiscale": [4, 16]},
                {"loss_assay_weight": "equal"}, {"loss_assay_weight": "learned"},
                {"loss_bin_equal": True}, {"loss": "nb_crps"},
                {"loss_weights": {"impute": 2.0, "base": 0.5}}]
LOSS_REFERENCE = {
    "arm64": ["0x1.2f16afd98f1d5p+2", "0x1.83b50c10f5e95p+2", "0x1.432dbbf37dcc3p+4",
              "0x1.bdb9721522356p+1", "0x1.7cb3fc4b8b01cp+2", "0x1.64211d83abf75p+2",
              "0x1.14a23a09c544ep+2", "0x1.182610d624e3ap+2"],
    "x86_64": ["0x1.2f16afd98f1d4p+2", "0x1.83b50c10f5e95p+2", "0x1.432dbbf37dcc2p+4",
               "0x1.bdb9721522355p+1", "0x1.7cb3fc4b8b01cp+2", "0x1.64211d83abf74p+2",
               "0x1.14a23a09c544ep+2", "0x1.182610d624e39p+2"],
}


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> Path:
    return synth.make_synthetic_tables(tmp_path_factory.mktemp("onion_tr"))


@pytest.fixture(scope="module")
def half_masked(tmp_path_factory, tables) -> Path:
    """A copy of the tables whose chr1 mask is 0 on bins [1128, 1256): in the row chr1:1000+256
    the starts 1000..1096 pass the mask test at L = 64 and the starts 1097..1192 fail (half)."""
    root = tmp_path_factory.mktemp("onion_tr_half")
    src = tables.parent.parent
    shutil.copytree(src, root, dirs_exist_ok=True)
    td = root / tables.relative_to(src)
    with h5py.File(root / "genome" / "mask.h5", "r+") as f:
        m = f["chr1"][:]
        m[1128:1256] = 0
        f["chr1"][...] = m
    return td


def _digest(batches) -> str:
    h = hashlib.sha256()
    for b in batches:
        for k in sorted(b):
            h.update(k.encode())
            if k == "meta":
                h.update(json.dumps(b[k], sort_keys=True).encode())
            else:
                t = b[k].contiguous()
                h.update(str(t.dtype).encode() + str(tuple(t.shape)).encode())
                h.update(t.numpy().tobytes())
    return h.hexdigest()


def _write_list(path: Path, rows, header: str = HEADER) -> Path:
    lines = [header + "\t".join(D.REGION_LIST_COLUMNS)]
    lines += ["\t".join(str(x) for x in r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _chi2_crit(df: int, z: float = 3.719) -> float:
    """Wilson-Hilferty upper quantile of chi-square(df) at the normal quantile z (3.719: 1e-4)."""
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


def _chi2(obs, exp) -> float:
    obs, exp = np.asarray(obs, float), np.asarray(exp, float)
    return float(((obs - exp) ** 2 / exp).sum())


# ------------------------------------------------------------------------------- defaults


@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_without_the_keys_batches_and_stream_equal_the_base(tables, name):
    cfg, want, nxt = REFERENCE[name]
    ds = D.OnionDataset(tables, "train", cfg, seed=5)
    rng = np.random.default_rng([5, 1])
    batches = [ds.batch(int(ds.config["batch_size"]), rng) for _ in range(40)]
    assert _digest(batches) == want
    assert int(rng.integers(1 << 62)) == nxt          # plan() drew no extra random number
    assert "w_window" not in batches[0]
    assert all("stratum" not in m for m in batches[0]["meta"])
    assert ds.region_record() == {}
    p = ds.plan(np.random.default_rng(0))
    assert set(p) == {"cell", "chrom", "start_bin", "entries"}


def _fixed_out(tables):
    ds = D.OnionDataset(tables, "train", {"context_bins": L, "two_step": True}, seed=5)
    b = ds.batch(6, np.random.default_rng(1))
    g = torch.Generator().manual_seed(0)
    shape = b["y_counts"].shape
    out = {"mu": torch.rand(shape, generator=g, dtype=torch.float64) * 5 + 0.1,
           "n": torch.rand(shape, generator=g, dtype=torch.float64) * 3 + 0.2}
    out["mu_base"] = out["mu"] * 1.3
    out["n_base"] = out["n"] * 0.7
    out["assay_log_w"] = torch.rand(shape[-1], generator=g, dtype=torch.float64)
    return out, b


@pytest.mark.parametrize("i", range(len(LOSS_CONFIGS)))
def test_without_w_window_the_loss_equals_the_base(tables, i):
    out, b = _fixed_out(tables)
    got = float(LS.total_loss(out, b, LOSS_CONFIGS[i])[0])
    ref = LOSS_REFERENCE.get(platform.machine())
    if ref is not None:
        assert got.hex() == ref[i]
    else:                                   # another machine: within the last bits
        assert got == pytest.approx(float.fromhex(LOSS_REFERENCE["x86_64"][i]), rel=1e-13)


# ------------------------------------------------------------------------------- refusals


def test_both_keys_together_are_refused(tables, tmp_path):
    lst = _write_list(tmp_path / "l.tsv", [("chr1", 100, 512, "a", 1.0, 1.0)])
    ex = tmp_path / "ex.json"
    ex.write_text(json.dumps({"regions": [], "fit_regions": [], "region_bins": 256}))
    with pytest.raises(ValueError, match="together"):
        D.OnionDataset(tables, "train", {"context_bins": L, "train_regions": str(lst),
                                         "exclude_regions": str(ex)})


def test_a_list_without_the_roles_line_is_refused(tables, tmp_path):
    lst = _write_list(tmp_path / "l.tsv", [("chr1", 100, 512, "a", 1.0, 1.0)],
                      header="# universe=test\n# sha256_inputs=0\n")
    with pytest.raises(D.RoleError, match="roles_read=train"):
        D.OnionDataset(tables, "train", {"context_bins": L, "train_regions": str(lst)})
    lst = _write_list(tmp_path / "v.tsv", [("chr1", 100, 512, "a", 1.0, 1.0)],
                      header="# roles_read=train,val\n")
    with pytest.raises(D.RoleError):
        D.OnionDataset(tables, "train", {"context_bins": L, "train_regions": str(lst)})


def test_a_row_shorter_than_the_window_is_refused(tables, tmp_path):
    lst = _write_list(tmp_path / "l.tsv", [("chr1", 100, 512, "a", 0.5, 1.0),
                                           ("chr2", 100, L - 1, "b", 0.5, 1.0)])
    with pytest.raises(ValueError, match="n_bins"):
        D.OnionDataset(tables, "train", {"context_bins": L, "train_regions": str(lst)})


def test_seed_resolves_in_the_path(tables, tmp_path):
    _write_list(tmp_path / "terr_s0.tsv", [("chr1", 0, 256, "s0", 1.0, 1.0)])
    _write_list(tmp_path / "terr_s7.tsv", [("chr2", 512, 256, "s7", 1.0, 1.0)])
    cfg = {"context_bins": L, "train_regions": str(tmp_path / "terr_s{seed}.tsv")}
    for seed, chrom in ((0, "chr1"), (7, "chr2")):
        ds = D.OnionDataset(tables, "train", cfg, seed=seed)
        rec = ds.region_record()
        assert rec["train_regions"].endswith(f"terr_s{seed}.tsv")
        assert rec["train_regions_sha256"] == hashlib.sha256(
            (tmp_path / f"terr_s{seed}.tsv").read_bytes()).hexdigest()
        rng = np.random.default_rng(1)
        assert {ds.plan(rng)["chrom"] for _ in range(50)} == {chrom}


# ------------------------------------------------------------------------------- train_regions


ROWS = [("chr1", 20, 300, "a", 0.1, 2.0), ("chr1", 1000, 256, "b", 0.2, 0.5),
        ("chr2", 300, 200, "c", 0.3, 1.0), ("chr2", 700, 64, "d", 0.4, 1.25)]


def _row_draws(td, tmp_path, n):
    lst = _write_list(tmp_path / "rows.tsv", ROWS)
    ds = D.OnionDataset(td, "train", {"context_bins": L, "train_regions": str(lst)}, seed=0)
    rng = np.random.default_rng(11)
    return ds, [ds.plan(rng) for _ in range(n)]


def test_every_window_lies_inside_a_listed_row_and_carries_its_weight(tables, tmp_path):
    ds, plans = _row_draws(tables, tmp_path, 3000)
    for p in plans:
        inside = [r for r in ROWS if r[0] == p["chrom"] and r[1] <= p["start_bin"]
                  and p["start_bin"] + L <= r[1] + r[2]]
        assert inside, p
        assert p["stratum"] == inside[0][3] and p["loss_weight"] == inside[0][5]
    b = ds.batch(5, np.random.default_rng(2))
    assert b["w_window"].dtype == torch.float32 and tuple(b["w_window"].shape) == (5,)
    want = {r[3]: r[5] for r in ROWS}
    assert b["w_window"].tolist() == [want[m["stratum"]] for m in b["meta"]]


@pytest.mark.parametrize("which", ["full_mask", "half_starts_fail"])
def test_row_frequencies_follow_sample_weight(tables, half_masked, tmp_path, which):
    td = tables if which == "full_mask" else half_masked
    ds, plans = _row_draws(td, tmp_path, 20_000)
    obs = [sum(p["stratum"] == r[3] for p in plans) for r in ROWS]
    exp = [20_000 * r[4] for r in ROWS]
    assert _chi2(obs, exp) < _chi2_crit(len(ROWS) - 1), (obs, exp)
    if which == "half_starts_fail":
        starts = np.asarray([p["start_bin"] for p in plans if p["stratum"] == "b"])
        assert (starts <= 1096).mean() > 0.99          # failed starts were redrawn in the row


def test_weighted_tilted_draw_has_the_uniform_mean_loss(tables, tmp_path):
    """Planted two-stratum case: windows on chr2 get a far worse fake prediction than on chr1.
    Uniform: q in proportion to the starts of each row, w = 1. Tilted: q = 1/2 each, w = g/q."""
    rows = [("chr1", 0, 1600), ("chr2", 0, 400)]
    starts = np.asarray([n - L + 1 for _, _, n in rows], float)
    g = starts / starts.sum()
    uni = _write_list(tmp_path / "uni.tsv", [(c, s, n, c, g[i], 1.0)
                                             for i, (c, s, n) in enumerate(rows)])
    tw = _write_list(tmp_path / "tw.tsv", [(c, s, n, c, 0.5, g[i] / 0.5)
                                           for i, (c, s, n) in enumerate(rows)])
    tu = _write_list(tmp_path / "tu.tsv", [(c, s, n, c, 0.5, 1.0)
                                           for i, (c, s, n) in enumerate(rows)])

    def losses(path, n=1500, seed=0):
        ds = D.OnionDataset(tables, "train", {"context_bins": L, "train_regions": str(path)},
                            seed=seed)
        rng = np.random.default_rng([seed, 1])
        out = []
        for _ in range(n):
            b = ds.batch(1, rng)
            if not C.loss_mask(b).any():
                continue
            mu = torch.full(b["y_counts"].shape, 1.0 if b["meta"][0]["chrom"] == "chr1" else 40.0)
            o = {"mu": mu, "n": torch.full_like(mu, 2.0)}
            out.append(float(LS.total_loss(o, b, {})[0]))
        return np.asarray(out)

    u, w, t = losses(uni), losses(tw, seed=1), losses(tu, seed=2)
    se = math.sqrt(u.var() / len(u) + w.var() / len(w))
    assert abs(w.mean() - u.mean()) < 4 * se, (u.mean(), w.mean(), se)
    se_t = math.sqrt(u.var() / len(u) + t.var() / len(t))
    assert abs(t.mean() - u.mean()) > 8 * se_t, (u.mean(), t.mean(), se_t)


def _weighted_batch(tables, tmp_path):
    lst = _write_list(tmp_path / "w.tsv", [("chr1", 0, 2048, "a", 1.0, 1.0)])
    ds = D.OnionDataset(tables, "train", {"context_bins": L, "two_step": True,
                                          "train_regions": str(lst)}, seed=0)
    b = ds.batch(4, np.random.default_rng(3))
    g = torch.Generator().manual_seed(1)
    shape = b["y_counts"].shape
    out = {"mu": torch.rand(shape, generator=g, dtype=torch.float64) * 5 + 0.1,
           "n": torch.rand(shape, generator=g, dtype=torch.float64) * 3 + 0.2}
    out["mu_base"], out["n_base"] = out["mu"] * 1.3, out["n"] * 0.7
    return out, b


@pytest.mark.parametrize("part,cfg", [("main", {}),
                                      ("main", {"loss_assay_weight": "learned"}),
                                      ("base", {}),
                                      ("shape", {"loss_shape_weight": 1.0}),
                                      ("multiscale_4", {"loss_multiscale": [4]})])
def test_the_weight_reaches_each_term(tables, tmp_path, part, cfg):
    out, b = _weighted_batch(tables, tmp_path)
    if cfg.get("loss_assay_weight") == "learned":
        out["assay_log_w"] = torch.zeros(C.A, dtype=torch.float64)
    B = b["w_window"].shape[0]

    def term(w):
        return LS.total_loss(out, dict(b, w_window=torch.as_tensor(w, dtype=torch.float32)),
                             cfg)[1][part]

    plain = LS.total_loss(out, {k: v for k, v in b.items() if k != "w_window"}, cfg)[1][part]
    assert term([1.0] * B) == pytest.approx(plain, rel=1e-12)
    w = [0.5, 2.0, 0.0, 3.0][:B]
    ones = np.eye(B)
    singles = [term(list(ones[j])) for j in range(B)]
    assert term(w) == pytest.approx(sum(wj * sj for wj, sj in zip(w, singles)), rel=1e-9)
    assert len({round(s, 9) for s in singles}) > 1      # the windows' shares differ


# ------------------------------------------------------------------------------- exclusion


def _exclusion(tmp_path):
    spec = {"regions": [["chr1", 512], ["chr2", 256]], "fit_regions": [["chr1", 1280]],
            "region_bins": 256}
    p = tmp_path / "regions.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    blocked = {"chr1": [(512, 768), (1280, 1536)], "chr2": [(256, 512)]}
    return p, blocked


def test_exclusion_never_overlaps_and_the_rest_is_uniform(tables, tmp_path):
    p, blocked = _exclusion(tmp_path)
    ds = D.OnionDataset(tables, "train", {"context_bins": L, "exclude_regions": str(p)}, seed=0)
    assert ds.region_record()["exclude_regions_sha256"] == hashlib.sha256(
        p.read_bytes()).hexdigest()
    rng = np.random.default_rng(4)
    plans = [ds.plan(rng) for _ in range(50_000)]
    assert "loss_weight" not in plans[0]
    for q in plans:
        s = q["start_bin"]
        assert not any(a < s + L and s < e for a, e in blocked[q["chrom"]]), q
    # uniform over the allowed starts (the synth mask passes every L = 64 window): chi-square
    # over groups of 128 starts
    obs, exp = {}, {}
    allowed = {c: [s for s in range(ds.n_bins[c] - L + 1)
                   if not any(a < s + L and s < e for a, e in blocked[c])] for c in ds.chroms}
    n_allowed = sum(len(v) for v in allowed.values())
    for c, ss in allowed.items():
        for s in ss:
            exp[(c, s // 128)] = exp.get((c, s // 128), 0) + 50_000 / n_allowed
    for q in plans:
        k = (q["chrom"], q["start_bin"] // 128)
        obs[k] = obs.get(k, 0) + 1
    keys = sorted(exp)
    assert set(obs) <= set(exp)
    assert _chi2([obs.get(k, 0) for k in keys], [exp[k] for k in keys]) < _chi2_crit(len(keys) - 1)


def test_exclusion_stream_is_deterministic_per_seed(tables, tmp_path):
    p, _ = _exclusion(tmp_path)
    cfg = {"context_bins": L, "exclude_regions": str(p)}

    def run(seed):
        ds = D.OnionDataset(tables, "train", cfg, seed=seed)
        it = iter(ds)
        return _digest(next(it) for _ in range(20))

    assert run(3) == run(3)
    assert run(3) != run(4)
    plain = D.OnionDataset(tables, "train", {"context_bins": L}, seed=3)
    assert _digest(next(iter(plain)) for _ in range(1)) != run(3)


def test_exclusion_of_everything_raises(tables, tmp_path):
    spec = {"regions": [["chr1", 0], ["chr2", 0]], "fit_regions": [], "region_bins": 2048}
    p = tmp_path / "all.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    ds = D.OnionDataset(tables, "train", {"context_bins": L, "exclude_regions": str(p)}, seed=0)
    with pytest.raises(RuntimeError, match="free of the"):
        ds.plan(np.random.default_rng(0))


# ------------------------------------------------------------------------------- train loop


def test_train_records_the_file_and_the_windows_per_stratum(tables, tmp_path):
    lst = _write_list(tmp_path / "t.tsv", [("chr1", 0, 1024, "a", 0.5, 1.0),
                                           ("chr2", 0, 1024, "b", 0.5, 1.0)])
    cfg = {"rung": 1, "context_bins": L, "batch_size": 4, "param_budget": 1.4e5, "lr": 3e-3,
           "dna": "own", "windows_seen": 16, "train_regions": str(lst)}
    T.train(cfg, tables, 0, tmp_path / "run", max_windows=8, log_every=1)
    info = T.train(cfg, tables, 0, tmp_path / "run", log_every=1)
    assert info["train_regions_sha256"] == hashlib.sha256(lst.read_bytes()).hexdigest()
    logs = [json.loads(x) for x in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    last = [r for r in logs if "windows_per_stratum" in r][-1]
    assert sum(last["windows_per_stratum"].values()) == 16
    assert set(last["windows_per_stratum"]) <= {"a", "b"}


TINY = {"rung": 1, "context_bins": L, "batch_size": 4, "param_budget": 1.4e5, "lr": 3e-3,
        "dna": "own"}


def _log(run: Path) -> list:
    return [json.loads(x) for x in (run / "log.jsonl").read_text().splitlines()]


def _same_log_line(a: dict, b: dict) -> dict:
    return {k: v for k, v in a.items() if k != "wall_seconds"} == \
        {k: v for k, v in b.items() if k != "wall_seconds"}


def test_train_regions_resume_is_bit_exact(tables, tmp_path):
    lst = _write_list(tmp_path / "t.tsv", [("chr1", 0, 1024, "a", 0.3, 2.0),
                                           ("chr2", 0, 1024, "b", 0.7, 4.0 / 7.0)])
    cfg = dict(TINY, windows_seen=32, train_regions=str(lst), loss_shape_weight=0.5)
    T.train(cfg, tables, 3, tmp_path / "a", log_every=1)
    T.train(cfg, tables, 3, tmp_path / "b", max_windows=16, log_every=1)
    T.train(cfg, tables, 3, tmp_path / "b", log_every=1)
    la, lb = _log(tmp_path / "a"), _log(tmp_path / "b")
    assert [r["step"] for r in lb] == list(range(1, 9))
    assert all(_same_log_line(x, y) for x, y in zip(la, lb))   # losses, parts, strata per step
    assert sum(lb[-1]["windows_per_stratum"].values()) == 32
    sa = torch.load(tmp_path / "a" / "ckpt.pt", weights_only=False)["state_dict"]
    sb = torch.load(tmp_path / "b" / "ckpt.pt", weights_only=False)["state_dict"]
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_train_regions_stream_is_deterministic_per_seed(tables, tmp_path):
    lst = _write_list(tmp_path / "t.tsv", ROWS)
    cfg = {"context_bins": L, "batch_size": 4, "two_step": True, "train_regions": str(lst)}

    def run(seed):
        it = iter(D.OnionDataset(tables, "train", cfg, seed=seed))
        return _digest(next(it) for _ in range(20))

    assert run(2) == run(2)
    assert run(2) != run(3)


def test_resume_refuses_a_changed_region_file(tables, tmp_path):
    lst = _write_list(tmp_path / "t.tsv", [("chr1", 0, 1024, "a", 1.0, 1.0)])
    cfg = dict(TINY, windows_seen=16, train_regions=str(lst))
    T.train(cfg, tables, 0, tmp_path / "r", max_windows=8)
    _write_list(lst, [("chr1", 0, 1000, "a", 1.0, 1.0)])          # same path, new rows
    with pytest.raises(ValueError, match="train_regions_sha256"):
        T.train(cfg, tables, 0, tmp_path / "r")


def test_exclusion_rejections_are_counted_and_logged(tables, tmp_path):
    p, blocked = _exclusion(tmp_path)
    ds = D.OnionDataset(tables, "train", {"context_bins": L, "exclude_regions": str(p)}, seed=0)
    rng = np.random.default_rng(5)
    plans = [ds.plan(rng) for _ in range(20_000)]
    rej = sum(q["exclude_rejections"] for q in plans)
    n_all = sum(ds.n_bins[c] - L + 1 for c in ds.chroms)
    n_bad = sum(1 for c in ds.chroms for s in range(ds.n_bins[c] - L + 1)
                if any(a < s + L and s < e for a, e in blocked[c]))
    rate = rej / (rej + len(plans))           # the mask passes every L = 64 window here
    se = math.sqrt(n_bad / n_all * (1 - n_bad / n_all) / (rej + len(plans)))
    assert abs(rate - n_bad / n_all) < 5 * se, (rate, n_bad / n_all)
    b = ds.batch(3, np.random.default_rng(0))
    assert all("exclude_rejections" in m for m in b["meta"]) and "w_window" not in b
    cfg = dict(TINY, windows_seen=8, exclude_regions=str(p))
    T.train(cfg, tables, 0, tmp_path / "x", log_every=1)
    last = _log(tmp_path / "x")[-1]
    assert last["exclude_rejections"] >= 0 and 0 <= last["exclude_rejection_rate"] < 1
    T.train(dict(TINY, windows_seen=8), tables, 0, tmp_path / "plain", log_every=1)
    assert all("exclude_rejections" not in r and "windows_per_stratum" not in r
               for r in _log(tmp_path / "plain"))
