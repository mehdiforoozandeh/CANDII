"""AGd: the scorer's dump hook writes method track files (plan Amendment 5, pinned interface F).

Pins:
  - without the option, scores.json, suffstats.npz and the blind-ledger rows are byte-identical (after
    the tmp root path and the wall-clock keys are normalized) to digests taken on the base commit
    b647359, one set per platform (the last float bits differ between the laptop and the Nibi nodes);
  - with the option, scores.json and suffstats.npz equal a run without it at the same lambda source;
  - the arrays of interface F per kind (run, NB baseline, external), checked against values computed
    here from the h5 files by a second route; no mask array; the `tracks.json` keys;
  - `--dump-targets` limits the dump; a blind dump happens only inside a ledgered blind pass;
  - the option reaches the scorer from score.py, external_signal.py and baseline_retrieval.py."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from minicandi.contract import MiniCandi  # noqa: E402
from minicandi.dataset import tables_sha256  # noqa: E402
from minicandi.synth import make_synthetic_tables  # noqa: E402
from onion_eval import evalset as E  # noqa: E402

VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (2, "H3K4me3"), (3, "DNase-seq"), (4, "DNase-seq"),
       (5, "DNase-seq"), (7, "chipseq-control"))
BLIND = ((8, "H3K4me3"),)
CFG = {"context_bins": 128, "batch_size": 4}
KW = dict(n_boot=10, guardrail_windows=4, log=lambda *a: None)
RB = 256
#: wall-clock keys of scores.json, and the eval set hash (it hashes the tmp tables path)
VOLATILE = ("wall_seconds", "timing_seconds", "torch_threads", "evalset_sha256")
#: digests of the normalized outputs of `recipes`, taken on the base commit b647359 (before AGd);
#: darwin: the laptop (candii env); linux: Nibi (kit venv of b647359, check job 27085822)
_DIGESTS = {
    "darwin": {"blind_model": {"ledger": "b25b2d2fc5c4dccd4f04c1fe4712816029f1b8901efd4796f95961b2fd866c5e", "scores": "9acf015c53119163e2ab6765b2b9687112087b760922bd8f5018fa06278dcc1e", "suff": "83644d763ee69406fe4c926b65b2f5b013b94cdddd640722416abd1249dacda6"}, "val_model_auto": {"scores": "f2c4fa88de15c9cbaf8d9197b5e8f271010ab299ddccd9d9d05d896336645e33", "suff": "9ee72dcc3357e39e42ff641c15446074db09a22fce44d510db1634ad709633d1"}, "val_pa_pc": {"scores": "c5a0e7ff0a14f214f23278ece2bc03b1b8ca5d7c2414d4a19b0e84fd16613f12", "suff": "1a0858e4863ed754d54e6d8705dccc76bf7db00c9c789cd612f952c02e5811d1"}},
    "linux": {"blind_model": {"ledger": "b25b2d2fc5c4dccd4f04c1fe4712816029f1b8901efd4796f95961b2fd866c5e", "scores": "7909275118388962f4233f67585e79e1962a5fb045a72561c7425b103c17294f", "suff": "e9dd2f2d96997038732e312e72576d08aadf36620785b4cc194146edc6050ed0"}, "val_model_auto": {"scores": "88bfbed5a68249471e37e694aad141b852c071859a96a2ac48747346c3720dcf", "suff": "5f82184561440093995da633a6940b0fe957ef80cc9ca8b6b02a12fef303ebdd"}, "val_pa_pc": {"scores": "e9cc9a10d89c110f6e9e9297d69ca06cea74eb195ded6ff4fc97a74cbb94e80d", "suff": "4b2e21028af03d8f8b6acdd884f37f6bef88bc8755aa903d7d28748f62d7a08e"}},
}
BASE_DIGESTS = _DIGESTS.get(sys.platform, _DIGESTS["linux"])


class Reader(MiniCandi):
    """mu = (1 + mean visible input count) x the target depth from C' (control included)."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)

    def forward(self, b):
        x = (b["x_counts"].float() * b["x_avail"][:, None, :].float()).mean(-1, keepdim=True)
        depth = (2.0 ** b["c_out"][..., 0] / 2 ** 24)[:, None, :]
        mu = (self.w * (1.0 + x) * depth).clamp_min(1e-3)
        return {"n": torch.full_like(mu, 5.0), "mu": mu}


def _tsv(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _write_tsv(p, rows):
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _trim(ev):
    ev["imputation_targets"] = [x for x in ev["imputation_targets"]
                                if x["assay"] in ("H3K4me3", "DNase-seq")]
    ev["correction_pairs"], ev["replicate_pairs"] = [], []
    ev["scorable"]["imputation"] = ["DNase-seq", "H3K4me3"]
    return ev


def make_env(root: Path) -> dict:
    """Synthetic tables: H3K4me3 targets with (s0, s8) and without (s1, s2) an observed control,
    DNase-seq targets, a validation control track (s7); pipeline signal tracks = 0.7 x counts."""
    t = make_synthetic_tables(root, n_cells=9, val_tracks=VAL, blind_tracks=BLIND)
    rows = _tsv(t / "measurements.tsv")
    base = {(r["cell"], r["assay"]): r for r in rows if r["arm_knob"] == ""}
    for cell in ("s0", "s8"):
        base[(cell, "H3K4me3")]["control_measurement_id"] = base[(cell, "chipseq-control")][
            "measurement_id"]
    _write_tsv(t / "measurements.tsv", rows)
    targets = {}
    for (cell, a), r in base.items():
        if a in ("H3K4me3", "DNase-seq") and r["in_eic"] in ("val", "blind"):
            d = Path(r["counts_path"][3:]).parent
            for nm, src in (("signal", "counts"), ("pr1_signal", "pr1"), ("pr2_signal", "pr2")):
                with h5py.File(d / f"{src}.h5", "r") as fi, h5py.File(d / f"{nm}.h5", "w") as fo:
                    for c in fi:
                        fo.create_dataset(c, data=np.asarray(fi[c][:], np.float32) * 0.7)
            targets[(cell, a)] = r["measurement_id"]
    ev_dir = root / "eval" / "synth"
    val = ev_dir / "evalset.json"
    val.write_text(json.dumps(_trim(E.build(t, val, n_regions=2, n_fit_regions=1, region_bins=RB,
                                            seed=0))))
    blind = ev_dir / "evalset_blind.json"
    blind.write_text(json.dumps(_trim(E.build(t, blind, include_blind=True, n_regions=2,
                                              n_fit_regions=1, region_bins=RB, seed=0))))
    return {"root": root, "tables": t, "evalset": val, "evalset_blind": blind, "targets": targets}


def _normalize(text: str, root: Path) -> str:
    tsha = tables_sha256(Path(root) / "tables" / "synth")
    text = text.replace(tsha, "<TABLES_SHA256>")
    for p in sorted({str(root), str(Path(root).resolve())}, key=len, reverse=True):
        text = text.replace(p, "<ROOT>")
    return text


def scores_digest(out: Path, root: Path) -> str:
    s = json.loads((out / "scores.json").read_text(encoding="utf-8"))
    for k in VOLATILE:
        s.pop(k, None)
    s.pop("scorer_version", None)   # SF0 (Amendment 8): the one key added after the pinned base
    return hashlib.sha256(_normalize(json.dumps(s, indent=1, sort_keys=True, allow_nan=False),
                                     root).encode()).hexdigest()


def suff_digest(out: Path) -> str:
    """The arrays of suffstats.npz (the zip's own timestamps are not content)."""
    h = hashlib.sha256()
    with np.load(out / "suffstats.npz") as z:
        for k in sorted(z.files):
            a = np.ascontiguousarray(z[k])
            h.update(f"{k}|{a.dtype.str}|{a.shape}|".encode())
            h.update(a.tobytes())
    return h.hexdigest()


def ledger_digest(ledger: Path, root: Path) -> str:
    rows = _tsv(ledger)
    for r in rows:
        for k in ("time", "git_sha", "evalset_sha256"):
            r.pop(k, None)
    return hashlib.sha256(_normalize(json.dumps(rows, sort_keys=True), root).encode()).hexdigest()


def _score(env, out, *, blind=False, **kw):
    from onion_eval import score as S
    model = kw.pop("model", None)
    base = kw.pop("baseline", None)
    if model is None and base is None:
        model = Reader()
    ev = env["evalset_blind"] if blind else env["evalset"]
    extra = dict(split="blind", allow_blind=True) if blind else {}
    cfg = kw.pop("config", dict(CFG))
    return S.score(ev, out, model=model, baseline=base, config=cfg,
                   run_tables_sha256=tables_sha256(env["tables"]) if model is not None else None,
                   **extra, **KW, **kw)


def recipes(env, out_root: Path, dump_root=None) -> dict:
    """The three runs whose outputs are pinned; with `dump_root`, each also dumps into
    `dump_root/<recipe>` (same configs as without it)."""
    out_root = Path(out_root)

    def dump(name):
        return {} if dump_root is None else {"dump_signal": Path(dump_root) / name}
    got = {}
    o = out_root / "val_model_auto"
    _score(env, o, **dump("val_model_auto"))
    got["val_model_auto"] = {"scores": scores_digest(o, env["root"]), "suff": suff_digest(o)}
    o = out_root / "val_pa_pc"
    _score(env, o, baseline="position_average", lambda_source="predicted_control",
           **dump("val_pa_pc"))
    got["val_pa_pc"] = {"scores": scores_digest(o, env["root"]), "suff": suff_digest(o)}
    o = out_root / "bm" / "blind"
    _score(env, o, blind=True, **dump("blind_model"))
    got["blind_model"] = {"scores": scores_digest(o, env["root"]), "suff": suff_digest(o),
                          "ledger": ledger_digest(env["evalset_blind"].parent / "BLIND_PASSES.tsv",
                                                  env["root"])}
    return got


# ---------------------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def dumped(tmp_path_factory):
    """The pinned recipes twice, on two copies of the fixture: without and with the dump."""
    a = make_env(tmp_path_factory.mktemp("plain") / "env")
    b = make_env(tmp_path_factory.mktemp("dump") / "env")
    plain = recipes(a, a["root"] / "out")
    got = recipes(b, b["root"] / "out", dump_root=b["root"] / "dump")
    return {"env": b, "plain": plain, "dumped": got, "dump": b["root"] / "dump"}


def _meta(d: Path) -> dict:
    return json.loads((d / "tracks.json").read_text(encoding="utf-8"))


def _h5_regions(path: Path, regions) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.concatenate([np.asarray(f[c][s:s + RB], np.float64) for c, s in regions])


def _rows(env) -> dict:
    return {r["measurement_id"]: r for r in _tsv(env["tables"] / "measurements.tsv")}


# ------------------------------------------------------------------------------------- byte identity

def test_without_option_byte_identical_to_base(dumped):
    """scores.json, suffstats.npz and the ledger rows equal the digests taken on b647359."""
    assert all(v is not None for v in BASE_DIGESTS.values()), "base digests not pinned"
    assert dumped["plain"] == BASE_DIGESTS


def test_option_leaves_outputs_unchanged(dumped):
    """With the option, scores.json and suffstats.npz (and the ledger rows) equal a run without it
    at the same lambda source (auto: val_model_auto, blind_model; predicted_control: val_pa_pc)."""
    assert dumped["dumped"] == dumped["plain"]


# ---------------------------------------------------------------------------------------- contents

F_KEYS = {"method", "kind", "evalset_sha256", "regions_sha256", "split", "regime", "region_bins",
          "n_regions", "lambda_source", "targets", "lambda_source_mode", "trained_on_roles"}


def test_run_dump_interface_f(dumped):
    from onion_eval import ag2_space as AS
    from onion_eval.ag2_baselines import regions_sha256
    env = dumped["env"]
    ev = json.loads(env["evalset"].read_text(encoding="utf-8"))
    regions = [(c, int(s)) for c, s in ev["regions"]]
    rows = _rows(env)
    for regime in ("leave_one_assay_out", "accessibility_only"):
        d = dumped["dump"] / "val_model_auto" / regime
        m = _meta(d)
        assert F_KEYS <= set(m)
        assert (m["kind"], m["method"], m["split"], m["regime"]) == ("run", "model", "val", regime)
        assert m["trained_on_roles"] == ["train"]
        assert m["regions_sha256"] == regions_sha256(regions, RB)
        assert m["evalset_sha256"] == hashlib.sha256(env["evalset"].read_bytes()).hexdigest()
        assert (m["region_bins"], m["n_regions"], m["lambda_source_mode"]) == (RB, 2, "auto")
        assert set(m["lambda_source"]) == set(m["targets"])
        assert m["trained_on_roles_source"] == "dataset_train_roles"
        assert "signal_pc_reason" in m
        want = {t["target"] for t in ev["imputation_targets"]
                if t["regime"] == regime and t["split"] == "val"}
        assert set(m["targets"]) == want and len(want) >= 3
        assert not list(d.glob("*mask*")) and not list(d.glob("*.signal_pc.npy"))
        for mid in m["targets"]:
            row = rows[mid]
            arr = {k: np.load(d / f"{mid}.{k}.npy") for k in
                   ("signal", "truth", "native", "native_truth", "peakprob")}
            for a in arr.values():
                assert a.dtype == np.float32 and a.shape == (2 * RB,) and np.isfinite(a).all()
            assert 0.0 <= arr["peakprob"].min() and arr["peakprob"].max() <= 1.0
            # second route: the truth and the native truth from the h5 files
            cdir = Path(row["counts_path"][3:]).parent
            np.testing.assert_allclose(arr["truth"], np.arcsinh(_h5_regions(cdir / "signal.h5",
                                                                            regions)), rtol=1e-6)
            y = _h5_regions(cdir / "counts.h5", regions)
            if row["assay"] == "DNase-seq":
                nt = y * 25 / (float(row["read_length"]) + 24) * 1e8 / float(row["depth_reads"])
                np.testing.assert_allclose(arr["native_truth"], nt, rtol=1e-5)
            elif row["control_measurement_id"]:
                crow = rows[row["control_measurement_id"]]
                ctl = _h5_regions(Path(crow["counts_path"][3:]), regions)
                cov = {"depth_reads": row["depth_reads"], "read_length": row["read_length"],
                       "frag_len": row["frag_len"], "control_depth_reads": crow["depth_reads"],
                       "control_read_length": crow["read_length"], "control_is_predicted": False}
                fe = np.concatenate([AS.fold_change_track(y[i * RB:(i + 1) * RB],
                                                          ctl[i * RB:(i + 1) * RB], cov)[0]
                                     for i in range(2)])
                np.testing.assert_allclose(arr["native_truth"], fe, rtol=1e-5)
                assert m["lambda_source"][mid] == "observed_control"


def test_baseline_dump_predicted_control(dumped):
    """A baseline under predicted_control: signal_pc from the training-average control on every
    ChIP target; on DNase-seq targets (no control on either route) signal_pc is signal."""
    env = dumped["env"]
    rows = _rows(env)
    d = dumped["dump"] / "val_pa_pc" / "leave_one_assay_out"
    m = _meta(d)
    assert (m["kind"], m["method"], m["lambda_source_mode"]) == ("baseline", "position_average",
                                                                 "predicted_control")
    assert m["trained_on_roles"] == ["train"]
    chip = [t for t in m["targets"] if rows[t]["assay"] == "H3K4me3"]
    dnase = [t for t in m["targets"] if rows[t]["assay"] == "DNase-seq"]
    assert len(chip) == 3 and len(dnase) == 3
    for t in chip:
        pc = np.load(d / f"{t}.signal_pc.npy")
        assert pc.dtype == np.float32 and np.isfinite(pc).all()
        assert m["lambda_source_predicted_control"][t] == "training_average_control"
    s0 = env["targets"][("s0", "H3K4me3")]       # observed control: the two lambdas differ
    assert not np.array_equal(np.load(d / f"{s0}.signal.npy"), np.load(d / f"{s0}.signal_pc.npy"))
    for t in dnase:
        np.testing.assert_array_equal(np.load(d / f"{t}.signal_pc.npy"), np.load(d / f"{t}.signal.npy"))
        assert t in m["signal_pc_same_as_signal"]
        assert m["lambda_source_predicted_control"][t] == m["lambda_source"][t]
    assert not m["signal_pc_left_out"]
    for t in m["targets"]:
        assert (d / f"{t}.native.npy").is_file() and (d / f"{t}.peakprob.npy").is_file()


def test_blind_dump_inside_the_ledgered_pass(dumped):
    env = dumped["env"]
    m = _meta(dumped["dump"] / "blind_model" / "leave_one_assay_out")
    assert m["split"] == "blind" and m["targets"] == [env["targets"][("s8", "H3K4me3")]]
    rows = _tsv(env["evalset_blind"].parent / "BLIND_PASSES.tsv")
    assert [r["status"] for r in rows] == ["started", "scored"]


# ------------------------------------------------------------------------------------- refusals

def test_blind_dump_outside_a_ledgered_pass_refused(dumped, tmp_path):
    from onion_eval import score as S
    env = dumped["env"]
    sc = S.Scorer(env["evalset_blind"], main="model", model=Reader(),
                  config=dict(CFG), split="blind", allow_blind=True,
                  run_tables_sha256=tables_sha256(env["tables"]), dump_signal=tmp_path / "d", **KW)
    with pytest.raises(S.ScoreRefused, match="ledger"):
        sc.run()
    assert not (tmp_path / "d").exists()


def test_trained_on_roles_from_the_run_record(dumped, tmp_path):
    from onion_eval import score as S
    env = dumped["env"]
    kw = dict(main="model", model=Reader(), run_tables_sha256=tables_sha256(env["tables"]),
              dump_signal=tmp_path / "d", **KW)
    sc = S.Scorer(env["evalset"], config=dict(CFG), **kw)       # the dataset's default
    assert sc.trained_on_roles == ["train"]
    sc = S.Scorer(env["evalset"], config=dict(CFG, train_on_validation=True, final_model=True),
                  **kw)
    assert sc.trained_on_roles == ["train", "val"]
    sc = S.Scorer(env["evalset"], config=dict(CFG), **dict(kw, dump_signal=None))  # no dump: no rule
    assert sc.trained_on_roles is None


# ------------------------------------------------------------------------------ external, targets

def test_external_writes_signal_only(dumped, tmp_path):
    from onion_eval import score as S
    env = dumped["env"]

    def entry(row, regions):
        p = Path(row["counts_path"][3:]).parent / "signal.h5"
        return [x.astype(np.float32) for x in np.split(_h5_regions(p, regions), len(regions))]
    S.score(env["evalset"], tmp_path / "o", signal_override=entry, dump_signal=tmp_path / "d",
            dump_method="some_entry", **KW)
    d = tmp_path / "d" / "leave_one_assay_out"
    m = _meta(d)
    assert (m["kind"], m["method"]) == ("external", "some_entry") and "trained_on_roles" not in m
    assert m["targets"] and sorted(p.name for p in d.glob("*.npy")) == sorted(
        f"{t}.signal.npy" for t in m["targets"])
    for t in m["targets"]:
        np.testing.assert_allclose(np.load(d / f"{t}.signal.npy"), np.load(
            dumped["dump"] / "val_model_auto" / "leave_one_assay_out" / f"{t}.truth.npy"), rtol=1e-6)


def test_dump_targets_limits_the_dump(dumped, tmp_path):
    env = dumped["env"]
    one = env["targets"][("s0", "H3K4me3")]       # observed control: defined for marginal_nb
    _score(env, tmp_path / "o", baseline="marginal_nb", dump_signal=tmp_path / "d",
           dump_targets={one})
    for regime in ("leave_one_assay_out", "accessibility_only"):
        d = tmp_path / "d" / regime
        assert _meta(d)["targets"] == [one]
        assert {p.name for p in d.glob("*.npy")} == {f"{one}.{k}.npy" for k in
                                                     ("signal", "truth", "native", "native_truth",
                                                      "peakprob")}


# ---------------------------------------------------------------------------------- pass-through

class _Captured(Exception):
    pass


def test_score_cli_passes_the_option(dumped, tmp_path, monkeypatch):
    from onion_eval import score as S
    env = dumped["env"]
    f = tmp_path / "ids.txt"
    f.write_text("a\n# note\nb\n\n", encoding="utf-8")
    seen = {}

    def fake(*a, **kw):
        seen.update(kw)
        raise _Captured
    monkeypatch.setattr(S, "score", fake)
    with pytest.raises(_Captured):
        S.main(["--baseline", "position_average", "--evalset", str(env["evalset"]), "--out",
                str(tmp_path / "o"), "--dump-signal", str(tmp_path / "d"), "--dump-targets", str(f)])
    assert seen["dump_signal"] == str(tmp_path / "d") and seen["dump_targets"] == {"a", "b"}
    seen.clear()
    with pytest.raises(_Captured):
        S.main(["--baseline", "position_average", "--evalset", str(env["evalset"]), "--out",
                str(tmp_path / "o")])
    assert "dump_signal" not in seen


def test_external_signal_cli_passes_the_option(dumped, tmp_path, monkeypatch):
    from onion_eval import eic_score as ES
    from onion_eval import external_signal as X
    from onion_eval import score as S
    env = dumped["env"]
    seen = {}

    class Tracks:
        def __init__(self, d, rb):
            pass

        def path(self, name):
            return "present"

    def fake(*a, **kw):
        seen.update(kw)
        raise _Captured
    monkeypatch.setattr(X, "EntryTracks", Tracks)
    monkeypatch.setattr(S, "score", fake)
    team = sorted(ES.TEAM_NAME)[0]
    args = ["score", "--entry-dir", str(tmp_path), "--entry", str(team), "--evalset",
            str(env["evalset"]), "--out", str(tmp_path / "o"), "--dump-signal", str(tmp_path / "d")]
    assert X.main(args) == 2 and not seen        # no record of the control: refused
    with pytest.raises(_Captured):
        X.main(args + ["--uses-target-cell-control", "no"])
    assert seen["dump_signal"] == str(tmp_path / "d") and seen["dump_targets"] is None
    assert seen["dump_method"] == ES.TEAM_NAME[team]


def test_external_signal_stamps_uses_target_cell_control(dumped, tmp_path):
    """external_signal.py writes the entry's record into each tracks.json of a real external dump."""
    from onion_eval import external_signal as X
    from onion_eval import score as S
    env = dumped["env"]

    def entry(row, regions):
        return _h5_regions(Path(row["counts_path"][3:]).parent / "signal.h5", regions)
    S.score(env["evalset"], tmp_path / "o", signal_override=entry, dump_signal=tmp_path / "d", **KW)
    got = X.stamp_control_record(tmp_path / "d", "no")
    assert len(got) == 2
    for tj in got:
        m = json.loads(tj.read_text(encoding="utf-8"))
        assert m["uses_target_cell_control"] == "no" and m["kind"] == "external"
        assert set(m["lambda_source"].values()) == {"external_entry"}
    with pytest.raises(ValueError):
        X.stamp_control_record(tmp_path / "d", "maybe")


def test_baseline_retrieval_cli_dumps(dumped, tmp_path):
    from onion_eval import baseline_retrieval as BR
    env = dumped["env"]
    assert BR.main(["run", "--tables", str(env["tables"]), "--evalset", str(env["evalset"]),
                    "--out", str(tmp_path / "o"), "--dump-signal", str(tmp_path / "d")]) == 0
    d = tmp_path / "d" / "leave_one_assay_out"
    m = _meta(d)
    assert (m["kind"], m["method"], m["trained_on_roles"]) == ("baseline", "retrieval", ["train"])
    assert m["targets"] and all((d / f"{t}.native.npy").is_file() for t in m["targets"])


def test_lambda_source_keys(dumped):
    """`lambda_source` and `lambda_source_predicted_control` are per-track dicts of the source
    actually used; the scorer's mode is `lambda_source_mode`."""
    env = dumped["env"]
    rows = _rows(env)
    m = _meta(dumped["dump"] / "val_model_auto" / "leave_one_assay_out")
    assert set(m["lambda_source"]) == set(m["targets"]) and "lambda_source_predicted_control" not in m
    for t, src in m["lambda_source"].items():
        if rows[t]["assay"] == "H3K4me3":
            assert src == ("observed_control" if rows[t]["control_measurement_id"]
                           else "predicted_control")
    p = _meta(dumped["dump"] / "val_pa_pc" / "leave_one_assay_out")
    assert set(p["lambda_source_predicted_control"]) == set(p["targets"]) == set(p["lambda_source"])


# ------------------------------------------------------------------- round trip through ag2_score

def test_round_trip_through_the_ag2_score_loader(dumped, tmp_path):
    """The merged reader (`ag2_score._resolve`, `_roles`, `_trained`, `lambda_and_control`) on the
    dumps of a run and a baseline: control label and lambda column per track."""
    from onion_eval import ag2_score as G
    env = dumped["env"]
    t = env["targets"]
    s0, s1, dn = t[("s0", "H3K4me3")], t[("s1", "H3K4me3")], t[("s3", "DNase-seq")]
    regime = "leave_one_assay_out"
    _, run = G._resolve(dumped["dump"] / "val_model_auto", regime)
    assert G.lambda_and_control(run, s0, "headline") == ("observed_control", "yes")
    assert G.lambda_and_control(run, s1, "headline") == ("predicted_control", "no")
    lam, ctl = G.lambda_and_control(run, dn, "headline")
    assert lam in G.CONTROL_OF_LAMBDA and ctl == "no"
    assert G._roles(run) == {"train"} and not G._trained("run", G._roles(run), "val", "unseen")
    _, base = G._resolve(dumped["dump"] / "val_pa_pc", regime)
    assert G.lambda_and_control(base, s0, "headline") == ("observed_control", "yes")
    assert G.lambda_and_control(base, s1, "headline") == ("training_average_control", "no")
    for x in (s0, s1):
        assert G.lambda_and_control(base, x, "control_free") == ("training_average_control", "no")
    assert G.lambda_and_control(base, dn, "control_free")[1] == "no"
    assert G._roles(base) == {"train"}
    # a final model trained on train + val: its validation tracks leave every comparison set
    from onion_eval import score as S
    o = tmp_path / "tv"
    S.score(env["evalset"], o, model=Reader(), config=dict(CFG, train_on_validation=True,
                                                            final_model=True),
            run_tables_sha256=tables_sha256(env["tables"]), dump_signal=tmp_path / "d", **KW)
    _, tv = G._resolve(tmp_path / "d", regime)
    assert tv["trained_on_roles"] == ["train", "val"] and G._roles(tv) == {"train", "val"}
    assert G._trained("run", G._roles(tv), "val", "unseen")
    assert not G._trained("run", G._roles(tv), "blind", "unseen")
