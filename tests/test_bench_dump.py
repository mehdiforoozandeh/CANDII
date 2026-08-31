"""t63 — dump a CANDI checkpoint to the §4.1 prediction root (`candi.bench.dump`).

The headline is the t47 gate pointed the other way: stream a checkpoint with the dump CLI's
own writer, score the root through `candi.bench.external`, and every shared numeric key must
agree with `candi.bench` on the same model. That is what makes "CANDI can be scored by the
same frozen instrument as every rival" a fact rather than a claim.

Nothing is mocked. `make_store` writes a real `CANDI_STORE`; a real `CandiModel` is the
checkpoint. The dump path reuses `harness.stream_tracks` and `cli._build_model` — this file
does not re-derive either.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from candi.bench import harness as H
from candi.bench.dump import (_write_npz, arrays_for_chrom, dump_predictions, main,
                              write_manifest)
from candi.bench.external import score_external, track_dirname
from candi.bench.harness import Pair, open_source
from candi.model import build_model
from candi.store import layout as L

from tests.test_store_reader import ASSAYS, N_BINS, make_store
from tests.test_store_regime import CTX, regime_dict

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

#: Two declared pairs, each posing imputation of H3K4me3. Copied from t47 so the dump and the
#: external scorer see the same panel the round-trip was proven on.
PAIRED_TRACKS = {
    "T_aa": ("ATAC-seq", "DNase-seq", L.CONTROL_TRACK),
    "V_aa": ("ATAC-seq", "H3K4me3"),
    "T_bb": ("ATAC-seq", "DNase-seq", L.CONTROL_TRACK),
    "V_bb": ("ATAC-seq", "H3K4me3"),
}
PAIRS = (Pair("T_aa", "V_aa"), Pair("T_bb", "V_bb"))
TARGET_ASSAY = "H3K4me3"
C_PAIRS = 2_000
METHOD = "candi-dump"


# ---------------------------------------------------------------------------
# a real store, a real regime, a real checkpoint
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Path:
    return make_store(tmp_path_factory.mktemp("dumpstore"), tracks=PAIRED_TRACKS)


@pytest.fixture(scope="module")
def regime_file(store, tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("dumpregime")
    obj = regime_dict(store, biosamples={"train": ["T_aa", "T_bb"], "eval": ["V_aa", "V_bb"]},
                      kinds=["counts", "peaks", "pval"],
                      eval_pairs=[["T_aa", "V_aa"], ["T_bb", "V_bb"]])
    p = d / "regime.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = build_model(num_assays=len(ASSAYS), context_length=CTX, resolution=25,
                    depth_center=24.25, heads=("count", "signal", "peak"))
    m.eval()
    return m


def _source(regime_file, **kw):
    return open_source(store=regime_file, **kw)


def _numeric(d):
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float)) and
            not isinstance(v, bool)}


@pytest.fixture(scope="module")
def model_scores(regime_file, model):
    """The normal bench path. `C` is off: the external entry cannot emit it."""
    src = _source(regime_file)
    try:
        return H.run_bench(model, src, "cpu", kinds=("impute",), batch_windows=4,
                           blocks=("E", "P", "D", "B"), c_index_pairs=C_PAIRS, seed=0)
    finally:
        src.close()


@pytest.fixture(scope="module")
def dumped_root(tmp_path_factory, regime_file, model) -> Path:
    root = tmp_path_factory.mktemp("dumppred") / "root"
    src = _source(regime_file)
    try:
        dump_predictions(model, src, root, method=METHOD, batch_windows=4, device="cpu")
    finally:
        src.close()
    return root


@pytest.fixture(scope="module")
def external_scores(regime_file, dumped_root):
    src = _source(regime_file)
    try:
        return score_external(src, dumped_root, seed=0, c_index_pairs=C_PAIRS)
    finally:
        src.close()


# ---------------------------------------------------------------------------
# 1 — the t47 gate: dump then score, same numbers as candi.bench
# ---------------------------------------------------------------------------

def test_the_dump_covers_both_declared_pairs(dumped_root) -> None:
    got = {p.name for p in dumped_root.iterdir() if p.is_dir()}
    assert got == {track_dirname(p, TARGET_ASSAY) for p in PAIRS}


def test_every_shared_numeric_key_survives_the_round_trip(model_scores, external_scores) -> None:
    """Same tracks, same arms, same keys, same numbers — the t47 tolerance."""
    assert external_scores["tracks"] == model_scores["tracks"]
    for key in model_scores["tracks"]:
        mine, theirs = model_scores["per_track"][key], external_scores["per_track"][key]
        assert set(mine) == set(theirs) == {"count", "pval"}, key
        for arm in ("count", "pval"):
            a, b = _numeric(mine[arm]), _numeric(theirs[arm])
            assert set(a) == set(b), f"{key}/{arm} key sets differ"
            for k in sorted(a):
                if np.isfinite(a[k]) or np.isfinite(b[k]):
                    assert b[k] == pytest.approx(a[k], rel=1e-6, abs=1e-6), f"{key}/{arm}/{k}"
                else:
                    assert not np.isfinite(a[k]) and not np.isfinite(b[k]), f"{key}/{arm}/{k}"


def test_the_macro_and_the_panel_survive_the_round_trip(model_scores, external_scores) -> None:
    for arm in ("count", "pval"):
        a, b = _numeric(model_scores["macro"][arm]), _numeric(external_scores["macro"][arm])
        assert set(a) == set(b)
        for k in sorted(a):
            assert b[k] == pytest.approx(a[k], rel=1e-6, abs=1e-6), f"macro/{arm}/{k}"
    assert set(external_scores["panel"]) == set(model_scores["panel"])
    mine = model_scores["panel"][TARGET_ASSAY]
    theirs = external_scores["panel"][TARGET_ASSAY]
    assert theirs["n_cell_types"] == mine["n_cell_types"] == 2
    assert theirs["tracks"] == mine["tracks"]
    for key in mine["tracks"]:
        a, b = _numeric(mine[key]), _numeric(theirs[key])
        assert set(a) == set(b)
        for k in sorted(a):
            if np.isfinite(a[k]) or np.isfinite(b[k]):
                assert b[k] == pytest.approx(a[k], rel=1e-6, abs=1e-6), f"panel/{key}/{k}"
            else:
                assert not np.isfinite(a[k]) and not np.isfinite(b[k]), f"panel/{key}/{k}"


def test_a_real_peak_score_is_recorded_as_a_real_peak_head(external_scores) -> None:
    for arms in external_scores["per_track"].values():
        assert np.isfinite(float(arms["pval"]["bernoulli_nll"]))


def test_npz_lengths_match_the_store_grid(dumped_root, regime_file) -> None:
    src = _source(regime_file)
    try:
        chroms = list(src.eval_chroms)
        want = {c: src.n_bins(c) for c in chroms}
    finally:
        src.close()
    for track in dumped_root.iterdir():
        if not track.is_dir():
            continue
        for c, n in want.items():
            with np.load(track / f"{c}.npz") as z:
                for k in z.files:
                    assert z[k].shape == (n,), f"{track.name}/{c}.npz `{k}` is {z[k].shape}"
                    assert n == N_BINS[c]


def test_the_manifest_names_the_method_and_the_declared_tracks(dumped_root) -> None:
    man = json.loads((dumped_root / "manifest.json").read_text(encoding="utf-8"))
    assert man["method"] == METHOD
    assert man["generated_by"] == "candi.bench.dump"
    assert set(man["declared_tracks"]) == {track_dirname(p, TARGET_ASSAY) for p in PAIRS}
    assert "pval" in man["arms"] and "count" in man["arms"]


# ---------------------------------------------------------------------------
# 2 — peak_score is emitted only when the checkpoint has a peak head
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def count_only_model():
    torch.manual_seed(0)
    m = build_model(num_assays=len(ASSAYS), context_length=CTX, resolution=25,
                    depth_center=24.25, heads=("count",))
    m.eval()
    return m


def test_a_count_only_checkpoint_does_not_fabricate_peak_score_or_signal(
        regime_file, count_only_model, tmp_path) -> None:
    src = _source(regime_file)
    try:
        rec = next(H.stream_tracks(count_only_model, src, "cpu", kind="impute", batch_windows=4))
        payload = arrays_for_chrom(rec, rec.chroms[0], N_BINS[rec.chroms[0]])
        root = tmp_path / "countonly"
        dump_predictions(count_only_model, src, root, method=METHOD, batch_windows=4, device="cpu")
    finally:
        src.close()
    assert rec.has_peak_head is False and rec.has_pval is False
    assert set(payload) == {"mu", "n"}
    with np.load(root / track_dirname(rec.pair, rec.assay) / f"{rec.chroms[0]}.npz") as z:
        assert set(z.files) == {"mu", "n"}


# ---------------------------------------------------------------------------
# 3 — the CLI, driven exactly as a shell drives it
# ---------------------------------------------------------------------------

def test_the_cli_writes_a_root_the_external_entry_can_score(
        regime_file, model, dumped_root, tmp_path) -> None:
    ck = tmp_path / "m.pt"
    torch.save(model.state_dict(), ck)
    out = tmp_path / "cli_root"
    rc = main(["--store", str(regime_file), "--ckpt", str(ck), "--out", str(out),
               "--method", METHOD, "--heads", "count,signal,peak",
               "--depth-center", "24.25", "--quiet"])
    assert rc == 0
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man["method"] == METHOD
    src = _source(regime_file)
    try:
        got = score_external(src, out, seed=0, c_index_pairs=C_PAIRS)
    finally:
        src.close()
    ref_src = _source(regime_file)
    try:
        ref = score_external(ref_src, dumped_root, seed=0, c_index_pairs=C_PAIRS)
    finally:
        ref_src.close()
    for key in got["tracks"]:
        assert got["per_track"][key]["pval"]["mse"] == pytest.approx(
            float(ref["per_track"][key]["pval"]["mse"]), rel=1e-6, abs=1e-6)


# ---------------------------------------------------------------------------
# 4 — t83: the root is deflated on write, and losslessly so
# ---------------------------------------------------------------------------

def _npz_members(root: Path):
    for track in sorted(p for p in root.iterdir() if p.is_dir()):
        for npz in sorted(track.glob("*.npz")):
            yield npz


def test_every_dumped_npz_is_deflated_rather_than_stored(dumped_root) -> None:
    """§12.6's footprint assumes compression. `np.savez` stores; `savez_compressed` deflates."""
    seen = 0
    for npz in _npz_members(dumped_root):
        with zipfile.ZipFile(npz) as z:
            members = z.infolist()
            assert members, f"{npz} holds no arrays"
            for m in members:
                assert m.compress_type == zipfile.ZIP_DEFLATED, f"{npz}::{m.filename} is stored"
                seen += 1
    assert seen > 0


@pytest.fixture(scope="module")
def stored_root(dumped_root, tmp_path_factory) -> Path:
    """The same root rewritten with `np.savez` — the uncompressed twin the scores must match."""
    out = tmp_path_factory.mktemp("dumpstored") / "root"
    shutil.copytree(dumped_root, out)
    for npz in _npz_members(out):
        with np.load(npz) as z:
            payload = {k: np.asarray(z[k]) for k in z.files}
        np.savez(npz.with_suffix(""), **payload)          # np.savez appends .npz itself
    return out


def test_compression_loses_no_bit_of_any_prediction_array(regime_file, model, tmp_path) -> None:
    """One real payload, written both ways, both read back against the arrays in memory.

    Comparing the two FILES would be circular — the uncompressed twin is made by reading the
    compressed one. The in-memory arrays `arrays_for_chrom` returned are the only reference that
    predates either write.
    """
    src = _source(regime_file)
    try:
        rec = next(H.stream_tracks(model, src, "cpu", kind="impute", batch_windows=4))
        chrom = rec.chroms[0]
        payload = arrays_for_chrom(rec, chrom, N_BINS[chrom])
    finally:
        src.close()
    assert set(payload) == {"mu", "n", "signal_mu", "signal_sigma", "peak_score"}
    _write_npz(tmp_path / "deflated.npz", payload)
    np.savez(tmp_path / "stored", **payload)
    for name in ("deflated.npz", "stored.npz"):
        with np.load(tmp_path / name) as z:
            assert sorted(z.files) == sorted(payload), name
            for k, want in payload.items():
                assert z[k].dtype == want.dtype, f"{name} `{k}`"
                assert np.array_equal(z[k], want), f"{name} `{k}`"


def test_a_compressed_root_scores_exactly_as_its_uncompressed_twin(
        regime_file, external_scores, stored_root) -> None:
    """Bit-identical arrays must give bit-identical numbers — no tolerance, and none needed."""
    src = _source(regime_file)
    try:
        stored = score_external(src, stored_root, seed=0, c_index_pairs=C_PAIRS)
    finally:
        src.close()
    assert stored["tracks"] == external_scores["tracks"]
    for key in external_scores["tracks"]:
        for arm in ("count", "pval"):
            a = _numeric(external_scores["per_track"][key][arm])
            b = _numeric(stored["per_track"][key][arm])
            assert set(a) == set(b), f"{key}/{arm}"
            for k in sorted(a):
                if np.isfinite(a[k]) or np.isfinite(b[k]):
                    assert b[k] == a[k], f"{key}/{arm}/{k}"
                else:
                    assert not np.isfinite(a[k]) and not np.isfinite(b[k]), f"{key}/{arm}/{k}"


def test_the_dump_reports_bytes_written_against_bytes_raw(regime_file, model, tmp_path,
                                                          capsys) -> None:
    """§12.6's table is a synthetic estimate until a real run prints this line."""
    src = _source(regime_file)
    try:
        dump_predictions(model, src, tmp_path / "sized", method=METHOD, batch_windows=4,
                         device="cpu", progress=True)
    finally:
        src.close()
    line = [ln for ln in capsys.readouterr().out.splitlines() if "written vs" in ln]
    assert len(line) == 1, "the dump must report its footprint exactly once"
    disk, raw = (int(x.replace(",", "")) for x in
                 re.match(r".*?([\d,]+) B written vs ([\d,]+) B raw", line[0]).groups())
    on_disk = sum(p.stat().st_size for p in _npz_members(tmp_path / "sized"))
    assert disk == on_disk and raw > disk        # the report is the real footprint, and it shrank


# ---------------------------------------------------------------------------
# 5 — t83: the manifest says which model wrote the root (§5's touch-once rule)
# ---------------------------------------------------------------------------

#: Every field §5 needs to audit "the B_ panel was predicted exactly once".
PROVENANCE_FIELDS = ("ckpt_sha256", "store_manifest_sha256", "regime_id", "panel", "code_sha",
                     "seed", "chroms")


def _manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def test_the_manifest_carries_every_provenance_field(dumped_root) -> None:
    man = _manifest(dumped_root)
    assert all(f in man for f in PROVENANCE_FIELDS), sorted(set(PROVENANCE_FIELDS) - set(man))


def test_the_manifest_identifies_the_store_the_regime_and_the_panel(dumped_root, store,
                                                                    regime_file) -> None:
    man = _manifest(dumped_root)
    want = hashlib.sha256(L.manifest_path(store).read_bytes()).hexdigest()
    assert man["store_manifest_sha256"] == want
    from candi.store.regime import Regime
    assert man["regime_id"] == Regime.from_file(regime_file).sha256
    assert man["panel"] == "V"                     # both declared pairs target a V_ cell
    assert man["chroms"] == ["chr2"]
    assert isinstance(man["seed"], int)
    assert len(str(man["code_sha"])) == 40         # the full HEAD sha, not the short one


def test_the_manifest_reaches_the_score_file_verbatim(external_scores, dumped_root) -> None:
    """`read_manifest` copies the manifest through, so the new fields cost the consumer nothing."""
    assert external_scores["provenance"]["manifest"] == _manifest(dumped_root)


def test_two_checkpoints_write_two_different_ckpt_sha256(tmp_path, model,
                                                         count_only_model) -> None:
    """The one field §5 rests on: two B_ roots from two models must be told apart on disk."""
    roots = []
    for name, m in (("a", model), ("b", count_only_model)):
        ck = tmp_path / f"{name}.pt"
        torch.save(m.state_dict(), ck)
        root = tmp_path / f"root_{name}"
        write_manifest(root, method=METHOD, declared_tracks=[], arms=["count"], ckpt=ck)
        roots.append(_manifest(root))
    assert roots[0]["ckpt_sha256"] != roots[1]["ckpt_sha256"]
    assert all(len(r["ckpt_sha256"]) == 64 for r in roots)


def test_the_cli_records_the_checkpoint_it_was_given(regime_file, model, tmp_path) -> None:
    ck = tmp_path / "m.pt"
    torch.save(model.state_dict(), ck)
    out = tmp_path / "ckpt_root"
    assert main(["--store", str(regime_file), "--ckpt", str(ck), "--out", str(out),
                 "--method", METHOD, "--heads", "count,signal,peak",
                 "--depth-center", "24.25", "--quiet"]) == 0
    man = _manifest(out)
    assert man["ckpt_sha256"] == hashlib.sha256(ck.read_bytes()).hexdigest()
    assert "ckpt_sha256" not in man["unknown"]


def test_an_undeterminable_field_is_an_explicit_null_with_a_reason(dumped_root) -> None:
    """`dumped_root` was written from a live model, so there is no checkpoint file to hash.

    An absent key and an unknown value must not look the same: the key stays, the value is null,
    and `unknown` says why. A reader that only sees a missing key cannot tell "not recorded" from
    "could not be determined".
    """
    man = _manifest(dumped_root)
    assert "ckpt_sha256" in man and man["ckpt_sha256"] is None
    assert man["unknown"]["ckpt_sha256"]
    for field in PROVENANCE_FIELDS:
        if man[field] is None:
            assert man["unknown"].get(field), f"{field} is null with no reason"
        else:
            assert field not in man["unknown"], f"{field} is known and still listed as unknown"
