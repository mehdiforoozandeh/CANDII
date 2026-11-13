"""SM speed options of the scorer: worker processes (`workers`) and staging to local disk
(`stage_dir`) leave every output unchanged.

Pins: on the synthetic case of `test_onion_eval_score_memory.py` (6 regions, a model with the
predicted-control route and the four baselines, with `--dump-signal`), scores.json (volatile keys
dropped), suffstats.npz and every dumped file are identical with 1 worker, with 3 workers, and with
3 workers plus staging, and equal the digests of the base commit fb3eeb7 pinned there."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import test_onion_eval_score_memory as M  # noqa: E402

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                reason="worker processes fork; run on the cluster (Linux)")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return M.make_env(tmp_path_factory.mktemp("sp") / "env")


def _digests(env, root: Path, **kw) -> dict:
    got = {}
    for name, rk in M.RECIPES:
        o, d = root / "out" / name, root / "dump" / name
        M._score(env, o, dump_signal=d, **rk, **kw)
        got[name] = {"scores": M.T.scores_digest(o, env["root"]), "suff": M.T.suff_digest(o),
                     "dump": M.dump_digest(d, env["root"])}
    return got


def test_workers_identical_to_serial_and_base(env, tmp_path):
    serial = _digests(env, tmp_path / "w1")
    par = _digests(env, tmp_path / "w3", workers=3)
    assert par == serial
    assert serial == M.BASE_DIGESTS


def test_staging_identical(env, tmp_path):
    staged = _digests(env, tmp_path / "st", workers=3, stage_dir=tmp_path / "stage")
    assert staged == M.BASE_DIGESTS
    # the rows point at the staged copies; every original file is still reachable by a link
    files = list((tmp_path / "stage").rglob("counts.h5"))
    assert files and all(f.is_file() and not f.is_symlink() for f in files)
