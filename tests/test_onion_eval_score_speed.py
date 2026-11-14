"""SM speed options of the scorer: worker processes (`workers`) and staging to local disk
(`stage_dir`) leave every output unchanged.

Pins: on the synthetic case of `test_onion_eval_score_memory.py` (6 regions, the four training-free
baselines, with `--dump-signal`), scores.json (volatile keys dropped), suffstats.npz and every
dumped file with 3 workers, and with 3 workers plus staging, equal the digests of the base commit
fb3eeb7 pinned there (the serial scorer equals them too: `test_outputs_identical_to_base`). A model
is always scored serially (`workers` is ignored), so only the baselines run in workers.

The worker scorings run in a fresh interpreter, as score.py does in a job: forked workers must not
inherit an OpenMP pool that earlier torch work in the same process started (GNU OpenMP is not
fork-safe; inside one pytest process that runs other torch tests first, the forked workers hung)."""
from __future__ import annotations

import json
import os
import subprocess
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

BASELINES = [r for r in M.RECIPES if "baseline" in r[1]]

_CHILD = r"""
import json, sys
from pathlib import Path
sys.path[:0] = [sys.argv[1] + "/tools", sys.argv[1] + "/tests"]
import test_onion_eval_score_memory as M
root, kw = Path(sys.argv[2]), json.loads(sys.argv[3])
env = M.make_env(root / "env")
got = {}
for name, rk in M.RECIPES:
    if "baseline" not in rk:
        continue
    o, d = root / "out" / name, root / "dump" / name
    M._score(env, o, dump_signal=d, **rk, **kw)
    got[name] = {"scores": M.T.scores_digest(o, env["root"]), "suff": M.T.suff_digest(o),
                 "dump": M.dump_digest(d, env["root"])}
print("DIGESTS " + json.dumps(got, sort_keys=True))
"""


def _digests(root: Path, **kw) -> dict:
    p = subprocess.run([sys.executable, "-c", _CHILD, str(REPO), str(root), json.dumps(kw)],
                       capture_output=True, text=True, timeout=1800,
                       env=dict(os.environ, PYTHONPATH=os.pathsep.join(
                           [str(REPO / "src"), os.environ.get("PYTHONPATH", "")])))
    assert p.returncode == 0, p.stderr[-3000:]
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("DIGESTS ")][-1]
    return json.loads(line[len("DIGESTS "):])


def test_workers_identical_to_base(tmp_path):
    want = {name: M.BASE_DIGESTS[name] for name, _ in BASELINES}
    assert _digests(tmp_path / "w3", workers=3) == want


def test_workers_and_staging_identical_to_base(tmp_path):
    want = {name: M.BASE_DIGESTS[name] for name, _ in BASELINES}
    stage = tmp_path / "stage"
    assert _digests(tmp_path / "st", workers=3, stage_dir=str(stage)) == want
    # the rows pointed at real copies: counts.h5 files, not links, were staged
    files = list(stage.rglob("counts.h5"))
    assert files and all(f.is_file() and not f.is_symlink() for f in files)
