"""miniCANDI base-model digests (t370 chunk C5).

`tools/golden.py` covers `src/candi` only. This test pins the three base miniCANDI models
(`tools/minicandi/configs/rung{2,3,4}/base_heldout.json`, at `context_bins` 64) to digests recorded
on the pre-change tree (exp/t370-minicandi-recipe at 57f3c9e8), so every later change proves that
the base configs still build the same model and give the same outputs bit for bit.

Per rung the JSON holds: the parameter count, the SHA-256 of the sorted `state_dict` key list, the
SHA-256 of the `state_dict` tensors (sorted keys, each key, dtype, shape and raw bytes), and on one
`minicandi.synth` batch in eval mode the SHA-256 of `mu` and `n` and the loss
(`losses.total_loss`) -- once at initialisation and once after a fixed perturbation of every
parameter (the NB head starts at zero, so the initial outputs do not depend on the trunk).
The batch itself is also hashed, so a failure tells a data change from a model change.

Re-record (only on purpose, on the tree that defines the reference):
    PYTHONPATH=$PWD/src:$PWD python tests/test_minicandi_base_golden.py --record
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import losses  # noqa: E402
from minicandi import registry as R  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi.dataset import OnionDataset  # noqa: E402

GOLDEN = _ROOT / "tests/data/minicandi_base_golden.json"
RUNGS = (2, 3, 4)
L = 64
BATCH = 4


def _cfg(r: int) -> dict:
    cfg = json.loads((_ROOT / f"tools/minicandi/configs/rung{r}/base_heldout.json").read_text())
    return {**cfg, "context_bins": L}


def _sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _sha_dict(d: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(d):
        v = d[k]
        h.update(k.encode())
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().contiguous()
            h.update(f"{v.dtype}{tuple(v.shape)}".encode())
            h.update(v.numpy().tobytes())
        elif isinstance(v, np.ndarray):
            v = np.ascontiguousarray(v)
            h.update(f"{v.dtype}{v.shape}".encode())
            h.update(v.tobytes())
        else:
            h.update(repr(v).encode())
    return h.hexdigest()


def _forward(model, batch, cfg) -> dict:
    model.eval()
    with torch.no_grad():
        out = model(batch)
        loss, _ = losses.total_loss(out, batch, cfg)
    return {"mu_sha256": _sha_tensor(out["mu"]), "n_sha256": _sha_tensor(out["n"]),
            "loss": float(loss), "loss_hex": float(loss).hex()}


def _digest(r: int, tables: Path) -> dict:
    cfg = _cfg(r)
    batch = OnionDataset(tables, "train", cfg, 0).batch(BATCH, np.random.default_rng(0))
    torch.manual_seed(0)
    model = R.build_model(cfg)
    sd = model.state_dict()
    rec = {
        "config": f"tools/minicandi/configs/rung{r}/base_heldout.json",
        "context_bins": L,
        "batch_size": BATCH,
        "batch_sha256": _sha_dict(batch),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "state_dict_keys_sha256": hashlib.sha256("\n".join(sorted(sd)).encode()).hexdigest(),
        "state_dict_sha256": _sha_dict(sd),
        "init": _forward(model, batch, cfg),
    }
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    rec["perturbed"] = _forward(model, batch, cfg)
    return rec


def _make_tables(root: Path) -> Path:
    td = synth.make_synthetic_tables(root)
    synth.make_evalset(td, n_regions=2, region_bins=4 * L)
    return td


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> Path:
    return _make_tables(tmp_path_factory.mktemp("minicandi_golden"))


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text())


@pytest.mark.parametrize("r", RUNGS)
def test_base_model_matches_recorded_digest(r, tables, golden):
    got = _digest(r, tables)
    want = golden["rungs"][str(r)]
    assert got["batch_sha256"] == want["batch_sha256"], "synthetic batch changed (data, not model)"
    for k in ("n_parameters", "state_dict_keys_sha256"):
        assert got[k] == want[k], f"rung {r}: {k} differs from the recorded base model"
    if (golden.get("torch"), golden.get("platform")) != (torch.__version__, sys.platform):
        pytest.skip("value digests are platform- and torch-specific; structure checked only")
    for k in ("state_dict_sha256", "init", "perturbed"):
        assert got[k] == want[k], f"rung {r}: {k} differs from the recorded base model"


if __name__ == "__main__" and "--record" in sys.argv:
    with tempfile.TemporaryDirectory() as d:
        td = _make_tables(Path(d))
        rec = {"recorded_on": "exp/t370-minicandi-recipe 57f3c9e8 (pre-change tree)",
               "torch": torch.__version__, "platform": sys.platform,
               "rungs": {str(r): _digest(r, td) for r in RUNGS}}
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(json.dumps({r: v["n_parameters"] for r, v in rec["rungs"].items()}))
