"""Ablation 7 (cov_embedding): the `numeric_per_value` covariate embedding.

The off switch was also checked against the unchanged code out of band (base state_dict and one
perturbed forward on a synthetic batch, equal on all three architectures); here the base build is
held to its published parameter counts and to having no per-value tensors."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import contract as C  # noqa: E402
from minicandi import covariates as COV  # noqa: E402
from minicandi import registry as R  # noqa: E402

ABL = _ROOT / "tools/minicandi/configs/ablations/cov_embedding"
#: rung -> (base parameter count, trunk width), from run.json of runs_fir/rung<r>_base_heldout
BASE = {2: (1_042_514, 60), 3: (1_038_018, 104), 4: (993_599, 25)}
#: per embedding: 4 tables x PV_TABLE x d_cat, plus 4 x (d_cat - 1) more inputs to the first layer
PV_EXTRA = 4 * COV.PV_TABLE * 8 + 4 * (8 - 1) * 64


def _row(read_length=36.0, run_type=0.0, dedup=1.0, mapq=30.0) -> torch.Tensor:
    v = torch.zeros(C.K_META)
    v[:5] = torch.tensor([25.0, run_type, read_length, dedup, mapq])
    v[5:] = torch.tensor([1.0, 1.0, 0.0, 3.0, 0.0, 0.0])
    return v


def _perturb(m: torch.nn.Module) -> torch.nn.Module:
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return m


def test_per_value_embedding_builds_with_the_expected_count():
    base = COV.CovariateEmbedding(C.K_META, 32)
    pv = COV.CovariateEmbedding(C.K_META, 32, per_value=True)
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert n(pv) - n(base) == PV_EXTRA
    assert not any(k.startswith("pv_") for k in base.state_dict())
    assert pv(torch.stack([_row(), _row(50.0)])).shape == (2, 32)


def test_gradient_reaches_every_per_value_table_and_values_are_separate():
    pv = _perturb(COV.CovariateEmbedding(C.K_META, 32, per_value=True))
    c = torch.stack([_row(36.0, 0.0, 1.0, 30.0), _row(50.0, 1.0, 0.0, 0.0)])
    pv(c).square().sum().backward()
    for t in pv.pv_tables:
        assert t.weight.grad is not None and t.weight.grad.abs().sum() > 0
    # 36 and 50 bp are different rows; 36.2 rounds to the 36 bp row
    e = pv(torch.stack([_row(36.0), _row(50.0), _row(36.2)]))
    assert (e[0] - e[1]).abs().max() > 1e-3 and torch.allclose(e[0], e[2], atol=1e-6)
    # flags use id = round(2 x value): 0, 0.5 (empty cell) and 1 are three rows
    f = pv(torch.stack([_row(run_type=0.0), _row(run_type=0.5), _row(run_type=1.0)]))
    assert min((f[i] - f[j]).abs().max() for i, j in ((0, 1), (0, 2), (1, 2))) > 1e-3


def test_per_value_keeps_the_null_vector_path():
    pv = COV.CovariateEmbedding(C.K_META, 32, per_value=True, null_vector=True)
    c = torch.stack([_row(), torch.full((C.K_META,), float("nan"))])
    e = pv(c)
    assert torch.equal(e[1], pv.null.detach())


@pytest.mark.parametrize("r", [2, 3, 4])
def test_configs_build_and_base_is_unchanged(r):
    ref = json.loads((_ROOT / f"tools/minicandi/configs/final/rung{r}/reference_equal.json")
                     .read_text())
    torch.manual_seed(0)
    m = R.build_model(ref)
    assert (sum(p.numel() for p in m.parameters()), m.width) == BASE[r]
    assert not any(".pv_" in k for k in m.state_dict())
    cfg = json.loads((ABL / f"rung{r}/cov_numeric_per_value.json").read_text())
    torch.manual_seed(0)
    pv = R.build_model(cfg)
    # the raised param_budget keeps the trunk width of the base; only the 2 embeddings grow
    assert pv.width == BASE[r][1]
    assert sum(p.numel() for p in pv.parameters()) == BASE[r][0] + 2 * PV_EXTRA
