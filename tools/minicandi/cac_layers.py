"""Pre and post covariate-correction layers (t370 chunk C2, decision D5; plan pinned interface C2).

    wrap(inner: MiniCandi, config: dict) -> CacLayers

The in-vitro CAC testbed form (t118 ladder): a generator g reads ONLY covariates and outputs the
parameters theta of a count map f_theta. Both sides act per track in log1p space.

  cac_pre   g_pre(emb_pre(c_in)) -> theta; x = log1p(x_counts) with every position that is not
            visible (`MaskHandler.visible(x_avail, x_bin_mask)` false) set to 0; loc = f(x, theta);
            the inner model reads a shallow copy of the batch with
            x_counts = expm1(loc).clamp_min(0) (float32). f removes the C artifact from the input.
  cac_post  out = inner(batch'); z = log1p(out["mu"]); theta = g_post(emb_post(c_out));
            loc, disp = f(z, theta); mu = expm1(loc) in [MU_MIN, MU_MAX];
            n = out["n"] * exp(disp) in [N_MIN, N_MAX]. f re-applies the C' artifact. Every other
            output key passes through.

Values of each key: absent or None (no layer), "curve" (`ladder.fforms.form_b.FormB`, the
monotone 12-knot curve) or "cnn" (`ladder.fforms.form_d.FormD`, the FiLM-conditioned dilated CNN;
30 zero bins of halo on each side). g is `ladder.model.G`; its last layer starts at
`f.init_theta()`, so both layers are the identity at the start. Both forms use the fixed knots
`KNOTS` and `STATS` (no fitted state). Each side has its own `CovariateEmbedding` and its own g.
g never reads counts or the track identity.

Sizes. The 1.10 x `param_budget` rule must admit `cac_pre = cac_post = cnn` on the conv
autoencoder (1,042,514 parameters at budget 1e6, so 57,486 to spare). With the testbed sizes
(FormD 32 channels, G hidden 64, embedding d_cat 8, d_cov 32) one cnn side alone is about 93 k
parameters, so this module uses smaller widths of the same classes: embedding d_cat 2, d_cov 16,
hidden 32; G hidden 32; FormD 16 channels. One curve side is 15,784 parameters and one cnn side
23,234 (the test file pins both).

Core FiLM off (D7) needs no key here: `covariate_drop_fields` neutralises the inner model's
covariates, while this wrapper reads the raw `batch["c_in"]` and `batch["c_out"]`.

Copy-leak guardrail part 1 holds: hidden-bin and missing-track counts are zeroed before f, g reads
no counts, and the post side reads only the inner model's output.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

_TOOLS = Path(__file__).resolve().parents[1]
for _p in (_TOOLS, _TOOLS / "t118"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder.fforms.form_b import FormB  # noqa: E402
from ladder.fforms.form_d import FormD  # noqa: E402
from ladder.model import G  # noqa: E402
from minicandi import contract as C  # noqa: E402
from minicandi.components import MaskHandler  # noqa: E402
from minicandi.covariates import CovariateEmbedding  # noqa: E402

FORMS = ("curve", "cnn")
KEYS = ("cac_pre", "cac_post")
KNOTS = np.linspace(0.0, 9.0, 12)
STATS = {"knots_x": KNOTS, "n0": 1.0, "sigma0": 1.0}
#: widths (module docstring, "Sizes")
EMB_D_CAT, EMB_D_OUT, EMB_HIDDEN = 2, 16, 32
G_HIDDEN = 32
CNN_CHANNELS = 16
BUDGET_SLACK = 1.10


def form_of(config: dict, key: str):
    v = config.get(key)
    if v is not None and v not in FORMS:
        raise ValueError(f"{key} {v!r} not in (None,) + {FORMS}")
    return v


# FormB extrapolates linearly past its last knot, so loc and disp are unbounded. Capping the
# exponent arguments keeps exp finite (an inf then clamped gives a 0 x inf = NaN gradient). The
# caps bind only beyond MU_MAX (1e6 counts per bin) and |disp| > 30, so normal outputs are unchanged.
LOC_CAP = math.log1p(C.MU_MAX)
DISP_CAP = 30.0


def _make_f(form: str) -> nn.Module:
    if form == "curve":
        return FormB("counts", STATS)
    return FormD("counts", STATS, channels=CNN_CHANNELS)


class CacSide(nn.Module):
    """emb + g + f of one side. `theta(c)` [B, A, n_theta]; `forward(v, c)` maps v [B, L, A]."""

    def __init__(self, form: str, k_in: int):
        super().__init__()
        self.form = form
        self.embed = CovariateEmbedding(k_in, EMB_D_OUT, d_cat=EMB_D_CAT, hidden=EMB_HIDDEN)
        self.f = _make_f(form)
        self.g = G(EMB_D_OUT, self.f.n_theta, self.f.init_theta(), hidden=G_HIDDEN)

    def theta(self, c: torch.Tensor) -> torch.Tensor:
        return self.g(self.embed(c))

    def forward(self, v: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, A = v.shape
        th = self.theta(c).reshape(B * A, -1)
        x = v.permute(0, 2, 1).reshape(B * A, L)
        h = self.f.context
        if h:
            x = F.pad(x, (h, h))
        loc, disp = self.f(x, th)
        back = lambda t: t.reshape(B, A, L).permute(0, 2, 1)  # noqa: E731
        return back(loc), back(disp)


class CacLayers(C.MiniCandi):
    """The inner model between the pre layer (on its input) and the post layer (on its NB)."""

    def __init__(self, inner: C.MiniCandi, config: dict):
        super().__init__()
        if not isinstance(inner, C.MiniCandi):
            raise TypeError(f"{type(inner).__name__} is not a contract.MiniCandi")
        pre, post = form_of(config, "cac_pre"), form_of(config, "cac_post")
        if pre is None and post is None:
            raise ValueError("CacLayers needs cac_pre or cac_post")
        self.inner = inner
        self.d_model = getattr(inner, "d_model", None)
        self.pre = CacSide(pre, C.k_input(config)) if pre else None
        self.post = CacSide(post, C.K_META) if post else None

    def forward(self, batch: dict) -> dict:
        if self.pre is not None:
            vis = MaskHandler.visible(batch["x_avail"], batch["x_bin_mask"])
            x = torch.log1p(batch["x_counts"].to(torch.float32).clamp_min(0))
            x = torch.where(vis, x, torch.zeros_like(x))
            loc, _ = self.pre(x, batch["c_in"])
            batch = dict(batch)
            batch["x_counts"] = torch.expm1(loc.clamp(max=LOC_CAP)).clamp_min(0).to(torch.float32)
        out = self.inner(batch)
        if self.post is None:
            return out
        out = dict(out)
        z = torch.log1p(out["mu"])
        loc, disp = self.post(z, batch["c_out"])
        out["mu"] = torch.expm1(loc.clamp(max=LOC_CAP)).clamp(C.MU_MIN, C.MU_MAX)
        out["n"] = (out["n"] * torch.exp(disp.clamp(-DISP_CAP, DISP_CAP))).clamp(C.N_MIN, C.N_MAX)
        return out


def wrap(inner: C.MiniCandi, config: dict) -> CacLayers:
    model = CacLayers(inner, config)
    budget = config.get("param_budget")
    if budget is not None and model.param_count() > BUDGET_SLACK * float(budget):
        raise ValueError(f"cac layers: {model.param_count()} parameters > {BUDGET_SLACK} x "
                         f"param_budget {budget}")
    return model


__all__ = ["BUDGET_SLACK", "CacLayers", "CacSide", "FORMS", "KEYS", "KNOTS", "STATS", "form_of",
           "wrap"]
