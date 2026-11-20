"""Two-step correction through the base covariates (decision D15; plan pinned interface E).

    wrap(inner: MiniCandi, config: dict) -> TwoStep

Leg 1. The inner model reads the batch as given, with `c_out := c_base`, and predicts the base
view of every target track, `(n_base, mu_base)`. A target track with no base view
(`c_base[..., 0] == 0`; log2 depth is never 0 for a real view) uses its own `c_out` in leg 1.
Call the per-track covariates that leg 1 decodes at `c_leg1` (= `c_base` where a base view
exists, else `c_out`).

Leg 2. The same inner model (shared weights) reads `x_counts := mu_base` (float, not detached,
and 0 on tracks without a target), `x_avail := y_avail`, `x_bin_mask := False`,
`c_in := c_leg1` (the covariates mu_base was decoded at, 0 on tracks without a target) and the
batch's own `c_out`. Its output is the correction to the target view.

`forward` returns leg 2's `{"n", "mu"}` plus leg 1's `{"n_base", "mu_base"}`. It reads only the
input side of the batch, `y_avail`, `c_out` and `c_base`; it never reads `y_counts` or `y_base`
(the base leg's loss lives in `losses.total_loss`). Guardrail part 1 holds because leg 1 is the
inner model on the batch's own inputs (J-invariant by the rung's construction) and leg 2 reads
only leg 1's output.

`two_step` with `qc_on_input` raises NotApplicable: the base view has no QC vector, so leg 2's
`c_in` would lack the 7 QC columns. The wrapper adds no parameters; it runs the inner model
twice, so its FLOPs per window are about twice the inner model's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from minicandi import contract as C  # noqa: E402
from minicandi.components import NotApplicable  # noqa: E402

QC_MESSAGE = ("two_step (D15) with qc_on_input: the base view has no QC vector, so leg 2 "
              "cannot read c_in with the QC columns")


class TwoStep(C.MiniCandi):
    """Leg 1 predicts the base view, leg 2 corrects it to the target view (module docstring)."""

    def __init__(self, inner: C.MiniCandi, config: dict):
        super().__init__()
        if config.get("qc_on_input"):
            raise NotApplicable(QC_MESSAGE)
        if not isinstance(inner, C.MiniCandi):
            raise TypeError(f"{type(inner).__name__} is not a contract.MiniCandi")
        self.inner = inner
        self.d_model = getattr(inner, "d_model", None)

    @staticmethod
    def leg1_covariates(batch: dict) -> torch.Tensor:
        """c_leg1 [B, A, K_META]: c_base where the track has a base view, else c_out."""
        if "c_base" not in batch:
            raise KeyError("two_step needs batch['c_base']: build the dataset with the same "
                           "config (two_step = true)")
        c_base, c_out = batch["c_base"], batch["c_out"]
        has_base = c_base[..., C.LOG2_DEPTH_COL] != 0
        return torch.where(has_base[..., None], c_base, c_out)

    def forward(self, batch: dict) -> dict:
        c_leg1 = self.leg1_covariates(batch)
        leg1 = dict(batch)
        leg1["c_out"] = c_leg1
        o1 = self.inner(leg1)
        tgt = batch["y_avail"]
        leg2 = dict(batch)
        leg2["x_counts"] = o1["mu"] * tgt[:, None, :].to(o1["mu"].dtype)
        leg2["x_avail"] = tgt
        leg2["x_bin_mask"] = torch.zeros_like(batch["x_bin_mask"])
        leg2["c_in"] = c_leg1 * tgt[..., None].to(c_leg1.dtype)
        o2 = self.inner(leg2)
        return {"n": o2["n"], "mu": o2["mu"], "n_base": o1["n"], "mu_base": o1["mu"]}


def wrap(inner: C.MiniCandi, config: dict) -> TwoStep:
    return TwoStep(inner, config)


__all__ = ["QC_MESSAGE", "TwoStep", "wrap"]
