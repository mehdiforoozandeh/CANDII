"""Activation probes for the metadata -> FiLM conditioning pathway (h75).

NEW FILE. Read-only instrumentation: forward hooks that measure what the conditioning signal looks
like on either side of the metadata-fusion LayerNorm and on either side of `decoder.film_proj`.
Nothing here participates in the graph, draws from any RNG, or touches the loss — the probe is
attached to a live model and the model computes exactly what it computed without it.

WHAT IS MEASURED, and why two quantities per site.

The captured tensors are `[B, A, E]`. The leading `B*A` axis is the PROMPT axis: one column per
(biosample, assay) conditioning prompt. Two different things can be called "the magnitude of the
metadata signal", and they behave very differently across a LayerNorm:

  * `_norm`   — mean over prompts of the per-prompt L2 norm. The AMBIENT activation magnitude.
  * `_spread` — subtract the mean prompt vector (mean over the `B*A` axis) and take the mean L2 norm
                of what is left. The CONDITIONING PERTURBATION: the across-prompt variation, which is
                exactly what a difference between two prompts is made of. A pathway can carry a large
                ambient norm and a vanishing spread, and only the spread can steer anything.

The source finding this instrument was built for (h48/F2) reports four numbers across the LayerNorm
and its prose does not say which of them are ambient norms and which are perturbations. Both
quantities are logged on both sides so the reading does not depend on resolving that.

`meta/perturbation_attenuation` = pre_ln_spread / post_ln_spread is the headline number.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

__all__ = ["MetaPathProbe", "effective_rank", "prompt_norm", "prompt_spread"]

_NAN = float("nan")
# Denominator floor for every ratio below. Ratios are reported as nan rather than inf/ZeroDivisionError
# so a dead pathway reads as "not measurable" instead of as a huge attenuation.
_DIV_EPS = 1e-12


def _safe_div(num: float, den: float) -> float:
    if not (abs(den) > _DIV_EPS):
        return _NAN
    return float(num) / float(den)


def _as_prompts(x: torch.Tensor) -> torch.Tensor:
    """`[B, A, E]` (or any leading shape) -> `[B*A, E]`, float32, detached."""
    return x.detach().reshape(-1, x.shape[-1]).float()


def prompt_norm(x: torch.Tensor) -> float:
    """Mean over prompts of the per-prompt L2 norm — the ambient activation magnitude."""
    m = _as_prompts(x)
    if m.numel() == 0:
        return _NAN
    return float(m.norm(dim=-1).mean())


def prompt_spread(x: torch.Tensor) -> float:
    """Mean L2 norm of each prompt's deviation from the mean prompt — the conditioning perturbation."""
    m = _as_prompts(x)
    if m.shape[0] < 1 or m.numel() == 0:
        return _NAN
    return float((m - m.mean(dim=0, keepdim=True)).norm(dim=-1).mean())


def effective_rank(x: torch.Tensor) -> float:
    """Entropy-based effective rank of a `[N, C]` matrix: `exp(-sum p_i log p_i)`, `p = s / sum(s)`.

    A rank-1 matrix reads ~1.0; a matrix with C equal singular values reads exactly C. An SVD
    convergence failure yields nan rather than an exception — this is a logging path.
    """
    try:
        m = _as_prompts(x)
        if m.shape[0] < 1 or m.shape[1] < 1:
            return _NAN
        s = torch.linalg.svdvals(m)
        tot = float(s.sum())
        if not (tot > _DIV_EPS):
            return _NAN
        p = (s / tot).clamp_min(1e-30)
        return float(torch.exp(-(p * p.log()).sum()))
    except Exception:                                  # noqa: BLE001 — instrumentation never raises
        return _NAN


class MetaPathProbe:
    """Forward hooks on the metadata-fusion LayerNorm and on `decoder.film_proj`.

    Usage per logging step:

        probe.arm()                 # capture the NEXT forward pass, and only that one
        out = model(...)            # unchanged, bit-exact, same autograd graph
        terms.update(probe.read())  # dict[str, float], buffer cleared

    When not armed the hooks return on their first statement: no tensor work, no allocation, no device
    synchronisation. That keeps the cost at exactly zero on the steps that do not log.

    LN-OFF ARM. `meta_embed_layernorm=False` removes the LayerNorm from `fusion`, so there is no
    "pre" and "post" to separate. The probe then hooks the last element of `fusion` (the final Linear)
    and reports its output as BOTH the pre and the post tensor. The key set is therefore identical
    across the two arms and directly comparable; `meta/ln_present` (1.0 / 0.0) says which regime
    produced it, and `meta/ln_gain` and `meta/perturbation_attenuation` read exactly 1.0 by
    construction when it is absent.
    """

    def __init__(self, model: nn.Module) -> None:
        self._armed = False
        self._buf: Dict[str, float] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

        decoder = model.decoder
        fusion = decoder.meta_embedding.fusion
        tail = fusion[-1]
        self.ln_present = isinstance(tail, nn.LayerNorm)
        self._handles.append(tail.register_forward_hook(self._fusion_hook))

        # candi_kit has ONE decoder FiLM projection; candi's SymmetricDecoder has one per tap
        # (after input_proj, then after each deconv). Every tap is hooked and keyed `meta/tN_*`; the
        # LAST tap is ALSO written to the historical unprefixed keys, so h75-era dashboards keep
        # resolving and the arms stay directly comparable on the same panel.
        if hasattr(decoder, "film_proj"):
            self._film_taps = [("t0", decoder.film_proj)]
        else:
            self._film_taps = [(f"t{i}", f.proj) for i, f in enumerate(decoder.film_layers)]
        if not self._film_taps:
            raise AttributeError("decoder exposes neither film_proj nor film_layers")
        self._last_tap = self._film_taps[-1][0]
        for tap, mod in self._film_taps:
            self._handles.append(
                mod.register_forward_hook(self._make_film_hook(tap)))

    # -- lifecycle ---------------------------------------------------------------------------------

    def arm(self) -> None:
        """Capture exactly the next forward pass. Auto-disarms once `film_proj` has fired."""
        self._buf.clear()
        self._armed = True

    def read(self) -> Dict[str, float]:
        """Return the captured scalars and clear the buffer. Empty dict when nothing was captured."""
        out = dict(self._buf)
        self._buf.clear()
        self._armed = False
        return out

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._armed = False
        self._buf.clear()

    def __enter__(self) -> "MetaPathProbe":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- hooks -------------------------------------------------------------------------------------

    def _fusion_hook(self, module, inputs, output) -> None:
        if not self._armed:
            return
        with torch.no_grad():
            post = output.detach()
            pre = inputs[0].detach() if self.ln_present else post
            pre_norm, post_norm = prompt_norm(pre), prompt_norm(post)
            pre_spread, post_spread = prompt_spread(pre), prompt_spread(post)
            self._buf.update({
                "meta/ln_present": 1.0 if self.ln_present else 0.0,
                "meta/pre_ln_norm": pre_norm,
                "meta/post_ln_norm": post_norm,
                "meta/ln_gain": _safe_div(post_norm, pre_norm),
                "meta/pre_ln_spread": pre_spread,
                "meta/post_ln_spread": post_spread,
                "meta/perturbation_attenuation": _safe_div(pre_spread, post_spread),
                "meta/pre_ln_snr": _safe_div(pre_spread, pre_norm),
                "meta/post_ln_snr": _safe_div(post_spread, post_norm),
            })

    def _make_film_hook(self, tap: str):
        def _film_hook(module, inputs, output) -> None:
            if not self._armed:
                return
            with torch.no_grad():
                fin = inputs[0].detach()
                fout = output.detach()
                in_norm, out_norm = prompt_norm(fin), prompt_norm(fout)
                in_spread, out_spread = prompt_spread(fin), prompt_spread(fout)
                # the decoder's own split, reproduced from the CAPTURED output — layer is not re-run
                gamma, beta = fout.chunk(2, dim=-1)
                vals = {
                    "film_in_norm": in_norm,
                    "film_out_norm": out_norm,
                    "film_gain": _safe_div(out_norm, in_norm),
                    "film_in_spread": in_spread,
                    "film_out_spread": out_spread,
                    "film_attenuation": _safe_div(in_spread, out_spread),
                    "gamma_absmax": float(gamma.abs().max()) if gamma.numel() else _NAN,
                    "beta_absmax": float(beta.abs().max()) if beta.numel() else _NAN,
                    "gamma_effrank": effective_rank(gamma),
                    "beta_effrank": effective_rank(beta),
                }
                self._buf.update({f"meta/{tap}_{k}": v for k, v in vals.items()})
                if tap == self._last_tap:
                    self._buf.update({f"meta/{k}": v for k, v in vals.items()})
            # one forward pass per arm(): disarm after the LAST tap in the decoder's own order
            if tap == self._last_tap:
                self._armed = False
        return _film_hook


def make_meta_path_probe(model: nn.Module) -> Optional[MetaPathProbe]:
    """Best-effort constructor: returns None instead of raising if the model has no such pathway."""
    try:
        return MetaPathProbe(model)
    except Exception as e:                             # noqa: BLE001 — instrumentation never raises
        print(f"[probe] DISABLED: {type(e).__name__}: {e}", flush=True)
        return None
