"""Mixed precision: one flag, two contexts, and the reason fp16 is not on the menu.

`--precision bf16` wraps the training forward in `torch.autocast`. Everything the audit found to be
low-magnitude, reduction-heavy, or the objective itself is fenced back into fp32 by `fp32_fence`,
which is called from inside the modules rather than from the training loop — a fence written at the
call site protects one caller, a fence written in the module protects every caller, including
`healthcheck` and anything a later reader adds.

WHY fp16 IS NOT A CHOICE, AND MUST NOT BECOME ONE
-------------------------------------------------
`SymmetricDecoder` clamps `log2_mu` at a ceiling of 30, so `mu = 2**log2_mu` legitimately reaches
2**30 ~ 1.07e9. fp16's largest finite value is 65504 = 2**15.99, so that `mu` overflows to `inf`.
The overflow then does not survive to be seen: the next line is
`p = (n / (n + mu)).clamp(eps, 1 - eps)`, and `n / inf` is 0.0, which the clamp turns into `eps` — a
finite, ordinary-looking probability. No NaN, no inf, no warning, and a calibration curve that is
merely wrong rather than obviously broken. bf16 carries fp32's 8-bit exponent and therefore fp32's
range, so the same ceiling is nowhere near it. That is the whole reason bf16 is the offer.

WHAT TO EXPECT FROM IT
----------------------
Memory, not speed. The activations halve; the parameters and the optimizer state do not, because
autocast keeps master weights in fp32. On the MIG slice this buys batch size. Any speedup is
incidental and this repo has not measured one, so a null speed result is the expected result and
not a bug.
"""
from __future__ import annotations

import contextlib
from typing import Iterator, Tuple

import torch

__all__ = ["PRECISIONS", "DEFAULT_PRECISION", "PRECISION_HELP", "autocast_region", "fp32_fence",
           "no_autocast", "assert_no_grad_scaler"]


# fp16 is absent DELIBERATELY — see the module docstring. Adding it here is not a one-line change.
PRECISIONS: Tuple[str, ...] = ("fp32", "bf16")
DEFAULT_PRECISION = "fp32"

PRECISION_HELP = (
    "fp32 (default, the recorded numerics) or bf16 mixed precision. bf16 buys MEMORY, not speed — "
    "activations halve, master weights and optimizer state stay fp32 — so a null speed result is "
    "the expected result, not a bug. fp16 is deliberately not offered: the decoder's log2_mu "
    "ceiling of 30 overflows fp16 and the NB head launders the inf into a plausible probability. "
    "The metadata embedders, every FiLM tap, LaneNorm, the NB head arithmetic and the whole of "
    "eval are fenced back into fp32 regardless of this flag."
)


def _device_type(device) -> str:
    """`'cuda'` / `'cpu'` from a string, a `torch.device`, or an index-carrying spelling."""
    return torch.device(device).type


def autocast_region(device, precision: str = DEFAULT_PRECISION):
    """The training forward's autocast context. `fp32` returns a context that changes nothing.

    Enabled on WHATEVER device is passed, CPU included. Gating it to CUDA would make
    `--precision bf16` a silent no-op on a CPU box — the exact failure `tests/test_flags.py` was
    written to catch — and it would leave the flag untestable anywhere without a GPU. The memory
    win is still a GPU win; the CPU path exists so the switch can be proven to reach the arithmetic.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}; got {precision!r}")
    return torch.amp.autocast(device_type=_device_type(device), dtype=torch.bfloat16,
                              enabled=(precision == "bf16"))


@contextlib.contextmanager
def fp32_fence(*tensors) -> Iterator[Tuple]:
    """Run a block in fp32 whatever autocast is doing outside it, and hand back its inputs cast.

    TWO THINGS ARE NEEDED AND ONLY ONE OF THEM IS OBVIOUS. Disabling autocast stops the ops inside
    the block from being demoted. It does nothing about a tensor that arrived ALREADY demoted — an
    activation produced by an autocast-ed matmul upstream is bf16 when it crosses the fence, and
    every op in here would then run in bf16 with autocast switched off. That is the failure this
    exists to prevent, wearing the disguise of the fix. So the cast and the disable are one call and
    cannot be half-applied:

        with fp32_fence(z, memb) as (z, memb):
            ...

    In fp32 this is exactly a no-op, which is why the golden gate still reads 0 ULP: nothing is
    enabled to disable, and `Tensor.float()` on an fp32 tensor returns the same object rather than a
    rounded copy. Non-float arguments (index tensors, None) pass through untouched — casting an
    integer index to float would corrupt an embedding lookup.

    The device type is read off the first tensor rather than passed in, so a module that has no idea
    which device it was moved to still fences correctly.
    """
    ref = next((t for t in tensors if torch.is_tensor(t)), None)
    dev = "cuda" if ref is None else ref.device.type
    with torch.amp.autocast(device_type=dev, enabled=False):
        yield tuple(t.float() if torch.is_tensor(t) and t.is_floating_point() else t
                    for t in tensors)


def no_autocast(device):
    """Disable autocast for a whole call tree. The entry-point form of `fp32_fence`.

    EVALUATION IS NEVER AUTOCAST. Every recorded number in this repo was measured in fp32, and a
    metric measured at a different precision than the one it is compared against is a difference
    nobody declared. It is used at the evaluation ENTRY POINT — `bench.cli.main` — rather than at
    each of the dozen forwards below it, because the guarantee wanted is "nothing under here", not
    "not this line".

    Takes a device instead of a tensor because at an entry point the device is what is in scope,
    and there is no activation yet to read it from.
    """
    return torch.amp.autocast(device_type=_device_type(device), enabled=False)


def assert_no_grad_scaler(scaler) -> None:
    """Refuse a `GradScaler`. bf16 does not need one, and one added quietly corrupts the grad probe.

    Loss scaling exists to stop fp16 gradients underflowing to zero in its 5-bit exponent. bf16 has
    fp32's 8-bit exponent, so there is nothing to rescue and a scaler is pure overhead — the
    PyTorch AMP recipe says as much. This repo therefore ships none.

    THE REASON THIS IS AN ASSERT AND NOT A COMMENT: `train._grad_norms` reads `p.grad` between
    `backward()` and `clip_grad_norm_`. With a scaler in play those gradients are still multiplied
    by the scale factor at that point, so every `grad/*` number and every `*_over_trunk` ratio would
    become a property of the scaler's current — and dynamically changing — scale rather than of the
    model. The whole probe exists to catch a covariate that cannot steer the model, and a drifting
    common factor is exactly what would hide one. If a scaler is ever genuinely needed, the fix is
    `scaler.unscale_(opt)` BEFORE the probe reads, not the removal of this line.
    """
    if scaler is not None:
        raise ValueError(
            "a GradScaler was passed, but this training loop has no unscale_ step. bf16 needs no "
            "scaler (it has fp32's exponent range), and fp16 is not offered. Adding one requires "
            "calling scaler.unscale_(opt) before train._grad_norms reads p.grad, or every reported "
            "gradient norm becomes a property of the scale factor instead of the model.")
