"""E4 copy-leak guardrail: planted copy models fail, a planted J-invariant model passes."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import guardrail as G  # noqa: E402

A, K, K_META = 36, 18, 11


def same_view_batch(seed: int, B: int = 4, L: int = 512, mask_frac: float = 0.15) -> dict:
    """Same-view batches: y = x on available tracks, a random 15% of their bins hidden."""
    rng = np.random.default_rng(seed)
    t = np.arange(L)
    rate = np.empty((B, L, A))
    for b in range(B):
        prof = 1.0 + 0.8 * np.sin(2 * np.pi * t / rng.uniform(40, 120) + rng.uniform(0, 6))
        rate[b] = prof[:, None] * rng.uniform(2, 20, A)[None, :]
    x = rng.poisson(rate).astype(np.int32)
    avail = rng.random((B, A)) < 0.6
    x = np.where(avail[:, None, :], x, 0)
    bin_mask = (rng.random((B, L, A)) < mask_frac) & avail[:, None, :]
    c_in = np.where(avail[:, :, None], rng.normal(size=(B, A, K)), 0).astype(np.float32)
    return {
        "x_counts": torch.from_numpy(x), "x_avail": torch.from_numpy(avail),
        "x_bin_mask": torch.from_numpy(bin_mask),
        "x_dna": torch.zeros(B, L * 25, dtype=torch.uint8),
        "c_in": torch.from_numpy(c_in), "c_out": torch.from_numpy(c_in[:, :, :K_META].copy()),
        "y_counts": torch.from_numpy(x.copy()), "y_avail": torch.from_numpy(avail.copy()),
        "y_kind": torch.from_numpy(np.where(avail, 3, 0).astype(np.uint8)),
        "meta": [{"cell": f"C{b}", "chrom": "chr1", "start_bin": 1000 * b} for b in range(B)],
    }


def _neighbour_mean(batch: dict, half: int = 6) -> torch.Tensor:
    """J-invariant: the mean of the visible bins within +-half of a bin, the bin itself excluded."""
    x = batch["x_counts"].double()
    vis = (~batch["x_bin_mask"].bool() & batch["x_avail"].bool()[:, None, :]).double()
    k = torch.ones(1, 1, 2 * half + 1, dtype=torch.float64)
    k[0, 0, half] = 0.0
    B, L, Ah = x.shape
    flat = lambda v: v.permute(0, 2, 1).reshape(B * Ah, 1, L)            # noqa: E731
    s = F.conv1d(flat(x * vis), k, padding=half)
    c = F.conv1d(flat(vis), k, padding=half)
    m = (s / c.clamp(min=1.0)).reshape(B, Ah, L).permute(0, 2, 1)
    return m.clamp(min=1e-3)


def honest(batch):
    mu = _neighbour_mean(batch)
    return {"n": torch.full_like(mu, 8.0).float(), "mu": mu.float()}


def identity_copy(batch):
    """Echoes its input on every bin, the hidden ones included."""
    mu = batch["x_counts"].double().clamp(min=1e-3)
    return {"n": torch.full_like(mu, 1e3).float(), "mu": mu.float()}


def make_masked_copy(w: float):
    """Respects the mask, but on visible bins blends weight w of the bin's own value."""
    def model(batch):
        nb = _neighbour_mean(batch)
        vis = ~batch["x_bin_mask"].bool() & batch["x_avail"].bool()[:, None, :]
        own = batch["x_counts"].double().clamp(min=1e-3)
        mu = torch.where(vis, w * own + (1 - w) * nb, nb)
        return {"n": torch.full_like(mu, 8.0).float(), "mu": mu.float()}
    return model


def leaks_unavailable(batch):
    mu = _neighbour_mean(batch) + 1e-3 * batch["x_counts"].double().sum(-1, keepdim=True)
    return {"n": torch.full_like(mu, 8.0).float(), "mu": mu.float()}


def test_honest_passes_both_parts():
    r = G.copy_leak(honest, [same_view_batch(s) for s in range(3)])
    assert set(G.KEYS) <= set(r)
    assert r["hidden_input_max_abs_change"] == 0.0
    assert r["unavailable_input_max_abs_change"] == 0.0
    assert r["part1_wiring_pass"] and r["copy_leak_pass"]
    assert r["same_view_copy_gain_flag"] is False and r["gate_on_copy_gain"] is False


def test_identity_copy_fails_wiring():
    r = G.copy_leak(identity_copy, [same_view_batch(0)])
    assert r["hidden_input_max_abs_change"] > 1e-3
    assert not r["part1_wiring_pass"] and not r["copy_leak_pass"]


def test_masked_copy_is_flagged_but_passes_the_gate():
    """The PI's gate is part 1 only: a model that echoes visible bins but never reads a hidden one
    passes, with the same-view copy gain flagged."""
    r = G.copy_leak(make_masked_copy(1.0), [same_view_batch(0)])
    assert r["part1_wiring_pass"]                        # it never reads a hidden bin ...
    assert r["same_view_copy_gain_flag"] is True         # ... but it echoes the visible ones
    assert r["same_view_copy_gain"] > 0 and r["copy_leak_pass"] is True


def test_gate_on_copy_gain_restores_the_old_gate(monkeypatch):
    monkeypatch.setattr(G, "GATE_ON_COPY_GAIN", True)
    r = G.copy_leak(make_masked_copy(1.0), [same_view_batch(0)])
    assert r["gate_on_copy_gain"] is True and r["part1_wiring_pass"]
    assert r["same_view_copy_gain_flag"] is True and r["copy_leak_pass"] is False
    h = G.copy_leak(honest, [same_view_batch(s) for s in range(3)])
    assert h["copy_leak_pass"] is True


def test_default_switch_is_off():
    assert G.GATE_ON_COPY_GAIN is False


def test_reading_unavailable_track_fails():
    r = G.copy_leak(leaks_unavailable, [same_view_batch(0)])
    assert r["hidden_input_max_abs_change"] > 0      # it reads hidden values through the sum
    assert r["unavailable_input_max_abs_change"] > G.REL_TOL and not r["copy_leak_pass"]


def test_no_same_view_entries(monkeypatch):
    """No y_kind 3 entry: the gain is undefined and not flagged; the gate (part 1) still decides.
    With the copy gain gating, an undefined gain fails closed."""
    b = same_view_batch(0)
    b["y_kind"] = torch.where(b["y_kind"] == 3, torch.tensor(1, dtype=torch.uint8), b["y_kind"])
    r = G.copy_leak(honest, [b])
    assert np.isnan(r["same_view_copy_gain"]) and r["same_view_copy_gain_flag"] is False
    assert r["copy_leak_pass"] is True
    monkeypatch.setattr(G, "GATE_ON_COPY_GAIN", True)
    assert G.copy_leak(honest, [b])["copy_leak_pass"] is False


def test_no_hidden_bin_fails_closed():
    """Part 1 with no hidden bin did not run: it fails rather than passing vacuously."""
    b = same_view_batch(0)
    b["x_bin_mask"] = torch.zeros_like(b["x_bin_mask"])
    r = G.copy_leak(honest, [b])
    assert r["n_hidden_bins"] == 0 and not r["part1_wiring_pass"] and not r["copy_leak_pass"]


def calibration(seeds=range(5)):
    """Relative gain per planted model and seed: the evidence behind GAIN_THRESHOLD."""
    models = {"honest": honest, "copy_w0.25": make_masked_copy(0.25),
              "copy_w0.5": make_masked_copy(0.5), "copy_w1": make_masked_copy(1.0)}
    return {name: [G.copy_leak(m, [same_view_batch(100 + s)])["same_view_copy_gain_relative"]
                   for s in seeds] for name, m in models.items()}


def test_threshold_separates_planted_models():
    cal = calibration()
    honest_gain = np.array(cal["honest"])
    assert np.all(np.abs(honest_gain) < G.GAIN_THRESHOLD / 4)       # margin of 4x over seed noise
    assert np.min(cal["copy_w0.25"]) > G.GAIN_THRESHOLD
    assert np.min(cal["copy_w1"]) > 3 * G.GAIN_THRESHOLD


def test_cli_help_does_not_need_minicandi():
    out = subprocess.run([sys.executable, str(REPO / "tools/onion_eval/guardrail.py"), "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "--run" in out.stdout


if __name__ == "__main__":
    for k, v in calibration().items():
        print(k, np.round(v, 3))
