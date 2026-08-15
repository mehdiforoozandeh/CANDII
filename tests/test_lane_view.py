"""The encoder's lane view is a view — it must never change a number.

`lanes_from_channels` / `channels_from_lanes` name the convention that grouped `Conv1d` leaves
implicit: track `t` owns channels `[t*C, (t+1)*C)`. Everything here exists to prove the naming is
free — same values, same order, same storage, and `FiLMLayer` computing the identical product it
computed under the old flat `view(bsz, channels)` form.

The one test that is NOT about the refactor is `test_lane_shapes_match_a_real_forward`: it pins the
documented per-track shape table to an actual forward pass, so the channel schedule can never drift
away from the comment that describes it.
"""
import pytest
import torch

from candi.encoder import (
    FiLMLayer,
    SignalConvTower,
    channels_from_lanes,
    lanes_from_channels,
)

T, C, L, B = 36, 8, 96, 2


def test_round_trip_is_identity():
    x = torch.randn(B, T * C, L)
    assert torch.equal(channels_from_lanes(lanes_from_channels(x, T)), x)


def test_lane_view_shares_storage():
    """A copy would be a silent memory cost on every FiLM tap — assert it is a real view."""
    x = torch.randn(B, T * C, L)
    assert lanes_from_channels(x, T).data_ptr() == x.data_ptr()


def test_lane_axis_matches_the_grouped_conv_convention():
    """Track t's slice must be exactly channels [t*C, (t+1)*C) — the layout groups=T produces."""
    x = torch.randn(B, T * C, L)
    lanes = lanes_from_channels(x, T)
    for t in (0, 1, T // 2, T - 1):
        assert torch.equal(lanes[:, t], x[:, t * C:(t + 1) * C])


def test_rejects_a_ragged_split():
    with pytest.raises(ValueError, match="not divisible"):
        lanes_from_channels(torch.randn(B, T * C + 1, L), T)


def test_film_matches_the_flat_formula_bit_for_bit():
    """The pre-refactor arithmetic, written out by hand, to 0 ULP."""
    torch.manual_seed(0)
    film = FiLMLayer(32, 2 * C).eval()
    x = torch.randn(B, T * C, L)
    memb = torch.randn(B, T, 32)

    with torch.no_grad():
        got = film(x, memb)
        scale, shift = film.proj(memb).chunk(2, dim=-1)
        scale = scale.contiguous().view(B, T * C).unsqueeze(-1)
        shift = shift.contiguous().view(B, T * C).unsqueeze(-1)
        want = x * (1.0 + scale) + shift

    assert torch.equal(got, want)


def test_film_modulates_each_track_independently():
    """Perturbing track 5's metadata must move track 5 and nothing else."""
    torch.manual_seed(0)
    film = FiLMLayer(32, 2 * C).eval()
    x = torch.randn(B, T * C, L)
    memb = torch.randn(B, T, 32)
    bumped = memb.clone()
    bumped[:, 5] += 3.0

    with torch.no_grad():
        delta = (film(x, bumped) - film(x, memb)).abs()
    lanes = lanes_from_channels(delta, T)

    assert lanes[:, 5].max() > 0
    others = torch.cat([lanes[:, :5], lanes[:, 6:]], dim=1)
    assert others.max() == 0


def test_film_rejects_a_width_mismatch():
    """A projection sized for the wrong panel used to misalign silently via the flat view."""
    film = FiLMLayer(32, 2 * C)
    with pytest.raises(ValueError, match="width mismatch"):
        film(torch.randn(B, T * (C + 1), L), torch.randn(B, T, 32))


def _tower():
    return SignalConvTower(
        num_tracks=T, n_layers=3, expansion_factor=2, kernel_size=3, pool_size=2,
        meta_embed_dim=32, conv_norm="layer", film_mode="per_conv",
    ).eval()


def test_return_lanes_is_the_same_numbers_as_the_flat_return():
    torch.manual_seed(0)
    tower = _tower()
    x = torch.randn(B, 768, T)
    memb = torch.randn(B, T, 32)

    with torch.no_grad():
        flat = tower(x, memb)
        lanes = tower(x, memb, return_lanes=True)

    assert lanes.shape == (B, L, T, C)
    assert torch.equal(lanes.reshape(B, L, T * C), flat)


def test_lane_shapes_match_a_real_forward():
    """The documented per-track table is asserted, not narrated."""
    torch.manual_seed(0)
    tower = _tower()
    x = torch.randn(B, 768, T)
    memb = torch.randn(B, T, 32)

    documented = tower.lane_shapes(768, batch=str(B))
    assert documented == [
        f"[{B}, 768, 36, 1]",
        f"[{B}, 384, 36, 2]",
        f"[{B}, 192, 36, 4]",
        f"[{B}, 96, 36, 8]",
    ]

    observed = [f"[{B}, 768, {T}, 1]"]
    h = x.permute(0, 2, 1).contiguous()
    with torch.no_grad():
        for i, block in enumerate(tower.blocks):
            h = block(h)
            h = tower.per_conv_film_layers[i](h, memb)
            lanes = lanes_from_channels(h, T)
            observed.append(f"[{B}, {lanes.shape[3]}, {lanes.shape[1]}, {lanes.shape[2]}]")

    assert observed == documented


def test_lane_shapes_does_not_enter_the_state_dict():
    """The schedule is bookkeeping; a buffer would change every checkpoint digest."""
    keys = set(_tower().state_dict())
    assert not any("_out_channels_list" in k or "_pool_sizes" in k for k in keys)
