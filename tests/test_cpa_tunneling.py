"""Unit tests for continuous-time CPA collision and tunneling detection."""

import pytest
import torch

from src.physics.cpa_collisions import (
    friendly_collision_mask,
    intercept_mask,
    pairwise_cpa,
)

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize("device", DEVICES)
def test_head_on_tunneling_detection(device: str):
    """Head-on tunneling case: R_INT = 2.0 m, dt = 0.05.
    Drone at x=-10 moving +250 m/s, missile at x=+10 moving -250 m/s.
    Assert start/end distance > R_INT, but pairwise_cpa detects d_min ~= 0,
    s_star = 0.8 (exact impact at t=0.04s in dt=0.05), and intercept_mask is True.
    Also verifies symmetric case at x=±6.25 yielding s_star = 0.5."""
    R_INT = 2.0
    dt = 0.05

    # Case 1: Prompt coordinates (x = -10, +10)
    pA0 = torch.tensor([[[-10.0, 0.0, 0.0]]], device=device, dtype=torch.float32)
    vA = torch.tensor([[[250.0, 0.0, 0.0]]], device=device, dtype=torch.float32)
    pA1 = pA0 + vA * dt  # +2.5 m

    pB0 = torch.tensor([[[10.0, 0.0, 0.0]]], device=device, dtype=torch.float32)
    vB = torch.tensor([[[-250.0, 0.0, 0.0]]], device=device, dtype=torch.float32)
    pB1 = pB0 + vB * dt  # -2.5 m

    start_dist = torch.linalg.norm(pB0 - pA0).item()
    end_dist = torch.linalg.norm(pB1 - pA1).item()

    assert start_dist == pytest.approx(20.0, abs=1e-3)
    assert end_dist == pytest.approx(5.0, abs=1e-3)
    assert start_dist > R_INT, "Start distance must be > R_INT for tunneling test"
    assert end_dist > R_INT, "End distance must be > R_INT for tunneling test"

    d_min, s_star = pairwise_cpa(pA0, pA1, pB0, pB1)
    assert d_min.item() < 1e-3, f"Expected d_min ~= 0, got {d_min.item()}"
    # Closing speed = 500 m/s, initial dist = 20 m -> impact at t = 20/500 = 0.04s -> s* = 0.04/0.05 = 0.8
    assert s_star.item() == pytest.approx(0.8, abs=1e-3)

    active_A = torch.tensor([[True]], device=device, dtype=torch.bool)
    active_B = torch.tensor([[True]], device=device, dtype=torch.bool)
    hit = intercept_mask(d_min, R_INT, active_A, active_B)
    assert hit.item() is True

    # Case 2: Symmetric start positions (-6.25, +6.25) where crossing is exactly at s_star = 0.5
    pA0_sym = torch.tensor([[[-6.25, 0.0, 0.0]]], device=device, dtype=torch.float32)
    pA1_sym = pA0_sym + vA * dt  # +6.25
    pB0_sym = torch.tensor([[[6.25, 0.0, 0.0]]], device=device, dtype=torch.float32)
    pB1_sym = pB0_sym + vB * dt  # -6.25

    assert torch.linalg.norm(pB0_sym - pA0_sym).item() > R_INT
    assert torch.linalg.norm(pB1_sym - pA1_sym).item() > R_INT

    d_min_sym, s_star_sym = pairwise_cpa(pA0_sym, pA1_sym, pB0_sym, pB1_sym)
    assert d_min_sym.item() < 1e-3
    assert s_star_sym.item() == pytest.approx(0.5, abs=1e-3)
    assert intercept_mask(d_min_sym, R_INT, active_A, active_B).item() is True


@pytest.mark.parametrize("device", DEVICES)
def test_near_miss(device: str):
    """Near-miss: missile offset 50 m in y -> d_min ~= 50, intercept False."""
    R_INT = 2.0
    dt = 0.05
    pA0 = torch.tensor([[[-10.0, 0.0, 0.0]]], device=device, dtype=torch.float32)
    vA = torch.tensor([[[250.0, 0.0, 0.0]]], device=device, dtype=torch.float32)
    pA1 = pA0 + vA * dt

    pB0 = torch.tensor([[[10.0, 50.0, 0.0]]], device=device, dtype=torch.float32)
    vB = torch.tensor([[[-250.0, 0.0, 0.0]]], device=device, dtype=torch.float32)
    pB1 = pB0 + vB * dt

    d_min, s_star = pairwise_cpa(pA0, pA1, pB0, pB1)
    assert d_min.item() == pytest.approx(50.0, abs=1e-3)
    assert s_star.item() == pytest.approx(0.8, abs=1e-3)

    active_A = torch.tensor([[True]], device=device, dtype=torch.bool)
    active_B = torch.tensor([[True]], device=device, dtype=torch.bool)
    assert intercept_mask(d_min, R_INT, active_A, active_B).item() is False


@pytest.mark.parametrize("device", DEVICES)
def test_zero_relative_velocity(device: str):
    """Zero relative velocity (identical velocities, 100 m apart) -> d_min == 100 (tol 1e-3), no NaN."""
    pA0 = torch.tensor([[[0.0, 0.0, 0.0]]], device=device, dtype=torch.float32)
    pA1 = torch.tensor([[[10.0, 20.0, 30.0]]], device=device, dtype=torch.float32)
    pB0 = torch.tensor([[[100.0, 0.0, 0.0]]], device=device, dtype=torch.float32)
    pB1 = torch.tensor([[[110.0, 20.0, 30.0]]], device=device, dtype=torch.float32)

    d_min, s_star = pairwise_cpa(pA0, pA1, pB0, pB1)

    assert not torch.isnan(d_min).any()
    assert not torch.isinf(d_min).any()
    assert not torch.isnan(s_star).any()
    assert not torch.isinf(s_star).any()
    assert d_min.item() == pytest.approx(100.0, abs=1e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_batched_shapes_and_convexity(device: str):
    """Batched shapes: E=3, A=5, B=7 -> output shapes [3,5,7], all finite,
    d_min <= min(start distance, end distance) + 1e-4 everywhere."""
    torch.manual_seed(123)
    E, A, B = 3, 5, 7

    pA0 = torch.randn(E, A, 3, device=device, dtype=torch.float32) * 100.0
    pA1 = pA0 + torch.randn(E, A, 3, device=device, dtype=torch.float32) * 50.0
    pB0 = torch.randn(E, B, 3, device=device, dtype=torch.float32) * 100.0
    pB1 = pB0 + torch.randn(E, B, 3, device=device, dtype=torch.float32) * 50.0

    d_min, s_star = pairwise_cpa(pA0, pA1, pB0, pB1)

    assert d_min.shape == (E, A, B)
    assert s_star.shape == (E, A, B)
    assert torch.isfinite(d_min).all()
    assert torch.isfinite(s_star).all()
    assert (s_star >= 0.0).all() and (s_star <= 1.0).all()

    start_dist = torch.linalg.norm(pB0.unsqueeze(1) - pA0.unsqueeze(2), dim=-1)
    end_dist = torch.linalg.norm(pB1.unsqueeze(1) - pA1.unsqueeze(2), dim=-1)
    min_endpoint_dist = torch.minimum(start_dist, end_dist)

    assert (d_min <= min_endpoint_dist + 1e-4).all()


@pytest.mark.parametrize("device", DEVICES)
def test_friendly_collision_mask(device: str):
    """Two drones crossing paths within one step are flagged; diagonal is False;
    an inactive drone is never flagged."""
    r_col = 1.0
    # Drone 0 moves [-5, 0, 0] -> [+5, 0, 0]
    # Drone 1 moves [0, -5, 0] -> [0, +5, 0]
    # They cross at the origin at s = 0.5, separation = 0 <= 2 * r_col
    pA0 = torch.tensor([[[-5.0, 0.0, 0.0], [0.0, -5.0, 0.0]]], device=device, dtype=torch.float32)
    pA1 = torch.tensor([[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]], device=device, dtype=torch.float32)

    # 1. Both active: off-diagonals True, diagonal False
    active_both = torch.tensor([[True, True]], device=device, dtype=torch.bool)
    mask = friendly_collision_mask(pA0, pA1, r_col, active_both)

    assert mask.shape == (1, 2, 2)
    assert mask[0, 0, 1].item() is True
    assert mask[0, 1, 0].item() is True
    assert mask[0, 0, 0].item() is False
    assert mask[0, 1, 1].item() is False

    # 2. Inactive drone: never flagged
    active_one = torch.tensor([[True, False]], device=device, dtype=torch.bool)
    mask_inactive = friendly_collision_mask(pA0, pA1, r_col, active_one)
    assert not mask_inactive.any()
