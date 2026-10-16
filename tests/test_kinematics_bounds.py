"""Unit tests for 3D kinematics bounds, drag, and trapezoidal integration."""

import pytest
import torch

from src.physics.kinematics_3d import G0, step_kinematics

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize("device", DEVICES)
def test_random_accel_clamping_200_steps(device: str):
    """Command accelerations of magnitude 1e6 in random directions for 200 steps on [4, 32, 3].
    Assert at every step ||a_applied|| <= a_max + 1e-3 and ||v|| <= v_max + 1e-3."""
    torch.manual_seed(42)
    E, N = 4, 32
    a_max = 12.0 * G0
    v_max = 350.0
    drag = 0.015
    dt = 0.05

    p = torch.zeros(E, N, 3, device=device, dtype=torch.float32)
    v = torch.zeros(E, N, 3, device=device, dtype=torch.float32)

    for _ in range(200):
        # Generate random unit directions scaled to 1e6 magnitude
        rnd = torch.randn(E, N, 3, device=device, dtype=torch.float32)
        rnd_norm = torch.linalg.norm(rnd, dim=-1, keepdim=True)
        a_cmd = (rnd / (rnd_norm + 1e-8)) * 1e6

        p, v, a_applied = step_kinematics(p, v, a_cmd, dt, a_max, v_max, drag)

        norm_a = torch.linalg.norm(a_applied, dim=-1)
        norm_v = torch.linalg.norm(v, dim=-1)

        assert (norm_a <= a_max + 1e-3).all(), f"Accel exceeded a_max: {norm_a.max().item()} > {a_max}"
        assert (norm_v <= v_max + 1e-3).all(), f"Speed exceeded v_max: {norm_v.max().item()} > {v_max}"


@pytest.mark.parametrize("device", DEVICES)
def test_diagonal_cheat_check(device: str):
    """Command a = (a_max, a_max, a_max); assert ||a_applied|| ~= a_max, NOT a_max*sqrt(3)."""
    a_max = 12.0 * G0
    a_cmd = torch.full((1, 1, 3), a_max, device=device, dtype=torch.float32)
    p = torch.zeros(1, 1, 3, device=device, dtype=torch.float32)
    v = torch.zeros(1, 1, 3, device=device, dtype=torch.float32)

    _, _, a_applied = step_kinematics(p, v, a_cmd, dt=0.05, a_max=a_max, v_max=350.0, drag=0.015)

    norm_applied = torch.linalg.norm(a_applied).item()
    assert norm_applied == pytest.approx(a_max, rel=1e-4)
    per_axis_clipped_norm = a_max * (3.0 ** 0.5)
    assert abs(norm_applied - per_axis_clipped_norm) > 1.0


@pytest.mark.parametrize("device", DEVICES)
def test_drag_deceleration(device: str):
    """With a_cmd = 0 and v0 = 100 m/s along x, after one step v = (1 - drag*dt)*100 within 1e-4."""
    drag = 0.015
    dt = 0.05
    p0 = torch.zeros(1, 1, 3, device=device, dtype=torch.float32)
    v0 = torch.tensor([[[100.0, 0.0, 0.0]]], device=device, dtype=torch.float32)
    a_cmd = torch.zeros(1, 1, 3, device=device, dtype=torch.float32)

    _, v_new, _ = step_kinematics(p0, v0, a_cmd, dt=dt, a_max=12.0 * G0, v_max=350.0, drag=drag)

    expected_vx = (1.0 - drag * dt) * 100.0
    assert v_new[0, 0, 0].item() == pytest.approx(expected_vx, rel=1e-4)
    assert v_new[0, 0, 1].item() == pytest.approx(0.0, abs=1e-6)
    assert v_new[0, 0, 2].item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_trapezoidal_integration_accuracy(device: str):
    """Constant accel 10 m/s^2 from rest, dt=0.05: verify against hand calculation."""
    dt = 0.05
    a_val = 10.0
    p0 = torch.zeros(1, 1, 3, device=device, dtype=torch.float32)
    v0 = torch.zeros(1, 1, 3, device=device, dtype=torch.float32)
    a_cmd = torch.tensor([[[a_val, 0.0, 0.0]]], device=device, dtype=torch.float32)

    # Hand calculation:
    # a_applied = 10.0
    # v_tilde = 0 + 10.0 * 0.05 = 0.5 m/s
    # v_new = 0.5 m/s (no clamping active)
    # p_new = 0 + 0.5 * (0 + 0.5) * 0.05 = 0.0125 m
    expected_v = a_val * dt
    expected_p = 0.5 * (0.0 + expected_v) * dt

    p_new, v_new, a_applied = step_kinematics(p0, v0, a_cmd, dt=dt, a_max=100.0, v_max=100.0, drag=0.0)

    assert a_applied[0, 0, 0].item() == pytest.approx(a_val, rel=1e-4)
    assert v_new[0, 0, 0].item() == pytest.approx(expected_v, rel=1e-4)
    assert p_new[0, 0, 0].item() == pytest.approx(expected_p, rel=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_inactive_entities_frozen(device: str):
    """Inactive entities do not move and have v=0 and a_applied=0."""
    active = torch.tensor([[True, False]], device=device, dtype=torch.bool)
    p = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]], device=device, dtype=torch.float32)
    v = torch.tensor([[[5.0, 0.0, 0.0], [15.0, 0.0, 0.0]]], device=device, dtype=torch.float32)
    a_cmd = torch.tensor([[[10.0, 0.0, 0.0], [50.0, 0.0, 0.0]]], device=device, dtype=torch.float32)

    p_new, v_new, a_applied = step_kinematics(
        p, v, a_cmd, dt=0.05, a_max=100.0, v_max=100.0, drag=0.01, active=active
    )

    # Active entity 0 should have updated
    assert not torch.equal(p_new[:, 0, :], p[:, 0, :])
    assert not torch.equal(v_new[:, 0, :], torch.zeros(1, 3, device=device))

    # Inactive entity 1 must remain frozen with 0 velocity and 0 applied accel
    assert torch.equal(p_new[:, 1, :], p[:, 1, :])
    assert torch.equal(v_new[:, 1, :], torch.zeros(1, 3, device=device))
    assert torch.equal(a_applied[:, 1, :], torch.zeros(1, 3, device=device))
