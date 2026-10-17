"""Unit tests for missile threat guidance models (ballistic, dive, weave)."""

import math
from pathlib import Path
import pytest
import torch

from src.config import load_config
from src.envs.threat_models import missile_accel_cmd
from src.physics.kinematics_3d import G0, step_kinematics

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default_sim.yaml"

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.fixture
def base_cfg():
    return load_config(CONFIG_PATH)


@pytest.mark.parametrize("device", DEVICES)
def test_dive_missile_intercepts_hva(base_cfg, device):
    """A dive missile (type 1) starting 5000 m from its HVA at ~400 m/s, offset 30 deg,
    reaches within 50 m of the HVA within 30 seconds of sim time."""
    dt = 0.05
    max_speed = 450.0
    a_max = 8.0 * G0
    drag = 0.0

    # HVA at origin
    hva_pos = torch.zeros((1, 1, 3), device=device, dtype=torch.float32)
    target_hva = torch.zeros((1, 1), device=device, dtype=torch.int64)
    mtype = torch.ones((1, 1), device=device, dtype=torch.int64)  # type 1 = dive
    weave_phase = torch.zeros((1, 1), device=device, dtype=torch.float32)

    # Start 5000 m away along x-axis
    pos = torch.tensor([[[5000.0, 0.0, 0.0]]], device=device, dtype=torch.float32)
    # Velocity offset 30 degrees in x-y plane
    speed = 400.0
    offset_rad = math.radians(30.0)
    vx = -speed * math.cos(offset_rad)
    vy = speed * math.sin(offset_rad)
    vel = torch.tensor([[[vx, vy, 0.0]]], device=device, dtype=torch.float32)

    min_dist = float("inf")
    t_sec = 0.0
    max_steps = int(30.0 / dt)

    for _ in range(max_steps):
        dist = torch.linalg.norm(pos - hva_pos).item()
        if dist < min_dist:
            min_dist = dist
        if dist <= 50.0:
            break

        a_cmd = missile_accel_cmd(pos, vel, hva_pos, target_hva, mtype, weave_phase, t_sec, base_cfg)
        pos, vel, _ = step_kinematics(pos, vel, a_cmd, dt, a_max, max_speed, drag)
        t_sec += dt

    assert min_dist <= 50.0, f"Dive missile failed to reach within 50 m (min dist was {min_dist:.2f} m)"


@pytest.mark.parametrize("device", DEVICES)
def test_ballistic_missile_constant_velocity(base_cfg, device):
    """A ballistic missile's velocity is exactly constant (drag 0)."""
    dt = 0.05
    max_speed = 450.0
    a_max = 8.0 * G0
    drag = 0.0

    hva_pos = torch.zeros((1, 1, 3), device=device, dtype=torch.float32)
    target_hva = torch.zeros((1, 1), device=device, dtype=torch.int64)
    mtype = torch.zeros((1, 1), device=device, dtype=torch.int64)  # type 0 = ballistic
    weave_phase = torch.zeros((1, 1), device=device, dtype=torch.float32)

    pos = torch.tensor([[[3000.0, 2000.0, 1500.0]]], device=device, dtype=torch.float32)
    vel_init = torch.tensor([[[-300.0, -150.0, -50.0]]], device=device, dtype=torch.float32)
    vel = vel_init.clone()

    t_sec = 0.0
    for _ in range(100):
        a_cmd = missile_accel_cmd(pos, vel, hva_pos, target_hva, mtype, weave_phase, t_sec, base_cfg)
        assert torch.equal(a_cmd, torch.zeros_like(a_cmd))
        pos, vel, a_applied = step_kinematics(pos, vel, a_cmd, dt, a_max, max_speed, drag)
        assert torch.allclose(vel, vel_init, atol=1e-5)
        t_sec += dt


@pytest.mark.parametrize("device", DEVICES)
def test_weave_missile_acceleration_and_lateral_deviation(base_cfg, device):
    """A weave missile's applied acceleration norm never exceeds a_max + 1e-3,
    and its lateral deviation is larger than a dive missile's for the same start."""
    dt = 0.05
    max_speed = 450.0
    a_max = 8.0 * G0
    drag = 0.0

    hva_pos = torch.zeros((1, 1, 3), device=device, dtype=torch.float32)
    target_hva = torch.zeros((1, 1), device=device, dtype=torch.int64)
    weave_phase = torch.zeros((1, 1), device=device, dtype=torch.float32)

    p0 = torch.tensor([[[5000.0, 0.0, 0.0]]], device=device, dtype=torch.float32)
    v0 = torch.tensor([[[-400.0, 0.0, 0.0]]], device=device, dtype=torch.float32)
    u_hat = torch.tensor([-1.0, 0.0, 0.0], device=device, dtype=torch.float32)

    # 1. Simulate dive missile (type 1)
    pos_dive = p0.clone()
    vel_dive = v0.clone()
    mtype_dive = torch.tensor([[1]], device=device, dtype=torch.int64)
    max_perp_dive = 0.0
    t_sec = 0.0

    for _ in range(200):
        a_cmd = missile_accel_cmd(pos_dive, vel_dive, hva_pos, target_hva, mtype_dive, weave_phase, t_sec, base_cfg)
        pos_dive, vel_dive, _ = step_kinematics(pos_dive, vel_dive, a_cmd, dt, a_max, max_speed, drag)
        disp = pos_dive[0, 0] - p0[0, 0]
        d_perp = disp - torch.dot(disp, u_hat) * u_hat
        perp_dist = torch.linalg.norm(d_perp).item()
        if perp_dist > max_perp_dive:
            max_perp_dive = perp_dist
        t_sec += dt

    # 2. Simulate weave missile (type 2)
    pos_weave = p0.clone()
    vel_weave = v0.clone()
    mtype_weave = torch.tensor([[2]], device=device, dtype=torch.int64)
    max_perp_weave = 0.0
    t_sec = 0.0

    for _ in range(200):
        a_cmd = missile_accel_cmd(pos_weave, vel_weave, hva_pos, target_hva, mtype_weave, weave_phase, t_sec, base_cfg)
        pos_weave, vel_weave, a_applied = step_kinematics(pos_weave, vel_weave, a_cmd, dt, a_max, max_speed, drag)
        # Check acceleration bound
        a_norm = torch.linalg.norm(a_applied, dim=-1).item()
        assert a_norm <= a_max + 1e-3, f"Weave accel exceeded a_max: {a_norm} > {a_max}"

        disp = pos_weave[0, 0] - p0[0, 0]
        d_perp = disp - torch.dot(disp, u_hat) * u_hat
        perp_dist = torch.linalg.norm(d_perp).item()
        if perp_dist > max_perp_weave:
            max_perp_weave = perp_dist
        t_sec += dt

    assert max_perp_weave > max_perp_dive + 5.0, (
        f"Expected weave lateral deviation ({max_perp_weave:.2f} m) > dive deviation ({max_perp_dive:.2f} m)"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_mode_2d_zeroes_z_trajectory(base_cfg, device):
    """2d mode keeps z == 0 for the whole trajectory."""
    dt = 0.05
    max_speed = 450.0
    a_max = 8.0 * G0
    drag = 0.0

    hva_pos = torch.zeros((1, 1, 3), device=device, dtype=torch.float32)
    target_hva = torch.zeros((1, 1), device=device, dtype=torch.int64)
    mtype = torch.tensor([[2]], device=device, dtype=torch.int64)  # type 2 weave
    weave_phase = torch.zeros((1, 1), device=device, dtype=torch.float32)

    pos = torch.tensor([[[4000.0, 3000.0, 0.0]]], device=device, dtype=torch.float32)
    vel = torch.tensor([[[-300.0, -200.0, 0.0]]], device=device, dtype=torch.float32)

    t_sec = 0.0
    for _ in range(100):
        a_cmd = missile_accel_cmd(pos, vel, hva_pos, target_hva, mtype, weave_phase, t_sec, base_cfg, mode="2d")
        assert a_cmd[0, 0, 2].item() == 0.0
        pos, vel, _ = step_kinematics(pos, vel, a_cmd, dt, a_max, max_speed, drag)
        assert pos[0, 0, 2].item() == 0.0
        assert vel[0, 0, 2].item() == 0.0
        t_sec += dt
