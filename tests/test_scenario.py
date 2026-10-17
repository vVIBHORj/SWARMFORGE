"""Unit tests for scenario generation."""

import copy
import math
from pathlib import Path
import pytest
import torch

from src.config import load_config
from src.envs.scenario_generator import generate_scenario

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default_sim.yaml"

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.fixture
def base_cfg():
    return load_config(CONFIG_PATH)


@pytest.mark.parametrize("device", DEVICES)
def test_scenario_shapes_and_dtypes(base_cfg, device):
    """Verify shapes and dtypes of all scenario tensors for E=4, N=20 (16+4), M=30, P=2."""
    E = 4
    num_interceptors = 16
    num_scouts = 4
    N = num_interceptors + num_scouts
    M = 30
    P = 2

    scen = generate_scenario(
        base_cfg,
        num_envs=E,
        num_interceptors=num_interceptors,
        num_scouts=num_scouts,
        num_missiles=M,
        device=device,
        seed=101,
    )

    # Check dtypes and shapes
    assert scen["hva_pos"].shape == (E, P, 3)
    assert scen["hva_pos"].dtype == torch.float32

    assert scen["drone_pos"].shape == (E, N, 3)
    assert scen["drone_pos"].dtype == torch.float32

    assert scen["drone_vel"].shape == (E, N, 3)
    assert scen["drone_vel"].dtype == torch.float32

    assert scen["drone_role"].shape == (E, N)
    assert scen["drone_role"].dtype == torch.int64

    assert scen["drone_active"].shape == (E, N)
    assert scen["drone_active"].dtype == torch.bool

    assert scen["missile_pos"].shape == (E, M, 3)
    assert scen["missile_pos"].dtype == torch.float32

    assert scen["missile_vel"].shape == (E, M, 3)
    assert scen["missile_vel"].dtype == torch.float32

    assert scen["missile_type"].shape == (E, M)
    assert scen["missile_type"].dtype == torch.int64

    assert scen["missile_target_hva"].shape == (E, M)
    assert scen["missile_target_hva"].dtype == torch.int64

    assert scen["missile_is_decoy"].shape == (E, M)
    assert scen["missile_is_decoy"].dtype == torch.bool

    assert scen["missile_K"].shape == (E, M)
    assert scen["missile_K"].dtype == torch.int64

    assert scen["missile_weave_phase"].shape == (E, M)
    assert scen["missile_weave_phase"].dtype == torch.float32

    assert scen["missile_spawn_time"].shape == (E, M)
    assert scen["missile_spawn_time"].dtype == torch.float32

    assert scen["missile_active"].shape == (E, M)
    assert scen["missile_active"].dtype == torch.bool

    assert isinstance(scen["meta"], dict)
    assert scen["meta"]["num_envs"] == E


@pytest.mark.parametrize("device", DEVICES)
def test_scenario_physics_and_bounds(base_cfg, device):
    """Verify entity positions inside bounds, drone roles, HVA z=0, missile speeds and directions."""
    E, num_interceptors, num_scouts, M = 4, 16, 4, 30
    N = num_interceptors + num_scouts
    W, L, H = [float(v) for v in base_cfg["environment"]["world_bounds_meters"]]
    max_speed = float(base_cfg["entities"]["missiles"]["max_speed"])

    scen = generate_scenario(
        base_cfg,
        num_envs=E,
        num_interceptors=num_interceptors,
        num_scouts=num_scouts,
        num_missiles=M,
        device=device,
        seed=202,
    )

    # 1. HVA bounds and ground z=0
    hva_pos = scen["hva_pos"]
    assert (hva_pos[..., 0] >= -W / 2.0).all() and (hva_pos[..., 0] <= W / 2.0).all()
    assert (hva_pos[..., 1] >= -L / 2.0).all() and (hva_pos[..., 1] <= L / 2.0).all()
    assert (hva_pos[..., 2] == 0.0).all()

    # 2. Drone bounds and roles
    drone_pos = scen["drone_pos"]
    assert (drone_pos[..., 0] >= -W / 2.0).all() and (drone_pos[..., 0] <= W / 2.0).all()
    assert (drone_pos[..., 1] >= -L / 2.0).all() and (drone_pos[..., 1] <= L / 2.0).all()
    assert (drone_pos[..., 2] >= 0.0).all() and (drone_pos[..., 2] <= H).all()

    expected_roles = torch.tensor([0] * num_interceptors + [1] * num_scouts, device=device, dtype=torch.int64)
    for e in range(E):
        assert torch.equal(scen["drone_role"][e], expected_roles)

    # 3. Missile bounds, speeds, and direction to target HVA
    missile_pos = scen["missile_pos"]
    assert (missile_pos[..., 0] >= -W / 2.0 - 1e-3).all() and (missile_pos[..., 0] <= W / 2.0 + 1e-3).all()
    assert (missile_pos[..., 1] >= -L / 2.0 - 1e-3).all() and (missile_pos[..., 1] <= L / 2.0 + 1e-3).all()
    assert (missile_pos[..., 2] >= 0.0).all() and (missile_pos[..., 2] <= H).all()

    speeds = torch.linalg.norm(scen["missile_vel"], dim=-1)
    min_speed = 0.7 * max_speed
    max_allowed_speed = 0.95 * max_speed
    assert (speeds >= min_speed - 1e-3).all()
    assert (speeds <= max_allowed_speed + 1e-3).all()

    # Target alignment check
    target_idx = scen["missile_target_hva"].unsqueeze(-1).expand(E, M, 3)
    target_pos = torch.gather(hva_pos, dim=1, index=target_idx)
    diff = target_pos - missile_pos
    cos_sim = (scen["missile_vel"] * diff).sum(dim=-1) / (
        torch.linalg.norm(scen["missile_vel"], dim=-1) * torch.linalg.norm(diff, dim=-1) + 1e-8
    )
    assert (cos_sim > 0.999).all()


@pytest.mark.parametrize("device", DEVICES)
def test_patterns(base_cfg, device):
    """Test saturation (30-degree arc, spawn_time 0) and wave patterns."""
    # Saturation pattern check
    cfg_sat = copy.deepcopy(base_cfg)
    cfg_sat["scenario"]["patterns"] = ["saturation"]

    scen_sat = generate_scenario(
        cfg_sat,
        num_envs=4,
        num_interceptors=16,
        num_scouts=4,
        num_missiles=30,
        device=device,
        seed=303,
    )
    assert (scen_sat["missile_spawn_time"] == 0.0).all()

    # Verify all spawn azimuths lie within a 30-degree arc
    mpos = scen_sat["missile_pos"]
    angles = torch.atan2(mpos[..., 1], mpos[..., 0])
    for e in range(4):
        ang_e = angles[e]
        ref_ang = ang_e[0]
        wrapped_diff = torch.atan2(torch.sin(ang_e - ref_ang), torch.cos(ang_e - ref_ang))
        arc_span = wrapped_diff.max() - wrapped_diff.min()
        assert arc_span.item() <= math.radians(30.0) + 1e-4

    # Wave pattern check
    cfg_wave = copy.deepcopy(base_cfg)
    cfg_wave["scenario"]["patterns"] = ["wave"]
    num_waves = cfg_wave["scenario"]["num_waves"]

    scen_wave = generate_scenario(
        cfg_wave,
        num_envs=4,
        num_interceptors=16,
        num_scouts=4,
        num_missiles=30,
        device=device,
        seed=404,
    )
    for e in range(4):
        unique_times = torch.unique(scen_wave["missile_spawn_time"][e])
        assert len(unique_times) == num_waves


@pytest.mark.parametrize("device", DEVICES)
def test_mode_2d(base_cfg, device):
    """Verify 2d mode zeroes all z positions for drones, missiles, and HVAs."""
    scen = generate_scenario(
        base_cfg,
        num_envs=4,
        num_interceptors=16,
        num_scouts=4,
        num_missiles=30,
        device=device,
        seed=505,
        mode="2d",
    )
    assert (scen["hva_pos"][..., 2] == 0.0).all()
    assert (scen["drone_pos"][..., 2] == 0.0).all()
    assert (scen["missile_pos"][..., 2] == 0.0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_weave_disabled(base_cfg, device):
    """Verify weave_enabled=False produces no type 2 missiles."""
    scen = generate_scenario(
        base_cfg,
        num_envs=4,
        num_interceptors=16,
        num_scouts=4,
        num_missiles=30,
        device=device,
        seed=606,
        weave_enabled=False,
    )
    assert (scen["missile_type"] != 2).all()


@pytest.mark.parametrize("device", DEVICES)
def test_decoy_fraction(base_cfg, device):
    """Verify decoy fraction over 64 envs x 30 missiles is within +-0.05 of 0.15."""
    scen = generate_scenario(
        base_cfg,
        num_envs=64,
        num_interceptors=16,
        num_scouts=4,
        num_missiles=30,
        device=device,
        seed=707,
    )
    mean_decoys = scen["missile_is_decoy"].float().mean().item()
    assert abs(mean_decoys - 0.15) <= 0.05
