"""Unit and closed-loop integration tests for guidance laws, CBF shield, and baseline controller."""

import copy
from pathlib import Path
import pytest
import torch

from src.baselines.baseline_controller import BaselineController
from src.config import load_config
from src.envs.vectorized_swarm_env import VectorizedSwarmEnv
from src.physics.guidance_laws import (
    cbf_repulsion_accel,
    engagement_kinematics,
    tpn_guidance_accel,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default_sim.yaml"

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.fixture
def base_cfg():
    return load_config(CONFIG_PATH)


@pytest.mark.parametrize("device", DEVICES)
def test_engagement_kinematics_closing_and_zem(device):
    """Test Vc positive on closing, negative on receding, and ZEM zero on collision course."""
    # 1. Head-on closing: drone at 0 moving +100, missile at 1000 moving -100
    drone_pos = torch.zeros(1, 1, 3, device=device)
    drone_vel = torch.tensor([[[100.0, 0.0, 0.0]]], device=device)
    missile_pos = torch.tensor([[[1000.0, 0.0, 0.0]]], device=device)
    missile_vel = torch.tensor([[[-100.0, 0.0, 0.0]]], device=device)

    kin_closing = engagement_kinematics(drone_pos, drone_vel, missile_pos, missile_vel)
    assert kin_closing["Vc"][0, 0, 0].item() == pytest.approx(200.0, abs=1e-3)
    assert kin_closing["zem_norm"][0, 0, 0].item() == pytest.approx(0.0, abs=1e-3)

    # 2. Receding: drone moving -200 (away from missile at -100)
    drone_vel_receding = torch.tensor([[[-200.0, 0.0, 0.0]]], device=device)
    kin_receding = engagement_kinematics(drone_pos, drone_vel_receding, missile_pos, missile_vel)
    assert kin_receding["Vc"][0, 0, 0].item() < 0.0

    # 3. Laterally offset missile: ZEM must be nonzero (~ offset)
    missile_pos_offset = torch.tensor([[[1000.0, 50.0, 0.0]]], device=device)
    kin_offset = engagement_kinematics(drone_pos, drone_vel, missile_pos_offset, missile_vel)
    assert kin_offset["zem_norm"][0, 0, 0].item() == pytest.approx(50.0, abs=1.0)


@pytest.mark.parametrize("device", DEVICES)
def test_tpn_guidance_accel_direction(device):
    """Test TPN produces lateral acceleration toward intercept, and unassigned drone gets zero."""
    drone_pos = torch.zeros(1, 2, 3, device=device)
    drone_vel = torch.tensor([[[200.0, 0.0, 0.0], [200.0, 0.0, 0.0]]], device=device)

    # Missile at x=1000, offset y=100, moving -200 along x
    missile_pos = torch.tensor([[[1000.0, 100.0, 0.0]]], device=device)
    missile_vel = torch.tensor([[[-200.0, 0.0, 0.0]]], device=device)

    # Drone 0 assigned to missile 0; Drone 1 unassigned
    assignment_z = torch.zeros(1, 2, 1, dtype=torch.bool, device=device)
    assignment_z[0, 0, 0] = True

    accel = tpn_guidance_accel(drone_pos, drone_vel, missile_pos, missile_vel, assignment_z, nav_gain=4.0)

    # Drone 0 should accelerate laterally in +y direction to intercept
    assert accel[0, 0, 1].item() > 0.0
    # Drone 1 unassigned receives exact 0 acceleration
    assert torch.equal(accel[0, 1], torch.zeros(3, device=device))


@pytest.mark.parametrize("device", DEVICES)
def test_cbf_repulsion(device):
    """Test CBF produces equal and opposite repulsion within r_safe, and 0 beyond r_safe."""
    drone_active = torch.tensor([[True, True]], dtype=torch.bool, device=device)

    # 1. Distance = 12m < r_safe (40m)
    p_close = torch.tensor([[[0.0, 0.0, 0.0], [12.0, 0.0, 0.0]]], device=device)
    a_cbf = cbf_repulsion_accel(p_close, drone_active, r_col=5.0, r_safe=40.0, k_cbf=500.0)

    # Drone 0 should be repelled to -x, Drone 1 to +x
    assert a_cbf[0, 0, 0].item() < 0.0
    assert a_cbf[0, 1, 0].item() > 0.0
    assert a_cbf[0, 0, 0].item() == pytest.approx(-a_cbf[0, 1, 0].item(), rel=1e-4)

    # 2. Distance = 60m > r_safe (40m) -> zero repulsion
    p_far = torch.tensor([[[0.0, 0.0, 0.0], [60.0, 0.0, 0.0]]], device=device)
    a_cbf_far = cbf_repulsion_accel(p_far, drone_active, r_col=5.0, r_safe=40.0, k_cbf=500.0)
    assert torch.equal(a_cbf_far, torch.zeros_like(a_cbf_far))


@pytest.mark.parametrize("device", DEVICES)
def test_stage1_closed_loop_benchmark(base_cfg, device):
    """Closed-loop Stage 1 benchmark (Milestone M2 criterion: >= 80% intercept rate).

    Parameters: E=8, N=6 interceptors, N_scouts=0, M=8 missiles, mode='2d',
    weave_enabled=False, decoy_fraction=0.0, seed=42, Hungarian allocator.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg["environment"]["num_parallel_envs"] = 8
    cfg["entities"]["drones"]["num_interceptors"] = 6
    cfg["entities"]["drones"]["num_scouts"] = 0
    cfg["entities"]["missiles"]["num_missiles"] = 8
    cfg["entities"]["missiles"]["decoy_fraction"] = 0.0
    cfg["scenario"]["mode"] = "2d"
    cfg["scenario"]["weave_enabled"] = False

    env = VectorizedSwarmEnv(cfg, num_envs=8, device=device, seed=42)
    controller = BaselineController(cfg, allocator="hungarian", use_cbf=True)

    max_steps = 400
    for step in range(max_steps):
        if env.done.all():
            break

        accel_cmd, z = controller.act(env)

        # Invariant checks at every step
        assert (z.sum(dim=2) <= 1).all(), f"Step {step}: drone assigned to multiple targets!"
        assert (z.sum(dim=1) <= env.missile_K).all(), f"Step {step}: missile exceeded capacity!"

        env.step(accel_cmd)

    metrics = env.metrics()
    mean_intercept_rate = metrics["intercept_rate"].mean().item()
    print(f"\nStage 1 Benchmark ({device}): Mean Intercept Rate = {mean_intercept_rate * 100:.1f}%")

    assert mean_intercept_rate >= 0.80, (
        f"Stage 1 mean intercept rate {mean_intercept_rate * 100:.1f}% fell below M2 criterion of 80%!"
    )
