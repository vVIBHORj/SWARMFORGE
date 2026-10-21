"""Unit and integration tests for observations, rewards, curriculum manager, and PettingZoo wrapper."""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.config import load_config
from src.envs.curriculum_manager import CurriculumManager
from src.envs.observations import build_observations
from src.envs.pettingzoo_wrapper import SwarmParallelEnv
from src.envs.rewards import RewardComputer
from src.envs.vectorized_swarm_env import VectorizedSwarmEnv

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default_sim.yaml"

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.fixture
def cfg():
    return load_config(CONFIG_PATH)


@pytest.mark.parametrize("device", DEVICES)
def test_observation_shapes_and_padding(cfg, device: str):
    """Test observation shapes and zero padding across curriculum stage variations."""
    # 1. Stage 1: N=6 (interceptor=6, scout=0), M=8 < K_threat=15, K_ally=10
    env1 = VectorizedSwarmEnv(cfg, num_envs=2, device=device)
    env1.reset(num_interceptors=6, num_scouts=0, num_missiles=8, mode="2d")

    obs1 = build_observations(env1, K_threat=15, K_ally=10)
    assert obs1["ego"].shape == (2, 6, 10)
    assert obs1["threat_tokens"].shape == (2, 6, 15, 15)
    assert obs1["threat_mask"].shape == (2, 6, 15)
    assert obs1["threat_indices"].shape == (2, 6, 15)
    assert obs1["ally_tokens"].shape == (2, 6, 10, 9)
    assert obs1["ally_mask"].shape == (2, 6, 10)
    assert obs1["ally_indices"].shape == (2, 6, 10)
    assert obs1["drone_mask"].shape == (2, 6)

    # Check finite and strictly 0.0 where masked
    assert torch.isfinite(obs1["ego"]).all()
    assert torch.isfinite(obs1["threat_tokens"]).all()
    assert torch.isfinite(obs1["ally_tokens"]).all()
    assert (obs1["threat_tokens"][~obs1["threat_mask"]] == 0.0).all()
    assert (obs1["ally_tokens"][~obs1["ally_mask"]] == 0.0).all()

    # Padded slots beyond M=8 and beyond active allies (6-1=5) must be False
    assert not obs1["threat_mask"][:, :, 8:].any()
    assert not obs1["ally_mask"][:, :, 5:].any()

    # 2. Stage 3: N=20 (interceptor=16, scout=4), M=30 > K_threat=15
    env3 = VectorizedSwarmEnv(cfg, num_envs=2, device=device)
    env3.reset(num_interceptors=16, num_scouts=4, num_missiles=30, mode="3d")

    obs3 = build_observations(env3, K_threat=15, K_ally=10)
    assert obs3["ego"].shape == (2, 20, 10)
    assert obs3["threat_tokens"].shape == (2, 20, 15, 15)
    assert obs3["threat_mask"].shape == (2, 20, 15)
    assert obs3["ally_tokens"].shape == (2, 20, 10, 9)
    assert obs3["ally_mask"].shape == (2, 20, 10)

    assert torch.isfinite(obs3["ego"]).all()
    assert torch.isfinite(obs3["threat_tokens"]).all()
    assert torch.isfinite(obs3["ally_tokens"]).all()
    assert (obs3["threat_tokens"][~obs3["threat_mask"]] == 0.0).all()
    assert (obs3["ally_tokens"][~obs3["ally_mask"]] == 0.0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_observation_threat_permutation_invariance_order(cfg, device: str):
    """Permuting missile order in env state yields identical sorted top-K threat tokens."""
    env = VectorizedSwarmEnv(cfg, num_envs=1, device=device)
    env.reset(num_interceptors=4, num_scouts=0, num_missiles=6, mode="3d", sigma_obs=0.0)

    obs1 = build_observations(env, K_threat=6, K_ally=3, use_estimated_state=False)

    # Permute missile order
    perm = torch.tensor([4, 1, 0, 5, 2, 3], device=device, dtype=torch.int64)
    env.missile_pos = env.missile_pos[:, perm, :]
    env.missile_vel = env.missile_vel[:, perm, :]
    env.missile_type = env.missile_type[:, perm]
    env.missile_target_hva = env.missile_target_hva[:, perm]
    env.missile_K = env.missile_K[:, perm]
    env.missile_is_decoy = env.missile_is_decoy[:, perm]
    env.missile_weave_phase = env.missile_weave_phase[:, perm]
    env.missile_spawn_time = env.missile_spawn_time[:, perm]
    env.missile_exists = env.missile_exists[:, perm]
    env.missile_status = env.missile_status[:, perm]
    env.z = env.z[:, :, perm]

    obs2 = build_observations(env, K_threat=6, K_ally=3, use_estimated_state=False)

    # Top-K threats must match within float precision
    assert torch.allclose(obs1["threat_tokens"], obs2["threat_tokens"], atol=1e-5)
    assert torch.equal(obs1["threat_mask"], obs2["threat_mask"])


@pytest.mark.parametrize("device", DEVICES)
def test_potential_based_shaping_zero_cycle_invariance(cfg, device: str):
    """Closed loop trajectory with gamma=1.0 yields net zero cumulative shaping reward."""
    env = VectorizedSwarmEnv(cfg, num_envs=1, device=device)
    env.reset(num_interceptors=1, num_scouts=0, num_missiles=1, mode="3d")

    env.drone_role.fill_(0)
    env.drone_active.fill_(True)
    env.z.fill_(False)
    env.z[0, 0, 0] = True
    env.missile_status.fill_(0)
    env.missile_exists.fill_(True)

    # Stationary missile
    env.missile_pos[0, 0] = torch.tensor([5000.0, 5000.0, 1000.0], device=device)
    env.missile_vel[0, 0] = torch.tensor([0.0, 0.0, 0.0], device=device)

    rc = RewardComputer(cfg, device=torch.device(device))
    rc.shaping_gamma = 1.0  # Discount = 1.0 to test telescoping invariance

    # Closed triangle trajectory: p0 -> p1 -> p2 -> p0
    p0 = torch.tensor([1000.0, 1000.0, 500.0], device=device)
    p1 = torch.tensor([2500.0, 1000.0, 600.0], device=device)
    p2 = torch.tensor([1800.0, 2200.0, 400.0], device=device)

    env.drone_pos[0, 0] = p0
    env.drone_vel[0, 0] = torch.zeros(3, device=device)
    phi0 = rc.compute_potential(env)

    dummy_info = {
        "intercept_event": torch.zeros((1, 1), dtype=torch.bool, device=device),
        "interceptor_of_missile": torch.full((1, 1), -1, dtype=torch.int64, device=device),
        "impact_event": torch.zeros((1, 1), dtype=torch.bool, device=device),
        "collision_mask": torch.zeros((1, 1, 1), dtype=torch.bool, device=device),
        "drone_a_applied": torch.zeros((1, 1, 3), device=device),
        "newly_done": torch.zeros((1,), dtype=torch.bool, device=device),
    }

    # Step 1: p0 -> p1
    env.drone_pos[0, 0] = p1
    _, comps1, phi1 = rc.compute_step_reward(env, phi0, prev_z=env.z, info=dummy_info)

    # Step 2: p1 -> p2
    env.drone_pos[0, 0] = p2
    _, comps2, phi2 = rc.compute_step_reward(env, phi1, prev_z=env.z, info=dummy_info)

    # Step 3: p2 -> p0
    env.drone_pos[0, 0] = p0
    _, comps3, phi3 = rc.compute_step_reward(env, phi2, prev_z=env.z, info=dummy_info)

    total_shaping = (comps1["shaping"] + comps2["shaping"] + comps3["shaping"])[0, 0].item()
    assert abs(total_shaping) < 1e-5, f"Cumulative shaping reward was {total_shaping}, expected ~0.0"


@pytest.mark.parametrize("device", DEVICES)
def test_reward_event_terms(cfg, device: str):
    """Verify specific numerical values for intercept bonuses, penalties, and collisions."""
    env = VectorizedSwarmEnv(cfg, num_envs=1, device=device)
    env.reset(num_interceptors=2, num_scouts=0, num_missiles=2, mode="3d")
    env.drone_active.fill_(True)
    env.missile_is_decoy.fill_(False)

    rc = RewardComputer(cfg, device=torch.device(device))
    phi0 = torch.zeros((1, 2), device=device)
    z = torch.zeros((1, 2, 2), dtype=torch.bool, device=device)
    z[0, 0, 0] = True  # Drone 0 assigned to missile 0
    env.z = z

    # 1. Assigned Intercept (+10 assigned + 5 team = +15 for drone 0; +5 for drone 1)
    info_int = {
        "intercept_event": torch.tensor([[True, False]], device=device),
        "interceptor_of_missile": torch.tensor([[0, -1]], device=device),
        "impact_event": torch.zeros((1, 2), dtype=torch.bool, device=device),
        "collision_mask": torch.zeros((1, 2, 2), dtype=torch.bool, device=device),
        "drone_a_applied": torch.zeros((1, 2, 3), device=device),
        "newly_done": torch.zeros((1,), dtype=torch.bool, device=device),
    }
    _, comps, _ = rc.compute_step_reward(env, phi0, prev_z=z, info=info_int)
    assert comps["assigned_intercept"][0, 0].item() == 10.0
    assert comps["assigned_intercept"][0, 1].item() == 0.0
    assert comps["team_intercept"][0, 0].item() == 5.0
    assert comps["team_intercept"][0, 1].item() == 5.0

    # 2. HVA Leaker Impact (-15.0 on active drones)
    info_leak = {
        "intercept_event": torch.zeros((1, 2), dtype=torch.bool, device=device),
        "interceptor_of_missile": torch.full((1, 2), -1, device=device),
        "impact_event": torch.tensor([[True, False]], device=device),
        "collision_mask": torch.zeros((1, 2, 2), dtype=torch.bool, device=device),
        "drone_a_applied": torch.zeros((1, 2, 3), device=device),
        "newly_done": torch.zeros((1,), dtype=torch.bool, device=device),
    }
    _, comps, _ = rc.compute_step_reward(env, phi0, prev_z=z, info=info_leak)
    assert comps["hva_leaker"][0, 0].item() == -15.0
    assert comps["hva_leaker"][0, 1].item() == -15.0

    # 3. Friendly Collision (-10.0 on colliding drones)
    col_mask = torch.zeros((1, 2, 2), dtype=torch.bool, device=device)
    col_mask[0, 0, 1] = True
    col_mask[0, 1, 0] = True
    info_col = {
        "intercept_event": torch.zeros((1, 2), dtype=torch.bool, device=device),
        "interceptor_of_missile": torch.full((1, 2), -1, device=device),
        "impact_event": torch.zeros((1, 2), dtype=torch.bool, device=device),
        "collision_mask": col_mask,
        "drone_a_applied": torch.zeros((1, 2, 3), device=device),
        "newly_done": torch.zeros((1,), dtype=torch.bool, device=device),
    }
    _, comps, _ = rc.compute_step_reward(env, phi0, prev_z=z, info=info_col)
    assert comps["collision"][0, 0].item() == -10.0
    assert comps["collision"][0, 1].item() == -10.0

    # 4. Duplicate Assignment Penalty (-2.0 if sum_i z_ij > K_j)
    env.missile_K.fill_(1)
    dup_z = torch.zeros((1, 2, 2), dtype=torch.bool, device=device)
    dup_z[0, 0, 0] = True
    dup_z[0, 1, 0] = True  # Both drones assigned to missile 0 (K=1)
    env.z = dup_z

    info_base = {
        "intercept_event": torch.zeros((1, 2), dtype=torch.bool, device=device),
        "interceptor_of_missile": torch.full((1, 2), -1, device=device),
        "impact_event": torch.zeros((1, 2), dtype=torch.bool, device=device),
        "collision_mask": torch.zeros((1, 2, 2), dtype=torch.bool, device=device),
        "drone_a_applied": torch.zeros((1, 2, 3), device=device),
        "newly_done": torch.zeros((1,), dtype=torch.bool, device=device),
    }
    _, comps, _ = rc.compute_step_reward(env, phi0, prev_z=dup_z, info=info_base)
    assert comps["duplicate"][0, 0].item() == -2.0
    assert comps["duplicate"][0, 1].item() == -2.0


def test_curriculum_progression(cfg):
    """Test 3-stage curriculum advancement conditions and capping at stage 3."""
    cm = CurriculumManager(cfg, initial_stage=1)
    assert cm.current_stage == 1

    # Stage 1: threshold = 0.90
    adv = cm.record_episodes([0.92] * 50)
    assert adv is True
    assert cm.current_stage == 2

    # Stage 2: threshold = 0.85; feeding 0.80 should stay in stage 2
    adv = cm.record_episodes([0.80] * 50)
    assert adv is False
    assert cm.current_stage == 2

    # Stage 2: feeding 0.88 should advance to stage 3
    adv = cm.record_episodes([0.88] * 50)
    assert adv is True
    assert cm.current_stage == 3

    # Stage 3: caps at stage 3 even with 100% win rate
    adv = cm.record_episodes([1.0] * 50)
    assert adv is False
    assert cm.current_stage == 3


@pytest.mark.parametrize("device", DEVICES)
def test_pettingzoo_parallel_wrapper(cfg, device: str):
    """Instantiate SwarmParallelEnv and run 10 multi-agent steps."""
    env = SwarmParallelEnv(cfg, device=device, seed=42)
    obs, infos = env.reset(seed=42)

    assert len(obs) == len(env.possible_agents)
    for agent in env.possible_agents:
        assert agent in obs
        assert obs[agent]["ego"].shape == (10,)
        assert obs[agent]["threat_tokens"].shape == (env.K_threat, 15)
        assert obs[agent]["ally_tokens"].shape == (env.K_ally, 9)

    for step_i in range(10):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)

        for agent in obs:
            assert np.all(np.isfinite(obs[agent]["ego"]))
            assert np.all(np.isfinite(obs[agent]["threat_tokens"]))
            assert np.all(np.isfinite(obs[agent]["ally_tokens"]))
            assert np.isfinite(rewards[agent])
            assert isinstance(terminations[agent], bool)
            assert isinstance(truncations[agent], bool)
            assert "shaping_reward" in infos[agent]

        if all(terminations.values()) or all(truncations.values()) or len(env.agents) == 0:
            break
