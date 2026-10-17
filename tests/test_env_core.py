"""Comprehensive test suite for VectorizedSwarmEnv core mechanics and invariants."""

import copy
from pathlib import Path
import pytest
import torch

from src.config import load_config
from src.envs.vectorized_swarm_env import VectorizedSwarmEnv

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default_sim.yaml"

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


def make_test_cfg(**overrides) -> dict:
    cfg = copy.deepcopy(load_config(CONFIG_PATH))
    for k, v in overrides.items():
        keys = k.split(".")
        d = cfg
        for sub_k in keys[:-1]:
            d = d.setdefault(sub_k, {})
        d[keys[-1]] = v
    return cfg


@pytest.mark.parametrize("device", DEVICES)
def test_env_reset_shapes_and_states(device):
    """(a) reset creates all expected state tensors with right shapes, t=0, done=False, status=0."""
    E, N_int, N_sct, M = 4, 16, 4, 30
    N = N_int + N_sct
    cfg = make_test_cfg(
        **{
            "environment.num_parallel_envs": E,
            "entities.drones.num_interceptors": N_int,
            "entities.drones.num_scouts": N_sct,
            "entities.missiles.num_missiles": M,
        }
    )
    env = VectorizedSwarmEnv(cfg, num_envs=E, device=device, seed=42)

    assert env.t.shape == (E,) and (env.t == 0).all()
    assert env.done.shape == (E,) and (~env.done).all()
    assert env.truncated.shape == (E,) and (~env.truncated).all()

    assert env.hva_pos.shape == (E, 2, 3)
    assert env.drone_pos.shape == (E, N, 3)
    assert env.drone_vel.shape == (E, N, 3)
    assert env.drone_role.shape == (E, N)
    assert env.drone_active.shape == (E, N) and env.drone_active.all()

    assert env.missile_pos.shape == (E, M, 3)
    assert env.missile_vel.shape == (E, M, 3)
    assert env.missile_type.shape == (E, M)
    assert env.missile_target_hva.shape == (E, M)
    assert env.missile_K.shape == (E, M)
    assert env.missile_is_decoy.shape == (E, M)
    assert env.missile_status.shape == (E, M) and (env.missile_status == 0).all()

    assert env.z.shape == (E, N, M)
    assert env.delta_v.shape == (E,) and (env.delta_v == 0.0).all()
    assert env.friendly_collisions.shape == (E,) and (env.friendly_collisions == 0.0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_head_on_tunneling_interception(device):
    """(b) Head-on tunneling at env level with intercept_radius=2.0 detects intercept."""
    cfg = make_test_cfg(
        **{
            "environment.num_parallel_envs": 1,
            "entities.drones.num_interceptors": 1,
            "entities.drones.num_scouts": 0,
            "entities.missiles.num_missiles": 1,
            "entities.drones.intercept_radius": 2.0,
            "entities.drones.linear_drag": 0.0,
            "entities.missiles.linear_drag": 0.0,
        }
    )
    env = VectorizedSwarmEnv(cfg, num_envs=1, device=device, seed=123)

    # Handcraft tunneling geometry: closing speed 500 m/s over dt=0.05 s
    env.drone_pos[0, 0] = torch.tensor([-10.0, 0.0, 100.0], device=device)
    env.drone_vel[0, 0] = torch.tensor([250.0, 0.0, 0.0], device=device)
    env.drone_role[0, 0] = 0  # interceptor
    env.drone_active[0, 0] = True

    env.missile_pos[0, 0] = torch.tensor([10.0, 0.0, 100.0], device=device)
    env.missile_vel[0, 0] = torch.tensor([-250.0, 0.0, 0.0], device=device)
    env.missile_type[0, 0] = 0  # ballistic
    env.missile_spawn_time[0, 0] = 0.0
    env.missile_status[0, 0] = 0

    zero_cmd = torch.zeros(1, 1, 3, device=device)
    info = env.step(zero_cmd)

    assert env.missile_status[0, 0].item() == 1
    assert info["interceptor_of_missile"][0, 0].item() == 0
    assert info["intercept_event"][0, 0].item() is True


@pytest.mark.parametrize("device", DEVICES)
def test_scout_cannot_intercept(device):
    """(c) Same tunneling geometry but drone role is scout (1) -> status stays 0."""
    cfg = make_test_cfg(
        **{
            "environment.num_parallel_envs": 1,
            "entities.drones.num_interceptors": 0,
            "entities.drones.num_scouts": 1,
            "entities.missiles.num_missiles": 1,
            "entities.drones.intercept_radius": 2.0,
            "entities.drones.linear_drag": 0.0,
            "entities.missiles.linear_drag": 0.0,
        }
    )
    env = VectorizedSwarmEnv(cfg, num_envs=1, device=device, seed=123)

    env.drone_pos[0, 0] = torch.tensor([-10.0, 0.0, 100.0], device=device)
    env.drone_vel[0, 0] = torch.tensor([250.0, 0.0, 0.0], device=device)
    env.drone_role[0, 0] = 1  # scout
    env.drone_active[0, 0] = True

    env.missile_pos[0, 0] = torch.tensor([10.0, 0.0, 100.0], device=device)
    env.missile_vel[0, 0] = torch.tensor([-250.0, 0.0, 0.0], device=device)
    env.missile_type[0, 0] = 0
    env.missile_spawn_time[0, 0] = 0.0
    env.missile_status[0, 0] = 0

    zero_cmd = torch.zeros(1, 1, 3, device=device)
    info = env.step(zero_cmd)

    assert env.missile_status[0, 0].item() == 0
    assert info["interceptor_of_missile"][0, 0].item() == -1
    assert info["intercept_event"][0, 0].item() is False


@pytest.mark.parametrize("device", DEVICES)
def test_decoy_intercept_metric(device):
    """(d) Decoy intercepted -> status 1, intercept_rate=0 (real only), decoy_intercepts=1."""
    cfg = make_test_cfg(
        **{
            "environment.num_parallel_envs": 1,
            "entities.drones.num_interceptors": 1,
            "entities.drones.num_scouts": 0,
            "entities.missiles.num_missiles": 1,
            "entities.drones.intercept_radius": 10.0,
            "entities.drones.linear_drag": 0.0,
            "entities.missiles.linear_drag": 0.0,
        }
    )
    env = VectorizedSwarmEnv(cfg, num_envs=1, device=device, seed=123)

    env.drone_pos[0, 0] = torch.tensor([0.0, 0.0, 100.0], device=device)
    env.drone_vel[0, 0] = torch.zeros(3, device=device)
    env.drone_role[0, 0] = 0
    env.drone_active[0, 0] = True

    env.missile_pos[0, 0] = torch.tensor([1.0, 0.0, 100.0], device=device)
    env.missile_vel[0, 0] = torch.zeros(3, device=device)
    env.missile_is_decoy[0, 0] = True
    env.missile_spawn_time[0, 0] = 0.0
    env.missile_status[0, 0] = 0

    env.step(torch.zeros(1, 1, 3, device=device))

    assert env.missile_status[0, 0].item() == 1
    m = env.metrics()
    assert m["decoy_intercepts"][0].item() == 1.0
    assert m["intercept_rate"][0].item() == 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_hva_impact(device):
    """(e) Ballistic missile flying straight at an HVA from 1 km at ~400 m/s reaches status 2."""
    cfg = make_test_cfg(
        **{
            "environment.num_parallel_envs": 1,
            "entities.drones.num_interceptors": 1,
            "entities.drones.num_scouts": 0,
            "entities.missiles.num_missiles": 1,
            "scenario.hva_impact_radius": 50.0,
            "entities.missiles.linear_drag": 0.0,
        }
    )
    env = VectorizedSwarmEnv(cfg, num_envs=1, device=device, seed=123)

    # Place HVA 0 at origin, drone far away
    env.hva_pos[0, 0] = torch.zeros(3, device=device)
    env.drone_pos[0, 0] = torch.tensor([4000.0, 4000.0, 1000.0], device=device)
    env.drone_vel[0, 0] = torch.zeros(3, device=device)

    # Missile 1 km away heading at -400 m/s along x towards origin
    env.missile_pos[0, 0] = torch.tensor([1000.0, 0.0, 0.0], device=device)
    env.missile_vel[0, 0] = torch.tensor([-400.0, 0.0, 0.0], device=device)
    env.missile_type[0, 0] = 0
    env.missile_is_decoy[0, 0] = False
    env.missile_spawn_time[0, 0] = 0.0
    env.missile_status[0, 0] = 0

    zero_cmd = torch.zeros(1, 1, 3, device=device)
    for _ in range(60):
        if env.missile_status[0, 0].item() != 0:
            break
        env.step(zero_cmd)

    assert env.missile_status[0, 0].item() == 2
    m = env.metrics()
    assert m["leaker_rate"][0].item() == 1.0


@pytest.mark.parametrize("device", DEVICES)
def test_unspawned_missile_stationary_and_uninterceptable(device):
    """(f) Unspawned missile (spawn_time=100 s) does not move for 20 steps and cannot be intercepted."""
    cfg = make_test_cfg(
        **{
            "environment.num_parallel_envs": 1,
            "entities.drones.num_interceptors": 1,
            "entities.drones.num_scouts": 0,
            "entities.missiles.num_missiles": 1,
            "entities.drones.intercept_radius": 50.0,
        }
    )
    env = VectorizedSwarmEnv(cfg, num_envs=1, device=device, seed=123)

    pos_init = torch.tensor([500.0, 500.0, 500.0], device=device)
    env.missile_pos[0, 0] = pos_init.clone()
    env.missile_vel[0, 0] = torch.tensor([-300.0, 0.0, 0.0], device=device)
    env.missile_spawn_time[0, 0] = 100.0
    env.missile_status[0, 0] = 0

    # Place drone directly on the missile
    env.drone_pos[0, 0] = pos_init.clone()
    env.drone_vel[0, 0] = torch.zeros(3, device=device)

    zero_cmd = torch.zeros(1, 1, 3, device=device)
    for _ in range(20):
        info = env.step(zero_cmd)
        assert not info["intercept_event"].any()
        assert env.missile_status[0, 0].item() == 0
        assert torch.equal(env.missile_pos[0, 0], pos_init)


@pytest.mark.parametrize("device", DEVICES)
def test_termination_and_frozen_done_envs(device):
    """(g) Termination on all resolved or max_episode_steps; done envs are frozen."""
    cfg = make_test_cfg(
        **{
            "environment.num_parallel_envs": 2,
            "entities.drones.num_interceptors": 1,
            "entities.drones.num_scouts": 0,
            "entities.missiles.num_missiles": 1,
            "environment.max_episode_steps": 5,
        }
    )
    env = VectorizedSwarmEnv(cfg, num_envs=2, device=device, seed=123)

    # Env 0: immediately resolve missile (intercepted)
    env.missile_status[0, 0] = 1

    zero_cmd = torch.zeros(2, 1, 3, device=device)
    env.step(zero_cmd)
    assert env.done[0].item() is True
    assert env.truncated[0].item() is False

    # Snapshot env 0 state
    d_pos_snap = env.drone_pos[0].clone()
    m_pos_snap = env.missile_pos[0].clone()
    t_snap = env.t[0].item()

    # Step until max_episode_steps (5)
    for _ in range(10):
        env.step(zero_cmd)

    # Env 0 was done and must remain completely unchanged
    assert env.t[0].item() == t_snap
    assert torch.equal(env.drone_pos[0], d_pos_snap)
    assert torch.equal(env.missile_pos[0], m_pos_snap)

    # Env 1 should have timed out at t=5 and become truncated
    assert env.t[1].item() == 5
    assert env.done[1].item() is True
    assert env.truncated[1].item() is True


@pytest.mark.parametrize("device", DEVICES)
def test_reset_envs_isolation(device):
    """(h) reset_envs resets only masked env; unmasked envs remain bit-identical."""
    cfg = make_test_cfg(**{"environment.num_parallel_envs": 4})
    env = VectorizedSwarmEnv(cfg, num_envs=4, device=device, seed=123)

    # Step once with random commands
    cmd = torch.randn(4, 20, 3, device=device) * 50.0
    env.step(cmd)

    # Snapshot unmasked envs: 0, 2, 3
    unmasked = [0, 2, 3]
    d_pos_snap = env.drone_pos[unmasked].clone()
    m_pos_snap = env.missile_pos[unmasked].clone()
    t_snap = env.t[unmasked].clone()

    mask = torch.tensor([False, True, False, False], device=device, dtype=torch.bool)
    env.reset_envs(mask)

    assert env.t[1].item() == 0
    assert env.done[1].item() is False

    assert torch.equal(env.drone_pos[unmasked], d_pos_snap)
    assert torch.equal(env.missile_pos[unmasked], m_pos_snap)
    assert torch.equal(env.t[unmasked], t_snap)


@pytest.mark.parametrize("device", DEVICES)
def test_interceptor_expended_on_intercept(device):
    """(i) interceptor_expended_on_intercept: True -> inactive; False -> stays active."""
    for expended_flag in [True, False]:
        cfg = make_test_cfg(
            **{
                "environment.num_parallel_envs": 1,
                "entities.drones.num_interceptors": 1,
                "entities.drones.num_scouts": 0,
                "entities.missiles.num_missiles": 1,
                "entities.drones.intercept_radius": 10.0,
                "environment.interceptor_expended_on_intercept": expended_flag,
            }
        )
        env = VectorizedSwarmEnv(cfg, num_envs=1, device=device, seed=42)
        env.drone_pos[0, 0] = torch.tensor([0.0, 0.0, 100.0], device=device)
        env.drone_vel[0, 0] = torch.zeros(3, device=device)
        env.drone_role[0, 0] = 0
        env.drone_active[0, 0] = True

        env.missile_pos[0, 0] = torch.tensor([1.0, 0.0, 100.0], device=device)
        env.missile_vel[0, 0] = torch.zeros(3, device=device)
        env.missile_spawn_time[0, 0] = 0.0
        env.missile_status[0, 0] = 0

        env.step(torch.zeros(1, 1, 3, device=device))
        assert env.missile_status[0, 0].item() == 1
        expected_active = not expended_flag
        assert env.drone_active[0, 0].item() is expected_active


@pytest.mark.parametrize("device", DEVICES)
def test_drone_boundary_containment(device):
    """(j) Drone commanded with 1e6 accel for 100 steps stays inside box and z >= 0."""
    cfg = make_test_cfg(**{"environment.num_parallel_envs": 1})
    env = VectorizedSwarmEnv(cfg, num_envs=1, device=device, seed=42)
    W, L, H = env.world_bounds

    # Extreme outward acceleration
    accel_cmd = torch.full((1, env.num_interceptors + env.num_scouts, 3), 1e6, device=device)
    for _ in range(100):
        env.step(accel_cmd)

    dp = env.drone_pos
    assert (dp[..., 0] >= -W / 2.0 - 1e-4).all() and (dp[..., 0] <= W / 2.0 + 1e-4).all()
    assert (dp[..., 1] >= -L / 2.0 - 1e-4).all() and (dp[..., 1] <= L / 2.0 + 1e-4).all()
    assert (dp[..., 2] >= -1e-4).all() and (dp[..., 2] <= H + 1e-4).all()


@pytest.mark.parametrize("device", DEVICES)
def test_delta_v_accounting(device):
    """(k) Constant 10 m/s^2 command for 10 steps with drag 0 -> delta_v == 5.0 (tol 1e-3)."""
    cfg = make_test_cfg(
        **{
            "environment.num_parallel_envs": 1,
            "entities.drones.num_interceptors": 1,
            "entities.drones.num_scouts": 0,
            "entities.missiles.num_missiles": 1,
            "entities.drones.linear_drag": 0.0,
            "environment.max_episode_steps": 100,
        }
    )
    env = VectorizedSwarmEnv(cfg, num_envs=1, device=device, seed=42)
    # Ensure missile doesn't terminate the episode early
    env.missile_spawn_time[0, 0] = 1000.0

    cmd = torch.tensor([[[10.0, 0.0, 0.0]]], device=device)
    for _ in range(10):
        env.step(cmd)

    expected_dv = 10.0 * 0.05 * 10
    assert env.delta_v[0].item() == pytest.approx(expected_dv, abs=1e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_rollout_invariants(device):
    """(l) 300 steps random rollout on E=8: no NaNs, status monotonicity, n_real accounting."""
    E, N_int, N_sct, M = 8, 16, 4, 30
    cfg = make_test_cfg(
        **{
            "environment.num_parallel_envs": E,
            "entities.drones.num_interceptors": N_int,
            "entities.drones.num_scouts": N_sct,
            "entities.missiles.num_missiles": M,
            "environment.max_episode_steps": 300,
        }
    )
    env = VectorizedSwarmEnv(cfg, num_envs=E, device=device, seed=999)
    rng = torch.Generator(device=device).manual_seed(101)

    prev_status = env.missile_status.clone()
    real_mask = env.missile_exists & (~env.missile_is_decoy)
    n_real = real_mask.sum(dim=-1)

    for _ in range(300):
        cmd = torch.randn(E, N_int + N_sct, 3, generator=rng, device=device) * 50.0
        env.step(cmd)

        # 1. No NaN / Inf in any state tensor
        assert torch.isfinite(env.drone_pos).all()
        assert torch.isfinite(env.drone_vel).all()
        assert torch.isfinite(env.missile_pos).all()
        assert torch.isfinite(env.missile_vel).all()

        # 2. Status never changes from nonzero back to 0
        status_went_back = (prev_status != 0) & (env.missile_status == 0)
        assert not status_went_back.any()
        prev_status = env.missile_status.clone()

        # 3. Real missile partition invariant
        s = env.missile_status
        int_real = (real_mask & (s == 1)).sum(dim=-1)
        imp_real = (real_mask & (s == 2)).sum(dim=-1)
        exp_real = (real_mask & (s == 3)).sum(dim=-1)
        inf_real = (real_mask & (s == 0)).sum(dim=-1)
        assert torch.equal(n_real, int_real + imp_real + exp_real + inf_real)
