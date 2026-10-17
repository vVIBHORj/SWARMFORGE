"""Tests for deterministic scenario generation and environment rollouts given identical seeds."""

from pathlib import Path
import pytest
import torch

from src.config import load_config
from src.envs.scenario_generator import generate_scenario
from src.envs.vectorized_swarm_env import VectorizedSwarmEnv

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default_sim.yaml"

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.fixture
def base_cfg():
    return load_config(CONFIG_PATH)


@pytest.mark.parametrize("device", DEVICES)
def test_scenario_seed_determinism(base_cfg, device):
    """Same seed twice => every tensor identical; different seed => positions differ."""
    E, N_int, N_sct, M = 4, 16, 4, 30

    s1 = generate_scenario(base_cfg, E, N_int, N_sct, M, device=device, seed=12345)
    s2 = generate_scenario(base_cfg, E, N_int, N_sct, M, device=device, seed=12345)

    # Every tensor in s1 and s2 must be bit-identical
    for k, v in s1.items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(v, s2[k]), f"Tensor mismatch for key '{k}' with identical seed"
        elif isinstance(v, dict):
            assert v == s2[k]

    # Different seed must produce different entity positions
    s3 = generate_scenario(base_cfg, E, N_int, N_sct, M, device=device, seed=99999)
    assert not torch.equal(s1["drone_pos"], s3["drone_pos"])
    assert not torch.equal(s1["missile_pos"], s3["missile_pos"])
    assert not torch.equal(s1["hva_pos"], s3["hva_pos"])


@pytest.mark.parametrize("device", DEVICES)
def test_env_rollout_determinism(base_cfg, device):
    """Two envs built with same seed and driven by identical actions give identical states/metrics."""
    E, N = 4, 20
    env1 = VectorizedSwarmEnv(base_cfg, num_envs=E, device=device, seed=42)
    env2 = VectorizedSwarmEnv(base_cfg, num_envs=E, device=device, seed=42)

    rng1 = torch.Generator(device=device).manual_seed(999)
    rng2 = torch.Generator(device=device).manual_seed(999)

    for _ in range(100):
        cmd1 = torch.randn(E, N, 3, generator=rng1, device=device) * 50.0
        cmd2 = torch.randn(E, N, 3, generator=rng2, device=device) * 50.0

        info1 = env1.step(cmd1)
        info2 = env2.step(cmd2)

        assert torch.equal(info1["intercept_event"], info2["intercept_event"])
        assert torch.equal(info1["impact_event"], info2["impact_event"])
        assert torch.equal(info1["expire_event"], info2["expire_event"])
        assert torch.equal(info1["interceptor_of_missile"], info2["interceptor_of_missile"])

    # State tensors must be bit-identical
    assert torch.equal(env1.drone_pos, env2.drone_pos)
    assert torch.equal(env1.drone_vel, env2.drone_vel)
    assert torch.equal(env1.missile_pos, env2.missile_pos)
    assert torch.equal(env1.missile_vel, env2.missile_vel)
    assert torch.equal(env1.missile_status, env2.missile_status)
    assert torch.equal(env1.t, env2.t)
    assert torch.equal(env1.done, env2.done)

    # Metrics must be bit-identical
    m1 = env1.metrics()
    m2 = env2.metrics()
    for k in m1:
        assert torch.equal(m1[k], m2[k]), f"Metric mismatch for '{k}'"

    # Different seed gives different scenario
    env3 = VectorizedSwarmEnv(base_cfg, num_envs=E, device=device, seed=999)
    assert not torch.equal(env1.missile_pos, env3.missile_pos)
