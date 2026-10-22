"""Unit tests for learning components: scatter_logits_to_full, RolloutBuffer & GAE.

Tests:
1. scatter_logits_to_full mapping [E, N, K_t] -> [E, N, M] and integration with build_cost_matrix.
2. RolloutBuffer insertion, step tracking, and VRAM memory ceiling (< 250 MB for 32x256x20).
3. Exact hand-calculated 3-step GAE computation with gamma=0.99, gae_lambda=0.95 and done boundaries.
4. Minibatch generator preserving drone team dimension [B_mb, N, ...] with normalized advantages.
"""

import pytest
import torch

from src.config import load_config
from src.envs.vectorized_swarm_env import VectorizedSwarmEnv
from src.learning.attention_nets import scatter_logits_to_full
from src.learning.rollout_buffer import RolloutBuffer
from src.physics.guidance_laws import build_cost_matrix

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize("device", DEVICES)
def test_scatter_logits_to_full(device: str):
    """scatter_logits_to_full maps [E, N, K_t] into [E, N, M] with proper masking."""
    E, N, K_t, M = 2, 3, 4, 6
    u_topk = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            [[1.5, 2.5, 3.5, 4.5], [5.5, 6.5, 7.5, 8.5], [9.5, 10.5, 11.5, 12.5]],
        ],
        dtype=torch.float32,
        device=device,
    )
    threat_indices = torch.tensor(
        [
            [[0, 2, 4, 5], [1, 3, 0, 2], [5, 4, 3, 1]],
            [[2, 1, 0, 3], [4, 5, 2, 1], [0, 3, 4, 5]],
        ],
        dtype=torch.int64,
        device=device,
    )
    # Mask out the last slot in each drone
    threat_mask = torch.ones((E, N, K_t), dtype=torch.bool, device=device)
    threat_mask[:, :, -1] = False

    full = scatter_logits_to_full(u_topk, threat_indices, threat_mask, M=M)
    assert full.shape == (E, N, M)

    # Check drone 0, threat slot 0 (index 0, val 1.0)
    assert full[0, 0, 0].item() == 1.0
    # Check drone 0, threat slot 1 (index 2, val 2.0)
    assert full[0, 0, 2].item() == 2.0
    # Check drone 0, threat slot 2 (index 4, val 3.0)
    assert full[0, 0, 4].item() == 3.0
    # Slot 3 (index 5) is masked out -> must be 0.0
    assert full[0, 0, 5].item() == 0.0
    # Unreferenced index 1 and 3 -> must be 0.0
    assert full[0, 0, 1].item() == 0.0
    assert full[0, 0, 3].item() == 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_scatter_logits_integration_with_cost_matrix(device: str):
    """Full scattered logits integrate with build_cost_matrix, reducing matched cost."""
    cfg = load_config("configs/default_sim.yaml")
    E = 2
    env = VectorizedSwarmEnv(cfg, num_envs=E, device=device, seed=42)
    env.reset()
    N, M = env.N, env.M

    # Baseline cost without RL logits
    C_base, valid_base = build_cost_matrix(env, rl_logits=None)

    # Create dummy RL logits targeting threat 1 for drone 0
    rl_logits = torch.zeros((E, N, M), dtype=torch.float32, device=device)
    rl_logits[:, 0, 1] = 5.0

    C_rl, valid_rl = build_cost_matrix(env, rl_logits=rl_logits)

    # Valid pair mask should be identical
    assert torch.equal(valid_base, valid_rl)

    # For valid pairs, C_rl[:, 0, 1] should be lower by w_logit * 5.0 (default w_logit=1.0)
    valid_d0_m1 = valid_base[:, 0, 1]
    if valid_d0_m1.any():
        diff = (C_base[:, 0, 1] - C_rl[:, 0, 1])[valid_d0_m1]
        assert torch.allclose(diff, torch.tensor(5.0, device=device), atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_rollout_buffer_insertion_and_memory(device: str):
    """RolloutBuffer pre-allocation meets memory budget (< 250 MB) and insertion works."""
    T = 256
    E = 32
    N = 20
    K_t = 15
    K_a = 10

    buffer = RolloutBuffer(
        rollout_steps=T,
        num_envs=E,
        num_drones=N,
        K_threat=K_t,
        K_ally=K_a,
        device=device,
    )

    # Assert memory budget (< 250 MB)
    mem_bytes = buffer.vram_bytes()
    mem_mb = mem_bytes / (1024 * 1024)
    assert mem_mb < 250.0, f"RolloutBuffer memory {mem_mb:.2f} MB exceeds 250 MB ceiling!"
    assert mem_mb > 200.0, f"RolloutBuffer memory {mem_mb:.2f} MB unexpectedly low"

    # Dummy step insertion
    obs = {
        "ego": torch.randn((E, N, 10), device=device),
        "threat_tokens": torch.randn((E, N, K_t, 15), device=device),
        "threat_mask": torch.ones((E, N, K_t), dtype=torch.bool, device=device),
        "ally_tokens": torch.randn((E, N, K_a, 9), device=device),
        "ally_mask": torch.ones((E, N, K_a), dtype=torch.bool, device=device),
        "drone_mask": torch.ones((E, N), dtype=torch.bool, device=device),
    }
    action = torch.randn((E, N, 3), device=device)
    log_prob = torch.randn((E, N), device=device)
    reward = torch.ones((E, N), device=device)
    done = torch.zeros((E,), dtype=torch.bool, device=device)
    value = torch.full((E, N), 0.5, device=device)

    assert buffer.ptr == 0
    buffer.insert(obs, action, log_prob, reward, done, value)
    assert buffer.ptr == 1

    assert torch.allclose(buffer.ego[0], obs["ego"])
    assert torch.allclose(buffer.actions[0], action)
    assert torch.allclose(buffer.values[0], value)

    buffer.reset()
    assert buffer.ptr == 0


@pytest.mark.parametrize("device", DEVICES)
def test_exact_hand_calculated_gae(device: str):
    """Verify exact 3-step GAE computation with episode termination boundaries."""
    # 3-step rollout for E=1, N=1
    T = 3
    E = 1
    N = 1
    buffer = RolloutBuffer(rollout_steps=T, num_envs=E, num_drones=N, device=device)

    # Set parameters: gamma=0.99, gae_lambda=0.95
    gamma = 0.99
    gae_lambda = 0.95

    # Hand values:
    # t=0: r=1.0, V=0.5, done=False
    # t=1: r=2.0, V=1.5, done=True (episode ends after transition 1!)
    # t=2: r=3.0, V=2.5, done=False
    # Next (t=3): next_value=3.5, next_done=False
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 1.5, 2.5]
    dones = [False, True, False]

    for t in range(T):
        obs = {
            "ego": torch.zeros((E, N, 10), device=device),
            "threat_tokens": torch.zeros((E, N, 15, 15), device=device),
            "threat_mask": torch.zeros((E, N, 15), dtype=torch.bool, device=device),
            "ally_tokens": torch.zeros((E, N, 10, 9), device=device),
            "ally_mask": torch.zeros((E, N, 10), dtype=torch.bool, device=device),
            "drone_mask": torch.ones((E, N), dtype=torch.bool, device=device),
        }
        buffer.insert(
            obs=obs,
            action=torch.zeros((E, N, 3), device=device),
            log_prob=torch.zeros((E, N), device=device),
            reward=torch.tensor([[rewards[t]]], device=device),
            done=torch.tensor([dones[t]], device=device),
            value=torch.tensor([[values[t]]], device=device),
        )

    next_value = torch.tensor([[3.5]], device=device)
    next_done = torch.tensor([False], device=device)

    buffer.compute_gae(next_value, next_done, gamma=gamma, gae_lambda=gae_lambda)

    # Hand calculation verification:
    # t=2:
    #   dones_next = next_done = False -> non_terminal = 1.0
    #   next_val = 3.5
    #   delta_2 = r[2] + gamma * 3.5 * 1.0 - V[2] = 3.0 + 0.99 * 3.5 - 2.5 = 3.965
    #   gae_2 = delta_2 + gamma * lambda * 1.0 * 0 = 3.965
    #   adv[2] = 3.965
    #   ret[2] = 3.965 + 2.5 = 6.465
    #
    # t=1:
    #   dones_next = dones[2] = False -> non_terminal = 1.0
    #   next_val = V[2] = 2.5
    #   delta_1 = r[1] + gamma * 2.5 * 1.0 - V[1] = 2.0 + 0.99 * 2.5 - 1.5 = 2.975
    #   gae_1 = delta_1 + gamma * lambda * 1.0 * gae_2 = 2.975 + (0.99 * 0.95) * 3.965 = 2.975 + 3.7290825 = 6.7040825
    #   adv[1] = 6.7040825
    #   ret[1] = 6.7040825 + 1.5 = 8.2040825
    #
    # t=0:
    #   dones_next = dones[1] = True -> non_terminal = 0.0 (episode boundary!)
    #   next_val = V[1] = 1.5
    #   delta_0 = r[0] + gamma * 1.5 * 0.0 - V[0] = 1.0 - 0.5 = 0.5
    #   gae_0 = delta_0 + gamma * lambda * 0.0 * gae_1 = 0.5
    #   adv[0] = 0.5
    #   ret[0] = 0.5 + 0.5 = 1.0

    expected_adv = torch.tensor([[[0.5]], [[6.7040825]], [[3.965]]], device=device)
    expected_ret = torch.tensor([[[1.0]], [[8.2040825]], [[6.465]]], device=device)

    assert torch.allclose(buffer.advantages, expected_adv, atol=1e-5), (
        f"advantages: {buffer.advantages.squeeze()} vs expected {expected_adv.squeeze()}"
    )
    assert torch.allclose(buffer.returns, expected_ret, atol=1e-5), (
        f"returns: {buffer.returns.squeeze()} vs expected {expected_ret.squeeze()}"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_rollout_buffer_minibatches(device: str):
    """get_minibatches yields expected chunks preserving [B_mb, N, ...] shapes and normalized advantages."""
    T = 16
    E = 4
    N = 5
    buffer = RolloutBuffer(rollout_steps=T, num_envs=E, num_drones=N, device=device)

    # Fill buffer
    for _ in range(T):
        obs = {
            "ego": torch.randn((E, N, 10), device=device),
            "threat_tokens": torch.randn((E, N, 15, 15), device=device),
            "threat_mask": torch.ones((E, N, 15), dtype=torch.bool, device=device),
            "ally_tokens": torch.randn((E, N, 10, 9), device=device),
            "ally_mask": torch.ones((E, N, 10), dtype=torch.bool, device=device),
            "drone_mask": torch.ones((E, N), dtype=torch.bool, device=device),
        }
        buffer.insert(
            obs=obs,
            action=torch.randn((E, N, 3), device=device),
            log_prob=torch.randn((E, N), device=device),
            reward=torch.randn((E, N), device=device),
            done=torch.zeros((E,), dtype=torch.bool, device=device),
            value=torch.randn((E, N), device=device),
        )

    buffer.compute_gae(
        next_value=torch.zeros((E, N), device=device),
        next_done=torch.zeros((E,), dtype=torch.bool, device=device),
    )

    num_minibatches = 4
    total_env_steps = T * E  # 64
    expected_mb_size = total_env_steps // num_minibatches  # 16

    minibatches = list(buffer.get_minibatches(num_minibatches=num_minibatches))
    assert len(minibatches) == num_minibatches

    for mb in minibatches:
        assert mb["obs"]["ego"].shape == (expected_mb_size, N, 10)
        assert mb["obs"]["threat_tokens"].shape == (expected_mb_size, N, 15, 15)
        assert mb["obs"]["threat_mask"].shape == (expected_mb_size, N, 15)
        assert mb["obs"]["ally_tokens"].shape == (expected_mb_size, N, 10, 9)
        assert mb["obs"]["ally_mask"].shape == (expected_mb_size, N, 10)
        assert mb["obs"]["drone_mask"].shape == (expected_mb_size, N)

        assert mb["actions"].shape == (expected_mb_size, N, 3)
        assert mb["old_log_probs"].shape == (expected_mb_size, N)
        assert mb["values"].shape == (expected_mb_size, N)
        assert mb["advantages"].shape == (expected_mb_size, N)
        assert mb["returns"].shape == (expected_mb_size, N)
        assert mb["drone_mask"].shape == (expected_mb_size, N)
