"""Pre-allocated GPU Rollout Buffer with Generalized Advantage Estimation (GAE).

Spec Section 6.7 & 8.2:
- Fixed-size float32 and bool tensors allocated once on device.
- Zero CPU host syncs during rollout step insertions.
- Vectorized GAE computation with episode done boundaries.
- Minibatch generator preserving the drone team dimension [B_mb, N, ...]
  for centralized critic team mean-pooling.
"""

from typing import Dict, Generator, Optional
import torch


class RolloutBuffer:
    """Pre-allocated GPU tensor buffer for multi-agent PPO rollouts."""

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        num_drones: int,
        K_threat: int = 15,
        K_ally: int = 10,
        ego_dim: int = 10,
        threat_dim: int = 15,
        ally_dim: int = 9,
        device: str = "cuda",
    ) -> None:
        """Allocate fixed GPU tensors for rollout collection.

        Args:
            rollout_steps: Number of transitions per env per rollout (T).
            num_envs: Number of parallel environments (E).
            num_drones: Number of drones per env (N).
            K_threat: Maximum observed threat tokens (K_t).
            K_ally: Maximum observed ally tokens (K_a).
            ego_dim: Dimension of ego feature vector (10).
            threat_dim: Dimension of threat token (15).
            ally_dim: Dimension of ally token (9).
            device: Storage device ("cuda" or "cpu").
        """
        self.rollout_steps = int(rollout_steps)
        self.num_envs = int(num_envs)
        self.num_drones = int(num_drones)
        self.K_threat = int(K_threat)
        self.K_ally = int(K_ally)
        self.ego_dim = int(ego_dim)
        self.threat_dim = int(threat_dim)
        self.ally_dim = int(ally_dim)
        self.device = torch.device(device)

        T = self.rollout_steps
        E = self.num_envs
        N = self.num_drones

        # Observations
        self.ego = torch.zeros((T, E, N, self.ego_dim), dtype=torch.float32, device=self.device)
        self.threat_tokens = torch.zeros(
            (T, E, N, self.K_threat, self.threat_dim), dtype=torch.float32, device=self.device
        )
        self.threat_mask = torch.zeros(
            (T, E, N, self.K_threat), dtype=torch.bool, device=self.device
        )
        self.ally_tokens = torch.zeros(
            (T, E, N, self.K_ally, self.ally_dim), dtype=torch.float32, device=self.device
        )
        self.ally_mask = torch.zeros((T, E, N, self.K_ally), dtype=torch.bool, device=self.device)
        self.drone_mask = torch.zeros((T, E, N), dtype=torch.bool, device=self.device)

        # Actions & policy outputs
        self.actions = torch.zeros((T, E, N, 3), dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros((T, E, N), dtype=torch.float32, device=self.device)

        # Rewards, dones & critic values
        self.rewards = torch.zeros((T, E, N), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((T, E), dtype=torch.bool, device=self.device)
        self.values = torch.zeros((T, E, N), dtype=torch.float32, device=self.device)

        # GAE targets
        self.advantages = torch.zeros((T, E, N), dtype=torch.float32, device=self.device)
        self.returns = torch.zeros((T, E, N), dtype=torch.float32, device=self.device)

        self.ptr = 0

    def reset(self) -> None:
        """Reset step pointer to begin new rollout collection."""
        self.ptr = 0

    def insert(
        self,
        obs: Dict[str, torch.Tensor],
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Write single rollout step directly into pre-allocated GPU slice with zero host syncs.

        Args:
            obs: Observation dict containing 'ego', 'threat_tokens', 'threat_mask',
                 'ally_tokens', 'ally_mask', 'drone_mask'.
            action: Continuous actions [E, N, 3] float32.
            log_prob: Log probability [E, N] float32.
            reward: Team/individual reward [E, N] float32.
            done: Env done flags [E] bool.
            value: Critic values [E, N] float32.
        """
        if self.ptr >= self.rollout_steps:
            raise IndexError(
                f"RolloutBuffer overflow: ptr={self.ptr} >= rollout_steps={self.rollout_steps}"
            )

        p = self.ptr
        self.ego[p].copy_(obs["ego"])
        self.threat_tokens[p].copy_(obs["threat_tokens"])
        self.threat_mask[p].copy_(obs["threat_mask"])
        self.ally_tokens[p].copy_(obs["ally_tokens"])
        self.ally_mask[p].copy_(obs["ally_mask"])
        self.drone_mask[p].copy_(obs["drone_mask"])

        self.actions[p].copy_(action)
        self.log_probs[p].copy_(log_prob)
        self.rewards[p].copy_(reward)
        self.dones[p].copy_(done)
        self.values[p].copy_(value)

        self.ptr += 1

    def compute_gae(
        self,
        next_value: torch.Tensor,
        next_done: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute Generalized Advantage Estimation (GAE) and discounted returns (Spec Eq. 6.7).

        Backward recurrence t = T - 1 ... 0:
            non_terminal = (~dones_next).float().unsqueeze(-1)
            delta = rewards[t] + gamma * next_val * non_terminal - values[t]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            advantages[t] = gae * drone_mask[t].float()
            returns[t] = advantages[t] + values[t]

        Args:
            next_value: Estimated value of observation at T [E, N].
            next_done: Episode done flags at T [E] bool.
            gamma: Discount factor.
            gae_lambda: GAE smoothing parameter.
        """
        next_val = next_value
        dones_next = next_done
        gae = torch.zeros_like(next_val)

        for t in reversed(range(self.rollout_steps)):
            # non_terminal shape: [E, 1] broadcastable over [E, N]
            if dones_next.dim() == 1:
                non_terminal = (~dones_next).float().unsqueeze(-1)
            else:
                non_terminal = (~dones_next).float()

            delta = self.rewards[t] + gamma * next_val * non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * non_terminal * gae

            # Mask inactive drones
            mask_t = self.drone_mask[t].float()
            self.advantages[t] = gae * mask_t
            self.returns[t] = self.advantages[t] + self.values[t]

            next_val = self.values[t]
            dones_next = self.dones[t]

    def get_minibatches(
        self,
        num_minibatches: int = 4,
        generator: Optional[torch.Generator] = None,
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Yield shuffled minibatches flattening (T, E) while preserving drone team dim N.

        Args:
            num_minibatches: Number of minibatches to yield.
            generator: Optional torch.Generator for reproducible shuffling.

        Yields:
            Dict containing:
                obs: Dict with 'ego', 'threat_tokens', 'threat_mask', 'ally_tokens',
                     'ally_mask', 'drone_mask' shaped [B_mb, N, ...].
                actions: [B_mb, N, 3]
                old_log_probs: [B_mb, N]
                values: [B_mb, N]
                advantages: [B_mb, N] (normalized over active drones)
                returns: [B_mb, N]
                drone_mask: [B_mb, N]
        """
        T = self.rollout_steps
        E = self.num_envs
        N = self.num_drones
        B_env = T * E

        batch_size = B_env // num_minibatches
        if batch_size == 0:
            raise ValueError(
                f"num_minibatches={num_minibatches} exceeds total environment steps B_env={B_env}"
            )

        # Normalize advantages over active drones across the buffer
        active_adv = self.advantages[self.drone_mask]
        if active_adv.numel() > 1:
            adv_mean = active_adv.mean()
            adv_std = active_adv.std()
            norm_advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
        else:
            norm_advantages = self.advantages.clone()
        norm_advantages = norm_advantages * self.drone_mask.float()

        # Flatten (T, E) -> B_env
        flat_ego = self.ego.view(B_env, N, self.ego_dim)
        flat_threat_tokens = self.threat_tokens.view(B_env, N, self.K_threat, self.threat_dim)
        flat_threat_mask = self.threat_mask.view(B_env, N, self.K_threat)
        flat_ally_tokens = self.ally_tokens.view(B_env, N, self.K_ally, self.ally_dim)
        flat_ally_mask = self.ally_mask.view(B_env, N, self.K_ally)
        flat_drone_mask = self.drone_mask.view(B_env, N)

        flat_actions = self.actions.view(B_env, N, 3)
        flat_old_log_probs = self.log_probs.view(B_env, N)
        flat_values = self.values.view(B_env, N)
        flat_advantages = norm_advantages.view(B_env, N)
        flat_returns = self.returns.view(B_env, N)

        # Random permutation of env steps
        if generator is not None and generator.device != self.device:
            indices = torch.randperm(B_env, generator=generator, device=generator.device).to(self.device)
        else:
            indices = torch.randperm(B_env, generator=generator, device=self.device)

        for i in range(num_minibatches):
            start = i * batch_size
            end = start + batch_size
            mb_idx = indices[start:end]

            yield {
                "obs": {
                    "ego": flat_ego[mb_idx],
                    "threat_tokens": flat_threat_tokens[mb_idx],
                    "threat_mask": flat_threat_mask[mb_idx],
                    "ally_tokens": flat_ally_tokens[mb_idx],
                    "ally_mask": flat_ally_mask[mb_idx],
                    "drone_mask": flat_drone_mask[mb_idx],
                },
                "actions": flat_actions[mb_idx],
                "old_log_probs": flat_old_log_probs[mb_idx],
                "values": flat_values[mb_idx],
                "advantages": flat_advantages[mb_idx],
                "returns": flat_returns[mb_idx],
                "drone_mask": flat_drone_mask[mb_idx],
            }

    def vram_bytes(self) -> int:
        """Calculate total VRAM bytes allocated across all internal buffer tensors."""
        tensors = [
            self.ego,
            self.threat_tokens,
            self.threat_mask,
            self.ally_tokens,
            self.ally_mask,
            self.drone_mask,
            self.actions,
            self.log_probs,
            self.rewards,
            self.dones,
            self.values,
            self.advantages,
            self.returns,
        ]
        return sum(t.element_size() * t.nelement() for t in tensors)
