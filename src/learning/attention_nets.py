"""Permutation-Invariant Cross-Attention Actor-Critic & Flat MLP Ablation.

Spec Section 7.1 & 7.2:
- Ego Encoder: Linear(10 -> d_model) + LayerNorm + SiLU -> Q_i [B, 1, d_model]
- Shared Threat Encoder: Linear(15 -> d_model) + SiLU + Linear + LayerNorm + SiLU -> E_threat [B, K_t, d_model]
- Shared Ally Encoder: Linear(9 -> d_model) + SiLU + Linear + LayerNorm + SiLU -> E_ally [B, K_a, d_model]
- Multi-Head Cross-Attention: nn.MultiheadAttention(embed_dim=d_model, num_heads=2)
- Residual Acceleration Head: tanh(Linear(d_model -> 3)) * a_res_max
- Coordination Logit Pointer Head: dot-product pointing over Top-K threats
- Centralized Critic: team mean-pooling over active drones -> value [E, N]
- FlatMLPAblationActorCritic: concatenated flat vector baseline for ablation comparison
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.distributions import Normal

from src.config import get


def scatter_logits_to_full(
    u_topk: torch.Tensor,
    threat_indices: torch.Tensor,
    threat_mask: torch.Tensor,
    M: int,
) -> torch.Tensor:
    """Scatter [E, N, K_t] coordination logits into the full [E, N, M] cost matrix.

    Args:
        u_topk: Tensor [E, N, K_t] float32 pointer scores
        threat_indices: Tensor [E, N, K_t] int64 indices of threats in [0, M-1]
        threat_mask: Tensor [E, N, K_t] bool valid visible threat mask
        M: Total number of missiles in the environment

    Returns:
        full_logits: Tensor [E, N, M] float32
    """
    E, N, K_t = u_topk.shape
    device = u_topk.device
    # Allocate with dummy column at index M to safely absorb invalid slots without overwriting index 0
    full_with_dummy = torch.zeros((E, N, M + 1), dtype=u_topk.dtype, device=device)

    valid = threat_mask & (threat_indices >= 0) & (threat_indices < M)
    safe_idx = torch.where(valid, threat_indices, torch.full_like(threat_indices, M))
    vals = torch.where(valid, u_topk, torch.zeros_like(u_topk))

    full_with_dummy.scatter_(dim=2, index=safe_idx, src=vals)
    return full_with_dummy[:, :, :M]


class CrossAttentionActorCritic(nn.Module):
    """Permutation-Invariant Cross-Attention Actor-Critic for Multi-Agent Swarm Defense."""

    def __init__(
        self,
        cfg: Optional[dict] = None,
        ego_dim: int = 10,
        threat_dim: int = 15,
        ally_dim: int = 9,
        d_model: int = 64,
        num_heads: int = 2,
        residual_accel_max_ms2: Optional[float] = None,
        logit_scale: float = 5.0,
    ):
        super().__init__()
        self.d_model = int(d_model if cfg is None else get(cfg, "training.d_model", d_model))
        self.num_heads = int(
            num_heads if cfg is None else get(cfg, "training.num_attention_heads", num_heads)
        )
        self.logit_scale = float(
            logit_scale if cfg is None else get(cfg, "training.logit_scale", logit_scale)
        )

        if residual_accel_max_ms2 is not None:
            self.a_res_max = float(residual_accel_max_ms2)
        elif cfg is not None:
            res_g = float(get(cfg, "training.residual_accel_max_g", 4.0))
            self.a_res_max = res_g * 9.80665
        else:
            self.a_res_max = 4.0 * 9.80665

        # 1. Ego Token Encoder: [B, 10] -> [B, 1, d_model]
        self.ego_encoder = nn.Sequential(
            nn.Linear(ego_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.SiLU(),
        )

        # 2. Shared Threat Token Encoder: [B, K_t, 15] -> [B, K_t, d_model]
        self.threat_encoder = nn.Sequential(
            nn.Linear(threat_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.SiLU(),
        )

        # 3. Shared Ally Token Encoder: [B, K_a, 9] -> [B, K_a, d_model]
        self.ally_encoder = nn.Sequential(
            nn.Linear(ally_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.SiLU(),
        )

        # 4. Multi-Head Cross-Attention over concatenated Threats + Allies
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            batch_first=True,
        )

        # 5. Fusion Trunk: Concat[Q_i, Attn_Out] (2 * d_model) -> h_i (d_model)
        self.fusion_trunk = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.SiLU(),
        )

        # 6. Residual Acceleration Actor Head: h_i -> mu (3), log_std (3)
        self.mu_head = nn.Linear(self.d_model, 3)
        self.log_std = nn.Parameter(torch.full((3,), -0.5, dtype=torch.float32))

        # 7. Coordination Logit Head: Cross-Pointer against E_threat
        self.W_ptr = nn.Linear(self.d_model, self.d_model, bias=False)

        # 8. Centralized Critic: Concat[h_team, h_i] (2 * d_model) -> value (1)
        self.critic = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, 1),
        )

    def _extract_and_encode(
        self, obs: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract tokens, run encoders, cross-attention with safe masking, and fusion trunk."""
        ego = obs["ego"]  # [..., N, 10]
        threat_tokens = obs["threat_tokens"]  # [..., N, K_t, 15]
        threat_mask = obs["threat_mask"]  # [..., N, K_t] bool
        ally_tokens = obs["ally_tokens"]  # [..., N, K_a, 9]
        ally_mask = obs["ally_mask"]  # [..., N, K_a] bool

        leading_shape = ego.shape[:-1]  # (E, N) or (B_env, N)
        B = int(math.prod(leading_shape))
        K_t = threat_tokens.shape[-2]
        K_a = ally_tokens.shape[-2]

        flat_ego = ego.reshape(B, -1)
        flat_threat = threat_tokens.reshape(B, K_t, -1)
        flat_threat_mask = threat_mask.reshape(B, K_t)
        flat_ally = ally_tokens.reshape(B, K_a, -1)
        flat_ally_mask = ally_mask.reshape(B, K_a)

        # Encode queries, threat keys, ally keys
        Q_i = self.ego_encoder(flat_ego).unsqueeze(1)  # [B, 1, d_model]
        E_threat = self.threat_encoder(flat_threat)  # [B, K_t, d_model]
        E_ally = self.ally_encoder(flat_ally)  # [B, K_a, d_model]

        # Concatenate keys/values and masks across threats and allies
        KV = torch.cat([E_threat, E_ally], dim=1)  # [B, K_t + K_a, d_model]
        valid_kv = torch.cat([flat_threat_mask, flat_ally_mask], dim=1)  # [B, K_t + K_a] bool

        # Safe all-masked handling: prevent NaN when a drone sees 0 neighbors
        has_any_valid = valid_kv.any(dim=-1)  # [B] bool
        safe_key_padding_mask = ~valid_kv  # True indicates masked / ignored slot
        # Force column 0 unmasked for rows that have zero valid keys
        safe_key_padding_mask = safe_key_padding_mask.clone()
        safe_key_padding_mask[~has_any_valid, 0] = False

        attn_out, _ = self.cross_attn(Q_i, KV, KV, key_padding_mask=safe_key_padding_mask)
        # Zero out rows with zero valid neighbors
        attn_out = torch.where(
            has_any_valid.unsqueeze(-1).unsqueeze(-1),
            attn_out,
            torch.zeros_like(attn_out),
        )

        # Fusion trunk
        concat_feat = torch.cat([Q_i.squeeze(1), attn_out.squeeze(1)], dim=-1)  # [B, 2 * d_model]
        h_i = self.fusion_trunk(concat_feat)  # [B, d_model]

        return h_i, E_threat, leading_shape, flat_threat_mask, K_t

    def forward(self, obs: dict) -> dict:
        """Forward pass computing continuous action distribution, pointer logits, and critic values."""
        h_i, E_threat, leading_shape, flat_threat_mask, K_t = self._extract_and_encode(obs)
        B = h_i.shape[0]

        # 1. Actor continuous residual acceleration distribution
        mu_flat = torch.tanh(self.mu_head(h_i)) * self.a_res_max  # [B, 3]
        std_flat = torch.exp(torch.clamp(self.log_std, -3.0, 1.0)).expand_as(mu_flat)  # [B, 3]

        # 2. Coordination logit head (Cross-pointer against E_threat)
        ptr_query = self.W_ptr(h_i).unsqueeze(-1)  # [B, d_model, 1]
        raw_scores = torch.bmm(E_threat, ptr_query).squeeze(-1) / math.sqrt(self.d_model)  # [B, K_t]
        u_topk_flat = self.logit_scale * torch.tanh(raw_scores)
        u_topk_flat = torch.where(flat_threat_mask, u_topk_flat, torch.zeros_like(u_topk_flat))

        # 3. Centralized Critic with permutation-invariant team mean-pooling
        # Reshape to [E_batch, N, d_model]
        E_batch, N = leading_shape
        h_team_2d = h_i.view(E_batch, N, self.d_model)

        drone_mask = obs.get("drone_mask", torch.ones(leading_shape, dtype=torch.bool, device=h_i.device))
        mask_f = drone_mask.float().unsqueeze(-1)  # [E_batch, N, 1]
        n_active = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        h_team = (h_team_2d * mask_f).sum(dim=1, keepdim=True) / n_active  # [E_batch, 1, d_model]
        h_team_expanded = h_team.expand(-1, N, -1)  # [E_batch, N, d_model]

        critic_in = torch.cat([h_team_expanded, h_team_2d], dim=-1).reshape(B, 2 * self.d_model)
        value_flat = self.critic(critic_in).squeeze(-1)  # [B]

        # Reshape outputs to original leading dimensions (E, N)
        mu = mu_flat.view(*leading_shape, 3)
        std = std_flat.view(*leading_shape, 3)
        u_topk = u_topk_flat.view(*leading_shape, K_t)
        value = value_flat.view(*leading_shape)

        return {
            "mu": mu,
            "std": std,
            "u_topk": u_topk,
            "value": value,
        }

    def get_action_and_value(
        self,
        obs: dict,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or evaluate actions under Normal(mu, std), returning action, log_prob, entropy, value, u_topk."""
        fwd = self.forward(obs)
        mu = fwd["mu"]
        std = fwd["std"]
        value = fwd["value"]
        u_topk = fwd["u_topk"]

        dist = Normal(mu, std)
        if action is None:
            action = mu if deterministic else dist.rsample()

        log_prob = dist.log_prob(action).sum(dim=-1)  # [E, N]
        entropy = dist.entropy().sum(dim=-1)  # [E, N]

        drone_mask = obs.get("drone_mask")
        if drone_mask is not None:
            log_prob = torch.where(drone_mask, log_prob, torch.zeros_like(log_prob))
            entropy = torch.where(drone_mask, entropy, torch.zeros_like(entropy))
            action = torch.where(drone_mask.unsqueeze(-1), action, torch.zeros_like(action))
            value = torch.where(drone_mask, value, torch.zeros_like(value))

        return action, log_prob, entropy, value, u_topk

    def get_value(self, obs: dict) -> torch.Tensor:
        """Compute state value estimates V_phi(s) [E, N]."""
        return self.forward(obs)["value"]

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FlatMLPAblationActorCritic(nn.Module):
    """Legacy Flat Concatenated MLP Actor-Critic baseline for ablation study (Spec Section 7.1)."""

    def __init__(
        self,
        cfg: Optional[dict] = None,
        K_threat: int = 15,
        K_ally: int = 10,
        ego_dim: int = 10,
        threat_dim: int = 15,
        ally_dim: int = 9,
        hidden_dim: int = 64,
        residual_accel_max_ms2: Optional[float] = None,
        logit_scale: float = 5.0,
    ):
        super().__init__()
        self.K_threat = K_threat
        self.K_ally = K_ally
        self.logit_scale = logit_scale
        input_dim = ego_dim + K_threat * threat_dim + K_ally * ally_dim  # 10 + 225 + 90 = 325

        if residual_accel_max_ms2 is not None:
            self.a_res_max = float(residual_accel_max_ms2)
        elif cfg is not None:
            res_g = float(get(cfg, "training.residual_accel_max_g", 4.0))
            self.a_res_max = res_g * 9.80665
        else:
            self.a_res_max = 4.0 * 9.80665

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
        )

        self.mu_head = nn.Linear(hidden_dim, 3)
        self.log_std = nn.Parameter(torch.full((3,), -0.5, dtype=torch.float32))
        self.logit_head = nn.Linear(hidden_dim, K_threat)

        self.critic = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.hidden_dim = hidden_dim

    def forward(self, obs: dict) -> dict:
        ego = obs["ego"]  # [E, N, 10]
        threats = obs["threat_tokens"]  # [E, N, K_t, 15]
        allies = obs["ally_tokens"]  # [E, N, K_a, 9]

        leading_shape = ego.shape[:-1]
        E_batch, N = leading_shape

        flat_threats = threats.reshape(E_batch, N, -1)
        flat_allies = allies.reshape(E_batch, N, -1)
        flat_input = torch.cat([ego, flat_threats, flat_allies], dim=-1)  # [E_batch, N, 325]

        h = self.mlp(flat_input)  # [E_batch, N, hidden_dim]
        mu = torch.tanh(self.mu_head(h)) * self.a_res_max
        std = torch.exp(torch.clamp(self.log_std, -3.0, 1.0)).expand_as(mu)

        raw_logits = self.logit_head(h)
        u_topk = self.logit_scale * torch.tanh(raw_logits)
        threat_mask = obs.get("threat_mask")
        if threat_mask is not None:
            u_topk = torch.where(threat_mask, u_topk, torch.zeros_like(u_topk))

        # Centralized critic with team pooling
        drone_mask = obs.get("drone_mask", torch.ones(leading_shape, dtype=torch.bool, device=h.device))
        mask_f = drone_mask.float().unsqueeze(-1)
        n_active = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        h_team = (h * mask_f).sum(dim=1, keepdim=True) / n_active
        h_team_exp = h_team.expand(-1, N, -1)

        critic_in = torch.cat([h_team_exp, h], dim=-1)
        value = self.critic(critic_in).squeeze(-1)

        return {"mu": mu, "std": std, "u_topk": u_topk, "value": value}

    def get_action_and_value(
        self,
        obs: dict,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fwd = self.forward(obs)
        mu = fwd["mu"]
        std = fwd["std"]
        value = fwd["value"]
        u_topk = fwd["u_topk"]

        dist = Normal(mu, std)
        if action is None:
            action = mu if deterministic else dist.rsample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        drone_mask = obs.get("drone_mask")
        if drone_mask is not None:
            log_prob = torch.where(drone_mask, log_prob, torch.zeros_like(log_prob))
            entropy = torch.where(drone_mask, entropy, torch.zeros_like(entropy))
            action = torch.where(drone_mask.unsqueeze(-1), action, torch.zeros_like(action))
            value = torch.where(drone_mask, value, torch.zeros_like(value))

        return action, log_prob, entropy, value, u_topk

    def get_value(self, obs: dict) -> torch.Tensor:
        return self.forward(obs)["value"]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
