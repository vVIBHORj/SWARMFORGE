"""Permutation-invariant set observations builder for multi-agent swarm defense.

Spec Section 4.2:
- Ego Token: [E, N, 10] float32
- Top-K Nearest Threat Tokens: [E, N, K_threat, 15] float32, threat_mask [E, N, K_threat] bool
- Top-K Nearest Ally Tokens: [E, N, K_ally, 9] float32, ally_mask [E, N, K_ally] bool
- Global Centralized State: [E, D_global] float32
- Drone Mask: [E, N] bool
"""

from typing import Optional
import torch
import torch.nn.functional as F

from src.config import get
from src.physics.guidance_laws import engagement_kinematics

EPS = 1e-6


@torch.no_grad()
def build_observations(
    env,
    K_threat: Optional[int] = None,
    K_ally: Optional[int] = None,
    use_estimated_state: bool = True,
) -> dict:
    """Build fully vectorized permutation-invariant set observations with zero host syncs.

    Args:
        env: VectorizedSwarmEnv instance
        K_threat: Number of nearest threat tokens to observe (default from cfg: 15)
        K_ally: Number of nearest ally tokens to observe (default from cfg: 10)
        use_estimated_state: If True and sigma_obs > 0, use Kalman-estimated missile tracks

    Returns:
        dict containing:
            ego: [E, N, 10] float32
            threat_tokens: [E, N, K_threat, 15] float32
            threat_mask: [E, N, K_threat] bool
            threat_indices: [E, N, K_threat] int64
            ally_tokens: [E, N, K_ally, 9] float32
            ally_mask: [E, N, K_ally] bool
            ally_indices: [E, N, K_ally] int64
            drone_mask: [E, N] bool
            global_state: [E, D_global] float32
    """
    E = env.num_envs
    N = env.N
    M = env.M
    device = env.device

    if K_threat is None:
        K_threat = int(get(env.cfg, "environment.max_neighbors_observed", 15))
    if K_ally is None:
        K_ally = int(get(env.cfg, "environment.max_allies_observed", 10))

    L_max = float(max(env.world_bounds))
    v_max = float(
        max(
            get(env.cfg, "entities.drones.max_speed", 350.0),
            get(env.cfg, "entities.missiles.max_speed", 450.0),
        )
    )
    T_max = float(env.max_episode_steps * env.dt)

    # -------------------------------------------------------------
    # 1. Ego Token [E, N, 10]
    # -------------------------------------------------------------
    p_norm = env.drone_pos / L_max  # [E, N, 3]
    v_norm = env.drone_vel / v_max  # [E, N, 3]
    role_i = env.drone_role.float().unsqueeze(-1)  # [E, N, 1]
    active_i = env.drone_active.float().unsqueeze(-1)  # [E, N, 1]
    has_assign_i = env.z.any(dim=-1).float().unsqueeze(-1)  # [E, N, 1]
    t_norm = (env.t.float() / float(env.max_episode_steps)).unsqueeze(-1).unsqueeze(-1).expand(E, N, 1)  # [E, N, 1]

    ego = torch.cat([p_norm, v_norm, role_i, active_i, has_assign_i, t_norm], dim=-1)  # [E, N, 10]
    ego = ego * active_i  # Zero out inactive drones

    # -------------------------------------------------------------
    # Threat State Selection (Estimated vs Ground Truth)
    # -------------------------------------------------------------
    if (
        use_estimated_state
        and env.sigma_obs > 0
        and hasattr(env, "track_state")
        and env.track_state is not None
    ):
        m_pos = env.track_state.get("pos_est", env.missile_pos)
        m_vel = env.track_state.get("vel_est", env.missile_vel)
        m_acc = env.track_state.get("acc_est", None)
        trace_P = env.track_state.get("trace_P", torch.zeros((E, M), device=device))
        valid_m_init = env.track_state.get("initialized", torch.ones_like(env.missile_live()))
        m_vis = env.missile_live() & valid_m_init
    else:
        m_pos = env.missile_pos
        m_vel = env.missile_vel
        m_acc = None
        trace_P = torch.zeros((E, M), device=device, dtype=torch.float32)
        m_vis = env.missile_live()

    # -------------------------------------------------------------
    # 2. Top-K Nearest Threat Tokens [E, N, K_threat, 15]
    # -------------------------------------------------------------
    kin = engagement_kinematics(
        env.drone_pos,
        env.drone_vel,
        m_pos,
        m_vel,
        missile_accel=m_acc,
        v_min=20.0,
        tgo_cap=T_max,
    )

    # HVA threat urgency [E, M]
    r_jp = env.hva_pos.unsqueeze(1) - m_pos.unsqueeze(2)  # [E, M, P, 3]
    v_rel_jp = -m_vel.unsqueeze(2)  # [E, M, P, 3]
    d_jp = torch.linalg.norm(r_jp, dim=-1)  # [E, M, P]
    Vc_jp = -(r_jp * v_rel_jp).sum(dim=-1) / (d_jp + EPS)  # [E, M, P]
    tgo_hva = (d_jp / torch.clamp(Vc_jp, min=20.0)).min(dim=-1).values  # [E, M]
    urgency = 1.0 / (tgo_hva + 1.0)  # [E, M]

    if env.sigma_obs > 0:
        tr_P = torch.log1p(trace_P) / 10.0
    else:
        tr_P = torch.zeros_like(trace_P)

    # 15 features across all M threats
    feat_r = kin["r"] / L_max  # [E, N, M, 3]
    feat_vrel = kin["v_rel"] / (2.0 * v_max)  # [E, N, M, 3]
    feat_vc = (kin["Vc"] / (2.0 * v_max)).unsqueeze(-1)  # [E, N, M, 1]
    feat_tgo = (kin["t_go"] / T_max).unsqueeze(-1)  # [E, N, M, 1]
    feat_zem = kin["ZEM"] / L_max  # [E, N, M, 3]
    feat_zij = env.z.float().unsqueeze(-1)  # [E, N, M, 1]
    other_assigned = (
        (env.z.float().sum(dim=1, keepdim=True) - env.z.float())
        .clamp(min=0.0)
        .unsqueeze(-1)
    )  # [E, N, M, 1]
    feat_urgency = urgency.unsqueeze(1).expand(E, N, M).unsqueeze(-1)  # [E, N, M, 1]
    feat_tr_P = tr_P.unsqueeze(1).expand(E, N, M).unsqueeze(-1)  # [E, N, M, 1]

    all_threat_features = torch.cat(
        [
            feat_r,
            feat_vrel,
            feat_vc,
            feat_tgo,
            feat_zem,
            feat_zij,
            other_assigned,
            feat_urgency,
            feat_tr_P,
        ],
        dim=-1,
    )  # [E, N, M, 15]

    # Rank visible threats by Euclidean distance
    dist = kin["dist"]  # [E, N, M]
    m_vis_exp = m_vis.unsqueeze(1).expand(E, N, M)
    masked_dist = torch.where(
        m_vis_exp,
        dist,
        torch.tensor(float("inf"), device=device, dtype=dist.dtype),
    )

    k_threat_select = min(K_threat, M)
    if k_threat_select > 0:
        topk_threat_dist, topk_threat_idx = torch.topk(
            masked_dist, k=k_threat_select, dim=-1, largest=False
        )
        valid_threat = torch.isfinite(topk_threat_dist) & (topk_threat_dist < 1e8)

        idx_exp = topk_threat_idx.unsqueeze(-1).expand(-1, -1, -1, 15)
        selected_threat_tokens = torch.gather(all_threat_features, dim=2, index=idx_exp)

        if k_threat_select < K_threat:
            pad_threat = K_threat - k_threat_select
            threat_tokens = F.pad(selected_threat_tokens, (0, 0, 0, pad_threat), value=0.0)
            threat_mask = F.pad(valid_threat, (0, pad_threat), value=False)
            threat_indices = F.pad(topk_threat_idx, (0, pad_threat), value=-1)
        else:
            threat_tokens = selected_threat_tokens
            threat_mask = valid_threat
            threat_indices = topk_threat_idx
    else:
        threat_tokens = torch.zeros((E, N, K_threat, 15), device=device, dtype=torch.float32)
        threat_mask = torch.zeros((E, N, K_threat), device=device, dtype=torch.bool)
        threat_indices = torch.full((E, N, K_threat), -1, device=device, dtype=torch.int64)

    # Inactive drones or masked slots must be zeroed out
    threat_mask = threat_mask & env.drone_active.unsqueeze(-1)
    threat_tokens = threat_tokens * threat_mask.unsqueeze(-1).float()
    threat_indices = torch.where(
        threat_mask, threat_indices, torch.tensor(-1, device=device, dtype=torch.int64)
    )

    # -------------------------------------------------------------
    # 3. Top-K Nearest Ally Tokens [E, N, K_ally, 9]
    # -------------------------------------------------------------
    # Pairwise relative position and velocity: [E, N, N, 3]
    # Index [e, i, k] represents (p_k - p_i)
    rel_p = env.drone_pos.unsqueeze(1) - env.drone_pos.unsqueeze(2)  # [E, N, N, 3]
    rel_v = env.drone_vel.unsqueeze(1) - env.drone_vel.unsqueeze(2)  # [E, N, N, 3]
    d_ik = torch.linalg.norm(rel_p, dim=-1)  # [E, N, N]

    # Target embedding for ally k
    k_has_target = env.z.any(dim=-1)  # [E, N] bool
    target_idx = torch.argmax(env.z.float(), dim=-1)  # [E, N] int64
    idx_3d = target_idx.unsqueeze(-1).expand(-1, -1, 3)  # [E, N, 3]
    p_target_k = torch.gather(m_pos, dim=1, index=idx_3d)  # [E, N, 3]
    p_target_k = torch.where(
        k_has_target.unsqueeze(-1), p_target_k, torch.zeros_like(p_target_k)
    )

    # Relative target vector (p_target_of_k - p_i) / L_max: [E, N, N, 3]
    rel_target = (p_target_k.unsqueeze(1) - env.drone_pos.unsqueeze(2)) / L_max
    rel_target = torch.where(
        k_has_target.unsqueeze(1).unsqueeze(-1), rel_target, torch.zeros_like(rel_target)
    )

    feat_rel_p = rel_p / L_max  # [E, N, N, 3]
    feat_rel_v = rel_v / v_max  # [E, N, N, 3]
    all_ally_features = torch.cat([feat_rel_p, feat_rel_v, rel_target], dim=-1)  # [E, N, N, 9]

    # Exclude self (diagonal) and inactive allies
    eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)  # [1, N, N]
    valid_ally_pair = (~eye) & env.drone_active.unsqueeze(1)  # [E, N, N]
    masked_ally_dist = torch.where(
        valid_ally_pair,
        d_ik,
        torch.tensor(float("inf"), device=device, dtype=d_ik.dtype),
    )

    k_ally_select = min(K_ally, max(0, N - 1))
    if k_ally_select > 0:
        topk_ally_dist, topk_ally_idx = torch.topk(
            masked_ally_dist, k=k_ally_select, dim=-1, largest=False
        )
        valid_ally = torch.isfinite(topk_ally_dist) & (topk_ally_dist < 1e8)

        idx_exp_ally = topk_ally_idx.unsqueeze(-1).expand(-1, -1, -1, 9)
        selected_ally_tokens = torch.gather(all_ally_features, dim=2, index=idx_exp_ally)

        if k_ally_select < K_ally:
            pad_ally = K_ally - k_ally_select
            ally_tokens = F.pad(selected_ally_tokens, (0, 0, 0, pad_ally), value=0.0)
            ally_mask = F.pad(valid_ally, (0, pad_ally), value=False)
            ally_indices = F.pad(topk_ally_idx, (0, pad_ally), value=-1)
        else:
            ally_tokens = selected_ally_tokens
            ally_mask = valid_ally
            ally_indices = topk_ally_idx
    else:
        ally_tokens = torch.zeros((E, N, K_ally, 9), device=device, dtype=torch.float32)
        ally_mask = torch.zeros((E, N, K_ally), device=device, dtype=torch.bool)
        ally_indices = torch.full((E, N, K_ally), -1, device=device, dtype=torch.int64)

    # Inactive drones or masked slots must be zeroed out
    ally_mask = ally_mask & env.drone_active.unsqueeze(-1)
    ally_tokens = ally_tokens * ally_mask.unsqueeze(-1).float()
    ally_indices = torch.where(
        ally_mask, ally_indices, torch.tensor(-1, device=device, dtype=torch.int64)
    )

    # -------------------------------------------------------------
    # 4. Centralized Global State [E, D_global]
    # -------------------------------------------------------------
    d_act_f = env.drone_active.float().unsqueeze(-1)  # [E, N, 1]
    n_act = d_act_f.sum(dim=1).clamp(min=1.0)
    mean_ego = (ego * d_act_f).sum(dim=1) / n_act  # [E, 10]

    live_m_f = env.missile_live().float().unsqueeze(-1)  # [E, M, 1]
    n_live_m = live_m_f.sum(dim=1).clamp(min=1.0)
    mean_m_pos = (m_pos * live_m_f).sum(dim=1) / n_live_m / L_max  # [E, 3]
    mean_m_vel = (m_vel * live_m_f).sum(dim=1) / n_live_m / v_max  # [E, 3]

    frac_active = env.drone_active.float().mean(dim=-1, keepdim=True)  # [E, 1]
    frac_live = env.missile_live().float().mean(dim=-1, keepdim=True)  # [E, 1]
    step_frac = (env.t.float() / float(env.max_episode_steps)).unsqueeze(-1)  # [E, 1]

    global_state = torch.cat(
        [mean_ego, mean_m_pos, mean_m_vel, frac_active, frac_live, step_frac], dim=-1
    )  # [E, 19]

    return {
        "ego": ego,
        "threat_tokens": threat_tokens,
        "threat_mask": threat_mask,
        "threat_indices": threat_indices,
        "ally_tokens": ally_tokens,
        "ally_mask": ally_mask,
        "ally_indices": ally_indices,
        "drone_mask": env.drone_active.clone(),
        "global_state": global_state,
    }
