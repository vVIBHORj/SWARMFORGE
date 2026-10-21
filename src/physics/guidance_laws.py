"""3D Proportional Navigation, engagement geometry, and Control Barrier Function (CBF) repulsion."""

from typing import Optional, Tuple
import torch

from src.config import get

EPS = 1e-6


@torch.no_grad()
def engagement_kinematics(
    drone_pos: torch.Tensor,
    drone_vel: torch.Tensor,
    missile_pos: torch.Tensor,
    missile_vel: torch.Tensor,
    missile_accel: Optional[torch.Tensor] = None,
    v_min: float = 20.0,
    tgo_cap: float = 120.0,
) -> dict:
    """Compute pairwise 3D engagement kinematics between drones and missiles.

    drone_pos, drone_vel: [E, N, 3]
    missile_pos, missile_vel: [E, M, 3]
    missile_accel: optional [E, M, 3]

    Returns dict with keys:
      r: [E, N, M, 3] relative position (p_m - p_d)
      v_rel: [E, N, M, 3] relative velocity (v_m - v_d)
      dist: [E, N, M] range ||r||
      Vc: [E, N, M] closing speed -d(||r||)/dt
      t_go: [E, N, M] estimated time-to-go clamped to [0, tgo_cap]
      ZEM: [E, N, M, 3] zero-effort miss vector
      zem_norm: [E, N, M] ||ZEM||
      omega: [E, N, M, 3] line-of-sight angular velocity vector
    """
    # Relative vectors: [E, N, M, 3]
    r = missile_pos.unsqueeze(1) - drone_pos.unsqueeze(2)
    v_rel = missile_vel.unsqueeze(1) - drone_vel.unsqueeze(2)

    dist = torch.linalg.norm(r, dim=-1)  # [E, N, M]
    Vc = -(r * v_rel).sum(dim=-1) / (dist + EPS)  # [E, N, M]

    Vc_clamped = torch.clamp(Vc, min=v_min)
    t_go = torch.clamp(dist / Vc_clamped, max=tgo_cap)  # [E, N, M]

    t_go_3d = t_go.unsqueeze(-1)
    if missile_accel is not None:
        a_m = missile_accel.unsqueeze(1)
    else:
        a_m = torch.zeros_like(r)

    ZEM = r + v_rel * t_go_3d + 0.5 * a_m * (t_go_3d ** 2)  # [E, N, M, 3]
    zem_norm = torch.linalg.norm(ZEM, dim=-1)  # [E, N, M]

    dist_sq = (dist.unsqueeze(-1)).square() + EPS
    omega = torch.linalg.cross(r, v_rel, dim=-1) / dist_sq  # [E, N, M, 3]

    return {
        "r": r,
        "v_rel": v_rel,
        "dist": dist,
        "Vc": Vc,
        "t_go": t_go,
        "ZEM": ZEM,
        "zem_norm": zem_norm,
        "omega": omega,
    }


@torch.no_grad()
def tpn_guidance_accel(
    drone_pos: torch.Tensor,
    drone_vel: torch.Tensor,
    missile_pos: torch.Tensor,
    missile_vel: torch.Tensor,
    assignment_z: torch.Tensor,
    missile_accel: Optional[torch.Tensor] = None,
    nav_gain: float = 4.0,
    v_min: float = 20.0,
) -> torch.Tensor:
    """Compute 3D True/Augmented Proportional Navigation acceleration command for assigned drones.

    drone_pos, drone_vel: [E, N, 3]
    missile_pos, missile_vel: [E, M, 3]
    assignment_z: [E, N, M] bool
    missile_accel: optional [E, M, 3]

    Returns: accel_cmd [E, N, 3] (unassigned drones receive [0, 0, 0]).
    """
    kin = engagement_kinematics(drone_pos, drone_vel, missile_pos, missile_vel, missile_accel=missile_accel, v_min=v_min)
    r = kin["r"]
    dist = kin["dist"]
    Vc = kin["Vc"]
    omega = kin["omega"]

    if missile_accel is not None:
        a_m = missile_accel.unsqueeze(1)
    else:
        a_m = torch.zeros_like(r)

    r_hat = r / (dist.unsqueeze(-1) + EPS)
    cross_term = torch.linalg.cross(omega, r, dim=-1) / (dist.unsqueeze(-1) + EPS)
    a_lat = nav_gain * Vc.unsqueeze(-1).clamp(min=v_min) * cross_term + 0.5 * nav_gain * a_m  # [E, N, M, 3]
    a_lat_sq = (a_lat ** 2).sum(dim=-1, keepdim=True)
    scale = torch.clamp((dist.unsqueeze(-1) - 10.0) / 50.0, min=0.0, max=1.0)
    a_fwd_mag = torch.sqrt(torch.clamp(117.6 ** 2 - a_lat_sq, min=0.0)) * scale
    a_pair = a_lat + a_fwd_mag * r_hat  # [E, N, M, 3]

    # Select assigned target acceleration per drone
    z_f = assignment_z.unsqueeze(-1).to(a_pair.dtype)
    a_drone = (a_pair * z_f).sum(dim=2)  # [E, N, 3]
    return a_drone


@torch.no_grad()
def cbf_repulsion_accel(
    drone_pos: torch.Tensor,
    drone_active: torch.Tensor,
    r_col: float = 5.0,
    r_safe: float = 40.0,
    k_cbf: float = 500.0,
) -> torch.Tensor:
    """Compute Control Barrier Function repulsion acceleration between active drones.

    drone_pos: [E, N, 3]
    drone_active: [E, N] bool

    Returns: a_cbf [E, N, 3]
    """
    N = drone_pos.shape[1]
    diff = drone_pos.unsqueeze(2) - drone_pos.unsqueeze(1)  # p_i - p_k: [E, N, N, 3]
    d_ik = torch.linalg.norm(diff, dim=-1)  # [E, N, N]

    diag = torch.eye(N, dtype=torch.bool, device=drone_pos.device).unsqueeze(0)
    active_pair = drone_active.unsqueeze(-1) & drone_active.unsqueeze(1)
    valid_pair = (~diag) & active_pair & (d_ik < r_safe)

    d_clamped = torch.clamp(d_ik, min=2.0 * r_col + 0.1)
    coeff = k_cbf * (1.0 / (d_clamped - 2.0 * r_col) - 1.0 / (r_safe - 2.0 * r_col)).clamp(min=0.0)
    coeff = torch.where(valid_pair, coeff, torch.zeros_like(coeff))

    a_cbf = (coeff.unsqueeze(-1) * diff / (d_clamped.unsqueeze(-1) ** 3 + EPS)).sum(dim=2)  # [E, N, 3]
    a_cbf = torch.where(drone_active.unsqueeze(-1), a_cbf, torch.zeros_like(a_cbf))
    return a_cbf


@torch.no_grad()
def build_cost_matrix(
    env,
    prev_z: Optional[torch.Tensor] = None,
    rl_logits: Optional[torch.Tensor] = None,
    weights: Optional[dict] = None,
    v_min: float = 20.0,
    tgo_cap: float = 120.0,
    invalid_cost: float = 1.0e6,
    use_estimated_state: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build hybrid assignment cost matrix C [E, N, M] and valid pair mask [E, N, M]."""
    if weights is None:
        weights = get(
            env.cfg,
            "guidance_and_allocation.cost_weights",
            {"w_tgo": 1.0, "w_zem": 0.02, "w_urgency": 50.0, "w_hysteresis": 2.5, "w_logit": 1.0},
        )
    w_tgo = float(weights.get("w_tgo", 1.0))
    w_zem = float(weights.get("w_zem", 0.02))
    w_urgency = float(weights.get("w_urgency", 50.0))
    w_hysteresis = float(weights.get("w_hysteresis", 2.5))
    w_logit = float(weights.get("w_logit", 1.0))

    if use_estimated_state and hasattr(env, "track_state") and env.track_state is not None:
        m_pos = env.track_state.get("pos_est", env.missile_pos)
        m_vel = env.track_state.get("vel_est", env.missile_vel)
        m_acc = env.track_state.get("acc_est", None)
        valid_m_init = env.track_state.get("initialized", torch.ones_like(env.missile_live()))
        valid_missile = env.missile_live() & valid_m_init
    else:
        m_pos = env.missile_pos
        m_vel = env.missile_vel
        m_acc = None
        valid_missile = env.missile_live()

    kin = engagement_kinematics(env.drone_pos, env.drone_vel, m_pos, m_vel, missile_accel=m_acc, v_min=v_min, tgo_cap=tgo_cap)
    t_go = kin["t_go"]
    zem_norm = kin["zem_norm"]

    # Compute HVA threat urgency [E, M]
    r_jp = env.hva_pos.unsqueeze(1) - m_pos.unsqueeze(2)  # [E, M, P, 3]
    v_rel_jp = -m_vel.unsqueeze(2)  # [E, M, P, 3]
    d_jp = torch.linalg.norm(r_jp, dim=-1)  # [E, M, P]
    Vc_jp = -(r_jp * v_rel_jp).sum(dim=-1) / (d_jp + EPS)  # [E, M, P]
    tgo_hva = (d_jp / torch.clamp(Vc_jp, min=v_min)).min(dim=-1).values  # [E, M]
    urgency = 1.0 / (tgo_hva + 1.0)  # [E, M]

    prev_z_f = prev_z.float() if prev_z is not None else 0.0
    rl_logits_f = rl_logits.float() if rl_logits is not None else 0.0

    C = w_tgo * t_go + w_zem * zem_norm - w_urgency * urgency.unsqueeze(1) - w_hysteresis * prev_z_f - w_logit * rl_logits_f

    valid_drone = env.drone_active & (env.drone_role == 0)  # active interceptors only
    valid_pair = valid_drone.unsqueeze(-1) & valid_missile.unsqueeze(1)

    C = torch.where(valid_pair, C, torch.full_like(C, invalid_cost))
    return C, valid_pair
