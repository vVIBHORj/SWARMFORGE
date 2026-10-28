"""Classical coordination baseline controller combining task allocation and proportional navigation."""

from typing import Tuple
import torch

from src.baselines.auction_allocator import auction_assign
from src.baselines.greedy_allocator import greedy_assign
from src.baselines.hungarian_allocator import hungarian_assign
from src.config import get
from src.physics.guidance_laws import (
    build_cost_matrix,
    cbf_repulsion_accel,
    engagement_kinematics,
    tpn_guidance_accel,
)


class BaselineController:
    """Multi-agent baseline controller with interchangeable allocation and TPN guidance."""

    def __init__(
        self,
        cfg: dict,
        allocator: str = "hungarian",
        use_cbf: bool = True,
        use_apn: bool = False,
        use_estimated_state: bool = False,
    ):
        self.cfg = cfg
        self.allocator = allocator.lower()
        if self.allocator not in {"greedy", "hungarian", "auction"}:
            raise ValueError(f"Unsupported allocator: {allocator}. Must be 'greedy', 'hungarian', or 'auction'.")

        self.use_cbf = use_cbf
        self.use_apn = use_apn
        self.use_estimated_state = use_estimated_state

        self.nav_gain = float(get(cfg, "guidance_and_allocation.navigation_gain_N", 4.0))
        self.v_min = float(get(cfg, "guidance_and_allocation.v_min_closing", 20.0))
        self.tgo_cap = float(get(cfg, "guidance_and_allocation.tgo_cap", 120.0))
        self.k_cbf = float(get(cfg, "guidance_and_allocation.cbf_gain", 500.0))
        self.r_col = float(get(cfg, "entities.drones.collision_radius", 5.0))
        self.r_safe = float(get(cfg, "entities.drones.cbf_safe_radius", 40.0))
        self.invalid_cost = float(get(cfg, "guidance_and_allocation.invalid_cost", 1.0e6))
        self.auction_num_iters = int(get(cfg, "guidance_and_allocation.auction_num_iters", 10))
        self.auction_eps = float(get(cfg, "guidance_and_allocation.auction_eps", 0.05))

    @torch.no_grad()
    def act(self, env) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute coordination actions and assignment for the current env state."""
        valid_drone = env.drone_active & (env.drone_role == 0)
        if self.use_estimated_state and hasattr(env, "track_state") and env.track_state is not None:
            valid_m_init = env.track_state.get("initialized", torch.ones_like(env.missile_live()))
            valid_missile = env.missile_live() & valid_m_init
            m_pos = env.track_state.get("pos_est", env.missile_pos)
            m_vel = env.track_state.get("vel_est", env.missile_vel)
            m_acc = env.track_state.get("acc_est", None)
        else:
            valid_missile = env.missile_live()
            m_pos = env.missile_pos
            m_vel = env.missile_vel
            m_acc = None

        kin = engagement_kinematics(
            env.drone_pos,
            env.drone_vel,
            m_pos,
            m_vel,
            missile_accel=m_acc,
            v_min=self.v_min,
            tgo_cap=self.tgo_cap,
        )
        dist = kin["dist"]
        Vc = kin["Vc"]

        if self.allocator == "greedy":
            # True uncoordinated nearest-target baseline: cost is purely range dist
            C = dist.clone()
            valid_pair = valid_drone.unsqueeze(-1) & valid_missile.unsqueeze(1)
            C = torch.where(valid_pair, C, torch.full_like(C, self.invalid_cost))
        else:
            C, _ = build_cost_matrix(
                env,
                prev_z=env.z,
                v_min=self.v_min,
                tgo_cap=self.tgo_cap,
                invalid_cost=self.invalid_cost,
                use_estimated_state=self.use_estimated_state,
            )

        # Dynamic terminal endgame commitment lock:
        # Only hard-lock targets when inside terminal engagement (dist < 800m and Vc > 0)
        is_terminal = (dist < 800.0) & (Vc > 0.0)
        active_commit = env.z & valid_drone.unsqueeze(-1) & valid_missile.unsqueeze(1) & is_terminal
        free_drone = valid_drone & (~active_commit.any(dim=-1))
        assigned_per_m = active_commit.sum(dim=1)
        rem_K = torch.clamp(env.missile_K - assigned_per_m, min=0)
        unassigned_missile = valid_missile & (rem_K > 0)

        if self.allocator == "greedy":
            z_alloc = greedy_assign(C, free_drone, unassigned_missile, rem_K, invalid_cost=self.invalid_cost)
        elif self.allocator == "hungarian":
            z_alloc = hungarian_assign(C, free_drone, unassigned_missile, rem_K, invalid_cost=self.invalid_cost)
        elif self.allocator == "auction":
            z_alloc = auction_assign(
                C,
                free_drone,
                unassigned_missile,
                rem_K,
                num_iters=self.auction_num_iters,
                eps=self.auction_eps,
                invalid_cost=self.invalid_cost,
            )

        z = active_commit | z_alloc

        # Enforce strict single-missile ownership per drone: if assigned to multiple, keep minimum cost
        multi_assigned = z.sum(dim=-1) > 1  # [E, N]
        if multi_assigned.any():
            masked_c = torch.where(z, C, torch.tensor(float('inf'), device=env.device, dtype=C.dtype))
            best_m = torch.argmin(masked_c, dim=-1, keepdim=True)
            clean_z = torch.zeros_like(z)
            clean_z.scatter_(dim=-1, index=best_m, src=z.any(dim=-1, keepdim=True))
            z = clean_z

        env.z = z

        # Compute TPN guidance acceleration
        m_accel = m_acc if self.use_apn else None
        a_tpn = tpn_guidance_accel(
            env.drone_pos,
            env.drone_vel,
            m_pos,
            m_vel,
            z,
            missile_accel=m_accel,
            nav_gain=self.nav_gain,
            v_min=self.v_min,
        )

        if self.use_cbf:
            a_cbf = cbf_repulsion_accel(
                env.drone_pos,
                env.drone_active,
                r_col=self.r_col,
                r_safe=self.r_safe,
                k_cbf=self.k_cbf,
            )
            accel_cmd = a_tpn + a_cbf
        else:
            accel_cmd = a_tpn

        # Non-interceptors (scouts) do not steer aggressively into missile paths
        is_interceptor = (env.drone_role == 0).unsqueeze(-1)
        accel_cmd = torch.where(is_interceptor, accel_cmd, torch.zeros_like(accel_cmd))

        return accel_cmd, z
