"""Multi-component reward calculation with potential-based reward shaping.

Spec Section 6.6:
- Intercept & HVA Defense:
    - Assigned intercept bonus (+10.0)
    - Team intercept bonus (+5.0 per intercepted real threat)
    - HVA leaker penalty (-15.0 per real missile impact)
- Potential-based shaping:
    Phi_i(s_t) = -w_zem * (||ZEM_{i, j*}|| / L_max) - w_dist * (||r_{i, j*}|| / L_max)
    r_shaping = gamma * Phi_i(s_{t+1}) - Phi_i(s_t)
- Penalties:
    - Duplicate ownership violation (-2.0)
    - Friendly collision (-10.0)
    - Control effort penalty (-effort_weight * ||a_effort||^2 * dt)
"""

from typing import Optional, Tuple
import torch

from src.config import get
from src.physics.guidance_laws import engagement_kinematics
from src.physics.kinematics_3d import g_to_ms2


class RewardComputer:
    """Vectorized stateful/pure multi-component reward computer with potential-based shaping."""

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.intercept_assigned_bonus = float(
            get(cfg, "rewards.intercept_assigned_bonus", 10.0)
        )
        self.intercept_team_bonus = float(get(cfg, "rewards.intercept_team_bonus", 5.0))
        self.hva_leaker_penalty = float(get(cfg, "rewards.hva_leaker_penalty", 15.0))
        self.duplicate_penalty = float(get(cfg, "rewards.duplicate_penalty", 2.0))
        self.collision_penalty = float(get(cfg, "rewards.collision_penalty", 10.0))
        self.effort_weight = float(get(cfg, "rewards.effort_weight", 0.01))
        self.shaping_gamma = float(get(cfg, "rewards.shaping_gamma", 0.99))
        self.shaping_zem_weight = float(get(cfg, "rewards.shaping_zem_weight", 1.0))
        self.shaping_dist_weight = float(get(cfg, "rewards.shaping_dist_weight", 0.5))

    @torch.no_grad()
    def compute_potential(self, env) -> torch.Tensor:
        """Compute state potential Phi_i(s_t) [E, N] for each active interceptor."""
        E, N, M = env.num_envs, env.N, env.M
        L_max = float(max(env.world_bounds))

        is_interceptor = (env.drone_role == 0) & env.drone_active  # [E, N]
        has_target = env.z.any(dim=-1) & is_interceptor  # [E, N]

        target_idx = torch.argmax(env.z.float(), dim=-1)  # [E, N]
        target_idx_expanded = target_idx.unsqueeze(-1).expand(-1, -1, 3)

        p_m = torch.gather(env.missile_pos, dim=1, index=target_idx_expanded)  # [E, N, 3]
        v_m = torch.gather(env.missile_vel, dim=1, index=target_idx_expanded)  # [E, N, 3]

        r = p_m - env.drone_pos  # [E, N, 3]
        v_rel = v_m - env.drone_vel  # [E, N, 3]

        dist = torch.linalg.norm(r, dim=-1)  # [E, N]
        Vc = -(r * v_rel).sum(dim=-1) / (dist + 1e-6)  # [E, N]

        t_go = torch.clamp(dist / torch.clamp(Vc, min=20.0), max=120.0)  # [E, N]
        ZEM = r + v_rel * t_go.unsqueeze(-1)  # [E, N, 3]
        zem_norm = torch.linalg.norm(ZEM, dim=-1)  # [E, N]

        live_m = env.missile_live()  # [E, M]
        target_live = torch.gather(live_m, dim=1, index=target_idx)  # [E, N]
        valid_eval = has_target & target_live

        phi = -self.shaping_zem_weight * (zem_norm / L_max) - self.shaping_dist_weight * (dist / L_max)
        phi = torch.where(valid_eval, phi, torch.zeros_like(phi))
        return phi

    @torch.no_grad()
    def compute_step_reward(
        self,
        env,
        prev_phi: torch.Tensor,
        prev_z: Optional[torch.Tensor],
        info: dict,
        rl_residual_accel: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """Compute multi-component step reward and return (total_reward, components, new_phi).

        Args:
            env: VectorizedSwarmEnv
            prev_phi: Tensor [E, N] potential from previous step
            prev_z: Tensor [E, N, M] bool assignment before this step
            info: dict returned by env.step()
            rl_residual_accel: Optional Tensor [E, N, 3] RL continuous residual acceleration

        Returns:
            total_reward: Tensor [E, N] float32
            components: dict of individual reward terms [E, N]
            new_phi: Tensor [E, N] float32 potential at new state
        """
        E, N, M = env.num_envs, env.N, env.M
        new_phi = self.compute_potential(env)

        # Envs actively running during this step
        running = (~env.done) | info.get("newly_done", torch.zeros_like(env.done))

        # -------------------------------------------------------------
        # 1. Intercept & HVA Defense
        # -------------------------------------------------------------
        real_hit = info["intercept_event"] & (~env.missile_is_decoy)  # [E, M]
        interceptor_of_m = info["interceptor_of_missile"]  # [E, M]

        # Intercept bonus (+10.0) for assigned/intercepting drone
        drone_indices = torch.arange(N, device=env.device).unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
        hit_by_drone = (interceptor_of_m.unsqueeze(1) == drone_indices) & real_hit.unsqueeze(1)  # [E, N, M]

        if prev_z is not None:
            assigned_to_hit = prev_z & real_hit.unsqueeze(1)  # [E, N, M]
            got_assigned_hit = (hit_by_drone | assigned_to_hit).any(dim=-1)  # [E, N]
        else:
            got_assigned_hit = hit_by_drone.any(dim=-1)

        r_assigned_intercept = torch.where(
            got_assigned_hit & env.drone_active,
            torch.full_like(new_phi, self.intercept_assigned_bonus),
            torch.zeros_like(new_phi),
        )

        # Team intercept bonus (+5.0 per intercepted threat in env e)
        team_hits = real_hit.sum(dim=-1, keepdim=True).float()  # [E, 1]
        r_team_intercept = torch.where(
            env.drone_active,
            team_hits * self.intercept_team_bonus,
            torch.zeros_like(new_phi),
        )

        # HVA leaker penalty (-15.0 per real missile impacting HVA in env e)
        real_impact = info["impact_event"] & (~env.missile_is_decoy)  # [E, M]
        team_impacts = real_impact.sum(dim=-1, keepdim=True).float()  # [E, 1]
        r_hva_leaker = torch.where(
            env.drone_active,
            -team_impacts * self.hva_leaker_penalty,
            torch.zeros_like(new_phi),
        )

        # -------------------------------------------------------------
        # 2. Potential-Based Reward Shaping (Spec Eq. 6.6)
        # -------------------------------------------------------------
        if prev_z is not None:
            prev_has_target = prev_z.any(dim=-1)  # [E, N]
            curr_has_target = env.z.any(dim=-1)  # [E, N]
            prev_target = torch.argmax(prev_z.float(), dim=-1)  # [E, N]
            curr_target = torch.argmax(env.z.float(), dim=-1)  # [E, N]

            same_target = prev_has_target & curr_has_target & (prev_target == curr_target)
            target_resolved = torch.gather(real_hit | real_impact, dim=1, index=curr_target)

            # Only shape when drone keeps the same active target and target wasn't just resolved
            shaping_valid = same_target & (~target_resolved) & env.drone_active & (env.drone_role == 0)
        else:
            shaping_valid = torch.zeros_like(new_phi, dtype=torch.bool)

        r_shaping = torch.where(
            shaping_valid,
            self.shaping_gamma * new_phi - prev_phi,
            torch.zeros_like(new_phi),
        )

        # -------------------------------------------------------------
        # 3. Penalties
        # -------------------------------------------------------------
        # Duplicate ownership violation (-2.0)
        owners_per_missile = env.z.sum(dim=1)  # [E, M]
        dup_missile = owners_per_missile > env.missile_K  # [E, M]
        drone_in_dup = (env.z & dup_missile.unsqueeze(1)).any(dim=-1)  # [E, N]
        r_duplicate = torch.where(
            drone_in_dup & env.drone_active,
            torch.full_like(new_phi, -self.duplicate_penalty),
            torch.zeros_like(new_phi),
        )

        # Friendly collision (-10.0)
        collision_mask = info["collision_mask"]  # [E, N, N]
        had_collision = collision_mask.any(dim=-1)  # [E, N]
        r_collision = torch.where(
            had_collision & env.drone_active,
            torch.full_like(new_phi, -self.collision_penalty),
            torch.zeros_like(new_phi),
        )

        # Control effort penalty
        if rl_residual_accel is not None:
            effort_sq = (rl_residual_accel ** 2).sum(dim=-1)  # [E, N]
        else:
            max_a_ms2 = g_to_ms2(float(get(env.cfg, "entities.drones.max_accel_g", 12.0)))
            applied_a = info["drone_a_applied"]  # [E, N, 3]
            norm_a = applied_a / max_a_ms2
            effort_sq = (norm_a ** 2).sum(dim=-1)  # [E, N]

        r_effort = torch.where(
            env.drone_active,
            -self.effort_weight * effort_sq * env.dt,
            torch.zeros_like(new_phi),
        )

        # -------------------------------------------------------------
        # 4. Total and Env Done Masking
        # -------------------------------------------------------------
        total_reward = (
            r_assigned_intercept
            + r_team_intercept
            + r_hva_leaker
            + r_shaping
            + r_duplicate
            + r_collision
            + r_effort
        )

        # Zero out rewards for envs that were not running during this step
        run_mask = running.unsqueeze(-1)
        total_reward = torch.where(run_mask, total_reward, torch.zeros_like(total_reward))

        components = {
            "assigned_intercept": torch.where(run_mask, r_assigned_intercept, torch.zeros_like(r_assigned_intercept)),
            "team_intercept": torch.where(run_mask, r_team_intercept, torch.zeros_like(r_team_intercept)),
            "hva_leaker": torch.where(run_mask, r_hva_leaker, torch.zeros_like(r_hva_leaker)),
            "shaping": torch.where(run_mask, r_shaping, torch.zeros_like(r_shaping)),
            "duplicate": torch.where(run_mask, r_duplicate, torch.zeros_like(r_duplicate)),
            "collision": torch.where(run_mask, r_collision, torch.zeros_like(r_collision)),
            "effort": torch.where(run_mask, r_effort, torch.zeros_like(r_effort)),
        }

        return total_reward, components, new_phi
