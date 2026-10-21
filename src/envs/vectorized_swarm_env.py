"""Vectorized 3D simulation environment for multi-agent swarm defense.

Conventions:
- Coordinates: World x,y in [-W/2, +W/2], z in [0, H] (altitude in meters).
- Roles: 0 = interceptor, 1 = scout.
- Missile Types: 0 = ballistic, 1 = dive, 2 = weave.
- Missile Status: 0 = in flight/unresolved, 1 = intercepted, 2 = impacted HVA, 3 = expired.
- Event Order per step:
    1. Missile guidance and kinematics (active = missile_live).
    2. Drone kinematics with per-role limits (scouts have reduced speed/accel).
    3. Boundary enforcement and 2D mode constraints.
    4. CPA drone-missile evaluation (interceptors only).
    5. CPA missile-HVA evaluation (interception takes precedence over impact).
    6. Out-of-bounds missile expiration.
    7. Missile status resolution and optional interceptor expenditure.
    8. Friendly drone collision detection and counting.
    9. Delta-v accumulation and episode termination checks.
"""

from typing import Optional
import torch

from src.config import get
from src.envs.observations import build_observations
from src.envs.rewards import RewardComputer
from src.envs.scenario_generator import generate_scenario
from src.envs.threat_models import missile_accel_cmd
from src.estimation.kalman_tracker import (
    VectorizedKalmanTracker,
    compute_detections_and_measurements,
)
from src.physics.cpa_collisions import friendly_collision_mask, pairwise_cpa
from src.physics.kinematics_3d import g_to_ms2, step_kinematics


class VectorizedSwarmEnv:
    """Vectorized simulation environment coordinating N drones against M missiles across E envs."""

    def __init__(self, cfg: dict, num_envs: Optional[int] = None, device: Optional[str | torch.device] = None, seed: Optional[int] = None):
        self.cfg = cfg
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_envs = int(num_envs if num_envs is not None else get(cfg, "environment.num_parallel_envs", 4))
        self.seed = int(seed if seed is not None else get(cfg, "environment.seed", 42))

        self.num_interceptors = int(get(cfg, "entities.drones.num_interceptors", 16))
        self.num_scouts = int(get(cfg, "entities.drones.num_scouts", 4))
        self.num_missiles = int(get(cfg, "entities.missiles.num_missiles", 30))
        self.M = self.num_missiles
        self.N = self.num_interceptors + self.num_scouts
        self.mode = get(cfg, "scenario.mode", "3d")
        self.weave_enabled = get(cfg, "scenario.weave_enabled", True)

        self.enable_sensor_noise = bool(get(cfg, "estimation.enable_sensor_noise", True))
        pos_noise = float(get(cfg, "estimation.position_noise_std_meters", 15.0))
        self.sigma_obs = pos_noise if self.enable_sensor_noise else 0.0

        self.dt = float(get(cfg, "environment.dt", 0.05))
        self.max_episode_steps = int(get(cfg, "environment.max_episode_steps", 400))
        self.interceptor_expended = bool(get(cfg, "environment.interceptor_expended_on_intercept", False))
        self.expire_margin = float(get(cfg, "environment.expire_margin_m", 1000.0))
        self.intercept_radius = float(get(cfg, "entities.drones.intercept_radius", 25.0))
        self.collision_radius = float(get(cfg, "entities.drones.collision_radius", 5.0))
        self.hva_impact_radius = float(get(cfg, "scenario.hva_impact_radius", 50.0))

        self.world_bounds = [float(v) for v in get(cfg, "environment.world_bounds_meters", [10000.0, 10000.0, 3000.0])]

        self._rng = torch.Generator(device="cpu").manual_seed(self.seed)
        self.reward_computer = RewardComputer(self.cfg, self.device)
        self.reset()

    def reset(self, seed: Optional[int] = None, num_interceptors: Optional[int] = None, num_scouts: Optional[int] = None,
              num_missiles: Optional[int] = None, mode: Optional[str] = None, weave_enabled: Optional[bool] = None,
              sigma_obs: Optional[float] = None) -> None:
        if seed is not None:
            self.seed = seed
            self._rng.manual_seed(seed)
        if num_interceptors is not None: self.num_interceptors = num_interceptors
        if num_scouts is not None: self.num_scouts = num_scouts
        if num_missiles is not None: self.num_missiles = num_missiles
        self.M = self.num_missiles
        self.N = self.num_interceptors + self.num_scouts
        if mode is not None: self.mode = mode
        if weave_enabled is not None: self.weave_enabled = weave_enabled
        if sigma_obs is not None: self.sigma_obs = sigma_obs

        sub_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self._rng).item())
        scen = generate_scenario(self.cfg, self.num_envs, self.num_interceptors, self.num_scouts, self.num_missiles,
                                 self.device, sub_seed, self.mode, self.weave_enabled, self.sigma_obs)

        E, N, M = self.num_envs, self.num_interceptors + self.num_scouts, self.num_missiles
        self.t = torch.zeros(E, dtype=torch.int64, device=self.device)
        self.done = torch.zeros(E, dtype=torch.bool, device=self.device)
        self.truncated = torch.zeros(E, dtype=torch.bool, device=self.device)

        self.hva_pos = scen["hva_pos"]
        self.drone_pos = scen["drone_pos"]
        self.drone_vel = scen["drone_vel"]
        self.drone_role = scen["drone_role"]
        self.drone_active = scen["drone_active"]

        self.missile_pos = scen["missile_pos"]
        self.missile_vel = scen["missile_vel"]
        self.missile_type = scen["missile_type"]
        self.missile_target_hva = scen["missile_target_hva"]
        self.missile_K = scen["missile_K"]
        self.missile_is_decoy = scen["missile_is_decoy"]
        self.missile_weave_phase = scen["missile_weave_phase"]
        self.missile_spawn_time = scen["missile_spawn_time"]
        self.missile_exists = scen["missile_active"]
        self.missile_status = torch.zeros((E, M), dtype=torch.int64, device=self.device)

        self.z = torch.zeros((E, N, M), dtype=torch.bool, device=self.device)
        self.delta_v = torch.zeros(E, dtype=torch.float32, device=self.device)
        self.friendly_collisions = torch.zeros(E, dtype=torch.float32, device=self.device)

        self.tracker = VectorizedKalmanTracker(self.cfg, self.num_envs, self.M, self.device)
        self.update_tracker()
        self._prev_phi = self.reward_computer.compute_potential(self)

    def reset_envs(self, env_mask: torch.Tensor) -> None:
        sub_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self._rng).item())
        scen = generate_scenario(self.cfg, self.num_envs, self.num_interceptors, self.num_scouts, self.num_missiles,
                                 self.device, sub_seed, self.mode, self.weave_enabled, self.sigma_obs)
        m1, m2, m3 = env_mask, env_mask.unsqueeze(-1), env_mask.unsqueeze(-1).unsqueeze(-1)

        self.t = torch.where(m1, torch.zeros_like(self.t), self.t)
        self.done = torch.where(m1, torch.zeros_like(self.done), self.done)
        self.truncated = torch.where(m1, torch.zeros_like(self.truncated), self.truncated)
        self.delta_v = torch.where(m1, torch.zeros_like(self.delta_v), self.delta_v)
        self.friendly_collisions = torch.where(m1, torch.zeros_like(self.friendly_collisions), self.friendly_collisions)

        self.hva_pos = torch.where(m3, scen["hva_pos"], self.hva_pos)
        self.drone_pos = torch.where(m3, scen["drone_pos"], self.drone_pos)
        self.drone_vel = torch.where(m3, scen["drone_vel"], self.drone_vel)
        self.drone_role = torch.where(m2, scen["drone_role"], self.drone_role)
        self.drone_active = torch.where(m2, scen["drone_active"], self.drone_active)

        self.missile_pos = torch.where(m3, scen["missile_pos"], self.missile_pos)
        self.missile_vel = torch.where(m3, scen["missile_vel"], self.missile_vel)
        self.missile_type = torch.where(m2, scen["missile_type"], self.missile_type)
        self.missile_target_hva = torch.where(m2, scen["missile_target_hva"], self.missile_target_hva)
        self.missile_K = torch.where(m2, scen["missile_K"], self.missile_K)
        self.missile_is_decoy = torch.where(m2, scen["missile_is_decoy"], self.missile_is_decoy)
        self.missile_weave_phase = torch.where(m2, scen["missile_weave_phase"], self.missile_weave_phase)
        self.missile_spawn_time = torch.where(m2, scen["missile_spawn_time"], self.missile_spawn_time)
        self.missile_exists = torch.where(m2, scen["missile_active"], self.missile_exists)
        self.missile_status = torch.where(m2, torch.zeros_like(self.missile_status), self.missile_status)
        self.z = torch.where(m3, torch.zeros_like(self.z), self.z)

        self.tracker.reset(env_mask)
        self.update_tracker(env_mask=env_mask)
        new_phi = self.reward_computer.compute_potential(self)
        self._prev_phi = torch.where(env_mask.unsqueeze(-1), new_phi, self._prev_phi)

    def missile_live(self) -> torch.Tensor:
        time_sec = self.t.unsqueeze(-1).float() * self.dt
        return self.missile_exists & (self.missile_status == 0) & (time_sec >= self.missile_spawn_time) & (~self.done.unsqueeze(-1))

    def step(self, drone_accel_cmd: torch.Tensor) -> dict:
        running = ~self.done
        live = self.missile_live()
        t_sec = (self.t.float() * self.dt).unsqueeze(-1)

        # 1. Missile guidance and kinematics
        m_cmd = missile_accel_cmd(self.missile_pos, self.missile_vel, self.hva_pos, self.missile_target_hva,
                                  self.missile_type, self.missile_weave_phase, t_sec, self.cfg, mode=self.mode)
        a_max_m = g_to_ms2(float(get(self.cfg, "entities.missiles.max_accel_g", 8.0)))
        v_max_m = float(get(self.cfg, "entities.missiles.max_speed", 450.0))
        drag_m = float(get(self.cfg, "entities.missiles.linear_drag", 0.0))
        p_m_new, v_m_new, _ = step_kinematics(self.missile_pos, self.missile_vel, m_cmd, self.dt, a_max_m, v_max_m, drag_m, active=live)

        # Unspawned missiles retain their initial launch velocity
        unspawned = self.missile_exists & (self.missile_status == 0) & (t_sec < self.missile_spawn_time)
        v_m_new = torch.where(unspawned.unsqueeze(-1), self.missile_vel, v_m_new)

        # 2. Drone kinematics
        is_scout = (self.drone_role == 1).unsqueeze(-1)
        base_a_d = g_to_ms2(float(get(self.cfg, "entities.drones.max_accel_g", 12.0)))
        base_v_d = float(get(self.cfg, "entities.drones.max_speed", 350.0))
        scout_a_f = float(get(self.cfg, "entities.drones.scout_accel_fraction", 0.6))
        scout_v_f = float(get(self.cfg, "entities.drones.scout_speed_fraction", 0.6))
        drag_d = float(get(self.cfg, "entities.drones.linear_drag", 0.015))

        a_max_d = torch.where(is_scout, base_a_d * scout_a_f, base_a_d)
        v_max_d = torch.where(is_scout, base_v_d * scout_v_f, base_v_d)
        d_active = self.drone_active & running.unsqueeze(-1)
        p_d_new, v_d_new, a_d_applied = step_kinematics(self.drone_pos, self.drone_vel, drone_accel_cmd, self.dt, a_max_d, v_max_d, drag_d, active=d_active)

        # 3. 2D mode and world box boundary clamping
        W, L, H = self.world_bounds
        if self.mode == "2d":
            p_d_new[..., 2] = 0.0; v_d_new[..., 2] = 0.0
            p_m_new[..., 2] = 0.0; v_m_new[..., 2] = 0.0

        vx, vy, vz = v_d_new[..., 0], v_d_new[..., 1], v_d_new[..., 2]
        px, py, pz = p_d_new[..., 0], p_d_new[..., 1], p_d_new[..., 2]
        vx = torch.where((px >= W / 2.0) & (vx > 0), torch.zeros_like(vx), vx)
        vx = torch.where((px <= -W / 2.0) & (vx < 0), torch.zeros_like(vx), vx)
        vy = torch.where((py >= L / 2.0) & (vy > 0), torch.zeros_like(vy), vy)
        vy = torch.where((py <= -L / 2.0) & (vy < 0), torch.zeros_like(vy), vy)
        vz = torch.where((pz >= H) & (vz > 0), torch.zeros_like(vz), vz)
        vz = torch.where((pz <= 0.0) & (vz < 0), torch.zeros_like(vz), vz)
        v_d_new = torch.stack([vx, vy, vz], dim=-1)
        p_d_new = torch.stack([torch.clamp(px, -W / 2.0, W / 2.0), torch.clamp(py, -L / 2.0, L / 2.0), torch.clamp(pz, 0.0, H)], dim=-1)

        # 4. Drone-missile CPA interceptions
        d_min, _ = pairwise_cpa(self.drone_pos, p_d_new, self.missile_pos, p_m_new)
        can_hit = (d_min <= self.intercept_radius) & d_active.unsqueeze(-1) & (self.drone_role == 0).unsqueeze(-1) & live.unsqueeze(1)
        m_hit = can_hit.any(dim=1)
        d_min_masked = torch.where(can_hit, d_min, torch.tensor(float('inf'), device=self.device, dtype=d_min.dtype))
        best_drone = torch.argmin(d_min_masked, dim=1)
        interceptor_of_m = torch.where(m_hit, best_drone, torch.tensor(-1, device=self.device, dtype=torch.int64))

        # 5. Missile-HVA impact (interception takes precedence over impact)
        d_min_hva, _ = pairwise_cpa(self.missile_pos, p_m_new, self.hva_pos, self.hva_pos)
        m_impact = (d_min_hva.min(dim=-1).values <= self.hva_impact_radius) & live & (~m_hit)

        # 6. Missile expiration
        mx, my, mz = p_m_new[..., 0], p_m_new[..., 1], p_m_new[..., 2]
        oob = (mz < -10.0) | (torch.abs(mx) > (W / 2.0 + self.expire_margin)) | (torch.abs(my) > (L / 2.0 + self.expire_margin)) | (mz > (H + self.expire_margin))
        m_expired = live & (~m_hit) & (~m_impact) & oob

        # 7. Update missile status & expended drones
        run_m = running.unsqueeze(-1)
        ev_intercept = m_hit & run_m
        ev_impact = m_impact & run_m
        ev_expire = m_expired & run_m

        self.missile_status = torch.where(ev_intercept, torch.tensor(1, device=self.device, dtype=torch.int64), self.missile_status)
        self.missile_status = torch.where(ev_impact, torch.tensor(2, device=self.device, dtype=torch.int64), self.missile_status)
        self.missile_status = torch.where(ev_expire, torch.tensor(3, device=self.device, dtype=torch.int64), self.missile_status)

        if self.interceptor_expended:
            valid_exp = ev_intercept & (interceptor_of_m >= 0)
            exp_drones = torch.zeros_like(self.drone_active)
            safe_idx = torch.where(valid_exp, interceptor_of_m, torch.zeros_like(interceptor_of_m))
            exp_drones.scatter_(dim=1, index=safe_idx, src=valid_exp)
            self.drone_active = self.drone_active & (~exp_drones)

        # 8. Friendly collisions
        f_mask = friendly_collision_mask(self.drone_pos, p_d_new, self.collision_radius, active_A=d_active)
        col_count = f_mask.sum(dim=(1, 2)) // 2
        self.friendly_collisions += torch.where(running, col_count.float(), torch.zeros_like(self.friendly_collisions))

        # 9. Delta-v accumulation
        norm_a = torch.linalg.norm(a_d_applied, dim=-1)
        step_dv = (norm_a * self.dt).sum(dim=1)
        self.delta_v += torch.where(running, step_dv, torch.zeros_like(self.delta_v))

        # Update running states
        m3 = running.unsqueeze(-1).unsqueeze(-1)
        self.drone_pos = torch.where(m3, p_d_new, self.drone_pos)
        self.drone_vel = torch.where(m3, v_d_new, self.drone_vel)
        self.missile_pos = torch.where(m3, p_m_new, self.missile_pos)
        self.missile_vel = torch.where(m3, v_m_new, self.missile_vel)

        # 10. Termination
        self.t = torch.where(running, self.t + 1, self.t)
        all_resolved = ~(self.missile_exists & (self.missile_status == 0)).any(dim=-1)
        time_out = self.t >= self.max_episode_steps
        done_new = (all_resolved | time_out) & running
        self.truncated = self.truncated | (time_out & (~all_resolved) & running)
        self.done = self.done | done_new

        # 11. Estimation / tracker update for active envs
        self.update_tracker(env_mask=running)

        return {
            "intercept_event": ev_intercept,
            "interceptor_of_missile": torch.where(run_m, interceptor_of_m, torch.tensor(-1, device=self.device, dtype=torch.int64)),
            "impact_event": ev_impact,
            "expire_event": ev_expire,
            "collision_mask": f_mask,
            "collision_count": col_count,
            "d_min": d_min,
            "drone_a_applied": a_d_applied,
            "newly_done": done_new,
        }

    def metrics(self) -> dict:
        real_mask = self.missile_exists & (~self.missile_is_decoy)
        n_real = real_mask.sum(dim=-1).float()
        n_real_safe = torch.clamp(n_real, min=1.0)

        int_real = (real_mask & (self.missile_status == 1)).sum(dim=-1).float()
        imp_real = (real_mask & (self.missile_status == 2)).sum(dim=-1).float()
        exp_real = (real_mask & (self.missile_status == 3)).sum(dim=-1).float()

        int_rate = torch.where(n_real > 0, int_real / n_real_safe, torch.zeros_like(n_real))
        leak_rate = torch.where(n_real > 0, imp_real / n_real_safe, torch.zeros_like(n_real))
        exp_rate = torch.where(n_real > 0, exp_real / n_real_safe, torch.zeros_like(n_real))

        decoy_ints = (self.missile_exists & self.missile_is_decoy & (self.missile_status == 1)).sum(dim=-1).float()
        int_total = (self.missile_exists & (self.missile_status == 1)).sum(dim=-1).float()
        dv_per_int = self.delta_v / torch.clamp(int_total, min=1.0)

        return {
            "n_real": n_real,
            "intercept_rate": int_rate,
            "leaker_rate": leak_rate,
            "expired_rate": exp_rate,
            "decoy_intercepts": decoy_ints,
            "friendly_collisions": self.friendly_collisions.clone(),
            "delta_v_total": self.delta_v.clone(),
            "delta_v_per_intercept": dv_per_int,
            "steps": self.t.float(),
        }

    def update_tracker(self, env_mask: Optional[torch.Tensor] = None) -> dict:
        """Compute detections and step the multi-target Kalman tracker."""
        meas = compute_detections_and_measurements(self, rng=self._rng)
        track_dict = self.tracker.step(
            z_pos=meas["z_pos"],
            detected=meas["detected"],
            sigma_m=meas["sigma_m"],
            missile_live=self.missile_live(),
            env_mask=env_mask,
        )
        if env_mask is not None and hasattr(self, "track_state"):
            m = env_mask.unsqueeze(-1)
            m_drone = env_mask.unsqueeze(-1).unsqueeze(-1)
            track_dict["sigma_m"] = torch.where(m, meas["sigma_m"], self.track_state.get("sigma_m", meas["sigma_m"]))
            track_dict["drone_detects"] = torch.where(m_drone, meas["drone_detects"], self.track_state.get("drone_detects", meas["drone_detects"]))
        else:
            track_dict["sigma_m"] = meas["sigma_m"]
            track_dict["drone_detects"] = meas["drone_detects"]

        self.track_state = track_dict
        return track_dict

    def get_observations(
        self,
        K_threat: Optional[int] = None,
        K_ally: Optional[int] = None,
        use_estimated_state: bool = True,
    ) -> dict:
        """Build permutation-invariant set observations."""
        return build_observations(
            self,
            K_threat=K_threat,
            K_ally=K_ally,
            use_estimated_state=use_estimated_state,
        )

    def compute_rewards(
        self,
        info: dict,
        prev_z: Optional[torch.Tensor] = None,
        rl_residual_accel: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute multi-component rewards with potential-based shaping."""
        rewards, comps, new_phi = self.reward_computer.compute_step_reward(
            self,
            self._prev_phi,
            prev_z=prev_z,
            info=info,
            rl_residual_accel=rl_residual_accel,
        )
        self._prev_phi = new_phi
        return rewards, comps
