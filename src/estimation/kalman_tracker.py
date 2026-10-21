"""Vectorized 3D Linear Kalman Filter and Cooperative Sensor Model for Multi-Target Tracking."""

from typing import Optional
import torch

from src.config import get


@torch.no_grad()
def compute_detections_and_measurements(
    env,
    rng: Optional[torch.Generator] = None,
    full_observability: bool = False,
) -> dict:
    """Compute pairwise drone-to-missile detections, fused noise std, and noisy position measurements.

    Args:
        env: VectorizedSwarmEnv instance with drone_pos, missile_pos, drone_role, drone_active, etc.
        rng: Optional torch.Generator for reproducible noise generation.
        full_observability: If True, all live missiles are detected regardless of range.

    Returns:
        dict containing:
            z_pos: [E, M, 3] noisy measured missile positions (or true positions if noise disabled)
            detected: [E, M] bool indicating whether each missile is observed by at least one active drone
            sigma_m: [E, M] float32 best (minimum) fused observation noise std per missile
            drone_detects: [E, N, M] bool indicating detection per (drone, missile) pair
    """
    drone_pos = env.drone_pos  # [E, N, 3]
    missile_pos = env.missile_pos  # [E, M, 3]

    # Pairwise distance [E, N, M]
    diff = missile_pos.unsqueeze(1) - drone_pos.unsqueeze(2)  # [E, N, M, 3]
    dist = torch.linalg.norm(diff, dim=-1)  # [E, N, M]

    interceptor_range = float(get(env.cfg, "entities.drones.interceptor_sensor_range", 4500.0))
    scout_range = float(get(env.cfg, "entities.drones.scout_sensor_range", 9000.0))
    sensor_range = torch.where(env.drone_role == 0, interceptor_range, scout_range).unsqueeze(-1)  # [E, N, 1]

    live_m = env.missile_live()  # [E, M]
    valid_detector = env.drone_active.unsqueeze(-1) & live_m.unsqueeze(1) & (dist <= sensor_range)  # [E, N, M]

    if full_observability:
        detected = live_m
    else:
        detected = valid_detector.any(dim=1)  # [E, M]

    # Measurement noise computation
    enable_noise = bool(get(env.cfg, "estimation.enable_sensor_noise", True))
    if hasattr(env, "sigma_obs") and env.sigma_obs is not None:
        sigma_base = float(env.sigma_obs)
    else:
        sigma_base = float(get(env.cfg, "estimation.position_noise_std_meters", 15.0)) if enable_noise else 0.0

    scout_mult = float(get(env.cfg, "estimation.scout_noise_multiplier", 0.4))
    range_noise_scale = float(get(env.cfg, "estimation.range_noise_scale", 0.5))

    role_mult = torch.where(env.drone_role == 0, 1.0, scout_mult).unsqueeze(-1)  # [E, N, 1]
    dist_scale = 1.0 + range_noise_scale * (dist / (sensor_range + 1e-6))
    sigma_ij = sigma_base * role_mult * dist_scale  # [E, N, M]

    # Mask invalid detectors with infinity
    sigma_ij = torch.where(valid_detector, sigma_ij, torch.full_like(sigma_ij, float("inf")))

    # Best team sensor fusion per missile
    sigma_m = sigma_ij.min(dim=1).values  # [E, M]
    # For undetected missiles, clamp to safe finite placeholder and mask with detected
    sigma_m = torch.where(detected, sigma_m, torch.full_like(sigma_m, 1e3))

    if enable_noise and sigma_base > 0.0:
        if rng is not None:
            if rng.device.type != env.device.type:
                noise_raw = torch.randn(missile_pos.shape, generator=rng, dtype=torch.float32).to(device=env.device)
            else:
                noise_raw = torch.randn(missile_pos.shape, generator=rng, device=env.device, dtype=torch.float32)
        else:
            noise_raw = torch.randn(missile_pos.shape, device=env.device, dtype=torch.float32)
        noise = noise_raw * sigma_m.unsqueeze(-1)
    else:
        noise = torch.zeros_like(missile_pos)

    z_pos = missile_pos + noise

    return {
        "z_pos": z_pos,
        "detected": detected,
        "sigma_m": sigma_m,
        "drone_detects": valid_detector,
    }


class VectorizedKalmanTracker:
    """Axis-decoupled 3D Constant-Acceleration Linear Kalman Filter for multi-target tracking.

    Tracks 9 kinematic states [pos, vel, acc] per target across 3 isotropic spatial axes.
    State representation:
        x_hat: [E, M, 3, 3] where dim 2 is (x, y, z) and dim 3 is (pos, vel, acc).
        P:     [E, M, 3, 3] shared covariance across the 3 isotropic spatial axes.
    """

    def __init__(self, cfg: dict, num_envs: int, num_missiles: int, device: torch.device | str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_missiles = num_missiles
        self.device = torch.device(device)

        self.dt = float(get(cfg, "environment.dt", 0.05))
        dt = self.dt

        # Constant 1D transition matrix F for [p, v, a]
        self.F = torch.tensor(
            [
                [1.0, dt, 0.5 * dt**2],
                [0.0, 1.0, dt],
                [0.0, 0.0, 1.0],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        self.FT = self.F.T.clone()

        # Discrete acceleration process noise covariance Q
        sigma_a = float(get(cfg, "estimation.kf_process_accel_std", 35.0))
        gamma = torch.tensor([[0.5 * dt**2], [dt], [1.0]], device=self.device, dtype=torch.float32)
        self.Q = (sigma_a**2) * (gamma @ gamma.T)  # [3, 3]

        self.init_pos_var = float(get(cfg, "estimation.kf_init_pos_var", 225.0))
        self.init_vel_var = float(get(cfg, "estimation.kf_init_vel_var", 10000.0))
        self.init_acc_var = float(get(cfg, "estimation.kf_init_acc_var", 2500.0))

        # Joseph form update helpers
        self.I = torch.eye(3, device=self.device, dtype=torch.float32)
        self.H_row = torch.tensor([[1.0, 0.0, 0.0]], device=self.device, dtype=torch.float32)

        # Allocate internal state tensors
        self.x_hat = torch.zeros((num_envs, num_missiles, 3, 3), device=self.device, dtype=torch.float32)
        self.P = torch.zeros((num_envs, num_missiles, 3, 3), device=self.device, dtype=torch.float32)
        self.initialized = torch.zeros((num_envs, num_missiles), device=self.device, dtype=torch.bool)
        self.steps_since_seen = torch.zeros((num_envs, num_missiles), device=self.device, dtype=torch.int64)

        self.prev_z_pos = torch.zeros((num_envs, num_missiles, 3), device=self.device, dtype=torch.float32)
        self.has_prev_z = torch.zeros((num_envs, num_missiles), device=self.device, dtype=torch.bool)
        self.raw_pos = torch.zeros((num_envs, num_missiles, 3), device=self.device, dtype=torch.float32)
        self.raw_vel = torch.zeros((num_envs, num_missiles, 3), device=self.device, dtype=torch.float32)
        self.detected = torch.zeros((num_envs, num_missiles), device=self.device, dtype=torch.bool)

    @torch.no_grad()
    def reset(self, env_mask: Optional[torch.Tensor] = None) -> None:
        """Reset state for all envs or only masked envs without host syncs."""
        if env_mask is None:
            self.x_hat.zero_()
            self.P.zero_()
            self.initialized.zero_()
            self.steps_since_seen.zero_()
            self.prev_z_pos.zero_()
            self.has_prev_z.zero_()
            self.raw_pos.zero_()
            self.raw_vel.zero_()
            self.detected.zero_()
        else:
            m2 = env_mask.unsqueeze(-1)  # [E, 1]
            m3 = m2.unsqueeze(-1)  # [E, 1, 1]
            m4 = m3.unsqueeze(-1)  # [E, 1, 1, 1]

            self.x_hat = torch.where(m4, torch.zeros_like(self.x_hat), self.x_hat)
            self.P = torch.where(m4, torch.zeros_like(self.P), self.P)
            self.initialized = torch.where(m2, torch.zeros_like(self.initialized), self.initialized)
            self.steps_since_seen = torch.where(m2, torch.zeros_like(self.steps_since_seen), self.steps_since_seen)
            self.prev_z_pos = torch.where(m3, torch.zeros_like(self.prev_z_pos), self.prev_z_pos)
            self.has_prev_z = torch.where(m2, torch.zeros_like(self.has_prev_z), self.has_prev_z)
            self.raw_pos = torch.where(m3, torch.zeros_like(self.raw_pos), self.raw_pos)
            self.raw_vel = torch.where(m3, torch.zeros_like(self.raw_vel), self.raw_vel)
            self.detected = torch.where(m2, torch.zeros_like(self.detected), self.detected)

    @torch.no_grad()
    def step(
        self,
        z_pos: torch.Tensor,
        detected: torch.Tensor,
        sigma_m: torch.Tensor,
        missile_live: torch.Tensor,
        env_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Execute one filter predict-update cycle across all targets and environments.

        Args:
            z_pos: [E, M, 3] noisy measured positions
            detected: [E, M] bool indicating whether measurement is valid
            sigma_m: [E, M] float32 measurement noise std
            missile_live: [E, M] bool indicating active/live missiles
            env_mask: Optional [E] bool indicating which environments to update (others remain frozen)

        Returns:
            dict containing filtered kinematic estimates and covariance metrics.
        """
        if env_mask is not None:
            prev_x_hat = self.x_hat.clone()
            prev_P = self.P.clone()
            prev_init = self.initialized.clone()
            prev_steps = self.steps_since_seen.clone()
            prev_raw_p = self.raw_pos.clone()
            prev_raw_v = self.raw_vel.clone()
            prev_prev_z = self.prev_z_pos.clone()
            prev_has_prev = self.has_prev_z.clone()
            prev_det = self.detected.clone()

        self.detected = detected & missile_live

        # Tracking status masks
        was_init = self.initialized & missile_live
        first_seen = (~self.initialized) & missile_live & detected
        coasting = was_init & (~detected)
        updating = was_init & detected
        dead = ~missile_live

        # --- Raw finite-difference baseline tracking ---
        diff_v = (z_pos - self.prev_z_pos) / self.dt
        valid_raw_v = self.has_prev_z & detected & missile_live
        curr_raw_vel = torch.where(valid_raw_v.unsqueeze(-1), diff_v, self.raw_vel)
        curr_raw_pos = torch.where((detected & missile_live).unsqueeze(-1), z_pos, self.raw_pos)

        new_has_prev = torch.where(
            dead,
            torch.zeros_like(self.has_prev_z),
            torch.where(detected, torch.ones_like(self.has_prev_z), self.has_prev_z),
        )
        new_prev_z = torch.where((detected & missile_live).unsqueeze(-1), z_pos, self.prev_z_pos)

        # --- First-seen state and covariance initialization ---
        pos_var_init = torch.clamp(sigma_m.square(), min=self.init_pos_var)
        P_init = torch.zeros_like(self.P)
        P_init[:, :, 0, 0] = pos_var_init
        P_init[:, :, 1, 1] = self.init_vel_var
        P_init[:, :, 2, 2] = self.init_acc_var

        x_init = torch.zeros_like(self.x_hat)
        x_init[:, :, :, 0] = z_pos

        # --- Predict step (vectorized over [E, M]) ---
        x_pred = torch.matmul(self.x_hat, self.FT)  # [E, M, 3, 3]
        P_pred = torch.matmul(torch.matmul(self.F, self.P), self.FT) + self.Q  # [E, M, 3, 3]

        # --- Update step (vectorized over [E, M]) ---
        # Innovation in position: z - x_pred_pos
        y = z_pos - x_pred[:, :, :, 0]  # [E, M, 3]
        R = sigma_m.square().clamp(min=1e-4)  # [E, M]
        S = P_pred[:, :, 0, 0] + R  # [E, M]

        # Kalman gain K: [E, M, 3]
        K = P_pred[:, :, :, 0] / S.unsqueeze(-1)  # [E, M, 3]

        # State update: x_upd = x_pred + y * K
        x_upd = x_pred + y.unsqueeze(-1) * K.unsqueeze(2)  # [E, M, 3, 3]

        # Joseph-form stabilized covariance update
        I_KH = self.I - K.unsqueeze(-1) * self.H_row  # [E, M, 3, 3]
        P_joseph_prior = torch.matmul(torch.matmul(I_KH, P_pred), I_KH.transpose(-1, -2))  # [E, M, 3, 3]
        KRKT = (K.unsqueeze(-1) @ K.unsqueeze(-2)) * R.unsqueeze(-1).unsqueeze(-1)  # [E, M, 3, 3]
        P_upd = P_joseph_prior + KRKT  # [E, M, 3, 3]

        # --- Combine states across categories ---
        m_first_seen_4d = first_seen.unsqueeze(-1).unsqueeze(-1)
        m_updating_4d = updating.unsqueeze(-1).unsqueeze(-1)
        m_coasting_4d = coasting.unsqueeze(-1).unsqueeze(-1)

        self.x_hat = torch.where(
            m_first_seen_4d,
            x_init,
            torch.where(m_updating_4d, x_upd, torch.where(m_coasting_4d, x_pred, torch.zeros_like(self.x_hat))),
        )
        self.P = torch.where(
            m_first_seen_4d,
            P_init,
            torch.where(m_updating_4d, P_upd, torch.where(m_coasting_4d, P_pred, torch.zeros_like(self.P))),
        )
        self.initialized = torch.where(dead, torch.zeros_like(self.initialized), self.initialized | first_seen)
        self.steps_since_seen = torch.where(
            dead | first_seen | updating,
            torch.zeros_like(self.steps_since_seen),
            self.steps_since_seen + 1,
        )

        self.raw_pos = torch.where(dead.unsqueeze(-1), torch.zeros_like(curr_raw_pos), curr_raw_pos)
        self.raw_vel = torch.where(dead.unsqueeze(-1), torch.zeros_like(curr_raw_vel), curr_raw_vel)
        self.prev_z_pos = new_prev_z
        self.has_prev_z = new_has_prev

        if env_mask is not None:
            m = env_mask.unsqueeze(-1)
            m3 = m.unsqueeze(-1)
            m4 = m3.unsqueeze(-1)
            self.x_hat = torch.where(m4, self.x_hat, prev_x_hat)
            self.P = torch.where(m4, self.P, prev_P)
            self.initialized = torch.where(m, self.initialized, prev_init)
            self.steps_since_seen = torch.where(m, self.steps_since_seen, prev_steps)
            self.raw_pos = torch.where(m3, self.raw_pos, prev_raw_p)
            self.raw_vel = torch.where(m3, self.raw_vel, prev_raw_v)
            self.prev_z_pos = torch.where(m3, self.prev_z_pos, prev_prev_z)
            self.has_prev_z = torch.where(m, self.has_prev_z, prev_has_prev)
            self.detected = torch.where(m, self.detected, prev_det)

        return self.get_estimates()

    def trace_P(self) -> torch.Tensor:
        """Total 3D trace of the 9-state covariance matrix (sum across x, y, z axes)."""
        return (self.P[:, :, 0, 0] + self.P[:, :, 1, 1] + self.P[:, :, 2, 2]) * 3.0

    def get_estimates(self) -> dict:
        """Return dict of current filtered kinematic estimates and covariance properties."""
        return {
            "pos_est": self.x_hat[:, :, :, 0],
            "vel_est": self.x_hat[:, :, :, 1],
            "acc_est": self.x_hat[:, :, :, 2],
            "trace_P": self.trace_P(),
            "pos_var": self.P[:, :, 0, 0],
            "vel_var": self.P[:, :, 1, 1],
            "acc_var": self.P[:, :, 2, 2],
            "initialized": self.initialized.clone(),
            "detected": self.detected.clone(),
            "steps_since_seen": self.steps_since_seen.clone(),
            "raw_pos": self.raw_pos.clone(),
            "raw_vel": self.raw_vel.clone(),
        }
