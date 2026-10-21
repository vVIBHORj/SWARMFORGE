"""Unit and integration tests for VectorizedKalmanTracker and cooperative sensor model."""

import copy
from pathlib import Path
import pytest
import torch

from src.config import load_config
from src.baselines.baseline_controller import BaselineController
from src.envs.vectorized_swarm_env import VectorizedSwarmEnv
from src.estimation.kalman_tracker import (
    VectorizedKalmanTracker,
    compute_detections_and_measurements,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default_sim.yaml"

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


def make_test_cfg(**overrides) -> dict:
    cfg = copy.deepcopy(load_config(CONFIG_PATH))
    for k, v in overrides.items():
        keys = k.split(".")
        d = cfg
        for sub_k in keys[:-1]:
            d = d.setdefault(sub_k, {})
        d[keys[-1]] = v
    return cfg


@pytest.mark.parametrize("device", DEVICES)
def test_m3_velocity_error_acceptance_criterion(device):
    """Milestone M3 Acceptance Criterion: Kalman vel RMSE < 0.35 * raw finite-diff vel RMSE.

    Simulate E=8, M=20 constant-velocity + weaving missiles over 200 steps (dt=0.05)
    with Gaussian position noise sigma_obs = 15.0 m.
    Compute RMS velocity estimation error over steps 40..200 for Kalman vel_est vs
    raw finite-difference raw_vel = (z_t - z_{t-1}) / dt.
    """
    E, M = 8, 20
    dt = 0.05
    steps = 200
    sigma_obs = 15.0

    cfg = make_test_cfg(
        **{
            "environment.dt": dt,
            "estimation.position_noise_std_meters": sigma_obs,
            "estimation.enable_sensor_noise": True,
            "estimation.kf_process_accel_std": 35.0,
        }
    )

    tracker = VectorizedKalmanTracker(cfg, num_envs=E, num_missiles=M, device=device)
    tracker.reset()

    # Create true trajectories: 10 constant velocity, 10 weaving per env
    # Initial positions and velocities
    rng = torch.Generator(device=device).manual_seed(42)
    p_0 = (torch.rand(E, M, 3, generator=rng, device=device) - 0.5) * 6000.0
    v_0 = torch.zeros(E, M, 3, device=device)
    # Velocity heading generally inward / non-zero
    v_0[..., 0] = -300.0 + torch.randn(E, M, generator=rng, device=device) * 20.0
    v_0[..., 1] = 50.0 + torch.randn(E, M, generator=rng, device=device) * 20.0
    v_0[..., 2] = -20.0 + torch.randn(E, M, generator=rng, device=device) * 10.0

    # Weaving parameters for second half of missiles
    is_weave = torch.zeros(E, M, 1, dtype=torch.bool, device=device)
    is_weave[:, M // 2 :, :] = True
    omega_w = 1.5
    amp_w = 4.0 * 9.81 / omega_w  # velocity amplitude: ~26.16 m/s
    phi = torch.rand(E, M, 1, generator=rng, device=device) * 6.28

    detected = torch.ones((E, M), dtype=torch.bool, device=device)
    missile_live = torch.ones((E, M), dtype=torch.bool, device=device)
    sigma_m = torch.full((E, M), sigma_obs, dtype=torch.float32, device=device)

    kf_errors = []
    raw_errors = []

    for step_idx in range(steps):
        t_sec = step_idx * dt
        # True kinematics
        p_true = p_0 + v_0 * t_sec
        v_true = v_0.clone()

        # Add sinusoidal weave in y-axis for weaving missiles
        weave_pos_offset = (amp_w / omega_w) * torch.sin(omega_w * t_sec + phi)
        weave_vel_offset = amp_w * torch.cos(omega_w * t_sec + phi)

        p_true[..., 1:2] = torch.where(is_weave, p_true[..., 1:2] + weave_pos_offset, p_true[..., 1:2])
        v_true[..., 1:2] = torch.where(is_weave, v_true[..., 1:2] + weave_vel_offset, v_true[..., 1:2])

        # Noisy position measurement
        noise = torch.randn(p_true.shape, generator=rng, device=device) * sigma_obs
        z_pos = p_true + noise

        est = tracker.step(z_pos, detected, sigma_m, missile_live)

        # Collect errors over steady-state steps 40..200
        if step_idx >= 40:
            kf_err = est["vel_est"] - v_true  # [E, M, 3]
            raw_err = est["raw_vel"] - v_true  # [E, M, 3]

            kf_errors.append(kf_err)
            raw_errors.append(raw_err)

    kf_err_tensor = torch.cat(kf_errors, dim=0)  # [N_samples, M, 3]
    raw_err_tensor = torch.cat(raw_errors, dim=0)

    kf_rmse = torch.sqrt(torch.mean(torch.sum(kf_err_tensor**2, dim=-1))).item()
    raw_rmse = torch.sqrt(torch.mean(torch.sum(raw_err_tensor**2, dim=-1))).item()
    ratio = kf_rmse / raw_rmse

    print(f"\n[Milestone M3] Kalman Vel RMSE: {kf_rmse:.2f} m/s | Raw Vel RMSE: {raw_rmse:.2f} m/s | Ratio: {ratio:.4f}")
    assert ratio < 0.35, f"Kalman velocity RMSE ratio {ratio:.4f} must be < 0.35 of raw velocity RMSE"


@pytest.mark.parametrize("device", DEVICES)
def test_scout_vs_interceptor_sensor_range_and_noise(device):
    """Place a missile at 6500m: only scout detects it. At 3000m: scout has lower noise."""
    cfg = make_test_cfg(
        **{
            "environment.num_parallel_envs": 1,
            "entities.drones.num_interceptors": 1,
            "entities.drones.num_scouts": 1,
            "entities.missiles.num_missiles": 1,
            "entities.drones.interceptor_sensor_range": 4500.0,
            "entities.drones.scout_sensor_range": 9000.0,
            "estimation.enable_sensor_noise": True,
            "estimation.position_noise_std_meters": 15.0,
            "estimation.scout_noise_multiplier": 0.4,
            "estimation.range_noise_scale": 0.5,
        }
    )
    env = VectorizedSwarmEnv(cfg, num_envs=1, device=device, seed=42)

    # Place interceptor (idx 0) and scout (idx 1) at origin
    env.drone_pos[0, 0] = torch.tensor([0.0, 0.0, 500.0], device=device)
    env.drone_role[0, 0] = 0  # interceptor
    env.drone_active[0, 0] = True

    env.drone_pos[0, 1] = torch.tensor([0.0, 0.0, 500.0], device=device)
    env.drone_role[0, 1] = 1  # scout
    env.drone_active[0, 1] = True

    # Case 1: Missile at 6500m along x
    env.missile_pos[0, 0] = torch.tensor([6500.0, 0.0, 500.0], device=device)
    env.missile_spawn_time[0, 0] = 0.0
    env.missile_status[0, 0] = 0

    meas1 = compute_detections_and_measurements(env)
    # Drone detects: [E, N, M] -> [1, 2, 1]
    interceptor_detects = meas1["drone_detects"][0, 0, 0].item()
    scout_detects = meas1["drone_detects"][0, 1, 0].item()
    team_detected = meas1["detected"][0, 0].item()

    assert not interceptor_detects, "Interceptor (range 4500m) should not detect missile at 6500m"
    assert scout_detects, "Scout (range 9000m) must detect missile at 6500m"
    assert team_detected, "Team detection must be True when scout detects missile"

    # Case 2: Missile at 3000m along x (both in range)
    env.missile_pos[0, 0] = torch.tensor([3000.0, 0.0, 500.0], device=device)
    meas2 = compute_detections_and_measurements(env)

    assert meas2["drone_detects"][0, 0, 0].item() is True
    assert meas2["drone_detects"][0, 1, 0].item() is True

    # Fused sigma_m should be determined by the scout's cleaner measurement (0.4 multiplier)
    sigma_fused = meas2["sigma_m"][0, 0].item()
    # Interceptor alone would have noise: 15.0 * 1.0 * (1 + 0.5 * (3000/4500)) = 20.0
    # Scout alone has noise: 15.0 * 0.4 * (1 + 0.5 * (3000/9000)) = 7.0
    assert sigma_fused < 10.0, f"Scout fusion should yield low noise std (<10m), got {sigma_fused:.2f}"


@pytest.mark.parametrize("device", DEVICES)
def test_coasting_and_covariance_growth(device):
    """Track missile for 30 steps (P shrinks), then coast for 20 steps (P grows monotonically)."""
    cfg = make_test_cfg(
        **{
            "environment.dt": 0.05,
            "estimation.kf_process_accel_std": 35.0,
            "estimation.kf_init_vel_var": 25000.0,
            "estimation.kf_init_acc_var": 20000.0,
        }
    )
    tracker = VectorizedKalmanTracker(cfg, num_envs=1, num_missiles=1, device=device)
    tracker.reset()

    # Missile moving at constant 300 m/s along x
    dt = 0.05
    v_true = torch.tensor([[[300.0, 0.0, 0.0]]], device=device)
    p_true = torch.tensor([[[0.0, 0.0, 1000.0]]], device=device)
    sigma_m = torch.tensor([[15.0]], device=device)
    live = torch.tensor([[True]], device=device)

    # 30 steps with detection
    det_true = torch.tensor([[True]], device=device)
    trace_history_det = []
    for _ in range(30):
        p_true = p_true + v_true * dt
        est = tracker.step(p_true, det_true, sigma_m, live)
        trace_history_det.append(est["trace_P"][0, 0].item())

    # Covariance should have converged and decreased significantly from initial
    assert trace_history_det[-1] < trace_history_det[0]

    # 20 steps coasting (detected=False)
    det_false = torch.tensor([[False]], device=device)
    trace_history_coast = [trace_history_det[-1]]
    pos_est_before_coast = est["pos_est"].clone()
    vel_est_before_coast = est["vel_est"].clone()
    acc_est_before_coast = est["acc_est"].clone()

    for step_c in range(20):
        est = tracker.step(p_true, det_false, sigma_m, live)
        trace_history_coast.append(est["trace_P"][0, 0].item())
        assert est["steps_since_seen"][0, 0].item() == step_c + 1

    # Monotonic covariance growth during coasting
    for k in range(len(trace_history_coast) - 1):
        assert trace_history_coast[k + 1] > trace_history_coast[k], (
            f"trace_P must monotonically increase during coasting: {trace_history_coast[k+1]} <= {trace_history_coast[k]}"
        )

    # Verify kinematic extrapolation along velocity and acceleration
    t_coast = 20 * dt
    expected_pos = (
        pos_est_before_coast
        + vel_est_before_coast * t_coast
        + 0.5 * acc_est_before_coast * (t_coast**2)
    )
    assert torch.allclose(est["pos_est"], expected_pos, atol=1e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_selective_reset_envs_tracker(device):
    """tracker.reset(env_mask) resets only masked env; unmasked env states remain bit-identical."""
    cfg = make_test_cfg(**{"environment.dt": 0.05})
    E, M = 4, 5
    tracker = VectorizedKalmanTracker(cfg, num_envs=E, num_missiles=M, device=device)
    tracker.reset()

    # Step 15 times with random inputs
    rng = torch.Generator(device=device).manual_seed(99)
    for _ in range(15):
        z_pos = torch.randn(E, M, 3, generator=rng, device=device) * 1000.0
        det = torch.ones((E, M), dtype=torch.bool, device=device)
        sig = torch.full((E, M), 15.0, device=device)
        live = torch.ones((E, M), dtype=torch.bool, device=device)
        tracker.step(z_pos, det, sig, live)

    # Snapshot states of envs [0, 2, 3]
    unmasked = [0, 2, 3]
    x_hat_snap = tracker.x_hat[unmasked].clone()
    P_snap = tracker.P[unmasked].clone()
    init_snap = tracker.initialized[unmasked].clone()
    steps_snap = tracker.steps_since_seen[unmasked].clone()

    # Reset only env 1
    mask = torch.tensor([False, True, False, False], device=device, dtype=torch.bool)
    tracker.reset(mask)

    # Env 1 should be completely reset
    assert (tracker.x_hat[1] == 0.0).all()
    assert (tracker.P[1] == 0.0).all()
    assert not tracker.initialized[1].any()
    assert (tracker.steps_since_seen[1] == 0).all()

    # Envs [0, 2, 3] must remain bit-identical
    assert torch.equal(tracker.x_hat[unmasked], x_hat_snap)
    assert torch.equal(tracker.P[unmasked], P_snap)
    assert torch.equal(tracker.initialized[unmasked], init_snap)
    assert torch.equal(tracker.steps_since_seen[unmasked], steps_snap)


@pytest.mark.parametrize("device", DEVICES)
def test_closed_loop_noisy_kalman_intercept(device):
    """Run closed-loop episode with sigma_obs=15.0 and BaselineController(use_estimated_state=True).

    Assert all tensors remain finite and intercept rate exceeds 0.70.
    """
    E = 4
    cfg = make_test_cfg(
        **{
            "environment.num_parallel_envs": E,
            "entities.drones.num_interceptors": 16,
            "entities.drones.num_scouts": 4,
            "entities.missiles.num_missiles": 16,
            "estimation.enable_sensor_noise": True,
            "estimation.position_noise_std_meters": 15.0,
            "environment.max_episode_steps": 350,
        }
    )
    env = VectorizedSwarmEnv(cfg, num_envs=E, device=device, seed=42)
    env.reset(sigma_obs=15.0)

    controller = BaselineController(cfg, allocator="hungarian", use_cbf=True, use_estimated_state=True)

    for _ in range(350):
        if env.done.all():
            break

        accel_cmd, _ = controller.act(env)

        assert torch.isfinite(accel_cmd).all(), "accel_cmd contains NaN or Inf"
        assert torch.isfinite(env.track_state["pos_est"]).all(), "pos_est contains NaN or Inf"
        assert torch.isfinite(env.track_state["vel_est"]).all(), "vel_est contains NaN or Inf"
        assert torch.isfinite(env.track_state["trace_P"]).all(), "trace_P contains NaN or Inf"

        env.step(accel_cmd)

    m = env.metrics()
    mean_int_rate = m["intercept_rate"].mean().item()
    print(f"\n[Closed-Loop Kalman Intercept] Mean Intercept Rate: {mean_int_rate:.2%}")
    assert mean_int_rate >= 0.70, f"Closed-loop Kalman intercept rate {mean_int_rate:.2%} must exceed 70%"
