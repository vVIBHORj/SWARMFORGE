"""Unit and integration tests for Evaluation Metrics, Multi-Seed Benchmarks, and Ablations.

Spec Section 10 & 12:
- Exact duplicate ownership and decoy waste accounting in EpisodeMetricsTracker.
- Verification of ablation flags (CBF shield, Kalman filter vs raw measurements, RL residual/logits).
- Benchmark suite execution in quick mode and verification of all generated artifacts.
"""

import os
from pathlib import Path
import pytest
import torch

from src.config import load_config
from src.envs.vectorized_swarm_env import VectorizedSwarmEnv
from src.evaluation.metrics import EpisodeMetricsTracker, evaluate_policy_episodes
from src.evaluation.run_ablations import run_benchmark_suite


def test_episode_metrics_tracker_exact_accounting():
    """Verify EpisodeMetricsTracker exact accounting on controlled E=2 scenario."""
    cfg = load_config("configs/default_sim.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = VectorizedSwarmEnv(cfg, num_envs=2, device=device, seed=42)

    tracker = EpisodeMetricsTracker(num_envs=2, device=device)

    # Configure controlled state:
    # Env 0:
    # - Real missiles only, all intercepted (status=1)
    # - Non-conflicting assignments: drone 0 to missile 0, drone 1 to missile 1
    # - No decoy assignments
    env.missile_is_decoy[0, :] = False
    env.missile_status[0, :] = 1  # 100% intercepted
    env.z[0, :, :] = False
    env.z[0, 0, 0] = True
    env.z[0, 1, 1] = True

    # Env 1:
    # - Missile 0 is real (status=0), missile 1 is decoy (status=0)
    # - Forced duplicate: drones 0, 1, 2 all assigned to missile 0 (when K=1)
    # - Forced decoy assignment: drone 3 assigned to decoy missile 1
    env.missile_is_decoy[1, 0] = False
    env.missile_is_decoy[1, 1] = True
    env.missile_status[1, 0] = 0  # In flight / live
    env.missile_status[1, 1] = 0  # In flight / live
    env.missile_K[1, :] = 1

    env.z[1, :, :] = False
    env.z[1, 0, 0] = True
    env.z[1, 1, 0] = True
    env.z[1, 2, 0] = True
    env.z[1, 3, 1] = True  # Assigned to decoy

    # Drones must be active interceptors
    env.drone_role[:, :4] = 0
    env.drone_active[:, :4] = True

    # Run one update step
    tracker.update(env, step_time_ms=5.0)

    # Finalize metrics
    res = tracker.finalize(env)

    # 1. Env 0 duplicate ownership must be exactly 0.0%
    assert res["duplicate_ownership_rate_pct"][0].item() == 0.0

    # 2. Env 1 duplicate ownership must be > 0.0%
    assert res["duplicate_ownership_rate_pct"][1].item() > 0.0

    # 3. Env 0 false target waste must be exactly 0.0%
    assert res["false_target_waste_pct"][0].item() == 0.0

    # 4. Env 1 false target waste must be > 0.0%
    assert res["false_target_waste_pct"][1].item() > 0.0

    # 5. Delta-V per intercept is finite
    assert torch.isfinite(res["mean_delta_v_per_intercept"]).all()

    # 6. Latency is recorded
    assert res["mean_step_latency_ms"] == pytest.approx(5.0, abs=1e-3)


def test_ablation_flags_cbf_and_kalman():
    """Verify that evaluate_policy_episodes supports all ablation flags and Kalman filtering advantage."""
    cfg = load_config("configs/default_sim.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Test clean execution of each ablation flag
    res_no_cbf = evaluate_policy_episodes(
        cfg=cfg,
        policy_name="auction_tpn",
        num_episodes=2,
        num_envs=2,
        seeds=(42,),
        scale_override={"num_interceptors": 4, "num_scouts": 1, "num_missiles": 5, "max_episode_steps": 30},
        ablation_flags={"use_cbf": False},
        device=device,
    )
    assert "intercept_rate_pct" in res_no_cbf

    res_no_residual = evaluate_policy_episodes(
        cfg=cfg,
        policy_name="attention_mappo",
        num_episodes=2,
        num_envs=2,
        seeds=(42,),
        scale_override={"num_interceptors": 4, "num_scouts": 1, "num_missiles": 5, "max_episode_steps": 30},
        ablation_flags={"use_residual_rl": False, "use_coord_logits": False},
        device=device,
    )
    assert "intercept_rate_pct" in res_no_residual

    # 2. Compare Kalman Filter (use_kalman=True) vs Raw Noisy Measurements (use_kalman=False) under noise
    scale_noise = {
        "num_interceptors": 6,
        "num_scouts": 2,
        "num_missiles": 8,
        "sigma_obs": 25.0,
        "max_episode_steps": 120,
    }

    res_kf = evaluate_policy_episodes(
        cfg=cfg,
        policy_name="auction_tpn",
        num_episodes=2,
        num_envs=2,
        seeds=(42,),
        scale_override=scale_noise,
        ablation_flags={"use_kalman": True},
        device=device,
    )

    res_raw = evaluate_policy_episodes(
        cfg=cfg,
        policy_name="auction_tpn",
        num_episodes=2,
        num_envs=2,
        seeds=(42,),
        scale_override=scale_noise,
        ablation_flags={"use_kalman": False},
        device=device,
    )

    # Kalman filter achieves equal or higher intercept rate or lower Delta-V waste under noisy measurements
    kf_intercept = res_kf["intercept_rate_pct"]["mean"]
    raw_intercept = res_raw["intercept_rate_pct"]["mean"]
    kf_dv = res_kf["mean_delta_v_per_intercept"]["mean"]
    raw_dv = res_raw["mean_delta_v_per_intercept"]["mean"]

    assert (kf_intercept >= raw_intercept) or (kf_dv <= raw_dv)


def test_run_benchmark_suite_quick_mode_artifacts(tmp_path):
    """Verify run_benchmark_suite generates all JSON, CSV, MD, and PNG plots."""
    cfg = load_config("configs/default_sim.yaml")
    output_dir = str(tmp_path / "benchmarks")

    result = run_benchmark_suite(
        cfg=cfg,
        seeds=(42, 43),
        episodes_per_seed=2,
        output_dir=output_dir,
        quick_mode=True,
    )

    # Verify structured dictionary return
    assert "results" in result
    assert "dataframe" in result
    assert "report_markdown" in result
    assert "artifacts" in result

    # Verify all expected artifact files exist and are non-empty
    expected_files = [
        "benchmark_results.json",
        "benchmark_summary.csv",
        "benchmark_report.md",
        "scale_generalization.png",
        "ablation_study.png",
        "pareto_latency_vs_intercept.png",
    ]

    for filename in expected_files:
        file_path = os.path.join(output_dir, filename)
        assert os.path.exists(file_path), f"Missing artifact: {filename}"
        assert os.path.getsize(file_path) > 0, f"Artifact is empty: {filename}"

    # Verify high intercept rates (>= 70.0%) for attention_mappo and hungarian_tpn on Small scale (6v8)
    scale_gen = result["results"]["scale_generalization"]
    small_scale_key = "Small (6v8)"
    attn_intercept = scale_gen[small_scale_key]["attention_mappo"]["summary"]["intercept_rate_pct"]["mean"]
    hung_intercept = scale_gen[small_scale_key]["hungarian_tpn"]["summary"]["intercept_rate_pct"]["mean"]
    assert attn_intercept >= 70.0, f"attention_mappo intercept rate was {attn_intercept:.1f}%, expected >= 70.0%"
    assert hung_intercept >= 70.0, f"hungarian_tpn intercept rate was {hung_intercept:.1f}%, expected >= 70.0%"

