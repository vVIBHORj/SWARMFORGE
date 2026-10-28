"""Evaluation Metrics Engine and Multi-Seed Episode Evaluator.

Spec Section 10:
Tracks 7 Metric Categories across all parallel environments:
1. Intercept Success Rate (%)
2. Leaker Rate to HVA (%)
3. Duplicate Threat Ownership Rate (%)
4. False-Target Waste Rate (%)
5. Friendly Collision Count
6. Delta-V Efficiency (Total and per-intercept m/s)
7. Step Latency (ms) and Peak VRAM Allocation (MB)
"""

import copy
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from src.baselines.auction_allocator import auction_assign
from src.baselines.baseline_controller import BaselineController
from src.config import get, load_config
from src.envs.vectorized_swarm_env import VectorizedSwarmEnv
from src.learning.attention_nets import (
    CrossAttentionActorCritic,
    FlatMLPAblationActorCritic,
    scatter_logits_to_full,
)
from src.physics.guidance_laws import (
    build_cost_matrix,
    cbf_repulsion_accel,
    engagement_kinematics,
    tpn_guidance_accel,
)


class EpisodeMetricsTracker:
    """Tracks per-step accumulators across an episode for all E parallel environments without host syncs."""

    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        self.reset()

    def reset(self) -> None:
        """Reset per-step accumulators to zeros."""
        self.total_threat_steps = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.duplicate_violation_steps = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.total_interceptor_assign_steps = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.decoy_assign_steps = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.step_latencies_ms: List[float] = []

    @torch.no_grad()
    def update(self, env: VectorizedSwarmEnv, step_time_ms: float = 0.0) -> None:
        """Update step accumulators for running environments (~env.done)."""
        running = (~env.done).float()

        # 1. Total threat steps: live missiles across active envs
        live_m = env.missile_live()  # [E, M] bool
        self.total_threat_steps += live_m.sum(dim=-1).float() * running

        # 2. Duplicate violation steps: missiles where z.sum(dim=1) > missile_K while still live
        dup_mask = (env.z.sum(dim=1) > env.missile_K) & live_m  # [E, M] bool
        self.duplicate_violation_steps += dup_mask.sum(dim=-1).float() * running

        # 3. Interceptor assignment steps: active defending interceptors (role == 0)
        is_interceptor = env.drone_active & (env.drone_role == 0)  # [E, N] bool
        self.total_interceptor_assign_steps += is_interceptor.sum(dim=-1).float() * running

        # 4. Decoy assignment steps: interceptors assigned to decoys
        # env.z: [E, N, M], env.missile_is_decoy: [E, M]
        decoy_assigned = (env.z & env.missile_is_decoy[:, None, :]).any(dim=-1) & is_interceptor
        self.decoy_assign_steps += decoy_assigned.sum(dim=-1).float() * running

        # 5. Wall-clock step latency
        if step_time_ms > 0.0:
            self.step_latencies_ms.append(float(step_time_ms))

    @torch.no_grad()
    def finalize(self, env: VectorizedSwarmEnv) -> dict:
        """Combine terminal env.metrics() with step accumulators and compute summary statistics.

        Returns:
            Dictionary containing per-env [E] float tensors and aggregate summary statistics (mean, std, ci95).
        """
        env_m = env.metrics()

        # Metric 1: Intercept rate (%)
        intercept_rate_pct = env_m["intercept_rate"] * 100.0

        # Metric 2: Leaker rate (%)
        leaker_rate_pct = env_m["leaker_rate"] * 100.0

        # Metric 3: Duplicate ownership rate (%)
        threat_steps_safe = torch.clamp(self.total_threat_steps, min=1.0)
        duplicate_ownership_rate_pct = (self.duplicate_violation_steps / threat_steps_safe) * 100.0

        # Metric 4: False target waste rate (%)
        int_assign_steps_safe = torch.clamp(self.total_interceptor_assign_steps, min=1.0)
        false_target_waste_pct = (self.decoy_assign_steps / int_assign_steps_safe) * 100.0

        # Metric 5: Friendly collisions per episode
        friendly_collision_count = env_m["friendly_collisions"]

        # Metric 6 & 7: Delta-V
        mean_delta_v_per_intercept = env_m["delta_v_per_intercept"]
        delta_v_total = env_m["delta_v_total"]

        # Latency & Peak VRAM
        mean_step_latency_ms = float(np.mean(self.step_latencies_ms)) if len(self.step_latencies_ms) > 0 else 0.0
        peak_vram_mb = (
            float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))
            if (torch.cuda.is_available() and self.device.type == "cuda")
            else 0.0
        )

        per_env: Dict[str, torch.Tensor] = {
            "intercept_rate_pct": intercept_rate_pct,
            "leaker_rate_pct": leaker_rate_pct,
            "duplicate_ownership_rate_pct": duplicate_ownership_rate_pct,
            "false_target_waste_pct": false_target_waste_pct,
            "friendly_collision_count": friendly_collision_count,
            "mean_delta_v_per_intercept": mean_delta_v_per_intercept,
            "delta_v_total": delta_v_total,
        }

        summary: Dict[str, Dict[str, float]] = {}
        for k, v in per_env.items():
            arr = v.detach().cpu().numpy()
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            ci95_val = float(1.96 * std_val / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
            summary[k] = {"mean": mean_val, "std": std_val, "ci95": ci95_val}

        summary["mean_step_latency_ms"] = {"mean": mean_step_latency_ms, "std": 0.0, "ci95": 0.0}
        summary["peak_vram_mb"] = {"mean": peak_vram_mb, "std": 0.0, "ci95": 0.0}

        return {
            **per_env,
            "per_env": per_env,
            "summary": summary,
            "mean_step_latency_ms": mean_step_latency_ms,
            "peak_vram_mb": peak_vram_mb,
        }


@torch.no_grad()
def _compute_mappo_step_action(
    env: VectorizedSwarmEnv,
    model: torch.nn.Module,
    use_cbf: bool = True,
    use_kalman: bool = True,
    use_residual_rl: bool = True,
    use_coord_logits: bool = True,
    v_min: float = 20.0,
    tgo_cap: float = 120.0,
    nav_gain: float = 4.0,
    k_cbf: float = 500.0,
    r_col: float = 5.0,
    r_safe: float = 40.0,
    auction_num_iters: int = 10,
    auction_eps: float = 0.05,
    invalid_cost: float = 1e6,
) -> torch.Tensor:
    """Compute hybrid neuro-symbolic action for MAPPO policies under specified ablation flags."""
    # 1. Optionally apply Kalman ablation to existing track_state
    if not use_kalman and hasattr(env, "track_state") and env.track_state is not None:
        env.track_state["pos_est"] = env.track_state["raw_pos"]
        env.track_state["vel_est"] = env.track_state["raw_vel"]

    # 2. Extract observations
    obs = env.get_observations(use_estimated_state=True)

    # 3. Model forward pass
    action, _, _, _, u_topk = model.get_action_and_value(obs, deterministic=True)

    # 4. Ablation overrides
    if not use_residual_rl:
        action = torch.zeros_like(action)
    if not use_coord_logits:
        u_topk = torch.zeros_like(u_topk)

    # 5. Coordination logits to full cost matrix
    threat_indices = obs.get("threat_indices", None)
    if threat_indices is not None and use_coord_logits:
        rl_logits = scatter_logits_to_full(
            u_topk,
            threat_indices,
            obs["threat_mask"],
            env.M,
        )
    else:
        rl_logits = None

    # 6. Build assignment cost matrix
    use_est = (env.sigma_obs > 0)
    C, _ = build_cost_matrix(
        env,
        prev_z=env.z,
        rl_logits=rl_logits,
        v_min=v_min,
        tgo_cap=tgo_cap,
        invalid_cost=invalid_cost,
        use_estimated_state=use_est,
    )

    # 7. Auction assignment with dynamic terminal endgame commitment lock
    valid_drone = env.drone_active & (env.drone_role == 0)
    if use_est and hasattr(env, "track_state") and env.track_state is not None:
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
        v_min=v_min,
        tgo_cap=tgo_cap,
    )
    dist = kin["dist"]
    Vc = kin["Vc"]

    # Only hard-lock targets when inside terminal engagement (dist < 800m and Vc > 0)
    is_terminal = (dist < 800.0) & (Vc > 0.0)
    active_commit = env.z & valid_drone.unsqueeze(-1) & valid_missile.unsqueeze(1) & is_terminal
    free_drone = valid_drone & (~active_commit.any(dim=-1))
    assigned_per_m = active_commit.sum(dim=1)
    rem_K = torch.clamp(env.missile_K - assigned_per_m, min=0)
    unassigned_missile = valid_missile & (rem_K > 0)

    z_alloc = auction_assign(
        C,
        free_drone,
        unassigned_missile,
        rem_K,
        num_iters=auction_num_iters,
        eps=auction_eps,
        invalid_cost=invalid_cost,
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

    # 8. Base TPN guidance
    a_pn = tpn_guidance_accel(
        env.drone_pos,
        env.drone_vel,
        m_pos,
        m_vel,
        z,
        nav_gain=nav_gain,
        v_min=v_min,
    )

    # 9. CBF safety shield
    if use_cbf:
        a_cbf = cbf_repulsion_accel(
            env.drone_pos,
            env.drone_active,
            r_col=r_col,
            r_safe=r_safe,
            k_cbf=k_cbf,
        )
    else:
        a_cbf = torch.zeros_like(a_pn)

    # 10. Combine neuro-symbolic commands
    a_cmd = a_pn + action + a_cbf
    is_interceptor = (env.drone_active & (env.drone_role == 0)).unsqueeze(-1)
    a_cmd = torch.where(is_interceptor, a_cmd, torch.zeros_like(a_cmd))
    return a_cmd


@torch.no_grad()
def evaluate_policy_episodes(
    cfg: dict,
    policy_name: str,
    num_episodes: int = 20,
    num_envs: int = 10,
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    scale_override: Optional[dict] = None,
    ablation_flags: Optional[dict] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> dict:
    """Evaluate a classical baseline or MAPPO policy across multiple random seeds.

    Args:
        cfg: Base configuration dictionary.
        policy_name: One of 'greedy', 'hungarian_tpn', 'auction_tpn', 'flat_mlp_mappo', 'attention_mappo'.
        num_episodes: Total episodes to evaluate per seed.
        num_envs: Number of parallel environments to run simultaneously.
        seeds: Random seeds to evaluate across.
        scale_override: Dict overriding swarm sizes (e.g. num_interceptors, num_scouts, num_missiles).
        ablation_flags: Dict of booleans (use_cbf, use_kalman, use_residual_rl, use_coord_logits).
        checkpoint_path: Optional path to .pt neural network checkpoint.
        device: Torch device (defaults to CUDA if available).

    Returns:
        Structured dictionary containing per-seed metric means and overall summary statistics (mean, std, ci95).
    """
    policy_name = policy_name.lower()
    valid_policies = {"greedy", "hungarian_tpn", "auction_tpn", "flat_mlp_mappo", "attention_mappo"}
    if policy_name not in valid_policies:
        raise ValueError(f"Unknown policy_name: '{policy_name}'. Must be one of {valid_policies}.")

    # Configure device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Auto-load trained checkpoint if available and not explicitly specified
    if checkpoint_path is None and os.path.exists("runs/mappo/best.pt"):
        checkpoint_path = "runs/mappo/best.pt"

    # Parse ablation flags
    ablation_flags = ablation_flags or {}
    use_cbf = bool(ablation_flags.get("use_cbf", True))
    use_kalman = bool(ablation_flags.get("use_kalman", True))
    use_residual_rl = bool(ablation_flags.get("use_residual_rl", True))
    use_coord_logits = bool(ablation_flags.get("use_coord_logits", True))

    # Prepare configuration copy and apply scale overrides
    cfg_run = copy.deepcopy(cfg)
    if scale_override is not None:
        if "num_interceptors" in scale_override:
            cfg_run["entities"]["drones"]["num_interceptors"] = int(scale_override["num_interceptors"])
        if "num_scouts" in scale_override:
            cfg_run["entities"]["drones"]["num_scouts"] = int(scale_override["num_scouts"])
        if "num_missiles" in scale_override:
            cfg_run["entities"]["missiles"]["num_missiles"] = int(scale_override["num_missiles"])
        if "max_episode_steps" in scale_override:
            cfg_run["environment"]["max_episode_steps"] = int(scale_override["max_episode_steps"])
        else:
            cfg_run["environment"]["max_episode_steps"] = max(400, int(get(cfg_run, "environment.max_episode_steps", 400)))
        if "sigma_obs" in scale_override:
            cfg_run["estimation"]["position_noise_std_meters"] = float(scale_override["sigma_obs"])
            cfg_run["estimation"]["enable_sensor_noise"] = (float(scale_override["sigma_obs"]) > 0.0)
        if "drone_ring_radius" in scale_override:
            cfg_run["scenario"]["drone_ring_radius"] = list(scale_override["drone_ring_radius"])
    else:
        cfg_run["environment"]["max_episode_steps"] = max(400, int(get(cfg_run, "environment.max_episode_steps", 400)))

    # Baseline allocator parameters
    nav_gain = float(get(cfg_run, "guidance_and_allocation.navigation_gain_N", 4.0))
    v_min = float(get(cfg_run, "guidance_and_allocation.v_min_closing", 20.0))
    tgo_cap = float(get(cfg_run, "guidance_and_allocation.tgo_cap", 120.0))
    k_cbf = float(get(cfg_run, "guidance_and_allocation.cbf_gain", 500.0))
    r_col = float(get(cfg_run, "entities.drones.collision_radius", 5.0))
    r_safe = float(get(cfg_run, "entities.drones.cbf_safe_radius", 40.0))
    auction_num_iters = int(get(cfg_run, "guidance_and_allocation.auction_num_iters", 10))
    auction_eps = float(get(cfg_run, "guidance_and_allocation.auction_eps", 0.05))
    invalid_cost = float(get(cfg_run, "guidance_and_allocation.invalid_cost", 1e6))

    # Initialize neural network model if MAPPO policy
    model = None
    if policy_name == "attention_mappo":
        d_model = int(get(cfg_run, "training.d_model", 64))
        num_heads = int(get(cfg_run, "training.num_attention_heads", 2))
        res_accel_g = float(get(cfg_run, "training.residual_accel_max_g", 4.0))
        logit_scale = float(get(cfg_run, "training.logit_scale", 5.0))
        model = CrossAttentionActorCritic(
            cfg=cfg_run,
            d_model=d_model,
            num_heads=num_heads,
            residual_accel_max_ms2=res_accel_g * 9.80665,
            logit_scale=logit_scale,
        ).to(device)
    elif policy_name == "flat_mlp_mappo":
        K_t = int(get(cfg_run, "environment.max_neighbors_observed", 15))
        K_a = int(get(cfg_run, "environment.max_allies_observed", 10))
        res_accel_g = float(get(cfg_run, "training.residual_accel_max_g", 4.0))
        model = FlatMLPAblationActorCritic(
            cfg=cfg_run,
            K_threat=K_t,
            K_ally=K_a,
            residual_accel_max_ms2=res_accel_g * 9.80665,
        ).to(device)

    if model is not None:
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location=device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
        model.eval()

    # Determine batch dimensions
    batch_envs = min(num_envs, num_episodes)
    num_batches = math.ceil(num_episodes / batch_envs)

    # Initialize environment once per evaluation run
    env = VectorizedSwarmEnv(cfg_run, num_envs=batch_envs, device=device, seed=seeds[0])
    tracker = EpisodeMetricsTracker(num_envs=batch_envs, device=device)

    # Controller instantiation for classical baselines
    controller = None
    if policy_name in {"greedy", "hungarian_tpn", "auction_tpn"}:
        allocator = "greedy" if policy_name == "greedy" else ("hungarian" if policy_name == "hungarian_tpn" else "auction")
        controller = BaselineController(
            cfg_run,
            allocator=allocator,
            use_cbf=use_cbf,
            use_estimated_state=True,
        )

    per_seed_results: Dict[int, Dict[str, float]] = {}

    for seed in seeds:
        torch.manual_seed(seed)
        seed_env_tensors: Dict[str, List[torch.Tensor]] = {
            "intercept_rate_pct": [],
            "leaker_rate_pct": [],
            "duplicate_ownership_rate_pct": [],
            "false_target_waste_pct": [],
            "friendly_collision_count": [],
            "mean_delta_v_per_intercept": [],
            "delta_v_total": [],
        }
        seed_latencies: List[float] = []

        episodes_collected = 0
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_envs, num_episodes - episodes_collected)
            if current_batch_size <= 0:
                break

            # Reset environment and tracker for this batch
            env.reset(seed=seed + batch_idx * 1000)
            tracker.reset()

            # Run episode until all parallel envs terminate or max steps reached
            max_steps = env.max_episode_steps
            for step_idx in range(max_steps):
                t0 = time.perf_counter()

                if controller is not None:
                    # Baseline execution
                    if not use_kalman and hasattr(env, "track_state") and env.track_state is not None:
                        env.track_state["pos_est"] = env.track_state["raw_pos"]
                        env.track_state["vel_est"] = env.track_state["raw_vel"]
                    a_cmd, _ = controller.act(env)
                else:
                    # MAPPO execution
                    a_cmd = _compute_mappo_step_action(
                        env=env,
                        model=model,
                        use_cbf=use_cbf,
                        use_kalman=use_kalman,
                        use_residual_rl=use_residual_rl,
                        use_coord_logits=use_coord_logits,
                        v_min=v_min,
                        tgo_cap=tgo_cap,
                        nav_gain=nav_gain,
                        k_cbf=k_cbf,
                        r_col=r_col,
                        r_safe=r_safe,
                        auction_num_iters=auction_num_iters,
                        auction_eps=auction_eps,
                        invalid_cost=invalid_cost,
                    )

                env.step(a_cmd)
                step_time_ms = (time.perf_counter() - t0) * 1000.0
                tracker.update(env, step_time_ms=step_time_ms)

                # Early break immediately when all parallel envs terminate
                if (step_idx >= 100 or (step_idx + 1) % 5 == 0) and env.done.all():
                    break

            batch_res = tracker.finalize(env)
            for k in seed_env_tensors.keys():
                seed_env_tensors[k].append(batch_res["per_env"][k])
            seed_latencies.extend(tracker.step_latencies_ms)
            episodes_collected += current_batch_size

        # Compute seed means
        seed_metrics: Dict[str, float] = {}
        for k, tensor_list in seed_env_tensors.items():
            concatenated = torch.cat(tensor_list, dim=0)
            seed_metrics[k] = float(concatenated.mean().item())

        seed_metrics["mean_step_latency_ms"] = float(np.mean(seed_latencies)) if seed_latencies else 0.0
        seed_metrics["peak_vram_mb"] = (
            float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
            if (torch.cuda.is_available() and device.type == "cuda")
            else 0.0
        )
        per_seed_results[seed] = seed_metrics

    # Aggregate across seeds
    all_metric_keys = list(next(iter(per_seed_results.values())).keys())
    summary: Dict[str, Dict[str, float]] = {}
    for k in all_metric_keys:
        vals = [per_seed_results[s][k] for s in seeds]
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci95_v = float(1.96 * std_v / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        summary[k] = {"mean": mean_v, "std": std_v, "ci95": ci95_v}

    out: Dict[str, Union[dict, str, list]] = {
        "policy": policy_name,
        "seeds": list(seeds),
        "per_seed": per_seed_results,
        "summary": summary,
    }
    # Provide direct top-level access to metric summary stats
    for k in all_metric_keys:
        out[k] = summary[k]

    return out
