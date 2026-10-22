"""Centralized Training with Decentralized Execution (CTDE) MAPPO Trainer.

Spec Sections 6.1, 6.7, 8.1, 8.2:
- Mixed-Precision AMP (bfloat16) on GPU.
- Hybrid Neuro-Symbolic Action Execution (TPN + continuous RL residual + CBF shield).
- Continuous coordination logit scattering into GPU auction allocator.
- Automated 3-Stage Curriculum progression with zero-shot swarm size resizing.
- Host-sync-free inner rollout collection loop.
"""

import copy
from contextlib import nullcontext
import os
import time
from typing import Optional, Tuple
import torch
import torch.nn as nn

from src.baselines.auction_allocator import auction_assign
from src.config import get, load_config
from src.envs.curriculum_manager import CurriculumManager
from src.envs.observations import build_observations
from src.envs.rewards import RewardComputer
from src.envs.vectorized_swarm_env import VectorizedSwarmEnv
from src.learning.attention_nets import (
    CrossAttentionActorCritic,
    FlatMLPAblationActorCritic,
    scatter_logits_to_full,
)
from src.learning.rollout_buffer import RolloutBuffer
from src.physics.cpa_collisions import friendly_collision_mask
from src.physics.guidance_laws import (
    build_cost_matrix,
    cbf_repulsion_accel,
    tpn_guidance_accel,
)


class MAPPOTrainer:
    """Multi-Agent PPO Trainer with Permutation-Invariant Cross-Attention & Hybrid Action Execution."""

    def __init__(
        self,
        cfg: dict,
        device: Optional[str | torch.device] = None,
        use_curriculum: bool = True,
        model_type: str = "attention",
        log_dir: Optional[str] = None,
    ) -> None:
        self.cfg = copy.deepcopy(cfg)
        if device is None:
            dev_str = get(self.cfg, "training.device", "cuda" if torch.cuda.is_available() else "cpu")
            if dev_str == "cuda" and not torch.cuda.is_available():
                dev_str = "cpu"
            self.device = torch.device(dev_str)
        else:
            self.device = torch.device(device)

        self.seed = int(get(self.cfg, "environment.seed", 42))
        torch.manual_seed(self.seed)
        self.rng = torch.Generator(device=self.device).manual_seed(self.seed)

        # 1. Environment & Curriculum
        self.use_curriculum = use_curriculum
        if self.use_curriculum:
            self.curriculum = CurriculumManager(self.cfg)
        else:
            self.curriculum = None

        self.env = VectorizedSwarmEnv(self.cfg, device=self.device, seed=self.seed)
        if self.curriculum is not None:
            self.curriculum.apply_to_env(self.env, seed=self.seed)

        self.reward_computer = self.env.reward_computer

        # 2. Model Architecture
        self.model_type = model_type.lower()
        d_model = int(get(self.cfg, "training.d_model", 64))
        num_heads = int(get(self.cfg, "training.num_attention_heads", 2))
        res_accel_g = float(get(self.cfg, "training.residual_accel_max_g", 4.0))
        res_accel_ms2 = res_accel_g * 9.80665
        logit_scale = float(get(self.cfg, "training.logit_scale", 5.0))

        if self.model_type == "attention":
            self.model = CrossAttentionActorCritic(
                cfg=self.cfg,
                d_model=d_model,
                num_heads=num_heads,
                residual_accel_max_ms2=res_accel_ms2,
                logit_scale=logit_scale,
            ).to(self.device)
        elif self.model_type in {"mlp", "flat_mlp"}:
            K_t = int(get(self.cfg, "environment.max_neighbors_observed", 15))
            K_a = int(get(self.cfg, "environment.max_allies_observed", 10))
            self.model = FlatMLPAblationActorCritic(
                cfg=self.cfg,
                K_threat=K_t,
                K_ally=K_a,
                residual_accel_max_ms2=res_accel_ms2,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'attention' or 'mlp'.")

        # 3. Rollout Buffer
        self.rollout_steps = int(get(self.cfg, "training.rollout_steps", 256))
        self.K_threat = int(get(self.cfg, "environment.max_neighbors_observed", 15))
        self.K_ally = int(get(self.cfg, "environment.max_allies_observed", 10))
        self._init_rollout_buffer()

        # 4. Optimizer & Cosine LR Scheduler
        lr = float(get(self.cfg, "training.learning_rate", 3e-4))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, eps=1e-5)
        total_steps = int(get(self.cfg, "training.total_env_steps", 10_000_000))
        total_updates = max(1, total_steps // (self.env.num_envs * self.rollout_steps))
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_updates, eta_min=1e-6
        )

        # 5. Mixed Precision AMP
        cfg_amp = bool(get(self.cfg, "training.mixed_precision_amp", True))
        self.use_amp = cfg_amp and (self.device.type == "cuda")

        # 6. PPO Hyperparameters
        self.gamma = float(get(self.cfg, "training.gamma", 0.99))
        self.gae_lambda = float(get(self.cfg, "training.gae_lambda", 0.95))
        self.clip_eps = float(get(self.cfg, "training.clip_epsilon", 0.2))
        self.entropy_coef = float(get(self.cfg, "training.entropy_coef", 0.01))
        self.value_loss_coef = float(get(self.cfg, "training.value_loss_coef", 0.5))
        self.max_grad_norm = float(get(self.cfg, "training.max_grad_norm", 0.5))
        self.ppo_epochs = int(get(self.cfg, "training.ppo_epochs", 4))
        self.num_minibatches = int(get(self.cfg, "training.num_minibatches", 4))

        # 7. Hybrid Guidance & Allocation Parameters
        self.v_min = float(get(self.cfg, "guidance_and_allocation.v_min", 20.0))
        self.tgo_cap = float(get(self.cfg, "guidance_and_allocation.tgo_cap", 120.0))
        self.nav_gain = float(get(self.cfg, "guidance_and_allocation.tpn_nav_gain", 4.0))
        self.k_cbf = float(get(self.cfg, "guidance_and_allocation.cbf_gain", 500.0))
        self.r_col = float(get(self.cfg, "entities.drones.collision_radius", 5.0))
        self.r_safe = float(get(self.cfg, "entities.drones.cbf_safe_radius", 40.0))
        self.invalid_cost = float(get(self.cfg, "guidance_and_allocation.invalid_cost", 1.0e6))
        self.auction_num_iters = int(get(self.cfg, "guidance_and_allocation.auction_num_iters", 10))
        self.auction_eps = float(get(self.cfg, "guidance_and_allocation.auction_eps", 0.05))

        self.log_dir = log_dir
        self.update_idx = 0
        self.total_env_steps = 0
        self.current_obs = self.env.get_observations()

    def _init_rollout_buffer(self) -> None:
        """Allocate or re-allocate GPU rollout buffer matching env's current drone count."""
        self.buffer = RolloutBuffer(
            rollout_steps=self.rollout_steps,
            num_envs=self.env.num_envs,
            num_drones=self.env.N,
            K_threat=self.K_threat,
            K_ally=self.K_ally,
            device=self.device,
        )

    @torch.no_grad()
    def compute_hybrid_env_action(
        self,
        obs: dict,
        delta_a_rl: torch.Tensor,
        u_topk: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hybrid neuro-symbolic action (Spec Eq. 6.1: a_cmd = a_PN + Δa_RL + a_CBF).

        Args:
            obs: Vectorized observation dict.
            delta_a_rl: Continuous residual acceleration [E, N, 3] from actor.
            u_topk: Continuous pointer coordination logits [E, N, K_t].

        Returns:
            a_cmd: Hybrid acceleration command [E, N, 3] for environment step.
        """
        threat_indices = obs.get("threat_indices", None)
        if threat_indices is not None:
            rl_logits = scatter_logits_to_full(
                u_topk,
                threat_indices,
                obs["threat_mask"],
                self.env.M,
            )
        else:
            rl_logits = None

        # Build assignment cost matrix
        use_est = (self.env.sigma_obs > 0)
        C, _ = build_cost_matrix(
            self.env,
            prev_z=self.env.z,
            rl_logits=rl_logits,
            v_min=self.v_min,
            tgo_cap=self.tgo_cap,
            invalid_cost=self.invalid_cost,
            use_estimated_state=use_est,
        )

        valid_drone = self.env.drone_active & (self.env.drone_role == 0)
        if use_est and hasattr(self.env, "track_state") and self.env.track_state is not None:
            valid_m_init = self.env.track_state.get("initialized", torch.ones_like(self.env.missile_live()))
            valid_missile = self.env.missile_live() & valid_m_init
            m_pos = self.env.track_state.get("pos_est", self.env.missile_pos)
            m_vel = self.env.track_state.get("vel_est", self.env.missile_vel)
        else:
            valid_missile = self.env.missile_live()
            m_pos = self.env.missile_pos
            m_vel = self.env.missile_vel

        # Retain active in-flight commitments to prevent uncoordinated chattering
        active_commit = self.env.z & valid_drone.unsqueeze(-1) & valid_missile.unsqueeze(1)
        free_drone = valid_drone & (~active_commit.any(dim=-1))
        assigned_per_m = active_commit.sum(dim=1)
        rem_K = torch.clamp(self.env.missile_K - assigned_per_m, min=0)
        unassigned_missile = valid_missile & (rem_K > 0)

        # 100% GPU parallelized auction allocator
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
        self.env.z = z

        # Base 3D TPN guidance acceleration
        a_pn = tpn_guidance_accel(
            self.env.drone_pos,
            self.env.drone_vel,
            m_pos,
            m_vel,
            z,
            nav_gain=self.nav_gain,
            v_min=self.v_min,
        )

        # Control Barrier Function (CBF) inter-drone collision avoidance shield
        a_cbf = cbf_repulsion_accel(
            self.env.drone_pos,
            self.env.drone_active,
            r_col=self.r_col,
            r_safe=self.r_safe,
            k_cbf=self.k_cbf,
        )

        # Combine neuro-symbolic commands: a_cmd = a_pn + delta_a_rl + a_cbf
        a_cmd = a_pn + delta_a_rl + a_cbf

        # Only active interceptors execute high-g interception maneuvers
        is_interceptor = (self.env.drone_active & (self.env.drone_role == 0)).unsqueeze(-1)
        a_cmd = torch.where(is_interceptor, a_cmd, torch.zeros_like(a_cmd))
        return a_cmd

    @torch.no_grad()
    def collect_rollout(self) -> dict:
        """Collect fixed rollout steps on GPU without host synchronization inside the loop."""
        self.buffer.reset()
        start_time = time.time()

        episode_rewards = torch.zeros(self.env.num_envs, device=self.device)
        completed_records = []

        for _ in range(self.rollout_steps):
            # Model forward
            action, log_prob, _, value, u_topk = self.model.get_action_and_value(self.current_obs)

            # Compute hybrid env action
            a_cmd = self.compute_hybrid_env_action(self.current_obs, action, u_topk)

            # Step environment
            prev_z = self.env.z.clone()
            info = self.env.step(a_cmd)

            # Compute shaped reward
            reward, rew_comps, new_phi = self.reward_computer.compute_step_reward(
                self.env,
                self.env._prev_phi,
                prev_z=prev_z,
                info=info,
                rl_residual_accel=action,
            )
            self.env._prev_phi = new_phi

            # Insert step into rollout buffer
            newly_done = info["newly_done"]
            self.buffer.insert(self.current_obs, action, log_prob, reward, newly_done, value)

            # Accumulate reward and record completed episode metrics
            team_reward = reward.mean(dim=-1)
            episode_rewards += team_reward

            step_metrics = self.env.metrics()
            completed_records.append((
                step_metrics["intercept_rate"],
                step_metrics["leaker_rate"],
                episode_rewards.clone(),
                newly_done,
            ))

            episode_rewards = torch.where(newly_done, torch.zeros_like(episode_rewards), episode_rewards)

            # In-place masked GPU reset
            self.env.reset_envs(newly_done)
            self.current_obs = self.env.get_observations()

        # Compute Generalized Advantage Estimation
        next_value = self.model.get_value(self.current_obs)
        self.buffer.compute_gae(next_value, self.env.done, self.gamma, self.gae_lambda)

        elapsed = max(1e-5, time.time() - start_time)
        steps_collected = self.rollout_steps * self.env.num_envs
        self.total_env_steps += steps_collected
        sps = steps_collected / elapsed

        # Aggregate completed episode metrics across the rollout (single host sync point)
        all_done = torch.cat([r[3] for r in completed_records])
        all_ints = torch.cat([r[0] for r in completed_records])
        all_leaks = torch.cat([r[1] for r in completed_records])
        all_rets = torch.cat([r[2] for r in completed_records])

        episodes_completed = int(all_done.sum().item())
        if episodes_completed > 0:
            fin_ints = all_ints[all_done]
            fin_leaks = all_leaks[all_done]
            fin_rets = all_rets[all_done]
            mean_int = float(fin_ints.mean().item())
            mean_leak = float(fin_leaks.mean().item())
            mean_rew = float(fin_rets.mean().item())

            # Check curriculum progression
            if self.curriculum is not None:
                advanced = self.curriculum.record_episodes(fin_ints)
                if advanced:
                    self.on_curriculum_stage_advance()
        else:
            mean_int = 0.0
            mean_leak = 0.0
            mean_rew = float(self.buffer.rewards.mean().item())

        peak_vram = (
            torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            if self.device.type == "cuda"
            else 0.0
        )

        return {
            "mean_reward": mean_rew,
            "mean_intercept_rate": mean_int,
            "mean_leaker_rate": mean_leak,
            "episodes_completed": episodes_completed,
            "sps": sps,
            "peak_vram_mb": peak_vram,
        }

    def ppo_update(self) -> dict:
        """Perform PPO mini-batch policy and value updates with clipped surrogate objectives."""
        self.update_idx += 1
        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        clip_fracs = []

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp)
            if self.device.type == "cuda"
            else nullcontext()
        )

        for _ in range(self.ppo_epochs):
            for mb in self.buffer.get_minibatches(self.num_minibatches, generator=self.rng):
                with autocast_ctx:
                    _, new_log_prob, entropy, new_value, _ = self.model.get_action_and_value(
                        mb["obs"], action=mb["actions"]
                    )

                    mask = mb["drone_mask"].float()
                    mask_sum = mask.sum().clamp(min=1.0)

                    # 1. Clipped Surrogate Policy Loss
                    log_ratio = new_log_prob - mb["old_log_probs"]
                    ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))

                    surr1 = ratio * mb["advantages"]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb["advantages"]
                    policy_loss = -(torch.min(surr1, surr2) * mask).sum() / mask_sum

                    # 2. Clipped Value Loss
                    v_clipped = mb["values"] + torch.clamp(new_value - mb["values"], -self.clip_eps, self.clip_eps)
                    v_loss1 = (new_value - mb["returns"]).square()
                    v_loss2 = (v_clipped - mb["returns"]).square()
                    value_loss = 0.5 * (torch.max(v_loss1, v_loss2) * mask).sum() / mask_sum

                    # 3. Entropy Loss
                    entropy_loss = (entropy * mask).sum() / mask_sum

                    # Total Loss
                    loss = (
                        policy_loss
                        + self.value_loss_coef * value_loss
                        - self.entropy_coef * entropy_loss
                    )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Metric logging
                with torch.no_grad():
                    approx_kl = (((ratio - 1.0) - log_ratio) * mask).sum() / mask_sum
                    clip_frac = (((ratio - 1.0).abs() > self.clip_eps).float() * mask).sum() / mask_sum

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_loss.item())
                approx_kls.append(approx_kl.item())
                clip_fracs.append(clip_frac.item())

        self.lr_scheduler.step()
        current_lr = self.optimizer.param_groups[0]["lr"]

        return {
            "policy_loss": sum(policy_losses) / len(policy_losses),
            "value_loss": sum(value_losses) / len(value_losses),
            "entropy": sum(entropies) / len(entropies),
            "approx_kl": sum(approx_kls) / len(approx_kls),
            "clip_frac": sum(clip_fracs) / len(clip_fracs),
            "lr": current_lr,
        }

    def on_curriculum_stage_advance(self) -> None:
        """Handle zero-shot scaling when curriculum advances to next stage."""
        stage = self.curriculum.current_stage
        self.curriculum.apply_to_env(self.env)
        self._init_rollout_buffer()
        self.current_obs = self.env.get_observations()
        print(
            f"[CURRICULUM] Advanced to Stage {stage}: "
            f"N={self.env.N} ({self.env.num_interceptors} int, {self.env.num_scouts} sct), "
            f"M={self.env.M} missiles, mode={self.env.mode}"
        )

    def save_checkpoint(self, path: str, extra_meta: Optional[dict] = None) -> None:
        """Save training checkpoint including model, optimizer, curriculum, and metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "curriculum_stage": self.curriculum.current_stage if self.curriculum is not None else None,
            "update_idx": self.update_idx,
            "total_env_steps": self.total_env_steps,
            "cfg": self.cfg,
            "model_type": self.model_type,
        }
        if extra_meta:
            ckpt["extra_meta"] = extra_meta
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str) -> dict:
        """Load training checkpoint and restore model, optimizer, and curriculum stage."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "lr_scheduler_state_dict" in ckpt:
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
        if self.curriculum is not None and ckpt.get("curriculum_stage") is not None:
            target_stage = ckpt["curriculum_stage"]
            if target_stage != self.curriculum.current_stage:
                self.curriculum.current_stage = target_stage
                self.on_curriculum_stage_advance()
        self.update_idx = ckpt.get("update_idx", 0)
        self.total_env_steps = ckpt.get("total_env_steps", 0)
        return ckpt
