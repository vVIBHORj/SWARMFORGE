"""PettingZoo ParallelEnv wrapper for SwarmForge multi-agent drone defense.

Deliverable 1: SwarmParallelEnv wraps a single-environment (E=1) VectorizedSwarmEnv
with a BaselineController (for base TPN + allocator + CBF shield) and exposes the
standard PettingZoo ParallelEnv API.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from pettingzoo import ParallelEnv

from src.baselines.baseline_controller import BaselineController
from src.config import get
from src.envs.observations import build_observations
from src.envs.vectorized_swarm_env import VectorizedSwarmEnv


class SwarmParallelEnv(ParallelEnv):
    """PettingZoo ParallelEnv wrapper for single-env multi-agent swarm defense."""

    metadata = {"render_modes": ["human", "rgb_array"], "name": "swarm_parallel_v0"}

    def __init__(
        self,
        cfg: Optional[dict] = None,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
        allocator: str = "hungarian",
        use_cbf: bool = True,
        use_estimated_state: bool = False,
    ):
        super().__init__()
        if cfg is None:
            from src.config import load_config
            from pathlib import Path
            cfg_path = Path(__file__).resolve().parent.parent.parent / "configs" / "default_sim.yaml"
            cfg = load_config(cfg_path)

        self.cfg = cfg
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seed_val = seed
        self.use_estimated_state = use_estimated_state

        # Instantiate single-environment VectorizedSwarmEnv
        self.env = VectorizedSwarmEnv(cfg, num_envs=1, device=self.device, seed=seed)
        self.baseline = BaselineController(
            cfg,
            allocator=allocator,
            use_cbf=use_cbf,
            use_apn=False,
            use_estimated_state=use_estimated_state,
        )

        self.K_threat = int(get(cfg, "environment.max_neighbors_observed", 15))
        self.K_ally = int(get(cfg, "environment.max_allies_observed", 10))
        self.a_res_max = float(get(cfg, "training.a_res_max", 30.0))

        self.possible_agents = [f"drone_{i}" for i in range(self.env.N)]
        self.agents = list(self.possible_agents)

        # Build action and observation spaces
        self._action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3 + self.K_threat,),
            dtype=np.float32,
        )

        self._observation_space = spaces.Dict(
            {
                "ego": spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
                "threat_tokens": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.K_threat, 15), dtype=np.float32
                ),
                "threat_mask": spaces.Box(
                    low=0, high=1, shape=(self.K_threat,), dtype=bool
                ),
                "ally_tokens": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.K_ally, 9), dtype=np.float32
                ),
                "ally_mask": spaces.Box(
                    low=0, high=1, shape=(self.K_ally,), dtype=bool
                ),
            }
        )

        self.action_spaces = {a: self._action_space for a in self.possible_agents}
        self.observation_spaces = {a: self._observation_space for a in self.possible_agents}

    def observation_space(self, agent: str) -> spaces.Dict:
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Box:
        return self._action_space

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, dict], Dict[str, dict]]:
        """Reset the environment and return initial observations and infos."""
        if seed is not None:
            self.seed_val = seed
        self.env.reset(seed=self.seed_val)

        self.agents = [
            f"drone_{i}"
            for i in range(self.env.N)
            if bool(self.env.drone_active[0, i].item())
        ]

        obs_all = build_observations(
            self.env,
            K_threat=self.K_threat,
            K_ally=self.K_ally,
            use_estimated_state=self.use_estimated_state,
        )

        obs_dict = {}
        infos_dict = {}
        for agent in self.agents:
            i = int(agent.split("_")[1])
            obs_dict[agent] = {
                "ego": obs_all["ego"][0, i].cpu().numpy(),
                "threat_tokens": obs_all["threat_tokens"][0, i].cpu().numpy(),
                "threat_mask": obs_all["threat_mask"][0, i].cpu().numpy(),
                "ally_tokens": obs_all["ally_tokens"][0, i].cpu().numpy(),
                "ally_mask": obs_all["ally_mask"][0, i].cpu().numpy(),
            }
            infos_dict[agent] = {}

        return obs_dict, infos_dict

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, dict], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]]:
        """Step the environment with multi-agent residual actions."""
        # 1. Classical baseline guidance and allocation
        accel_cmd, _ = self.baseline.act(self.env)
        prev_z = self.env.z.clone()

        # 2. Apply RL residual accelerations if provided
        rl_residual_accel = torch.zeros_like(accel_cmd)
        for agent, act in actions.items():
            if agent in self.possible_agents:
                i = int(agent.split("_")[1])
                act_arr = np.asarray(act, dtype=np.float32)
                if act_arr.shape == (3 + self.K_threat,):
                    delta_a = torch.from_numpy(act_arr[:3]).to(self.device) * self.a_res_max
                    rl_residual_accel[0, i] = delta_a
                    accel_cmd[0, i] = accel_cmd[0, i] + delta_a
                elif act_arr.shape == (3,):
                    delta_a = torch.from_numpy(act_arr).to(self.device) * self.a_res_max
                    rl_residual_accel[0, i] = delta_a
                    accel_cmd[0, i] = accel_cmd[0, i] + delta_a

        # 3. Environment physical step
        info = self.env.step(accel_cmd)

        # 4. Compute rewards
        rewards_tensor, comps = self.env.compute_rewards(
            info, prev_z=prev_z, rl_residual_accel=rl_residual_accel
        )

        # 5. Build observations
        obs_all = build_observations(
            self.env,
            K_threat=self.K_threat,
            K_ally=self.K_ally,
            use_estimated_state=self.use_estimated_state,
        )

        # 6. Formulate PettingZoo return dicts
        obs_dict = {}
        rewards_dict = {}
        terminations_dict = {}
        truncations_dict = {}
        infos_dict = {}

        is_done = bool(self.env.done[0].item())
        is_trunc = bool(self.env.truncated[0].item())

        active_agents_this_step = list(self.agents)
        for agent in active_agents_this_step:
            i = int(agent.split("_")[1])
            drone_active = bool(self.env.drone_active[0, i].item())

            obs_dict[agent] = {
                "ego": obs_all["ego"][0, i].cpu().numpy(),
                "threat_tokens": obs_all["threat_tokens"][0, i].cpu().numpy(),
                "threat_mask": obs_all["threat_mask"][0, i].cpu().numpy(),
                "ally_tokens": obs_all["ally_tokens"][0, i].cpu().numpy(),
                "ally_mask": obs_all["ally_mask"][0, i].cpu().numpy(),
            }
            rewards_dict[agent] = float(rewards_tensor[0, i].item())
            terminations_dict[agent] = is_done or (not drone_active)
            truncations_dict[agent] = is_trunc
            infos_dict[agent] = {
                "role": int(self.env.drone_role[0, i].item()),
                "has_assignment": bool(self.env.z[0, i].any().item()),
                "assigned_intercept_reward": float(comps["assigned_intercept"][0, i].item()),
                "team_intercept_reward": float(comps["team_intercept"][0, i].item()),
                "shaping_reward": float(comps["shaping"][0, i].item()),
            }

        # Update remaining active agents
        self.agents = [
            a
            for a in self.agents
            if not (terminations_dict.get(a, False) or truncations_dict.get(a, False))
        ]

        return obs_dict, rewards_dict, terminations_dict, truncations_dict, infos_dict
