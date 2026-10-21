"""Three-stage automated curriculum manager for multi-agent swarm defense.

Spec Section 8.1:
- Stage 1 (Foundational Intercept & Deconfliction):
    N=6 interceptors, M=8 missiles, 2D plane, no decoys, no weave, sigma_obs=0, drag=0.
    Advances at >= 90% intercept rate.
- Stage 2 (3D Maneuvering Salvage & Drag):
    N=12 interceptors, 3 scouts, M=25 missiles, 3D volume, weave active, drag=0.02.
    Advances at >= 85% intercept rate.
- Stage 3 (Full Swarm Saturation & Sensor Fog):
    N=16 interceptors, 4 scouts, M=30 missiles, 15% decoys, sigma_obs=15m, drag=0.015.
    Caps at Stage 3.
"""

from collections import deque
from typing import Optional, Union
import torch

from src.config import get


class CurriculumManager:
    """Manages the 3-stage training progression based on rolling evaluation intercept rates."""

    DEFAULT_STAGES = {
        1: {
            "num_interceptors": 6,
            "num_scouts": 0,
            "num_missiles": 8,
            "mode": "2d",
            "weave_enabled": False,
            "decoy_fraction": 0.0,
            "sigma_obs": 0.0,
            "linear_drag": 0.0,
            "win_threshold": 0.90,
        },
        2: {
            "num_interceptors": 12,
            "num_scouts": 3,
            "num_missiles": 25,
            "mode": "3d",
            "weave_enabled": True,
            "decoy_fraction": 0.0,
            "sigma_obs": 0.0,
            "linear_drag": 0.02,
            "win_threshold": 0.85,
        },
        3: {
            "num_interceptors": 16,
            "num_scouts": 4,
            "num_missiles": 30,
            "mode": "3d",
            "weave_enabled": True,
            "decoy_fraction": 0.15,
            "sigma_obs": 15.0,
            "linear_drag": 0.015,
            "win_threshold": 1.0,
        },
    }

    def __init__(self, cfg: dict, initial_stage: int = 1):
        self.cfg = cfg
        self.current_stage = int(initial_stage)
        if self.current_stage not in {1, 2, 3}:
            raise ValueError(f"initial_stage must be in {{1, 2, 3}}, got {initial_stage}")

        self.window_size = int(get(cfg, "curriculum.window_size", 50))

        # Load stage configurations from cfg with fallbacks
        self.stages = {}
        for s in (1, 2, 3):
            stage_cfg = get(cfg, f"curriculum.stage{s}", self.DEFAULT_STAGES[s])
            self.stages[s] = {
                "num_interceptors": int(stage_cfg.get("num_interceptors", self.DEFAULT_STAGES[s]["num_interceptors"])),
                "num_scouts": int(stage_cfg.get("num_scouts", self.DEFAULT_STAGES[s]["num_scouts"])),
                "num_missiles": int(stage_cfg.get("num_missiles", self.DEFAULT_STAGES[s]["num_missiles"])),
                "mode": str(stage_cfg.get("mode", self.DEFAULT_STAGES[s]["mode"])),
                "weave_enabled": bool(stage_cfg.get("weave_enabled", self.DEFAULT_STAGES[s]["weave_enabled"])),
                "decoy_fraction": float(stage_cfg.get("decoy_fraction", self.DEFAULT_STAGES[s]["decoy_fraction"])),
                "sigma_obs": float(stage_cfg.get("sigma_obs", self.DEFAULT_STAGES[s]["sigma_obs"])),
                "linear_drag": float(stage_cfg.get("linear_drag", self.DEFAULT_STAGES[s]["linear_drag"])),
                "win_threshold": float(stage_cfg.get("win_threshold", self.DEFAULT_STAGES[s]["win_threshold"])),
            }

        self.buffer = deque(maxlen=self.window_size)

    def get_stage_kwargs(self) -> dict:
        """Return the environment configuration arguments for the current stage."""
        stage = self.stages[self.current_stage]
        return {
            "num_interceptors": stage["num_interceptors"],
            "num_scouts": stage["num_scouts"],
            "num_missiles": stage["num_missiles"],
            "mode": stage["mode"],
            "weave_enabled": stage["weave_enabled"],
            "decoy_fraction": stage["decoy_fraction"],
            "sigma_obs": stage["sigma_obs"],
            "linear_drag": stage["linear_drag"],
        }

    def apply_to_env(self, env, seed: Optional[int] = None) -> None:
        """Apply the current curriculum stage parameters to an environment."""
        kwargs = self.get_stage_kwargs()

        # Update environment config overrides
        if "entities" not in env.cfg:
            env.cfg["entities"] = {}
        if "missiles" not in env.cfg["entities"]:
            env.cfg["entities"]["missiles"] = {}
        if "drones" not in env.cfg["entities"]:
            env.cfg["entities"]["drones"] = {}

        env.cfg["entities"]["missiles"]["decoy_fraction"] = kwargs["decoy_fraction"]
        env.cfg["entities"]["missiles"]["linear_drag"] = kwargs["linear_drag"]
        env.cfg["entities"]["drones"]["linear_drag"] = kwargs["linear_drag"]

        env.reset(
            seed=seed,
            num_interceptors=kwargs["num_interceptors"],
            num_scouts=kwargs["num_scouts"],
            num_missiles=kwargs["num_missiles"],
            mode=kwargs["mode"],
            weave_enabled=kwargs["weave_enabled"],
            sigma_obs=kwargs["sigma_obs"],
        )

    def record_episodes(
        self, intercept_rates: Union[torch.Tensor, list[float], float]
    ) -> bool:
        """Append completed episode intercept rates to the rolling buffer.

        If the buffer is full (len >= window_size), mean >= win_threshold,
        and current_stage < 3, advances current_stage, clears the buffer,
        and returns True. Otherwise returns False.
        """
        if isinstance(intercept_rates, torch.Tensor):
            rates = intercept_rates.detach().cpu().flatten().tolist()
        elif isinstance(intercept_rates, (int, float)):
            rates = [float(intercept_rates)]
        else:
            rates = [float(r) for r in intercept_rates]

        for r in rates:
            self.buffer.append(r)

        if len(self.buffer) >= self.window_size:
            mean_val = sum(self.buffer) / len(self.buffer)
            threshold = self.stages[self.current_stage]["win_threshold"]
            if mean_val >= threshold and self.current_stage < 3:
                self.current_stage += 1
                self.buffer.clear()
                return True

        return False
