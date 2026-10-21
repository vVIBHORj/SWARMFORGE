"""Environment modules: scenario generator, threat models, vectorized environment, observations, rewards, curriculum, and PettingZoo wrapper."""

from src.envs.curriculum_manager import CurriculumManager
from src.envs.observations import build_observations
from src.envs.pettingzoo_wrapper import SwarmParallelEnv
from src.envs.rewards import RewardComputer
from src.envs.scenario_generator import generate_scenario
from src.envs.threat_models import missile_accel_cmd
from src.envs.vectorized_swarm_env import VectorizedSwarmEnv

__all__ = [
    "generate_scenario",
    "missile_accel_cmd",
    "VectorizedSwarmEnv",
    "build_observations",
    "RewardComputer",
    "CurriculumManager",
    "SwarmParallelEnv",
]
