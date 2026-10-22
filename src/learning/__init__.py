"""Learning package for multi-agent reinforcement learning.

Exports:
    CrossAttentionActorCritic: Permutation-invariant attention actor-critic network.
    FlatMLPAblationActorCritic: Flat MLP ablation baseline network.
    scatter_logits_to_full: Helper to scatter top-K threat logits into full [E, N, M].
    RolloutBuffer: Pre-allocated GPU rollout buffer with GAE computation.
    MAPPOTrainer: CTDE MAPPO trainer with AMP, hybrid action execution, and curriculum.
"""

from src.learning.attention_nets import (
    CrossAttentionActorCritic,
    FlatMLPAblationActorCritic,
    scatter_logits_to_full,
)
from src.learning.mappo_trainer import MAPPOTrainer
from src.learning.rollout_buffer import RolloutBuffer

__all__ = [
    "CrossAttentionActorCritic",
    "FlatMLPAblationActorCritic",
    "scatter_logits_to_full",
    "RolloutBuffer",
    "MAPPOTrainer",
]
