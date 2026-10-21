"""Classical multi-agent baseline allocation and control algorithms."""

from src.baselines.auction_allocator import auction_assign
from src.baselines.baseline_controller import BaselineController
from src.baselines.greedy_allocator import greedy_assign
from src.baselines.hungarian_allocator import hungarian_assign

__all__ = [
    "greedy_assign",
    "hungarian_assign",
    "auction_assign",
    "BaselineController",
]
