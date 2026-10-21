"""Multi-target estimation and sensor fusion package."""

from src.estimation.kalman_tracker import (
    VectorizedKalmanTracker,
    compute_detections_and_measurements,
)

__all__ = [
    "VectorizedKalmanTracker",
    "compute_detections_and_measurements",
]
