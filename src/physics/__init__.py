"""Physics package: kinematics, continuous-time collision checking, and 3D guidance laws."""

from src.physics.kinematics_3d import G0, EPS, g_to_ms2, step_kinematics
from src.physics.cpa_collisions import (
    pairwise_cpa,
    intercept_mask,
    friendly_collision_mask,
)
from src.physics.guidance_laws import (
    engagement_kinematics,
    tpn_guidance_accel,
    cbf_repulsion_accel,
    build_cost_matrix,
)

__all__ = [
    "G0",
    "EPS",
    "g_to_ms2",
    "step_kinematics",
    "pairwise_cpa",
    "intercept_mask",
    "friendly_collision_mask",
    "engagement_kinematics",
    "tpn_guidance_accel",
    "cbf_repulsion_accel",
    "build_cost_matrix",
]
