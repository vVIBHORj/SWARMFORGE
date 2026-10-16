"""3D vectorized kinematics module with Euclidean-norm limits."""

from typing import Optional, Tuple, Union
import torch

EPS = 1e-8
G0 = 9.80665


def g_to_ms2(g: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
    """Convert acceleration from g to m/s^2."""
    return g * G0


@torch.no_grad()
def step_kinematics(
    p: torch.Tensor,
    v: torch.Tensor,
    a_cmd: torch.Tensor,
    dt: float,
    a_max: Union[float, torch.Tensor],
    v_max: Union[float, torch.Tensor],
    drag: Union[float, torch.Tensor],
    active: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """p,v,a_cmd: [E,N,3]. active: optional bool [E,N] (True = alive).
    Returns p_new, v_new, a_applied (all [E,N,3])."""
    # 1. Euclidean-norm acceleration clamping
    norm_a = torch.linalg.norm(a_cmd, dim=-1, keepdim=True)
    if isinstance(a_max, torch.Tensor) and a_max.dim() == 2:
        a_max = a_max.unsqueeze(-1)
    a_scale = torch.clamp(a_max / (norm_a + EPS), max=1.0)
    a = a_cmd * a_scale

    # 2. Velocity update with drag
    if isinstance(drag, torch.Tensor) and drag.dim() == 2:
        drag = drag.unsqueeze(-1)
    v_tilde = (1.0 - drag * dt) * v + a * dt

    # 3. Euclidean-norm velocity clamping
    norm_v = torch.linalg.norm(v_tilde, dim=-1, keepdim=True)
    if isinstance(v_max, torch.Tensor) and v_max.dim() == 2:
        v_max = v_max.unsqueeze(-1)
    v_scale = torch.clamp(v_max / (norm_v + EPS), max=1.0)
    v_new = v_tilde * v_scale

    # 4. Trapezoidal position integration
    p_new = p + 0.5 * (v + v_new) * dt

    # 5. Mask out inactive entities
    if active is not None:
        mask = active.unsqueeze(-1) if active.dim() == 2 else active
        p_new = torch.where(mask, p_new, p)
        v_new = torch.where(mask, v_new, torch.zeros_like(v_new))
        a = torch.where(mask, a, torch.zeros_like(a))

    return p_new, v_new, a
