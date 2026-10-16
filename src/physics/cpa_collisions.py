"""Continuous-time closest-point-of-approach (CPA) and collision detection."""

from typing import Optional, Tuple, Union
import torch

EPS = 1e-8


@torch.no_grad()
def pairwise_cpa(
    pA0: torch.Tensor,
    pA1: torch.Tensor,
    pB0: torch.Tensor,
    pB1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """pA0,pA1: [E,A,3] positions at start/end of step. pB0,pB1: [E,B,3].
    Returns d_min [E,A,B] and s_star [E,A,B] (fraction of the step in [0,1] at closest approach)."""
    # r0: displacement at start of step [E, A, B, 3]
    r0 = pB0.unsqueeze(1) - pA0.unsqueeze(2)
    # dr: change in displacement over step [E, A, B, 3]
    dr = (pB1.unsqueeze(1) - pB0.unsqueeze(1)) - (pA1.unsqueeze(2) - pA0.unsqueeze(2))

    r0_dot_dr = (r0 * dr).sum(dim=-1)
    dr_sq = (dr * dr).sum(dim=-1)

    s_star = torch.clamp(-r0_dot_dr / (dr_sq + EPS), min=0.0, max=1.0)
    r_cpa = r0 + s_star.unsqueeze(-1) * dr
    d_min = torch.linalg.norm(r_cpa, dim=-1)

    return d_min, s_star


@torch.no_grad()
def intercept_mask(
    d_min: torch.Tensor,
    r_int: Union[float, torch.Tensor],
    active_A: Optional[torch.Tensor] = None,
    active_B: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """d_min: [E,A,B]. active_A: [E,A], active_B: [E,B].
    Returns bool [E,A,B], True where d_min <= r_int and both active."""
    mask = d_min <= r_int
    if active_A is not None:
        mask = mask & (active_A.unsqueeze(-1) if active_A.dim() == 2 else active_A)
    if active_B is not None:
        mask = mask & (active_B.unsqueeze(1) if active_B.dim() == 2 else active_B)
    return mask


@torch.no_grad()
def friendly_collision_mask(
    pA0: torch.Tensor,
    pA1: torch.Tensor,
    r_col: Union[float, torch.Tensor],
    active_A: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """pA0,pA1: [E,A,3], r_col: float. active_A: [E,A].
    Returns bool [E,A,A]; pairs within 2*r_col using pairwise_cpa(A,A);
    diagonal forced False; both active required."""
    d_min, _ = pairwise_cpa(pA0, pA1, pA0, pA1)
    mask = intercept_mask(d_min, 2.0 * r_col, active_A, active_A)
    num_entities = pA0.shape[1]
    diag = torch.eye(num_entities, dtype=torch.bool, device=pA0.device).unsqueeze(0)
    return mask & (~diag)
