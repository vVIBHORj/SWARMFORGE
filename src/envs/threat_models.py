"""Threat dynamics and guidance models for incoming missiles."""

from typing import Union
import torch

from src.config import get

EPS = 1e-8
G0 = 9.80665


@torch.no_grad()
def missile_accel_cmd(
    pos: torch.Tensor,
    vel: torch.Tensor,
    hva_pos: torch.Tensor,
    target_hva: torch.Tensor,
    mtype: torch.Tensor,
    weave_phase: torch.Tensor,
    t_sec: Union[float, torch.Tensor],
    cfg: dict,
    mode: str = "3d",
) -> torch.Tensor:
    """Compute guidance acceleration commands for missiles.

    pos, vel: [E, M, 3]
    hva_pos: [E, P, 3]
    target_hva: [E, M] (int64 index into hva_pos)
    mtype: [E, M] (0 = ballistic, 1 = dive, 2 = weave)
    weave_phase: [E, M] (radians)
    t_sec: float sim time in seconds
    cfg: sim configuration dict
    mode: '3d' or '2d'

    Returns: a_cmd [E, M, 3] (unclamped acceleration command).
    """
    E, M, _ = pos.shape

    # Gather target HVA position for each missile: [E, M, 3]
    target_idx = target_hva.unsqueeze(-1).expand(E, M, 3)
    target_pos = torch.gather(hva_pos, dim=1, index=target_idx)

    # Line of sight vector to target HVA
    diff = target_pos - pos
    if mode == "2d":
        diff = diff.clone()
        diff[..., 2] = 0.0

    diff_norm = torch.linalg.norm(diff, dim=-1, keepdim=True)
    r_hat = diff / (diff_norm + EPS)
    speed = torch.linalg.norm(vel, dim=-1, keepdim=True)
    v_des = r_hat * speed

    # Type 1: Dive guidance (pure-pursuit steering)
    steer_gain = 2.0  # 1/s
    a_dive = steer_gain * (v_des - vel)

    # Type 2: Weave guidance (dive command + sinusoidal lateral acceleration)
    weave_amp_g = float(get(cfg, "entities.missiles.weave_amplitude_g", 4.0))
    omega = float(get(cfg, "entities.missiles.weave_frequency_rad_s", 1.5))
    A_weave = weave_amp_g * G0

    z_up = torch.tensor([0.0, 0.0, 1.0], device=vel.device, dtype=vel.dtype)
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=vel.device, dtype=vel.dtype)

    n1 = torch.cross(vel, z_up.expand_as(vel), dim=-1)
    n1_norm = torch.linalg.norm(n1, dim=-1, keepdim=True)
    n2 = torch.cross(vel, x_axis.expand_as(vel), dim=-1)

    n_vec = torch.where(n1_norm > 1e-4, n1, n2)
    n_hat = n_vec / (torch.linalg.norm(n_vec, dim=-1, keepdim=True) + EPS)

    phase = weave_phase.unsqueeze(-1)
    t_val = t_sec.unsqueeze(-1) if (isinstance(t_sec, torch.Tensor) and t_sec.dim() == 2) else t_sec
    a_w = A_weave * torch.sin(omega * t_val + phase) * n_hat
    a_weave = a_dive + a_w

    # Compose command using torch.where over mtype
    mtype_mask = mtype.unsqueeze(-1)
    a_cmd = torch.zeros_like(vel)
    a_cmd = torch.where(mtype_mask == 1, a_dive, a_cmd)
    a_cmd = torch.where(mtype_mask == 2, a_weave, a_cmd)

    if mode == "2d":
        a_cmd[..., 2] = 0.0

    return a_cmd
