"""Scenario generation module for SwarmForge.

World convention:
Meters. World x,y in [-W/2, +W/2], z in [0, H] (z = altitude),
with W,L,H = environment.world_bounds_meters. HVAs sit on the ground (z=0).
Roles: 0 = interceptor, 1 = scout (interceptors first, then scouts, in drone index order).
Missile types: 0 = ballistic (straight line), 1 = dive (steers toward its HVA),
2 = weave (steers toward HVA + sinusoidal lateral acceleration).
"""

import math
from typing import Optional, Union
import torch

from src.config import get


def generate_scenario(
    cfg: dict,
    num_envs: int,
    num_interceptors: int,
    num_scouts: int,
    num_missiles: int,
    device: Union[str, torch.device],
    seed: int,
    mode: Optional[str] = None,
    weave_enabled: Optional[bool] = None,
    sigma_obs: Optional[float] = None,
) -> dict:
    """Generate synthetic initial scenario state for defending drones and incoming missiles."""
    device = torch.device(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    E, N, M = num_envs, num_interceptors + num_scouts, num_missiles

    mode = mode if mode is not None else get(cfg, "scenario.mode", "3d")
    weave_enabled = weave_enabled if weave_enabled is not None else get(cfg, "scenario.weave_enabled", True)
    sigma_obs = float(sigma_obs if sigma_obs is not None else get(cfg, "estimation.position_noise_std_meters", 0.0))
    dt = float(get(cfg, "environment.dt", 0.05))

    W, L, H = [float(v) for v in get(cfg, "environment.world_bounds_meters", [10000.0, 10000.0, 3000.0])]

    # 1. HVAs: ground level z=0 within +-hva_region_fraction of world half-extent
    P = int(get(cfg, "environment.num_defended_hvas", 2))
    hva_frac = float(get(cfg, "scenario.hva_region_fraction", 0.25))
    hva_rand = torch.rand((E, P, 2), generator=gen, device=device, dtype=torch.float32)
    hva_xy = (hva_rand * 2.0 - 1.0) * torch.tensor([(W / 2.0) * hva_frac, (L / 2.0) * hva_frac], device=device)
    hva_pos = torch.cat([hva_xy, torch.zeros((E, P, 1), device=device, dtype=torch.float32)], dim=-1)

    # 2. Drones: spawn on ring around HVA centroid
    hva_centroid = hva_pos.mean(dim=1, keepdim=True)
    ring_min, ring_max = get(cfg, "scenario.drone_ring_radius", [300.0, 1500.0])
    r_drone = ring_min + torch.rand((E, N, 1), generator=gen, device=device) * (ring_max - ring_min)
    az_drone = torch.rand((E, N, 1), generator=gen, device=device) * (2.0 * math.pi)

    dx = hva_centroid[..., 0:1] + r_drone * torch.cos(az_drone)
    dy = hva_centroid[..., 1:2] + r_drone * torch.sin(az_drone)
    if mode == "2d":
        dz = torch.zeros((E, N, 1), device=device, dtype=torch.float32)
    else:
        alt_d_min, alt_d_max = get(cfg, "scenario.drone_altitude_range", [100.0, 800.0])
        dz = alt_d_min + torch.rand((E, N, 1), generator=gen, device=device) * (alt_d_max - alt_d_min)

    drone_pos = torch.cat([dx, dy, dz], dim=-1)
    drone_vel = torch.zeros((E, N, 3), device=device, dtype=torch.float32)
    roles_1d = torch.cat([torch.zeros(num_interceptors, dtype=torch.int64, device=device),
                          torch.ones(num_scouts, dtype=torch.int64, device=device)])
    drone_role = roles_1d.unsqueeze(0).expand(E, N).clone()
    drone_active = torch.ones((E, N), device=device, dtype=torch.bool)

    # 3. Missiles: pattern-based perimeter spawn
    patterns = list(get(cfg, "scenario.patterns", ["saturation", "pincer", "wave"]))
    phi_base = torch.rand((E, 1), generator=gen, device=device) * (2.0 * math.pi)
    num_waves = int(get(cfg, "scenario.num_waves", 3))
    wave_interval_s = float(get(cfg, "scenario.wave_interval_s", 2.0))
    wave_k = (torch.arange(M, device=device, dtype=torch.float32) * num_waves) // M
    m_idx = torch.arange(M, device=device).unsqueeze(0).expand(E, M)

    p_idx = 0 if len(patterns) == 1 else torch.randint(0, len(patterns), (E, 1), generator=gen, device=device)
    azimuth = torch.zeros((E, M), device=device, dtype=torch.float32)
    spawn_time = torch.zeros((E, M), device=device, dtype=torch.float32)

    for idx, p_name in enumerate(patterns):
        mask = (p_idx == idx) if isinstance(p_idx, torch.Tensor) else True
        if p_name == "saturation":
            delta = (torch.rand((E, M), generator=gen, device=device) * 2.0 - 1.0) * math.radians(15.0)
            az = phi_base + delta
            azimuth = torch.where(mask, az, azimuth) if isinstance(mask, torch.Tensor) else az
        elif p_name == "pincer":
            sec_off = torch.where(m_idx < (M // 2), 0.0, math.pi)
            delta = (torch.rand((E, M), generator=gen, device=device) * 2.0 - 1.0) * math.radians(15.0)
            az = phi_base + sec_off + delta
            azimuth = torch.where(mask, az, azimuth) if isinstance(mask, torch.Tensor) else az
        elif p_name == "wave":
            delta = (torch.rand((E, M), generator=gen, device=device) * 2.0 - 1.0) * math.radians(30.0)
            az = phi_base + delta
            st = wave_k.unsqueeze(0).expand(E, M) * wave_interval_s
            azimuth = torch.where(mask, az, azimuth) if isinstance(mask, torch.Tensor) else az
            spawn_time = torch.where(mask, st, spawn_time) if isinstance(mask, torch.Tensor) else st

    tx = (W / 2.0) / (torch.abs(torch.cos(azimuth)) + 1e-8)
    ty = (L / 2.0) / (torch.abs(torch.sin(azimuth)) + 1e-8)
    t = torch.minimum(tx, ty)
    mx = torch.clamp(t * torch.cos(azimuth), -W / 2.0, W / 2.0)
    my = torch.clamp(t * torch.sin(azimuth), -L / 2.0, L / 2.0)

    if mode == "2d":
        mz = torch.zeros((E, M, 1), device=device, dtype=torch.float32)
    else:
        alt_m_min, alt_m_max = get(cfg, "scenario.missile_altitude_range", [1500.0, 3000.0])
        mz = alt_m_min + torch.rand((E, M, 1), generator=gen, device=device) * (alt_m_max - alt_m_min)
    missile_pos = torch.cat([mx.unsqueeze(-1), my.unsqueeze(-1), mz], dim=-1)

    missile_target_hva = torch.randint(0, P, (E, M), generator=gen, device=device, dtype=torch.int64)
    target_idx = missile_target_hva.unsqueeze(-1).expand(E, M, 3)
    target_hva_pos = torch.gather(hva_pos, dim=1, index=target_idx)

    diff = target_hva_pos - missile_pos
    unit_dir = diff / (torch.linalg.norm(diff, dim=-1, keepdim=True) + 1e-8)
    spd_min_frac, spd_max_frac = get(cfg, "scenario.missile_speed_fraction", [0.7, 0.95])
    max_speed = float(get(cfg, "entities.missiles.max_speed", 450.0))
    speed_frac = spd_min_frac + torch.rand((E, M, 1), generator=gen, device=device) * (spd_max_frac - spd_min_frac)
    missile_vel = unit_dir * (max_speed * speed_frac)

    type_upper = 3 if weave_enabled else 2
    missile_type = torch.randint(0, type_upper, (E, M), generator=gen, device=device, dtype=torch.int64)
    decoy_fraction = float(get(cfg, "entities.missiles.decoy_fraction", 0.15))
    missile_is_decoy = torch.rand((E, M), generator=gen, device=device) < decoy_fraction
    max_salvo_k = int(get(cfg, "entities.missiles.max_salvo_owners_k", 1))
    missile_K = torch.full((E, M), max_salvo_k, device=device, dtype=torch.int64)
    missile_weave_phase = torch.rand((E, M), generator=gen, device=device) * (2.0 * math.pi)

    return {
        "hva_pos": hva_pos,
        "drone_pos": drone_pos,
        "drone_vel": drone_vel,
        "drone_role": drone_role,
        "drone_active": torch.ones((E, N), device=device, dtype=torch.bool),
        "missile_pos": missile_pos,
        "missile_vel": missile_vel,
        "missile_type": missile_type,
        "missile_target_hva": missile_target_hva,
        "missile_is_decoy": missile_is_decoy,
        "missile_K": missile_K,
        "missile_weave_phase": missile_weave_phase,
        "missile_spawn_time": spawn_time,
        "missile_active": torch.ones((E, M), device=device, dtype=torch.bool),
        "meta": {"dt": dt, "sigma_obs": sigma_obs, "num_envs": E},
    }
