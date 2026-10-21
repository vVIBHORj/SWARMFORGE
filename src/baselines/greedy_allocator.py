"""GPU-vectorized greedy task allocator."""

import torch


@torch.no_grad()
def greedy_assign(
    cost_matrix: torch.Tensor,
    valid_drone: torch.Tensor,
    valid_missile: torch.Tensor,
    missile_K: torch.Tensor,
    invalid_cost: float = 1.0e6,
) -> torch.Tensor:
    """Greedy assignment selecting minimal cost (drone, missile) pairs sequentially.

    cost_matrix: [E, N, M] float32
    valid_drone: [E, N] bool
    valid_missile: [E, M] bool
    missile_K: [E, M] int64

    Returns: z [E, N, M] bool
    """
    E, N, M = cost_matrix.shape
    device = cost_matrix.device

    avail_drone = valid_drone.clone()
    cap = torch.where(valid_missile, missile_K, torch.zeros_like(missile_K))
    z = torch.zeros((E, N, M), dtype=torch.bool, device=device)
    e_idx = torch.arange(E, device=device)

    for _ in range(N):
        pair_mask = avail_drone.unsqueeze(-1) & (cap.unsqueeze(1) > 0) & (cost_matrix < invalid_cost * 0.5)
        if not pair_mask.any():
            break

        masked_cost = torch.where(pair_mask, cost_matrix, torch.tensor(float('inf'), device=device, dtype=cost_matrix.dtype))
        flat_cost = masked_cost.view(E, -1)
        best_pair = torch.argmin(flat_cost, dim=-1)
        min_val = torch.gather(flat_cost, 1, best_pair.unsqueeze(1)).squeeze(1)
        valid_step = min_val < invalid_cost * 0.5

        if not valid_step.any():
            break

        best_drone = best_pair // M
        best_missile = best_pair % M

        # Update assignment
        z[e_idx, best_drone, best_missile] = torch.where(valid_step, True, z[e_idx, best_drone, best_missile])

        # Mask assigned drone
        d_mask = torch.zeros_like(avail_drone)
        d_mask[e_idx, best_drone] = valid_step
        avail_drone = avail_drone & (~d_mask)

        # Decrement capacity
        c_mask = torch.zeros_like(cap)
        c_mask[e_idx, best_missile] = valid_step.to(cap.dtype)
        cap = cap - c_mask

    return z
