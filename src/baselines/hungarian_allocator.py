"""Hungarian (Munkres) optimal assignment baseline using scipy."""

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


@torch.no_grad()
def hungarian_assign(
    cost_matrix: torch.Tensor,
    valid_drone: torch.Tensor,
    valid_missile: torch.Tensor,
    missile_K: torch.Tensor,
    invalid_cost: float = 1.0e6,
) -> torch.Tensor:
    """Optimal bipartite matching per env using linear sum assignment with capacity expansion.

    cost_matrix: [E, N, M] float32
    valid_drone: [E, N] bool
    valid_missile: [E, M] bool
    missile_K: [E, M] int64

    Returns: z [E, N, M] bool
    """
    E, N, M = cost_matrix.shape
    device = cost_matrix.device

    C_cpu = cost_matrix.cpu().numpy()
    vd_cpu = valid_drone.cpu().numpy()
    vm_cpu = valid_missile.cpu().numpy()
    k_cpu = missile_K.cpu().numpy()

    z_np = np.zeros((E, N, M), dtype=bool)
    threshold = invalid_cost * 0.5

    for e in range(E):
        c_mat = C_cpu[e]  # [N, M]
        vd = vd_cpu[e]    # [N]
        vm = vm_cpu[e]    # [M]
        k = k_cpu[e]      # [M]

        # Capacity expansion: up to K_max = 2 columns per missile
        # Column 2*j: slot 1 (active if vm and k >= 1)
        # Column 2*j + 1: slot 2 (active if vm and k >= 2)
        exp_cost = np.full((N, 2 * M), invalid_cost, dtype=np.float32)

        for j in range(M):
            if not vm[j] or k[j] < 1:
                continue

            cost_col = np.where(vd, c_mat[:, j], invalid_cost)
            exp_cost[:, 2 * j] = cost_col

            if k[j] >= 2:
                exp_cost[:, 2 * j + 1] = cost_col

        row_ind, col_ind = linear_sum_assignment(exp_cost)
        for r, c in zip(row_ind, col_ind):
            if exp_cost[r, c] < threshold:
                j = c // 2
                z_np[e, r, j] = True

    return torch.from_numpy(z_np).to(device=device, dtype=torch.bool)
