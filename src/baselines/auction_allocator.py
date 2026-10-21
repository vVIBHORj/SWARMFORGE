"""GPU-parallelized forward auction / CBBA allocator."""

import torch


@torch.no_grad()
def auction_assign(
    cost_matrix: torch.Tensor,
    valid_drone: torch.Tensor,
    valid_missile: torch.Tensor,
    missile_K: torch.Tensor,
    num_iters: int = 10,
    eps: float = 0.05,
    invalid_cost: float = 1.0e6,
) -> torch.Tensor:
    """Forward auction assignment operating fully in parallel across environments.

    cost_matrix: [E, N, M] float32
    valid_drone: [E, N] bool
    valid_missile: [E, M] bool
    missile_K: [E, M] int64

    Returns: z [E, N, M] bool
    """
    E, N, M = cost_matrix.shape
    device = cost_matrix.device

    valid_pair = (
        valid_drone.unsqueeze(-1)
        & valid_missile.unsqueeze(1)
        & (missile_K.unsqueeze(1) > 0)
        & (cost_matrix < invalid_cost * 0.5)
    )
    U = torch.where(valid_pair, invalid_cost - cost_matrix, torch.zeros_like(cost_matrix))
    p = torch.zeros((E, 1, M), device=device, dtype=cost_matrix.dtype)
    z = torch.zeros((E, N, M), dtype=torch.bool, device=device)
    m_range = torch.arange(M, device=device).view(1, 1, M)

    for _ in range(num_iters):
        assigned = z.any(dim=-1)
        unassigned = valid_drone & (~assigned)

        payoff = U - p
        payoff = torch.where(
            valid_pair & unassigned.unsqueeze(-1),
            payoff,
            torch.tensor(-1e9, device=device, dtype=cost_matrix.dtype),
        )

        top_v, top_i = torch.topk(payoff, k=min(2, M), dim=-1)
        best_v = top_v[..., 0]
        best_m = top_i[..., 0]
        second_v = top_v[..., 1] if M > 1 else torch.zeros_like(best_v)
        second_v = torch.where(second_v < 0, torch.zeros_like(second_v), second_v)

        has_bid = unassigned & (best_v > 0)

        bid_incr = best_v - second_v + eps
        new_bid_mask = has_bid.unsqueeze(-1) & (m_range == best_m.unsqueeze(-1))

        # Update prices for missiles receiving bids
        max_incr = (new_bid_mask * bid_incr.unsqueeze(-1)).max(dim=1, keepdim=True).values
        p = torch.where(new_bid_mask.any(dim=1, keepdim=True), p + max_incr, p)

        # Candidates for each missile: current holders + new bidders
        cands = z | new_bid_mask
        cand_val = torch.where(cands, U, torch.tensor(-1e9, device=device, dtype=cost_matrix.dtype))

        # Select top K_j candidates
        top_c_v, top_c_i = torch.topk(cand_val, k=min(2, N), dim=1)
        new_z = torch.zeros_like(z)

        v0 = (top_c_v[:, 0, :] > 0) & (missile_K >= 1)
        d0 = top_c_i[:, 0, :]
        new_z.scatter_(dim=1, index=d0.unsqueeze(1), src=v0.unsqueeze(1))

        if N > 1:
            v1 = (top_c_v[:, 1, :] > 0) & (missile_K >= 2)
            d1 = top_c_i[:, 1, :]
            new_z.scatter_(dim=1, index=d1.unsqueeze(1), src=v1.unsqueeze(1))

        z = new_z

    return z
