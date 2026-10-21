"""Unit tests verifying ownership invariants across all allocators and random batches."""

import pytest
import torch

from src.baselines.auction_allocator import auction_assign
from src.baselines.greedy_allocator import greedy_assign
from src.baselines.hungarian_allocator import hungarian_assign

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("allocator_fn", [greedy_assign, hungarian_assign, auction_assign])
def test_ownership_invariants_10000_instances(device, allocator_fn):
    """Verify ownership constraints on 100 batches of E=100 (10,000 instances total)."""
    torch.manual_seed(42)
    if allocator_fn == hungarian_assign:
        E, N, M = 50, 12, 16
        num_batches = 10
    else:
        E, N, M = 100, 20, 30
        num_batches = 100

    for _ in range(num_batches):
        cost_matrix = torch.rand(E, N, M, device=device, dtype=torch.float32) * 500.0

        # Random active masks
        valid_drone = torch.rand(E, N, device=device) > 0.1  # ~90% active interceptors
        valid_missile = torch.rand(E, M, device=device) > 0.1  # ~90% live missiles
        # Random capacity K in {1, 2}
        missile_K = torch.randint(1, 3, (E, M), device=device, dtype=torch.int64)

        z = allocator_fn(cost_matrix, valid_drone, valid_missile, missile_K)

        # 1. Every drone owns at most 1 missile
        assert (z.sum(dim=2) <= 1).all(), "Drone assigned to more than 1 missile!"

        # 2. Every missile has at most K_j owners
        assert (z.sum(dim=1) <= missile_K).all(), "Missile exceeded ownership capacity K_j!"

        # 3. No invalid pair is ever assigned
        valid_pair = valid_drone.unsqueeze(-1) & valid_missile.unsqueeze(1) & (missile_K.unsqueeze(1) > 0)
        invalid_assignments = (~valid_pair) & z
        assert not invalid_assignments.any(), "Invalid drone or missile was assigned!"


@pytest.mark.parametrize("device", DEVICES)
def test_exact_4x4_global_optimum(device):
    """On known 4x4 cost matrix with unique optimum, hungarian and auction find the exact global minimum."""
    c = torch.tensor(
        [
            [
                [10.0, 5.0, 12.0, 15.0],
                [4.0, 8.0, 11.0, 20.0],
                [14.0, 9.0, 3.0, 7.0],
                [16.0, 12.0, 8.0, 2.0],
            ]
        ],
        device=device,
        dtype=torch.float32,
    )
    K = torch.ones((1, 4), device=device, dtype=torch.int64)
    valid_d = torch.ones((1, 4), device=device, dtype=torch.bool)
    valid_m = torch.ones((1, 4), device=device, dtype=torch.bool)

    # Expected unique minimum: 0->1 (5), 1->0 (4), 2->2 (3), 3->3 (2)
    expected_z = torch.zeros((1, 4, 4), device=device, dtype=torch.bool)
    expected_z[0, 0, 1] = True
    expected_z[0, 1, 0] = True
    expected_z[0, 2, 2] = True
    expected_z[0, 3, 3] = True

    z_hung = hungarian_assign(c, valid_d, valid_m, K)
    assert torch.equal(z_hung, expected_z), f"Hungarian did not find global optimum: {z_hung.nonzero()}"

    z_auc = auction_assign(c, valid_d, valid_m, K, num_iters=15, eps=0.01)
    assert torch.equal(z_auc, expected_z), f"Auction did not find global optimum: {z_auc.nonzero()}"
