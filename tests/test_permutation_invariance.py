"""Spec Section 11 Mandatory Verification Tests: Permutation Invariance & Equivariance.

Verifies:
1. Threat slot permutation invariance of continuous action mu and centralized critic value,
   and equivariance of pointer coordination logits u_topk.
2. Ally slot permutation invariance of mu, u_topk, and value.
3. Drone team permutation equivariance across active drones.
4. Contrast failure with FlatMLPAblationActorCritic under threat permutation.
5. All-masked safety test (zero visible threats & allies produces zero NaNs/Infs).
6. Strict parameter budget check (< 100,000 trainable parameters).
"""

import pytest
import torch

from src.learning.attention_nets import (
    CrossAttentionActorCritic,
    FlatMLPAblationActorCritic,
)

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


def _build_dummy_obs(
    E: int = 4,
    N: int = 8,
    K_t: int = 15,
    K_a: int = 10,
    device: str = "cpu",
    all_masked: bool = False,
) -> dict:
    """Helper to generate well-formed random observation tensors."""
    torch.manual_seed(42)
    ego = torch.randn(E, N, 10, device=device)
    threat_tokens = torch.randn(E, N, K_t, 15, device=device)
    ally_tokens = torch.randn(E, N, K_a, 9, device=device)

    if all_masked:
        threat_mask = torch.zeros(E, N, K_t, dtype=torch.bool, device=device)
        ally_mask = torch.zeros(E, N, K_a, dtype=torch.bool, device=device)
    else:
        # At least some valid tokens
        threat_mask = torch.rand(E, N, K_t, device=device) > 0.3
        threat_mask[:, :, 0] = True  # ensure at least one valid
        ally_mask = torch.rand(E, N, K_a, device=device) > 0.3
        ally_mask[:, :, 0] = True

    drone_mask = torch.ones(E, N, dtype=torch.bool, device=device)
    drone_mask[:, -1] = False  # test with one inactive drone

    return {
        "ego": ego,
        "threat_tokens": threat_tokens,
        "threat_mask": threat_mask,
        "ally_tokens": ally_tokens,
        "ally_mask": ally_mask,
        "drone_mask": drone_mask,
    }


@pytest.mark.parametrize("device", DEVICES)
def test_threat_permutation_invariance(device: str):
    """Permuting threat tokens must leave mu and value identical, and permute u_topk equivariantly."""
    model = CrossAttentionActorCritic(d_model=64, num_heads=2).to(device).eval()
    obs = _build_dummy_obs(E=4, N=8, K_t=15, K_a=10, device=device)

    with torch.no_grad():
        out = model(obs)
        mu, value, u_topk = out["mu"], out["value"], out["u_topk"]

        # Generate fixed permutation of threat slots
        perm = torch.randperm(15, device=device)
        obs_perm = {
            "ego": obs["ego"].clone(),
            "threat_tokens": obs["threat_tokens"][:, :, perm, :].clone(),
            "threat_mask": obs["threat_mask"][:, :, perm].clone(),
            "ally_tokens": obs["ally_tokens"].clone(),
            "ally_mask": obs["ally_mask"].clone(),
            "drone_mask": obs["drone_mask"].clone(),
        }

        out_perm = model(obs_perm)
        mu_perm, value_perm, u_topk_perm = (
            out_perm["mu"],
            out_perm["value"],
            out_perm["u_topk"],
        )

    # Actor continuous action mu and Critic value must be numerically identical within 1e-5
    assert torch.allclose(mu, mu_perm, atol=1e-5), (
        f"mu max diff: {(mu - mu_perm).abs().max().item()}"
    )
    assert torch.allclose(value, value_perm, atol=1e-5), (
        f"value max diff: {(value - value_perm).abs().max().item()}"
    )

    # Coordination logits u_topk must be permuted equivariantly
    # u_topk[:, :, perm] should match u_topk_perm
    u_topk_expected = u_topk[:, :, perm]
    assert torch.allclose(u_topk_expected, u_topk_perm, atol=1e-5), (
        f"u_topk perm max diff: {(u_topk_expected - u_topk_perm).abs().max().item()}"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_ally_permutation_invariance(device: str):
    """Permuting ally tokens must leave mu, value, and u_topk identical."""
    model = CrossAttentionActorCritic(d_model=64, num_heads=2).to(device).eval()
    obs = _build_dummy_obs(E=4, N=8, K_t=15, K_a=10, device=device)

    with torch.no_grad():
        out = model(obs)
        mu, value, u_topk = out["mu"], out["value"], out["u_topk"]

        perm = torch.randperm(10, device=device)
        obs_perm = {
            "ego": obs["ego"].clone(),
            "threat_tokens": obs["threat_tokens"].clone(),
            "threat_mask": obs["threat_mask"].clone(),
            "ally_tokens": obs["ally_tokens"][:, :, perm, :].clone(),
            "ally_mask": obs["ally_mask"][:, :, perm].clone(),
            "drone_mask": obs["drone_mask"].clone(),
        }

        out_perm = model(obs_perm)
        mu_perm, value_perm, u_topk_perm = (
            out_perm["mu"],
            out_perm["value"],
            out_perm["u_topk"],
        )

    assert torch.allclose(mu, mu_perm, atol=1e-5)
    assert torch.allclose(value, value_perm, atol=1e-5)
    assert torch.allclose(u_topk, u_topk_perm, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_drone_team_permutation_invariance(device: str):
    """Permuting drones across dim=1 must permute mu, u_topk, and value equivariantly."""
    model = CrossAttentionActorCritic(d_model=64, num_heads=2).to(device).eval()
    obs = _build_dummy_obs(E=4, N=8, K_t=15, K_a=10, device=device)

    with torch.no_grad():
        out = model(obs)
        mu, value, u_topk = out["mu"], out["value"], out["u_topk"]

        perm = torch.randperm(8, device=device)
        obs_perm = {
            "ego": obs["ego"][:, perm, :].clone(),
            "threat_tokens": obs["threat_tokens"][:, perm, :, :].clone(),
            "threat_mask": obs["threat_mask"][:, perm, :].clone(),
            "ally_tokens": obs["ally_tokens"][:, perm, :, :].clone(),
            "ally_mask": obs["ally_mask"][:, perm, :].clone(),
            "drone_mask": obs["drone_mask"][:, perm].clone(),
        }

        out_perm = model(obs_perm)
        mu_perm, value_perm, u_topk_perm = (
            out_perm["mu"],
            out_perm["value"],
            out_perm["u_topk"],
        )

    assert torch.allclose(mu[:, perm, :], mu_perm, atol=1e-5)
    assert torch.allclose(value[:, perm], value_perm, atol=1e-5)
    assert torch.allclose(u_topk[:, perm, :], u_topk_perm, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_contrast_with_flat_mlp_ablation(device: str):
    """FlatMLPAblationActorCritic FAILS threat permutation invariance (justifying Cross-Attention)."""
    mlp_model = (
        FlatMLPAblationActorCritic(
            ego_dim=10,
            K_threat=15,
            threat_dim=15,
            K_ally=10,
            ally_dim=9,
        )
        .to(device)
        .eval()
    )
    obs = _build_dummy_obs(E=2, N=4, K_t=15, K_a=10, device=device)

    with torch.no_grad():
        out = mlp_model(obs)
        mu = out["mu"]

        perm = torch.randperm(15, device=device)
        obs_perm = {
            "ego": obs["ego"].clone(),
            "threat_tokens": obs["threat_tokens"][:, :, perm, :].clone(),
            "threat_mask": obs["threat_mask"][:, :, perm].clone(),
            "ally_tokens": obs["ally_tokens"].clone(),
            "ally_mask": obs["ally_mask"].clone(),
            "drone_mask": obs["drone_mask"].clone(),
        }
        out_perm = mlp_model(obs_perm)
        mu_perm = out_perm["mu"]

    # In a flat MLP, changing input slot ordering changes the linear layer weights applied to tokens!
    # Therefore outputs MUST differ significantly.
    diff = (mu - mu_perm).abs().max().item()
    assert diff > 1e-3, f"Flat MLP unexpectedly invariant: diff={diff}"


@pytest.mark.parametrize("device", DEVICES)
def test_all_masked_safety(device: str):
    """When a drone has zero visible threats and zero visible allies, outputs must contain zero NaNs or Infs."""
    model = CrossAttentionActorCritic(d_model=64, num_heads=2).to(device).eval()
    obs = _build_dummy_obs(E=4, N=8, K_t=15, K_a=10, device=device, all_masked=True)

    with torch.no_grad():
        out = model(obs)
        mu = out["mu"]
        value = out["value"]
        u_topk = out["u_topk"]

        assert not torch.isnan(mu).any(), "NaN found in mu with all masks False"
        assert not torch.isinf(mu).any(), "Inf found in mu with all masks False"
        assert not torch.isnan(value).any(), "NaN found in value with all masks False"
        assert not torch.isinf(value).any(), "Inf found in value with all masks False"
        assert not torch.isnan(u_topk).any(), "NaN found in u_topk with all masks False"
        assert not torch.isinf(u_topk).any(), "Inf found in u_topk with all masks False"

        # Action sampling
        action, log_prob, entropy, val, logits = model.get_action_and_value(obs)
        assert not torch.isnan(action).any(), "NaN in sampled action"
        assert not torch.isnan(log_prob).any(), "NaN in log_prob"
        assert not torch.isnan(entropy).any(), "NaN in entropy"


def test_parameter_budget():
    """Assert CrossAttentionActorCritic parameters < 100,000 (targeting ~42k-50k)."""
    model = CrossAttentionActorCritic(d_model=64, num_heads=2)
    param_count = model.count_parameters()
    assert param_count < 100_000, f"Parameter count {param_count} exceeds 100,000 budget!"
    assert 30_000 <= param_count <= 60_000, f"Parameter count {param_count} deviates from ~42k-50k target"
