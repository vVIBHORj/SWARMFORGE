"""Unit tests for MAPPOTrainer, curriculum zero-shot scaling, checkpoints, and VRAM budget.

Tests:
1. Single rollout collection and PPO update (finite losses, parameter movement, zero NaNs).
2. Zero-shot curriculum stage transition and swarm size scaling (Stage 1 -> Stage 2).
3. Checkpoint save & load roundtrip with exact output reproduction.
4. RTX 3050 VRAM budget (< 2.8 GB) with 32 parallel environments on CUDA.
"""

import copy
import pytest
import torch

from src.config import load_config
from src.learning.mappo_trainer import MAPPOTrainer

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


def _make_test_cfg(**overrides) -> dict:
    cfg = copy.deepcopy(load_config("configs/default_sim.yaml"))
    for k, v in overrides.items():
        keys = k.split(".")
        d = cfg
        for sub_k in keys[:-1]:
            d = d.setdefault(sub_k, {})
        d[keys[-1]] = v
    return cfg


@pytest.mark.parametrize("device", DEVICES)
def test_mappo_single_rollout_and_ppo_update(device: str):
    """MAPPOTrainer collects a rollout and performs a PPO gradient step with finite losses."""
    cfg = _make_test_cfg(
        **{
            "environment.num_parallel_envs": 4,
            "training.rollout_steps": 16,
            "training.ppo_epochs": 2,
            "training.num_minibatches": 2,
            "training.device": device,
        }
    )

    trainer = MAPPOTrainer(cfg=cfg, device=device, use_curriculum=False, model_type="attention")

    # Capture initial weights of ego encoder
    initial_param = trainer.model.ego_encoder[0].weight.clone()

    # Collect rollout
    rollout_stats = trainer.collect_rollout()
    assert rollout_stats["episodes_completed"] >= 0
    assert rollout_stats["sps"] > 0
    assert trainer.buffer.ptr == trainer.rollout_steps

    # PPO update
    ppo_stats = trainer.ppo_update()

    # Assert finite losses
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl"):
        val = ppo_stats[key]
        assert torch.isfinite(torch.tensor(val)), f"Metric {key} is not finite: {val}"

    # Assert parameter weights have updated
    updated_param = trainer.model.ego_encoder[0].weight
    assert not torch.equal(initial_param, updated_param), "Model parameters did not update during PPO step!"

    # Assert zero NaNs in weights
    for name, p in trainer.model.named_parameters():
        assert not torch.isnan(p).any(), f"NaN found in parameter {name}"
        assert not torch.isinf(p).any(), f"Inf found in parameter {name}"


@pytest.mark.parametrize("device", DEVICES)
def test_mappo_zero_shot_curriculum_stage_resize(device: str):
    """Curriculum stage advance resizes env and buffer while model weights seamlessly generalize."""
    cfg = _make_test_cfg(
        **{
            "environment.num_parallel_envs": 2,
            "training.rollout_steps": 8,
            "training.ppo_epochs": 1,
            "training.num_minibatches": 2,
            "training.device": device,
        }
    )

    trainer = MAPPOTrainer(cfg=cfg, device=device, use_curriculum=True, model_type="attention")
    assert trainer.curriculum.current_stage == 1
    assert trainer.env.N == 6
    assert trainer.env.M == 8
    assert trainer.buffer.num_drones == 6

    # 1. Run 1 rollout & update on Stage 1
    trainer.collect_rollout()
    trainer.ppo_update()

    # 2. Feed 50 winning episodes (1.0 intercept rate) to trigger stage transition
    advanced = trainer.curriculum.record_episodes([1.0] * 50)
    assert advanced, "Curriculum failed to advance with 50 perfect episodes!"
    trainer.on_curriculum_stage_advance()

    # Verify stage 2 properties (12 interceptors + 3 scouts = 15 drones, 25 missiles)
    assert trainer.curriculum.current_stage == 2
    assert trainer.env.N == 15
    assert trainer.env.M == 25
    assert trainer.buffer.num_drones == 15

    # 3. Collect rollout and update on Stage 2 without shape error
    rollout_stats = trainer.collect_rollout()
    ppo_stats = trainer.ppo_update()

    assert torch.isfinite(torch.tensor(ppo_stats["policy_loss"]))
    assert rollout_stats["sps"] > 0


@pytest.mark.parametrize("device", DEVICES)
def test_checkpoint_save_and_load_roundtrip(device: str, tmp_path):
    """Saved checkpoint restores model weights, optimizer, curriculum stage, and deterministic actions."""
    cfg = _make_test_cfg(
        **{
            "environment.num_parallel_envs": 2,
            "training.rollout_steps": 8,
            "training.device": device,
        }
    )

    trainer1 = MAPPOTrainer(cfg=cfg, device=device, use_curriculum=True)
    # Run 1 update to alter initial weights
    trainer1.collect_rollout()
    trainer1.ppo_update()

    ckpt_path = str(tmp_path / "test_ckpt.pt")
    trainer1.save_checkpoint(ckpt_path, extra_meta={"test_metric": 42.0})

    # Instantiate second trainer and load checkpoint
    trainer2 = MAPPOTrainer(cfg=cfg, device=device, use_curriculum=True)
    trainer2.load_checkpoint(ckpt_path)

    assert trainer2.update_idx == trainer1.update_idx
    assert trainer2.total_env_steps == trainer1.total_env_steps
    assert trainer2.curriculum.current_stage == trainer1.curriculum.current_stage

    # Verify identical deterministic actions on same observation
    dummy_obs = trainer1.current_obs
    with torch.no_grad():
        a1, _, _, v1, u1 = trainer1.model.get_action_and_value(dummy_obs, deterministic=True)
        a2, _, _, v2, u2 = trainer2.model.get_action_and_value(dummy_obs, deterministic=True)

    assert torch.allclose(a1, a2, atol=1e-6), "Loaded model continuous action does not match saved model!"
    assert torch.allclose(v1, v2, atol=1e-6), "Loaded model value does not match saved model!"
    assert torch.allclose(u1, u2, atol=1e-6), "Loaded model pointer logits do not match saved model!"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this runner")
def test_rtx3050_vram_and_throughput_budget():
    """32 parallel environments on CUDA must strictly stay below 2800 MB (< 2.8 GB) VRAM budget."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cfg = _make_test_cfg(
        **{
            "environment.num_parallel_envs": 32,
            "training.rollout_steps": 64,
            "training.ppo_epochs": 2,
            "training.num_minibatches": 4,
            "training.device": "cuda",
            "training.mixed_precision_amp": True,
        }
    )

    trainer = MAPPOTrainer(cfg=cfg, device="cuda", use_curriculum=True)
    rollout_stats = trainer.collect_rollout()
    ppo_stats = trainer.ppo_update()

    peak_bytes = torch.cuda.max_memory_allocated()
    peak_mb = peak_bytes / (1024 * 1024)

    # Acceptance criterion: Strictly < 2800 MB (< 2.8 GB)
    print(f"\n[RTX 3050 Benchmark] E=32 Peak VRAM: {peak_mb:.2f} MB | Throughput: {rollout_stats['sps']:.1f} SPS")
    assert peak_mb < 2800.0, f"Peak VRAM {peak_mb:.1f} MB exceeds 2800 MB RTX 3050 budget!"
    assert rollout_stats["sps"] > 100, f"SPS unexpectedly low: {rollout_stats['sps']:.1f}"
    assert torch.isfinite(torch.tensor(ppo_stats["policy_loss"]))
