#!/usr/bin/env python3
"""Training entrypoint for CTDE MAPPO on SWARMFORGE.

Features:
- Config loading with CLI overrides.
- TensorBoard logging & JSONL streaming metrics.
- Automated best/latest checkpoint tracking.
- Fast smoke-test mode for CI/smoke validation.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get, load_config
from src.learning.mappo_trainer import MAPPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train MAPPO on SWARMFORGE.")
    parser.add_argument("--config", type=str, default="configs/train_mappo_rtx3050.yaml", help="Path to YAML config.")
    parser.add_argument("--total-steps", type=int, default=None, help="Override total environment steps.")
    parser.add_argument("--updates", type=int, default=None, help="Explicit number of rollout updates to run.")
    parser.add_argument("--num-envs", type=int, default=None, help="Override number of parallel envs.")
    parser.add_argument("--rollout-steps", type=int, default=None, help="Override rollout steps per update.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda' or 'cpu').")
    parser.add_argument("--model-type", type=str, default="attention", choices=["attention", "mlp"], help="Actor-critic architecture.")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum manager progression.")
    parser.add_argument("--save-dir", type=str, default="runs/mappo", help="Directory for logs and checkpoints.")
    parser.add_argument("--smoke-test", action="store_true", help="Quick 2-update verification run.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.seed is not None:
        cfg["environment"]["seed"] = args.seed
    if args.num_envs is not None:
        cfg["environment"]["num_parallel_envs"] = args.num_envs
    if args.rollout_steps is not None:
        cfg["training"]["rollout_steps"] = args.rollout_steps
    if args.total_steps is not None:
        cfg["training"]["total_env_steps"] = args.total_steps
    if args.device is not None:
        cfg["training"]["device"] = args.device

    # Smoke test settings
    if args.smoke_test:
        cfg["environment"]["num_parallel_envs"] = 4
        cfg["training"]["rollout_steps"] = 16
        cfg["training"]["ppo_epochs"] = 2
        cfg["training"]["num_minibatches"] = 2
        target_updates = 2
    else:
        target_updates = args.updates

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard initialization
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(save_dir))
    except Exception:
        pass

    metrics_file = save_dir / "metrics.jsonl"

    print("=" * 78)
    print("SWARMFORGE MAPPO TRAINING")
    print(f"Config: {args.config} | Model: {args.model_type} | Device: {cfg.get('training', {}).get('device', 'auto')}")
    print(f"Save Directory: {save_dir.resolve()}")
    if args.smoke_test:
        print("[SMOKE TEST MODE] Running 2 updates with E=4, T=16")
    print("=" * 78)

    trainer = MAPPOTrainer(
        cfg=cfg,
        device=args.device,
        use_curriculum=not args.no_curriculum,
        model_type=args.model_type,
        log_dir=str(save_dir),
    )

    best_intercept_rate = -1.0
    total_target_steps = int(get(cfg, "training.total_env_steps", 10_000_000))
    start_total_time = time.time()

    try:
        while True:
            # Check stopping criteria
            if target_updates is not None and trainer.update_idx >= target_updates:
                break
            if target_updates is None and trainer.total_env_steps >= total_target_steps:
                break

            # 1. Rollout Collection
            rollout_stats = trainer.collect_rollout()

            # 2. PPO Gradient Updates
            ppo_stats = trainer.ppo_update()

            stage_num = trainer.curriculum.current_stage if trainer.curriculum else 1

            # Combined metrics
            combined = {
                "update": trainer.update_idx,
                "total_env_steps": trainer.total_env_steps,
                "stage": stage_num,
                **rollout_stats,
                **ppo_stats,
            }

            # TensorBoard logging
            if writer is not None:
                for k, v in combined.items():
                    if isinstance(v, (int, float)):
                        writer.add_scalar(k, v, global_step=trainer.total_env_steps)

            # JSONL metrics logging
            with open(metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(combined) + "\n")

            # Checkpoints
            trainer.save_checkpoint(str(save_dir / "latest.pt"), extra_meta=combined)
            if rollout_stats["mean_intercept_rate"] > best_intercept_rate:
                best_intercept_rate = rollout_stats["mean_intercept_rate"]
                trainer.save_checkpoint(str(save_dir / "best.pt"), extra_meta=combined)

            # Concise terminal progress
            print(
                f"[Update {trainer.update_idx:4d}] steps={trainer.total_env_steps:8d} stage={stage_num} | "
                f"int_rate={rollout_stats['mean_intercept_rate']:5.1%} leak_rate={rollout_stats['mean_leaker_rate']:5.1%} | "
                f"p_loss={ppo_stats['policy_loss']:+7.4f} v_loss={ppo_stats['value_loss']:6.4f} ent={ppo_stats['entropy']:5.3f} | "
                f"SPS={rollout_stats['sps']:6.0f} VRAM={rollout_stats['peak_vram_mb']:5.1f}MB"
            )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving emergency checkpoint...")
        trainer.save_checkpoint(str(save_dir / "interrupted.pt"))

    finally:
        if writer is not None:
            writer.close()

    total_time = time.time() - start_total_time
    print("=" * 78)
    print(f"Training completed in {total_time:.2f}s across {trainer.update_idx} updates ({trainer.total_env_steps} steps).")
    print(f"Best Intercept Rate: {best_intercept_rate:5.1%}")
    print(f"Checkpoints saved to: {save_dir.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    main()
