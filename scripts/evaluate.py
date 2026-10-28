#!/usr/bin/env python3
"""CLI Evaluation and Multi-Seed Ablation Runner for SwarmForge.

Spec Section 12:
- Supports evaluation of classical baselines and trained MAPPO checkpoints.
- Evaluates zero-shot scale generalization (Small, Medium, Large scales).
- Conducts component ablation study (Cross-Attention, CBF, Kalman, RL residual/logits).
- Produces publication-ready PNG figures, JSON results, CSV summary, and Markdown report.
"""

import argparse
from pathlib import Path
import sys
import time

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.evaluation.run_ablations import run_benchmark_suite


def parse_seeds(seed_arg: str):
    """Parse integer count or comma-separated seed list."""
    if "," in seed_arg:
        return tuple(int(s.strip()) for s in seed_arg.split(",") if s.strip())
    try:
        count = int(seed_arg)
        return tuple(range(42, 42 + count))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid seeds specification: '{seed_arg}'. Provide an int count or comma-separated list.")


def main():
    parser = argparse.ArgumentParser(description="SwarmForge Evaluation & Multi-Seed Statistical Benchmark Suite")
    parser.add_argument("--config", type=str, default="configs/default_sim.yaml", help="Path to simulation YAML config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained .pt checkpoint (optional)")
    parser.add_argument("--seeds", type=str, default="5", help="Number of seeds or comma-separated list (e.g. 5 or 42,43,44)")
    parser.add_argument("--episodes-per-seed", type=int, default=20, help="Number of evaluation episodes per seed")
    parser.add_argument("--output-dir", type=str, default="runs/benchmarks", help="Output directory for benchmark artifacts")
    parser.add_argument("--quick", "--smoke-test", dest="quick_mode", action="store_true", help="Run fast smoke test (<15s) across minimal scales")

    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = parse_seeds(args.seeds)

    print("================================================================================")
    print("  SWARMFORGE EVALUATION & STATISTICAL BENCHMARK SUITE")
    print(f"  Mode: {'Quick Smoke-Test (<15s)' if args.quick_mode else 'Full Multi-Seed Protocol'}")
    print(f"  Seeds: {(42, 43) if args.quick_mode else seeds} | Episodes/seed: {2 if args.quick_mode else args.episodes_per_seed}")
    print(f"  Output directory: {args.output_dir}")
    print("================================================================================\n")

    start_time = time.perf_counter()
    res = run_benchmark_suite(
        cfg=cfg,
        seeds=seeds,
        episodes_per_seed=args.episodes_per_seed,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        quick_mode=args.quick_mode,
    )
    elapsed = time.perf_counter() - start_time

    # Print the formatted markdown report to stdout
    print("\n" + res["report_markdown"] + "\n")

    print(f"Benchmark finished in {elapsed:.2f}s.")
    print("Generated artifacts:")
    for name, path in res["artifacts"].items():
        print(f"  - {name}: {path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
