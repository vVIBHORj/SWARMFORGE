"""Benchmark VectorizedSwarmEnv step throughput and memory on GPU."""

import json
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import load_config
from src.envs.vectorized_swarm_env import VectorizedSwarmEnv


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Running env benchmark on: {gpu_name}")

    config_path = Path(__file__).resolve().parent.parent / "configs" / "default_sim.yaml"
    cfg = load_config(config_path)

    configs = [
        {"name": "Stress Test (E=1, N=300, M=500)", "E": 1, "N": 300, "M": 500, "N_int": 240, "N_sct": 60},
        {"name": "Standard Train (E=16, N=60, M=100)", "E": 16, "N": 60, "M": 100, "N_int": 48, "N_sct": 12},
        {"name": "Batched Small (E=32, N=25, M=40)", "E": 32, "N": 25, "M": 40, "N_int": 20, "N_sct": 5},
        {"name": "Lightweight (E=16, N=15, M=25)", "E": 16, "N": 15, "M": 25, "N_int": 12, "N_sct": 3},
    ]

    results = []

    for c in configs:
        E, N, M = c["E"], c["N"], c["M"]
        N_int, N_sct = c["N_int"], c["N_sct"]
        print(f"\n--- {c['name']} ---")

        env = VectorizedSwarmEnv(cfg, num_envs=E, device=device, seed=42)
        env.reset(num_interceptors=N_int, num_scouts=N_sct, num_missiles=M)

        cmd = torch.randn(E, N, 3, device=device, dtype=torch.float32) * 50.0

        # Warmup
        for _ in range(5):
            env.step(cmd)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # Timed 100 steps
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            env.step(cmd)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration_s = time.perf_counter() - start

        mean_ms = (duration_s / iterations) * 1000.0
        env_steps_per_sec = (E * iterations) / duration_s
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

        print(f"Mean step time: {mean_ms:.3f} ms")
        print(f"Environment steps/sec: {env_steps_per_sec:.1f}")
        print(f"Peak GPU memory: {peak_mem_mb:.2f} MB")

        results.append({
            "config": c,
            "mean_step_ms": round(mean_ms, 3),
            "env_steps_per_sec": round(env_steps_per_sec, 1),
            "peak_memory_mb": round(peak_mem_mb, 2),
        })

    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / "bench_env.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"gpu_name": gpu_name, "benchmarks": results}, f, indent=2)

    print(f"\nSaved benchmark report to {out_path}")


if __name__ == "__main__":
    main()
