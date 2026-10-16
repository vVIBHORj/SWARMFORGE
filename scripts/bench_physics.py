"""Benchmark physics module (kinematics, pairwise CPA, friendly collision) on GPU."""

import json
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.physics.cpa_collisions import friendly_collision_mask, pairwise_cpa
from src.physics.kinematics_3d import G0, step_kinematics


def benchmark_fn(fn, warmup: int = 5, iterations: int = 50) -> float:
    """Run warmup and timed iterations with CUDA synchronization, returning mean ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    return (duration / iterations) * 1000.0


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Physics benchmark requires GPU.")
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Running physics benchmark on: {gpu_name}")

    configs = [
        {"name": "Stress Test (E=1, N=300, M=500)", "E": 1, "N": 300, "M": 500},
        {"name": "Standard Train (E=16, N=60, M=100)", "E": 16, "N": 60, "M": 100},
        {"name": "Batched Small (E=32, N=25, M=40)", "E": 32, "N": 25, "M": 40},
    ]

    dt = 0.05
    drone_a_max = 12.0 * G0
    drone_v_max = 350.0
    drone_drag = 0.015
    drone_r_col = 1.5

    missile_a_max = 25.0 * G0
    missile_v_max = 800.0
    missile_drag = 0.005

    results = []

    for cfg in configs:
        E, N, M = cfg["E"], cfg["N"], cfg["M"]
        print(f"\n--- {cfg['name']} ---")

        torch.cuda.reset_peak_memory_stats()

        # Drones state
        p_d = torch.randn(E, N, 3, device=device, dtype=torch.float32) * 500.0
        v_d = torch.randn(E, N, 3, device=device, dtype=torch.float32) * 100.0
        a_d = torch.randn(E, N, 3, device=device, dtype=torch.float32) * 50.0
        active_d = torch.ones(E, N, device=device, dtype=torch.bool)

        # Missiles state
        p_m = torch.randn(E, M, 3, device=device, dtype=torch.float32) * 1000.0
        v_m = torch.randn(E, M, 3, device=device, dtype=torch.float32) * 200.0
        a_m = torch.randn(E, M, 3, device=device, dtype=torch.float32) * 100.0
        active_m = torch.ones(E, M, device=device, dtype=torch.bool)

        # Precompute next positions for CPA benchmarks
        p_d_next, _, _ = step_kinematics(p_d, v_d, a_d, dt, drone_a_max, drone_v_max, drone_drag, active_d)
        p_m_next, _, _ = step_kinematics(p_m, v_m, a_m, dt, missile_a_max, missile_v_max, missile_drag, active_m)

        # 1. Step kinematics (drones)
        t_drones = benchmark_fn(
            lambda: step_kinematics(p_d, v_d, a_d, dt, drone_a_max, drone_v_max, drone_drag, active_d)
        )
        print(f"step_kinematics (drones [{E},{N},3]): {t_drones:.3f} ms")

        # 2. Step kinematics (missiles)
        t_missiles = benchmark_fn(
            lambda: step_kinematics(p_m, v_m, a_m, dt, missile_a_max, missile_v_max, missile_drag, active_m)
        )
        print(f"step_kinematics (missiles [{E},{M},3]): {t_missiles:.3f} ms")

        # 3. Pairwise CPA (drones vs missiles)
        t_cpa = benchmark_fn(
            lambda: pairwise_cpa(p_d, p_d_next, p_m, p_m_next)
        )
        print(f"pairwise_cpa (drones x missiles [{E},{N},{M}]): {t_cpa:.3f} ms")

        # 4. Friendly collision mask (drones)
        t_friendly = benchmark_fn(
            lambda: friendly_collision_mask(p_d, p_d_next, drone_r_col, active_d)
        )
        print(f"friendly_collision_mask (drones [{E},{N},{N}]): {t_friendly:.3f} ms")

        # Full physics step
        def full_step():
            p_d1, v_d1, _ = step_kinematics(p_d, v_d, a_d, dt, drone_a_max, drone_v_max, drone_drag, active_d)
            p_m1, v_m1, _ = step_kinematics(p_m, v_m, a_m, dt, missile_a_max, missile_v_max, missile_drag, active_m)
            pairwise_cpa(p_d, p_d1, p_m, p_m1)
            friendly_collision_mask(p_d, p_d1, drone_r_col, active_d)

        t_total = benchmark_fn(full_step)
        print(f"total physics step: {t_total:.3f} ms")

        max_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak GPU memory: {max_mem_mb:.2f} MB")

        results.append({
            "config": cfg,
            "step_kinematics_drones_ms": round(t_drones, 4),
            "step_kinematics_missiles_ms": round(t_missiles, 4),
            "pairwise_cpa_ms": round(t_cpa, 4),
            "friendly_collision_mask_ms": round(t_friendly, 4),
            "total_step_ms": round(t_total, 4),
            "peak_memory_mb": round(max_mem_mb, 2),
        })

    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / "bench_physics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"gpu_name": gpu_name, "benchmarks": results}, f, indent=2)
    print(f"\nSaved benchmark results to {out_path}")


if __name__ == "__main__":
    main()
