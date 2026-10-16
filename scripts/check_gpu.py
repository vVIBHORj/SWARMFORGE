import json
from pathlib import Path
import sys
import time

try:
    import torch
except ImportError:
    print("Error: PyTorch is not installed.")
    sys.exit(1)

try:
    import numba
except ImportError:
    numba = None

try:
    import scipy
except ImportError:
    scipy = None


def main():
    print(f"Python version: {sys.version.splitlines()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")

    if not cuda_available:
        print("Notice: CUDA is not available on this system. Exiting gracefully.")
        sys.exit(0)

    gpu_name = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    total_vram_mb = total_bytes / (1024 ** 2)
    free_vram_mb = free_bytes / (1024 ** 2)
    bf16_supported = torch.cuda.is_bf16_supported()
    numba_ver = numba.__version__ if numba else "not installed"
    scipy_ver = scipy.__version__ if scipy else "not installed"

    print(f"GPU Name: {gpu_name}")
    print(f"Total VRAM: {total_vram_mb:.2f} MB")
    print(f"Free VRAM: {free_vram_mb:.2f} MB")
    print(f"bfloat16 supported: {bf16_supported}")
    print(f"Numba version: {numba_ver}")
    print(f"SciPy version: {scipy_ver}")

    # Benchmark: 2048x2048 matmul on GPU
    print("Running 2048x2048 matmul benchmark on GPU...")
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
    y = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        _ = torch.matmul(x, y)
    torch.cuda.synchronize()

    iterations = 20
    start = time.perf_counter()
    for _ in range(iterations):
        _ = torch.matmul(x, y)
    torch.cuda.synchronize()
    duration_s = time.perf_counter() - start
    avg_matmul_ms = (duration_s / iterations) * 1000.0

    print(f"2048x2048 matmul average time: {avg_matmul_ms:.3f} ms")

    baseline_data = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "total_vram_mb": round(total_vram_mb, 2),
        "free_vram_mb": round(free_vram_mb, 2),
        "bfloat16_supported": bf16_supported,
        "numba_version": numba_ver,
        "scipy_version": scipy_ver,
        "matmul_2048_avg_ms": round(avg_matmul_ms, 3),
    }

    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / "baseline_gpu.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2)

    print(f"Wrote baseline GPU report to {out_path}")


if __name__ == "__main__":
    main()
