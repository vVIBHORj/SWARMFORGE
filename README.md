# SWARMFORGE: Multi-Agent Swarm & Intercept Coordination Simulator

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2Bcu126-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Platform](https://img.shields.io/badge/Platform-WSL2%20Ubuntu-orange.svg)](https://ubuntu.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

A high-performance, abstract mathematical simulation where $N$ defending autonomous drones coordinate to intercept $M$ incoming dynamic threats protecting High-Value Assets (HVAs). 

Built from the ground up for vectorized GPU physics, classical guidance/allocation baselines, cross-attention Multi-Agent PPO (MAPPO), and real-time Three.js 3D telemetry streaming.

> **Note**: This is a purely mathematical, synthetic simulation. No real hardware, military, or weapon interfaces are modeled or supported.

---

## Key Capabilities

- **Vectorized PyTorch GPU Physics**: Fully batched simulations across parallel environments (`num_parallel_envs`). State tensors strictly shaped `[E, N, ...]` (defenders) and `[E, M, ...]`, with boolean active masks and zero entity-level Python loops.
- **Guidance & Baseline Allocation**:
  - True Proportional Navigation (TPN) with configurable navigation gains $N$.
  - Classical target allocators: Greedy, Hungarian matching, and Hybrid Auction + Attention with hysteresis bonuses.
- **MARL Cross-Attention Coordination**:
  - Centralized Training with Decentralized Execution (CTDE) attention-based MAPPO.
  - Multi-head cross-attention across defender and missile observation tokens.
  - Mixed-precision AMP (`float32` physics, `bfloat16` for neural network forward/backward passes).
- **Sensor Estimation & Noise**:
  - Configurable spherical/Cartesian measurement noise with 3D Kalman filtering.
- **Telemetry & 3D Visualization**:
  - Lightweight FastAPI/WebSocket server streaming packed binary float32 state packets at 60 FPS to an interactive Three.js 3D web viewer.

---

## Hardware Profile & Memory Constraints

Optimized specifically for consumer laptops with tight VRAM budgets:
- **Host OS**: Windows 11 + WSL2 Ubuntu (Python 3.12, PyTorch CUDA 12.6)
- **Target GPU**: NVIDIA GeForce RTX 3050 Laptop (4 GB VRAM)
- **VRAM Budget Rule**: Strictly constrained to $\le 3.0$ GB VRAM allocation to ensure stability alongside desktop rendering.

---

## Repository Structure

```text
SWARMFORGE/
├── configs/                    # YAML scenario & training configurations
│   ├── default_sim.yaml        # Default baseline simulation configuration
│   ├── train_mappo_rtx3050.yaml# MAPPO training tuned for RTX 3050 4GB
│   └── stress_test_300v500.yaml# High-density stress test (300 vs 500)
├── src/                        # Core simulation & learning modules
│   ├── config.py               # YAML config loader with recursive inheritance
│   ├── envs/                   # Gymnasium / PettingZoo swarm environments
│   ├── physics/                # Vectorized 3D dynamics & CPA computation
│   ├── estimation/             # Sensor noise & Kalman filters
│   ├── baselines/              # Greedy / Hungarian / Auction / PN baselines
│   ├── learning/               # Attention MAPPO agent & actor-critic networks
│   ├── evaluation/             # Metrics, win-rate trackers, benchmarks
│   └── server/                 # FastAPI / WebSocket binary telemetry server
├── viewer3d/                   # Three.js 3D web client
│   └── src/                    # Frontend rendering code
├── tests/                      # Pytest unit & integration test suites
├── scripts/                    # Operational & validation scripts
│   ├── setup_wsl.sh            # Automated WSL2 environment setup
│   └── check_gpu.py            # GPU verification & matrix multiplication benchmark
├── runs/                       # Experiment logs, checkpoints, telemetry
├── PROJECT_CONTEXT.md          # Architectural constraints & milestone roadmap
├── requirements.txt            # Python dependencies (NumPy, SciPy, Numba, etc.)
├── pytest.ini                  # Pytest configuration
└── README.md                   # Project documentation
```

---

## Quickstart (WSL2 Ubuntu)

### 1. Environment Setup
Run the automated environment setup script inside your WSL Ubuntu terminal. It will create a Python 3.12 `.venv`, install PyTorch with CUDA 12.6, and resolve all dependencies:

```bash
bash scripts/setup_wsl.sh
```

### 2. Activate & Validate GPU
Activate the virtual environment and run the hardware validation benchmark:

```bash
source .venv/bin/activate
python scripts/check_gpu.py
```

This verifies CUDA availability, VRAM headroom, `bfloat16` support, and executes a $2048 \times 2048$ GPU matrix multiplication benchmark written to `runs/baseline_gpu.json`.

### 3. Run Test Suite
Validate configuration parsing and schema validation rules:

```bash
pytest -q
```

---

## Development Milestones

- [x] **M0: Environment & GPU Setup** (WSL2 Ubuntu, CUDA 12.6, configs, tests, baseline benchmarks)
- [ ] **M1: Vectorized 3D Physics & CPA**
- [ ] **M2: Classical Baselines** (Greedy, Hungarian, Hybrid Auction + TPN)
- [ ] **M3: Sensor Noise & Kalman Filtering**
- [ ] **M4: Cross-Attention MAPPO & Curriculum Training**
- [ ] **M5: Three.js 3D Telemetry Viewer**
- [ ] **M6: Ablations, Stress Testing & Release**
