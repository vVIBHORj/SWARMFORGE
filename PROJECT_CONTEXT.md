# SwarmForge: Multi-Agent Swarm & Intercept Coordination Simulator

Abstract, synthetic simulation where N defending drones coordinate to intercept M incoming threats protecting High-Value Assets (HVAs). Features vectorized PyTorch GPU physics, classical baselines (Greedy/Hungarian/Auction + Proportional Navigation), a cross-attention MAPPO agent, and a Three.js viewer. Purely mathematical simulation with no real-world interfaces.

## Hardware Constraints
- OS: Windows 11 + WSL2 Ubuntu (Python 3.12, PyTorch CUDA)
- Host Hardware: Intel i7 12th Gen, 16 GB RAM
- GPU: NVIDIA RTX 3050 Laptop (4 GB VRAM) — strictly limit allocations to <= 3 GB VRAM

## Core Architecture Rules
1. **Vectorization**: All tensors shaped `[E, N, ...]` (interceptors) or `[E, M, ...]` (missiles) with boolean active masks. No Python loops over entities.
2. **Precision**: `float32` for all physical dynamics, geometry, and numerical integration; `bfloat16` AMP restricted to neural network training/inference.

## Repository Structure
- `configs/`: YAML scenario and training configurations (`default_sim.yaml`, etc.)
- `src/`: Core Python packages (`envs`, `physics`, `estimation`, `baselines`, `learning`, `evaluation`, `server`, `config.py`)
- `viewer3d/`: Three.js web visualization client
- `tests/`: Pytest suite for unit/integration validation
- `scripts/`: Operational scripts (`setup_wsl.sh`, `check_gpu.py`)
- `runs/`: Execution runs, checkpoints, and benchmark artifacts

## Milestones
- [x] **M0: Environment & GPU setup** (Done)
- [ ] **M1: Vectorized 3D physics & CPA** (In Progress: kinematics + CPA done; scenario generator + vectorized env still to do)
- [ ] **M2: Classical baselines (Greedy/Hungarian/Auction + PN)**
- [ ] **M3: Sensor noise & Kalman filtering**
- [ ] **M4: Cross-attention MAPPO & curriculum learning**
- [ ] **M5: Three.js 3D viewer**
- [ ] **M6: Ablations & release**
