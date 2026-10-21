# SWARMFORGE: Multi-Agent Swarm & Intercept Coordination Simulator

[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2Bcu126-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900.svg?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-ParallelEnv-blueviolet.svg?style=flat)](https://pettingzoo.farama.org/)
[![Tests](https://img.shields.io/badge/Tests-109%20Passed-brightgreen.svg?style=flat)](tests/)
[![Platform](https://img.shields.io/badge/Platform-WSL2%20Ubuntu%20%7C%20Win11-orange.svg?style=flat)](https://ubuntu.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg?style=flat)](LICENSE)

A high-performance, abstract mathematical simulation where $N$ defending autonomous drones coordinate in real time to track, prioritize, and intercept $M$ incoming dynamic threats protecting High-Value Assets (HVAs). 

Built from the ground up for vectorized PyTorch GPU physics, intra-step continuous Closest Point of Approach (CPA) anti-tunneling, multi-target 3D Kalman tracking, classical guidance/allocation baselines, cross-attention Multi-Agent Reinforcement Learning (Attention-MAPPO), and real-time Three.js 3D instanced tactical visualization.

> **Research & Safety Boundary**: Purely synthetic, abstract mathematical simulation focusing on multi-agent optimization, spatial geometry, and resource allocation. No real-world operational hardware, military seekers, or weapon systems are modeled or interfaced.

---

## 3D Tactical Airspace Overview

```text
                                  [ AIRSPACE VOLUME: 10 km x 10 km x 3 km ]
                          
         Incoming Maneuvering Threats (M)                     Defending Swarm (N)
     [Ballistic / Dives / Evasive Weave]                 [Interceptors & Radar Scouts]
               \         |         /                                   ^
                \        |        /                                    |
                 v       v       v                           [ Sensor Table / Tracks ]
              +---------------------+                        +-----------------------+
              |   Threat Salvo M    |                        | Interceptors (80% N)  |
              |   (v <= 450 m/s)    |                        |   v <= 350 m/s, 12G   |
              +---------------------+                        | Scouts (20% N)        |
                         |                                   |   Wide FOV, 0.4x Noise|
                         v                                   +-----------------------+
           [ Continuous Swept-Sphere CPA ]                               |
              (R_int = 25m, Anti-Tunnel)                                 |
                         |                                               v
                         +---------------------> * <---------------------+
                                            Intercepts!
                                                 |
                                       [ Defended HVAs (P) ]
                                      Ground Coordinates (x, y, 0)
```

---

## System Architecture

```mermaid
flowchart TB
    subgraph Scenario["1. Scenario & Salvo Generator"]
        direction TB
        SG["Procedural Spawn Engine<br/>• Saturation, Pincer & Waves<br/>• Maneuvering Ballistic / Dives / Weaves<br/>• Decoys & Spawn Staggers"]
    end

    subgraph Core["2. Vectorized Swarm Environment (PyTorch GPU)"]
        direction TB
        Kin["3D G-Limited Kinematics<br/>[E, N, 3] & [E, M, 3]<br/>Trapezoidal Integration + Drag"]
        CPA["Continuous-Time CPA Engine<br/>Intra-step Swept-Sphere Intercepts<br/>Friendly Collision Detection"]
        Kalman["Vectorized 3D Kalman Tracker<br/>9-State Const-Accel Filter [p, v, a]<br/>Joseph Stabilized Covariance"]
        Kin --> CPA --> Kalman
    end

    subgraph Allocation["3. Allocation & Guidance Baselines"]
        direction TB
        CostMat["Hybrid Cost Matrix C_ij<br/>t_go + ZEM + Urgency - Hysteresis - u_ij"]
        Alloc["Exclusive Ownership Solvers<br/>• Greedy O(N·M)<br/>• Hungarian Bipartite (Exact Optimal)<br/>• GPU-Vectorized Auction"]
        Guidance["Guidance & Safety Shields<br/>• 3D True Proportional Navigation (TPN)<br/>• Control Barrier Function (CBF) Shield"]
        CostMat --> Alloc --> Guidance
    end

    subgraph MARL["4. MARL & PettingZoo Interface (Deliverable 1)"]
        direction TB
        Obs["Permutation-Invariant Observations<br/>• Ego Token [E, N, 10]<br/>• Top-K Threats [E, N, 15, 15]<br/>• Top-K Allies [E, N, 10, 9]<br/>• Global State [E, 19]"]
        Rew["Potential-Based Reward Shaping<br/>Phi_i(s) = -w_zem·ZEM - w_dist·r<br/>Event Bonuses & Collision Penalties"]
        Curric["3-Stage Curriculum Manager<br/>Stage 1 (2D) ➔ Stage 2 (3D) ➔ Stage 3 (Fog)<br/>Rolling 50-Episode Win Rate Gates"]
        PZ["PettingZoo ParallelEnv API<br/>SwarmParallelEnv Wrapper"]
        Obs --> PZ
        Rew --> PZ
        Curric --> PZ
    end

    subgraph Presentation["5. Evaluation & 3D Visualization"]
        direction TB
        Tele["FastAPI / WebSocket Binary Server<br/>Packed Float32Array @ 60 Hz"]
        Viewer["Three.js 3D Instanced Client<br/>THREE.InstancedMesh + Trails + HUD"]
        Tele --> Viewer
    end

    Scenario --> Core
    Core --> Allocation
    Core --> MARL
    MARL --> Core
    Core --> Presentation
```

---

## Permutation-Invariant Set Observations (Spec Section 4.2)

Instead of a fixed flat array that breaks when swarm sizes change, each drone observes structured sets of ego, threat, and ally tokens padded and masked with zero host synchronization:

```mermaid
flowchart LR
    subgraph InputTokens["Structured Observation Sets"]
        direction TB
        Ego["Ego Token (10)<br/>p/L_max, v/v_max, role, active, assigned, t_norm"]
        Threats["Top-K_t Threat Tokens (15)<br/>r, v_rel, Vc, t_go, ZEM, z_ij, other_assigned, urgency, tr_P"]
        Allies["Top-K_a Ally Tokens (9)<br/>r_ik, v_rel_ik, target_embed_k"]
    end

    subgraph AttentionNet["Cross-Attention Network (<45k Params)"]
        direction TB
        Q["Linear + LayerNorm ➔ Query Q_i (1x64)"]
        KV["Shared MLPs ➔ Keys & Values (K_t + K_a x 64)"]
        Attn["Multi-Head Cross-Attention (2 heads, d=64)"]
        Q --> Attn
        KV --> Attn
    end

    subgraph Outputs["Hybrid Action Outputs"]
        direction TB
        ResA["Residual Acceleration Head<br/>Δa_RL in [-a_res, a_res]^3"]
        CoordLogits["Coordination Logit Head<br/>Priority adjustments u_ij for Allocator"]
    end

    Ego --> Q
    Threats --> KV
    Allies --> KV
    Attn --> ResA
    Attn --> CoordLogits
```

### Observation Feature Breakdown

| Token Type | Shape | Features | Description |
| :--- | :--- | :--- | :--- |
| **Ego Token** | `[E, N, 10]` | `p_i / L_max` (3), `v_i / v_max` (3), `role_i` (1), `active_i` (1), `has_assignment_i` (1), `t / T_max` (1) | Normalized self-state and active status. Rows are zeroed out if drone is inactive. |
| **Threat Tokens** | `[E, N, K_t, 15]` | `r_ij / L_max` (3), `v_rel / 2v_max` (3), `Vc / 2v_max` (1), `t_go / T_max` (1), `ZEM / L_max` (3), `z_ij` (1), `other_assigned` (1), `urgency` (1), `tr_P` (1) | Top-$K_t$ nearest visible threats ranked by distance. Masked slots are strictly `0.0`. |
| **Ally Tokens** | `[E, N, K_a, 9]` | `(p_k - p_i) / L_max` (3), `(v_k - v_i) / v_max` (3), `(p_target_k - p_i) / L_max` (3) | Top-$K_a$ nearest active friendly drones $k \neq i$ with target vector embeddings. |
| **Global State** | `[E, 19]` | `mean(ego)` (10), `mean(p_threats)` (3), `mean(v_threats)` (3), `frac_active` (1), `frac_live` (1), `t_norm` (1) | Centralized state vector for CTDE critic value estimation. |

---

## Potential-Based Reward Shaping & Curriculum Pipeline

```mermaid
stateDiagram-v2
    direction TB
    [*] --> Stage1
    
    state Stage1 {
        direction LR
        S1_Desc: Foundational Intercept & Deconfliction<br/>• 6 Interceptors, 0 Scouts, 8 Missiles<br/>• 2D Mode (z=0), No Weave, No Decoys, sigma=0
    }
    
    Stage1 --> Stage2: Rolling 50-Episode Win Rate >= 90%
    
    state Stage2 {
        direction LR
        S2_Desc: 3D Maneuvering Salvage & Aerodynamic Drag<br/>• 12 Interceptors, 3 Scouts, 25 Missiles<br/>• 3D Volume, Sinusoidal Weave, Drag cd=0.02
    }
    
    Stage2 --> Stage3: Rolling 50-Episode Win Rate >= 85%
    
    state Stage3 {
        direction LR
        S3_Desc: Full Swarm Saturation & Sensor Fog<br/>• 16 Interceptors, 4 Scouts, 30 Missiles<br/>• 15% Decoys, Kalman Noise sigma=15m, Drag cd=0.015
    }
    
    Stage3 --> [*]: Capped at Stage 3
```

### Reward Formulation (Spec Section 6.6)

$$r_i(t) = r_{intercept} + r_{team} - r_{leaker} + r_{shaping} - r_{duplicate} - r_{collision} - r_{effort}$$

1. **Assigned Intercept**: $+10.0$ for drone $i$ intercepting its assigned target.
2. **Team Intercept Bonus**: $+5.0$ per real intercepted missile to all active swarm defenders.
3. **HVA Defense Penalty**: $-15.0$ per real missile impacting a defended asset.
4. **Potential-Based Shaping**:
   $$\Phi_i(s_t) = -w_{zem} \frac{\|ZEM_{i, j^*}\|_2}{L_{max}} - w_{dist} \frac{\|r_{i, j^*}\|_2}{L_{max}}, \quad r_{shaping} = \gamma \Phi_i(s_{t+1}) - \Phi_i(s_t)$$
   Guarantees optimal policy invariance; closed trajectories have zero net gain, preventing circling exploits.
5. **Penalties**: Duplicate assignment violation ($-2.0$ if $\sum_i z_{ij} > K_j$), friendly collision ($-10.0$ if $d_{ik} \le 2 R_{col}$), and control effort ($-0.01 \|\Delta a_i^{RL}\|^2 \Delta t$).

---

## Hardware Profile & Memory Constraints

Optimized specifically for consumer laptops with tight VRAM budgets:
- **Host System**: Windows 11 + WSL2 Ubuntu (Python 3.12, PyTorch CUDA 12.6)
- **Target GPU**: NVIDIA GeForce RTX 3050 Laptop (4 GB GDDR6 VRAM)
- **VRAM Budget Rule**: Strictly constrained to $\le 3.0$ GB VRAM allocation to ensure zero display driver evictions alongside desktop applications.

| Operating Mode | Active Entities | Recommended Workload | Peak VRAM | Primary Bottleneck |
| :--- | :--- | :--- | :--- | :--- |
| **Development & Tests** | $N=6$, $M=8$ | 1–4 Vectorized Envs | < 350 MB | Instantaneous (<1s) test suite |
| **RL Training (Headless)** | $N=20$, $M=30$ | 16–32 Vectorized GPU Envs | < 2.4 GB | Attention activations & backward pass |
| **Live 3D Telemetry** | $N=60$, $M=100$ | 1 GPU Env @ 60 FPS | < 1.2 GB | WebSocket serialization & Three.js draw calls |
| **Stress Benchmark** | $N=300$, $M=500$ | 1 Env (Inference mode) | < 2.9 GB | $O(N \cdot M)$ pairwise distance matrix |

---

## Repository Structure

```text
SWARMFORGE/
├── configs/                        # YAML scenario & training configurations
│   ├── default_sim.yaml            # Default baseline simulation configuration
│   ├── train_mappo_rtx3050.yaml    # MAPPO training tuned for RTX 3050 4GB
│   └── stress_test_300v500.yaml    # High-density stress test (300 vs 500)
├── src/                            # Core simulation & learning modules
│   ├── config.py                   # YAML config loader with recursive inheritance
│   ├── envs/                       # Vectorized simulation environments & MARL APIs
│   │   ├── vectorized_swarm_env.py # PyTorch batched GPU simulation core
│   │   ├── observations.py         # Permutation-invariant set observations
│   │   ├── rewards.py              # Multi-component potential-based rewards
│   │   ├── curriculum_manager.py   # 3-stage rolling evaluation curriculum
│   │   ├── pettingzoo_wrapper.py   # PettingZoo ParallelEnv wrapper (Deliverable 1)
│   │   ├── scenario_generator.py   # Procedural salvo & formation spawning
│   │   └── threat_models.py        # Ballistic, diving, and sinusoidal weaving threats
│   ├── physics/                    # Vectorized 3D dynamics & collision detection
│   │   ├── kinematics_3d.py        # G-limited 3D velocity & position integration
│   │   ├── cpa_collisions.py       # Continuous intra-step CPA swept-spheres
│   │   └── guidance_laws.py        # 3D True Proportional Navigation (TPN) & CBF
│   ├── estimation/                 # Multi-target Kalman tracking & sensor noise
│   │   └── kalman_tracker.py       # Vectorized 9-state linear Kalman filter
│   ├── baselines/                  # Classical allocation & guidance baselines
│   │   ├── baseline_controller.py  # Closed-loop baseline controller
│   │   ├── greedy_allocator.py     # Vectorized greedy allocator
│   │   ├── hungarian_allocator.py  # Optimal bipartite matching allocator
│   │   └── auction_allocator.py    # 100% GPU-vectorized auction allocator
│   ├── learning/                   # Attention MAPPO agent & actor-critic networks
│   ├── evaluation/                 # Metrics, win-rate trackers, benchmarks
│   └── server/                     # FastAPI / WebSocket binary telemetry server
├── viewer3d/                       # Three.js 3D web client
├── tests/                          # Pytest test suite (109 passed tests)
├── scripts/                        # Operational & validation scripts
│   ├── setup_wsl.sh                # Automated WSL2 environment setup
│   └── check_gpu.py                # GPU verification & matrix multiplication benchmark
├── runs/                           # Experiment logs, checkpoints, telemetry
├── PROJECT_CONTEXT.md              # Architectural constraints & milestone roadmap
├── requirements.txt                # Pinned Python dependencies
├── pytest.ini                      # Pytest configuration
└── README.md                       # Project documentation
```

---

## Quickstart

### 1. Environment Setup (WSL2 Ubuntu)
```bash
# Automated environment setup
bash scripts/setup_wsl.sh
```

### 2. Verify GPU & VRAM Allocation
```bash
.venv/bin/python scripts/check_gpu.py
```

### 3. Run Test Suite
```bash
.venv/bin/pytest -q
```
*Current test status: **109 passed tests** across CPU and CUDA backends.*

### 4. PettingZoo ParallelEnv Usage

```python
from src.config import load_config
from src.envs.pettingzoo_wrapper import SwarmParallelEnv

cfg = load_config("configs/default_sim.yaml")
env = SwarmParallelEnv(cfg, allocator="hungarian", use_cbf=True)

observations, infos = env.reset(seed=42)

for step in range(100):
    # Sample multi-agent actions (residual acceleration + coordination logits)
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    if len(env.agents) == 0:
        break
```

---

## Development Milestones

- [x] **Deliverable 1: Vectorized Environment Core & PettingZoo API**
- [x] **M0: Environment & GPU Setup** (WSL2 Ubuntu, CUDA 12.6, configs, tests, baseline benchmarks)
- [x] **M1: Vectorized 3D Physics & CPA Anti-Tunneling**
- [x] **M2: Classical Baselines** (Greedy, Hungarian, Hybrid Auction + TPN)
- [x] **M3: Sensor Noise & 3D Kalman Filtering**
- [ ] **M4: Cross-Attention MAPPO & Curriculum Training** *(In Progress)*
- [ ] **M5: Three.js 3D Instanced Tactical Viewer**
- [ ] **M6: Ablations, Stress Testing & Research Report**

---

## License

This project is licensed under the [MIT License](LICENSE).
