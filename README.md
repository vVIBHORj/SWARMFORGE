# SWARMFORGE: Multi-Agent Swarm & Intercept Coordination Simulator

[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2Bcu126-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900.svg?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-ParallelEnv-blueviolet.svg?style=flat)](https://pettingzoo.farama.org/)
[![Tests](https://img.shields.io/badge/Tests-151%20Passed-brightgreen.svg?style=flat)](tests/)
[![Milestones](https://img.shields.io/badge/Milestones-M0--M6%20Complete-success.svg?style=flat)](PROJECT_CONTEXT.md)
[![Platform](https://img.shields.io/badge/Platform-WSL2%20Ubuntu%20%7C%20Win11-orange.svg?style=flat)](https://ubuntu.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg?style=flat)](LICENSE)

A high-performance, abstract mathematical simulation where $N$ defending autonomous drones coordinate in real time to track, prioritize, and intercept $M$ incoming dynamic threats protecting High-Value Assets (HVAs). 

Built from the ground up for vectorized PyTorch GPU physics, intra-step continuous Closest Point of Approach (CPA) anti-tunneling, multi-target 3D Kalman tracking, classical guidance/allocation baselines, cross-attention Multi-Agent Reinforcement Learning (Attention-MAPPO), and real-time Three.js 3D instanced tactical visualization.

> **Research & Safety Boundary**: Purely synthetic, abstract mathematical simulation focusing on multi-agent optimization, spatial geometry, and resource allocation. No real-world operational hardware, military seekers, or weapon systems are modeled or interfaced.

---

## 3D Tactical Instanced Viewer

SWARMFORGE includes a zero-build Three.js client connected via binary `Float32Array` WebSockets streaming at 60 Hz. The instanced renderer draws up to 2,000 entities in just **2 GPU draw calls**, featuring architectural facility compounds, dynamic defense domes, continuous time-scale controls, and procedural PBR aerospace geometries:

![3D Tactical Web Viewer](runs/benchmarks/viewer_screenshot.png)

*Live tactical display at `http://localhost:8765`: 5 architectural defended facility compounds (HQ tower, Airbase hangars/runway, Refinery cooling towers/spheres, Radar station with rotating dish, Urban district) with procedural window illumination grids, pulsing defense domes, delta-wing interceptors with assignment lock lines (cyan), hypersonic threats with thruster plumes (crimson), kinetic CPA burst effects, and Obsidian telemetry HUD with interactive playback controls.*

### Key Viewer Features
- **Architectural Defended Facility Compounds (5 HVAs)**: High-Value Assets are rendered as distinct structural complexes rather than simple markers—including Command HQ tower, Airbase hangars with asphalt runway and taxiway, Energy Refinery cooling towers and pressurized spherical tanks, Phased-Array Radar station with rotating dish, and Urban High-Rise District. Compounds feature canvas-generated window illumination grids, flashing red FAA beacons, glowing defense perimeter domes, structural health rings, and real-time damage states with hit counters.
- **PBR Aerospace Geometries**: High-detail procedural instanced geometries featuring delta-wing interceptors with twin canted vertical stabilizers, high-altitude radar scout pickets, hypersonic cruise/ballistic threats with continuous axial roll and radiant thruster exhaust plumes, and electronic warfare decoys.
- **Continuous 60 FPS Interpolation & Working Playback Controls**: Decoupled client-side `PlaybackController` uses an accumulator clock for butter-smooth 60 FPS position/orientation interpolation across variable simulation tick rates and time scales (`0.05x`, `0.1x`, `0.25x`, `0.5x`, `1.0x`, `2.0x`, `4.0x`, `8.0x`), automatic slow-motion during terminal kinetic intercepts, single-step inspection, and instant pause/resume.
- **Atmospheric Environments & Lighting**: Dynamic tactical lighting presets (Night, Dusk, Day) with normal-mapped ground terrain, soft smoke puff particle trails behind incoming threats, and dynamic point-light explosion bursts upon kinetic Closest Point of Approach (CPA) intercepts.
- **Standalone Offline Demo Mode**: Zero-dependency client-side simulation fallback accessible via `?demo=1` or triggered automatically after a 3-second connection timeout, allowing full inspection of swarm dynamics, 3D models, camera presets, and controls without a running Python backend.

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
├── tests/                          # Pytest test suite (151 passed tests)
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

## Quickstart & Operational Commands

> **Important**: Always execute operational commands from the repository root directory (`cd ~/swarmforge`) so the virtual environment `.venv` and modules are properly resolved.

### 1. Environment Verification & Test Suite
```bash
# Ensure working directory is repo root
cd ~/swarmforge

# Verify GPU acceleration and strictly bounded VRAM allocation (< 250 MB)
.venv/bin/python scripts/check_gpu.py

# Run complete pytest validation suite (151 tests passing across CPU/CUDA)
.venv/bin/pytest -q

# Run frontend viewer unit tests (5 Node.js tests passing)
export PATH="$HOME/.local/node/bin:$PATH"
node --test viewer3d/tests/*.test.js
```

### 2. Train Production Cross-Attention MAPPO Checkpoint
```bash
# Train on CUDA with 3-stage automated curriculum and CTDE PPO
.venv/bin/python scripts/train.py --updates 20 --num-envs 16 --rollout-steps 128 --device cuda

# Checkpoints saved to: runs/mappo/best.pt (88.9% peak win rate) and latest.pt
```

### 3. Run Multi-Seed Statistical Benchmark & Ablation Suite
```bash
# Fast smoke-test suite (< 20s, generates all publication plots and markdown tables)
.venv/bin/python scripts/evaluate.py --smoke-test

# Complete 5-seed full-horizon statistical protocol (Small 10v15, Medium 30v50, Large 100v150)
.venv/bin/python scripts/evaluate.py --seeds 5 --episodes-per-seed 4
```

### 4. Launch 3D Tactical Instanced Viewer & WebSocket Telemetry Server
```bash
# Start FastAPI binary WebSocket server at 60 Hz from repo root
PYTHONPATH=. .venv/bin/python scripts/demo_server.py --port 8765

# Open your browser at http://localhost:8765 to access:
# - Three.js InstancedMesh rendering (<2 GPU draw calls for 2,000 entities)
# - 5 Defended facility complexes with structural health and hit counters
# - Interactive Red-Team Salvo Threat Spawner & Trajectory Trails
# - Smooth variable-speed playback (0.05x to 8x) and Auto-Slow-Mo
# - Real-time algorithm hot-swapping (Greedy / Hungarian / Auction / Attention-MAPPO)
#
# Standalone offline demo mode is also available directly at:
# http://localhost:8765/?demo=1
```

---

## 5-Seed Statistical Benchmark & Ablation Results

Evaluated across 5 random seeds `(42, 43, 44, 45, 46)` and 20 full-horizon Monte Carlo episodes per condition:

### Algorithms Evaluated & Accuracy Overview

1. **Greedy + TPN**: Sequential nearest-target allocation combined with 3D True Proportional Navigation. Baseline uncoordinated clustering achieves 40.0%–46.5% intercept rates under dense salvos due to threat overlap and lack of team-level deconfliction.
2. **Hungarian + TPN**: Global optimal bipartite matching via capacity-expanded linear sum assignment. Achieves strong Small/Medium defense coordination (59.3%–65.2% intercept rate).
3. **Auction + TPN**: 100% GPU-vectorized $\epsilon$-complementary slackness auction allocator with zero host syncs. Scales seamlessly up to **82.8% intercept rate** on Large zero-shot scale (100v150).
4. **Flat-MLP MAPPO**: CTDE reinforcement learning baseline using flattened fixed-size observation vectors. Achieves 74.4%–82.1% intercept rate.
5. **Cross-Attention MAPPO**: Permutation-invariant attention network with continuous residual steering ($\Delta a_{RL}$) and coordination pointer logits ($u_{ij}$) scattered into the GPU auction allocator. Achieves **88.9% peak win rate** in training and top-tier zero-shot scalability (**81.1%–82.1% intercept rate** on Large 100v150 with ~18 ms step latency).

### 1. Zero-Shot Swarm Scale Generalization Benchmark

| Swarm Scale | Algorithm | Intercept Rate (%) | Leaker Rate (%) | Duplicate Conflict (%) | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Small (10v15) | `greedy` | 46.4 ± 6.3 | 37.0 ± 9.1 | 0.0 ± 0.0 | 12.48 |
| Small (10v15) | `hungarian_tpn` | 60.2 ± 7.1 | 24.3 ± 13.6 | 0.0 ± 0.0 | 10.03 |
| Small (10v15) | `auction_tpn` | 77.5 ± 7.5 | 14.0 ± 8.2 | 0.0 ± 0.0 | 12.85 |
| Small (10v15) | `flat_mlp_mappo` | 79.5 ± 3.6 | 11.7 ± 6.8 | 0.0 ± 0.0 | 16.76 |
| Small (10v15) | `attention_mappo` | 78.5 ± 6.8 | 11.8 ± 8.4 | 0.0 ± 0.0 | 18.30 |
| Medium (30v50) | `greedy` | 40.0 ± 5.1 | 42.5 ± 7.2 | 0.0 ± 0.0 | 20.32 |
| Medium (30v50) | `hungarian_tpn` | 59.3 ± 2.6 | 24.8 ± 5.6 | 0.0 ± 0.0 | 10.23 |
| Medium (30v50) | `auction_tpn` | 74.4 ± 3.4 | 14.5 ± 6.5 | 0.0 ± 0.0 | 13.04 |
| Medium (30v50) | `flat_mlp_mappo` | 74.4 ± 1.7 | 14.4 ± 5.7 | 0.0 ± 0.0 | 16.97 |
| Medium (30v50) | `attention_mappo` | 74.5 ± 3.7 | 14.6 ± 5.6 | 0.0 ± 0.0 | 18.12 |
| Large (100v150) | `greedy` | 46.5 ± 3.0 | 39.5 ± 5.6 | 0.0 ± 0.0 | 44.72 |
| Large (100v150) | `hungarian_tpn` | 65.2 ± 2.0 | 22.9 ± 4.2 | 0.0 ± 0.0 | 14.57 |
| Large (100v150) | `auction_tpn` | **82.8 ± 1.4** | **8.7 ± 4.1** | **0.0 ± 0.0** | 13.20 |
| Large (100v150) | `flat_mlp_mappo` | 82.1 ± 3.0 | 8.9 ± 2.2 | 0.0 ± 0.0 | 17.19 |
| Large (100v150) | `attention_mappo` | 81.1 ± 2.7 | 9.5 ± 2.8 | 0.0 ± 0.0 | 18.44 |

![Scale Generalization](runs/benchmarks/scale_generalization.png)

### 2. Component Ablation Study (Medium 30v50 Scale)

| Architecture / Component Variant | Intercept Rate (%) | Friendly Collisions | Duplicate Conflict (%) | Mean ΔV (m/s) |
| :--- | :---: | :---: | :---: | :---: |
| **Full (Cross-Attn + TPN + CBF)** | **74.5 ± 3.7** | **0.0 ± 0.0** | **0.0 ± 0.0** | **1035.0 ± 40.2** |
| **w/o Cross-Attention (Flat-MLP)** | 74.4 ± 1.7 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1036.3 ± 48.9 |
| **w/o CBF Safety Shield** | 69.2 ± 1.9 | **89.0 ± 100.1** | 0.0 ± 0.0 | 986.5 ± 71.5 |
| **w/o Kalman (Raw Noisy Obs)** | **6.5 ± 3.4** | 1.1 ± 0.5 | 0.0 ± 0.0 | **18888.1 ± 8566.9** |
| **w/o RL Residual & Coord Logits** | 74.4 ± 3.4 | 0.1 ± 0.1 | 0.0 ± 0.0 | 1026.6 ± 45.6 |

![Component Ablation Study](runs/benchmarks/ablation_study.png)

### 3. Step Latency vs Intercept Pareto Trade-off

![Pareto Trade-Off](runs/benchmarks/pareto_latency_vs_intercept.png)

- **Throughput & Real-Time Performance**: All baseline algorithms and neural policies execute within **10–19 ms** per step even under the 100 defending drones vs 150 missiles regime, allowing 50–100 Hz closed-loop control.
- **Safety Criticality**: Disabling the Control Barrier Function (`w/o CBF`) causes friendly collision spikes (**89.0 collisions/ep**), validating the necessity of the analytical safety shield.
- **Filtering Under High Sensor Noise**: Operating with raw finite-difference velocity (`w/o Kalman`) degrades intercept success down to **6.5%** and wastes **18,888 m/s $\Delta V$**, proving the critical importance of the 9-state 3D Kalman filter.

---

## Development Milestones (100% Complete)

- [x] **Deliverable 1: Vectorized Environment Core & PettingZoo API**
- [x] **M0: Environment & GPU Setup** (WSL2 Ubuntu, CUDA 12.6, configs, tests, baseline benchmarks)
- [x] **M1: Vectorized 3D Physics & CPA Anti-Tunneling**
- [x] **M2: Classical Baselines** (Greedy, Hungarian, Hybrid Auction + TPN)
- [x] **M3: Sensor Noise & 3D Kalman Filtering**
- [x] **M4: Cross-Attention MAPPO & Curriculum Training** (`runs/mappo/best.pt`, 87.5% win rate)
- [x] **M5: Three.js 3D Instanced Tactical Viewer & WebSocket Server** (60 Hz binary Float32Array protocol)
- [x] **M6: Full-Horizon Engagement Benchmarks, Multi-Seed Statistical Tables & Portfolio Release**

---

## License

This project is licensed under the [MIT License](LICENSE).
