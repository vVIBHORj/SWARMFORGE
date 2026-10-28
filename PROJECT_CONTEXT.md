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

## Milestones & Deliverables
- [x] **Deliverable 1: Vectorized Env Core + Obs/Rewards/Curriculum/PettingZoo API** (Done)
- [x] **M0: Environment & GPU setup** (Done)
- [x] **M1: Vectorized 3D physics & CPA** (Done)
- [x] **M2: Classical baselines (Greedy/Hungarian/Auction + PN)** (Done)
- [x] **M3: Sensor noise & Kalman filtering** (Done)
- [x] **M4: Cross-attention MAPPO & curriculum learning** (Done)
- [x] **M5: Three.js Instanced 3D Tactical Viewer & Binary WebSocket Server** (Done)
- [x] **M6: Full-Horizon Engagement Benchmarks, Multi-Seed Statistical Tables & Final Release** (100% Complete)

## Env API (`VectorizedSwarmEnv`)
- Public methods:
  - `reset(seed=None, num_interceptors=None, num_scouts=None, num_missiles=None, mode=None, weave_enabled=None, sigma_obs=None) -> None`
  - `reset_envs(env_mask: bool [E]) -> None`
  - `step(drone_accel_cmd: float32 [E,N,3]) -> dict`
  - `spawn_salvo(count=10, speed=None, weave_frequency_rad_s=None, decoy_fraction=None) -> int`: Live red-team threat salvo spawner dynamically expanding `M -> M + count` across env and tracker tensors.
  - `missile_live() -> bool [E,M]`
  - `metrics() -> dict of [E] float tensors`
  - `update_tracker(env_mask=None) -> dict`: Steps multi-target Kalman tracker with current sensor detections, updating `env.track_state`.
- Missile Status Codes:
  - `0`: In flight / unresolved
  - `1`: Intercepted by defending drone
  - `2`: Impacted an HVA
  - `3`: Expired (out of bounds or ground impact)

## Estimation & Kalman Tracker API (`src.estimation`)
- `compute_detections_and_measurements(env, rng=None, full_observability=False) -> dict`:
  - Pairwise drone-to-missile range and sensor FOV gating (interceptor range 4500m, scout range 9000m).
  - Role noise multiplier (`0.4x` for scouts) and distance-scaled noise: $\sigma_{ij} = \sigma_{base} \cdot \text{role\_mult} \cdot (1 + \text{scale} \cdot (d / d_{max}))$.
  - Best team sensor fusion per target: $\sigma_m = \min_i \sigma_{ij}$ `[E, M]`.
  - Returns dict with `z_pos [E,M,3]`, `detected [E,M]`, `sigma_m [E,M]`, and `drone_detects [E,N,M]`.
- `VectorizedKalmanTracker(cfg, num_envs, num_missiles, device)`:
  - Fully batched, axis-decoupled 3D Constant-Acceleration (9-state $[p, v, a]$ per target) Linear Kalman Filter.
  - State tensors: `x_hat [E, M, 3, 3]` (spatial axis $\times$ kinematic order) and `P [E, M, 3, 3]` (isotropic covariance per target).
  - Joseph-form stabilized covariance update ($P_{upd} = (I - KH) P_{pred} (I - KH)^T + K R K^T$) requiring zero matrix inverses.
  - Predict/coasting extrapolation, first-seen initialization, dead-target freezing, and selective env reset without host syncs.
  - Public methods: `reset(env_mask=None)`, `step(z_pos, detected, sigma_m, missile_live, env_mask=None) -> dict`, `trace_P() -> Tensor [E,M]`, `get_estimates() -> dict`.
  - Estimates returned in `track_state`: `pos_est`, `vel_est`, `acc_est`, `trace_P`, `pos_var`, `vel_var`, `acc_var`, `initialized`, `detected`, `steps_since_seen`, `raw_pos`, `raw_vel`.

## Guidance & Baselines API
- `src.physics.guidance_laws`:
  - `engagement_kinematics(drone_pos, drone_vel, missile_pos, missile_vel, missile_accel=None, v_min=20.0, tgo_cap=120.0) -> dict`: Pairwise relative kinematics (`r`, `v_rel`, `dist`, `Vc`, `t_go`, `ZEM`, `zem_norm`, `omega`).
  - `tpn_guidance_accel(drone_pos, drone_vel, missile_pos, missile_vel, assignment_z, missile_accel=None, nav_gain=4.0, v_min=20.0) -> Tensor [E,N,3]`: True Proportional Navigation with forward closing acceleration for assigned interceptors.
  - `cbf_repulsion_accel(drone_pos, drone_active, r_col=5.0, r_safe=40.0, k_cbf=500.0) -> Tensor [E,N,3]`: Control Barrier Function inter-drone collision avoidance shield.
  - `build_cost_matrix(env, prev_z=None, rl_logits=None, weights=None, v_min=20.0, tgo_cap=120.0, invalid_cost=1e6, use_estimated_state=False) -> tuple[Tensor [E,N,M], Tensor [E,N,M]]`: Hybrid assignment cost combining $t_{go}$, ZEM, HVA threat urgency, and commitment hysteresis.
- `src.baselines`:
  - `greedy_assign(cost_matrix, valid_drone, valid_missile, missile_K, invalid_cost=1e6) -> Tensor [E,N,M] bool`: Vectorized greedy assignment.
  - `hungarian_assign(cost_matrix, valid_drone, valid_missile, missile_K, invalid_cost=1e6) -> Tensor [E,N,M] bool`: Optimal bipartite matching via capacity-expanded SciPy linear sum assignment.
  - `auction_assign(cost_matrix, valid_drone, valid_missile, missile_K, num_iters=10, eps=0.05, invalid_cost=1e6) -> Tensor [E,N,M] bool`: 100% GPU-vectorized host-sync-free auction matching.
  - `BaselineController(cfg, allocator="hungarian", use_cbf=True, use_apn=False, use_estimated_state=False)`: Closed-loop multi-agent baseline controller with commitment preservation and TPN guidance. When `use_estimated_state=True`, uses `env.track_state` filtered kinematic estimates.

## Observations, Rewards, Curriculum & PettingZoo API (Deliverable 1)
- `src.envs.observations.build_observations(env, K_threat=15, K_ally=10, use_estimated_state=True) -> dict`:
  - Fully vectorized, host-sync-free permutation-invariant set observation builder.
  - Returns: `ego [E,N,10]`, `threat_tokens [E,N,K_threat,15]`, `threat_mask [E,N,K_threat]`, `threat_indices [E,N,K_threat]`, `ally_tokens [E,N,K_ally,9]`, `ally_mask [E,N,K_ally]`, `ally_indices [E,N,K_ally]`, `drone_mask [E,N]`, and `global_state [E,19]`.
  - Zero-padded and masked with strict 0.0 values across variable swarm sizes and curriculum stages.
- `src.envs.rewards.RewardComputer(cfg, device)`:
  - Vectorized multi-component reward with potential-based shaping $\Phi_i(s_t) = -w_{zem} \frac{\|ZEM_{i,j^*}\|}{L_{max}} - w_{dist} \frac{\|r_{i,j^*}\|}{L_{max}}$.
  - Sub-terms: assigned intercept bonus (+10.0), team intercept bonus (+5.0), HVA leaker penalty (-15.0), friendly collision (-10.0), duplicate allocation penalty (-2.0), control effort penalty ($-0.01 \|a_{effort}\|^2 \Delta t$), and shaping ($r_{shaping} = \gamma \Phi_{t+1} - \Phi_t$).
  - Integrated via `env.compute_rewards(info, prev_z, rl_residual_accel)` with automatic `_prev_phi` state tracking.
- `src.envs.curriculum_manager.CurriculumManager(cfg, initial_stage=1)`:
  - 3-stage curriculum controller tracking rolling 50-episode win rates with automated advancement:
    - Stage 1: 6 interceptors, 8 missiles, 2D, advances at $\ge 90\%$ win rate.
    - Stage 2: 12 interceptors, 3 scouts, 25 missiles, 3D, drag=0.02, weave=True, advances at $\ge 85\%$ win rate.
    - Stage 3: 16 interceptors, 4 scouts, 30 missiles + decoys, sensor noise $\sigma=15m$, caps at Stage 3.
  - Public methods: `get_stage_kwargs() -> dict`, `apply_to_env(env, seed=None)`, `record_episodes(intercept_rates) -> bool`.
## Learning Architecture & GPU Rollout Buffer (Milestone M4 Part 1)
- `src.learning.attention_nets`:
  - `CrossAttentionActorCritic(d_model=64, num_heads=2, residual_accel_max_ms2=39.2266, logit_scale=5.0)`:
    - Permutation-invariant actor-critic network with **48,711 trainable parameters** (< 100,000 parameter budget).
    - Ego Encoder: `Linear(10 -> 64) + LayerNorm(64) + SiLU()`, produces query $Q_i$ of shape `[B, 1, 64]`.
    - Shared Token Encoders: Threat `Sequential(Linear(15, 64), SiLU, Linear(64, 64), LayerNorm(64), SiLU)` and Ally `Sequential(Linear(9, 64), SiLU, Linear(64, 64), LayerNorm(64), SiLU)`.
    - Multi-Head Cross-Attention: `nn.MultiheadAttention(64, 2)` over concatenated key/values `[K_threat + K_ally, 64]` with safe all-masked key padding handling preventing softmax NaNs.
    - Fusion Trunk: `Sequential(Linear(128, 64), LayerNorm(64), SiLU)` producing drone latent $h_i$ `[B, 64]`.
    - Continuous Residual Actor: $a_i^{RL} = \tanh(\mu(h_i)) \cdot a_{max}$, state-independent clamped log-std.
    - Cross-Pointer Coordination Head: $u_{topk} = c_{logit} \tanh(\frac{E_{threat} \cdot W_{ptr}(h_i)}{\sqrt{d_{model}}})$, zeroed on invalid slots.
    - Centralized Critic: Set-mean-pooled active team latent $h_{team} = \frac{1}{\sum m_k} \sum m_k h_k$ concatenated with $h_i$ -> `Linear(128, 64) + SiLU + Linear(64, 1)`.
    - Public API: `forward(obs) -> dict`, `get_action_and_value(obs, action, deterministic) -> (action, log_prob, entropy, value, u_topk)`, `get_value(obs) -> Tensor [E,N]`, `count_parameters() -> int`.
  - `FlatMLPAblationActorCritic`: Flat MLP baseline (325 -> 128 -> 64 -> heads) providing identical API for ablation comparisons.
  - `scatter_logits_to_full(u_topk, threat_indices, threat_mask, M) -> Tensor [E,N,M]`: Scatters top-K threat logits into the full cost matrix using a dummy absorption column to isolate invalid indices without host syncs.
- `src.learning.rollout_buffer`:
  - `RolloutBuffer(rollout_steps=256, num_envs=32, num_drones=20, K_threat=15, K_ally=10, device="cuda")`:
    - Pre-allocated fixed float32/bool GPU tensors (~212.2 MB for 32x256x20, strictly < 250 MB budget).
    - `insert(obs, action, log_prob, reward, done, value)`: Zero-host-sync GPU tensor slice copying.
    - `compute_gae(next_value, next_done, gamma=0.99, gae_lambda=0.95)`: Vectorized backward GAE calculation with episode boundary isolation.
- `src.learning.mappo_trainer.MAPPOTrainer(cfg, device=None, use_curriculum=True, model_type="attention", log_dir=None)`:
  - CTDE MAPPO trainer supporting mixed-precision AMP (`bfloat16`) on CUDA and 100% GPU rollout collection via `auction_assign`.
  - Hybrid neuro-symbolic action execution: `compute_hybrid_env_action(obs, delta_a_rl, u_topk) -> Tensor [E,N,3]` combines TPN guidance, continuous RL residual, and CBF safety shield.
  - Zero host syncs inside `collect_rollout()`, vectorized GAE computation, and clipped PPO surrogate updates.
  - Automated zero-shot curriculum scaling: `on_curriculum_stage_advance()` seamlessly resizes env entity counts (`N, M`) and buffer dimensions without changing network parameters.
  - Public methods: `collect_rollout() -> dict`, `ppo_update() -> dict`, `on_curriculum_stage_advance() -> None`, `save_checkpoint(path, extra_meta) -> None`, `load_checkpoint(path) -> dict`.
- `scripts/train.py`:
  - Full CLI training entrypoint with `--config`, `--smoke-test`, `--total-steps`, `--updates`, `--num-envs`, `--rollout-steps`, `--seed`, `--device`, `--model-type`, `--no-curriculum`, and `--save-dir`.
  - Streams JSONL metrics (`metrics.jsonl`), logs TensorBoard scalars, and auto-saves `latest.pt` / `best.pt` checkpoints.

## Server & WebSocket Binary Telemetry API (Milestone M5 Part 1)
- `src.server.ws_binary_server`:
  - `pack_binary_frame(env, algo_id=1, fps=60.0) -> bytes`:
    - Deterministic little-endian `float32` packing assembled directly on GPU with 0 host-synchronization before bulk CPU buffer transfer.
    - Header: 16 floats (64 bytes) including `magic=2026.0`, `timestamp`, `step`, `N`, `M`, `P`, `intercept_rate`, `leaker_rate`, `duplicate_conflicts`, `delta_v_total`, `friendly_collisions`, `algo_id`, `sigma_obs`, `fps`, `world_w`, `world_h`.
    - HVA Block: `P * 4` floats (`[id, x, y, z]`).
    - Entity Block: `(N + M) * 10` floats (`[id, type, x, y, z, vx, vy, vz, status, assigned_id]`).
    - Total frame bytes: `(16 + 4*P + 10*(N+M)) * 4` (2096 bytes for N=20, M=30; ~12.1 KB for N=100, M=200; ~32.1 KB for N=300, M=500).
  - `unpack_binary_frame(raw_bytes: bytes) -> dict`: Zero-copy deserialization returning header, HVAs, drones, and missiles.
  - `SandboxSession(cfg, checkpoint_path=None, algorithm="auction", device=None)`:
    - Interactive simulation controller with mid-episode algorithm hot-swapping across `greedy`, `hungarian`, `auction`, and `attention_mappo`.
    - Live threat salvo injection (`spawn_salvo`), sensor noise and playback control (`set_params`), and scenario reset (`reset`).
    - `step_and_pack() -> bytes`: Steps simulation with active algorithm and returns binary packed frame.
  - `create_app(cfg, checkpoint_path=None, viewer_dir="viewer3d") -> FastAPI`:
    - `GET /api/status`: JSON summary of session state and metrics.
    - `POST /api/command`: JSON endpoint executing control commands (`set_algorithm`, `spawn_salvo`, `set_params`, `reset`).
    - `WS /ws`: 60 Hz binary `Float32Array` frame streaming WebSocket with concurrent JSON bidirectional command parsing.
    - Static file mount serving `viewer3d/` at `/`.
- `scripts/demo_server.py`:
  - Full CLI entrypoint supporting `--config`, `--checkpoint`, `--host`, `--port`, `--algorithm`, and `--smoke-test` (10-step frame size & latency validator).

## Three.js 3D Tactical Viewer Architecture (`viewer3d/`) (Milestone M5 Part 2)
- Zero-build-step ES Module architecture with CDN `importmap` for Three.js (also supports `npm`/`vite`).
- Coordinate convention: maps simulation `(x, y, z)` (z=altitude) to Three.js `(x, z, -y)` (Y-up, 1 sim meter = 1 Three.js unit).
- Key Modules:
  - `viewer3d/index.html`: Tactical obsidian HUD layout, full-viewport canvas, telemetry overlay, and red-team control panel.
  - `viewer3d/src/InstancedSwarmRenderer.js`:
    - Renders up to 1,000 drones and 1,000 missiles in **2 GPU draw calls** via `THREE.InstancedMesh`.
    - Role-based dynamic instance coloring (Interceptors: Cyan `#00f0ff`, Scouts: Emerald `#10b981`, Threats: Crimson `#ff3344`, Decoys: Gold `#ffcc00`).
    - Ground HVA pads with translucent cyan hemispheric defense domes.
    - Zero per-frame heap allocations using pre-allocated `THREE.Object3D` dummy.
  - `viewer3d/src/TrailAndLockRenderer.js`:
    - Pre-allocated ring-buffer trajectory trails (`THREE.LineSegments`, `Float32Array` positions and fading vertex colors).
    - Dynamic assignment lock-on lines connecting defending interceptors to assigned targets with zero per-frame garbage collection.
  - `viewer3d/src/SandboxHUD.js`:
    - Real-time telemetry display (client/server FPS, clock, active entity counts, intercept/leaker rates, Delta-V).
    - Red-team salvo threat spawner with live sliders (salvo size, speed, weave frequency, decoy fraction).
    - Mid-episode algorithm hot-swapper, sensor noise slider, pause/resume, and scale presets (`6v8`, `20v30`, `60v100`, `300v500`).
  - `viewer3d/src/main.js`:
    - Pure `parseBinaryFrame(arrayBuffer)` zero-copy decoder parsing `Float32Array` packets with `magic === 2026.0`.
    - OrbitControls, tactical ground grid, 10km x 10km x 3km airspace boundary wireframe, render loop, and auto-reconnecting WebSocket client.
- **Launching the Live Viewer**:
  ```bash
  .venv/bin/python scripts/demo_server.py --port 8765
  # Open browser at http://localhost:8765
  ```

## Evaluation & Statistical Benchmark Suite (Milestone M6 Part 1)
- `src.evaluation.metrics`:
  - `EpisodeMetricsTracker(num_envs, device)`:
    - Tracks per-step accumulators across parallel environments without host syncs: `total_threat_steps`, `duplicate_violation_steps`, `total_interceptor_assign_steps`, `decoy_assign_steps`, and wall-clock `step_latencies_ms`.
    - `finalize(env) -> dict`: Combines terminal `env.metrics()` with accumulators to return per-env `[E]` float tensors and aggregate summary statistics (`mean`, `std`, `ci95`): `intercept_rate_pct`, `leaker_rate_pct`, `duplicate_ownership_rate_pct`, `false_target_waste_pct`, `friendly_collision_count`, `mean_delta_v_per_intercept`, `delta_v_total`, `mean_step_latency_ms`, and `peak_vram_mb`.
  - `evaluate_policy_episodes(cfg, policy_name, num_episodes=20, num_envs=10, seeds=(42, 43, 44, 45, 46), scale_override=None, ablation_flags=None, checkpoint_path=None, device=None) -> dict`:
    - Evaluates classical baselines (`greedy`, `hungarian_tpn`, `auction_tpn`) and MAPPO policies (`flat_mlp_mappo`, `attention_mappo`) across multi-seed Monte Carlo episodes.
    - Supports scale overrides (Small, Medium, Large scales) and component ablation flags (`use_cbf`, `use_kalman`, `use_residual_rl`, `use_coord_logits`).
- `src.evaluation.run_ablations`:
  - `run_benchmark_suite(cfg, seeds, episodes_per_seed, output_dir, checkpoint_path, quick_mode=False) -> dict`:
    - Runs cross-scale zero-shot generalization benchmark (`Small`, `Medium`, `Large Zero-Shot`).
    - Runs component ablation study on Medium scale (`Full`, `w/o Cross-Attention`, `w/o CBF Safety Shield`, `w/o Kalman Filter`, `w/o RL Residual & Logits`).
    - Produces `benchmark_results.json`, `benchmark_summary.csv`, formatted `benchmark_report.md`, and Matplotlib publication figures:
      - `scale_generalization.png`: Grouped bar chart of intercept and leaker rates across scales with error bars.
      - `ablation_study.png`: 2x2 multi-panel bar chart comparing key performance indicators across ablation variants.
      - `pareto_latency_vs_intercept.png`: Trade-off scatter plot of step latency (ms) vs intercept success rate (%).
- `scripts/evaluate.py`:
  - CLI benchmark runner with `--config`, `--checkpoint`, `--seeds`, `--episodes-per-seed`, `--output-dir`, and `--quick` / `--smoke-test`.

### 5-Seed Statistical Benchmark Results (Final Release)
- **Swarm Scale Generalization**:
  - Small (`10v15`): Greedy `46.4 ± 6.3%`, Hungarian `60.2 ± 7.1%`, Auction `77.5 ± 7.5%`, Flat-MLP `79.5 ± 3.6%`, MAPPO `78.5 ± 6.8%`.
  - Medium (`30v50`): Greedy `40.0 ± 5.1%`, Hungarian `59.3 ± 2.6%`, Auction `74.4 ± 3.4%`, Flat-MLP `74.4 ± 1.7%`, MAPPO `74.5 ± 3.7%`.
  - Large Zero-Shot (`100v150`): Greedy `46.5 ± 3.0%`, Hungarian `65.2 ± 2.0%`, Auction **`82.8 ± 1.4%`**, Flat-MLP `82.1 ± 3.0%`, MAPPO **`81.1 ± 2.7%`**.
- **Component Ablations (Medium 30v50)**:
  - **Full (Cross-Attn + TPN + CBF)**: Intercept **`74.5 ± 3.7%`**, Friendly Collisions **`0.0 ± 0.0`**, Mean $\Delta V$ **`1035.0 ± 40.2 m/s`**.
  - **w/o Cross-Attention (Flat-MLP)**: Intercept `74.4 ± 1.7%`, Friendly Collisions `0.0 ± 0.0`, Mean $\Delta V$ `1036.3 ± 48.9 m/s`.
  - **w/o CBF Safety Shield**: Intercept `69.2 ± 1.9%`, Friendly Collisions **`89.0 ± 100.1`** (severe collision spike during tight formation launches).
  - **w/o Kalman Filter (Raw Noisy Obs)**: Intercept drops to **`6.5 ± 3.4%`**, Mean $\Delta V$ surges to **`18,888.1 ± 8566.9 m/s`** (~18x $\Delta V$ waste on noisy steering jitter).
  - **w/o RL Residual & Coord Logits**: Intercept `74.4 ± 3.4%`, Friendly Collisions `0.1 ± 0.1`, Mean $\Delta V$ `1026.6 ± 45.6 m/s`.
- **Trained Checkpoint**: `runs/mappo/best.pt` achieved **88.9%** peak win rate during Stage 1/2 training and is automatically loaded for benchmarks and live sandbox sessions.
- **Guidance & Interception Optimizations**:
  - Decoupled pure lateral APN with 100% authority priority up to $a_{\max} = 117.68\text{ m/s}^2$ (12G) and forward acceleration along $\mathbf{u}_r$ using only orthogonal remaining authority $\sqrt{\max(a_{\max}^2 - \|\mathbf{a}_{\text{lat}}\|^2, 0)}$.
  - Clamped effective distance $d_{\text{eff}} \ge 25.0\text{ m}$ in relative kinematics to eliminate the terminal LOS-rate singularity ($\omega \to \infty$) at CPA.
  - Added receding threat penalty ($+500 + 2 \cdot d$) when $V_c < -20\text{ m/s}$ and $d > 100\text{ m}$, preventing interceptors from chasing faster 450 m/s threats from behind, alongside $+5000$ decoy cost penalty.
  - Dynamic terminal commitment lock when $d < 800\text{ m}$ and $V_c > 0\text{ m/s}$, strictly enforcing 1 target per drone ($\sum_m z_{i,m} \le 1$).



