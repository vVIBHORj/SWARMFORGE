# SwarmForge Multi-Seed Statistical Benchmark & Ablation Report

- **Execution Mode**: `Full Multi-Seed Protocol`
- **Random Seeds**: `[42, 43, 44, 45, 46]` (5 seeds)
- **Episodes per Seed**: `4` (Total 20 episodes per condition)
- **Timestamp**: `2026-10-28 16:18:48`

## 1. Zero-Shot Swarm Scale Generalization Benchmark

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
| Large (100v150) | `auction_tpn` | 82.8 ± 1.4 | 8.7 ± 4.1 | 0.0 ± 0.0 | 13.20 |
| Large (100v150) | `flat_mlp_mappo` | 82.1 ± 3.0 | 8.9 ± 2.2 | 0.0 ± 0.0 | 17.19 |
| Large (100v150) | `attention_mappo` | 81.1 ± 2.7 | 9.5 ± 2.8 | 0.0 ± 0.0 | 18.44 |

## 2. Component Ablation Study (Medium (30v50))

| Architecture / Component Variant | Intercept Rate (%) | Friendly Collisions | Duplicate Conflict (%) | Mean ΔV (m/s) |
| :--- | :---: | :---: | :---: | :---: |
| **Full (Cross-Attn + TPN + CBF)** | 74.5 ± 3.7 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1035.0 ± 40.2 |
| **w/o Cross-Attention (Flat-MLP)** | 74.4 ± 1.7 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1036.3 ± 48.9 |
| **w/o CBF Safety Shield** | 69.2 ± 1.9 | 89.0 ± 100.1 | 0.0 ± 0.0 | 986.5 ± 71.5 |
| **w/o Kalman (Raw Noisy Obs)** | 6.5 ± 3.4 | 1.1 ± 0.5 | 0.0 ± 0.0 | 18888.1 ± 8566.9 |
| **w/o RL Residual & Coord Logits** | 74.4 ± 3.4 | 0.1 ± 0.1 | 0.0 ± 0.0 | 1026.6 ± 45.6 |

## 3. Key Observations & Takeaways
1. **Scale Generalization**: Cross-Attention MAPPO and Auction-TPN maintain high intercept rates across expanding swarm scales without combinatorial runtime explosion.
2. **Safety Shielding**: Disabling CBF (`w/o CBF`) significantly increases friendly drone collisions during tight formation launches.
3. **State Estimation Robustness**: In noisy environments, the batched Kalman Filter (`VectorizedKalmanTracker`) effectively filters high-variance sensor measurements, preventing ΔV-wasting steering chatter.
