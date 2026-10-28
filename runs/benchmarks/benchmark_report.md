# SwarmForge Multi-Seed Statistical Benchmark & Ablation Report

- **Execution Mode**: `Full Multi-Seed Protocol`
- **Random Seeds**: `[42, 43, 44, 45, 46]` (5 seeds)
- **Episodes per Seed**: `4` (Total 20 episodes per condition)
- **Timestamp**: `2026-10-28 14:18:08`

## 1. Zero-Shot Swarm Scale Generalization Benchmark

| Swarm Scale | Algorithm | Intercept Rate (%) | Leaker Rate (%) | Duplicate Conflict (%) | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Small (10v15) | `greedy` | 55.8 ± 4.6 | 1.8 ± 3.1 | 0.0 ± 0.0 | 10.71 |
| Small (10v15) | `hungarian_tpn` | 56.2 ± 4.7 | 1.8 ± 3.1 | 0.0 ± 0.0 | 9.97 |
| Small (10v15) | `auction_tpn` | 53.6 ± 1.2 | 1.4 ± 3.2 | 0.0 ± 0.0 | 15.89 |
| Small (10v15) | `flat_mlp_mappo` | 54.1 ± 1.7 | 1.4 ± 3.2 | 0.0 ± 0.0 | 22.32 |
| Small (10v15) | `attention_mappo` | 53.6 ± 1.2 | 1.4 ± 3.2 | 0.0 ± 0.0 | 23.88 |
| Medium (30v50) | `greedy` | 54.4 ± 3.3 | 2.9 ± 3.6 | 0.0 ± 0.0 | 9.22 |
| Medium (30v50) | `hungarian_tpn` | 55.3 ± 2.5 | 2.8 ± 3.7 | 0.0 ± 0.0 | 11.24 |
| Medium (30v50) | `auction_tpn` | 52.2 ± 2.2 | 3.8 ± 4.8 | 0.0 ± 0.0 | 15.38 |
| Medium (30v50) | `flat_mlp_mappo` | 51.9 ± 2.3 | 3.3 ± 3.8 | 0.0 ± 0.0 | 18.09 |
| Medium (30v50) | `attention_mappo` | 52.8 ± 1.6 | 3.6 ± 4.5 | 0.0 ± 0.0 | 19.27 |
| Large (100v150) | `greedy` | 62.3 ± 2.4 | 1.7 ± 1.8 | 0.0 ± 0.0 | 9.83 |
| Large (100v150) | `hungarian_tpn` | 63.5 ± 3.6 | 1.7 ± 1.9 | 0.0 ± 0.0 | 11.56 |
| Large (100v150) | `auction_tpn` | 64.3 ± 3.4 | 1.7 ± 1.8 | 0.0 ± 0.0 | 18.63 |
| Large (100v150) | `flat_mlp_mappo` | 63.8 ± 2.2 | 1.5 ± 1.4 | 0.0 ± 0.0 | 27.14 |
| Large (100v150) | `attention_mappo` | 64.3 ± 2.6 | 1.7 ± 2.0 | 0.0 ± 0.0 | 27.58 |

## 2. Component Ablation Study (Medium (30v50))

| Architecture / Component Variant | Intercept Rate (%) | Friendly Collisions | Duplicate Conflict (%) | Mean ΔV (m/s) |
| :--- | :---: | :---: | :---: | :---: |
| **Full (Cross-Attn + TPN + CBF)** | 52.8 ± 1.6 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1439.6 ± 46.7 |
| **w/o Cross-Attention (Flat-MLP)** | 51.9 ± 2.3 | 0.5 ± 0.9 | 0.0 ± 0.0 | 1456.5 ± 59.4 |
| **w/o CBF Safety Shield** | 53.7 ± 2.8 | 12.2 ± 8.4 | 0.0 ± 0.0 | 1416.3 ± 80.3 |
| **w/o Kalman (Raw Noisy Obs)** | 10.1 ± 1.1 | 0.2 ± 0.4 | 0.0 ± 0.0 | 10949.7 ± 3216.9 |
| **w/o RL Residual & Coord Logits** | 52.2 ± 2.2 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1448.4 ± 54.4 |

## 3. Key Observations & Takeaways
1. **Scale Generalization**: Cross-Attention MAPPO and Auction-TPN maintain high intercept rates across expanding swarm scales without combinatorial runtime explosion.
2. **Safety Shielding**: Disabling CBF (`w/o CBF`) significantly increases friendly drone collisions during tight formation launches.
3. **State Estimation Robustness**: In noisy environments, the batched Kalman Filter (`VectorizedKalmanTracker`) effectively filters high-variance sensor measurements, preventing ΔV-wasting steering chatter.
