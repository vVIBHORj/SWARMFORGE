"""Unit and integration tests for FastAPI packed binary Float32Array WebSocket server.

Tests:
1. test_binary_frame_pack_and_unpack_roundtrip
2. test_hot_swap_algorithms_mid_episode
3. test_live_salvo_spawner
4. test_fastapi_websocket_and_command_endpoints
5. test_300v500_frame_budget_latency
"""

import json
import time
from fastapi.testclient import TestClient
import numpy as np
import pytest
import torch

from src.config import load_config
from src.envs.vectorized_swarm_env import VectorizedSwarmEnv
from src.server.ws_binary_server import (
    ALGO_ID_TO_NAME,
    ALGO_NAME_TO_ID,
    SandboxSession,
    create_app,
    pack_binary_frame,
    unpack_binary_frame,
)


@pytest.fixture
def default_cfg():
    return load_config("configs/default_sim.yaml")


@pytest.fixture
def stress_cfg():
    return load_config("configs/stress_test_300v500.yaml")


def test_binary_frame_pack_and_unpack_roundtrip(default_cfg):
    """Verify deterministic Float32Array frame byte length, packing, and roundtrip fidelity."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = VectorizedSwarmEnv(default_cfg, num_envs=1, device=device)

    # N=20 (16 interceptors + 4 scouts), M=30, P=2
    N, M, P = env.N, env.M, env.hva_pos.shape[1]
    assert N == 20
    assert M == 30
    assert P == 2

    # Pack frame and verify byte length: (16 + 4*P + 10*(N+M)) * 4 = (16 + 8 + 500) * 4 = 2096 bytes
    expected_bytes = (16 + 4 * P + 10 * (N + M)) * 4
    assert expected_bytes == 2096

    raw_frame = pack_binary_frame(env, algo_id=1, fps=60.0)
    assert len(raw_frame) == expected_bytes

    # Also test N=100, M=200, P=2 scale matching Spec Section 9.1: (16 + 8 + 3000) * 4 = 12096 bytes (~12.1 KB)
    cfg_100v200 = load_config("configs/default_sim.yaml")
    cfg_100v200["entities"]["drones"]["num_interceptors"] = 80
    cfg_100v200["entities"]["drones"]["num_scouts"] = 20
    cfg_100v200["entities"]["missiles"]["num_missiles"] = 200
    env_large = VectorizedSwarmEnv(cfg_100v200, num_envs=1, device=device)
    raw_large = pack_binary_frame(env_large, algo_id=2, fps=60.0)
    expected_large_bytes = (16 + 4 * 2 + 10 * (100 + 200)) * 4
    assert expected_large_bytes == 12096
    assert len(raw_large) == 12096

    # Unpack default frame and assert roundtrip fields
    unpacked = unpack_binary_frame(raw_frame)
    hdr = unpacked["header"]
    assert np.isclose(hdr["magic"], 2026.0)
    assert np.isclose(hdr["timestamp"], float(env.t[0].item()) * env.dt)
    assert hdr["step"] == int(env.t[0].item())
    assert hdr["N"] == N
    assert hdr["M"] == M
    assert hdr["P"] == P
    assert hdr["algo_id"] == 1
    assert np.isclose(hdr["fps"], 60.0)
    assert np.isclose(hdr["sigma_obs"], env.sigma_obs)
    assert np.isclose(hdr["world_w"], env.world_bounds[0])
    assert np.isclose(hdr["world_h"], env.world_bounds[2])

    # Assert HVA coordinates match env tensors within 1e-5
    hvas_np = unpacked["hvas"]
    assert hvas_np.shape == (P, 4)
    for p in range(P):
        assert np.isclose(hvas_np[p, 0], float(p))
        np.testing.assert_allclose(hvas_np[p, 1:], env.hva_pos[0, p].cpu().numpy(), atol=1e-5)

    # Assert Drones match env tensors
    drones_np = unpacked["drones"]
    assert drones_np.shape == (N, 10)
    drone_pos_np = env.drone_pos[0].cpu().numpy()
    drone_vel_np = env.drone_vel[0].cpu().numpy()
    drone_roles_np = env.drone_role[0].float().cpu().numpy()
    drone_active_np = env.drone_active[0].float().cpu().numpy()

    np.testing.assert_allclose(drones_np[:, 0], np.arange(N, dtype=np.float32))
    np.testing.assert_allclose(drones_np[:, 1], drone_roles_np)
    np.testing.assert_allclose(drones_np[:, 2:5], drone_pos_np, atol=1e-5)
    np.testing.assert_allclose(drones_np[:, 5:8], drone_vel_np, atol=1e-5)
    np.testing.assert_allclose(drones_np[:, 8], drone_active_np)

    # Assert Missiles match env tensors
    missiles_np = unpacked["missiles"]
    assert missiles_np.shape == (M, 10)
    m_pos_np = env.missile_pos[0].cpu().numpy()
    m_vel_np = env.missile_vel[0].cpu().numpy()
    m_target_np = env.missile_target_hva[0].float().cpu().numpy()

    np.testing.assert_allclose(missiles_np[:, 0], np.arange(M, dtype=np.float32))
    np.testing.assert_allclose(missiles_np[:, 2:5], m_pos_np, atol=1e-5)
    np.testing.assert_allclose(missiles_np[:, 5:8], m_vel_np, atol=1e-5)
    np.testing.assert_allclose(missiles_np[:, 9], m_target_np)


def test_hot_swap_algorithms_mid_episode(default_cfg):
    """Step SandboxSession 5 steps per algorithm (greedy -> hungarian -> auction -> attention_mappo)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    session = SandboxSession(default_cfg, algorithm="greedy", device=device)

    # 1. Greedy (5 steps)
    for _ in range(5):
        frame = session.step_and_pack()
    unpacked = unpack_binary_frame(frame)
    assert session.env.t[0].item() == 5
    assert unpacked["header"]["algo_id"] == 0
    assert (session.env.z[0].sum(dim=0) <= session.env.missile_K[0]).all()

    # 2. Hungarian (5 steps)
    res = session.handle_command({"type": "set_algorithm", "algorithm": "hungarian"})
    assert res["status"] == "ok"
    assert session.algo_id == 1
    for _ in range(5):
        frame = session.step_and_pack()
    unpacked = unpack_binary_frame(frame)
    assert session.env.t[0].item() == 10
    assert unpacked["header"]["algo_id"] == 1
    assert (session.env.z[0].sum(dim=0) <= session.env.missile_K[0]).all()

    # 3. Auction (5 steps)
    res = session.handle_command({"type": "set_algorithm", "algorithm": "auction"})
    assert res["status"] == "ok"
    assert session.algo_id == 2
    for _ in range(5):
        frame = session.step_and_pack()
    unpacked = unpack_binary_frame(frame)
    assert session.env.t[0].item() == 15
    assert unpacked["header"]["algo_id"] == 2
    assert (session.env.z[0].sum(dim=0) <= session.env.missile_K[0]).all()

    # 4. Attention MAPPO (5 steps)
    res = session.handle_command({"type": "set_algorithm", "algorithm": "attention_mappo"})
    assert res["status"] == "ok"
    assert session.algo_id == 3
    for _ in range(5):
        frame = session.step_and_pack()
    unpacked = unpack_binary_frame(frame)
    assert session.env.t[0].item() == 20
    assert unpacked["header"]["algo_id"] == 3
    assert (session.env.z[0].sum(dim=0) <= session.env.missile_K[0]).all()


def test_live_salvo_spawner(default_cfg):
    """Step SandboxSession 10 steps, trigger live red-team salvo, and verify shape consistency."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    session = SandboxSession(default_cfg, algorithm="auction", device=device)
    init_M = session.env.M
    init_N = session.env.N
    P = session.env.hva_pos.shape[1]

    for _ in range(10):
        session.step_and_pack()
    assert session.env.t[0].item() == 10

    # Live red-team salvo spawn
    spawn_res = session.handle_command({
        "type": "spawn_salvo",
        "count": 12,
        "speed": 420.0,
        "weave_freq": 2.0,
        "decoy_fraction": 0.25,
    })
    assert spawn_res["status"] == "ok"
    assert spawn_res["spawned"] == 12
    assert session.env.M == init_M + 12

    # Verify new live threats are present
    live_m = session.env.missile_live()
    assert live_m[0].sum().item() > 0

    # Verify frame size updates to reflect new entity count
    new_frame = session.step_and_pack()
    expected_bytes = (16 + 4 * P + 10 * (init_N + init_M + 12)) * 4
    assert len(new_frame) == expected_bytes

    unpacked = unpack_binary_frame(new_frame)
    assert unpacked["header"]["M"] == init_M + 12

    # Subsequent 10 steps run cleanly without shape mismatch
    for _ in range(10):
        frame = session.step_and_pack()
        assert len(frame) == expected_bytes

    assert session.env.t[0].item() == 21


def test_fastapi_websocket_and_command_endpoints(default_cfg):
    """Test HTTP REST endpoints and concurrent binary WebSocket streaming."""
    app = create_app(default_cfg)
    client = TestClient(app)

    # 1. GET /api/status
    res = client.get("/api/status")
    assert res.status_code == 200
    status = res.json()
    assert status["algorithm"] in ALGO_NAME_TO_ID
    assert status["N"] == 20
    assert status["M"] == 30
    assert "metrics" in status
    assert "intercept_rate" in status["metrics"]

    # 2. POST /api/command: set_algorithm
    res_cmd = client.post("/api/command", json={"type": "set_algorithm", "algorithm": "hungarian"})
    assert res_cmd.status_code == 200
    assert res_cmd.json()["algorithm"] == "hungarian"
    assert res_cmd.json()["algo_id"] == 1

    # 3. POST /api/command: spawn_salvo
    res_salvo = client.post("/api/command", json={"type": "spawn_salvo", "count": 6})
    assert res_salvo.status_code == 200
    assert res_salvo.json()["spawned"] == 6
    assert res_salvo.json()["M"] == 36

    # 4. POST /api/command: set_params
    res_params = client.post("/api/command", json={"type": "set_params", "sigma_obs": 18.0, "paused": True})
    assert res_params.status_code == 200
    assert res_params.json()["sigma_obs"] == 18.0
    assert res_params.json()["paused"] is True

    # 5. WS /ws binary stream and concurrent command handling
    # Unpause first
    client.post("/api/command", json={"type": "set_params", "paused": False})

    with client.websocket_connect("/ws") as ws:
        # Receive binary Float32Array frame
        raw_bytes = ws.receive_bytes()
        arr = np.frombuffer(raw_bytes, dtype=np.float32)
        assert len(arr) >= 16
        assert arr[0] == 2026.0

        # Send command over websocket
        ws.send_text(json.dumps({"type": "set_algorithm", "algorithm": "auction"}))

        # Receive next binary frame and verify hot-swapped algo_id == 2 (auction)
        raw_bytes2 = ws.receive_bytes()
        arr2 = np.frombuffer(raw_bytes2, dtype=np.float32)
        assert arr2[0] == 2026.0
        assert arr2[11] == 2.0  # algo_id for auction


def test_300v500_frame_budget_latency(stress_cfg):
    """Verify 300 drones vs 500 missiles step+pack latency is < 16.6 ms (60 Hz budget)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    session = SandboxSession(stress_cfg, algorithm="auction", device=device)

    assert session.env.N == 300
    assert session.env.M == 500

    P = session.env.hva_pos.shape[1]
    expected_bytes = (16 + 4 * P + 10 * (300 + 500)) * 4
    # (16 + 8 + 8000) * 4 = 8024 * 4 = 32096 bytes (~32.1 KB)
    assert expected_bytes == 32096

    # 5 warmup steps
    for _ in range(5):
        frame = session.step_and_pack()
        assert len(frame) == expected_bytes

    # 20 timed steps
    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        frame = session.step_and_pack()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)

    mean_lat_ms = float(np.mean(latencies))
    print(f"\n[Mode 3 Stress Test 300v500] Mean latency: {mean_lat_ms:.2f} ms (budget: 16.6 ms, device: {device})")
    assert mean_lat_ms < 16.6, f"Mean latency {mean_lat_ms:.2f} ms exceeded 16.6 ms 60 Hz frame budget"
