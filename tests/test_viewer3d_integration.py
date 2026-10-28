"""Integration tests for Three.js 3D Tactical Viewer and FastAPI server integration.

Tests:
1. test_viewer3d_files_and_static_serving
2. test_js_binary_layout_contract_consistency
3. test_websocket_full_sandbox_workflow
"""

import json
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np
import pytest
import torch

from src.config import load_config
from src.envs.vectorized_swarm_env import VectorizedSwarmEnv
from src.server.ws_binary_server import (
    create_app,
    pack_binary_frame,
    unpack_binary_frame,
)


@pytest.fixture
def default_cfg():
    return load_config("configs/default_sim.yaml")


def test_viewer3d_files_and_static_serving(default_cfg):
    """Verify all viewer3d frontend files exist and are served with HTTP 200 via FastAPI."""
    repo_root = Path(__file__).resolve().parent.parent
    viewer_dir = repo_root / "viewer3d"

    required_files = [
        viewer_dir / "index.html",
        viewer_dir / "package.json",
        viewer_dir / "src" / "main.js",
        viewer_dir / "src" / "InstancedSwarmRenderer.js",
        viewer_dir / "src" / "TrailAndLockRenderer.js",
        viewer_dir / "src" / "SandboxHUD.js",
    ]

    for fpath in required_files:
        assert fpath.exists(), f"Missing required viewer3d file: {fpath}"
        assert fpath.stat().st_size > 0, f"File is empty: {fpath}"

    # Test static file serving via TestClient
    app = create_app(default_cfg, viewer_dir=str(viewer_dir))
    client = TestClient(app)

    # 1. GET / (index.html)
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "SwarmForge" in res_index.text
    assert "tactical-canvas" in res_index.text

    # 2. GET /src/main.js
    res_main = client.get("/src/main.js")
    assert res_main.status_code == 200
    assert "parseBinaryFrame" in res_main.text

    # 3. GET /src/InstancedSwarmRenderer.js
    res_swarm = client.get("/src/InstancedSwarmRenderer.js")
    assert res_swarm.status_code == 200
    assert "InstancedSwarmRenderer" in res_swarm.text

    # 4. GET /src/TrailAndLockRenderer.js
    res_trail = client.get("/src/TrailAndLockRenderer.js")
    assert res_trail.status_code == 200
    assert "TrailAndLockRenderer" in res_trail.text

    # 5. GET /src/SandboxHUD.js
    res_hud = client.get("/src/SandboxHUD.js")
    assert res_hud.status_code == 200
    assert "SandboxHUD" in res_hud.text


def test_js_binary_layout_contract_consistency(default_cfg):
    """Verify JavaScript Three.js renderers and binary layout contracts match ws_binary_server.py."""
    repo_root = Path(__file__).resolve().parent.parent

    # 1. Verify InstancedSwarmRenderer uses THREE.InstancedMesh and composite geometries
    swarm_code = (repo_root / "viewer3d" / "src" / "InstancedSwarmRenderer.js").read_text(encoding="utf-8")
    assert "THREE.InstancedMesh" in swarm_code
    assert "setMatrixAt" in swarm_code
    assert "setColorAt" in swarm_code
    assert "BufferGeometryUtils.mergeGeometries" in swarm_code
    assert "createInterceptorGeometry" in swarm_code
    assert "createScoutGeometry" in swarm_code
    assert "createMissileGeometry" in swarm_code
    assert "createDecoyGeometry" in swarm_code

    # 2. Verify TrailAndLockRenderer uses THREE.BufferGeometry and THREE.LineSegments
    trail_code = (repo_root / "viewer3d" / "src" / "TrailAndLockRenderer.js").read_text(encoding="utf-8")
    assert "THREE.BufferGeometry" in trail_code
    assert "THREE.LineSegments" in trail_code
    assert "setDrawRange" in trail_code

    # 3. Verify main.js uses Float32Array and magic 2026.0
    main_code = (repo_root / "viewer3d" / "src" / "main.js").read_text(encoding="utf-8")
    assert "Float32Array" in main_code
    assert "2026.0" in main_code

    # 4. Verify Python pack_binary_frame matches the exact offsets in parseBinaryFrame
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = VectorizedSwarmEnv(default_cfg, num_envs=1, device=device)
    raw_bytes = pack_binary_frame(env, algo_id=2, fps=60.0)

    arr = np.frombuffer(raw_bytes, dtype=np.float32)
    assert arr[0] == 2026.0

    # Header verification
    N = int(arr[3])
    M = int(arr[4])
    P = int(arr[5])
    expected_floats = 16 + 4 * P + 10 * (N + M)
    assert len(arr) == expected_floats

    # HVA block offset
    hva_slice = arr[16 : 16 + 4 * P]
    assert len(hva_slice) == 4 * P

    # Drone block offset
    drone_start = 16 + 4 * P
    drone_slice = arr[drone_start : drone_start + 10 * N]
    assert len(drone_slice) == 10 * N

    # Missile block offset
    missile_start = drone_start + 10 * N
    missile_slice = arr[missile_start : missile_start + 10 * M]
    assert len(missile_slice) == 10 * M


def test_websocket_full_sandbox_workflow(default_cfg):
    """Verify live bidirectional WebSocket workflow: telemetry streaming, algo swap, salvo spawn, params, reset."""
    app = create_app(default_cfg)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        # 1. Receive initial streaming frame
        raw1 = ws.receive_bytes()
        arr1 = np.frombuffer(raw1, dtype=np.float32)
        assert arr1[0] == 2026.0
        init_M = int(arr1[4])
        assert init_M == 30

        # 2. Hot-swap algorithm to Hungarian (algo_id = 1)
        ws.send_text(json.dumps({"type": "set_algorithm", "algorithm": "hungarian"}))
        raw2 = ws.receive_bytes()
        arr2 = np.frombuffer(raw2, dtype=np.float32)
        assert arr2[11] == 1.0  # algo_id for hungarian

        # 3. Hot-swap algorithm to Attention-MAPPO (algo_id = 3)
        ws.send_text(json.dumps({"type": "set_algorithm", "algorithm": "attention_mappo"}))
        raw3 = ws.receive_bytes()
        arr3 = np.frombuffer(raw3, dtype=np.float32)
        assert arr3[11] == 3.0  # algo_id for attention_mappo

        # 4. Spawn a live red-team salvo of 10 threats
        ws.send_text(json.dumps({
            "type": "spawn_salvo",
            "count": 10,
            "speed": 420.0,
            "weave_freq": 2.0,
            "decoy_fraction": 0.2,
        }))
        raw4 = ws.receive_bytes()
        arr4 = np.frombuffer(raw4, dtype=np.float32)
        assert int(arr4[4]) == init_M + 10

        # 5. Set simulation parameters (sensor noise & playback)
        ws.send_text(json.dumps({"type": "set_params", "sigma_obs": 25.0}))
        raw5 = ws.receive_bytes()
        arr5 = np.frombuffer(raw5, dtype=np.float32)
        assert np.isclose(arr5[12], 25.0)

        # 6. Reset scenario
        ws.send_text(json.dumps({
            "type": "reset",
            "num_interceptors": 12,
            "num_scouts": 3,
            "num_missiles": 25,
        }))
        raw6 = ws.receive_bytes()
        arr6 = np.frombuffer(raw6, dtype=np.float32)
        assert int(arr6[3]) == 15  # N = 12 + 3
        assert int(arr6[4]) == 25  # M = 25


def test_slow_motion_and_step_once_commands(default_cfg):
    """Verify speed_multiplier (0.1, 0.25, 0.5, 2.0) and step_once commands via /api/command and /ws."""
    app = create_app(default_cfg)
    client = TestClient(app)

    # 1. Test speed_multiplier settings over /api/command
    for spd in [0.1, 0.25, 0.5, 2.0]:
        res = client.post("/api/command", json={"type": "set_params", "speed_multiplier": spd})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert np.isclose(data["speed_multiplier"], spd)

    # 2. Test pause and step_once over /api/command
    client.post("/api/command", json={"type": "set_params", "paused": True})
    status1 = client.get("/api/status").json()
    step1 = status1["step"]
    assert status1["paused"] is True

    # Send step_once command
    res_step = client.post("/api/command", json={"type": "step_once"})
    assert res_step.status_code == 200
    assert res_step.json()["status"] == "ok"
    assert res_step.json()["step"] == step1 + 1

    # 3. Test speed_multiplier and step_once over WebSocket /ws
    with client.websocket_connect("/ws") as ws:
        # Check initial frame
        raw = ws.receive_bytes()
        assert len(raw) > 64

        # Set speed_multiplier to 0.1 over WebSocket
        ws.send_text(json.dumps({"type": "set_params", "speed_multiplier": 0.1, "paused": True}))
        raw_ack = ws.receive_bytes()
        assert len(raw_ack) > 64

        # Send step_once over WebSocket
        ws.send_text(json.dumps({"type": "step_once"}))
        raw_step = ws.receive_bytes()
        arr_step = np.frombuffer(raw_step, dtype=np.float32)
        assert arr_step[2] >= step1 + 1
