#!/usr/bin/env python3
"""CLI entry point for running the SwarmForge live simulation WebSocket server."""

import argparse
from pathlib import Path
import sys
import time

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import get, load_config
from src.server.ws_binary_server import (
    ALGO_NAME_TO_ID,
    SandboxSession,
    create_app,
    unpack_binary_frame,
)


def run_smoke_test(cfg: dict, checkpoint: str = None, algorithm: str = "auction"):
    """Execute smoke test stepping 10 frames, verifying binary frame size and latency."""
    print("=" * 70)
    print("Running SwarmForge Demo Server Smoke Test...")
    print("=" * 70)
    session = SandboxSession(cfg, checkpoint_path=checkpoint, algorithm=algorithm)
    N = session.env.N
    M = session.env.M
    P = session.env.hva_pos.shape[1]
    expected_bytes = (16 + 4 * P + 10 * (N + M)) * 4

    print(f"Session initialized with algorithm='{algorithm}', N={N}, M={M}, P={P}")
    print(f"Expected binary frame size: {expected_bytes} bytes")

    latencies_ms = []
    for step in range(10):
        t0 = time.perf_counter()
        raw_frame = session.step_and_pack()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

        if len(raw_frame) != expected_bytes:
            print(f"ERROR: Frame size mismatch at step {step}: got {len(raw_frame)} bytes, expected {expected_bytes} bytes")
            sys.exit(1)

        unpacked = unpack_binary_frame(raw_frame)
        assert unpacked["header"]["magic"] == 2026.0
        assert unpacked["header"]["step"] == step + 1

    mean_lat_ms = float(np.mean(latencies_ms))
    max_lat_ms = float(np.max(latencies_ms))
    print(f"Smoke test passed! 10 steps completed.")
    print(f"Frame size: {expected_bytes} bytes | Mean step+pack latency: {mean_lat_ms:.2f} ms (max: {max_lat_ms:.2f} ms)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="SwarmForge Live Simulation WebSocket Server")
    parser.add_argument("--config", type=str, default="configs/default_sim.yaml", help="Path to YAML configuration")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to MAPPO policy checkpoint")
    parser.add_argument("--host", type=str, default=None, help="Server host IP (default from config or 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Server WebSocket/HTTP port (default from config or 8765)")
    parser.add_argument("--algorithm", type=str, default="auction", choices=list(ALGO_NAME_TO_ID.keys()), help="Initial guidance/allocation algorithm")
    parser.add_argument("--smoke-test", action="store_true", help="Run 10-step smoke test and exit cleanly")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.smoke_test:
        run_smoke_test(cfg, checkpoint=args.checkpoint, algorithm=args.algorithm)
        return

    import uvicorn

    host = args.host or str(get(cfg, "visualization.host", "0.0.0.0"))
    port = args.port or int(get(cfg, "visualization.websocket_port", 8765))

    app = create_app(cfg, checkpoint_path=args.checkpoint)
    app.state.session.algorithm = args.algorithm.lower()
    app.state.session.algo_id = ALGO_NAME_TO_ID.get(app.state.session.algorithm, 2)

    print(f"Starting SwarmForge simulation server at http://{host}:{port} (WS ws://{host}:{port}/ws)")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
