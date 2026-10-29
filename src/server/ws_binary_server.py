"""FastAPI Packed Binary Float32Array WebSocket Server & Interactive Simulation Session.

Implements Spec Section 9.1 & 9.2:
- Vectorized deterministic binary Float32Array frame protocol (packing env 0 state with zero Python loops).
- Interactive SandboxSession supporting mid-episode algorithm hot-swapping and live red-team salvo spawning.
- FastAPI WebSocket + HTTP REST command & status server.
"""

import asyncio
import json
from pathlib import Path
import time
from typing import Optional, Union

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import torch

from src.baselines.auction_allocator import auction_assign
from src.baselines.baseline_controller import BaselineController
from src.config import get
from src.envs.vectorized_swarm_env import VectorizedSwarmEnv
from src.learning.attention_nets import CrossAttentionActorCritic, scatter_logits_to_full
from src.physics.guidance_laws import (
    build_cost_matrix,
    cbf_repulsion_accel,
    tpn_guidance_accel,
)

ALGO_NAME_TO_ID = {
    "greedy": 0,
    "hungarian": 1,
    "auction": 2,
    "attention_mappo": 3,
}

ALGO_ID_TO_NAME = {v: k for k, v in ALGO_NAME_TO_ID.items()}


@torch.no_grad()
def pack_binary_frame(env: VectorizedSwarmEnv, algo_id: int = 1, fps: float = 60.0) -> bytes:
    """Pack env 0 state into a single contiguous little-endian float32 binary frame.

    Protocol Layout:
    - Header: 16 floats (64 bytes)
        [0]: magic (2026.0)
        [1]: timestamp (t * dt)
        [2]: step t
        [3]: N (num drones)
        [4]: M (num missiles)
        [5]: P (num HVAs)
        [6]: intercept_rate
        [7]: leaker_rate
        [8]: duplicate_conflicts (missiles violating sum_i z_ij > K_j in env 0)
        [9]: delta_v_total
        [10]: friendly_collisions
        [11]: algo_id (0=greedy, 1=hungarian, 2=auction, 3=attention_mappo)
        [12]: sigma_obs
        [13]: fps
        [14]: world_w
        [15]: world_h
    - HVA Block: P * 4 floats ([id, x, y, z])
    - Entities Block: (N + M) * 10 floats ([id, type, x, y, z, vx, vy, vz, status, assigned_id])
    """
    device = env.device

    t_val = env.t[0].float()
    dt = float(env.dt)
    timestamp = t_val * dt
    N = int(env.N)
    M = int(env.M)
    P = int(env.hva_pos.shape[1])

    total_len = 16 + 4 * P + 10 * (N + M)
    total_frame = torch.empty(total_len, dtype=torch.float32, device=device)

    real_m = env.missile_exists[0] & (~env.missile_is_decoy[0])
    n_real = torch.clamp(real_m.sum().float(), min=1.0)
    int_rate = (real_m & (env.missile_status[0] == 1)).sum().float() / n_real
    leak_rate = (real_m & (env.missile_status[0] == 2)).sum().float() / n_real

    total_frame[0] = 2026.0
    total_frame[1] = timestamp
    total_frame[2] = t_val
    total_frame[3] = float(N)
    total_frame[4] = float(M)
    total_frame[5] = float(P)
    total_frame[6] = int_rate
    total_frame[7] = leak_rate
    total_frame[8] = (env.z[0].sum(dim=0) > env.missile_K[0]).sum().float()
    total_frame[9] = env.delta_v[0]
    total_frame[10] = env.friendly_collisions[0]
    total_frame[11] = float(algo_id)
    total_frame[12] = float(env.sigma_obs)
    total_frame[13] = float(fps)
    total_frame[14] = float(env.world_bounds[0])
    total_frame[15] = float(env.world_bounds[2])

    # 1. HVA Block: P * 4 floats
    hva_slice = total_frame[16 : 16 + 4 * P].view(P, 4)
    hva_slice[:, 0] = torch.arange(P, dtype=torch.float32, device=device)
    hva_slice[:, 1:4] = env.hva_pos[0]

    # 2. Drone Entities Block: N * 10 floats
    d_start = 16 + 4 * P
    d_slice = total_frame[d_start : d_start + 10 * N].view(N, 10)
    d_slice[:, 0] = torch.arange(N, dtype=torch.float32, device=device)
    d_slice[:, 1] = env.drone_role[0].float()
    d_slice[:, 2:5] = env.drone_pos[0]
    d_slice[:, 5:8] = env.drone_vel[0]
    d_slice[:, 8] = env.drone_active[0].float()

    z0 = env.z[0]  # [N, M]
    has_target = z0.any(dim=-1)
    target_idx = torch.argmax(z0.float(), dim=-1).float()
    d_slice[:, 9] = torch.where(has_target, target_idx, -1.0)

    # 3. Missile Entities Block: M * 10 floats
    m_start = d_start + 10 * N
    m_slice = total_frame[m_start : m_start + 10 * M].view(M, 10)
    m_slice[:, 0] = torch.arange(M, dtype=torch.float32, device=device)
    m_slice[:, 1] = torch.where(env.missile_is_decoy[0], 3.0, 2.0)
    m_slice[:, 2:5] = env.missile_pos[0]
    m_slice[:, 5:8] = env.missile_vel[0]

    t_sec = env.t[0].float() * env.dt
    unspawned = env.missile_spawn_time[0] > t_sec
    inactive = (~env.missile_exists[0]) | unspawned
    m_status = env.missile_status[0].float()
    m_slice[:, 8] = torch.where(inactive, -1.0, m_status)
    m_slice[:, 9] = env.missile_target_hva[0].float()

    return total_frame.cpu().numpy().tobytes()


def unpack_binary_frame(raw_bytes: bytes) -> dict:
    """Unpack raw float32 binary frame bytes into a structured python dictionary."""
    arr = np.frombuffer(raw_bytes, dtype=np.float32)
    if len(arr) < 16:
        raise ValueError(f"Binary frame too short: {len(arr)} floats (minimum 16 required)")

    magic = float(arr[0])
    if magic != 2026.0:
        raise ValueError(f"Invalid magic number: {magic}, expected 2026.0")

    timestamp = float(arr[1])
    step = int(arr[2])
    N = int(arr[3])
    M = int(arr[4])
    P = int(arr[5])
    intercept_rate = float(arr[6])
    leaker_rate = float(arr[7])
    duplicate_conflicts = int(arr[8])
    delta_v_total = float(arr[9])
    friendly_collisions = int(arr[10])
    algo_id = int(arr[11])
    sigma_obs = float(arr[12])
    fps = float(arr[13])
    world_w = float(arr[14])
    world_h = float(arr[15])

    expected_len = 16 + 4 * P + 10 * (N + M)
    if len(arr) != expected_len:
        raise ValueError(f"Frame length mismatch: got {len(arr)} floats, expected {expected_len}")

    offset = 16
    hvas = arr[offset : offset + 4 * P].reshape(P, 4)
    offset += 4 * P
    drones = arr[offset : offset + 10 * N].reshape(N, 10)
    offset += 10 * N
    missiles = arr[offset : offset + 10 * M].reshape(M, 10)

    return {
        "header": {
            "magic": magic,
            "timestamp": timestamp,
            "step": step,
            "N": N,
            "M": M,
            "P": P,
            "intercept_rate": intercept_rate,
            "leaker_rate": leaker_rate,
            "duplicate_conflicts": duplicate_conflicts,
            "delta_v_total": delta_v_total,
            "friendly_collisions": friendly_collisions,
            "algo_id": algo_id,
            "sigma_obs": sigma_obs,
            "fps": fps,
            "world_w": world_w,
            "world_h": world_h,
        },
        "hvas": hvas,
        "drones": drones,
        "missiles": missiles,
    }


class SandboxSession:
    """Interactive simulation controller managing environment, live policies, and hot-swapping."""

    def __init__(
        self,
        cfg: dict,
        checkpoint_path: Optional[str] = None,
        algorithm: str = "auction",
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.cfg = cfg
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.env = VectorizedSwarmEnv(cfg, num_envs=1, device=self.device)

        # Baseline controllers
        self.baseline_controllers = {
            algo: BaselineController(
                cfg,
                allocator=algo,
                use_cbf=True,
                use_estimated_state=(self.env.sigma_obs > 0),
            )
            for algo in ["greedy", "hungarian", "auction"]
        }

        # Attention MAPPO policy
        self.policy = CrossAttentionActorCritic(cfg=cfg).to(self.device)
        if checkpoint_path is None and Path("runs/mappo/best.pt").exists():
            checkpoint_path = "runs/mappo/best.pt"
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.policy.load_state_dict(state_dict)
        self.policy.eval()

        self.algorithm = algorithm.lower()
        self.algo_id = ALGO_NAME_TO_ID.get(self.algorithm, 2)
        self.target_fps = float(get(cfg, "visualization.target_fps", 60.0))
        self.paused = False
        self.auto_reset = True
        self.speed_multiplier = 1.0
        self.step_once = False
        self._step_accum = 0.0

    @torch.no_grad()
    def _act_attention_mappo(self) -> torch.Tensor:
        """Compute actions with CrossAttentionActorCritic policy and auction allocation."""
        obs = self.env.get_observations(use_estimated_state=(self.env.sigma_obs > 0))
        fwd = self.policy(obs)
        delta_a_rl = fwd["mu"]  # [E, N, 3]
        u_topk = fwd["u_topk"]  # [E, N, K_t]

        threat_indices = obs.get("threat_indices", None)
        if threat_indices is not None:
            rl_logits = scatter_logits_to_full(
                u_topk,
                threat_indices,
                obs["threat_mask"],
                self.env.M,
            )
        else:
            rl_logits = None

        use_est = self.env.sigma_obs > 0
        v_min = float(get(self.cfg, "guidance_and_allocation.v_min_closing", 20.0))
        tgo_cap = float(get(self.cfg, "guidance_and_allocation.tgo_cap", 120.0))
        invalid_cost = float(get(self.cfg, "guidance_and_allocation.invalid_cost", 1.0e6))
        auction_iters = int(get(self.cfg, "guidance_and_allocation.auction_num_iters", 10))
        auction_eps = float(get(self.cfg, "guidance_and_allocation.auction_eps", 0.05))
        nav_gain = float(get(self.cfg, "guidance_and_allocation.navigation_gain_N", 4.0))
        r_col = float(get(self.cfg, "entities.drones.collision_radius", 5.0))
        r_safe = float(get(self.cfg, "entities.drones.cbf_safe_radius", 40.0))
        k_cbf = float(get(self.cfg, "guidance_and_allocation.cbf_gain", 500.0))

        C, _ = build_cost_matrix(
            self.env,
            prev_z=self.env.z,
            rl_logits=rl_logits,
            v_min=v_min,
            tgo_cap=tgo_cap,
            invalid_cost=invalid_cost,
            use_estimated_state=use_est,
        )

        valid_drone = self.env.drone_active & (self.env.drone_role == 0)
        if use_est and hasattr(self.env, "track_state") and self.env.track_state is not None:
            valid_m_init = self.env.track_state.get("initialized", torch.ones_like(self.env.missile_live()))
            valid_missile = self.env.missile_live() & valid_m_init
            m_pos = self.env.track_state.get("pos_est", self.env.missile_pos)
            m_vel = self.env.track_state.get("vel_est", self.env.missile_vel)
        else:
            valid_missile = self.env.missile_live()
            m_pos = self.env.missile_pos
            m_vel = self.env.missile_vel

        active_commit = self.env.z & valid_drone.unsqueeze(-1) & valid_missile.unsqueeze(1)
        free_drone = valid_drone & (~active_commit.any(dim=-1))
        assigned_per_m = active_commit.sum(dim=1)
        rem_K = torch.clamp(self.env.missile_K - assigned_per_m, min=0)
        unassigned_missile = valid_missile & (rem_K > 0)

        z_alloc = auction_assign(
            C,
            free_drone,
            unassigned_missile,
            rem_K,
            num_iters=auction_iters,
            eps=auction_eps,
            invalid_cost=invalid_cost,
        )
        z = active_commit | z_alloc
        self.env.z = z

        a_pn = tpn_guidance_accel(
            self.env.drone_pos,
            self.env.drone_vel,
            m_pos,
            m_vel,
            z,
            nav_gain=nav_gain,
            v_min=v_min,
        )

        a_cbf = cbf_repulsion_accel(
            self.env.drone_pos,
            self.env.drone_active,
            r_col=r_col,
            r_safe=r_safe,
            k_cbf=k_cbf,
        )

        accel_cmd = a_pn + delta_a_rl + a_cbf
        is_interceptor = (self.env.drone_role == 0).unsqueeze(-1)
        accel_cmd = torch.where(is_interceptor, accel_cmd, torch.zeros_like(accel_cmd))
        return accel_cmd

    def step_simulation(self) -> None:
        """Advance one physical simulation step."""
        is_done = bool(self.env.done[0])
        if is_done and self.auto_reset:
            self.env.reset()
            is_done = False
        if not is_done:
            if self.algorithm in {"greedy", "hungarian", "auction"}:
                ctrl = self.baseline_controllers[self.algorithm]
                ctrl.use_estimated_state = self.env.sigma_obs > 0
                accel_cmd, _ = ctrl.act(self.env)
            elif self.algorithm == "attention_mappo":
                accel_cmd = self._act_attention_mappo()
            else:
                ctrl = self.baseline_controllers["auction"]
                accel_cmd, _ = ctrl.act(self.env)
            self.env.step(accel_cmd)

    def step_and_pack(self) -> bytes:
        """Step simulation according to speed_multiplier / paused state, and pack env 0 frame."""
        if self.step_once:
            self.step_once = False
            self.step_simulation()
        elif not self.paused:
            self._step_accum += self.speed_multiplier
            steps_to_run = int(self._step_accum)
            self._step_accum -= steps_to_run
            for _ in range(steps_to_run):
                self.step_simulation()
        return pack_binary_frame(self.env, algo_id=self.algo_id, fps=self.target_fps)

    def pack_current_frame(self) -> bytes:
        """Pack current state into binary frame without advancing simulation time."""
        return pack_binary_frame(self.env, algo_id=self.algo_id, fps=self.target_fps)

    def handle_command(self, cmd: dict) -> dict:
        """Process an interactive command received from HTTP POST or WebSocket."""
        cmd_type = cmd.get("type", "")
        if cmd_type == "set_algorithm":
            algo = str(cmd.get("algorithm", "auction")).lower()
            if algo in ALGO_NAME_TO_ID:
                self.algorithm = algo
                self.algo_id = ALGO_NAME_TO_ID[algo]
                return {"status": "ok", "algorithm": self.algorithm, "algo_id": self.algo_id}
            return {"status": "error", "message": f"Unsupported algorithm: {algo}"}

        elif cmd_type == "spawn_salvo":
            count = int(cmd.get("count", 10))
            speed = cmd.get("speed", None)
            if speed is not None:
                speed = float(speed)
            weave_freq = cmd.get("weave_freq", cmd.get("weave_frequency_rad_s", None))
            if weave_freq is not None:
                weave_freq = float(weave_freq)
            decoy_fraction = cmd.get("decoy_fraction", None)
            if decoy_fraction is not None:
                decoy_fraction = float(decoy_fraction)
            spawned = self.env.spawn_salvo(
                count=count,
                speed=speed,
                weave_frequency_rad_s=weave_freq,
                decoy_fraction=decoy_fraction,
            )
            return {"status": "ok", "spawned": spawned, "M": self.env.M}

        elif cmd_type == "set_params":
            if "sigma_obs" in cmd:
                self.env.sigma_obs = float(cmd["sigma_obs"])
            if "paused" in cmd:
                self.paused = bool(cmd["paused"])
            if "speed_multiplier" in cmd:
                self.speed_multiplier = max(0.05, min(4.0, float(cmd["speed_multiplier"])))
            if cmd.get("step_once", False):
                self.step_once = True
            return {
                "status": "ok",
                "sigma_obs": float(self.env.sigma_obs),
                "paused": bool(self.paused),
                "speed_multiplier": float(self.speed_multiplier),
                "step_once": bool(self.step_once),
            }

        elif cmd_type == "step_once":
            self.step_once = True
            self.step_simulation()
            return {"status": "ok", "step": int(self.env.t[0].item())}

        elif cmd_type == "reset":
            reset_kwargs = {}
            for k in ["num_interceptors", "num_scouts", "num_missiles", "seed", "mode", "weave_enabled", "sigma_obs"]:
                if k in cmd and cmd[k] is not None:
                    reset_kwargs[k] = cmd[k]
            self.env.reset(**reset_kwargs)
            return {
                "status": "ok",
                "step": int(self.env.t[0].item()),
                "N": int(self.env.N),
                "M": int(self.env.M),
            }

        return {"status": "error", "message": f"Unknown command type: {cmd_type}"}

    def get_status(self) -> dict:
        """Return JSON status of the simulation session."""
        metrics = self.env.metrics()
        return {
            "algorithm": self.algorithm,
            "algo_id": self.algo_id,
            "N": int(self.env.N),
            "M": int(self.env.M),
            "P": int(self.env.hva_pos.shape[1]),
            "step": int(self.env.t[0].item()),
            "fps": float(self.target_fps),
            "paused": bool(self.paused),
            "sigma_obs": float(self.env.sigma_obs),
            "speed_multiplier": float(self.speed_multiplier),
            "metrics": {
                "intercept_rate": float(metrics["intercept_rate"][0].item()),
                "leaker_rate": float(metrics["leaker_rate"][0].item()),
                "friendly_collisions": float(metrics["friendly_collisions"][0].item()),
                "delta_v_total": float(metrics["delta_v_total"][0].item()),
            },
        }


def create_app(
    cfg: dict,
    checkpoint_path: Optional[str] = None,
    viewer_dir: str = "viewer3d",
) -> FastAPI:
    """Create and configure FastAPI application for binary streaming and simulation control."""
    app = FastAPI(title="SwarmForge Live Simulation Server")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    session = SandboxSession(cfg, checkpoint_path=checkpoint_path)
    app.state.session = session

    @app.get("/api/status")
    async def api_status():
        return session.get_status()

    @app.post("/api/command")
    async def api_command(cmd: dict):
        return session.handle_command(cmd)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()

        async def sender():
            try:
                while True:
                    t0 = time.perf_counter()
                    frame = session.step_and_pack()
                    await websocket.send_bytes(frame)
                    elapsed = time.perf_counter() - t0
                    dt = 1.0 / max(1.0, session.target_fps)
                    sleep_time = max(0.001, dt - elapsed)
                    await asyncio.sleep(sleep_time)
            except (WebSocketDisconnect, asyncio.CancelledError):
                pass

        async def receiver():
            try:
                while True:
                    data = await websocket.receive_text()
                    try:
                        cmd = json.loads(data)
                        session.handle_command(cmd)
                    except json.JSONDecodeError:
                        pass
            except (WebSocketDisconnect, asyncio.CancelledError):
                pass

        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())
        try:
            done, pending = await asyncio.wait(
                [sender_task, receiver_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
        except WebSocketDisconnect:
            sender_task.cancel()
            receiver_task.cancel()

    # Serve static viewer client if available
    vpath = Path(viewer_dir)
    if not vpath.is_absolute() and not vpath.exists():
        cand = Path(__file__).resolve().parent.parent.parent / viewer_dir
        if cand.exists():
            vpath = cand
    if vpath.exists():
        public_dir = vpath / "public"
        if public_dir.exists():
            assets_dir = public_dir / "assets"
            if assets_dir.exists():
                app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
            draco_dir = public_dir / "draco"
            if draco_dir.exists():
                app.mount("/draco", StaticFiles(directory=str(draco_dir)), name="draco")
        app.mount("/", StaticFiles(directory=str(vpath), html=True), name="viewer3d")

    return app
