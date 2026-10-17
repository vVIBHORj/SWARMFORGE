"""Generate and preview top-down 2D visualization of a scenario and missile trajectories."""

from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

from src.config import load_config
from src.envs.scenario_generator import generate_scenario
from src.envs.threat_models import missile_accel_cmd
from src.physics.kinematics_3d import G0, step_kinematics


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    print(f"Generating preview scenario with seed: {seed}")

    config_path = Path(__file__).resolve().parent.parent / "configs" / "default_sim.yaml"
    cfg = load_config(config_path)

    E = 1
    num_interceptors = 16
    num_scouts = 4
    num_missiles = 30
    device = torch.device("cpu")

    scen = generate_scenario(
        cfg=cfg,
        num_envs=E,
        num_interceptors=num_interceptors,
        num_scouts=num_scouts,
        num_missiles=num_missiles,
        device=device,
        seed=seed,
    )

    # Simulate only missiles for 20 seconds
    dt = float(cfg["environment"]["dt"])
    sim_time = 20.0
    num_steps = int(sim_time / dt)

    a_max = float(cfg["entities"]["missiles"]["max_accel_g"]) * G0
    v_max = float(cfg["entities"]["missiles"]["max_speed"])
    drag = float(cfg["entities"]["missiles"]["linear_drag"])

    pos = scen["missile_pos"].clone()
    vel = scen["missile_vel"].clone()
    hva_pos = scen["hva_pos"]
    target_hva = scen["missile_target_hva"]
    mtype = scen["missile_type"]
    weave_phase = scen["missile_weave_phase"]

    traj_history = [pos[0].numpy().copy()]

    for step in range(num_steps):
        t_sec = step * dt
        a_cmd = missile_accel_cmd(pos, vel, hva_pos, target_hva, mtype, weave_phase, t_sec, cfg)
        pos, vel, _ = step_kinematics(pos, vel, a_cmd, dt, a_max, v_max, drag)
        traj_history.append(pos[0].numpy().copy())

    # Build plot in km
    world_bounds = cfg["environment"]["world_bounds_meters"]
    W_km = world_bounds[0] / 1000.0
    L_km = world_bounds[1] / 1000.0

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

    # 1. World boundary box
    rect = patches.Rectangle(
        (-W_km / 2.0, -L_km / 2.0),
        W_km,
        L_km,
        linewidth=1.5,
        edgecolor="black",
        facecolor="#f8f9fa",
        label="World Boundary",
        zorder=1,
    )
    ax.add_patch(rect)

    # 2. Missile trajectories
    plotted_missile = False
    plotted_decoy = False
    for m in range(num_missiles):
        is_decoy = scen["missile_is_decoy"][0, m].item()
        m_pts = [frame[m] / 1000.0 for frame in traj_history]
        tx = [pt[0] for pt in m_pts]
        ty = [pt[1] for pt in m_pts]

        if is_decoy:
            label = "Decoy Trajectory" if not plotted_decoy else None
            ax.plot(tx, ty, color="#e6a100", linestyle="--", linewidth=1.2, alpha=0.8, label=label, zorder=2)
            plotted_decoy = True
        else:
            label = "Missile Trajectory" if not plotted_missile else None
            ax.plot(tx, ty, color="#d90429", linestyle="-", linewidth=1.0, alpha=0.6, label=label, zorder=2)
            plotted_missile = True

    # 3. Missile spawn points
    spawn_km = traj_history[0] / 1000.0
    ax.scatter(
        spawn_km[:, 0],
        spawn_km[:, 1],
        c="#d90429",
        marker="^",
        s=45,
        edgecolors="black",
        linewidths=0.5,
        label="Missile Spawn",
        zorder=3,
    )

    # 4. Defending drones
    drone_km = scen["drone_pos"][0].numpy() / 1000.0
    ax.scatter(
        drone_km[:, 0],
        drone_km[:, 1],
        c="#0077b6",
        marker="o",
        s=30,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.5,
        label="Drone",
        zorder=4,
    )

    # 5. Defended HVAs
    hva_km = scen["hva_pos"][0].numpy() / 1000.0
    ax.scatter(
        hva_km[:, 0],
        hva_km[:, 1],
        c="#2b9348",
        marker="s",
        s=120,
        edgecolors="black",
        linewidths=1.2,
        label="HVA Target",
        zorder=5,
    )

    margin = 0.8
    ax.set_xlim(-W_km / 2.0 - margin, W_km / 2.0 + margin)
    ax.set_ylim(-L_km / 2.0 - margin, L_km / 2.0 + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (km)", fontsize=11)
    ax.set_ylabel("Y (km)", fontsize=11)
    ax.set_title(f"SwarmForge Scenario Preview (Top-down, Seed={seed})", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=10)

    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / "scenario_preview.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved scenario preview image to {out_path}")


if __name__ == "__main__":
    main()
