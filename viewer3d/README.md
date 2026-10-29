# SwarmForge 3D Tactical Airspace Viewer

High-performance Three.js WebGL tactical airspace visualizer for large-scale multi-agent drone swarm engagements against hypersonic missile threats.

---

## Key Capabilities

- **Zero-Allocation 60 FPS Engine**: Renders up to 1,000 defending UAVs and 1,000 inbound hypersonic threats using GPU `InstancedMesh` with 0 heap allocations in the hot render loop.
- **Client-Side Snapshot Buffer & Interpolation Clock**: Continuous monotonic simulation clock with multi-frame buffering (`PlaybackController`) eliminating jitter across time dilations from `0.05x` to `8.0x`.
- **Architectural Defended Facilities**: 5 distinct High-Value Asset compounds (HQ Command Tower, Airbase Command & Hangars, Energy Facility & Cooling Towers, Early-Warning Radar Station with rotating dish, and Urban Core District) featuring illuminated window grids, blinking warning beacons, dynamic impact damage states, and hit counter billboards.
- **Realistic Aerospace Aerodynamics**:
  - **Interceptors**: Delta-wing composite UAVs with ogive nose, cockpit canopy ridge, wingtip nav pods (red/green), canted V-tails, and speed-scaled afterburner nozzle.
  - **Scouts**: High-aspect-ratio surveillance gliders with twin boom tails and dorsal rotodome radar dish.
  - **Hypersonic Missiles**: Ogive nose, cylindrical booster, cruciform strakes, control fins, rocket motor plume, and continuous axial roll spin.
  - **EW Decoys**: Forward-swept wings with wingtip flare pods in high-visibility gold.
  - **Kinematic Attitude**: Dynamic velocity heading, climb/dive pitch, and lateral roll banking into turns.
- **Atmospheric Environments & Lighting**:
  - Switchable **Night**, **Dusk**, and **Day** environments with horizon fog and directional sun/moonlight.
  - Textured ground plane (PBR normal map) with overlaid tactical grid coordinates.
  - Post-processing **UnrealBloomPass** for neon emissive engine plumes, laser lock lines, and explosions.
  - Soft rocket smoke puff particle trails or tactical laser vector lines.
  - Contact blob shadows on the ground plane beneath low-altitude aircraft.
- **Red-Team Interactive Sandbox**:
  - Live algorithm hot-swapping (`auction`, `hungarian`, `greedy`, `attention_mappo`).
  - Interactive salvo spawner (missile count, speed 200–650 m/s, weave frequency, decoy fraction).
  - Real-time sensor noise tuning ($\sigma_{\text{obs}}$).
- **Standalone Demo Mode**:
  - Works out of the box with zero backend dependencies (`?demo=1` or auto-fallback after 3s).
  - Simulates 20 defending UAVs, 30 inbound threats, 2 defended facilities with realistic kinematics and intercepts.

---

## Controls & Hotkeys

| Key / Control | Action |
| :--- | :--- |
| **Space** | Toggle Play / Pause |
| **S** | Step 1 frame forward (when paused) |
| **`[`** / **`]`** | Decrease / Increase playback speed (0.1x, 0.25x, 0.5x, 1x, 2x, 4x, 8x) |
| **`0`** | Reset speed to 1.00x |
| **`1`** | Camera: Tactical Overview (high-angle orbital) |
| **`2`** | Camera: Action Cam (low-altitude dynamic view) |
| **`3`** | Camera: Follow Drone (chase-cam tracking lead interceptor) |
| **`4`** | Camera: Top-Down (2D radar tactical map) |
| **Mouse Left Drag** | Orbit camera around tactical target |
| **Mouse Right Drag** | Pan tactical target |
| **Mouse Wheel** | Zoom in / out (50 m to 60,000 m) |

---

## Project Structure

```
viewer3d/
├── index.html                   # Tactical HUD layout, styling, and importmap
├── package.json                 # Vite + Three.js 0.165 build configuration
├── vite.config.js               # Dev server and build pipeline
├── public/
│   ├── draco/                   # Google Draco 3D mesh decoders
│   └── assets/
│       ├── CREDITS.md           # Full asset licensing and source documentation
│       ├── textures/            # PBR terrain, normal map, cloud smoke, and flares
│       └── models/              # Kenney CC0 facility 3D models
├── src/
│   ├── main.js                  # Scene, camera presets, render loop, WebSocket client
│   ├── PlaybackController.js    # Snapshot buffer, monotonic sim clock, auto slow-mo
│   ├── HvaBuildings.js          # 5 facility compounds, beacons, damage states, domes
│   ├── InstancedSwarmRenderer.js# Instanced PBR drones, missiles, banking, point-light bursts
│   ├── TrailAndLockRenderer.js  # Ring-buffer trails (lines/smoke) and lock-on lines
│   ├── SandboxHUD.js            # Interactive DOM overlay and command dispatcher
│   └── MockTelemetry.js         # Standalone client-side demo simulator
└── tests/
    ├── playback.test.js         # Unit tests for PlaybackController interpolation
    └── mock_telemetry.test.js   # Unit tests for standalone MockTelemetry protocol
```

---

## Development & Testing

```bash
# Install dependencies
npm install

# Start Vite live development server
npm run dev

# Run frontend unit tests
node --test tests/*.test.js

# Production bundle build
npm run build
```

---

## Asset Credits & Licenses

All external assets are documented with links, licenses, and dates in [`public/assets/CREDITS.md`](file:///home/vibhor/swarmforge/viewer3d/public/assets/CREDITS.md):
- **Building Models**: Kenney Space Kit & City Kit (Creative Commons CC0 1.0 Universal).
- **PBR Textures**: Three.js Examples Repository (Creative Commons CC0 1.0 Universal / MIT).
- **Draco Decoders**: Google Draco 3D Compression Library (Apache License 2.0).

