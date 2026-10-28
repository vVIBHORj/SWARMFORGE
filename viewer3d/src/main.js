/**
 * main.js
 * Main entrypoint for SwarmForge Three.js 3D Tactical Viewer.
 *
 * Spec Section 9.1 & 9.2 & Step 15:
 * - High-performance WebGL renderer with high-performance power preference.
 * - Zero-copy Float32Array binary frame decoder.
 * - 60 FPS sub-frame kinematic interpolation during 0.1x / slow-motion playback.
 * - Camera presets (Overview, Action Cam, Follow Interceptor, Top-Down Radar View).
 * - Keyboard hotkeys (Space to Pause, S to Step, 1-4 for camera views).
 * - Instanced swarm meshes, ring-buffer trails, dynamic lock lines, and Sandbox HUD.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { InstancedSwarmRenderer } from './InstancedSwarmRenderer.js';
import { TrailAndLockRenderer } from './TrailAndLockRenderer.js';
import { SandboxHUD } from './SandboxHUD.js';

/**
 * Pure zero-copy decoder parsing binary telemetry frame into structured JavaScript object.
 *
 * @param {ArrayBuffer} arrayBuffer Raw bytes received from WebSocket
 * @returns {Object|null} Decoded frame object
 */
export function parseBinaryFrame(arrayBuffer) {
  if (!arrayBuffer || arrayBuffer.byteLength < 64) {
    return null;
  }

  const arr = new Float32Array(arrayBuffer);
  if (arr[0] !== 2026.0) {
    console.warn(`[SwarmForge] Frame magic mismatch: expected 2026.0, got ${arr[0]}`);
    return null;
  }

  const header = {
    magic: arr[0],
    timestamp: arr[1],
    step: Math.round(arr[2]),
    N: Math.round(arr[3]),
    M: Math.round(arr[4]),
    P: Math.round(arr[5]),
    intercept_rate: arr[6],
    leaker_rate: arr[7],
    duplicate_conflicts: Math.round(arr[8]),
    delta_v_total: arr[9],
    friendly_collisions: Math.round(arr[10]),
    algo_id: Math.round(arr[11]),
    sigma_obs: arr[12],
    fps: arr[13],
    world_w: arr[14],
    world_h: arr[15],
  };

  const P = header.P;
  const N = header.N;
  const M = header.M;

  const expectedLength = 16 + 4 * P + 10 * (N + M);
  if (arr.length !== expectedLength) {
    console.warn(`[SwarmForge] Length mismatch: got ${arr.length} floats, expected ${expectedLength}`);
    return null;
  }

  let offset = 16;
  const hvas = [];
  for (let p = 0; p < P; p++) {
    hvas.push({
      id: arr[offset],
      x: arr[offset + 1],
      y: arr[offset + 2],
      z: arr[offset + 3],
    });
    offset += 4;
  }

  const drones = [];
  for (let i = 0; i < N; i++) {
    drones.push({
      id: arr[offset],
      type: arr[offset + 1],
      x: arr[offset + 2],
      y: arr[offset + 3],
      z: arr[offset + 4],
      vx: arr[offset + 5],
      vy: arr[offset + 6],
      vz: arr[offset + 7],
      status: arr[offset + 8],
      assigned_id: arr[offset + 9],
    });
    offset += 10;
  }

  const missiles = [];
  for (let j = 0; j < M; j++) {
    missiles.push({
      id: arr[offset],
      type: arr[offset + 1],
      x: arr[offset + 2],
      y: arr[offset + 3],
      z: arr[offset + 4],
      vx: arr[offset + 5],
      vy: arr[offset + 6],
      vz: arr[offset + 7],
      status: arr[offset + 8],
      assigned_id: arr[offset + 9],
    });
    offset += 10;
  }

  return { header, hvas, drones, missiles, raw: arr };
}

// Global Application Initialization
function initApp() {
  const canvas = document.getElementById('tactical-canvas');
  if (!canvas) return;

  // 1. Three.js Scene, Camera, Renderer
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x060a12);
  scene.fog = new THREE.FogExp2(0x060a12, 0.000035);

  const camera = new THREE.PerspectiveCamera(
    45,
    window.innerWidth / window.innerHeight,
    10,
    80000
  );
  camera.position.set(0, 5200, 10500);

  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    powerPreference: 'high-performance',
  });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxPolarAngle = Math.PI / 2 - 0.02; // Keep camera above ground plane
  controls.minDistance = 50;
  controls.maxDistance = 55000;
  controls.target.set(0, 400, 0);

  // Camera presets
  let cameraMode = 'overview'; // 'overview', 'action', 'follow', 'topdown'

  function applyCameraPreset(mode) {
    cameraMode = mode;
    if (mode === 'overview') {
      controls.target.set(0, 400, 0);
      camera.position.set(0, 5200, 10500);
    } else if (mode === 'action') {
      controls.target.set(0, 750, 0);
      camera.position.set(0, 550, 2400);
    } else if (mode === 'topdown') {
      controls.target.set(0, 0, 0);
      camera.position.set(0, 15000, 50);
    }
    controls.update();

    // Update active button state in HUD
    document.querySelectorAll('.btn-cam').forEach(b => {
      b.classList.toggle('active', b.getAttribute('data-cam') === mode);
    });
  }

  // 2. Tactical Airspace Grid & Bounding Box
  const gridHelper = new THREE.GridHelper(10000, 50, 0x00f0ff, 0x1e293b);
  gridHelper.position.y = 0;
  scene.add(gridHelper);

  const boxGeom = new THREE.BoxGeometry(10000, 3000, 10000);
  const boxWire = new THREE.LineSegments(
    new THREE.EdgesGeometry(boxGeom),
    new THREE.LineBasicMaterial({ color: 0x0ea5e9, transparent: true, opacity: 0.22 })
  );
  boxWire.position.set(0, 1500, 0);
  scene.add(boxWire);

  // 3. Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
  scene.add(ambientLight);

  const dirLight = new THREE.DirectionalLight(0xffffff, 1.3);
  dirLight.position.set(4000, 8000, 4000);
  scene.add(dirLight);

  // 4. Tactical Renderers & Telemetry HUD
  const swarmRenderer = new InstancedSwarmRenderer(scene, 1000, 1000, 16);
  const trailAndLockRenderer = new TrailAndLockRenderer(scene, 1600, 24, 1000);

  let socket = null;
  const sendCommand = (cmdObj) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(cmdObj));
    }
  };

  const hud = new SandboxHUD(sendCommand);

  // Bind checkbox visibility toggles
  if (hud.chkTrails) {
    hud.chkTrails.addEventListener('change', (e) => {
      trailAndLockRenderer.showTrails = e.target.checked;
    });
  }
  if (hud.chkLocks) {
    hud.chkLocks.addEventListener('change', (e) => {
      trailAndLockRenderer.showLocks = e.target.checked;
    });
  }
  if (hud.chkDomes) {
    hud.chkDomes.addEventListener('change', (e) => {
      swarmRenderer.showDomes = e.target.checked;
    });
  }

  // Bind camera view buttons
  document.querySelectorAll('.btn-cam').forEach(btn => {
    btn.addEventListener('click', () => {
      applyCameraPreset(btn.getAttribute('data-cam'));
    });
  });

  // Keyboard Shortcuts: Space (Pause/Resume), S (Step 1 Frame), 1-4 (Camera Views)
  window.addEventListener('keydown', (e) => {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT')) {
      return;
    }
    if (e.code === 'Space') {
      e.preventDefault();
      hud.togglePause();
    } else if (e.code === 'KeyS') {
      e.preventDefault();
      hud.stepOnce();
    } else if (e.code === 'Digit1') {
      applyCameraPreset('overview');
    } else if (e.code === 'Digit2') {
      applyCameraPreset('action');
    } else if (e.code === 'Digit3') {
      applyCameraPreset('follow');
    } else if (e.code === 'Digit4') {
      applyCameraPreset('topdown');
    }
  });

  // 5. WebSocket Telemetry Streaming Connection
  let latestFrame = null;
  let lastPacketTime = performance.now();
  let reconnectTimeout = null;

  // Zero-allocation interpolated frame structure
  const interpFrame = {
    header: null,
    hvas: [],
    drones: [],
    missiles: [],
  };

  function getInterpolatedFrame(baseFrame, subDt, speedMult) {
    interpFrame.header = baseFrame.header;
    interpFrame.hvas = baseFrame.hvas;

    const N = baseFrame.drones.length;
    if (interpFrame.drones.length !== N) {
      interpFrame.drones = new Array(N);
      for (let i = 0; i < N; i++) {
        interpFrame.drones[i] = { ...baseFrame.drones[i] };
      }
    }

    const dArr = interpFrame.drones;
    for (let i = 0; i < N; i++) {
      const src = baseFrame.drones[i];
      const dst = dArr[i];
      dst.id = src.id;
      dst.type = src.type;
      dst.status = src.status;
      dst.assigned_id = src.assigned_id;
      dst.vx = src.vx;
      dst.vy = src.vy;
      dst.vz = src.vz;
      if (src.status > 0.5) {
        dst.x = src.x + src.vx * subDt * speedMult;
        dst.y = src.y + src.vy * subDt * speedMult;
        dst.z = src.z + src.vz * subDt * speedMult;
      } else {
        dst.x = src.x;
        dst.y = src.y;
        dst.z = src.z;
      }
    }

    const M = baseFrame.missiles.length;
    if (interpFrame.missiles.length !== M) {
      interpFrame.missiles = new Array(M);
      for (let j = 0; j < M; j++) {
        interpFrame.missiles[j] = { ...baseFrame.missiles[j] };
      }
    }

    const mArr = interpFrame.missiles;
    for (let j = 0; j < M; j++) {
      const src = baseFrame.missiles[j];
      const dst = mArr[j];
      dst.id = src.id;
      dst.type = src.type;
      dst.status = src.status;
      dst.assigned_id = src.assigned_id;
      dst.vx = src.vx;
      dst.vy = src.vy;
      dst.vz = src.vz;
      if (src.status === 0.0) {
        dst.x = src.x + src.vx * subDt * speedMult;
        dst.y = src.y + src.vy * subDt * speedMult;
        dst.z = src.z + src.vz * subDt * speedMult;
      } else {
        dst.x = src.x;
        dst.y = src.y;
        dst.z = src.z;
      }
    }

    return interpFrame;
  }

  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || 'localhost:8765';
    const wsUrl = `${protocol}//${host}/ws`;

    console.log(`[SwarmForge] Connecting to WebSocket: ${wsUrl}`);
    socket = new WebSocket(wsUrl);
    socket.binaryType = 'arraybuffer';

    socket.onopen = () => {
      console.log('[SwarmForge] WebSocket connected.');
      hud.setConnected(true);
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
        reconnectTimeout = null;
      }
    };

    socket.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        const frame = parseBinaryFrame(event.data);
        if (frame) {
          latestFrame = frame;
          lastPacketTime = performance.now();
        }
      }
    };

    socket.onclose = () => {
      console.warn('[SwarmForge] WebSocket disconnected. Reconnecting in 1.5s...');
      hud.setConnected(false);
      reconnectTimeout = setTimeout(connectWebSocket, 1500);
    };

    socket.onerror = (err) => {
      console.error('[SwarmForge] WebSocket error:', err);
      socket.close();
    };
  }

  connectWebSocket();

  // 6. Window Resize Handler
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  });

  // 7. High-Performance Render Loop
  let lastTime = performance.now();
  let frameCount = 0;
  let fpsTimer = performance.now();
  let currentClientFps = 60;

  function animate() {
    requestAnimationFrame(animate);

    const now = performance.now();
    const dt = Math.min((now - lastTime) / 1000.0, 0.1);
    lastTime = now;

    frameCount++;
    if (now - fpsTimer >= 500) {
      currentClientFps = (frameCount * 1000) / (now - fpsTimer);
      frameCount = 0;
      fpsTimer = now;
    }

    // Kinematic sub-frame interpolation for smooth 60 FPS motion at slow speeds
    let displayFrame = latestFrame;
    if (latestFrame) {
      const speedMult = hud.speedMultiplier || 1.0;
      const isPaused = hud.isPaused;
      const subDt = isPaused ? 0.0 : Math.max(0.0, Math.min((now - lastPacketTime) / 1000.0, 0.5));

      if (!isPaused && speedMult < 0.99 && subDt > 0.001) {
        displayFrame = getInterpolatedFrame(latestFrame, subDt, speedMult);
      }

      // Follow Camera Mode: track primary active interceptor
      if (cameraMode === 'follow' && displayFrame.drones.length > 0) {
        let activeInt = null;
        for (let i = 0; i < displayFrame.drones.length; i++) {
          const d = displayFrame.drones[i];
          if (d.status > 0.5 && d.type === 0) {
            activeInt = d;
            break;
          }
        }
        if (activeInt) {
          const tx = activeInt.x;
          const ty = activeInt.z;
          const tz = -activeInt.y;
          controls.target.lerp(new THREE.Vector3(tx, ty, tz), 0.1);

          const spd = Math.hypot(activeInt.vx, activeInt.vz, -activeInt.vy);
          if (spd > 0.5) {
            const chaseDist = 180.0;
            const chaseHeight = 55.0;
            const cx = tx - (activeInt.vx / spd) * chaseDist;
            const cy = ty + chaseHeight;
            const cz = tz - (-activeInt.vy / spd) * chaseDist;
            camera.position.lerp(new THREE.Vector3(cx, cy, cz), 0.08);
          }
        }
      }

      swarmRenderer.update(displayFrame, dt);
      trailAndLockRenderer.update(displayFrame);
      hud.update(latestFrame, currentClientFps);
    }

    controls.update();
    renderer.render(scene, camera);
  }

  animate();
}

// Start application when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
