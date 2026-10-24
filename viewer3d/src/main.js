/**
 * main.js
 * Main entrypoint for SwarmForge Three.js 3D Tactical Viewer.
 *
 * Spec Section 9.1 & 9.2:
 * - High-performance WebGL renderer with high-performance power preference.
 * - Zero-copy Float32Array binary frame decoder.
 * - Instanced swarm meshes, ring-buffer trails, dynamic lock lines, and Sandbox HUD.
 * - Bidirectional WebSocket streaming with auto-reconnection.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { InstancedSwarmRenderer } from './InstancedSwarmRenderer.js';
import { TrailAndLockRenderer } from './TrailAndLockRenderer.js';
import { SandboxHUD } from './SandboxHUD.js';

/**
 * Pure zero-copy decoder parsing binary telemetry frame into structured JavaScript object.
 * Layout:
 * - Header (16 floats): magic, timestamp, step, N, M, P, metrics, algo_id, sigma_obs, fps, world_w, world_h
 * - HVA Block (P * 4 floats): id, x, y, z
 * - Drone Block (N * 10 floats): id, type, x, y, z, vx, vy, vz, status, assigned_id
 * - Missile Block (M * 10 floats): id, type, x, y, z, vx, vy, vz, status, assigned_id
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
  scene.fog = new THREE.FogExp2(0x060a12, 0.00004);

  const camera = new THREE.PerspectiveCamera(
    45,
    window.innerWidth / window.innerHeight,
    10,
    70000
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
  controls.minDistance = 100;
  controls.maxDistance = 50000;
  controls.target.set(0, 400, 0);

  // 2. Tactical Airspace Grid & Bounding Box
  // 10km x 10km ground grid with cyan radar aesthetics
  const gridHelper = new THREE.GridHelper(10000, 50, 0x00f0ff, 0x1e293b);
  gridHelper.position.y = 0;
  scene.add(gridHelper);

  // 10km x 10km x 3km airspace box wireframe
  const boxGeom = new THREE.BoxGeometry(10000, 3000, 10000);
  const boxWire = new THREE.LineSegments(
    new THREE.EdgesGeometry(boxGeom),
    new THREE.LineBasicMaterial({ color: 0x0ea5e9, transparent: true, opacity: 0.22 })
  );
  boxWire.position.set(0, 1500, 0);
  scene.add(boxWire);

  // 3. Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambientLight);

  const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
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

  // 5. WebSocket Telemetry Streaming Connection
  let latestFrame = null;
  let reconnectTimeout = null;

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

    controls.update();

    if (latestFrame) {
      swarmRenderer.update(latestFrame, dt);
      trailAndLockRenderer.update(latestFrame);
      hud.update(latestFrame, currentClientFps);
    }

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
