/**
 * main.js
 * Main entrypoint for SwarmForge Three.js 3D Tactical Viewer.
 *
 * Spec Section 9.1 & 9.2 & Step 15 Upgrades:
 * - High-performance WebGL renderer with high-performance power preference.
 * - Zero-copy Float32Array binary frame decoder conforming to frozen binary protocol.
 * - Client-side snapshot buffer & continuous interpolation clock (PlaybackController).
 * - Multi-facility architectural HVA compounds with window textures, beacons, and damage states (HvaBuildings).
 * - PBR composite drones & hypersonic missiles with 3D banking, roll spin, and point-light burst FX (InstancedSwarmRenderer).
 * - Pre-allocated ring-buffer trajectory trails & dynamic lock lines with soft rocket smoke toggle (TrailAndLockRenderer).
 * - Atmospheric lighting environments (Day / Dusk / Night) and textured terrain.
 * - Post-processing UnrealBloomPass neon glow on engines, lasers, and bursts.
 * - Interactive Speed controls (0.1x to 8x, continuous slider, hotkeys [, ], 0, toast notifications).
 * - Auto-fallback Standalone Demo Mode (MockTelemetry) with URL override (?demo=1).
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

import { InstancedSwarmRenderer } from './InstancedSwarmRenderer.js';
import { TrailAndLockRenderer } from './TrailAndLockRenderer.js';
import { SandboxHUD } from './SandboxHUD.js';
import { PlaybackController } from './PlaybackController.js';
import { HvaBuildings } from './HvaBuildings.js';
import { MockTelemetry } from './MockTelemetry.js';

/**
 * Pure zero-copy decoder parsing binary telemetry frame into structured JavaScript object.
 * Layout: 16-float header, 4*P HVA floats, 10*(N+M) entity floats.
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
    90000
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
  controls.maxDistance = 60000;
  controls.target.set(0, 400, 0);

  // 2. Post-Processing Bloom Setup
  let bloomEnabled = true;
  const composer = new EffectComposer(renderer);
  const renderPass = new RenderPass(scene, camera);
  composer.addPass(renderPass);

  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    0.7, // intensity
    0.35, // radius
    0.85  // threshold
  );
  composer.addPass(bloomPass);

  const outputPass = new OutputPass();
  composer.addPass(outputPass);

  // 3. Lighting & Environments (Night, Dusk, Day)
  const ambientLight = new THREE.AmbientLight(0x1e293b, 0.5);
  scene.add(ambientLight);

  const dirLight = new THREE.DirectionalLight(0x38bdf8, 0.85);
  dirLight.position.set(4000, 7500, 3000);
  scene.add(dirLight);

  function setEnvironment(env) {
    if (env === 'day') {
      scene.background.setHex(0x78a5c2);
      scene.fog.color.setHex(0x8cb6d1);
      scene.fog.density = 0.000025;
      ambientLight.color.setHex(0xcedeee);
      ambientLight.intensity = 1.2;
      dirLight.color.setHex(0xfffaed);
      dirLight.intensity = 2.0;
      dirLight.position.set(4000, 9000, 4000);
    } else if (env === 'dusk') {
      scene.background.setHex(0x1e1528);
      scene.fog.color.setHex(0x1e1528);
      scene.fog.density = 0.00003;
      ambientLight.color.setHex(0x55354e);
      ambientLight.intensity = 0.8;
      dirLight.color.setHex(0xff7b00);
      dirLight.intensity = 1.7;
      dirLight.position.set(8000, 2200, 4000);
    } else {
      // Night (default)
      scene.background.setHex(0x060a14);
      scene.fog.color.setHex(0x060a14);
      scene.fog.density = 0.000035;
      ambientLight.color.setHex(0x1e293b);
      ambientLight.intensity = 0.5;
      dirLight.color.setHex(0x38bdf8);
      dirLight.intensity = 0.85;
      dirLight.position.set(4000, 7500, 3000);
    }
  }

  // 4. Ground Plane with PBR Terrain Texture & Tactical Airspace Grid
  const textureLoader = new THREE.TextureLoader();
  const terrainTex = textureLoader.load('/assets/textures/grasslight-big.jpg');
  terrainTex.wrapS = THREE.RepeatWrapping;
  terrainTex.wrapT = THREE.RepeatWrapping;
  terrainTex.repeat.set(36, 36);

  const terrainNorm = textureLoader.load('/assets/textures/grasslight-big-nm.jpg');
  terrainNorm.wrapS = THREE.RepeatWrapping;
  terrainNorm.wrapT = THREE.RepeatWrapping;
  terrainNorm.repeat.set(36, 36);

  const terrainGeom = new THREE.PlaneGeometry(10000, 10000);
  terrainGeom.rotateX(-Math.PI / 2);
  const terrainMat = new THREE.MeshStandardMaterial({
    map: terrainTex,
    normalMap: terrainNorm,
    roughness: 0.85,
    metalness: 0.1,
    color: 0x334155,
  });
  const terrainMesh = new THREE.Mesh(terrainGeom, terrainMat);
  terrainMesh.position.y = -0.5;
  scene.add(terrainMesh);

  // Overlay tactical grid lines
  const gridHelper = new THREE.GridHelper(10000, 50, 0x00f0ff, 0x1e293b);
  gridHelper.position.y = 0.5;
  scene.add(gridHelper);

  // Tactical bounding box (10km x 3km x 10km airspace)
  const boxGeom = new THREE.BoxGeometry(10000, 3000, 10000);
  const boxWire = new THREE.LineSegments(
    new THREE.EdgesGeometry(boxGeom),
    new THREE.LineBasicMaterial({ color: 0x0ea5e9, transparent: true, opacity: 0.25 })
  );
  boxWire.position.set(0, 1500, 0);
  scene.add(boxWire);

  // 5. Tactical Renderers & Subsystems
  const swarmRenderer = new InstancedSwarmRenderer(scene, 1000, 1000, 16);
  const trailAndLockRenderer = new TrailAndLockRenderer(scene, 1600, 24, 1000);
  const hvaBuildings = new HvaBuildings(scene, 16);
  const playback = new PlaybackController({ bufferCapacity: 8, targetLagSeconds: 0.08 });

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

    document.querySelectorAll('.btn-cam').forEach((b) => {
      b.classList.toggle('active', b.getAttribute('data-cam') === mode);
    });
  }

  // 6. WebSocket Connection & Standalone Demo Mode Setup
  let socket = null;
  let isDemoMode = false;
  let mockTelemetry = null;
  let mockLastTick = performance.now();

  const urlParams = new URLSearchParams(window.location.search);
  const forceDemo = urlParams.get('demo') === '1';

  const sendCommand = (cmdObj) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(cmdObj));
    } else if (isDemoMode && mockTelemetry) {
      // Direct mock telemetry handling
      if (cmdObj.type === 'set_algorithm') {
        const idMap = { greedy: 0, hungarian: 1, auction: 2, attention_mappo: 3 };
        mockTelemetry.algoId = idMap[cmdObj.algorithm] ?? 2;
      } else if (cmdObj.type === 'reset') {
        mockTelemetry.reset(
          cmdObj.num_interceptors || 16,
          cmdObj.num_scouts || 4,
          cmdObj.num_missiles || 30,
          2
        );
        playback.reset();
        trailAndLockRenderer.resetHistory();
      } else if (cmdObj.type === 'spawn_salvo') {
        mockTelemetry.spawnSalvo(
          cmdObj.count || 15,
          cmdObj.speed || 450,
          cmdObj.weave_freq || 1.5,
          cmdObj.decoy_fraction || 0.15
        );
      } else if (cmdObj.type === 'set_params') {
        if (cmdObj.sigma_obs !== undefined) mockTelemetry.sigmaObs = cmdObj.sigma_obs;
      }
    }
  };

  const hud = new SandboxHUD(sendCommand, {
    onSpeedChange: (spd) => playback.setSpeed(spd),
    onPauseChange: (p) => playback.setPaused(p),
    onStepOnce: () => playback.stepOnce(),
    onAutoSlowMoChange: (en) => playback.setAutoSlowMo(en),
    onVisualScaleChange: (sc) => swarmRenderer.setVisualScale(sc),
    onEnvironmentChange: (env) => setEnvironment(env),
    onTrailModeChange: (m) => trailAndLockRenderer.setTrailMode(m),
    onBloomChange: (b) => { bloomEnabled = b; },
  });

  // Checkbox layer toggles
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
      hvaBuildings.showDomes = e.target.checked;
    });
  }

  // Camera buttons
  document.querySelectorAll('.btn-cam').forEach((btn) => {
    btn.addEventListener('click', () => {
      applyCameraPreset(btn.getAttribute('data-cam'));
      btn.blur();
    });
  });

  // Activate Demo Mode
  function activateDemoMode() {
    if (isDemoMode) return;
    console.log('[SwarmForge] Activating standalone demo mode.');
    isDemoMode = true;
    mockTelemetry = new MockTelemetry({ N: 20, M: 30, P: 2 });
    mockLastTick = performance.now();
    hud.setConnected(false, true);
    hud.hideLoadingScreen();
  }

  // 7. WebSocket Streaming Connection with 3-second Auto Demo Fallback
  let connectionTimeout = null;

  function connectWebSocket() {
    if (forceDemo) {
      activateDemoMode();
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || 'localhost:8765';
    const wsUrl = `${protocol}//${host}/ws`;

    console.log(`[SwarmForge] Connecting to WebSocket: ${wsUrl}`);
    socket = new WebSocket(wsUrl);
    socket.binaryType = 'arraybuffer';

    connectionTimeout = setTimeout(() => {
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        console.warn('[SwarmForge] WebSocket connection timed out after 3.0s. Falling back to Demo Mode.');
        activateDemoMode();
      }
    }, 3000);

    socket.onopen = () => {
      console.log('[SwarmForge] WebSocket connected.');
      if (connectionTimeout) {
        clearTimeout(connectionTimeout);
        connectionTimeout = null;
      }
      isDemoMode = false;
      hud.setConnected(true, false);
      hud.hideLoadingScreen();
    };

    socket.onmessage = (event) => {
      hud.hideLoadingScreen();
      if (event.data instanceof ArrayBuffer) {
        const frame = parseBinaryFrame(event.data);
        if (frame) {
          playback.pushFrame(frame);
        }
      }
    };

    socket.onclose = () => {
      console.warn('[SwarmForge] WebSocket closed.');
      if (!isDemoMode) {
        hud.setConnected(false, false);
        // Fallback to demo mode if disconnected
        setTimeout(() => {
          if (!socket || socket.readyState !== WebSocket.OPEN) {
            activateDemoMode();
          }
        }, 1500);
      }
    };

    socket.onerror = (err) => {
      console.error('[SwarmForge] WebSocket error:', err);
      socket.close();
    };
  }

  connectWebSocket();

  // 8. Keyboard Hotkeys
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
    } else if (e.code === 'BracketLeft') {
      e.preventDefault();
      hud.adjustSpeedStep(-1);
    } else if (e.code === 'BracketRight') {
      e.preventDefault();
      hud.adjustSpeedStep(1);
    } else if (e.code === 'Digit0') {
      e.preventDefault();
      hud.setSpeedMultiplier(1.0);
      hud.showSpeedToast('1.00x SPEED');
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

  // 9. Window Resize
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    composer.setSize(window.innerWidth, window.innerHeight);
  });

  // 10. High-Performance Render Loop
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

    // Update demo telemetry if in demo mode
    if (isDemoMode && mockTelemetry) {
      const mockDt = (now - mockLastTick) / 1000.0;
      if (mockDt >= 0.05) { // 20 Hz simulation tick
        mockLastTick = now;
        if (!playback.isPaused) {
          const rawBuf = mockTelemetry.step(mockDt * playback.speedMultiplier);
          const frame = parseBinaryFrame(rawBuf.buffer);
          if (frame) {
            playback.pushFrame(frame);
          }
        }
      }
    }

    // Advance continuous client-side interpolation clock
    const { frame: displayFrame, simDt } = playback.update(dt);

    if (displayFrame) {
      const simTime = displayFrame.header ? displayFrame.header.timestamp : 0.0;

      // Auto slow-mo on intercept trigger
      playback.checkAutoSlowMo(displayFrame);

      // Follow Camera: track primary active interceptor
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
            const chaseDist = 220.0;
            const chaseHeight = 65.0;
            const cx = tx - (activeInt.vx / spd) * chaseDist;
            const cy = ty + chaseHeight;
            const cz = tz - (-activeInt.vy / spd) * chaseDist;
            camera.position.lerp(new THREE.Vector3(cx, cy, cz), 0.08);
          }
        }
      }

      // Update Subsystems
      try {
        swarmRenderer.update(displayFrame, dt, simTime);
      } catch (err) {
        console.warn('[SwarmForge] swarmRenderer update error:', err);
      }
      try {
        hvaBuildings.update(displayFrame, simDt);
      } catch (err) {
        console.warn('[SwarmForge] hvaBuildings update error:', err);
      }
      try {
        trailAndLockRenderer.update(displayFrame, playback.isPaused);
      } catch (err) {
        console.warn('[SwarmForge] trailAndLockRenderer update error:', err);
      }
      try {
        hud.update(displayFrame, currentClientFps, playback.getEffectiveSimRate());
      } catch (err) {
        console.warn('[SwarmForge] hud update error:', err);
      }
    }

    controls.update();

    try {
      if (bloomEnabled) {
        composer.render();
      } else {
        renderer.render(scene, camera);
      }
    } catch (e) {
      console.warn('[SwarmForge] Bloom pass failed, falling back to standard renderer:', e);
      bloomEnabled = false;
      renderer.render(scene, camera);
    }
  }

  animate();
}

// Start application when DOM is ready in browser
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
  } else {
    initApp();
  }
}
