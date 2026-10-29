/**
 * MockTelemetry.js
 * Standalone offline demo simulation engine producing exact binary telemetry frames.
 *
 * Implements Demo Mode:
 * - Runs completely client-side in the browser when backend is absent or ?demo=1 is active.
 * - Simulates 2-4 HVAs, N defending interceptors/scouts, and M incoming maneuvering threats.
 * - Simulates 3D kinematics, proportional navigation pursuit, weaving trajectories, and intercepts.
 * - Responds to reset, spawn_salvo, set_params (speed_multiplier, paused), step_once, and scale presets.
 * - Packs state into binary Float32Array frames matching the frozen protocol (magic 2026.0).
 */

export class MockTelemetry {
  constructor(options = {}) {
    this.targetFps = options.fps || 60.0;
    this.dt = 0.05; // 20 Hz physics step
    this.t = 0;
    this.timestamp = 0.0;
    this.paused = false;
    this.speedMultiplier = 1.0;
    this.stepAccumulator = 0.0;
    this.algoId = 2; // Auction + TPN default

    this.numInterceptors = 16;
    this.numScouts = 4;
    this.numMissiles = 30;
    this.numHvas = 2;

    this.hvas = [];
    this.drones = [];
    this.missiles = [];

    this.totalIntercepts = 0;
    this.totalLeakers = 0;
    this.deltaVTotal = 0.0;

    this.reset();
  }

  reset(options = {}) {
    if (options.num_interceptors !== undefined) this.numInterceptors = options.num_interceptors;
    if (options.num_scouts !== undefined) this.numScouts = options.num_scouts;
    if (options.num_missiles !== undefined) this.numMissiles = options.num_missiles;

    this.t = 0;
    this.timestamp = 0.0;
    this.totalIntercepts = 0;
    this.totalLeakers = 0;
    this.deltaVTotal = 0.0;

    // 1. Defended HVAs (spread around center)
    this.hvas = [
      { id: 0, x: -900, y: -400, z: 0 },
      { id: 1, x: 900, y: 500, z: 0 },
    ];
    this.numHvas = this.hvas.length;

    // 2. Drones (Interceptors + Scouts)
    this.drones = [];
    const N = this.numInterceptors + this.numScouts;
    for (let i = 0; i < N; i++) {
      const isScout = i >= this.numInterceptors;
      const angle = (i / N) * Math.PI * 2 + (Math.random() - 0.5) * 0.4;
      const radius = 900 + Math.random() * 2000;
      const alt = isScout ? 1200 + Math.random() * 800 : 350 + Math.random() * 900;
      const x = Math.cos(angle) * radius;
      const y = Math.sin(angle) * radius;

      // Inward patrol velocity (60 - 120 m/s)
      const spd = isScout ? 70 : 110;
      const vx = -Math.cos(angle) * spd;
      const vy = -Math.sin(angle) * spd;
      const vz = 0;

      this.drones.push({
        id: i,
        type: isScout ? 1 : 0,
        x,
        y,
        z: alt,
        vx,
        vy,
        vz,
        status: 1.0, // Active
        assigned_id: -1,
      });
    }

    // 3. Missiles (Multi-Vector 360-degree perimeter salvo)
    this.missiles = [];
    for (let j = 0; j < this.numMissiles; j++) {
      this._spawnSingleMissile(j);
    }

    // Initial target allocation
    this._assignTargets();
  }

  _spawnSingleMissile(id) {
    const angle = Math.random() * Math.PI * 2;
    const perimeterR = 4800 + Math.random() * 200;
    const isLow = Math.random() < 0.35;
    const alt = isLow ? 200 + Math.random() * 300 : 2000 + Math.random() * 900;

    const mx = Math.cos(angle) * perimeterR;
    const my = Math.sin(angle) * perimeterR;

    // Target random HVA
    const targetHva = this.hvas[Math.floor(Math.random() * this.hvas.length)];
    const dx = targetHva.x - mx;
    const dy = targetHva.y - my;
    const dist = Math.hypot(dx, dy);

    const speed = 400 + Math.random() * 50;
    const vx = (dx / dist) * speed;
    const vy = (dy / dist) * speed;
    const vz = -alt / (dist / speed);

    const isDecoy = Math.random() < 0.15;

    this.missiles.push({
      id,
      type: isDecoy ? 3.0 : 2.0,
      x: mx,
      y: my,
      z: alt,
      vx,
      vy,
      vz,
      status: 0.0, // In-flight
      assigned_id: targetHva.id,
      weavePhase: Math.random() * Math.PI * 2,
      weaveFreq: 1.5,
    });
  }

  spawnSalvo(count = 10, speed = 420.0, weaveFreq = 1.5, decoyFraction = 0.15) {
    const startId = this.missiles.length;
    for (let k = 0; k < count; k++) {
      const id = startId + k;
      this._spawnSingleMissile(id);
      const m = this.missiles[id];
      if (speed) {
        const currentSpd = Math.hypot(m.vx, m.vy);
        m.vx = (m.vx / currentSpd) * speed;
        m.vy = (m.vy / currentSpd) * speed;
      }
      if (weaveFreq) m.weaveFreq = weaveFreq;
      if (decoyFraction !== undefined) {
        m.type = Math.random() < decoyFraction ? 3.0 : 2.0;
      }
    }
    this.numMissiles = this.missiles.length;
    this._assignTargets();
    return count;
  }

  _assignTargets() {
    // Greedy nearest-target assignment for drones without target
    const liveMissiles = this.missiles.filter((m) => m.status === 0.0);
    if (liveMissiles.length === 0) return;

    for (let i = 0; i < this.drones.length; i++) {
      const d = this.drones[i];
      if (d.type !== 0 || d.status <= 0.5) continue; // Only active interceptors

      // If current target is dead, find closest live missile
      const currentTarget = this.missiles[d.assigned_id];
      if (!currentTarget || currentTarget.status !== 0.0) {
        let bestDist = Infinity;
        let bestId = -1;
        for (let j = 0; j < liveMissiles.length; j++) {
          const m = liveMissiles[j];
          const dist = Math.hypot(m.x - d.x, m.y - d.y, m.z - d.z);
          if (dist < bestDist) {
            bestDist = dist;
            bestId = m.id;
          }
        }
        d.assigned_id = bestId;
      }
    }
  }

  step() {
    this.t++;
    this.timestamp = this.t * this.dt;

    // 1. Step Missiles
    for (let j = 0; j < this.missiles.length; j++) {
      const m = this.missiles[j];
      if (m.status !== 0.0) continue;

      // Sinusoidal weave acceleration
      m.weavePhase += m.weaveFreq * this.dt;
      const lateralAccel = Math.sin(m.weavePhase) * 35.0; // ~3.5G lateral weave
      const heading = Math.atan2(m.vy, m.vx);
      const ax = -Math.sin(heading) * lateralAccel;
      const ay = Math.cos(heading) * lateralAccel;

      m.vx += ax * this.dt;
      m.vy += ay * this.dt;

      m.x += m.vx * this.dt;
      m.y += m.vy * this.dt;
      m.z += m.vz * this.dt;

      // HVA impact check
      const targetHva = this.hvas[m.assigned_id] || this.hvas[0];
      const distHva = Math.hypot(m.x - targetHva.x, m.y - targetHva.y);
      if (distHva < 80.0 && m.z <= 40.0) {
        m.status = 2.0; // Impacted HVA!
        this.totalLeakers++;
      } else if (m.z <= 0.0 || Math.abs(m.x) > 5200 || Math.abs(m.y) > 5200) {
        m.status = 3.0; // Expired / ground impact
      }
    }

    // 2. Step Drones (Proportional Navigation pursuit)
    this._assignTargets();

    const maxDroneSpeed = 350.0;
    const maxDroneAccel = 117.68; // 12G

    for (let i = 0; i < this.drones.length; i++) {
      const d = this.drones[i];
      if (d.status <= 0.5) continue;

      if (d.type === 0 && d.assigned_id >= 0) {
        const m = this.missiles[d.assigned_id];
        if (m && m.status === 0.0) {
          // Relative kinematics
          const rx = m.x - d.x;
          const ry = m.y - d.y;
          const rz = m.z - d.z;
          const dist = Math.hypot(rx, ry, rz);

          // Intercept check (25m radius)
          if (dist <= 28.0) {
            m.status = 1.0; // Intercepted!
            this.totalIntercepts++;
            d.assigned_id = -1;
            continue;
          }

          // True Proportional Navigation steering
          const relVx = m.vx - d.vx;
          const relVy = m.vy - d.vy;
          const relVz = m.vz - d.vz;
          const closingSpeed = -(rx * relVx + ry * relVy + rz * relVz) / Math.max(1.0, dist);

          const N_gain = 4.0;
          let cmdAx = (rx / Math.max(1.0, dist)) * 25.0 + (relVx / Math.max(1.0, dist)) * N_gain * Math.max(20.0, closingSpeed);
          let cmdAy = (ry / Math.max(1.0, dist)) * 25.0 + (relVy / Math.max(1.0, dist)) * N_gain * Math.max(20.0, closingSpeed);
          let cmdAz = (rz / Math.max(1.0, dist)) * 25.0 + (relVz / Math.max(1.0, dist)) * N_gain * Math.max(20.0, closingSpeed);

          const accelMag = Math.hypot(cmdAx, cmdAy, cmdAz);
          if (accelMag > maxDroneAccel) {
            cmdAx = (cmdAx / accelMag) * maxDroneAccel;
            cmdAy = (cmdAy / accelMag) * maxDroneAccel;
            cmdAz = (cmdAz / accelMag) * maxDroneAccel;
          }

          d.vx += cmdAx * this.dt;
          d.vy += cmdAy * this.dt;
          d.vz += cmdAz * this.dt;
          this.deltaVTotal += accelMag * this.dt;

          const currentSpd = Math.hypot(d.vx, d.vy, d.vz);
          if (currentSpd > maxDroneSpeed) {
            d.vx = (d.vx / currentSpd) * maxDroneSpeed;
            d.vy = (d.vy / currentSpd) * maxDroneSpeed;
            d.vz = (d.vz / currentSpd) * maxDroneSpeed;
          }
        }
      }

      d.x += d.vx * this.dt;
      d.y += d.vy * this.dt;
      d.z += d.vz * this.dt;

      // Keep drones within tactical airspace
      d.z = Math.max(50.0, Math.min(2800.0, d.z));
    }
  }

  /**
   * Handle control commands from HUD.
   */
  handleCommand(cmd) {
    if (!cmd) return;
    const type = cmd.type;

    if (type === 'set_params') {
      if (cmd.paused !== undefined) this.paused = !!cmd.paused;
      if (cmd.speed_multiplier !== undefined) this.speedMultiplier = parseFloat(cmd.speed_multiplier);
    } else if (type === 'step_once') {
      this.step();
    } else if (type === 'spawn_salvo') {
      this.spawnSalvo(cmd.count || 10, cmd.speed, cmd.weave_freq, cmd.decoy_fraction);
    } else if (type === 'reset') {
      this.reset(cmd);
    } else if (type === 'set_algorithm') {
      const map = { greedy: 0, hungarian: 1, auction: 2, attention_mappo: 3 };
      this.algoId = map[cmd.algorithm] ?? 2;
    }
  }

  /**
   * Pack current state into an ArrayBuffer matching the frozen binary protocol.
   * Layout:
   * - Header: 16 floats
   * - HVA Block: P * 4 floats
   * - Drones Block: N * 10 floats
   * - Missiles Block: M * 10 floats
   */
  packBinaryFrame() {
    const N = this.drones.length;
    const M = this.missiles.length;
    const P = this.hvas.length;

    const totalFloats = 16 + 4 * P + 10 * (N + M);
    const buffer = new ArrayBuffer(totalFloats * 4);
    const arr = new Float32Array(buffer);

    // Header (16 floats)
    arr[0] = 2026.0; // Magic
    arr[1] = this.timestamp;
    arr[2] = this.t;
    arr[3] = N;
    arr[4] = M;
    arr[5] = P;
    arr[6] = this.totalIntercepts / Math.max(1, this.totalIntercepts + this.totalLeakers);
    arr[7] = this.totalLeakers / Math.max(1, this.totalIntercepts + this.totalLeakers);
    arr[8] = 0; // duplicate_conflicts
    arr[9] = this.deltaVTotal;
    arr[10] = 0; // friendly_collisions
    arr[11] = this.algoId;
    arr[12] = 15.0; // sigma_obs
    arr[13] = this.targetFps;
    arr[14] = 10000.0; // world_w
    arr[15] = 3000.0;  // world_h

    // HVA Block: P * 4 floats
    let offset = 16;
    for (let p = 0; p < P; p++) {
      const hva = this.hvas[p];
      arr[offset] = hva.id;
      arr[offset + 1] = hva.x;
      arr[offset + 2] = hva.y;
      arr[offset + 3] = hva.z;
      offset += 4;
    }

    // Drone Block: N * 10 floats
    for (let i = 0; i < N; i++) {
      const d = this.drones[i];
      arr[offset] = d.id;
      arr[offset + 1] = d.type;
      arr[offset + 2] = d.x;
      arr[offset + 3] = d.y;
      arr[offset + 4] = d.z;
      arr[offset + 5] = d.vx;
      arr[offset + 6] = d.vy;
      arr[offset + 7] = d.vz;
      arr[offset + 8] = d.status;
      arr[offset + 9] = d.assigned_id;
      offset += 10;
    }

    // Missile Block: M * 10 floats
    for (let j = 0; j < M; j++) {
      const m = this.missiles[j];
      arr[offset] = m.id;
      arr[offset + 1] = m.type;
      arr[offset + 2] = m.x;
      arr[offset + 3] = m.y;
      arr[offset + 4] = m.z;
      arr[offset + 5] = m.vx;
      arr[offset + 6] = m.vy;
      arr[offset + 7] = m.vz;
      arr[offset + 8] = m.status;
      arr[offset + 9] = m.assigned_id;
      offset += 10;
    }

    return buffer;
  }
}

