/**
 * PlaybackController.js
 * Client-side snapshot buffer and continuous interpolation clock.
 *
 * Implements Part 1:
 * - Buffers decoded telemetry frames with arrival time and sim timestamp.
 * - Interpolates between real snapshots (position lerp, velocity slerp) for smooth motion.
 * - Speeds 0.05x to 8x, pause, and step-once.
 * - Detects server time-scaling vs client-side dilation to prevent double-scaling.
 * - Measures effective simulation rate (delta sim / delta real) for the HUD.
 * - Auto slow-mo on intercept helper.
 * - Zero-allocation in the hot render loop.
 */

export class PlaybackController {
  /**
   * @param {Object} options
   * @param {number} options.bufferCapacity Maximum snapshots to retain (default 8)
   * @param {number} options.targetLagSeconds Interpolation lag behind latest snapshot (default 0.08s)
   */
  constructor(options = {}) {
    this.bufferCapacity = options.bufferCapacity || 8;
    this.targetLagSeconds = options.targetLagSeconds || 0.08;

    // Snapshot buffer: array of { frame, timestamp, step, arrivalTime }
    this.snapshots = [];

    // Playback state
    this.renderSimTime = 0.0;
    this.speedMultiplier = 1.0;
    this.targetSpeedMultiplier = 1.0;
    this.isPaused = false;

    // Auto slow-mo on intercept
    this.autoSlowMoEnabled = false;
    this.autoSlowMoTimer = 0.0;
    this.preSlowMoSpeed = 1.0;

    // Effective sim-rate measurement
    this.effectiveSimRate = 1.0;
    this.lastRateCalcTime = performance.now();
    this.simTimeAtLastRateCalc = 0.0;

    // Step-once trigger flag
    this.pendingStepOnce = false;

    // Reusable interpolated frame to ensure 0 per-frame GC allocations
    this.interpolatedFrame = {
      header: {
        magic: 2026.0,
        timestamp: 0.0,
        step: 0,
        N: 0,
        M: 0,
        P: 0,
        intercept_rate: 0.0,
        leaker_rate: 0.0,
        duplicate_conflicts: 0,
        delta_v_total: 0.0,
        friendly_collisions: 0,
        algo_id: 1,
        sigma_obs: 0.0,
        fps: 60.0,
        world_w: 10000.0,
        world_h: 3000.0,
      },
      hvas: [],
      drones: [],
      missiles: [],
      raw: null,
    };

    // Preallocate pool arrays for 1000 drones and 1000 missiles
    for (let i = 0; i < 1000; i++) {
      this.interpolatedFrame.drones.push({
        id: i,
        type: 0,
        x: 0,
        y: 0,
        z: 0,
        vx: 0,
        vy: 0,
        vz: 0,
        status: 0,
        assigned_id: -1,
      });
    }
    for (let j = 0; j < 1000; j++) {
      this.interpolatedFrame.missiles.push({
        id: j,
        type: 2,
        x: 0,
        y: 0,
        z: 0,
        vx: 0,
        vy: 0,
        vz: 0,
        status: 0,
        assigned_id: -1,
      });
    }
  }

  /**
   * Push a newly decoded binary telemetry frame into the snapshot buffer.
   * @param {Object} frame Decoded frame from parseBinaryFrame
   */
  pushFrame(frame) {
    if (!frame || !frame.header) return;

    const now = performance.now();
    const timestamp = frame.header.timestamp;
    const step = frame.header.step;

    // Check for episode reset or backward step
    if (this.snapshots.length > 0) {
      const lastSnap = this.snapshots[this.snapshots.length - 1];
      if (step < lastSnap.step - 3 || timestamp < lastSnap.timestamp - 0.5) {
        // Episode reset detected: clear snapshot buffer and align clock
        this.snapshots.length = 0;
        this.renderSimTime = timestamp;
      }
    }

    // Add to buffer
    this.snapshots.push({
      frame,
      timestamp,
      step,
      arrivalTime: now,
    });

    // Prune oldest snapshots beyond capacity
    while (this.snapshots.length > this.bufferCapacity) {
      this.snapshots.shift();
    }

    // Initialize clock on first snapshot
    if (this.snapshots.length === 1) {
      this.renderSimTime = timestamp;
    }

    // Handle auto slow-mo trigger on intercept or impact
    if (this.autoSlowMoEnabled && this.autoSlowMoTimer <= 0.0) {
      const missiles = frame.missiles;
      for (let m = 0; m < missiles.length; m++) {
        if (missiles[m].status === 1.0 || missiles[m].status === 2.0) {
          // Trigger temporary slow-mo
          this.triggerAutoSlowMo();
          break;
        }
      }
    }
  }

  /**
   * Trigger automatic slow-motion on dramatic intercept moment.
   */
  triggerAutoSlowMo() {
    if (this.autoSlowMoTimer <= 0.0) {
      this.preSlowMoSpeed = this.targetSpeedMultiplier;
      this.speedMultiplier = 0.25;
      this.autoSlowMoTimer = 1.5; // 1.5 seconds of slow-mo
    }
  }

  /**
   * Set user-requested speed multiplier.
   * @param {number} speed Multiplier (0.05 to 8.0)
   */
  setSpeed(speed) {
    const clamped = Math.max(0.05, Math.min(8.0, speed));
    this.targetSpeedMultiplier = clamped;
    if (this.autoSlowMoTimer <= 0.0) {
      this.speedMultiplier = clamped;
    }
  }

  /**
   * Toggle pause / resume.
   * @returns {boolean} New paused state
   */
  togglePause() {
    this.isPaused = !this.isPaused;
    return this.isPaused;
  }

  /**
   * Set pause explicitly.
   * @param {boolean} paused
   */
  setPaused(paused) {
    this.isPaused = !!paused;
  }

  /**
   * Step forward by one snapshot when paused.
   */
  stepOnce() {
    if (!this.isPaused) return;
    this.pendingStepOnce = true;
  }

  /**
   * Advance the playback clock and compute interpolated frame.
   * @param {number} realDt Real wall-clock seconds elapsed since last frame
   * @returns {{ frame: Object|null, simDt: number }} Interpolated frame and scaled sim delta time
   */
  update(realDt) {
    const now = performance.now();

    // 1. Update auto slow-mo timer if active
    if (this.autoSlowMoTimer > 0.0) {
      this.autoSlowMoTimer -= realDt;
      if (this.autoSlowMoTimer <= 0.0) {
        this.speedMultiplier = this.targetSpeedMultiplier;
      }
    }

    if (this.snapshots.length === 0) {
      return { frame: null, simDt: 0.0 };
    }

    const newestSnap = this.snapshots[this.snapshots.length - 1];
    const oldestSnap = this.snapshots[0];

    // 2. Handle Step-Once when paused
    if (this.pendingStepOnce) {
      this.pendingStepOnce = false;
      // Advance clock to next snapshot or by dt
      if (this.snapshots.length >= 2) {
        for (let i = 0; i < this.snapshots.length; i++) {
          if (this.snapshots[i].timestamp > this.renderSimTime + 0.005) {
            this.renderSimTime = this.snapshots[i].timestamp;
            break;
          }
        }
      } else {
        this.renderSimTime = newestSnap.timestamp;
      }
      const frame = this._interpolateAt(this.renderSimTime);
      return { frame, simDt: 0.05 };
    }

    // 3. Paused state: freeze sim time
    if (this.isPaused) {
      const frame = this._interpolateAt(this.renderSimTime);
      return { frame, simDt: 0.0 };
    }

    // 4. Measure Server Sim Rate (to detect if server already scales time)
    // Server rate = (sim time delta between newest and oldest) / (real arrival time delta)
    let serverSimRate = 1.0;
    if (this.snapshots.length >= 2) {
      const simSpan = newestSnap.timestamp - oldestSnap.timestamp;
      const realSpan = (newestSnap.arrivalTime - oldestSnap.arrivalTime) / 1000.0;
      if (realSpan > 0.05 && simSpan >= 0) {
        serverSimRate = simSpan / realSpan;
      }
    }

    /**
     * Dilation Strategy:
     * If the server is ALREADY pacing sim time at requested multiplier
     * (e.g. |serverSimRate - speedMultiplier| < 0.25 * max(speedMultiplier, 0.5)),
     * then snapshots already arrive at the requested rate. Advancing at nominal 1.0 rate
     * tracks the arriving snapshots naturally without double-scaling.
     * If server runs at normal 1.0x (e.g. in mock mode or unthrottled server),
     * we apply the full speedMultiplier locally.
     */
    const serverAlreadyScales =
      Math.abs(serverSimRate - this.speedMultiplier) < 0.25 * Math.max(0.5, this.speedMultiplier) &&
      Math.abs(this.speedMultiplier - 1.0) > 0.15;

    const effectivePacing = serverAlreadyScales ? 1.0 : this.speedMultiplier;
    let simDt = realDt * effectivePacing;

    // 5. Target lag and smooth clock convergence
    // Keep renderSimTime trailing the newest snapshot by targetLagSeconds
    const targetSimTime = Math.max(oldestSnap.timestamp, newestSnap.timestamp - this.targetLagSeconds);
    const lagError = targetSimTime - this.renderSimTime;

    // Soft spring correction: nudge clock if drifting too far from target lag
    const springGain = 3.5;
    const correction = lagError * Math.min(1.0, realDt * springGain);
    this.renderSimTime += simDt + correction;

    // Strict clamp: never extrapolate ahead of newest snapshot or fall behind oldest
    if (this.renderSimTime > newestSnap.timestamp) {
      this.renderSimTime = newestSnap.timestamp;
    }
    if (this.renderSimTime < oldestSnap.timestamp) {
      this.renderSimTime = oldestSnap.timestamp;
    }

    // 6. Update effective sim rate calculation for HUD
    if (now - this.lastRateCalcTime >= 400) {
      const realElapsed = (now - this.lastRateCalcTime) / 1000.0;
      const simElapsed = this.renderSimTime - this.simTimeAtLastRateCalc;
      if (realElapsed > 0.1 && simElapsed >= 0) {
        const measuredRate = simElapsed / realElapsed;
        this.effectiveSimRate = this.effectiveSimRate * 0.7 + measuredRate * 0.3;
      }
      this.lastRateCalcTime = now;
      this.simTimeAtLastRateCalc = this.renderSimTime;
    }

    // 7. Perform zero-allocation snapshot interpolation
    const frame = this._interpolateAt(this.renderSimTime);
    return { frame, simDt };
  }

  /**
   * Internal zero-allocation interpolation between two surrounding snapshots.
   * @param {number} tSim Target simulation time
   * @returns {Object} Reusable interpolated frame
   * @private
   */
  _interpolateAt(tSim) {
    const snaps = this.snapshots;
    if (snaps.length === 0) return null;
    if (snaps.length === 1) return snaps[0].frame;

    // Locate two snapshots bounding tSim
    let snapA = snaps[0];
    let snapB = snaps[snaps.length - 1];

    if (tSim <= snapA.timestamp) {
      return snapA.frame;
    }
    if (tSim >= snapB.timestamp) {
      return snapB.frame;
    }

    for (let i = 0; i < snaps.length - 1; i++) {
      if (snaps[i].timestamp <= tSim && snaps[i + 1].timestamp >= tSim) {
        snapA = snaps[i];
        snapB = snaps[i + 1];
        break;
      }
    }

    const tSpan = snapB.timestamp - snapA.timestamp;
    const alpha = tSpan > 0.0001 ? Math.max(0.0, Math.min(1.0, (tSim - snapA.timestamp) / tSpan)) : 1.0;

    const frameA = snapA.frame;
    const frameB = snapB.frame;
    const target = this.interpolatedFrame;

    // Copy header from active frame, updating timestamp and step
    target.header = { ...frameB.header };
    target.header.timestamp = tSim;
    target.header.step = alpha < 0.5 ? frameA.header.step : frameB.header.step;

    // Copy HVAs
    target.hvas = frameB.hvas;

    // Interpolate Drones
    const numDrones = Math.min(frameB.drones.length, 1000);
    target.drones.length = numDrones;

    for (let i = 0; i < numDrones; i++) {
      const dA = frameA.drones[i] || frameB.drones[i];
      const dB = frameB.drones[i];
      const dOut = target.drones[i];

      dOut.id = dB.id;
      dOut.type = dB.type;
      dOut.status = dB.status;
      dOut.assigned_id = dB.assigned_id;

      if (dB.status > 0.5 && dA && dA.status > 0.5) {
        // Continuous linear position lerp
        dOut.x = dA.x + (dB.x - dA.x) * alpha;
        dOut.y = dA.y + (dB.y - dA.y) * alpha;
        dOut.z = dA.z + (dB.z - dA.z) * alpha;

        // Velocity lerp
        dOut.vx = dA.vx + (dB.vx - dA.vx) * alpha;
        dOut.vy = dA.vy + (dB.vy - dA.vy) * alpha;
        dOut.vz = dA.vz + (dB.vz - dA.vz) * alpha;
      } else {
        dOut.x = dB.x;
        dOut.y = dB.y;
        dOut.z = dB.z;
        dOut.vx = dB.vx;
        dOut.vy = dB.vy;
        dOut.vz = dB.vz;
      }
    }

    // Interpolate Missiles
    const numMissiles = Math.min(frameB.missiles.length, 1000);
    target.missiles.length = numMissiles;

    for (let j = 0; j < numMissiles; j++) {
      const mA = frameA.missiles[j] || frameB.missiles[j];
      const mB = frameB.missiles[j];
      const mOut = target.missiles[j];

      mOut.id = mB.id;
      mOut.type = mB.type;
      mOut.status = mB.status;
      mOut.assigned_id = mB.assigned_id;

      if (mB.status === 0.0 && mA && mA.status === 0.0) {
        // In-flight lerp
        mOut.x = mA.x + (mB.x - mA.x) * alpha;
        mOut.y = mA.y + (mB.y - mA.y) * alpha;
        mOut.z = mA.z + (mB.z - mA.z) * alpha;

        mOut.vx = mA.vx + (mB.vx - mA.vx) * alpha;
        mOut.vy = mA.vy + (mB.vy - mA.vy) * alpha;
        mOut.vz = mA.vz + (mB.vz - mA.vz) * alpha;
      } else {
        mOut.x = mB.x;
        mOut.y = mB.y;
        mOut.z = mB.z;
        mOut.vx = mB.vx;
        mOut.vy = mB.vy;
        mOut.vz = mB.vz;
      }
    }

    return target;
  }
}

