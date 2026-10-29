/**
 * TrailAndLockRenderer.js
 * Pre-allocated ring-buffer trajectory trails and dynamic assignment lock-on lines
 * implemented with single GPU draw calls and zero per-frame heap allocations.
 *
 * Spec Section 9.1 & 9.2:
 * - Pre-Allocated Ring-Buffer Trajectory Trails for swarm entities.
 * - Dynamic Assignment Lock-On Lines connecting defending interceptors to assigned threats.
 * - Dual Trail Modes: Tactical vector lines or Atmospheric soft rocket smoke puffs.
 */

import * as THREE from 'three';

export class TrailAndLockRenderer {
  /**
   * @param {THREE.Scene} scene
   * @param {number} maxEntities Combined capacity of drones + missiles (default 1600)
   * @param {number} trailLength Number of historical points per entity (default 24)
   * @param {number} maxLocks Maximum simultaneous lock lines (default 1000)
   */
  constructor(scene, maxEntities = 1600, trailLength = 24, maxLocks = 1000) {
    this.scene = scene;
    this.maxEntities = maxEntities;
    this.trailLength = trailLength;
    this.maxLocks = maxLocks;

    this.showTrails = true;
    this.showLocks = true;
    this.trailMode = 'lines'; // 'lines' | 'smoke'

    this.prevStep = -1;

    // --- 1. Ring-Buffer Trail Data Structures ---
    this.histX = new Float32Array(this.maxEntities * this.trailLength);
    this.histY = new Float32Array(this.maxEntities * this.trailLength);
    this.histZ = new Float32Array(this.maxEntities * this.trailLength);
    this.histCount = new Uint8Array(this.maxEntities);
    this.histHead = new Uint8Array(this.maxEntities);

    // Pre-allocated LineSegments vertex positions and colors:
    const maxTrailSegments = this.maxEntities * (this.trailLength - 1);
    this.trailPositions = new Float32Array(maxTrailSegments * 2 * 3);
    this.trailColors = new Float32Array(maxTrailSegments * 2 * 3);

    this.trailGeometry = new THREE.BufferGeometry();
    this.trailGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(this.trailPositions, 3).setUsage(THREE.DynamicDrawUsage)
    );
    this.trailGeometry.setAttribute(
      'color',
      new THREE.BufferAttribute(this.trailColors, 3).setUsage(THREE.DynamicDrawUsage)
    );

    this.trailLineMaterial = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    this.trailLines = new THREE.LineSegments(this.trailGeometry, this.trailLineMaterial);
    this.trailGeometry.setDrawRange(0, 0);
    this.scene.add(this.trailLines);

    // --- Soft Smoke Particle Trail Mesh ---
    const maxSmokePuffs = this.maxEntities * this.trailLength;
    this.smokePositions = new Float32Array(maxSmokePuffs * 3);
    this.smokeColors = new Float32Array(maxSmokePuffs * 3);
    this.smokeSizes = new Float32Array(maxSmokePuffs);

    this.smokeGeometry = new THREE.BufferGeometry();
    this.smokeGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(this.smokePositions, 3).setUsage(THREE.DynamicDrawUsage)
    );
    this.smokeGeometry.setAttribute(
      'color',
      new THREE.BufferAttribute(this.smokeColors, 3).setUsage(THREE.DynamicDrawUsage)
    );

    // Load cloud/smoke texture
    const texLoader = new THREE.TextureLoader();
    const smokeTex = texLoader.load('/assets/textures/cloud.png');

    this.smokeMaterial = new THREE.PointsMaterial({
      size: 45,
      map: smokeTex,
      transparent: true,
      opacity: 0.65,
      vertexColors: true,
      blending: THREE.NormalBlending,
      depthWrite: false,
    });

    this.smokePoints = new THREE.Points(this.smokeGeometry, this.smokeMaterial);
    this.smokeGeometry.setDrawRange(0, 0);
    this.smokePoints.visible = false;
    this.scene.add(this.smokePoints);

    // --- 2. Dynamic Assignment Lock-On Lines ---
    this.lockPositions = new Float32Array(this.maxLocks * 2 * 3);
    this.lockColors = new Float32Array(this.maxLocks * 2 * 3);

    this.lockGeometry = new THREE.BufferGeometry();
    this.lockGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(this.lockPositions, 3).setUsage(THREE.DynamicDrawUsage)
    );
    this.lockGeometry.setAttribute(
      'color',
      new THREE.BufferAttribute(this.lockColors, 3).setUsage(THREE.DynamicDrawUsage)
    );

    const lockMaterial = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    this.lockLines = new THREE.LineSegments(this.lockGeometry, lockMaterial);
    this.lockGeometry.setDrawRange(0, 0);
    this.scene.add(this.lockLines);
  }

  setTrailMode(mode) {
    this.trailMode = mode === 'smoke' ? 'smoke' : 'lines';
    if (this.trailMode === 'smoke') {
      this.trailLines.visible = true;
      this.trailLineMaterial.opacity = 0.35;
      this.smokePoints.visible = true;
    } else {
      this.trailLines.visible = true;
      this.trailLineMaterial.opacity = 0.85;
      this.smokePoints.visible = false;
    }
  }

  /**
   * Reset all history buffers cleanly (called on episode reset).
   */
  resetHistory() {
    this.histCount.fill(0);
    this.histHead.fill(0);
    this.trailGeometry.setDrawRange(0, 0);
    this.smokeGeometry.setDrawRange(0, 0);
    this.lockGeometry.setDrawRange(0, 0);
  }

  /**
   * Push position into an entity's ring buffer.
   */
  pushPoint(entityIdx, x, y, z) {
    if (entityIdx >= this.maxEntities) return;
    const base = entityIdx * this.trailLength;
    const head = this.histHead[entityIdx];
    this.histX[base + head] = x;
    this.histY[base + head] = y;
    this.histZ[base + head] = z;

    this.histHead[entityIdx] = (head + 1) % this.trailLength;
    if (this.histCount[entityIdx] < this.trailLength) {
      this.histCount[entityIdx]++;
    }
  }

  /**
   * Update trails and lock-on lines for current telemetry frame.
   * @param {Object} frame Decoded telemetry frame
   * @param {boolean} isPaused Whether playback is paused
   */
  update(frame, isPaused = false) {
    if (!frame) return;

    const { header, drones, missiles } = frame;
    const currentStep = header.step;

    // Detect episode reset or backward step
    if (currentStep < this.prevStep || currentStep === 0) {
      this.resetHistory();
    }
    this.prevStep = currentStep;

    const N = drones.length;
    const M = missiles.length;

    // 1. Update Ring-Buffer Histories only if not paused or step advanced
    if (!isPaused) {
      for (let i = 0; i < N && i < this.maxEntities; i++) {
        const d = drones[i];
        if (d.status > 0.5) {
          this.pushPoint(i, d.x, d.z, -d.y);
        } else {
          this.histCount[i] = 0;
        }
      }

      const missileOffset = 800;
      for (let j = 0; j < M && (missileOffset + j) < this.maxEntities; j++) {
        const m = missiles[j];
        const eIdx = missileOffset + j;
        if (m.status === 0.0) {
          this.pushPoint(eIdx, m.x, m.z, -m.y);
        } else {
          this.histCount[eIdx] = 0;
        }
      }
    }

    // 2. Build Trail Geometry
    if (this.showTrails) {
      let vIdx = 0;
      let sIdx = 0;
      const pos = this.trailPositions;
      const col = this.trailColors;
      const sPos = this.smokePositions;
      const sCol = this.smokeColors;

      // Render Drone Trails
      for (let i = 0; i < N && i < this.maxEntities; i++) {
        const count = this.histCount[i];
        if (count < 2) continue;

        const d = drones[i];
        const cr = (d.type === 0) ? 0.0 : 0.06;
        const cg = (d.type === 0) ? 0.94 : 0.72;
        const cb = (d.type === 0) ? 1.0 : 0.5;

        const base = i * this.trailLength;
        const head = this.histHead[i];

        for (let k = 0; k < count - 1; k++) {
          const idx1 = (head - count + k + this.trailLength) % this.trailLength;
          const idx2 = (idx1 + 1) % this.trailLength;

          const p1 = base + idx1;
          const p2 = base + idx2;

          pos[vIdx] = this.histX[p1];
          pos[vIdx + 1] = this.histY[p1];
          pos[vIdx + 2] = this.histZ[p1];

          const alpha1 = (k / count) * 0.75;
          col[vIdx] = cr * alpha1;
          col[vIdx + 1] = cg * alpha1;
          col[vIdx + 2] = cb * alpha1;
          vIdx += 3;

          pos[vIdx] = this.histX[p2];
          pos[vIdx + 1] = this.histY[p2];
          pos[vIdx + 2] = this.histZ[p2];

          const alpha2 = ((k + 1) / count) * 0.95;
          col[vIdx] = cr * alpha2;
          col[vIdx + 1] = cg * alpha2;
          col[vIdx + 2] = cb * alpha2;
          vIdx += 3;
        }
      }

      // Render Missile Trails
      const missileOffset = 800;
      for (let j = 0; j < M && (missileOffset + j) < this.maxEntities; j++) {
        const eIdx = missileOffset + j;
        const count = this.histCount[eIdx];
        if (count < 2) continue;

        const m = missiles[j];
        const isDecoy = (m.type === 3.0);
        const cr = 1.0;
        const cg = isDecoy ? 0.8 : 0.22;
        const cb = isDecoy ? 0.0 : 0.25;

        const base = eIdx * this.trailLength;
        const head = this.histHead[eIdx];

        for (let k = 0; k < count - 1; k++) {
          const idx1 = (head - count + k + this.trailLength) % this.trailLength;
          const idx2 = (idx1 + 1) % this.trailLength;

          const p1 = base + idx1;
          const p2 = base + idx2;

          pos[vIdx] = this.histX[p1];
          pos[vIdx + 1] = this.histY[p1];
          pos[vIdx + 2] = this.histZ[p1];

          const alpha1 = (k / count) * 0.75;
          col[vIdx] = cr * alpha1;
          col[vIdx + 1] = cg * alpha1;
          col[vIdx + 2] = cb * alpha1;
          vIdx += 3;

          pos[vIdx] = this.histX[p2];
          pos[vIdx + 1] = this.histY[p2];
          pos[vIdx + 2] = this.histZ[p2];

          const alpha2 = ((k + 1) / count) * 0.95;
          col[vIdx] = cr * alpha2;
          col[vIdx + 1] = cg * alpha2;
          col[vIdx + 2] = cb * alpha2;
          vIdx += 3;

          // Soft smoke puff particle
          if (this.trailMode === 'smoke' && k % 2 === 0) {
            sPos[sIdx] = this.histX[p2];
            sPos[sIdx + 1] = this.histY[p2];
            sPos[sIdx + 2] = this.histZ[p2];

            const smkAlpha = 0.3 + 0.4 * (k / count);
            sCol[sIdx] = (isDecoy ? 0.9 : 0.8) * smkAlpha;
            sCol[sIdx + 1] = (isDecoy ? 0.7 : 0.75) * smkAlpha;
            sCol[sIdx + 2] = (isDecoy ? 0.4 : 0.78) * smkAlpha;
            sIdx += 3;
          }
        }
      }

      const totalVertices = vIdx / 3;
      this.trailGeometry.setDrawRange(0, totalVertices);
      this.trailGeometry.attributes.position.needsUpdate = true;
      this.trailGeometry.attributes.color.needsUpdate = true;

      if (this.trailMode === 'smoke') {
        const totalSmoke = sIdx / 3;
        this.smokeGeometry.setDrawRange(0, totalSmoke);
        this.smokeGeometry.attributes.position.needsUpdate = true;
        this.smokeGeometry.attributes.color.needsUpdate = true;
      }
    } else {
      this.trailGeometry.setDrawRange(0, 0);
      this.smokeGeometry.setDrawRange(0, 0);
    }

    // 3. Build Dynamic Lock-On Lines
    if (this.showLocks) {
      let lockIdx = 0;
      const lPos = this.lockPositions;
      const lCol = this.lockColors;

      for (let i = 0; i < N; i++) {
        const d = drones[i];
        if (d.status <= 0.5) continue;
        const assignedId = Math.round(d.assigned_id);
        if (assignedId < 0 || assignedId >= M) continue;

        const targetMissile = missiles[assignedId];
        if (!targetMissile || targetMissile.status !== 0.0) continue;

        if (lockIdx / 6 >= this.maxLocks) break;

        // Vertex 1: Interceptor position (cyan)
        lPos[lockIdx] = d.x;
        lPos[lockIdx + 1] = d.z;
        lPos[lockIdx + 2] = -d.y;

        lCol[lockIdx] = 0.0;
        lCol[lockIdx + 1] = 0.94;
        lCol[lockIdx + 2] = 1.0;
        lockIdx += 3;

        // Vertex 2: Assigned Missile position (amber/crimson)
        lPos[lockIdx] = targetMissile.x;
        lPos[lockIdx + 1] = targetMissile.z;
        lPos[lockIdx + 2] = -targetMissile.y;

        lCol[lockIdx] = 1.0;
        lCol[lockIdx + 1] = 0.65;
        lCol[lockIdx + 2] = 0.1;
        lockIdx += 3;
      }

      const totalLockVertices = lockIdx / 3;
      this.lockGeometry.setDrawRange(0, totalLockVertices);
      this.lockGeometry.attributes.position.needsUpdate = true;
      this.lockGeometry.attributes.color.needsUpdate = true;
    } else {
      this.lockGeometry.setDrawRange(0, 0);
    }
  }
}
