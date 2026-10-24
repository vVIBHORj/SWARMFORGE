/**
 * InstancedSwarmRenderer.js
 * High-performance Three.js InstancedMesh renderer for up to 1,000 drones and 1,000 missiles
 * in 2 GPU draw calls with zero per-frame heap allocations.
 *
 * Spec Section 9.1:
 * - Drones: InstancedMesh with role colors (Interceptor #00f0ff, Scout #10b981).
 * - Missiles: InstancedMesh with threat colors (Real #ff3344, Decoy #ffcc00).
 * - Defended HVAs: Ground glowing pads + translucent hemispheric defense domes.
 * - Dynamic intercept & impact burst visual effects.
 */

import * as THREE from 'three';

export class InstancedSwarmRenderer {
  /**
   * @param {THREE.Scene} scene
   * @param {number} maxDrones
   * @param {number} maxMissiles
   * @param {number} maxHvas
   */
  constructor(scene, maxDrones = 1000, maxMissiles = 1000, maxHvas = 16) {
    this.scene = scene;
    this.maxDrones = maxDrones;
    this.maxMissiles = maxMissiles;
    this.maxHvas = maxHvas;

    // Zero-allocation reusable dummy and math vectors
    this.dummy = new THREE.Object3D();
    this.forwardDir = new THREE.Vector3(0, 0, 1);
    this.velDir = new THREE.Vector3();

    // Color cache
    this.colorInterceptor = new THREE.Color(0x00f0ff); // Cyan
    this.colorScout = new THREE.Color(0x10b981);       // Emerald
    this.colorThreat = new THREE.Color(0xff3344);      // Crimson
    this.colorDecoy = new THREE.Color(0xffcc00);       // Gold

    // 1. Drones InstancedMesh
    // Cone geometry rotated so apex points in +Z (forward direction)
    const droneGeom = new THREE.ConeGeometry(9.0, 26.0, 5);
    droneGeom.rotateX(Math.PI / 2);
    const droneMat = new THREE.MeshStandardMaterial({
      roughness: 0.25,
      metalness: 0.85,
    });
    this.dronesMesh = new THREE.InstancedMesh(droneGeom, droneMat, this.maxDrones);
    this.dronesMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    if (this.dronesMesh.instanceColor) {
      this.dronesMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    }
    this.dronesMesh.count = 0;
    this.scene.add(this.dronesMesh);

    // 2. Missiles InstancedMesh
    const missileGeom = new THREE.ConeGeometry(7.0, 22.0, 5);
    missileGeom.rotateX(Math.PI / 2);
    const missileMat = new THREE.MeshStandardMaterial({
      roughness: 0.2,
      metalness: 0.9,
    });
    this.missilesMesh = new THREE.InstancedMesh(missileGeom, missileMat, this.maxMissiles);
    this.missilesMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    if (this.missilesMesh.instanceColor) {
      this.missilesMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    }
    this.missilesMesh.count = 0;
    this.scene.add(this.missilesMesh);

    // Track missile statuses for burst detection
    this.prevMissileStatus = new Float32Array(this.maxMissiles);
    this.prevMissileStatus.fill(-1.0);

    // 3. Intercept & Impact Burst Markers Pool
    this.maxBursts = 48;
    this.bursts = [];
    const burstGeom = new THREE.SphereGeometry(18, 12, 8);
    const burstMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.9,
      wireframe: true,
    });
    for (let b = 0; b < this.maxBursts; b++) {
      const mesh = new THREE.Mesh(burstGeom, burstMat.clone());
      mesh.visible = false;
      this.scene.add(mesh);
      this.bursts.push({ mesh, life: 0.0, maxLife: 0.35, isImpact: false });
    }

    // 4. Defended HVA Markers & Hemispheric Domes
    this.hvaGroup = new THREE.Group();
    this.scene.add(this.hvaGroup);
    this.hvaObjects = [];

    const baseGeom = new THREE.CylinderGeometry(70, 90, 16, 20);
    const baseMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      emissive: 0x0284c7,
      emissiveIntensity: 0.4,
      metalness: 0.9,
      roughness: 0.3,
    });

    const spireGeom = new THREE.ConeGeometry(24, 70, 8);
    spireGeom.translate(0, 35, 0);
    const spireMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });

    const domeGeom = new THREE.SphereGeometry(800, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2);
    const domeMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.12,
      wireframe: true,
    });

    for (let p = 0; p < this.maxHvas; p++) {
      const g = new THREE.Group();
      const base = new THREE.Mesh(baseGeom, baseMat);
      const spire = new THREE.Mesh(spireGeom, spireMat);
      const dome = new THREE.Mesh(domeGeom, domeMat);
      g.add(base);
      g.add(spire);
      g.add(dome);
      g.visible = false;
      this.hvaGroup.add(g);
      this.hvaObjects.push(g);
    }

    this.showDomes = true;
  }

  /**
   * Spawn a visual burst when an intercept or HVA impact occurs.
   */
  triggerBurst(x, y, z, isImpact = false) {
    for (let i = 0; i < this.maxBursts; i++) {
      const b = this.bursts[i];
      if (b.life <= 0) {
        b.mesh.position.set(x, y, z);
        b.mesh.scale.set(1, 1, 1);
        b.mesh.visible = true;
        b.isImpact = isImpact;
        b.mesh.material.color.setHex(isImpact ? 0xef4444 : 0x00f0ff);
        b.life = b.maxLife;
        break;
      }
    }
  }

  /**
   * Update all entity matrices and colors from decoded telemetry frame.
   * @param {Object} frame Decoded telemetry frame
   * @param {number} deltaSeconds Time elapsed since last render
   */
  update(frame, deltaSeconds = 0.016) {
    if (!frame) return;

    const { header, hvas, drones, missiles } = frame;
    const N = Math.min(drones.length, this.maxDrones);
    const M = Math.min(missiles.length, this.maxMissiles);
    const P = Math.min(hvas.length, this.maxHvas);

    // 1. Update Drones
    this.dronesMesh.count = N;
    for (let i = 0; i < N; i++) {
      const d = drones[i];
      const isActive = d.status > 0.5;

      if (!isActive) {
        this.dummy.position.set(d.x, d.z, -d.y);
        this.dummy.scale.set(0, 0, 0);
        this.dummy.updateMatrix();
        this.dronesMesh.setMatrixAt(i, this.dummy.matrix);
        continue;
      }

      this.dummy.position.set(d.x, d.z, -d.y);
      this.dummy.scale.set(1, 1, 1);

      // Orientation from 3D velocity vector (vx, vz, -vy)
      const spd = Math.hypot(d.vx, d.vz, -d.vy);
      if (spd > 0.5) {
        this.velDir.set(d.vx / spd, d.vz / spd, -d.vy / spd);
        this.dummy.quaternion.setFromUnitVectors(this.forwardDir, this.velDir);
      } else {
        this.dummy.quaternion.identity();
      }

      this.dummy.updateMatrix();
      this.dronesMesh.setMatrixAt(i, this.dummy.matrix);

      // Role color: 0 = interceptor (cyan), 1 = scout (emerald)
      const roleColor = (d.type === 0) ? this.colorInterceptor : this.colorScout;
      this.dronesMesh.setColorAt(i, roleColor);
    }
    this.dronesMesh.instanceMatrix.needsUpdate = true;
    if (this.dronesMesh.instanceColor) {
      this.dronesMesh.instanceColor.needsUpdate = true;
    }

    // 2. Update Missiles
    this.missilesMesh.count = M;
    for (let j = 0; j < M; j++) {
      const m = missiles[j];
      const status = m.status; // 0=in-flight, 1=intercepted, 2=impact, 3=expired, -1=unspawned
      const prevStat = this.prevMissileStatus[j];

      // Detect transition to intercepted or impacted
      if (prevStat === 0.0 && (status === 1.0 || status === 2.0)) {
        this.triggerBurst(m.x, m.z, -m.y, status === 2.0);
      }
      this.prevMissileStatus[j] = status;

      const isInFlight = (status === 0.0);
      if (!isInFlight) {
        this.dummy.position.set(m.x, m.z, -m.y);
        this.dummy.scale.set(0, 0, 0);
        this.dummy.updateMatrix();
        this.missilesMesh.setMatrixAt(j, this.dummy.matrix);
        continue;
      }

      this.dummy.position.set(m.x, m.z, -m.y);
      this.dummy.scale.set(1, 1, 1);

      const spd = Math.hypot(m.vx, m.vz, -m.vy);
      if (spd > 0.5) {
        this.velDir.set(m.vx / spd, m.vz / spd, -m.vy / spd);
        this.dummy.quaternion.setFromUnitVectors(this.forwardDir, this.velDir);
      } else {
        this.dummy.quaternion.identity();
      }

      this.dummy.updateMatrix();
      this.missilesMesh.setMatrixAt(j, this.dummy.matrix);

      // Threat type: 2.0 = real threat (crimson), 3.0 = decoy (gold)
      const typeColor = (m.type === 3.0) ? this.colorDecoy : this.colorThreat;
      this.missilesMesh.setColorAt(j, typeColor);
    }
    this.missilesMesh.instanceMatrix.needsUpdate = true;
    if (this.missilesMesh.instanceColor) {
      this.missilesMesh.instanceColor.needsUpdate = true;
    }

    // 3. Update Defended HVAs
    for (let p = 0; p < this.maxHvas; p++) {
      const g = this.hvaObjects[p];
      if (p < P) {
        const hva = hvas[p];
        g.position.set(hva.x, 0, -hva.y);
        g.visible = true;
        // Dome visibility toggle
        g.children[2].visible = this.showDomes;
      } else {
        g.visible = false;
      }
    }

    // 4. Update Burst Effects
    for (let b = 0; b < this.maxBursts; b++) {
      const burst = this.bursts[b];
      if (burst.life > 0) {
        burst.life -= deltaSeconds;
        const progress = 1.0 - (burst.life / burst.maxLife);
        const s = 1.0 + progress * 5.0;
        burst.mesh.scale.set(s, s, s);
        burst.mesh.material.opacity = Math.max(0.0, 1.0 - progress);
        if (burst.life <= 0) {
          burst.mesh.visible = false;
        }
      }
    }
  }
}
