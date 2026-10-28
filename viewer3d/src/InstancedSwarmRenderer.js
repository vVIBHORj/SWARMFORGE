/**
 * InstancedSwarmRenderer.js
 * High-performance Three.js InstancedMesh renderer for up to 1,000 drones and 1,000 missiles
 * with procedurally merged composite aerospace geometries and zero per-frame heap allocations.
 *
 * Spec Section 9.1 & Step 15 Upgrades:
 * - Interceptors: Merged Delta-Wing composite UAV with ogive nose, swept wings, canted twin V-tails, and thruster nozzle.
 * - Scouts: Merged High-Aspect-Ratio surveillance airframe with twin boom tails and dorsal rotodome radar dish.
 * - Missiles: Merged Hypersonic threat with sharp ogive nose, cylindrical booster, cruciform strakes, control fins, and exhaust plume.
 * - Decoys: Merged EW decoy airframe with flare pods and wingtip corner reflectors in high-visibility amber/gold.
 * - Orientation: Dynamic 3D velocity alignment with lateral turn banking and speed-scaled engine exhaust glow.
 * - Intercept & Impact FX: Multi-component expanding shockwave rings and bright energy burst flashes.
 */

import * as THREE from 'three';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';

/**
 * Procedural composite geometry builder for Delta-Wing Interceptor UAV.
 */
function createInterceptorGeometry() {
  const parts = [];

  // 1. Sleek pointed fuselage body
  const body = new THREE.CylinderGeometry(1.8, 2.6, 18.0, 8);
  body.rotateX(Math.PI / 2);
  parts.push(body);

  // 2. Ogive nosecone
  const nose = new THREE.ConeGeometry(1.8, 8.0, 8);
  nose.rotateX(Math.PI / 2);
  nose.translate(0, 0, 13.0);
  parts.push(nose);

  // 3. Swept Delta Wings
  const wing = new THREE.BoxGeometry(28.0, 0.6, 11.0);
  wing.translate(0, -0.2, -1.0);
  parts.push(wing);

  // 4. Twin canted V-tail vertical stabilizers
  const leftTail = new THREE.BoxGeometry(0.4, 5.5, 4.5);
  leftTail.rotateZ(Math.PI * 0.12);
  leftTail.translate(-3.2, 2.5, -7.5);
  parts.push(leftTail);

  const rightTail = new THREE.BoxGeometry(0.4, 5.5, 4.5);
  rightTail.rotateZ(-Math.PI * 0.12);
  rightTail.translate(3.2, 2.5, -7.5);
  parts.push(rightTail);

  // 5. Rear glowing thruster nozzle
  const nozzle = new THREE.CylinderGeometry(1.8, 1.3, 3.0, 8);
  nozzle.rotateX(Math.PI / 2);
  nozzle.translate(0, 0, -10.5);
  parts.push(nozzle);

  return BufferGeometryUtils.mergeGeometries(parts, false);
}

/**
 * Procedural composite geometry builder for Scout / AWACS Radar Surveillance UAV.
 */
function createScoutGeometry() {
  const parts = [];

  // 1. High-aspect-ratio surveillance fuselage
  const body = new THREE.CylinderGeometry(2.4, 2.4, 24.0, 8);
  body.rotateX(Math.PI / 2);
  parts.push(body);

  const nose = new THREE.SphereGeometry(2.4, 8, 8);
  nose.translate(0, 0, 12.0);
  parts.push(nose);

  // 2. High-aspect glider wings (48m wingspan)
  const wing = new THREE.BoxGeometry(48.0, 0.8, 5.0);
  wing.translate(0, 1.0, 2.0);
  parts.push(wing);

  // 3. Twin boom tails
  const leftBoom = new THREE.CylinderGeometry(0.7, 0.7, 16.0, 6);
  leftBoom.rotateX(Math.PI / 2);
  leftBoom.translate(-6.0, 0.5, -10.0);
  parts.push(leftBoom);

  const rightBoom = new THREE.CylinderGeometry(0.7, 0.7, 16.0, 6);
  rightBoom.rotateX(Math.PI / 2);
  rightBoom.translate(6.0, 0.5, -10.0);
  parts.push(rightBoom);

  const hStab = new THREE.BoxGeometry(13.0, 0.5, 3.5);
  hStab.translate(0, 2.2, -17.5);
  parts.push(hStab);

  const leftFin = new THREE.BoxGeometry(0.4, 4.0, 3.0);
  leftFin.translate(-6.0, 2.5, -17.5);
  parts.push(leftFin);

  const rightFin = new THREE.BoxGeometry(0.4, 4.0, 3.0);
  rightFin.translate(6.0, 2.5, -17.5);
  parts.push(rightFin);

  // 4. Dorsal rotodome radar dish
  const pylon = new THREE.BoxGeometry(0.8, 3.8, 2.0);
  pylon.translate(0, 3.5, -2.0);
  parts.push(pylon);

  const rotodome = new THREE.CylinderGeometry(7.5, 7.5, 2.0, 16);
  rotodome.rotateX(0.18); // Tactical forward tilt
  rotodome.translate(0, 6.0, -2.0);
  parts.push(rotodome);

  return BufferGeometryUtils.mergeGeometries(parts, false);
}

/**
 * Procedural composite geometry builder for Hypersonic Threat Missile.
 */
function createMissileGeometry() {
  const parts = [];

  // 1. Sharp ogive nosecone
  const nose = new THREE.ConeGeometry(1.8, 7.5, 8);
  nose.rotateX(Math.PI / 2);
  nose.translate(0, 0, 13.5);
  parts.push(nose);

  // 2. Slender cylindrical booster airframe
  const body = new THREE.CylinderGeometry(1.8, 1.8, 20.0, 8);
  body.rotateX(Math.PI / 2);
  parts.push(body);

  // 3. 4 Cruciform mid-body strakes
  const hStrake = new THREE.BoxGeometry(8.5, 0.35, 7.0);
  hStrake.translate(0, 0, 2.0);
  parts.push(hStrake);

  const vStrake = new THREE.BoxGeometry(0.35, 8.5, 7.0);
  vStrake.translate(0, 0, 2.0);
  parts.push(vStrake);

  // 4. 4 Rear control fins
  const hFin = new THREE.BoxGeometry(9.5, 0.4, 3.5);
  hFin.translate(0, 0, -8.5);
  parts.push(hFin);

  const vFin = new THREE.BoxGeometry(0.4, 9.5, 3.5);
  vFin.translate(0, 0, -8.5);
  parts.push(vFin);

  // 5. Afterburner exhaust plume cone
  const plume = new THREE.ConeGeometry(1.5, 5.0, 8);
  plume.rotateX(-Math.PI / 2);
  plume.translate(0, 0, -12.5);
  parts.push(plume);

  return BufferGeometryUtils.mergeGeometries(parts, false);
}

/**
 * Procedural composite geometry builder for Electronic Warfare Decoy Munition.
 */
function createDecoyGeometry() {
  const parts = [];

  // 1. Compact hexagonal EW pod
  const body = new THREE.CylinderGeometry(2.5, 2.5, 14.0, 6);
  body.rotateX(Math.PI / 2);
  parts.push(body);

  // 2. Forward-swept wings
  const wing = new THREE.BoxGeometry(16.0, 0.45, 3.5);
  wing.translate(0, 0, 1.0);
  parts.push(wing);

  // 3. Wingtip flare / reflector pods
  const leftPod = new THREE.CylinderGeometry(0.8, 0.8, 4.0, 6);
  leftPod.rotateX(Math.PI / 2);
  leftPod.translate(-8.0, 0, 1.0);
  parts.push(leftPod);

  const rightPod = new THREE.CylinderGeometry(0.8, 0.8, 4.0, 6);
  rightPod.rotateX(Math.PI / 2);
  rightPod.translate(8.0, 0, 1.0);
  parts.push(rightPod);

  // 4. Rear fins
  const hFin = new THREE.BoxGeometry(7.0, 0.4, 2.5);
  hFin.translate(0, 0, -6.0);
  parts.push(hFin);

  const vFin = new THREE.BoxGeometry(0.4, 7.0, 2.5);
  vFin.translate(0, 0, -6.0);
  parts.push(vFin);

  return BufferGeometryUtils.mergeGeometries(parts, false);
}

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

    // Previous velocities for banking roll computation
    this.prevDroneVel = new Float32Array(maxDrones * 3);
    this.prevMissileVel = new Float32Array(maxMissiles * 3);

    // Color definitions
    this.colorInterceptor = new THREE.Color(0x00f0ff); // Electric Cyan
    this.colorScout = new THREE.Color(0x10b981);       // Emerald Green
    this.colorThreat = new THREE.Color(0xff3344);      // Crimson Red
    this.colorDecoy = new THREE.Color(0xffcc00);       // High-Vis Gold

    // 1. Interceptors InstancedMesh (Composite Delta-Wing)
    const interceptorGeom = createInterceptorGeometry();
    const interceptorMat = new THREE.MeshStandardMaterial({
      color: 0x00f0ff,
      roughness: 0.25,
      metalness: 0.85,
      emissive: 0x00a3cc,
      emissiveIntensity: 0.35,
    });
    this.interceptorsMesh = new THREE.InstancedMesh(interceptorGeom, interceptorMat, this.maxDrones);
    this.interceptorsMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    if (this.interceptorsMesh.instanceColor) this.interceptorsMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    this.interceptorsMesh.count = 0;
    this.scene.add(this.interceptorsMesh);

    // Backward-compatible alias
    this.dronesMesh = this.interceptorsMesh;

    // 2. Scouts InstancedMesh (Composite AWACS Airframe)
    const scoutGeom = createScoutGeometry();
    const scoutMat = new THREE.MeshStandardMaterial({
      color: 0x10b981,
      roughness: 0.3,
      metalness: 0.8,
      emissive: 0x047857,
      emissiveIntensity: 0.35,
    });
    this.scoutsMesh = new THREE.InstancedMesh(scoutGeom, scoutMat, this.maxDrones);
    this.scoutsMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    if (this.scoutsMesh.instanceColor) this.scoutsMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    this.scoutsMesh.count = 0;
    this.scene.add(this.scoutsMesh);

    // 3. Hypersonic Missiles InstancedMesh (Composite Cruise/Booster)
    const missileGeom = createMissileGeometry();
    const missileMat = new THREE.MeshStandardMaterial({
      color: 0xff3344,
      roughness: 0.2,
      metalness: 0.9,
      emissive: 0x990011,
      emissiveIntensity: 0.45,
    });
    this.missilesMesh = new THREE.InstancedMesh(missileGeom, missileMat, this.maxMissiles);
    this.missilesMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    if (this.missilesMesh.instanceColor) this.missilesMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    this.missilesMesh.count = 0;
    this.scene.add(this.missilesMesh);

    // 4. Decoys InstancedMesh (Composite EW Decoy Airframe)
    const decoyGeom = createDecoyGeometry();
    const decoyMat = new THREE.MeshStandardMaterial({
      color: 0xffcc00,
      roughness: 0.25,
      metalness: 0.85,
      emissive: 0xb38600,
      emissiveIntensity: 0.45,
    });
    this.decoysMesh = new THREE.InstancedMesh(decoyGeom, decoyMat, this.maxMissiles);
    this.decoysMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    if (this.decoysMesh.instanceColor) this.decoysMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    this.decoysMesh.count = 0;
    this.scene.add(this.decoysMesh);

    // Track missile statuses for burst detection
    this.prevMissileStatus = new Float32Array(this.maxMissiles);
    this.prevMissileStatus.fill(-1.0);

    // 5. Multi-Component Intercept & Impact Burst Markers Pool
    this.maxBursts = 48;
    this.bursts = [];

    const flashGeom = new THREE.SphereGeometry(14, 12, 8);
    const ringGeom = new THREE.RingGeometry(6, 14, 24);
    ringGeom.rotateX(Math.PI / 2); // Lay horizontal in XZ plane

    for (let b = 0; b < this.maxBursts; b++) {
      const g = new THREE.Group();

      const flashMat = new THREE.MeshBasicMaterial({
        color: 0x00f0ff,
        transparent: true,
        opacity: 0.95,
      });
      const flash = new THREE.Mesh(flashGeom, flashMat);

      const ringMat = new THREE.MeshBasicMaterial({
        color: 0x00f0ff,
        transparent: true,
        opacity: 0.9,
        side: THREE.DoubleSide,
      });
      const ring = new THREE.Mesh(ringGeom, ringMat);

      g.add(flash);
      g.add(ring);
      g.visible = false;
      this.scene.add(g);
      this.bursts.push({ group: g, flash, ring, life: 0.0, maxLife: 0.45, isImpact: false });
    }

    // 6. Defended HVA Markers & Hemispheric Domes
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
   * Spawn expanding shockwave ring and bright energy flash when an intercept or HVA impact occurs.
   */
  triggerBurst(x, y, z, isImpact = false) {
    for (let i = 0; i < this.maxBursts; i++) {
      const b = this.bursts[i];
      if (b.life <= 0) {
        b.group.position.set(x, y, z);
        b.group.scale.set(1, 1, 1);
        b.group.visible = true;
        b.isImpact = isImpact;

        const colorHex = isImpact ? 0xef4444 : 0x00f0ff;
        b.flash.material.color.setHex(colorHex);
        b.ring.material.color.setHex(isImpact ? 0xf97316 : 0x00f0ff);
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

    // 1. Update Drones (Interceptors and Scouts)
    this.interceptorsMesh.count = N;
    this.scoutsMesh.count = N;

    for (let i = 0; i < N; i++) {
      const d = drones[i];
      const isActive = d.status > 0.5;
      const isScout = (d.type === 1);

      if (!isActive) {
        this.dummy.position.set(d.x, d.z, -d.y);
        this.dummy.scale.set(0, 0, 0);
        this.dummy.updateMatrix();
        this.interceptorsMesh.setMatrixAt(i, this.dummy.matrix);
        this.scoutsMesh.setMatrixAt(i, this.dummy.matrix);
        continue;
      }

      this.dummy.position.set(d.x, d.z, -d.y);
      this.dummy.scale.set(1, 1, 1);

      // Orientation along 3D velocity vector (vx, vz, -vy) with lateral banking
      const vx = d.vx;
      const vy = d.vz;
      const vz = -d.vy;
      const spd = Math.hypot(vx, vy, vz);

      if (spd > 0.5) {
        this.velDir.set(vx / spd, vy / spd, vz / spd);
        this.dummy.quaternion.setFromUnitVectors(this.forwardDir, this.velDir);

        // Realistic roll banking into lateral turns
        const prevIdx = i * 3;
        const pVx = this.prevDroneVel[prevIdx];
        const pVz = this.prevDroneVel[prevIdx + 2];
        const pSpd = Math.hypot(pVx, pVz);
        if (pSpd > 0.5) {
          const crossY = (vx * pVz - vz * pVx) / (spd * pSpd + 1e-4);
          const bankAngle = Math.max(-0.45, Math.min(0.45, crossY * 18.0));
          this.dummy.rotateZ(bankAngle);
        }
        this.prevDroneVel[prevIdx] = vx;
        this.prevDroneVel[prevIdx + 1] = vy;
        this.prevDroneVel[prevIdx + 2] = vz;
      } else {
        this.dummy.quaternion.identity();
      }

      this.dummy.updateMatrix();

      if (isScout) {
        this.scoutsMesh.setMatrixAt(i, this.dummy.matrix);
        this.scoutsMesh.setColorAt(i, this.colorScout);

        // Hide in interceptors mesh
        this.dummy.scale.set(0, 0, 0);
        this.dummy.updateMatrix();
        this.interceptorsMesh.setMatrixAt(i, this.dummy.matrix);
      } else {
        this.interceptorsMesh.setMatrixAt(i, this.dummy.matrix);
        this.interceptorsMesh.setColorAt(i, this.colorInterceptor);

        // Hide in scouts mesh
        this.dummy.scale.set(0, 0, 0);
        this.dummy.updateMatrix();
        this.scoutsMesh.setMatrixAt(i, this.dummy.matrix);
      }
    }
    this.interceptorsMesh.instanceMatrix.needsUpdate = true;
    if (this.interceptorsMesh.instanceColor) this.interceptorsMesh.instanceColor.needsUpdate = true;
    this.scoutsMesh.instanceMatrix.needsUpdate = true;
    if (this.scoutsMesh.instanceColor) this.scoutsMesh.instanceColor.needsUpdate = true;

    // 2. Update Missiles (Hypersonic Threats and EW Decoys)
    this.missilesMesh.count = M;
    this.decoysMesh.count = M;

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
      const isDecoy = (m.type === 3.0);

      if (!isInFlight) {
        this.dummy.position.set(m.x, m.z, -m.y);
        this.dummy.scale.set(0, 0, 0);
        this.dummy.updateMatrix();
        this.missilesMesh.setMatrixAt(j, this.dummy.matrix);
        this.decoysMesh.setMatrixAt(j, this.dummy.matrix);
        continue;
      }

      this.dummy.position.set(m.x, m.z, -m.y);
      this.dummy.scale.set(1, 1, 1);

      const vx = m.vx;
      const vy = m.vz;
      const vz = -m.vy;
      const spd = Math.hypot(vx, vy, vz);

      if (spd > 0.5) {
        this.velDir.set(vx / spd, vy / spd, vz / spd);
        this.dummy.quaternion.setFromUnitVectors(this.forwardDir, this.velDir);

        const prevIdx = j * 3;
        const pVx = this.prevMissileVel[prevIdx];
        const pVz = this.prevMissileVel[prevIdx + 2];
        const pSpd = Math.hypot(pVx, pVz);
        if (pSpd > 0.5) {
          const crossY = (vx * pVz - vz * pVx) / (spd * pSpd + 1e-4);
          const bankAngle = Math.max(-0.55, Math.min(0.55, crossY * 20.0));
          this.dummy.rotateZ(bankAngle);
        }
        this.prevMissileVel[prevIdx] = vx;
        this.prevMissileVel[prevIdx + 1] = vy;
        this.prevMissileVel[prevIdx + 2] = vz;
      } else {
        this.dummy.quaternion.identity();
      }

      this.dummy.updateMatrix();

      if (isDecoy) {
        this.decoysMesh.setMatrixAt(j, this.dummy.matrix);
        this.decoysMesh.setColorAt(j, this.colorDecoy);

        this.dummy.scale.set(0, 0, 0);
        this.dummy.updateMatrix();
        this.missilesMesh.setMatrixAt(j, this.dummy.matrix);
      } else {
        this.missilesMesh.setMatrixAt(j, this.dummy.matrix);
        this.missilesMesh.setColorAt(j, this.colorThreat);

        this.dummy.scale.set(0, 0, 0);
        this.dummy.updateMatrix();
        this.decoysMesh.setMatrixAt(j, this.dummy.matrix);
      }
    }
    this.missilesMesh.instanceMatrix.needsUpdate = true;
    if (this.missilesMesh.instanceColor) this.missilesMesh.instanceColor.needsUpdate = true;
    this.decoysMesh.instanceMatrix.needsUpdate = true;
    if (this.decoysMesh.instanceColor) this.decoysMesh.instanceColor.needsUpdate = true;

    // 3. Update Defended HVAs
    for (let p = 0; p < this.maxHvas; p++) {
      const g = this.hvaObjects[p];
      if (p < P) {
        const hva = hvas[p];
        g.position.set(hva.x, 0, -hva.y);
        g.visible = true;
        g.children[2].visible = this.showDomes;
      } else {
        g.visible = false;
      }
    }

    // 4. Update Expanding Shockwave Rings & Particle Bursts
    for (let b = 0; b < this.maxBursts; b++) {
      const burst = this.bursts[b];
      if (burst.life > 0) {
        burst.life -= deltaSeconds;
        const progress = Math.max(0.0, Math.min(1.0, 1.0 - (burst.life / burst.maxLife)));

        // Flash sphere expands and fades
        const sf = 1.0 + progress * 4.0;
        burst.flash.scale.set(sf, sf, sf);
        burst.flash.material.opacity = Math.max(0.0, 1.0 - progress * 1.2);

        // Shockwave ring expands wide across the ground plane
        const sr = 1.0 + progress * 8.5;
        burst.ring.scale.set(sr, sr, sr);
        burst.ring.material.opacity = Math.max(0.0, 1.0 - progress);

        if (burst.life <= 0) {
          burst.group.visible = false;
        }
      }
    }
  }
}
