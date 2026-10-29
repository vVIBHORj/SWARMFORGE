/**
 * InstancedSwarmRenderer.js
 * High-performance Three.js InstancedMesh renderer for up to 1,000 drones and 1,000 missiles
 * with procedurally merged composite aerospace geometries, realistic lighting, dynamic attitude,
 * speed-scaled plumes, dynamic point-light burst FX, and zero per-frame heap allocations.
 *
 * Spec Section 9.1 & Step 15 Upgrades:
 * - Interceptors: Sleek Delta-Wing composite UAV with ogive nose, cockpit canopy, swept wings,
 *   wingtip navigation pods (red/port, green/starboard), canted twin V-tails, and glowing thruster nozzle.
 * - Scouts: High-Aspect-Ratio surveillance airframe with twin boom tails, dorsal rotodome radar dish,
 *   and surveillance sensor pods.
 * - Missiles: Hypersonic threat with sharp ogive nose, cylindrical booster, cruciform strakes,
 *   control fins, and rear rocket motor plume with axial roll spin.
 * - Decoys: Electronic warfare decoy airframe with flare pods and wingtip corner reflectors.
 * - Dynamic Attitude: 3D velocity vector alignment with climb/dive pitch, lateral roll banking,
 *   and missile axial roll.
 * - Visual Scaling: Configurable 1x to 20x visual scale factor for tactical airspace clarity.
 * - Dynamic Burst FX: Multi-component expanding shockwave rings, fireball core, and pooled point-light flashes.
 * - Ground Blob Shadows: Instanced soft contact shadows beneath low-altitude aircraft.
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

  // 3. Cockpit canopy ridge
  const canopy = new THREE.SphereGeometry(1.5, 8, 6);
  canopy.scale(0.8, 1.2, 2.2);
  canopy.translate(0, 1.4, 4.0);
  parts.push(canopy);

  // 4. Swept Delta Wings
  const wing = new THREE.BoxGeometry(28.0, 0.6, 11.0);
  wing.translate(0, -0.2, -1.0);
  parts.push(wing);

  // 5. Wingtip navigation light pods
  const leftNavPod = new THREE.CylinderGeometry(0.5, 0.5, 2.5, 6);
  leftNavPod.rotateX(Math.PI / 2);
  leftNavPod.translate(-14.0, -0.2, -1.0);
  parts.push(leftNavPod);

  const rightNavPod = new THREE.CylinderGeometry(0.5, 0.5, 2.5, 6);
  rightNavPod.rotateX(Math.PI / 2);
  rightNavPod.translate(14.0, -0.2, -1.0);
  parts.push(rightNavPod);

  // 6. Twin canted V-tail vertical stabilizers
  const leftTail = new THREE.BoxGeometry(0.4, 5.5, 4.5);
  leftTail.rotateZ(Math.PI * 0.12);
  leftTail.translate(-3.2, 2.5, -7.5);
  parts.push(leftTail);

  const rightTail = new THREE.BoxGeometry(0.4, 5.5, 4.5);
  rightTail.rotateZ(-Math.PI * 0.12);
  rightTail.translate(3.2, 2.5, -7.5);
  parts.push(rightTail);

  // 7. Rear glowing thruster nozzle
  const nozzle = new THREE.CylinderGeometry(1.8, 1.3, 3.0, 8);
  nozzle.rotateX(Math.PI / 2);
  nozzle.translate(0, 0, -10.5);
  parts.push(nozzle);

  // 8. Thruster exhaust plume cone
  const plume = new THREE.ConeGeometry(1.2, 4.5, 8);
  plume.rotateX(-Math.PI / 2);
  plume.translate(0, 0, -13.5);
  parts.push(plume);

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
  rotodome.rotateX(0.18); // Forward tactical tilt
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

  // 5. Rocket motor nozzle and exhaust plume
  const nozzle = new THREE.CylinderGeometry(1.5, 1.2, 2.0, 8);
  nozzle.rotateX(Math.PI / 2);
  nozzle.translate(0, 0, -10.5);
  parts.push(nozzle);

  const plume = new THREE.ConeGeometry(1.5, 7.0, 8);
  plume.rotateX(-Math.PI / 2);
  plume.translate(0, 0, -14.5);
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

  // 5. Decoy motor plume
  const plume = new THREE.ConeGeometry(1.2, 5.0, 6);
  plume.rotateX(-Math.PI / 2);
  plume.translate(0, 0, -9.5);
  parts.push(plume);

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

    // Visual scale factor (1x = true scale, 8x = tactical high-vis)
    this.visualScale = 6.0;

    // Zero-allocation reusable dummy and math vectors
    this.dummy = new THREE.Object3D();
    this.forwardDir = new THREE.Vector3(0, 0, 1);
    this.velDir = new THREE.Vector3();
    this.upDir = new THREE.Vector3(0, 1, 0);

    // Previous velocities for banking roll computation
    this.prevDroneVel = new Float32Array(maxDrones * 3);
    this.prevMissileVel = new Float32Array(maxMissiles * 3);

    // Color definitions
    this.colorInterceptor = new THREE.Color(0x00f0ff); // Electric Cyan
    this.colorScout = new THREE.Color(0x10b981);       // Emerald Green
    this.colorThreat = new THREE.Color(0xff3344);      // Crimson Red
    this.colorDecoy = new THREE.Color(0xffcc00);       // High-Vis Gold

    // 1. Interceptors InstancedMesh (PBR Delta-Wing)
    const interceptorGeom = createInterceptorGeometry();
    const interceptorMat = new THREE.MeshStandardMaterial({
      color: 0x00f0ff,
      roughness: 0.28,
      metalness: 0.85,
      emissive: 0x007799,
      emissiveIntensity: 0.4,
    });
    this.interceptorsMesh = new THREE.InstancedMesh(interceptorGeom, interceptorMat, this.maxDrones);
    this.interceptorsMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    if (this.interceptorsMesh.instanceColor) this.interceptorsMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    this.interceptorsMesh.count = 0;
    this.scene.add(this.interceptorsMesh);

    // Backward-compatible alias
    this.dronesMesh = this.interceptorsMesh;

    // 2. Scouts InstancedMesh (PBR AWACS Airframe)
    const scoutGeom = createScoutGeometry();
    const scoutMat = new THREE.MeshStandardMaterial({
      color: 0x10b981,
      roughness: 0.32,
      metalness: 0.8,
      emissive: 0x047857,
      emissiveIntensity: 0.4,
    });
    this.scoutsMesh = new THREE.InstancedMesh(scoutGeom, scoutMat, this.maxDrones);
    this.scoutsMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    if (this.scoutsMesh.instanceColor) this.scoutsMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    this.scoutsMesh.count = 0;
    this.scene.add(this.scoutsMesh);

    // 3. Hypersonic Missiles InstancedMesh (PBR Threat Airframe)
    const missileGeom = createMissileGeometry();
    const missileMat = new THREE.MeshStandardMaterial({
      color: 0xff3344,
      roughness: 0.22,
      metalness: 0.88,
      emissive: 0xaa1122,
      emissiveIntensity: 0.5,
    });
    this.missilesMesh = new THREE.InstancedMesh(missileGeom, missileMat, this.maxMissiles);
    this.missilesMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    if (this.missilesMesh.instanceColor) this.missilesMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    this.missilesMesh.count = 0;
    this.scene.add(this.missilesMesh);

    // 4. Decoys InstancedMesh (PBR EW Airframe)
    const decoyGeom = createDecoyGeometry();
    const decoyMat = new THREE.MeshStandardMaterial({
      color: 0xffcc00,
      roughness: 0.25,
      metalness: 0.85,
      emissive: 0xb38600,
      emissiveIntensity: 0.5,
    });
    this.decoysMesh = new THREE.InstancedMesh(decoyGeom, decoyMat, this.maxMissiles);
    this.decoysMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    if (this.decoysMesh.instanceColor) this.decoysMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    this.decoysMesh.count = 0;
    this.scene.add(this.decoysMesh);

    // 5. Contact / Blob Shadows on Ground for low altitude aircraft
    const shadowGeom = new THREE.CircleGeometry(16.0, 16);
    shadowGeom.rotateX(-Math.PI / 2);
    const shadowMat = new THREE.MeshBasicMaterial({
      color: 0x000000,
      transparent: true,
      opacity: 0.35,
      depthWrite: false,
    });
    this.shadowsMesh = new THREE.InstancedMesh(shadowGeom, shadowMat, this.maxDrones);
    this.shadowsMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.shadowsMesh.count = 0;
    this.scene.add(this.shadowsMesh);

    // Track missile statuses for burst detection
    this.prevMissileStatus = new Float32Array(this.maxMissiles);
    this.prevMissileStatus.fill(-1.0);

    // 6. Multi-Component Intercept & Impact Burst Markers Pool
    this.maxBursts = 48;
    this.bursts = [];

    const flashGeom = new THREE.SphereGeometry(16, 12, 8);
    const ringGeom = new THREE.RingGeometry(8, 18, 28);
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
      this.bursts.push({ group: g, flash, ring, life: 0.0, maxLife: 0.5, isImpact: false });
    }

    // 7. Dynamic Point Light Pool for Explosions (6 lights)
    this.maxLights = 6;
    this.pointLights = [];
    for (let l = 0; l < this.maxLights; l++) {
      const pLight = new THREE.PointLight(0x00f0ff, 0, 1800, 1.5);
      pLight.visible = false;
      this.scene.add(pLight);
      this.pointLights.push({ light: pLight, life: 0.0, maxLife: 0.35 });
    }

    this.showDomes = true;
  }

  setVisualScale(scale) {
    this.visualScale = Math.max(1.0, Math.min(25.0, scale));
  }

  /**
   * Spawn expanding shockwave ring, bright energy flash, and dynamic point light.
   */
  triggerBurst(x, y, z, isImpact = false) {
    // 1. Visual burst group
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

    // 2. Point light illumination
    for (let l = 0; l < this.maxLights; l++) {
      const pl = this.pointLights[l];
      if (pl.life <= 0) {
        pl.light.position.set(x, Math.max(y + 20, 30), z);
        pl.light.color.setHex(isImpact ? 0xff4411 : 0x00f0ff);
        pl.light.intensity = isImpact ? 16000 : 12000;
        pl.light.visible = true;
        pl.life = pl.maxLife;
        break;
      }
    }
  }

  /**
   * Update all entity matrices and colors from decoded telemetry frame.
   * @param {Object} frame Decoded telemetry frame
   * @param {number} deltaSeconds Time elapsed since last render
   * @param {number} simTime Current simulation time
   */
  update(frame, deltaSeconds = 0.016, simTime = 0.0) {
    if (!frame) return;

    const { drones, missiles } = frame;
    const N = Math.min(drones.length, this.maxDrones);
    const M = Math.min(missiles.length, this.maxMissiles);
    const scale = this.visualScale;

    // 1. Update Drones (Interceptors and Scouts)
    this.interceptorsMesh.count = N;
    this.scoutsMesh.count = N;
    this.shadowsMesh.count = N;

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
        this.shadowsMesh.setMatrixAt(i, this.dummy.matrix);
        continue;
      }

      const posX = d.x;
      const posY = d.z;
      const posZ = -d.y;

      this.dummy.position.set(posX, posY, posZ);
      this.dummy.scale.set(scale, scale, scale);

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
          const bankAngle = Math.max(-0.55, Math.min(0.55, crossY * 20.0));
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

      // Contact blob shadow on ground (y = 0.5)
      if (posY < 900) {
        const shadowScale = Math.max(0.2, (1.0 - posY / 900.0) * scale * 1.5);
        this.dummy.position.set(posX, 0.5, posZ);
        this.dummy.scale.set(shadowScale, 1.0, shadowScale);
        this.dummy.quaternion.identity();
        this.dummy.updateMatrix();
        this.shadowsMesh.setMatrixAt(i, this.dummy.matrix);
      } else {
        this.dummy.scale.set(0, 0, 0);
        this.dummy.updateMatrix();
        this.shadowsMesh.setMatrixAt(i, this.dummy.matrix);
      }
    }
    this.interceptorsMesh.instanceMatrix.needsUpdate = true;
    if (this.interceptorsMesh.instanceColor) this.interceptorsMesh.instanceColor.needsUpdate = true;
    this.scoutsMesh.instanceMatrix.needsUpdate = true;
    if (this.scoutsMesh.instanceColor) this.scoutsMesh.instanceColor.needsUpdate = true;
    this.shadowsMesh.instanceMatrix.needsUpdate = true;

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
      this.dummy.scale.set(scale, scale, scale);

      const vx = m.vx;
      const vy = m.vz;
      const vz = -m.vy;
      const spd = Math.hypot(vx, vy, vz);

      if (spd > 0.5) {
        this.velDir.set(vx / spd, vy / spd, vz / spd);
        this.dummy.quaternion.setFromUnitVectors(this.forwardDir, this.velDir);

        // Continuous axial roll spin for hypersonic threat missile
        const rollAngle = (simTime * 8.0 + j * 0.7) % (Math.PI * 2);
        this.dummy.rotateZ(rollAngle);

        // Banking into lateral weave turns
        const prevIdx = j * 3;
        const pVx = this.prevMissileVel[prevIdx];
        const pVz = this.prevMissileVel[prevIdx + 2];
        const pSpd = Math.hypot(pVx, pVz);
        if (pSpd > 0.5) {
          const crossY = (vx * pVz - vz * pVx) / (spd * pSpd + 1e-4);
          const bankAngle = Math.max(-0.65, Math.min(0.65, crossY * 22.0));
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

    // 3. Update Expanding Shockwave Rings & Particle Bursts
    for (let b = 0; b < this.maxBursts; b++) {
      const burst = this.bursts[b];
      if (burst.life > 0) {
        burst.life -= deltaSeconds;
        const progress = Math.max(0.0, Math.min(1.0, 1.0 - (burst.life / burst.maxLife)));

        // Flash sphere expands and fades
        const sf = 1.0 + progress * 5.0;
        burst.flash.scale.set(sf, sf, sf);
        burst.flash.material.opacity = Math.max(0.0, 1.0 - progress * 1.3);

        // Shockwave ring expands wide across the ground plane
        const sr = 1.0 + progress * 10.0;
        burst.ring.scale.set(sr, sr, sr);
        burst.ring.material.opacity = Math.max(0.0, 1.0 - progress);

        if (burst.life <= 0) {
          burst.group.visible = false;
        }
      }
    }

    // 4. Update Dynamic Point Light Pulses
    for (let l = 0; l < this.maxLights; l++) {
      const pl = this.pointLights[l];
      if (pl.life > 0) {
        pl.life -= deltaSeconds;
        const p = Math.max(0.0, pl.life / pl.maxLife);
        pl.light.intensity = (pl.isImpact ? 16000 : 12000) * p * p;
        if (pl.life <= 0) {
          pl.light.visible = false;
        }
      }
    }
  }
}
