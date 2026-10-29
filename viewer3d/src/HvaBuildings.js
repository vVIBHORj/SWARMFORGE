/**
 * HvaBuildings.js
 * Realistic Defended High-Value Asset (HVA) Facility Compounds and Damage States.
 *
 * Implements Part 2:
 * - Replaces simple primitives with 5 distinct architectural compounds:
 *   1. HQ Command Tower (stepped mid-rise tower, antenna mast, comms domes, operations wings).
 *   2. Airbase Command (barrel-roof hangars, control tower with glazed cab, runway strip).
 *   3. Energy Facility / Refinery (cooling towers, containment dome, storage tanks, turbine hall).
 *   4. Early-Warning Radar Complex (giant radome sphere, rotating parabolic dish, operations bunker).
 *   5. Urban Core District (cluster of 3-4 commercial mid-rise glass towers with rooftop helipads).
 * - Legible at 10km scale (~350-450m compound footprint, 80-220m tall).
 * - Window-grid canvas textures with nighttime emissive glow.
 * - Blinking red aircraft warning beacons on tower masts.
 * - Translucent hemispheric defense domes (R = 800m).
 * - Damage state: flashes red on missile impact, darkens structure, spawns lingering smoke/fire columns.
 * - Per-HVA hit counter billboard sprite.
 * - Low draw-call overhead: merges sub-geometries per material group.
 */

import * as THREE from 'three';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';

/**
 * Procedural canvas texture generator for illuminated office/facility windows.
 */
function createWindowTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  // Background concrete/steel wall
  ctx.fillStyle = '#1e293b';
  ctx.fillRect(0, 0, 512, 512);

  // Structural panel lines
  ctx.strokeStyle = '#0f172a';
  ctx.lineWidth = 4;
  for (let x = 0; x <= 512; x += 32) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, 512);
    ctx.stroke();
  }
  for (let y = 0; y <= 512; y += 32) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(512, y);
    ctx.stroke();
  }

  // Windows grid
  for (let x = 6; x < 512; x += 32) {
    for (let y = 6; y < 512; y += 32) {
      const rand = Math.random();
      if (rand < 0.65) {
        // Lit window (warm yellow or tactical cyan)
        ctx.fillStyle = rand < 0.5 ? '#fef08a' : '#38bdf8';
      } else {
        // Dark unlit window
        ctx.fillStyle = '#090d16';
      }
      ctx.fillRect(x, y, 20, 20);
    }
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(2, 4);
  return texture;
}

/**
 * Procedural canvas texture for concrete tarmac pad with helipad marking.
 */
function createTarmacTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  // Concrete tarmac
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, 0, 512, 512);

  // Perimeter boundary line
  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth = 8;
  ctx.strokeRect(16, 16, 480, 480);

  // Helipad circle
  ctx.beginPath();
  ctx.arc(256, 256, 120, 0, Math.PI * 2);
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 12;
  ctx.stroke();

  // Helipad 'H'
  ctx.fillStyle = '#e2e8f0';
  ctx.font = 'bold 130px ui-monospace, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('H', 256, 256);

  const texture = new THREE.CanvasTexture(canvas);
  return texture;
}

/**
 * Create text billboard sprite for HVA identification and hit counter.
 */
function createLabelSprite(text) {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = 'rgba(10, 16, 28, 0.85)';
  ctx.strokeStyle = '#00f0ff';
  ctx.lineWidth = 4;
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(8, 8, 496, 112, 16);
  } else {
    ctx.rect(8, 8, 496, 112);
  }
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#00f0ff';
  ctx.font = 'bold 44px ui-monospace, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 256, 64);

  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(380, 95, 1);
  return { sprite, canvas, ctx, texture };
}

export class HvaBuildings {
  /**
   * @param {THREE.Scene} scene
   * @param {number} maxHvas
   */
  constructor(scene, maxHvas = 16) {
    this.scene = scene;
    this.maxHvas = maxHvas;
    this.showDomes = true;

    // Shared procedural textures
    this.windowTex = createWindowTexture();
    this.tarmacTex = createTarmacTexture();

    // Materials
    this.matBuilding = new THREE.MeshStandardMaterial({
      color: 0x94a3b8,
      map: this.windowTex,
      emissive: 0x38bdf8,
      emissiveIntensity: 0.15,
      roughness: 0.4,
      metalness: 0.6,
    });

    this.matIndustrial = new THREE.MeshStandardMaterial({
      color: 0x64748b,
      roughness: 0.6,
      metalness: 0.5,
    });

    this.matTarmac = new THREE.MeshStandardMaterial({
      map: this.tarmacTex,
      roughness: 0.8,
      metalness: 0.2,
    });

    this.matBeacon = new THREE.MeshBasicMaterial({ color: 0xef4444 });

    // Compound groups
    this.compounds = [];
    this.rootGroup = new THREE.Group();
    this.scene.add(this.rootGroup);

    // Damage particle columns pool
    this.smokeParticles = [];

    this._initCompounds();
  }

  _initCompounds() {
    const domeGeom = new THREE.SphereGeometry(800, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2);
    const domeMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.12,
      wireframe: true,
    });

    for (let p = 0; p < this.maxHvas; p++) {
      const g = new THREE.Group();

      // 1. Tarmac ground pad
      const padGeom = new THREE.PlaneGeometry(420, 420);
      padGeom.rotateX(-Math.PI / 2);
      const pad = new THREE.Mesh(padGeom, this.matTarmac);
      pad.position.y = 1.0;
      g.add(pad);

      // 2. Architectural building model by type (p % 5)
      const typeIdx = p % 5;
      const model = this._buildCompoundModel(typeIdx);
      g.add(model);

      // 3. Blinking beacon light at mast top
      const beaconGeom = new THREE.SphereGeometry(5, 8, 8);
      const beacon = new THREE.Mesh(beaconGeom, this.matBeacon.clone());
      beacon.position.set(0, 185, 0);
      g.add(beacon);

      // 4. Rotating radar dish (if applicable)
      let rotatingPart = null;
      if (typeIdx === 3) {
        rotatingPart = model.getObjectByName('radarDish');
      }

      // 5. Translucent Hemispheric Defense Dome
      const dome = new THREE.Mesh(domeGeom, domeMat);
      dome.position.y = 0;
      g.add(dome);

      // 6. Hit Counter & Status Billboard Label
      const labelObj = createLabelSprite(`HVA-${p + 1} DEFENSE SITE`);
      labelObj.sprite.position.set(0, 260, 0);
      g.add(labelObj.sprite);

      // 7. Fire & Smoke damage column
      const smokeColumn = this._createSmokeColumn();
      smokeColumn.group.position.set(0, 10, 0);
      smokeColumn.group.visible = false;
      g.add(smokeColumn.group);

      g.visible = false;
      this.rootGroup.add(g);

      this.compounds.push({
        group: g,
        dome,
        model,
        beacon,
        rotatingPart,
        labelObj,
        smokeColumn,
        typeIdx,
        hits: 0,
        damageFlashTimer: 0.0,
      });
    }
  }

  _buildCompoundModel(type) {
    const container = new THREE.Group();

    if (type === 0) {
      // 1. HQ Command Tower (stepped mid-rise tower + comms mast)
      const parts = [];
      const b1 = new THREE.BoxGeometry(110, 130, 90);
      b1.translate(0, 65, 0);
      parts.push(b1);

      const b2 = new THREE.BoxGeometry(75, 60, 65);
      b2.translate(0, 160, 0);
      parts.push(b2);

      const wingL = new THREE.BoxGeometry(130, 40, 55);
      wingL.translate(-80, 20, 0);
      parts.push(wingL);

      const wingR = new THREE.BoxGeometry(130, 40, 55);
      wingR.translate(80, 20, 0);
      parts.push(wingR);

      const mast = new THREE.CylinderGeometry(2, 4, 80, 8);
      mast.translate(0, 230, 0);
      parts.push(mast);

      const merged = BufferGeometryUtils.mergeGeometries(parts, false);
      const mesh = new THREE.Mesh(merged, this.matBuilding.clone());
      container.add(mesh);
    } else if (type === 1) {
      // 2. Airbase Command (barrel hangars + control tower)
      const parts = [];
      // Two aircraft hangars with arched roofs
      const h1 = new THREE.CylinderGeometry(35, 35, 140, 16, 1, false, 0, Math.PI);
      h1.rotateZ(Math.PI / 2);
      h1.translate(-90, 0, -40);
      parts.push(h1);

      const h2 = new THREE.CylinderGeometry(35, 35, 140, 16, 1, false, 0, Math.PI);
      h2.rotateZ(Math.PI / 2);
      h2.translate(-90, 0, 50);
      parts.push(h2);

      // Control tower
      const shaft = new THREE.CylinderGeometry(14, 18, 120, 8);
      shaft.translate(80, 60, 0);
      parts.push(shaft);

      const cab = new THREE.CylinderGeometry(26, 18, 24, 8);
      cab.translate(80, 132, 0);
      parts.push(cab);

      const merged = BufferGeometryUtils.mergeGeometries(parts, false);
      const mesh = new THREE.Mesh(merged, this.matIndustrial.clone());
      container.add(mesh);
    } else if (type === 2) {
      // 3. Energy Facility / Refinery (cooling towers + containment dome + tanks)
      const parts = [];
      const ct1 = new THREE.CylinderGeometry(28, 48, 130, 16);
      ct1.translate(-90, 65, -50);
      parts.push(ct1);

      const ct2 = new THREE.CylinderGeometry(28, 48, 130, 16);
      ct2.translate(-90, 65, 60);
      parts.push(ct2);

      // Reactor containment dome
      const dome = new THREE.SphereGeometry(45, 16, 12, 0, Math.PI * 2, 0, Math.PI / 2);
      dome.translate(70, 0, 0);
      parts.push(dome);

      // Fuel tanks
      const t1 = new THREE.CylinderGeometry(24, 24, 38, 16);
      t1.translate(20, 19, -80);
      parts.push(t1);

      const t2 = new THREE.CylinderGeometry(24, 24, 38, 16);
      t2.translate(80, 19, -80);
      parts.push(t2);

      const merged = BufferGeometryUtils.mergeGeometries(parts, false);
      const mesh = new THREE.Mesh(merged, this.matIndustrial.clone());
      container.add(mesh);
    } else if (type === 3) {
      // 4. Early-Warning Radar Station (large radome sphere + rotating parabolic dish)
      const parts = [];
      const base = new THREE.BoxGeometry(160, 35, 120);
      base.translate(0, 17.5, 0);
      parts.push(base);

      const radomePedestal = new THREE.CylinderGeometry(32, 38, 45, 12);
      radomePedestal.translate(-70, 45, 0);
      parts.push(radomePedestal);

      const radomeSphere = new THREE.SphereGeometry(48, 16, 12);
      radomeSphere.translate(-70, 115, 0);
      parts.push(radomeSphere);

      const merged = BufferGeometryUtils.mergeGeometries(parts, false);
      const mesh = new THREE.Mesh(merged, this.matIndustrial.clone());
      container.add(mesh);

      // Rotating radar dish (child object so it can rotate)
      const dishGroup = new THREE.Group();
      dishGroup.name = 'radarDish';
      dishGroup.position.set(60, 65, 0);

      const dishPylon = new THREE.Mesh(new THREE.CylinderGeometry(8, 12, 50, 8), this.matIndustrial);
      dishPylon.position.y = 25;
      dishGroup.add(dishPylon);

      const dishGeom = new THREE.SphereGeometry(32, 16, 12, 0, Math.PI * 2, 0, Math.PI / 3);
      dishGeom.rotateX(Math.PI / 2 + 0.3);
      const dishMesh = new THREE.Mesh(dishGeom, this.matBuilding.clone());
      dishMesh.position.y = 55;
      dishGroup.add(dishMesh);

      container.add(dishGroup);
    } else {
      // 5. Urban Core District (3 mid-rise commercial glass skyscrapers)
      const parts = [];
      const t1 = new THREE.BoxGeometry(80, 210, 80);
      t1.translate(-65, 105, -30);
      parts.push(t1);

      const t2 = new THREE.BoxGeometry(70, 160, 70);
      t2.translate(65, 80, -40);
      parts.push(t2);

      const t3 = new THREE.BoxGeometry(85, 120, 85);
      t3.translate(10, 60, 60);
      parts.push(t3);

      const skybridge = new THREE.BoxGeometry(60, 18, 18);
      skybridge.translate(0, 140, -35);
      parts.push(skybridge);

      const merged = BufferGeometryUtils.mergeGeometries(parts, false);
      const mesh = new THREE.Mesh(merged, this.matBuilding.clone());
      container.add(mesh);
    }

    return container;
  }

  _createSmokeColumn() {
    const g = new THREE.Group();
    const count = 12;
    const sprites = [];

    const fireMat = new THREE.SpriteMaterial({
      color: 0xf97316,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    const smokeMat = new THREE.SpriteMaterial({
      color: 0x334155,
      transparent: true,
      opacity: 0.7,
      depthWrite: false,
    });

    for (let i = 0; i < count; i++) {
      const isFire = i < 4;
      const sp = new THREE.Sprite(isFire ? fireMat.clone() : smokeMat.clone());
      sp.position.set(
        (Math.random() - 0.5) * 60,
        i * 20 + Math.random() * 15,
        (Math.random() - 0.5) * 60
      );
      const s = isFire ? 45 + Math.random() * 25 : 70 + i * 15;
      sp.scale.set(s, s, 1);
      g.add(sp);
      sprites.push(sp);
    }

    return { group: g, sprites, life: 0.0 };
  }

  /**
   * Register a missile impact on or near an HVA.
   */
  registerImpact(hvaId) {
    if (hvaId < 0 || hvaId >= this.compounds.length) return;
    const comp = this.compounds[hvaId];
    comp.hits++;
    comp.damageFlashTimer = 1.0; // Flash red for 1s
    comp.smokeColumn.group.visible = true;
    comp.smokeColumn.life = 25.0; // Smoke lingers for 25s sim time

    // Update billboard sprite
    const ctx = comp.labelObj.ctx;
    ctx.fillStyle = 'rgba(239, 68, 68, 0.9)';
    ctx.strokeStyle = '#f87171';
    ctx.lineWidth = 4;
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(8, 8, 496, 112, 16);
    } else {
      ctx.rect(8, 8, 496, 112);
    }
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 40px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`HVA-${hvaId + 1}: ${comp.hits} HITS [DAMAGED]`, 256, 64);
    comp.labelObj.texture.needsUpdate = true;
  }

  /**
   * Update HVA compounds from telemetry frame.
   * @param {Object} frame Decoded frame
   * @param {number} simDt Scaled simulation delta seconds
   */
  update(frame, simDt = 0.016) {
    if (!frame || !frame.hvas) return;

    const hvas = frame.hvas;
    const P = Math.min(hvas.length, this.maxHvas);
    const simTime = frame.header ? frame.header.timestamp : 0.0;

    // Detect missile impacts near HVAs
    if (frame.missiles) {
      for (let m = 0; m < frame.missiles.length; m++) {
        const mis = frame.missiles[m];
        if (mis.status === 2.0) { // HVA Impact status
          // Find closest HVA
          for (let p = 0; p < P; p++) {
            const hva = hvas[p];
            const dist = Math.hypot(mis.x - hva.x, mis.y - hva.y);
            if (dist < 200.0 && this.compounds[p].damageFlashTimer <= 0.01) {
              this.registerImpact(p);
            }
          }
        }
      }
    }

    for (let p = 0; p < this.maxHvas; p++) {
      const comp = this.compounds[p];
      if (p < P) {
        const hva = hvas[p];
        comp.group.position.set(hva.x, 0, -hva.y);
        comp.group.visible = true;
        comp.dome.visible = this.showDomes;

        // Rotate radar dish if present
        if (comp.rotatingPart) {
          comp.rotatingPart.rotation.y += simDt * 1.2;
        }

        // Blink red warning beacon on mast
        const blinkPhase = Math.sin(simTime * 4.0 + p);
        comp.beacon.visible = blinkPhase > 0.1;

        // Damage flash decay
        if (comp.damageFlashTimer > 0) {
          comp.damageFlashTimer -= simDt;
          const flashProgress = comp.damageFlashTimer / 1.0;
          comp.model.traverse((child) => {
            if (child.isMesh && child.material && child.material.emissive) {
              child.material.emissive.setHex(0xef4444);
              child.material.emissiveIntensity = flashProgress * 0.8;
            }
          });
        }

        // Animate lingering smoke/fire column
        if (comp.smokeColumn.life > 0) {
          comp.smokeColumn.life -= simDt;
          const sprites = comp.smokeColumn.sprites;
          for (let s = 0; s < sprites.length; s++) {
            sprites[s].position.y += simDt * (12.0 + s * 3.0);
            sprites[s].position.x += Math.sin(simTime + s) * simDt * 5.0;
            if (sprites[s].position.y > 220) {
              sprites[s].position.y = 15;
            }
          }
          if (comp.smokeColumn.life <= 0) {
            comp.smokeColumn.group.visible = false;
          }
        }
      } else {
        comp.group.visible = false;
      }
    }
  }
}

