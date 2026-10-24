/**
 * SandboxHUD.js
 * Interactive DOM telemetry overlay and Red-Team Sandbox command controller.
 *
 * Spec Section 9.2:
 * - Real-time telemetry: FPS, sim clock, entity counts, intercept/leaker rates, Delta-V.
 * - Algorithm hot-swapping mid-episode.
 * - Interactive Red-Team salvo threat spawner with live sliders.
 * - Sensor noise & playback control (pause/resume, reset, scale presets).
 */

export class SandboxHUD {
  /**
   * @param {Function} sendCommandCallback Function (cmdObject) => void to send JSON over WebSocket
   */
  constructor(sendCommandCallback) {
    this.sendCommand = sendCommandCallback || (() => {});

    this.isPaused = false;
    this.lastDomUpdate = 0;
    this.domUpdateIntervalMs = 50; // Update DOM at ~20 Hz to avoid reflow overhead

    this.algoMap = {
      0: 'Greedy Nearest + TPN',
      1: 'Hungarian Matching + TPN',
      2: 'GPU Parallel Auction + TPN',
      3: 'Attention-MAPPO Neuro-Symbolic',
    };

    this.algoIdToKey = {
      0: 'greedy',
      1: 'hungarian',
      2: 'auction',
      3: 'attention_mappo',
    };

    this.cacheDomElements();
    this.bindEvents();
  }

  cacheDomElements() {
    // Status
    this.connBadge = document.getElementById('conn-badge');
    this.connText = document.getElementById('conn-text');

    // Telemetry fields
    this.elAlgo = document.getElementById('tel-algo');
    this.elFpsClient = document.getElementById('tel-fps-client');
    this.elFpsServer = document.getElementById('tel-fps-server');
    this.elTime = document.getElementById('tel-time');
    this.elStep = document.getElementById('tel-step');
    this.elDrones = document.getElementById('tel-drones');
    this.elMissiles = document.getElementById('tel-missiles');
    this.elHvas = document.getElementById('tel-hvas');
    this.elInterceptRate = document.getElementById('tel-intercept-rate');
    this.elInterceptBar = document.getElementById('tel-intercept-bar');
    this.elLeakerRate = document.getElementById('tel-leaker-rate');
    this.elDupConflicts = document.getElementById('tel-dup-conflicts');
    this.elFriendlyCols = document.getElementById('tel-friendly-cols');
    this.elDeltaV = document.getElementById('tel-delta-v');

    // Controls
    this.selectAlgo = document.getElementById('select-algo');
    this.inputSalvoCount = document.getElementById('input-salvo-count');
    this.valSalvoCount = document.getElementById('val-salvo-count');
    this.inputSalvoSpeed = document.getElementById('input-salvo-speed');
    this.valSalvoSpeed = document.getElementById('val-salvo-speed');
    this.inputSalvoWeave = document.getElementById('input-salvo-weave');
    this.valSalvoWeave = document.getElementById('val-salvo-weave');
    this.inputSalvoDecoy = document.getElementById('input-salvo-decoy');
    this.valSalvoDecoy = document.getElementById('val-salvo-decoy');
    this.btnSpawnSalvo = document.getElementById('btn-spawn-salvo');

    this.inputNoise = document.getElementById('input-noise');
    this.valNoise = document.getElementById('val-noise');

    this.btnPausePlay = document.getElementById('btn-pause-play');
    this.pauseIcon = document.getElementById('pause-icon');
    this.pauseText = document.getElementById('pause-text');
    this.btnReset = document.getElementById('btn-reset');

    this.chkTrails = document.getElementById('chk-trails');
    this.chkLocks = document.getElementById('chk-locks');
    this.chkDomes = document.getElementById('chk-domes');

    this.presetButtons = document.querySelectorAll('.btn-preset');
  }

  bindEvents() {
    // 1. Algorithm Selection Dropdown
    if (this.selectAlgo) {
      this.selectAlgo.addEventListener('change', (e) => {
        const algo = e.target.value;
        this.sendCommand({ type: 'set_algorithm', algorithm: algo });
      });
    }

    // 2. Salvo Controls Sliders
    if (this.inputSalvoCount && this.valSalvoCount) {
      this.inputSalvoCount.addEventListener('input', (e) => {
        this.valSalvoCount.textContent = e.target.value;
      });
    }
    if (this.inputSalvoSpeed && this.valSalvoSpeed) {
      this.inputSalvoSpeed.addEventListener('input', (e) => {
        this.valSalvoSpeed.textContent = e.target.value;
      });
    }
    if (this.inputSalvoWeave && this.valSalvoWeave) {
      this.inputSalvoWeave.addEventListener('input', (e) => {
        this.valSalvoWeave.textContent = parseFloat(e.target.value).toFixed(1);
      });
    }
    if (this.inputSalvoDecoy && this.valSalvoDecoy) {
      this.inputSalvoDecoy.addEventListener('input', (e) => {
        this.valSalvoDecoy.textContent = `${Math.round(parseFloat(e.target.value) * 100)}%`;
      });
    }

    // 3. Salvo Spawn Button
    if (this.btnSpawnSalvo) {
      this.btnSpawnSalvo.addEventListener('click', () => {
        const count = parseInt(this.inputSalvoCount?.value || '15', 10);
        const speed = parseFloat(this.inputSalvoSpeed?.value || '450');
        const weave = parseFloat(this.inputSalvoWeave?.value || '1.5');
        const decoy = parseFloat(this.inputSalvoDecoy?.value || '0.15');

        this.sendCommand({
          type: 'spawn_salvo',
          count,
          speed,
          weave_freq: weave,
          decoy_fraction: decoy,
        });

        // Visual flash feedback on button
        this.btnSpawnSalvo.style.transform = 'scale(0.97)';
        setTimeout(() => {
          this.btnSpawnSalvo.style.transform = '';
        }, 120);
      });
    }

    // 4. Sensor Noise Slider
    if (this.inputNoise && this.valNoise) {
      this.inputNoise.addEventListener('input', (e) => {
        const noise = parseFloat(e.target.value);
        this.valNoise.textContent = `${noise.toFixed(0)}m`;
        this.sendCommand({ type: 'set_params', sigma_obs: noise });
      });
    }

    // 5. Pause / Resume Button
    if (this.btnPausePlay) {
      this.btnPausePlay.addEventListener('click', () => {
        this.isPaused = !this.isPaused;
        this.updatePauseButton();
        this.sendCommand({ type: 'set_params', paused: this.isPaused });
      });
    }

    // 6. Reset Button
    if (this.btnReset) {
      this.btnReset.addEventListener('click', () => {
        this.sendCommand({ type: 'reset' });
      });
    }

    // 7. Scale Presets
    this.presetButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const preset = btn.getAttribute('data-preset');
        let cmd = { type: 'reset' };
        if (preset === 'stage1') {
          cmd = { type: 'reset', num_interceptors: 6, num_scouts: 0, num_missiles: 8, mode: '2d' };
        } else if (preset === 'standard') {
          cmd = { type: 'reset', num_interceptors: 16, num_scouts: 4, num_missiles: 30, mode: '3d' };
        } else if (preset === 'swarm') {
          cmd = { type: 'reset', num_interceptors: 48, num_scouts: 12, num_missiles: 100, mode: '3d' };
        } else if (preset === 'stress') {
          cmd = { type: 'reset', num_interceptors: 240, num_scouts: 60, num_missiles: 500, mode: '3d' };
        }
        this.sendCommand(cmd);
      });
    });
  }

  updatePauseButton() {
    if (this.pauseIcon && this.pauseText) {
      if (this.isPaused) {
        this.pauseIcon.textContent = '▶';
        this.pauseText.textContent = 'Resume';
        this.btnPausePlay?.classList.add('active');
      } else {
        this.pauseIcon.textContent = '⏸';
        this.pauseText.textContent = 'Pause';
        this.btnPausePlay?.classList.remove('active');
      }
    }
  }

  setConnected(isConnected) {
    if (!this.connBadge || !this.connText) return;
    if (isConnected) {
      this.connBadge.style.background = 'rgba(16, 185, 129, 0.15)';
      this.connBadge.style.borderColor = 'rgba(16, 185, 129, 0.35)';
      this.connBadge.style.color = '#10b981';
      this.connText.textContent = 'ONLINE';
    } else {
      this.connBadge.style.background = 'rgba(239, 68, 68, 0.15)';
      this.connBadge.style.borderColor = 'rgba(239, 68, 68, 0.35)';
      this.connBadge.style.color = '#ef4444';
      this.connText.textContent = 'RECONNECTING';
    }
  }

  /**
   * Update HUD labels and gauges from latest telemetry frame and render FPS.
   * @param {Object} frame Decoded telemetry frame
   * @param {number} clientFps Current Three.js render loop FPS
   */
  update(frame, clientFps = 60) {
    if (!frame) return;

    const now = performance.now();
    if (now - this.lastDomUpdate < this.domUpdateIntervalMs) return;
    this.lastDomUpdate = now;

    const { header, drones, missiles, hvas } = frame;

    // Algorithm name
    if (this.elAlgo) {
      this.elAlgo.textContent = this.algoMap[header.algo_id] || `Algo #${header.algo_id}`;
    }
    if (this.selectAlgo && document.activeElement !== this.selectAlgo) {
      const currentKey = this.algoIdToKey[header.algo_id];
      if (currentKey && this.selectAlgo.value !== currentKey) {
        this.selectAlgo.value = currentKey;
      }
    }

    // FPS
    if (this.elFpsClient) this.elFpsClient.textContent = Math.round(clientFps);
    if (this.elFpsServer) this.elFpsServer.textContent = Math.round(header.fps);

    // Sim Time & Step
    if (this.elTime) this.elTime.textContent = header.timestamp.toFixed(2);
    if (this.elStep) this.elStep.textContent = header.step;

    // Active Counts
    let interceptorCount = 0;
    let scoutCount = 0;
    for (let i = 0; i < drones.length; i++) {
      if (drones[i].status > 0.5) {
        if (drones[i].type === 0) interceptorCount++;
        else scoutCount++;
      }
    }
    if (this.elDrones) {
      this.elDrones.textContent = `${header.N} (${interceptorCount} Int / ${scoutCount} Scout)`;
    }

    let liveMissiles = 0;
    for (let j = 0; j < missiles.length; j++) {
      if (missiles[j].status === 0.0) liveMissiles++;
    }
    if (this.elMissiles) {
      this.elMissiles.textContent = `${liveMissiles} live / ${header.M} total`;
    }

    if (this.elHvas) {
      this.elHvas.textContent = `${header.P} Defended HVAs`;
    }

    // Metrics
    const intPct = (header.intercept_rate * 100).toFixed(1);
    if (this.elInterceptRate) this.elInterceptRate.textContent = `${intPct}%`;
    if (this.elInterceptBar) this.elInterceptBar.style.width = `${intPct}%`;

    const leakPct = (header.leaker_rate * 100).toFixed(1);
    if (this.elLeakerRate) this.elLeakerRate.textContent = `${leakPct}%`;

    if (this.elDupConflicts) this.elDupConflicts.textContent = header.duplicate_conflicts;
    if (this.elFriendlyCols) this.elFriendlyCols.textContent = header.friendly_collisions;
    if (this.elDeltaV) this.elDeltaV.textContent = `${header.delta_v_total.toFixed(1)} m/s`;
  }
}
