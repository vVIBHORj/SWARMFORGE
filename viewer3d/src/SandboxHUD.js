/**
 * SandboxHUD.js
 * Interactive DOM telemetry overlay, speed controls, graphics customization,
 * and Red-Team Sandbox command controller.
 *
 * Spec Section 9.2 & Step 15:
 * - Real-time telemetry: FPS, sim clock, entity counts, intercept/leaker rates, Delta-V, Sim Rate.
 * - Algorithm hot-swapping mid-episode.
 * - Interactive Red-Team salvo threat spawner with live sliders.
 * - Time-Dilation / Slow-Motion controls (0.1x to 8x, continuous slider, hotkeys [, ], 0).
 * - Step 1 Frame (S) when paused, Spacebar to toggle Pause/Resume.
 * - Visual mesh scale slider (1x to 20x).
 * - Atmospheric lighting switcher (Night / Dusk / Day).
 * - Soft smoke / lines trail mode switcher.
 * - Auto slow-mo on intercept toggle with floating speed toast notification.
 */

export class SandboxHUD {
  /**
   * @param {Function} sendCommandCallback Function (cmdObject) => void to send JSON over WebSocket
   * @param {Object} callbacks Hooks for viewer visual controls
   */
  constructor(sendCommandCallback, callbacks = {}) {
    this.sendCommand = sendCommandCallback || (() => {});
    this.callbacks = callbacks;

    this.isPaused = false;
    this.speedMultiplier = 1.0;
    this.lastDomUpdate = 0;
    this.domUpdateIntervalMs = 50; // Update DOM at ~20 Hz to avoid reflow overhead
    this.toastTimeout = null;
    this.hasHiddenLoading = false;

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
    // Loading & Toast
    this.loadingScreen = document.getElementById('loading-screen');
    this.speedToast = document.getElementById('speed-toast');

    // Status
    this.connBadge = document.getElementById('conn-badge');
    this.connText = document.getElementById('conn-text');

    // Telemetry fields
    this.elAlgo = document.getElementById('tel-algo');
    this.elFpsClient = document.getElementById('tel-fps-client');
    this.elFpsServer = document.getElementById('tel-fps-server');
    this.elSimRate = document.getElementById('tel-sim-rate');
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

    // Visual Realism Controls
    this.inputVisualScale = document.getElementById('input-visual-scale');
    this.valVisualScale = document.getElementById('val-visual-scale');
    this.envButtons = document.querySelectorAll('.btn-env');

    // Playback & Speed Controls
    this.btnPausePlay = document.getElementById('btn-pause-play');
    this.pauseIcon = document.getElementById('pause-icon');
    this.pauseText = document.getElementById('pause-text');
    this.btnStep = document.getElementById('btn-step');
    this.btnReset = document.getElementById('btn-reset');

    this.inputSpeed = document.getElementById('input-speed');
    this.valSpeed = document.getElementById('val-speed');
    this.speedPresetButtons = document.querySelectorAll('.btn-speed-preset');
    this.chkAutoSlowMo = document.getElementById('chk-auto-slowmo');

    // Layer & Mode Toggles
    this.chkTrails = document.getElementById('chk-trails');
    this.btnTrailMode = document.getElementById('btn-trail-mode');
    this.chkLocks = document.getElementById('chk-locks');
    this.chkDomes = document.getElementById('chk-domes');
    this.chkBloom = document.getElementById('chk-bloom');

    this.presetButtons = document.querySelectorAll('.btn-preset');
  }

  showSpeedToast(text) {
    if (!this.speedToast) return;
    this.speedToast.textContent = text;
    this.speedToast.classList.add('show');
    if (this.toastTimeout) clearTimeout(this.toastTimeout);
    this.toastTimeout = setTimeout(() => {
      this.speedToast?.classList.remove('show');
    }, 850);
  }

  hideLoadingScreen() {
    if (this.hasHiddenLoading || !this.loadingScreen) return;
    this.hasHiddenLoading = true;
    this.loadingScreen.classList.add('hidden');
    setTimeout(() => {
      if (this.loadingScreen) this.loadingScreen.style.display = 'none';
    }, 550);
  }

  bindEvents() {
    // Blur helper to keep focus off buttons so Spacebar doesn't re-trigger them
    const preventFocusRetention = (el) => {
      if (el) {
        el.addEventListener('click', () => {
          el.blur();
        });
      }
    };

    // 1. Algorithm Selection Dropdown
    if (this.selectAlgo) {
      this.selectAlgo.addEventListener('change', (e) => {
        const algo = e.target.value;
        this.sendCommand({ type: 'set_algorithm', algorithm: algo });
        this.selectAlgo.blur();
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
      preventFocusRetention(this.btnSpawnSalvo);
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

        this.btnSpawnSalvo.style.transform = 'scale(0.97)';
        setTimeout(() => {
          if (this.btnSpawnSalvo) this.btnSpawnSalvo.style.transform = '';
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

    // 5. Visual Scale Slider
    if (this.inputVisualScale && this.valVisualScale) {
      this.inputVisualScale.addEventListener('input', (e) => {
        const sc = parseFloat(e.target.value);
        this.valVisualScale.textContent = `${sc.toFixed(0)}x`;
        if (this.callbacks.onVisualScaleChange) {
          this.callbacks.onVisualScaleChange(sc);
        }
      });
    }

    // 6. Atmospheric Lighting Buttons
    this.envButtons.forEach((btn) => {
      preventFocusRetention(btn);
      btn.addEventListener('click', () => {
        const env = btn.getAttribute('data-env');
        this.envButtons.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        if (this.callbacks.onEnvironmentChange) {
          this.callbacks.onEnvironmentChange(env);
        }
      });
    });

    // 7. Pause / Resume Button
    if (this.btnPausePlay) {
      preventFocusRetention(this.btnPausePlay);
      this.btnPausePlay.addEventListener('click', () => {
        this.togglePause();
      });
    }

    // 8. Step Forward 1 Frame Button
    if (this.btnStep) {
      preventFocusRetention(this.btnStep);
      this.btnStep.addEventListener('click', () => {
        this.stepOnce();
      });
    }

    // 9. Speed Slider
    if (this.inputSpeed && this.valSpeed) {
      this.inputSpeed.addEventListener('input', (e) => {
        const spd = parseFloat(e.target.value);
        this.setSpeedMultiplier(spd);
        this.showSpeedToast(`${spd.toFixed(2)}x SPEED`);
      });
    }

    // 10. Speed Preset Buttons
    this.speedPresetButtons.forEach((btn) => {
      preventFocusRetention(btn);
      btn.addEventListener('click', () => {
        const spd = parseFloat(btn.getAttribute('data-speed'));
        this.setSpeedMultiplier(spd);
        this.showSpeedToast(`${spd.toFixed(2)}x SPEED`);
      });
    });

    // 11. Auto Slow-Mo Toggle
    if (this.chkAutoSlowMo) {
      this.chkAutoSlowMo.addEventListener('change', (e) => {
        if (this.callbacks.onAutoSlowMoChange) {
          this.callbacks.onAutoSlowMoChange(e.target.checked);
        }
        this.showSpeedToast(e.target.checked ? 'AUTO SLOW-MO: ON' : 'AUTO SLOW-MO: OFF');
      });
    }

    // 12. Trail Mode Switcher (Lines vs Soft Smoke)
    if (this.btnTrailMode) {
      preventFocusRetention(this.btnTrailMode);
      let isSmoke = false;
      this.btnTrailMode.addEventListener('click', () => {
        isSmoke = !isSmoke;
        this.btnTrailMode.textContent = isSmoke ? 'Smoke' : 'Lines';
        if (this.callbacks.onTrailModeChange) {
          this.callbacks.onTrailModeChange(isSmoke ? 'smoke' : 'lines');
        }
      });
    }

    // 13. Bloom Toggle
    if (this.chkBloom) {
      this.chkBloom.addEventListener('change', (e) => {
        if (this.callbacks.onBloomChange) {
          this.callbacks.onBloomChange(e.target.checked);
        }
      });
    }

    // 14. Reset Button
    if (this.btnReset) {
      preventFocusRetention(this.btnReset);
      this.btnReset.addEventListener('click', () => {
        this.sendCommand({ type: 'reset' });
      });
    }

    // 15. Scale Presets
    this.presetButtons.forEach((btn) => {
      preventFocusRetention(btn);
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

  togglePause() {
    this.isPaused = !this.isPaused;
    this.updatePauseButton();
    if (this.callbacks.onPauseChange) {
      this.callbacks.onPauseChange(this.isPaused);
    }
    this.sendCommand({ type: 'set_params', paused: this.isPaused });
    this.showSpeedToast(this.isPaused ? 'PAUSED' : `${this.speedMultiplier.toFixed(2)}x SPEED`);
  }

  stepOnce() {
    if (this.callbacks.onStepOnce) {
      this.callbacks.onStepOnce();
    }
    this.sendCommand({ type: 'step_once' });
    this.showSpeedToast('STEP +1');
  }

  setSpeedMultiplier(spd) {
    const clamped = Math.max(0.05, Math.min(8.0, spd));
    this.speedMultiplier = clamped;
    if (this.valSpeed) {
      this.valSpeed.textContent = `${clamped.toFixed(2)}x`;
    }
    if (this.inputSpeed) {
      this.inputSpeed.value = clamped;
    }
    this.speedPresetButtons.forEach((b) => {
      const bSpd = parseFloat(b.getAttribute('data-speed'));
      b.classList.toggle('active', Math.abs(bSpd - clamped) < 0.04);
    });

    if (this.callbacks.onSpeedChange) {
      this.callbacks.onSpeedChange(clamped);
    }
    this.sendCommand({ type: 'set_params', speed_multiplier: clamped });
  }

  adjustSpeedStep(direction) {
    // direction > 0: faster; direction < 0: slower
    const speeds = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0];
    let newSpd = this.speedMultiplier;
    if (direction > 0) {
      for (let s of speeds) {
        if (s > this.speedMultiplier + 0.05) {
          newSpd = s;
          break;
        }
      }
      if (newSpd === this.speedMultiplier && this.speedMultiplier < 8.0) {
        newSpd = Math.min(8.0, this.speedMultiplier * 1.5);
      }
    } else {
      for (let i = speeds.length - 1; i >= 0; i--) {
        if (speeds[i] < this.speedMultiplier - 0.05) {
          newSpd = speeds[i];
          break;
        }
      }
      if (newSpd === this.speedMultiplier && this.speedMultiplier > 0.05) {
        newSpd = Math.max(0.05, this.speedMultiplier * 0.67);
      }
    }
    this.setSpeedMultiplier(newSpd);
    this.showSpeedToast(`${newSpd.toFixed(2)}x SPEED`);
  }

  updatePauseButton() {
    if (this.pauseIcon && this.pauseText) {
      if (this.isPaused) {
        this.pauseIcon.textContent = '▶';
        this.pauseText.textContent = 'Resume';
        this.btnPausePlay?.classList.add('active');
        if (this.btnStep) this.btnStep.disabled = false;
      } else {
        this.pauseIcon.textContent = '⏸';
        this.pauseText.textContent = 'Pause';
        this.btnPausePlay?.classList.remove('active');
        if (this.btnStep) this.btnStep.disabled = true;
      }
    }
  }

  setConnected(isConnected, isDemo = false) {
    if (!this.connBadge || !this.connText) return;
    if (isDemo) {
      this.connBadge.style.background = 'rgba(56, 189, 248, 0.15)';
      this.connBadge.style.borderColor = 'rgba(56, 189, 248, 0.35)';
      this.connBadge.style.color = '#38bdf8';
      this.connText.textContent = 'DEMO MODE';
    } else if (isConnected) {
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
   * @param {number} effectiveSimRate Measured sim seconds per real second
   */
  update(frame, clientFps = 60, effectiveSimRate = 1.0) {
    if (!frame) return;

    this.hideLoadingScreen();

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

    // FPS & Sim Rate
    if (this.elFpsClient) this.elFpsClient.textContent = Math.round(clientFps);
    if (this.elFpsServer) this.elFpsServer.textContent = Math.round(header.fps);
    if (this.elSimRate) {
      const simSecs = (effectiveSimRate * 20.0).toFixed(1);
      this.elSimRate.textContent = `${effectiveSimRate.toFixed(2)}x (${simSecs} sim/s)`;
    }

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
      this.elHvas.textContent = `${hvas ? hvas.length : header.P} Defended Facilities`;
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
