import test from 'node:test';
import assert from 'node:assert/strict';
import { PlaybackController } from '../src/PlaybackController.js';

function createDummyFrame(step, timestamp, droneX, missileX) {
  return {
    header: {
      magic: 2026.0,
      timestamp,
      step,
      N: 1,
      M: 1,
      P: 1,
      intercept_rate: 0.5,
      leaker_rate: 0.0,
      duplicate_conflicts: 0,
      delta_v_total: 100.0,
      friendly_collisions: 0,
      algo_id: 1,
      sigma_obs: 0.0,
      fps: 60.0,
      world_w: 10000.0,
      world_h: 3000.0,
    },
    hvas: [{ id: 0, x: 0, y: 0, z: 0 }],
    drones: [
      { id: 0, type: 0, x: droneX, y: 100, z: 50, vx: 50, vy: 0, vz: 0, status: 1, assigned_id: 0 }
    ],
    missiles: [
      { id: 0, type: 2, x: missileX, y: 200, z: 500, vx: -100, vy: 0, vz: 0, status: 0, assigned_id: 0 }
    ],
  };
}

test('PlaybackController buffers frames and interpolates positions', () => {
  const ctrl = new PlaybackController({ bufferCapacity: 4, targetLagSeconds: 0.05 });

  const f0 = createDummyFrame(0, 0.00, 0.0, 1000.0);
  const f1 = createDummyFrame(1, 0.10, 10.0, 900.0);
  const f2 = createDummyFrame(2, 0.20, 20.0, 800.0);

  ctrl.pushFrame(f0);
  ctrl.pushFrame(f1);
  ctrl.pushFrame(f2);

  // Test direct interpolation at 0.15s
  const interpFrame = ctrl._interpolateAt(0.15);
  assert.equal(interpFrame.header.timestamp, 0.15);
  // Drone X should be interpolated halfway between 10.0 and 20.0 = 15.0
  assert.ok(Math.abs(interpFrame.drones[0].x - 15.0) < 0.001, `Expected drone x ~ 15.0, got ${interpFrame.drones[0].x}`);
  // Missile X should be interpolated halfway between 900.0 and 800.0 = 850.0
  assert.ok(Math.abs(interpFrame.missiles[0].x - 850.0) < 0.001, `Expected missile x ~ 850.0, got ${interpFrame.missiles[0].x}`);

  // Test update advances time
  const tBefore = ctrl.renderSimTime;
  const res = ctrl.update(0.016);
  assert.ok(ctrl.renderSimTime > tBefore, 'update should advance sim time');
});

test('PlaybackController freezes on pause and advances on stepOnce', () => {
  const ctrl = new PlaybackController();
  const f0 = createDummyFrame(0, 0.0, 0.0, 1000.0);
  const f1 = createDummyFrame(1, 0.1, 10.0, 900.0);
  ctrl.pushFrame(f0);
  ctrl.pushFrame(f1);

  ctrl.setPaused(true);
  ctrl.renderSimTime = 0.05;

  const res1 = ctrl.update(0.016);
  assert.equal(res1.simDt, 0.0);
  assert.equal(ctrl.renderSimTime, 0.05);

  // Step once
  ctrl.stepOnce();
  const res2 = ctrl.update(0.016);
  assert.ok(ctrl.renderSimTime > 0.05, 'Step once should advance sim time');
});

test('PlaybackController handles episode reset', () => {
  const ctrl = new PlaybackController();
  ctrl.pushFrame(createDummyFrame(100, 10.0, 500.0, 200.0));
  ctrl.pushFrame(createDummyFrame(101, 10.1, 510.0, 190.0));

  // Episode resets: step drops to 0
  ctrl.pushFrame(createDummyFrame(0, 0.0, 0.0, 1000.0));
  assert.equal(ctrl.snapshots.length, 1);
  assert.equal(ctrl.renderSimTime, 0.0);
});
