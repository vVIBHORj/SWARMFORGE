import test from 'node:test';
import assert from 'node:assert/strict';
import { MockTelemetry } from '../src/MockTelemetry.js';
import { parseBinaryFrame } from '../src/main.js';

test('MockTelemetry outputs valid binary frames conforming to frozen protocol', () => {
  const mock = new MockTelemetry();
  const buffer = mock.packBinaryFrame();

  assert.ok(buffer instanceof ArrayBuffer, 'Should produce ArrayBuffer');
  assert.ok(buffer.byteLength > 64, 'Buffer size should exceed header size');

  const parsed = parseBinaryFrame(buffer);
  assert.ok(parsed !== null, 'parseBinaryFrame should successfully decode buffer');
  assert.equal(parsed.header.magic, 2026.0);
  assert.equal(parsed.header.P, mock.hvas.length);
  assert.equal(parsed.header.N, mock.drones.length);
  assert.equal(parsed.header.M, mock.missiles.length);
  assert.equal(parsed.hvas.length, mock.hvas.length);
  assert.equal(parsed.drones.length, mock.drones.length);
  assert.equal(parsed.missiles.length, mock.missiles.length);
});

test('MockTelemetry steps simulation and intercepts threats', () => {
  const mock = new MockTelemetry();
  const initTimestamp = mock.timestamp;

  for (let s = 0; s < 50; s++) {
    mock.step();
  }

  assert.ok(mock.timestamp > initTimestamp, 'Timestamp should advance with step()');
  assert.equal(mock.t, 50);

  // Check that command handling works
  mock.handleCommand({ type: 'spawn_salvo', count: 5 });
  assert.equal(mock.missiles.length, 35);
});

