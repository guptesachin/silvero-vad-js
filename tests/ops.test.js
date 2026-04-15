import { describe, it, expect } from 'vitest';
import { sigmoid, tanh, relu, matmul, addBias, conv1d } from '../src/ops.js';

describe('sigmoid', () => {
  it('returns 0.5 at x=0', () => {
    expect(sigmoid(0)).toBeCloseTo(0.5, 10);
  });
  it('saturates near 1 for large positive input', () => {
    expect(sigmoid(20)).toBeCloseTo(1, 6);
  });
  it('saturates near 0 for large negative input', () => {
    expect(sigmoid(-20)).toBeCloseTo(0, 6);
  });
});

describe('tanh', () => {
  it('matches Math.tanh', () => {
    for (const x of [-3, -1, 0, 1, 3]) {
      expect(tanh(x)).toBeCloseTo(Math.tanh(x), 10);
    }
  });
});

describe('relu', () => {
  it('passes positives, zeros negatives, in-place on Float32Array', () => {
    const arr = new Float32Array([-2, -0.001, 0, 0.5, 7]);
    const out = relu(arr);
    expect(Array.from(out)).toEqual([0, 0, 0, 0.5, 7]);
    expect(out).toBeInstanceOf(Float32Array);
  });
});

describe('matmul', () => {
  it('computes 2x3 * 3x2 = 2x2 correctly', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b = new Float32Array([7, 8, 9, 10, 11, 12]);
    const c = matmul(a, 2, 3, b, 2);
    expect(Array.from(c)).toEqual([58, 64, 139, 154]);
  });

  it('handles 1xN * NxM (vector-matrix)', () => {
    const a = new Float32Array([1, 2]);
    const b = new Float32Array([10, 20, 30, 40]);
    const c = matmul(a, 1, 2, b, 2);
    expect(Array.from(c)).toEqual([70, 100]);
  });
});

describe('addBias', () => {
  it('adds a per-row bias broadcast across columns', () => {
    const x = new Float32Array([1, 2, 3, 4, 5, 6]);
    const bias = new Float32Array([10, 20]);
    addBias(x, 2, 3, bias);
    expect(Array.from(x)).toEqual([11, 12, 13, 24, 25, 26]);
  });
});

describe('conv1d', () => {
  it('1-channel in, 1-channel out, kernel=3, stride=1, no pad', () => {
    const input = new Float32Array([1, 2, 3, 4, 5]);
    const weight = new Float32Array([1, 0, -1]);
    const bias = new Float32Array([0]);
    const out = conv1d(input, weight, bias, {
      inChannels: 1, outChannels: 1, kernelSize: 3, stride: 1, padding: 0, inputLength: 5,
    });
    expect(Array.from(out)).toEqual([-2, -2, -2]);
  });

  it('handles 2 input channels, 2 output channels, with bias', () => {
    const input = new Float32Array([1, 2, 3, 4, 5, 6]);
    const weight = new Float32Array([1, 1, 0, 0, 0, 0, 1, 1]);
    const bias = new Float32Array([10, 20]);
    const out = conv1d(input, weight, bias, {
      inChannels: 2, outChannels: 2, kernelSize: 2, stride: 1, padding: 0, inputLength: 3,
    });
    expect(Array.from(out)).toEqual([13, 15, 29, 31]);
  });

  it('respects stride > 1', () => {
    const input = new Float32Array([1, 2, 3, 4, 5, 6]);
    const weight = new Float32Array([1, 1]);
    const bias = new Float32Array([0]);
    const out = conv1d(input, weight, bias, {
      inChannels: 1, outChannels: 1, kernelSize: 2, stride: 2, padding: 0, inputLength: 6,
    });
    expect(Array.from(out)).toEqual([3, 7, 11]);
  });

  it('applies zero padding symmetrically', () => {
    // input [1,2,3], kernel [1,1,1], pad=1 → padded [0,1,2,3,0], out len = (3+2-3)/1+1 = 3
    // t=0: 0+1+2 = 3; t=1: 1+2+3 = 6; t=2: 2+3+0 = 5
    const input = new Float32Array([1, 2, 3]);
    const weight = new Float32Array([1, 1, 1]);
    const bias = new Float32Array([0]);
    const out = conv1d(input, weight, bias, {
      inChannels: 1, outChannels: 1, kernelSize: 3, stride: 1, padding: 1, inputLength: 3,
    });
    expect(Array.from(out)).toEqual([3, 6, 5]);
  });
});
