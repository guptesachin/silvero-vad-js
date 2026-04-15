import { describe, it, expect } from 'vitest';
import { sigmoid, tanh, relu, matmul, addBias, conv1d, lstmCell } from '../src/ops.js';

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
    // input [1,2,3], kernel [1,1,1], pad=1 → padded [0,1,2,3,0], outLen = 3
    // t=0: 0+1+2=3, t=1: 1+2+3=6, t=2: 2+3+0=5
    const input = new Float32Array([1, 2, 3]);
    const weight = new Float32Array([1, 1, 1]);
    const bias = new Float32Array([0]);
    const out = conv1d(input, weight, bias, {
      inChannels: 1, outChannels: 1, kernelSize: 3, stride: 1, padding: 1, inputLength: 3,
    });
    expect(Array.from(out)).toEqual([3, 6, 5]);
  });
});

describe('lstmCell (PyTorch gate order [i, f, g, o])', () => {
  it('with zero state and zero weights, returns zero state', () => {
    const hiddenSize = 4, inputSize = 2;
    const x = new Float32Array([0.5, -0.5]);
    const hPrev = new Float32Array(hiddenSize);
    const cPrev = new Float32Array(hiddenSize);
    const { h, c } = lstmCell(x, hPrev, cPrev, {
      W_ih: new Float32Array(4 * hiddenSize * inputSize),
      W_hh: new Float32Array(4 * hiddenSize * hiddenSize),
      b_ih: new Float32Array(4 * hiddenSize),
      b_hh: new Float32Array(4 * hiddenSize),
      inputSize, hiddenSize,
    });
    // All zeros: gates all = 0. i=f=o=sigmoid(0)=0.5, g=tanh(0)=0.
    // c = 0.5*0 + 0.5*0 = 0; h = 0.5*tanh(0) = 0.
    for (let i = 0; i < hiddenSize; i++) {
      expect(c[i]).toBeCloseTo(0, 10);
      expect(h[i]).toBeCloseTo(0, 10);
    }
  });

  it('accumulates cell state when forget≈1, input≈1, g≈1', () => {
    const hiddenSize = 1, inputSize = 1;
    const W_ih = new Float32Array(4);
    const W_hh = new Float32Array(4);
    const b_ih = new Float32Array([10, 10, 10, 10]); // huge bias on i,f,g,o
    const b_hh = new Float32Array(4);
    let h = new Float32Array(1), c = new Float32Array(1);
    const x = new Float32Array([0]);
    for (let step = 0; step < 5; step++) {
      const out = lstmCell(x, h, c, { W_ih, W_hh, b_ih, b_hh, inputSize, hiddenSize });
      h = out.h; c = out.c;
    }
    // Recurrence with saturating gates, exact: c_t = s * c_{t-1} + s * t, where
    // s = sigmoid(10) and t = tanh(10). Iterate analytically to match.
    const s = 1 / (1 + Math.exp(-10));
    const tg = Math.tanh(10);
    let cExpected = 0;
    for (let step = 0; step < 5; step++) cExpected = s * cExpected + s * tg;
    const hExpected = s * Math.tanh(cExpected);
    expect(c[0]).toBeCloseTo(cExpected, 5);
    expect(h[0]).toBeCloseTo(hExpected, 5);
  });

  it('matches analytical reference for uniform weights', () => {
    // inputSize=2, hiddenSize=2, W_ih=W_hh=all ones, biases=0.
    // gate_pre = sum(x) + sum(hPrev) for every gate unit.
    // With x=(0.3,0.1), hPrev=(0.5,-0.5): sum=0.4+0=0.4.
    const W_ih = new Float32Array(Array(16).fill(1));
    const W_hh = new Float32Array(Array(16).fill(1));
    const b_ih = new Float32Array(8);
    const b_hh = new Float32Array(8);
    const x = new Float32Array([0.3, 0.1]);
    const hPrev = new Float32Array([0.5, -0.5]);
    const cPrev = new Float32Array([0.2, -0.2]);
    const { h, c } = lstmCell(x, hPrev, cPrev, {
      W_ih, W_hh, b_ih, b_hh, inputSize: 2, hiddenSize: 2,
    });
    const sig = 1 / (1 + Math.exp(-0.4));
    const gGate = Math.tanh(0.4);
    expect(c[0]).toBeCloseTo(sig * 0.2 + sig * gGate, 5);
    expect(c[1]).toBeCloseTo(sig * -0.2 + sig * gGate, 5);
    expect(h[0]).toBeCloseTo(sig * Math.tanh(c[0]), 5);
    expect(h[1]).toBeCloseTo(sig * Math.tanh(c[1]), 5);
  });
});
