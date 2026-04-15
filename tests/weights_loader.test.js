import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { loadWeightsFromBuffers } from '../src/weights_loader.js';

const bin = readFileSync('weights/silero_vad_v5.bin');
const manifest = JSON.parse(readFileSync('weights/silero_vad_v5.manifest.json', 'utf8'));

describe('loadWeightsFromBuffers', () => {
  it('exposes all 15 tensors with correct shapes and non-zero data', () => {
    // Copy the buffer to guarantee correct alignment for Float32Array views.
    const arrayBuf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
    const weights = loadWeightsFromBuffers(arrayBuf, manifest);
    expect(Object.keys(weights).length).toBe(15);

    const e0 = weights['encoder.0.reparam_conv.weight'];
    expect(e0.shape).toEqual([128, 129, 3]);
    expect(e0.data).toBeInstanceOf(Float32Array);
    expect(e0.data.length).toBe(128 * 129 * 3);
    expect(e0.data.some((v) => v !== 0)).toBe(true);

    const rnn = weights['decoder.rnn.weight_ih'];
    expect(rnn.shape).toEqual([512, 128]);
    expect(rnn.data.length).toBe(512 * 128);

    const stft = weights['stft.forward_basis_buffer'];
    expect(stft.shape).toEqual([258, 1, 256]);
    expect(stft.data.length).toBe(258 * 256);
  });

  it('returns views (zero-copy) sharing the same underlying buffer', () => {
    const arrayBuf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
    const weights = loadWeightsFromBuffers(arrayBuf, manifest);
    const first = weights[Object.keys(weights)[0]];
    expect(first.data.buffer).toBe(arrayBuf);
  });
});
