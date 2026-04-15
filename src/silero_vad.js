import {
  conv1d, lstmCell, matmul, addBias, relu, sigmoid,
  reflectPad1d, stftMagnitude,
} from './ops.js';

// Fixed constants for the 16kHz branch, derived in docs/ARCHITECTURE.md.
const SR = 16000;
const FRAME_SAMPLES = 512;
const PAD = 32;                // reflection pad on each side → 576 samples (STFT out len 3)
const STFT_KERNEL = 256;
const STFT_STRIDE = 128;
const N_FREQS = 129;           // STFT basis has 258 rows = 129 real + 129 imag
const HIDDEN = 128;

/**
 * Pure-JS SileroVAD v5 inference engine, 16kHz-only.
 *
 *   const vad = new SileroVADJS(weights);
 *   vad.reset();                       // clear LSTM state between utterances
 *   const prob = vad.process(frame);   // frame: Float32Array(512), [-1, 1], 16kHz
 */
export class SileroVADJS {
  constructor(weights) {
    this.w = weights;
    this.h = new Array(HIDDEN).fill(0);  // f64 for recurrence precision
    this.c = new Array(HIDDEN).fill(0);
  }

  reset() {
    for (let i = 0; i < HIDDEN; i++) { this.h[i] = 0; this.c[i] = 0; }
  }

  /** Packed state (2, 1, 128) matching ORT's `stateN` output layout. */
  exportState() {
    const out = new Float32Array(2 * HIDDEN);
    out.set(this.h, 0);
    out.set(this.c, HIDDEN);
    return Array.from(out);
  }

  process(frame) {
    if (frame.length !== FRAME_SAMPLES) {
      throw new Error(`expected ${FRAME_SAMPLES}-sample frame, got ${frame.length}`);
    }

    // 1. Reflect-pad: 512 → 640
    const padded = reflectPad1d(frame, PAD);

    // 2. STFT magnitude: (576,) → (129, 3)
    const stftBasis = this.w['stft.forward_basis_buffer'].data;
    let x = stftMagnitude(padded, stftBasis, {
      kernelSize: STFT_KERNEL,
      stride: STFT_STRIDE,
      inputLength: padded.length,
      nFreqs: N_FREQS,
    });
    let inCh = N_FREQS;
    let inLen = Math.floor((padded.length - STFT_KERNEL) / STFT_STRIDE) + 1;

    // 3–6. Encoder: four Conv1D→ReLU layers.
    const enc = [
      { wName: 'encoder.0.reparam_conv.weight', bName: 'encoder.0.reparam_conv.bias', outCh: 128, stride: 1 },
      { wName: 'encoder.1.reparam_conv.weight', bName: 'encoder.1.reparam_conv.bias', outCh: 64,  stride: 2 },
      { wName: 'encoder.2.reparam_conv.weight', bName: 'encoder.2.reparam_conv.bias', outCh: 64,  stride: 2 },
      { wName: 'encoder.3.reparam_conv.weight', bName: 'encoder.3.reparam_conv.bias', outCh: 128, stride: 1 },
    ];
    for (const layer of enc) {
      const w = this.w[layer.wName].data;
      const b = this.w[layer.bName].data;
      x = conv1d(x, w, b, {
        inChannels: inCh,
        outChannels: layer.outCh,
        kernelSize: 3,
        stride: layer.stride,
        padding: 1,
        inputLength: inLen,
      });
      inLen = Math.floor((inLen + 2 - 3) / layer.stride) + 1;
      inCh = layer.outCh;
      x = relu(x);
    }
    // After encoder: shape (128, 1). Squeeze time → 128-dim vector.
    if (inLen !== 1 || inCh !== HIDDEN) {
      throw new Error(`unexpected encoder output: (${inCh}, ${inLen})`);
    }

    // 7. LSTM cell
    const { h, c } = lstmCell(x, this.h, this.c, {
      W_ih: this.w['decoder.rnn.weight_ih'].data,
      W_hh: this.w['decoder.rnn.weight_hh'].data,
      b_ih: this.w['decoder.rnn.bias_ih'].data,
      b_hh: this.w['decoder.rnn.bias_hh'].data,
      inputSize: HIDDEN,
      hiddenSize: HIDDEN,
    });
    this.h = h;
    this.c = c;

    // 8. ReLU(h)
    const hR = relu(new Float32Array(h));

    // 9. 1×1 Conv decoder: weight (1, 128, 1) → equivalent to matmul(hR, 1, 128, W^T, 1)
    // Weight data is stored as (1, 128, 1) row-major = a 128-vector. Dot product with hR.
    const wDec = this.w['decoder.decoder.2.weight'].data;
    const bDec = this.w['decoder.decoder.2.bias'].data;
    const logit = matmul(hR, 1, HIDDEN, wDec, 1);
    addBias(logit, 1, 1, bDec);

    // 10. Sigmoid
    return sigmoid(logit[0]);
  }
}
