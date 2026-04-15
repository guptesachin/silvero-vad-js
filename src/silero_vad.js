import {
  conv1d, lstmCell, matmul, addBias, relu, sigmoid, stftMagnitude,
} from './ops.js';

// Fixed constants for the 16kHz branch, derived in docs/ARCHITECTURE.md.
const SR = 16000;
const FRAME_SAMPLES = 512;
const CONTEXT_SAMPLES = 64;    // prepend last 64 samples from previous frame
const RIGHT_PAD = 64;          // reflection pad 64 samples on the right (matches ONNX graph)
const STFT_INPUT_LEN = CONTEXT_SAMPLES + FRAME_SAMPLES + RIGHT_PAD; // 640
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
    this.context = new Float32Array(CONTEXT_SAMPLES); // zeros on first frame
  }

  reset() {
    for (let i = 0; i < HIDDEN; i++) { this.h[i] = 0; this.c[i] = 0; }
    this.context.fill(0);
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

    // 1. Build the 640-sample STFT input:
    //    [64 context from last frame][512 new samples][64 right-reflection]
    //    Matches the ONNX graph's PyTorch ReflectionPad1d(pad=(0,64)) after the
    //    official OnnxWrapper prepends context.
    const padded = new Float32Array(STFT_INPUT_LEN);
    padded.set(this.context, 0);
    padded.set(frame, CONTEXT_SAMPLES);
    // Right-reflect the last 64 samples of [context+frame]: for a reflection
    // that excludes the boundary, index mapping is: padded[N + i] = padded[N - 2 - i].
    const N = CONTEXT_SAMPLES + FRAME_SAMPLES; // 576
    for (let i = 0; i < RIGHT_PAD; i++) padded[N + i] = padded[N - 2 - i];

    // 2. STFT magnitude: (640,) → (129, 4)
    const stftBasis = this.w['stft.forward_basis_buffer'].data;
    let x = stftMagnitude(padded, stftBasis, {
      kernelSize: STFT_KERNEL,
      stride: STFT_STRIDE,
      inputLength: padded.length,
      nFreqs: N_FREQS,
    });
    let inCh = N_FREQS;
    let inLen = Math.floor((padded.length - STFT_KERNEL) / STFT_STRIDE) + 1;

    // 3. Save last 64 samples of [context + frame] as context for next call.
    this.context = new Float32Array(CONTEXT_SAMPLES);
    for (let i = 0; i < CONTEXT_SAMPLES; i++) {
      this.context[i] = padded[N - CONTEXT_SAMPLES + i];
    }

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
