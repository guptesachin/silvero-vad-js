// Neural-net primitive ops. All tensors are row-major Float32Array.

export function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

export function tanh(x) {
  return Math.tanh(x);
}

/** ReLU in place. Returns the same array for chaining. */
export function relu(arr) {
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] < 0) arr[i] = 0;
  }
  return arr;
}

/**
 * Matrix multiply: C (aRows x bCols) = A (aRows x aCols) @ B (aCols x bCols).
 * Loop order i,k,j for cache locality on B.
 */
export function matmul(a, aRows, aCols, b, bCols) {
  const c = new Float32Array(aRows * bCols);
  for (let i = 0; i < aRows; i++) {
    for (let k = 0; k < aCols; k++) {
      const aik = a[i * aCols + k];
      if (aik === 0) continue;
      for (let j = 0; j < bCols; j++) {
        c[i * bCols + j] += aik * b[k * bCols + j];
      }
    }
  }
  return c;
}

/** Add a per-row bias in place. x has shape (rows, cols); bias length = rows. */
export function addBias(x, rows, cols, bias) {
  for (let i = 0; i < rows; i++) {
    const b = bias[i];
    for (let j = 0; j < cols; j++) {
      x[i * cols + j] += b;
    }
  }
  return x;
}

/**
 * 1D convolution with symmetric zero padding.
 * input:  (inChannels, inputLength) row-major
 * weight: (outChannels, inChannels, kernelSize) row-major
 * bias:   (outChannels,)
 * Returns (outChannels, outLen) where outLen = floor((inputLength + 2*padding - kernelSize)/stride) + 1
 */
export function conv1d(input, weight, bias, opts) {
  const { inChannels, outChannels, kernelSize, stride, padding, inputLength } = opts;
  const paddedLen = inputLength + 2 * padding;
  const outLen = Math.floor((paddedLen - kernelSize) / stride) + 1;
  const out = new Float32Array(outChannels * outLen);

  for (let oc = 0; oc < outChannels; oc++) {
    const wBase = oc * inChannels * kernelSize;
    for (let t = 0; t < outLen; t++) {
      let sum = bias[oc];
      const inStart = t * stride - padding;
      for (let ic = 0; ic < inChannels; ic++) {
        const inBase = ic * inputLength;
        const wcBase = wBase + ic * kernelSize;
        for (let k = 0; k < kernelSize; k++) {
          const idx = inStart + k;
          if (idx < 0 || idx >= inputLength) continue;
          sum += input[inBase + idx] * weight[wcBase + k];
        }
      }
      out[oc * outLen + t] = sum;
    }
  }
  return out;
}

/**
 * Single-timestep LSTM cell with PyTorch gate order [i, f, g, o].
 *
 *   gates = W_ih @ x + b_ih + W_hh @ hPrev + b_hh   // shape (4*hidden,)
 *   i, f, g, o = split(gates, 4 contiguous blocks of hiddenSize)
 *   c = sigmoid(f) * cPrev + sigmoid(i) * tanh(g)
 *   h = sigmoid(o) * tanh(c)
 *
 * SileroVAD v5's exported weights use PyTorch order (not ONNX [i,o,f,c]),
 * which is why this function hardcodes [i,f,g,o].
 */
export function lstmCell(x, hPrev, cPrev, params) {
  const { W_ih, W_hh, b_ih, b_hh, inputSize, hiddenSize } = params;
  const four = 4 * hiddenSize;
  const gates = new Float32Array(four);

  for (let r = 0; r < four; r++) gates[r] = b_ih[r] + b_hh[r];

  for (let r = 0; r < four; r++) {
    const base = r * inputSize;
    let sum = 0;
    for (let j = 0; j < inputSize; j++) sum += W_ih[base + j] * x[j];
    gates[r] += sum;
  }
  for (let r = 0; r < four; r++) {
    const base = r * hiddenSize;
    let sum = 0;
    for (let j = 0; j < hiddenSize; j++) sum += W_hh[base + j] * hPrev[j];
    gates[r] += sum;
  }

  const h = new Float32Array(hiddenSize);
  const c = new Float32Array(hiddenSize);
  for (let n = 0; n < hiddenSize; n++) {
    const i = 1 / (1 + Math.exp(-gates[n]));
    const f = 1 / (1 + Math.exp(-gates[hiddenSize + n]));
    const g = Math.tanh(gates[2 * hiddenSize + n]);
    const o = 1 / (1 + Math.exp(-gates[3 * hiddenSize + n]));
    c[n] = f * cPrev[n] + i * g;
    h[n] = o * Math.tanh(c[n]);
  }
  return { h, c };
}

/**
 * 1D reflection padding, matching PyTorch's ReflectionPad1d.
 * For input [a,b,c,d,e] and pad=2, returns [c,b,a,b,c,d,e,d,c]: boundary
 * samples appear exactly once; only interior samples are mirrored.
 * Requires pad < inputLength.
 */
export function reflectPad1d(input, pad) {
  const n = input.length;
  if (pad === 0) return new Float32Array(input);
  if (pad >= n) throw new Error(`reflect pad ${pad} >= input length ${n}`);
  const out = new Float32Array(n + 2 * pad);
  for (let i = 0; i < pad; i++) out[i] = input[pad - i];         // mirror left (skip boundary)
  for (let i = 0; i < n; i++) out[pad + i] = input[i];
  for (let i = 0; i < pad; i++) out[pad + n + i] = input[n - 2 - i]; // mirror right
  return out;
}

/**
 * STFT magnitude via SileroVAD's fused Conv1D basis.
 *
 * `basis` has shape (2 * nFreqs, 1, kernelSize), layout: first nFreqs rows
 * are the cosine (real) kernels, next nFreqs rows are the sine (imag) kernels.
 * Computes conv1d with stride, splits output into real/imag halves, returns
 * magnitude = sqrt(real^2 + imag^2) as shape (nFreqs, nFrames).
 */
export function stftMagnitude(input, basis, opts) {
  const { kernelSize, stride, inputLength, nFreqs } = opts;
  const outLen = Math.floor((inputLength - kernelSize) / stride) + 1;
  const mag = new Float32Array(nFreqs * outLen);

  // Real kernels at rows [0, nFreqs), imag at [nFreqs, 2*nFreqs).
  for (let f = 0; f < nFreqs; f++) {
    const wReBase = f * kernelSize;
    const wImBase = (nFreqs + f) * kernelSize;
    for (let t = 0; t < outLen; t++) {
      let re = 0, im = 0;
      const inStart = t * stride;
      for (let k = 0; k < kernelSize; k++) {
        const v = input[inStart + k];
        re += basis[wReBase + k] * v;
        im += basis[wImBase + k] * v;
      }
      mag[f * outLen + t] = Math.sqrt(re * re + im * im);
    }
  }
  return mag;
}
