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
