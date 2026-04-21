# Building a Pure JavaScript Inference Engine for SileroVAD

## Goal
Replace ONNX Runtime WASM dependency with a pure JavaScript inference engine that runs SileroVAD on all browsers, including iOS Safari.

## Why
- ONNX Runtime WASM does not work on iOS Safari (WASM-SIMD broken since iOS 16.4+)
- QuizVidya students on iPhones cannot use Voice Activity Detection
- Students fall back to hold-and-release mode, causing fatigue and garbled recordings in long sessions
- This is the #1 cause of false STT failures

## SileroVAD Model Details

### License
- **MIT License** — no restrictions, no telemetry, no vendor lock
- Source: https://github.com/snakers4/silero-vad
- ONNX weights on HuggingFace: https://huggingface.co/onnx-community/silero-vad

### Model Versions
- **v5** (latest): 3x faster than v4, supports 6000+ languages
- File: `silero_vad_v5.onnx` (~2MB)
- Also available: `silero_vad_legacy.onnx` (older, simpler architecture)

### Architecture
- Convolutional layers (feature extraction from raw audio)
- LSTM layers (temporal modeling — tracks speech/silence state over time)
- Small feedforward output layer (produces speech probability)

### Input/Output Specification

**Inputs:**
| Name | Shape | Description |
|------|-------|-------------|
| `input` | (batch, 512) | PCM audio samples at 16kHz (32ms window) |
| `sr` | (1,) | Sample rate: 16000 |
| `h` | (2, batch, 64) | LSTM hidden state |
| `c` | (2, batch, 64) | LSTM cell state |

**Outputs:**
| Name | Shape | Description |
|------|-------|-------------|
| `output` | (batch, 1) | Speech probability 0.0 to 1.0 |
| `hn` | (2, batch, 64) | Updated hidden state |
| `cn` | (2, batch, 64) | Updated cell state |

- Window size: 512 samples at 16kHz = 32ms per frame
- No overlap between frames
- LSTM state carries over between frames (maintains context)

## Implementation Plan

### Step 1: Extract Model Architecture from ONNX

```bash
# Install tools
pip install onnx onnxruntime netron

# Inspect model structure
python3 -c "
import onnx
model = onnx.load('silero_vad_v5.onnx')
print(onnx.helper.printable_graph(model.graph))
"

# Visual inspection
# Open https://netron.app and drag the .onnx file
```

This reveals:
- Exact layer sequence (Conv1d → ReLU → Conv1d → LSTM → Linear → Sigmoid)
- Weight tensor names and shapes
- Any conditional logic (subgraphs)

### Step 2: Export Weights to JSON

```python
import onnx
import numpy as np
import json

model = onnx.load('silero_vad_v5.onnx')

weights = {}
for tensor in model.graph.initializer:
    arr = onnx.numpy_helper.to_array(tensor)
    weights[tensor.name] = {
        'shape': list(arr.shape),
        'data': arr.flatten().tolist()
    }

with open('silero_vad_weights.json', 'w') as f:
    json.dump(weights, f)

print(f"Exported {len(weights)} tensors")
for name, w in weights.items():
    print(f"  {name}: {w['shape']}")
```

**Expected output:** ~2MB JSON file with all weight matrices.

### Step 3: Implement Forward Pass in JavaScript

Each neural network operation needs a JS implementation:

```javascript
// Matrix multiply: C = A × B
function matmul(a, aRows, aCols, b, bCols) {
    const c = new Float32Array(aRows * bCols);
    for (let i = 0; i < aRows; i++) {
        for (let j = 0; j < bCols; j++) {
            let sum = 0;
            for (let k = 0; k < aCols; k++) {
                sum += a[i * aCols + k] * b[k * bCols + j];
            }
            c[i * bCols + j] = sum;
        }
    }
    return c;
}

// 1D Convolution
function conv1d(input, weight, bias, inChannels, outChannels, kernelSize, stride) {
    const inputLen = input.length / inChannels;
    const outputLen = Math.floor((inputLen - kernelSize) / stride) + 1;
    const output = new Float32Array(outChannels * outputLen);

    for (let oc = 0; oc < outChannels; oc++) {
        for (let i = 0; i < outputLen; i++) {
            let sum = bias[oc];
            for (let ic = 0; ic < inChannels; ic++) {
                for (let k = 0; k < kernelSize; k++) {
                    const inputIdx = ic * inputLen + (i * stride + k);
                    const weightIdx = oc * inChannels * kernelSize + ic * kernelSize + k;
                    sum += input[inputIdx] * weight[weightIdx];
                }
            }
            output[oc * outputLen + i] = sum;
        }
    }
    return output;
}

// LSTM cell
function lstmCell(input, hPrev, cPrev, weights, hiddenSize) {
    const inputSize = input.length;

    // Gates: input, forget, cell, output
    // W_ih: (4*hidden, input), W_hh: (4*hidden, hidden), b_ih, b_hh
    const gates = new Float32Array(4 * hiddenSize);

    // gates = W_ih @ input + b_ih + W_hh @ hPrev + b_hh
    for (let i = 0; i < 4 * hiddenSize; i++) {
        let sum = weights.b_ih[i] + weights.b_hh[i];
        for (let j = 0; j < inputSize; j++) {
            sum += weights.W_ih[i * inputSize + j] * input[j];
        }
        for (let j = 0; j < hiddenSize; j++) {
            sum += weights.W_hh[i * hiddenSize + j] * hPrev[j];
        }
        gates[i] = sum;
    }

    const h = new Float32Array(hiddenSize);
    const c = new Float32Array(hiddenSize);

    for (let i = 0; i < hiddenSize; i++) {
        const ig = sigmoid(gates[i]);                        // input gate
        const fg = sigmoid(gates[hiddenSize + i]);           // forget gate
        const cg = Math.tanh(gates[2 * hiddenSize + i]);     // cell gate
        const og = sigmoid(gates[3 * hiddenSize + i]);       // output gate

        c[i] = fg * cPrev[i] + ig * cg;
        h[i] = og * Math.tanh(c[i]);
    }

    return { h, c };
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

// ReLU activation
function relu(arr) {
    return arr.map(x => Math.max(0, x));
}
```

### Step 4: Build the VAD Class

```javascript
class SileroVADJS {
    constructor(weightsUrl) {
        this.weights = null;
        this.h = null;  // LSTM hidden state
        this.c = null;  // LSTM cell state
        this.loaded = false;
    }

    async load(weightsUrl) {
        const response = await fetch(weightsUrl);
        this.weights = await response.json();

        // Initialize LSTM states to zeros
        this.h = new Float32Array(2 * 64).fill(0);  // (2, 1, 64)
        this.c = new Float32Array(2 * 64).fill(0);

        this.loaded = true;
    }

    // Process one 512-sample frame, return speech probability
    process(audioFrame) {
        if (!this.loaded) throw new Error('Model not loaded');

        // audioFrame: Float32Array of 512 samples at 16kHz

        // 1. Conv layers (feature extraction)
        let x = audioFrame;
        x = conv1d(x, this.weights['conv1.weight'].data,
                   this.weights['conv1.bias'].data, 1, 64, 3, 1);
        x = relu(x);
        // ... more conv layers based on actual architecture

        // 2. LSTM layers
        const lstmOut = lstmCell(x, this.h, this.c, {
            W_ih: this.weights['lstm.weight_ih_l0'].data,
            W_hh: this.weights['lstm.weight_hh_l0'].data,
            b_ih: this.weights['lstm.bias_ih_l0'].data,
            b_hh: this.weights['lstm.bias_hh_l0'].data,
        }, 64);
        this.h = lstmOut.h;
        this.c = lstmOut.c;

        // 3. Output layer (Linear + Sigmoid)
        let output = matmul(lstmOut.h, 1, 64,
                           this.weights['output.weight'].data, 1);
        output[0] += this.weights['output.bias'].data[0];
        output[0] = sigmoid(output[0]);

        return output[0];  // speech probability 0.0 to 1.0
    }

    // Reset state between utterances
    reset() {
        this.h.fill(0);
        this.c.fill(0);
    }
}
```

### Step 5: Integrate with Web Audio API

```javascript
class VADRecorder {
    constructor() {
        this.vad = new SileroVADJS();
        this.isRecording = false;
        this.speechStarted = false;
        this.silenceFrames = 0;
        this.audioChunks = [];
    }

    async init() {
        await this.vad.load('/static/js/silero_vad_weights.json');

        this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        this.audioContext = new AudioContext({ sampleRate: 16000 });
        const source = this.audioContext.createMediaStreamSource(this.stream);

        // AudioWorklet or ScriptProcessor to get raw PCM frames
        await this.audioContext.audioWorklet.addModule('/static/js/vad-processor.js');
        this.processor = new AudioWorkletNode(this.audioContext, 'vad-processor');

        this.processor.port.onmessage = (event) => {
            const audioFrame = event.data;  // Float32Array(512)
            const speechProb = this.vad.process(audioFrame);

            if (speechProb > 0.5) {
                if (!this.speechStarted) {
                    this.speechStarted = true;
                    this.onSpeechStart();
                }
                this.silenceFrames = 0;
                this.audioChunks.push(audioFrame);
            } else if (this.speechStarted) {
                this.silenceFrames++;
                // 500ms of silence = speech ended (500ms / 32ms per frame ≈ 15 frames)
                if (this.silenceFrames > 15) {
                    this.speechStarted = false;
                    this.onSpeechEnd(this.audioChunks);
                    this.audioChunks = [];
                    this.vad.reset();
                }
            }
        };

        source.connect(this.processor);
    }

    onSpeechStart() {
        console.log('Speech detected — recording...');
    }

    onSpeechEnd(chunks) {
        console.log('Speech ended — sending to STT...');
        // Concatenate chunks and send to Deepgram
    }
}
```

### Step 6: AudioWorklet Processor

File: `vad-processor.js`
```javascript
class VADProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = new Float32Array(512);
        this.bufferIndex = 0;
    }

    process(inputs) {
        const input = inputs[0][0];  // mono channel
        if (!input) return true;

        for (let i = 0; i < input.length; i++) {
            this.buffer[this.bufferIndex++] = input[i];
            if (this.bufferIndex >= 512) {
                this.port.postMessage(this.buffer.slice());
                this.bufferIndex = 0;
            }
        }
        return true;
    }
}

registerProcessor('vad-processor', VADProcessor);
```

## Experiment Steps

### Prerequisites
```bash
pip install onnx onnxruntime numpy
```

### Step-by-step

1. **Download the model:**
```bash
wget https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
# or v5:
wget https://github.com/ricky0123/vad/raw/master/silero_vad_v5.onnx
```

2. **Inspect with Netron:**
   - Go to https://netron.app
   - Drag the .onnx file
   - Document every layer: type, input shapes, output shapes, weight names

3. **Export weights:**
   - Run the Python script from Step 2
   - Check the tensor names match what you see in Netron

4. **Build the JS engine:**
   - Start with the legacy model (simpler architecture)
   - Implement one layer at a time
   - After each layer, compare output with ONNX Runtime's output for the same input (on a non-iOS machine)

5. **Validate:**
   - Feed the same 512-sample audio frame to both ONNX Runtime and your JS engine
   - Output probabilities should match to 4-5 decimal places
   - Test with 100 frames of speech + silence

6. **Test on iOS:**
   - Serve via HTTPS (required for getUserMedia on iOS)
   - Test on iPhone Safari
   - Verify AudioWorklet works (fallback to ScriptProcessor if not)

## Potential Issues

### Subgraphs
SileroVAD v5 ONNX uses conditional subgraphs (different processing for 8kHz vs 16kHz). You only need the 16kHz path. Inspect with Netron and only implement that branch.

### Performance
Pure JS matrix math is slower than WASM. But the model is tiny:
- 512 input samples → ~64 conv features → 64 LSTM hidden units → 1 output
- Estimated: <5ms per frame on modern phones
- Need to process 1 frame every 32ms — plenty of headroom

### AudioWorklet on iOS
AudioWorklet is supported on iOS Safari 14.5+. For older versions, fall back to ScriptProcessor (deprecated but works).

### Numerical Precision
JavaScript uses 64-bit floats, ONNX uses 32-bit. Results will differ slightly at the last decimal places. This is fine for VAD — you just need the probability to be roughly right (>0.5 = speech).

## Fallback Strategy

```javascript
async function initVAD() {
    // Try SileroVAD via ONNX Runtime first (works on Chrome, Firefox, desktop Safari)
    try {
        const vad = await SileroVADOnnx.init();
        return vad;
    } catch (e) {
        console.log('ONNX Runtime failed, using JS engine');
    }

    // Fall back to pure JS engine (works everywhere including iOS Safari)
    const vad = new SileroVADJS();
    await vad.load('/static/js/silero_vad_weights.json');
    return vad;
}
```

This way:
- Chrome/Firefox/desktop users get ONNX Runtime (fastest)
- iOS Safari users get the JS engine (slightly slower but works)

## Files to Create

| File | Purpose | Size |
|------|---------|------|
| `silero_vad_weights.json` | Model weights extracted from ONNX | ~2MB |
| `silero_vad_js.js` | Pure JS inference engine | ~5-10KB |
| `vad_processor.js` | AudioWorklet for audio frame buffering | ~1KB |
| `vad_recorder.js` | High-level API (speech start/end detection) | ~3KB |

## References

- SileroVAD GitHub: https://github.com/snakers4/silero-vad
- SileroVAD architecture (DeepWiki): https://deepwiki.com/snakers4/silero-vad/2-architecture
- ONNX model on HuggingFace: https://huggingface.co/onnx-community/silero-vad
- Netron (model visualizer): https://netron.app
- SileroVAD v5 discussion: https://github.com/snakers4/silero-vad/discussions/471
- ONNX Runtime iOS issues: https://github.com/microsoft/onnxruntime/issues/22776
- Web Audio API AudioWorklet: https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet
