# SileroVAD v5 — ONNX Architecture (inspected 2026-04-14)

Source: `weights/silero_vad_v5.onnx` (MIT, from snakers4/silero-vad).

## Graph I/O

| Name | Shape | Dtype | Notes |
|---|---|---|---|
| IN `input` | (N, samples) | float32 | raw PCM, 512 samples per frame at 16kHz |
| IN `state` | (2, N, 128) | float32 | packed LSTM state: state[0] = h, state[1] = c |
| IN `sr` | () | int64 | 16000 or 8000 |
| OUT `output` | (N, 1) | float32 | speech probability |
| OUT `stateN` | (2, N, 128) | float32 | updated state |

## Branch Selection

Root graph has only 5 nodes: a `Constant(16000)`, an `Equal(sr, 16000)`, then a top-level `If` dispatching to `then_branch` (16kHz) or `else_branch` (8kHz). **We only implement `then_branch` (16kHz)** — iPhone audio capture is always at 16kHz in our recorder.

## 16kHz Forward Pass (then_branch)

Weights are stored as `Constant` nodes *inside* the subgraph (not top-level initializers). The JS engine loads them from our exported `.bin` under their logical names (the `If_0_then_branch__Inline_0__` prefix is stripped at export time).

### 1. Build the 640-sample STFT input

The model expects **576 samples** to reach its input tensor: 64 samples of audio context from the **previous** frame prepended to the current 512-sample frame. This context is the last 64 samples of `[prev_context + prev_frame]`. On the first call the context is zeros.

The ONNX graph then applies **PyTorch `ReflectionPad1d(pad=(0, 64))`** — a 64-sample reflection on the *right side only* (matches `padded[N + i] = padded[N - 2 - i]`). Final tensor fed to the STFT Conv is **(N, 640)**.

In the JS port we don't call into ONNX's pad; we build the 640-sample tensor directly: `[64 context from last frame][512 new samples][64 right-reflection]`. After each call we save the last 64 of `[context + frame]` for the next call.

This is the #1 thing to get right. We originally (incorrectly) interpreted the pad as symmetric-32-on-each-side from a single 512 input, which produced STFT output of 3 time frames instead of 4 and silently broke speech detection on real audio (while still matching ORT's frame-by-frame output in the equally-broken regime — our first golden fixture validated the wrong behavior).

### 2. STFT via fixed-weight Conv1D

| Op | Weight | Shape | Attrs |
|---|---|---|---|
| Conv1D | `stft.forward_basis_buffer` | (258, 1, 256) | kernel=256, stride=128, pad=0 |

Input: (N, 1, 640). Output: (N, 258, 4). Rows 0–128 = real parts of 129 frequency bins; rows 129–257 = imag parts. (Output length = (640 − 256) / 128 + 1 = **4 time frames**.)

### 3. Magnitude

Split the 258 channels into real (0..128) and imag (129..257), square-and-add, sqrt:

```
mag[:, f, t] = sqrt(real[:, f, t]^2 + imag[:, f, t]^2)     f in 0..128
```

Output: (N, 129, 4).

### 4. Encoder (four ReLU-convs)

| # | Weight name | Weight shape | Bias shape | Stride | Pad | Output length |
|---|---|---|---|---|---|---|
| 0 | `encoder.0.reparam_conv.weight` | (128, 129, 3) | (128,) | 1 | 1 | 4 |
| 1 | `encoder.1.reparam_conv.weight` | (64, 128, 3) | (64,) | 2 | 1 | 2 |
| 2 | `encoder.2.reparam_conv.weight` | (64, 64, 3) | (64,) | 2 | 1 | 1 |
| 3 | `encoder.3.reparam_conv.weight` | (128, 64, 3) | (128,) | 1 | 1 | 1 |

Each followed by ReLU. Final shape: **(N, 128, 1)** — a single 128-dim feature vector per frame. (Length math: 4 → 2 → 1 → 1 with kernel=3, pad=1 at each stride.)

### 5. LSTM cell (single timestep per frame)

| Weight | Shape | Role |
|---|---|---|
| `decoder.rnn.weight_ih` | (512, 128) | input-to-hidden, all 4 gates stacked |
| `decoder.rnn.weight_hh` | (512, 128) | hidden-to-hidden |
| `decoder.rnn.bias_ih`   | (512,) | |
| `decoder.rnn.bias_hh`   | (512,) | |

Hidden size = 512 / 4 = **128**. The stored weights are in **PyTorch `[i, f, g, o]` order**. In the ONNX graph there's a dedicated `LSTM` op that expects ONNX `[i, o, f, c]` order, so the graph includes Slice/Concat reordering just before the LSTM op. In our JS port we read the weights directly in PyTorch order and implement the LSTM cell with `[i, f, g, o]` ordering — mathematically equivalent, no reorder step needed.

Input vector: squeeze the last encoder output (N, 128, 1) → (N, 128). Initial `(h_prev, c_prev)` = unpack `state`:

```
h_prev = state[0, :, :]   # (N, 128)
c_prev = state[1, :, :]   # (N, 128)
```

Standard LSTM cell math. Output `h` is fed to the decoder; new `(h_new, c_new)` are re-packed into `stateN`.

### 6. Decoder (ReLU → 1×1 Conv → Sigmoid)

| Op | Weight | Shape | Notes |
|---|---|---|---|
| ReLU | — | — | applied to `h` |
| Conv1D (1×1) | `decoder.decoder.2.weight` | (1, 128, 1) | kernel=1 → just a linear projection |
|  | `decoder.decoder.2.bias` | (1,) | scalar; for 16kHz branch value is `-0.6245977282524109` |
| Sigmoid | — | — | `output` |

## Total Weight Budget (16kHz branch only)

| Tensor | Floats |
|---|---|
| stft.forward_basis_buffer | 66,048 |
| encoder.0.weight + bias | 49,664 |
| encoder.1.weight + bias | 24,640 |
| encoder.2.weight + bias | 12,352 |
| encoder.3.weight + bias | 24,704 |
| rnn weights + biases | 132,096 |
| decoder.decoder.2.weight + bias | 129 |
| **TOTAL** | **309,633 floats (~1.18 MB)** |

(The `.onnx` file includes both 8kHz and 16kHz branches; the 16kHz-only JS payload is ~1.2 MB.)

## Notes on ONNX graph complexity

The ONNX graph uses a native `LSTM` op preceded by Slice/Concat to reorder PyTorch's `[i, f, g, o]` weight layout into ONNX's `[i, o, f, c]`. There are also nested `If` subgraphs that handle dynamic batch shapes. None of this matters for our JS port — we skip the ONNX-side plumbing and implement the standard LSTM cell formula directly against the raw PyTorch-order weights.

The STFT is implemented as a Conv1D with a pre-computed basis buffer (rows 0..128 = cosine kernels for real parts, rows 129..257 = sine kernels for imag). We don't need to recompute the basis; we just load it as a weight.

## Numerical notes

JS per-frame probability matches the official `silero-vad` Python wrapper within **1e-4** on all 10 golden fixture frames and on 100 captured mic frames (76 speech frames detected, identical to the wrapper). The golden fixtures were regenerated using the official `OnnxWrapper` rather than feeding the raw ONNX model — the wrapper's context-prepending + internal pad flow is what production usage requires.
